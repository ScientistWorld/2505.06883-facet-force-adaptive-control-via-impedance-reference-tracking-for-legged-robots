#!/usr/bin/env python3
"""Hardening harness: reference seeds, quality variants and shortcut probes.

Subcommands
  checks   simulator / metric consistency between data/, method/, eval/train, eval/test
  run      train variants (resumable), build probe submissions, score every case
           with the unmodified hidden-test evaluator (and the train evaluator)
  analyze  ranks under the gym's rule, seed noise and paired comparisons

Each case is a checkpoint root whose ``facet/policy.pt`` is the candidate, scored
by ``python eval/test/evaluate.py <root>`` (the code path of
scripts/evaluate_test.sh). Candidate construction and the brute-force selection
never read eval/test or test scores.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path("/home/user")
HD = ROOT / "redteam" / "hardened"
sys.path.insert(0, str(ROOT))

BUDGETS = {
    "full": {"teacher": 200, "student": 50, "envs": 256, "steps": 96},
    "smoke": {"teacher": 2, "student": 1, "envs": 32, "steps": 16},
}
# name, seed, overrides (scale multiplies the update budget)
TRAIN_RUNS = [
    ("better_longer", 17, {"scale": 3}),
    ("better_longer", 7, {"scale": 3}),
    ("better_longer", 27, {"scale": 3}),
    ("reference", 17, {}),
    ("reference", 27, {}),
    ("weaker_untrained", 7, {"teacher": 0, "student": 0}),
    ("weaker_mlp_estimator", 7, {"estimator": "mlp"}),
    ("weaker_mlp_estimator", 17, {"estimator": "mlp"}),
    ("weaker_open_loop", 7, {"target_mode": "open_loop"}),
]
GRID_KX = (0.25, 0.5, 1.0, 2.0, 4.0, 5.5)
GRID_KY = (0.0, 0.1, 0.3, 1.0, 4.0)


def out_dir(mode: str) -> Path:
    return ROOT / "outputs" / ("hardening" if mode == "full" else "hardening_smoke")


def cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def cmd_checks(_args) -> None:
    import data.facet_planar_env as public
    from method.facet_reference import ImpedanceCommand, ReferenceTargets

    test_eval = load_module("hidden_test_eval", ROOT / "eval/test/evaluate.py")
    train_eval = load_module("visible_train_eval", ROOT / "eval/train/evaluate.py")
    report: dict = {}
    builders = {
        "impulse": lambda m: m.impulse_scenario(123)[0],
        "wall": lambda m: m.wall_scenario(123),
        "tracking": lambda m: m.tracking_scenario(123),
    }
    for label, build in builders.items():
        sc_test = build(test_eval)
        sc_train = build(train_eval)
        sc_public = public.Scenario(**{k: np.array(v) for k, v in vars(sc_test).items()})
        sims = {
            "data": public.PlanarForceSim(sc_public),
            "eval_test": test_eval.PlanarForceSim(sc_test),
            "eval_train": train_eval.PlanarForceSim(sc_train),
        }
        rng = np.random.default_rng(5)
        traj = {key: [] for key in sims}
        for k in range(300):
            noise = rng.normal(0.0, 1.5, size=(sc_test.n, 2))
            for key, sim in sims.items():
                obs = sim.observation().astype(np.float64)
                action = obs[:, 3:4] * 20.0 * (obs[:, 0:2] - obs[:, 5:7]) / (obs[:, 4:5] * 20.0) + 0.05 * obs[:, 5:7] + noise
                sim.step(action)
                traj[key].append(np.concatenate([sim.pos, sim.vel, sim.tilt[:, None], sim.failed[:, None]], axis=1))
        arrays = {key: np.stack(val) for key, val in traj.items()}
        report[f"{label}_max_abs_diff_data_vs_test"] = float(np.max(np.abs(arrays["data"] - arrays["eval_test"])))
        report[f"{label}_max_abs_diff_train_vs_test"] = float(np.max(np.abs(arrays["eval_train"] - arrays["eval_test"])))

    # Training reward targets (method) vs evaluator tracking metric on one trajectory.
    sc = test_eval.tracking_scenario(321)
    traj = test_eval.rollout(test_eval.VelocityBaseline(1.0, 0.5), sc, 120)
    ev_eval, _ = test_eval.tracking_errors(traj, sc)
    ref = ReferenceTargets(sc.n, test_eval.DT, (8, 16, 32), "smoothed")
    ref.reset(np.arange(sc.n), traj.pos[0], traj.vel[0])
    command = ImpedanceCommand(sc.kp, sc.kd, sc.virtual_mass, sc.target_velocity)
    worst = 0.0
    for k in range(120):
        ref.push(traj.pos[k], traj.force[k], traj.pos[k + 1], traj.vel[k + 1], command)
        _, vs = ref.current_targets(command)
        ev_method = np.mean(np.sum((traj.vel[k + 1][None] - vs) ** 2, axis=-1), axis=0)
        alive = traj.fail_state >= k + 1
        worst = max(worst, float(np.max(np.abs(np.minimum(ev_method, 25.0) - ev_eval[k])[alive], initial=0.0)))
    report["reward_targets_vs_eval_metric_max_abs_diff"] = worst
    report["ok"] = all(v < 1e-9 for key, v in report.items() if key.endswith("diff"))
    write_json(HD / "checks.json", report)
    print(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def train_job(mode: str, name: str, seed: int, overrides: dict, threads: int) -> Path:
    budget = BUDGETS[mode]
    scale = overrides.get("scale", 1)
    target = out_dir(mode) / "train" / f"{name}_s{seed}"
    if (target / "policy.pt").exists():
        return target
    cmd = [
        sys.executable, "-u", "-m", "method.run_facet",
        "--output-dir", str(target),
        "--seed", str(seed),
        "--teacher-updates", str(overrides.get("teacher", budget["teacher"] * scale)),
        "--student-updates", str(overrides.get("student", budget["student"] * scale)),
        "--n-envs", str(budget["envs"]),
        "--rollout-steps", str(budget["steps"]),
        "--estimator", overrides.get("estimator", "gru"),
        "--target-mode", overrides.get("target_mode", "smoothed"),
        "--threads", str(threads),
    ]
    target.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with open(target / "train.log", "w") as log:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"[harness] trained {name}_s{seed} rc={rc} in {time.time() - started:.0f}s", flush=True)
    return target


def evaluate(root: Path, case: str, split: str, mode: str, output: Path | None = None) -> dict:
    output = output or (HD / "cases" / case / f"scores_{split}.json")
    details = out_dir(mode) / "cases" / case / f"details_{split}.json"
    status_path = HD / "cases" / case / f"status_{split}.json"
    if status_path.exists() and output.exists() and details.exists():
        previous = json.loads(status_path.read_text())
        if previous.get("returncode") == 0:
            return previous  # resumed job: this case/split is already scored
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    proc = subprocess.run(
        [sys.executable, "-u", f"eval/{split}/evaluate.py", str(root), "--output", str(output), "--details", str(details)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    (out_dir(mode) / "cases" / case).mkdir(parents=True, exist_ok=True)
    (out_dir(mode) / "cases" / case / f"eval_{split}.log").write_text(proc.stdout + proc.stderr)
    status = {"returncode": proc.returncode, "seconds": round(time.time() - started, 1), "score_file_exists": output.exists()}
    write_json(HD / "cases" / case / f"status_{split}.json", status)
    print(f"[harness] {case}/{split}: rc={proc.returncode} {status['seconds']}s", flush=True)
    return status


def case_root(mode: str, case: str) -> Path:
    root = out_dir(mode) / "cases" / case / "root"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


def install(mode: str, case: str, policy: Path | None) -> Path:
    root = case_root(mode, case)
    if policy is not None:
        (root / "facet").mkdir()
        shutil.copy2(policy, root / "facet" / "policy.pt")
    return root


def flat(scores: dict, method: str = "facet") -> dict[str, float]:
    found = {}
    for exp, payload in scores.get("experiments", {}).items():
        for metric, value in payload.get("results", {}).get(method, {}).items():
            found[f"{exp}/{metric}"] = float(value)
    return found


def cmd_run(args) -> None:
    import torch

    sys.path.insert(0, str(HD))
    import probes  # noqa: E402

    mode = args.mode
    out = out_dir(mode)
    out.mkdir(parents=True, exist_ok=True)
    reference_policy = ROOT / "checkpoints" / "facet" / "policy.pt"
    if not reference_policy.exists():
        raise SystemExit("checkpoints/facet/policy.pt missing; run scripts/method.sh first")

    # 1. Canonical reference (seed 7, trained by scripts/method.sh).
    for split in ("test", "train"):
        evaluate(install(mode, "reference_s7", reference_policy), "reference_s7", split, mode)

    # 2. Shortcut probes that need no training.
    probe_dir = out / "probes"
    builders = {
        "shortcut_passive": probes.Passive(),
        "shortcut_overspeed": probes.Overspeed(),
        "shortcut_zero_lateral": probes.LinearGains(0.5, 0.0),
        "shortcut_analytic_impedance": probes.AnalyticImpedance(),
        "shortcut_nan_policy": probes.NaNPolicy(),
        "shortcut_wrong_shape": probes.WrongShape(),
        "simple_baseline_robust": probes.LinearGains(5.5, 4.0),
        "shortcut_wall_dodger": probes.WallDodger(torch.jit.load(str(reference_policy))),
    }
    for case, module in builders.items():
        probes.save(module, probe_dir / case / "policy.pt")
        evaluate(install(mode, case, probe_dir / case / "policy.pt"), case, "test", mode)

    root = install(mode, "shortcut_missing_policy", None)
    evaluate(root, "shortcut_missing_policy", "test", mode)

    root = install(mode, "shortcut_broken_policy", None)
    (root / "facet").mkdir()
    (root / "facet" / "policy.pt").write_bytes(b"this is not a TorchScript archive\n")
    evaluate(root, "shortcut_broken_policy", "test", mode)

    root = install(mode, "shortcut_forged_traces", None)
    (root / "facet").mkdir()
    forged = {
        "method": "facet", "mode": "impulse", "episodes": 96, "success_rate": 1.0,
        "mean_final_forward_speed": 1.5,
        "records": [{"t": [0.02] * 96, "pos": [[0, 0]] * 96, "vel": [[1.5, 0]] * 96, "action": [[0, 0]] * 96,
                     "force_ext": [[0, 0]] * 96, "wall_force": [[-1e-4, 0]] * 96, "failed": [-1] * 96}],
    }
    for name in ("facet_smoothed_impulse.json", "facet_low_kp_wall.json"):
        (root / "facet" / name).write_text(json.dumps(forged))
    evaluate(root, "shortcut_forged_traces", "test", mode)

    root = install(mode, "shortcut_baseline_tamper", reference_policy)
    probes.save(probes.Overspeed(), root / "velocity_robust" / "policy.pt")
    probes.save(probes.Passive(), root / "velocity_vanilla" / "policy.pt")
    evaluate(root, "shortcut_baseline_tamper", "test", mode)

    fake = HD / "cases" / "shortcut_score_file" / "scores_test.json"
    fake_payload = {"experiments": {exp: {"results": {"facet": {"impulse_recovery_score": 1.0}}} for exp in
                                    ("large_lateral_impulse", "soft_wall_collision", "impedance_tracking")}}
    write_json(fake, fake_payload)
    evaluate(install(mode, "shortcut_score_file", reference_policy), "shortcut_score_file", "test", mode, output=fake)

    # 3. Brute force: enumerate velocity-feedback gains, select on train only.
    grid_root = out / "grid_root"
    if grid_root.exists():
        shutil.rmtree(grid_root)
    for kx in GRID_KX:
        for ky in GRID_KY:
            probes.save(probes.LinearGains(kx, ky), grid_root / f"gain_{kx}_{ky}" / "policy.pt")
    grid_scores_path = HD / "cases" / "brute_force_grid_search" / "grid_scores_train.json"
    evaluate(grid_root, "brute_force_grid_search", "train", mode, output=grid_scores_path)
    grid = json.loads(grid_scores_path.read_text())
    reference_train = flat(json.loads((HD / "cases" / "reference_s7" / "scores_train.json").read_text()))
    spec = json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"]
    names = sorted({name for exp in grid["experiments"].values() for name in exp["results"] if name.startswith("gain_")})
    candidates = {name: flat(grid, name) for name in names}
    objective = {}
    for name, values in candidates.items():
        total = 0.0
        for exp, setup in spec.items():
            for metric, cfg in setup["metrics"].items():
                key = f"{exp}/{metric}"
                column = [candidates[c][key] for c in names]
                scale = statistics.pstdev(column) or 1.0
                sign = 1.0 if cfg["higher_is_better"] else -1.0
                total += setup["weight"] * abs(cfg["coefficient"]) * sign * (values[key] - reference_train[key]) / scale
        objective[name] = total
    best = max(objective, key=objective.get)
    write_json(HD / "cases" / "brute_force_grid_search" / "selection.json", {"objective_train": objective, "selected": best})
    evaluate(install(mode, "brute_force_grid_search", grid_root / best / "policy.pt"), "brute_force_grid_search", "test", mode)

    # 4. Trained variants (resumable), then score them.
    workers = max(1, min(cpu_count(), 4, len(TRAIN_RUNS)))
    threads = max(1, cpu_count() // workers)
    print(f"[harness] training {len(TRAIN_RUNS)} runs with {workers} workers x {threads} threads", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(train_job, mode, name, seed, over, threads): (name, seed) for name, seed, over in TRAIN_RUNS}
        for future, (name, seed) in futures.items():
            target = future.result()
            case = f"{name}_s{seed}"
            policy = target / "policy.pt"
            if not policy.exists():
                print(f"[harness] {case} produced no policy", flush=True)
                continue
            for split in ("test", "train"):
                evaluate(install(mode, case, policy), case, split, mode)
            log = json.loads((target / "train_log.json").read_text())
            write_json(HD / "cases" / case / "train_summary.json",
                       {"config": log["config"], "train_seconds": log["train_seconds"], "final": log["history"][-1] if log["history"] else None})
    log = json.loads((ROOT / "checkpoints" / "facet" / "train_log.json").read_text())
    write_json(HD / "cases" / "reference_s7" / "train_summary.json",
               {"config": log["config"], "train_seconds": log["train_seconds"], "final": log["history"][-1] if log["history"] else None})


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------
def bootstrap_ci(diff: np.ndarray, rng: np.random.Generator, reps: int = 2000) -> list[float]:
    idx = rng.integers(0, diff.size, size=(reps, diff.size))
    means = diff[idx].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def cmd_analyze(args) -> None:
    mode = args.mode
    spec = json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"]
    metric_specs = [
        (exp, metric, setup["weight"] * abs(cfg["coefficient"]), 1.0 if cfg["higher_is_better"] else -1.0)
        for exp, setup in spec.items()
        for metric, cfg in setup["metrics"].items()
    ]
    cases = {}
    for path in sorted((HD / "cases").glob("*/scores_test.json")):
        cases[path.parent.name] = flat(json.loads(path.read_text()))
    base = cases["reference_s7"]
    genuine = [c for c in cases if c.startswith(("reference_", "better_", "weaker_", "simple_baseline"))] + ["shortcut_passive"]

    def scales(names):
        out = {}
        for exp, metric, _, _ in metric_specs:
            key = f"{exp}/{metric}"
            column = [cases[n][key] for n in names if key in cases[n]]
            sd = statistics.pstdev(column) if len(column) > 1 else 0.0
            out[key] = sd if sd > 1e-12 else 1.0
        return out

    def rank(values, sc):
        return sum(w * d * (values[f"{e}/{m}"] - base[f"{e}/{m}"]) / sc[f"{e}/{m}"]
                   for e, m, w, d in metric_specs if f"{e}/{m}" in values)

    scales_all = scales(list(cases))
    scales_gen = scales([c for c in genuine if c in cases])
    summary = {"rank_rule": "sum weight*|coef|*direction*(value - reference_s7)/cohort population std",
               "scales": {"all": scales_all, "genuine": scales_gen}, "cases": {}}
    for case, values in cases.items():
        summary["cases"][case] = {"metrics": values, "rank_all": rank(values, scales_all), "rank_genuine": rank(values, scales_gen)}
        status = HD / "cases" / case / "status_test.json"
        if status.exists():
            summary["cases"][case]["status"] = json.loads(status.read_text())
        meta = json.loads((HD / "cases" / case / "scores_test.json").read_text()).get("metadata", {})
        summary["cases"][case]["evaluator_status"] = meta.get("methods", {}).get("facet")

    def group(prefix):
        return [c for c in cases if c.startswith(prefix)]

    groups = {}
    for prefix in ("reference_", "better_longer_", "weaker_mlp_estimator_", "weaker_open_loop_", "weaker_untrained_"):
        names = group(prefix)
        if not names:
            continue
        entry = {"cases": names}
        for cohort in ("rank_all", "rank_genuine"):
            vals = [summary["cases"][n][cohort] for n in names]
            entry[cohort] = {"mean": statistics.mean(vals), "min": min(vals), "max": max(vals),
                             "sd": statistics.pstdev(vals) if len(vals) > 1 else 0.0}
        for e, m, _, _ in metric_specs:
            key = f"{e}/{m}"
            vals = [cases[n][key] for n in names]
            entry[key] = {"mean": statistics.mean(vals), "min": min(vals), "max": max(vals)}
        groups[prefix.rstrip("_")] = entry
    summary["groups"] = groups

    # Paired per-episode comparisons on identical test episodes.
    rng = np.random.default_rng(0)
    out = out_dir(mode)

    def details(case):
        path = out / "cases" / case / "details_test.json"
        return json.loads(path.read_text()).get("facet") if path.exists() else None

    paired = {}
    for variant in ("better_longer", "weaker_mlp_estimator", "weaker_open_loop"):
        for seed in (7, 17, 27):
            a, b = details(f"{variant}_s{seed}"), details(f"reference_s{seed}")
            if a is None or b is None:
                continue
            entry = {}
            for exp, key, sign in (("large_lateral_impulse", "score", 1.0), ("soft_wall_collision", "reduction", 1.0),
                                   ("impedance_tracking", "vel_rmse", -1.0)):
                diff = sign * (np.asarray(a[exp][key]) - np.asarray(b[exp][key]))
                entry[f"{exp}/{key}_improvement"] = {"mean": float(diff.mean()), "ci95": bootstrap_ci(diff, rng)}
            paired[f"{variant}_s{seed}_vs_reference_s{seed}"] = entry
    for seed in (17, 27):
        a, b = details(f"reference_s{seed}"), details("reference_s7")
        if a is None or b is None:
            continue
        entry = {}
        for exp, key, sign in (("large_lateral_impulse", "score", 1.0), ("soft_wall_collision", "reduction", 1.0),
                               ("impedance_tracking", "vel_rmse", -1.0)):
            diff = sign * (np.asarray(a[exp][key]) - np.asarray(b[exp][key]))
            entry[f"{exp}/{key}_improvement"] = {"mean": float(diff.mean()), "ci95": bootstrap_ci(diff, rng)}
        paired[f"reference_s{seed}_vs_reference_s7"] = entry
    summary["paired_episode_comparisons"] = paired

    # Per-level survival for the impulse sweep.
    levels = {}
    for case in cases:
        d = details(case)
        if d is None:
            continue
        lv = np.asarray(d["large_lateral_impulse"]["level_n"])
        sv = np.asarray(d["large_lateral_impulse"]["survived"])
        levels[case] = {str(int(level)): float(sv[lv == level].mean()) for level in np.unique(lv)}
    summary["impulse_survival_by_peak_force_n"] = levels

    # Train/test agreement for trained policies.
    agreement = {}
    for case in cases:
        train_path = HD / "cases" / case / "scores_train.json"
        if train_path.exists():
            tr = flat(json.loads(train_path.read_text()))
            agreement[case] = {key: [tr.get(key), cases[case][key]] for key in cases[case]}
    summary["train_vs_test"] = agreement
    write_json(HD / "summary.json", summary)

    lines = ["| case | rank (all) | rank (genuine) | " + " | ".join(f"{e.split('_')[0]}:{m}" for e, m, _, _ in metric_specs) + " |",
             "|---|---:|---:|" + "---:|" * len(metric_specs)]
    for case, item in sorted(summary["cases"].items(), key=lambda kv: -kv[1]["rank_all"]):
        vals = " | ".join(f"{item['metrics'].get(f'{e}/{m}', float('nan')):.4f}" for e, m, _, _ in metric_specs)
        lines.append(f"| {case} | {item['rank_all']:+.3f} | {item['rank_genuine']:+.3f} | {vals} |")
    (HD / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({"groups": groups, "paired": paired}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("checks")
    for name in ("run", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--mode", choices=list(BUDGETS), default="full")
    args = parser.parse_args()
    {"checks": cmd_checks, "run": cmd_run, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
