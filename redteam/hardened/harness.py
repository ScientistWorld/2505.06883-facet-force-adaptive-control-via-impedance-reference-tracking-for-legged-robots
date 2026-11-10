#!/usr/bin/env python3
"""Hardening harness: reference seeds, quality variants and shortcut probes.

Subcommands
  checks   simulator / metric consistency between data/, method/, eval/train, eval/test
  run      train every variant (parallel, single-threaded so each run equals
           ``SEED=s bash scripts/method.sh``; resumable), score them, install the
           median reference seed as the canonical checkpoints/facet, build the
           shortcut probes on it and score everything with the unmodified
           hidden-test evaluator (plus the train evaluator)
  analyze  ranks under the gym's rule, seed noise and paired bootstrap comparisons

Each case is a checkpoint root whose ``facet/policy.pt`` is the candidate, scored
by ``python eval/test/evaluate.py <root>`` (the code path of
scripts/evaluate_test.sh). Candidate construction and the brute-force selection
never read eval/test or test scores; the canonical-seed choice uses test scores
of reference seeds only (it decides which reference run is shipped, not a
candidate).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
REFERENCE_SEEDS = (7, 17, 27, 37, 47)
# name, seed, overrides (scale multiplies the update budget)
TRAIN_RUNS = (
    [("better_6x", s, {"scale": 6}) for s in (7, 17, 27)]
    + [("better_3x", s, {"scale": 3}) for s in REFERENCE_SEEDS]
    + [("reference", s, {}) for s in REFERENCE_SEEDS]
    + [("weaker_open_loop", s, {"target_mode": "open_loop"}) for s in (7, 17)]
    + [("alt_mlp_estimator", s, {"estimator": "mlp"}) for s in (7, 17)]
    + [("weaker_untrained", 7, {"teacher": 0, "student": 0})]
)
GRID_KX = (0.25, 0.5, 1.0, 2.0, 4.0, 5.5)
GRID_KY = (0.0, 0.1, 0.3, 1.0, 4.0)
TRAINED_PREFIXES = ("reference_", "better_", "weaker_", "alt_")


def out_dir(mode: str) -> Path:
    return ROOT / "outputs" / ("hardening" if mode == "full" else "hardening_smoke")


def cases_dir(mode: str) -> Path:
    return HD / ("cases" if mode == "full" else "cases_smoke")


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


def metric_specs() -> list[tuple[str, str, float, float]]:
    spec = json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"]
    return [
        (exp, metric, setup["weight"] * abs(cfg["coefficient"]), 1.0 if cfg["higher_is_better"] else -1.0)
        for exp, setup in spec.items()
        for metric, cfg in setup["metrics"].items()
    ]


def flat(scores: dict, method: str = "facet") -> dict[str, float]:
    found = {}
    for exp, payload in scores.get("experiments", {}).items():
        for metric, value in payload.get("results", {}).get(method, {}).items():
            found[f"{exp}/{metric}"] = float(value)
    return found


def cohort_scales(cases: dict[str, dict[str, float]], names: list[str]) -> dict[str, float]:
    out = {}
    for exp, metric, _, _ in metric_specs():
        key = f"{exp}/{metric}"
        column = [cases[n][key] for n in names if key in cases[n]]
        sd = statistics.pstdev(column) if len(column) > 1 else 0.0
        out[key] = sd if sd > 1e-12 else 1.0
    return out


def rank_of(values: dict[str, float], base: dict[str, float], scales: dict[str, float]) -> float:
    return sum(
        w * d * (values[f"{e}/{m}"] - base[f"{e}/{m}"]) / scales[f"{e}/{m}"]
        for e, m, w, d in metric_specs()
        if f"{e}/{m}" in values and f"{e}/{m}" in base
    )


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
def train_job(mode: str, name: str, seed: int, overrides: dict) -> Path:
    budget = BUDGETS[mode]
    scale = overrides.get("scale", 1)
    target = out_dir(mode) / "train" / f"{name}_s{seed}"
    if (target / "policy.pt").exists():
        return target
    # Same arguments as scripts/method.sh (single thread), so reference_s<seed>
    # is exactly what `SEED=<seed> bash scripts/method.sh` trains.
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
        "--threads", "1",
    ]
    target.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with open(target / "train.log", "a") as log:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"[harness] trained {name}_s{seed} rc={rc} in {time.time() - started:.0f}s", flush=True)
    return target


def evaluate(root: Path, case: str, split: str, mode: str, output: Path | None = None, legacy: bool = False) -> dict:
    tag = f"{split}_legacy" if legacy else split
    cdir = cases_dir(mode) / case
    output = output or (cdir / f"scores_{tag}.json")
    details = out_dir(mode) / "cases" / case / f"details_{tag}.json"
    status_path = cdir / f"status_{tag}.json"
    if status_path.exists() and output.exists() and details.exists():
        previous = json.loads(status_path.read_text())
        if previous.get("returncode") == 0:
            return previous  # resumed job: this case/split is already scored
    output.parent.mkdir(parents=True, exist_ok=True)
    details.parent.mkdir(parents=True, exist_ok=True)
    if legacy:
        cmd = [sys.executable, "-u", str(HD / "legacy_eval.py"), f"eval/{split}/evaluate.py", str(root), str(output), str(details)]
    else:
        cmd = [sys.executable, "-u", f"eval/{split}/evaluate.py", str(root), "--output", str(output), "--details", str(details)]
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    (details.parent / f"eval_{tag}.log").write_text(proc.stdout + proc.stderr)
    status = {"returncode": proc.returncode, "seconds": round(time.time() - started, 1), "score_file_exists": output.exists()}
    write_json(status_path, status)
    print(f"[harness] {case}/{tag}: rc={proc.returncode} {status['seconds']}s", flush=True)
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


def run_parallel(tasks: list, workers: int) -> list:
    """tasks: list of zero-argument callables; exceptions are reported, not raised."""
    results = []

    def guarded(fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            print(f"[harness] task failed: {type(exc).__name__}: {exc}", flush=True)
            return None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for result in pool.map(guarded, tasks):
            results.append(result)
    return results


def select_canonical(mode: str) -> int:
    """Median reference seed by rank over the trained cohort (test slice)."""
    cases = {}
    for path in sorted(cases_dir(mode).glob("*/scores_test.json")):
        if path.parent.name.startswith(TRAINED_PREFIXES):
            cases[path.parent.name] = flat(json.loads(path.read_text()))
    refs = [f"reference_s{s}" for s in REFERENCE_SEEDS if f"reference_s{s}" in cases]
    if not refs:
        raise SystemExit("no reference seed was scored")
    scales = cohort_scales(cases, list(cases))
    base = cases[refs[0]]
    ranks = {name: rank_of(cases[name], base, scales) for name in refs}
    ordered = sorted(refs, key=lambda n: ranks[n])
    chosen = ordered[(len(ordered) - 1) // 2]
    seed = int(chosen.split("_s")[-1])
    write_json(HD / ("canonical.json" if mode == "full" else "canonical_smoke.json"), {
        "rule": "median reference seed by rank (trained cohort scales, test slice)",
        "seed": seed, "case": chosen, "ranks_relative_to_first_seed": ranks, "order": ordered,
    })
    print(f"[harness] canonical reference: {chosen} (order {ordered})", flush=True)
    return seed


def install_canonical(mode: str, seed: int) -> Path:
    src = out_dir(mode) / "train" / f"reference_s{seed}"
    dst = ROOT / "checkpoints" / "facet" if mode == "full" else out_dir(mode) / "canonical" / "facet"
    dst.mkdir(parents=True, exist_ok=True)
    for item in dst.iterdir():
        if item.is_file():
            item.unlink()
    for name in ("policy.pt", "train_log.json", "agent_state.pt"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    return dst / "policy.pt"


def cmd_run(args) -> None:
    import torch

    sys.path.insert(0, str(HD))
    import probes  # noqa: E402

    mode = args.mode
    out = out_dir(mode)
    out.mkdir(parents=True, exist_ok=True)
    cdir = cases_dir(mode)
    workers = max(1, min(cpu_count(), len(TRAIN_RUNS)))

    # 1. Train every variant (longest first), single-threaded and in parallel.
    def cost(run):
        _, _, over = run
        return over.get("scale", 1) * (0 if over.get("teacher") == 0 else 1)

    runs = sorted(TRAIN_RUNS, key=cost, reverse=True)
    print(f"[harness] training {len(runs)} runs with {workers} workers", flush=True)
    run_parallel([lambda r=r: train_job(mode, r[0], r[1], r[2]) for r in runs], workers)

    # 2. Score every trained case on test and train.
    tasks = []
    for name, seed, _ in runs:
        case = f"{name}_s{seed}"
        target = out / "train" / case
        if not (target / "policy.pt").exists():
            print(f"[harness] {case} produced no policy", flush=True)
            continue
        log = json.loads((target / "train_log.json").read_text())
        write_json(cdir / case / "train_summary.json",
                   {"config": log["config"], "train_seconds": log["train_seconds"],
                    "final": log["history"][-1] if log["history"] else None})
        for split in ("test", "train"):
            # Separate install roots per split: the two scorings run concurrently.
            root_name = case if split == "test" else f"{case}__train"
            tasks.append(lambda c=case, s=split, r=root_name, p=target / "policy.pt": evaluate(install(mode, r, p), c, s, mode))
    run_parallel(tasks, workers)

    # 3. Canonical reference = median reference seed; shipped as checkpoints/facet.
    seed = select_canonical(mode)
    canonical = install_canonical(mode, seed)
    canonical_case = f"reference_s{seed}"

    # 4. Shortcut probes (wrappers build on the canonical reference).
    probe_dir = out / "probes"
    inner = lambda: torch.jit.load(str(canonical))  # noqa: E731
    builders = {
        "shortcut_passive": lambda: probes.Passive(),
        "shortcut_overspeed": lambda: probes.Overspeed(),
        "shortcut_zero_lateral": lambda: probes.LinearGains(0.5, 0.0),
        "shortcut_analytic_impedance": lambda: probes.AnalyticImpedance(),
        "shortcut_analytic_clipped": lambda: probes.AnalyticClipped(2.0),
        "shortcut_nan_policy": lambda: probes.NaNPolicy(),
        "shortcut_wrong_shape": lambda: probes.WrongShape(),
        "simple_baseline_robust": lambda: probes.LinearGains(5.5, 4.0),
        "shortcut_wall_dodger": lambda: probes.WallDodger(inner()),
        "shortcut_retreat_full": lambda: probes.RetreatOnContact(inner(), 16.0),
        "shortcut_retreat_mild": lambda: probes.RetreatOnContact(inner(), 2.5),
        "shortcut_reference_lateral_clip": lambda: probes.LateralClip(inner(), 2.0),
    }
    tasks = []
    for case, build in builders.items():
        try:
            probes.save(build(), probe_dir / case / "policy.pt")
        except Exception as exc:  # noqa: BLE001
            print(f"[harness] could not build {case}: {type(exc).__name__}: {exc}", flush=True)
            continue
        tasks.append(lambda c=case: evaluate(install(mode, c, probe_dir / c / "policy.pt"), c, "test", mode))

    def missing():
        return evaluate(install(mode, "shortcut_missing_policy", None), "shortcut_missing_policy", "test", mode)

    def broken():
        root = install(mode, "shortcut_broken_policy", None)
        (root / "facet").mkdir()
        (root / "facet" / "policy.pt").write_bytes(b"this is not a TorchScript archive\n")
        return evaluate(root, "shortcut_broken_policy", "test", mode)

    def forged():
        root = install(mode, "shortcut_forged_traces", None)
        (root / "facet").mkdir()
        payload = {
            "method": "facet", "mode": "impulse", "episodes": 96, "success_rate": 1.0,
            "mean_final_forward_speed": 1.5,
            "records": [{"t": [0.02] * 96, "pos": [[0, 0]] * 96, "vel": [[1.5, 0]] * 96, "action": [[0, 0]] * 96,
                         "force_ext": [[0, 0]] * 96, "wall_force": [[-1e-4, 0]] * 96, "failed": [-1] * 96}],
        }
        for name in ("facet_smoothed_impulse.json", "facet_low_kp_wall.json"):
            (root / "facet" / name).write_text(json.dumps(payload))
        return evaluate(root, "shortcut_forged_traces", "test", mode)

    def tamper():
        root = install(mode, "shortcut_baseline_tamper", canonical)
        probes.save(probes.Overspeed(), root / "velocity_robust" / "policy.pt")
        probes.save(probes.Passive(), root / "velocity_vanilla" / "policy.pt")
        return evaluate(root, "shortcut_baseline_tamper", "test", mode)

    def score_file():
        fake = cdir / "shortcut_score_file" / "scores_test.json"
        status = cdir / "shortcut_score_file" / "status_test.json"
        if not status.exists():
            write_json(fake, {"experiments": {exp: {"results": {"facet": {"impulse_recovery_score": 1.0}}}
                                              for exp in ("large_lateral_impulse", "soft_wall_collision", "impedance_tracking")}})
        return evaluate(install(mode, "shortcut_score_file", canonical), "shortcut_score_file", "test", mode, output=fake)

    tasks += [missing, broken, forged, tamper, score_file]
    # Pre-fix (retreat gate disabled) scores for the wall-retreat shortcut and its source.
    tasks.append(lambda: evaluate(install(mode, f"{canonical_case}__legacy", canonical), canonical_case, "test", mode, legacy=True))
    for case in ("shortcut_retreat_full", "shortcut_retreat_mild", "shortcut_passive"):
        if (probe_dir / case / "policy.pt").exists():
            tasks.append(lambda c=case: evaluate(install(mode, f"{c}__legacy", probe_dir / c / "policy.pt"), c, "test", mode, legacy=True))

    # 5. Brute force: enumerate velocity-feedback gains, select on train only.
    grid_names = []
    chunks = max(1, min(workers, 6))
    grid_roots = [out / f"grid_root_{i}" for i in range(chunks)]
    for root in grid_roots:
        if root.exists():
            shutil.rmtree(root)
    i = 0
    for kx in GRID_KX:
        for ky in GRID_KY:
            name = f"gain_{kx}_{ky}"
            probes.save(probes.LinearGains(kx, ky), grid_roots[i % chunks] / name / "policy.pt")
            grid_names.append(name)
            i += 1
    for j, root in enumerate(grid_roots):
        tasks.append(lambda r=root, j=j: evaluate(r, f"brute_force_grid_chunk{j}", "train", mode))
    run_parallel(tasks, workers)

    grid: dict[str, dict[str, float]] = {}
    for j in range(chunks):
        path = cdir / f"brute_force_grid_chunk{j}" / "scores_train.json"
        if path.exists():
            payload = json.loads(path.read_text())
            for name in grid_names:
                values = flat(payload, name)
                if values:
                    grid[name] = values
    reference_train = flat(json.loads((cdir / canonical_case / "scores_train.json").read_text()))
    scales = cohort_scales(grid, list(grid))
    objective = {name: rank_of(values, reference_train, scales) for name, values in grid.items()}
    best = max(objective, key=objective.get)
    write_json(cdir / "brute_force_grid_search" / "selection.json", {"objective_train": objective, "selected": best,
                                                                    "candidates_train": grid})
    best_policy = next(root / best / "policy.pt" for root in grid_roots if (root / best / "policy.pt").exists())
    evaluate(install(mode, "brute_force_grid_search", best_policy), "brute_force_grid_search", "test", mode)
    print(f"[harness] run complete; canonical {canonical_case}", flush=True)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------
def bootstrap_ci(diff: np.ndarray, rng: np.random.Generator, reps: int = 2000) -> list[float]:
    idx = rng.integers(0, diff.size, size=(reps, diff.size))
    means = diff[idx].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def episode_metrics(d: dict, idx: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Recompute every scored metric from per-episode details on resampled indices [reps, n]."""
    out = {}
    imp = d["large_lateral_impulse"]
    i = idx["large_lateral_impulse"]
    out["large_lateral_impulse/impulse_recovery_score"] = np.asarray(imp["score"])[i].mean(1)
    out["large_lateral_impulse/success_rate"] = np.asarray(imp["survived"])[i].mean(1)
    out["large_lateral_impulse/velocity_tracking_rmse_mps"] = np.asarray(imp["rmse"])[i].mean(1)
    wall = d["soft_wall_collision"]
    i = idx["soft_wall_collision"]
    charged = np.asarray(wall["charged"])[i].sum(1)
    base = np.maximum(np.asarray(wall["baseline_impulse"])[i].sum(1), 1e-9)
    out["soft_wall_collision/post_contact_impulse_reduction"] = np.clip(1.0 - charged / base, -1.0, 1.0)
    out["soft_wall_collision/wall_contact_rate"] = np.asarray(wall["valid"])[i].mean(1)
    out["soft_wall_collision/entry_speed_error_mps"] = np.asarray(wall["entry_error"])[i].mean(1)
    out["soft_wall_collision/contact_tracking_rmse_mps"] = np.asarray(wall["contact_rmse"])[i].mean(1)
    trk = d["impedance_tracking"]
    i = idx["impedance_tracking"]
    out["impedance_tracking/velocity_tracking_rmse_mps"] = np.asarray(trk["vel_rmse"])[i].mean(1)
    out["impedance_tracking/position_tracking_rmse_m"] = np.asarray(trk["pos_rmse"])[i].mean(1)
    out["impedance_tracking/success_rate"] = np.asarray(trk["survived"])[i].mean(1)
    return out


def paired_rank_ci(a: dict, b: dict, scales: dict[str, float], rng: np.random.Generator, reps: int = 1000) -> dict:
    """Bootstrap (episodes resampled jointly for both policies) of rank(a) - rank(b)."""
    sizes = {exp: len(a[exp][key]) for exp, key in (("large_lateral_impulse", "score"),
                                                     ("soft_wall_collision", "charged"),
                                                     ("impedance_tracking", "vel_rmse"))}
    idx = {exp: rng.integers(0, n, size=(reps, n)) for exp, n in sizes.items()}
    point = {exp: np.arange(n)[None, :] for exp, n in sizes.items()}
    ma, mb = episode_metrics(a, idx), episode_metrics(b, idx)
    pa, pb = episode_metrics(a, point), episode_metrics(b, point)
    diff = np.zeros(reps)
    est = 0.0
    for e, m, w, d in metric_specs():
        key = f"{e}/{m}"
        diff += w * d * (ma[key] - mb[key]) / scales[key]
        est += float(w * d * (pa[key][0] - pb[key][0]) / scales[key])
    return {"estimate": est, "ci95": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))]}


def cmd_analyze(args) -> None:
    mode = args.mode
    cdir = cases_dir(mode)
    out = out_dir(mode)
    specs = metric_specs()
    canonical = json.loads((HD / ("canonical.json" if mode == "full" else "canonical_smoke.json")).read_text())
    base_case = canonical["case"]
    cases = {}
    for path in sorted(cdir.glob("*/scores_test.json")):
        values = flat(json.loads(path.read_text()))
        if values:
            cases[path.parent.name] = values
    base = cases[base_case]
    trained = [c for c in cases if c.startswith(TRAINED_PREFIXES)]
    genuine = trained + [c for c in ("simple_baseline_robust", "shortcut_passive", "shortcut_analytic_impedance",
                                     "brute_force_grid_search") if c in cases]
    cohorts = {"all": list(cases), "genuine": genuine, "trained": trained}
    scales = {name: cohort_scales(cases, members) for name, members in cohorts.items()}
    summary = {"canonical": canonical,
               "rank_rule": "sum weight*|coef|*direction*(value - canonical reference)/cohort population std",
               "cohorts": cohorts, "scales": scales, "cases": {}}
    for case, values in cases.items():
        entry = {"metrics": values}
        for name in cohorts:
            entry[f"rank_{name}"] = rank_of(values, base, scales[name])
        status = cdir / case / "status_test.json"
        if status.exists():
            entry["status"] = json.loads(status.read_text())
        meta = json.loads((cdir / case / "scores_test.json").read_text()).get("metadata", {})
        entry["evaluator_status"] = meta.get("methods", {}).get("facet")
        legacy = cdir / case / "scores_test_legacy.json"
        if legacy.exists():
            lv = flat(json.loads(legacy.read_text()))
            entry["legacy_metrics"] = lv
            entry["legacy_rank_all_vs_legacy_reference"] = None
        summary["cases"][case] = entry
    legacy_base = cdir / base_case / "scores_test_legacy.json"
    if legacy_base.exists():
        lb = flat(json.loads(legacy_base.read_text()))
        for case, entry in summary["cases"].items():
            if "legacy_metrics" in entry:
                entry["legacy_rank_all_vs_legacy_reference"] = rank_of(entry["legacy_metrics"], lb, scales["all"])
                entry["legacy_rank_genuine_vs_legacy_reference"] = rank_of(entry["legacy_metrics"], lb, scales["genuine"])

    groups = {}
    for prefix in ("reference_", "better_3x_", "better_6x_", "alt_mlp_estimator_", "weaker_open_loop_", "weaker_untrained_"):
        names = [c for c in cases if c.startswith(prefix)]
        if not names:
            continue
        entry = {"cases": names}
        for name in cohorts:
            vals = [summary["cases"][n][f"rank_{name}"] for n in names]
            entry[f"rank_{name}"] = {"mean": statistics.mean(vals), "min": min(vals), "max": max(vals),
                                     "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0}
        for e, m, _, _ in specs:
            key = f"{e}/{m}"
            vals = [cases[n][key] for n in names]
            entry[key] = {"mean": statistics.mean(vals), "min": min(vals), "max": max(vals)}
        groups[prefix.rstrip("_")] = entry
    summary["groups"] = groups

    rng = np.random.default_rng(0)

    def details(case: str):
        path = out / "cases" / case / "details_test.json"
        return json.loads(path.read_text()).get("facet") if path.exists() else None

    # Paired, same-episode comparisons: composite rank difference vs the canonical
    # reference for every case, and primary-metric deltas for same-seed pairs.
    base_details = details(base_case)
    vs_canonical = {}
    if base_details is not None:
        for case in cases:
            d = details(case)
            if d is None or case == base_case:
                continue
            vs_canonical[case] = {name: paired_rank_ci(d, base_details, scales[name], rng) for name in ("all", "genuine")}
    summary["paired_rank_vs_canonical"] = vs_canonical

    pairs = []
    for s in REFERENCE_SEEDS:
        pairs += [(f"better_3x_s{s}", f"reference_s{s}"), (f"better_6x_s{s}", f"reference_s{s}"), (f"better_6x_s{s}", f"better_3x_s{s}"),
                  (f"alt_mlp_estimator_s{s}", f"reference_s{s}"), (f"weaker_open_loop_s{s}", f"reference_s{s}"),
                  (f"weaker_untrained_s{s}", f"reference_s{s}")]
    for case in ("shortcut_retreat_full", "shortcut_retreat_mild", "shortcut_wall_dodger"):
        pairs.append((case, base_case))
    paired = {}
    for a_name, b_name in pairs:
        a, b = details(a_name), details(b_name)
        if a is None or b is None:
            continue
        entry = {"rank_diff": {name: paired_rank_ci(a, b, scales[name], rng) for name in ("all", "genuine")}}
        for exp, key, sign in (("large_lateral_impulse", "score", 1.0), ("soft_wall_collision", "reduction", 1.0),
                               ("impedance_tracking", "vel_rmse", -1.0)):
            diff = sign * (np.asarray(a[exp][key]) - np.asarray(b[exp][key]))
            entry[f"{exp}/{key}_improvement"] = {"mean": float(diff.mean()), "ci95": bootstrap_ci(diff, rng)}
        paired[f"{a_name}_vs_{b_name}"] = entry
    summary["paired_episode_comparisons"] = paired

    levels, retreat = {}, {}
    for case in cases:
        d = details(case)
        if d is None:
            continue
        lv = np.asarray(d["large_lateral_impulse"]["level_n"])
        sv = np.asarray(d["large_lateral_impulse"]["survived"])
        levels[case] = {str(int(level)): float(sv[lv == level].mean()) for level in np.unique(lv)}
        wall = d["soft_wall_collision"]
        rr = np.asarray(wall.get("retreat_rms", [np.nan]))
        retreat[case] = {"max": float(np.nanmax(rr)), "p95": float(np.nanquantile(rr, 0.95)), "mean": float(np.nanmean(rr))}
        if "measured_impulse" in wall:
            ratio = np.asarray(wall["measured_impulse"]) / np.maximum(np.asarray(wall["commanded_impulse"]), 1e-9)
            retreat[case]["measured_over_commanded"] = {"mean": float(ratio.mean()), "p05": float(np.quantile(ratio, 0.05)),
                                                        "p95": float(np.quantile(ratio, 0.95))}
    summary["impulse_survival_by_peak_force_n"] = levels
    summary["wall_retreat_rms_per_episode"] = retreat

    agreement = {}
    for case in cases:
        train_path = cdir / case / "scores_train.json"
        if train_path.exists():
            tr = flat(json.loads(train_path.read_text()))
            agreement[case] = {key: [tr.get(key), cases[case][key]] for key in cases[case]}
    summary["train_vs_test"] = agreement
    write_json(HD / ("summary.json" if mode == "full" else "summary_smoke.json"), summary)

    lines = ["| case | rank (all) | rank (genuine) | rank (trained) | " + " | ".join(f"{e.split('_')[0]}:{m}" for e, m, _, _ in specs) + " |",
             "|---|---:|---:|---:|" + "---:|" * len(specs)]
    for case, item in sorted(summary["cases"].items(), key=lambda kv: -kv[1]["rank_all"]):
        vals = " | ".join(f"{item['metrics'].get(f'{e}/{m}', float('nan')):.4f}" for e, m, _, _ in specs)
        lines.append(f"| {case} | {item['rank_all']:+.3f} | {item['rank_genuine']:+.3f} | {item['rank_trained']:+.3f} | {vals} |")
    (HD / ("summary.md" if mode == "full" else "summary_smoke.md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({"canonical": canonical, "groups": groups}, indent=1))


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
