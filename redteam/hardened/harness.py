#!/usr/bin/env python3
"""Hardening harness (final round): reference configurations, quality variants and shortcut probes.

Subcommands (each resumable; finished trainings and scorings on disk are reused)
  checks          simulator / metric consistency between data/, method/, eval/train, eval/test
  score-existing  score every reduced-network run trained in the previous round
                  (test + train), the stand-alone controller probes, and the
                  brute-force gain grid (selected on train only)
  train-paper     train the paper-size-network runs (single-threaded, in parallel;
                  each equals ``SEED=s bash scripts/method.sh`` with the same flags)
  finalize        score the paper-size runs, choose the reference configuration
                  (pre-registered rule, train slice), install its median seed as
                  checkpoints/facet, build and score the probes that wrap it, and
                  score a few cases under the previous round's protocol ("v2")
  analyze         ranks under the gym's rule, seed noise, paired bootstrap comparisons

Each case is a checkpoint root whose ``facet/policy.pt`` is the candidate, scored
by ``python eval/<split>/evaluate.py <root>`` (the code path of
scripts/evaluate_<split>.sh). Candidate construction, the brute-force selection,
the configuration choice and the canonical-seed choice use the train slice only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
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

TAG = "v3"
BUDGETS = {
    # Reference budget: 600 teacher + 150 student PPO updates of 256 envs x 96 steps
    # (14.7M + 3.7M env steps; the paper's 4:1 teacher/student split).
    "full": {"teacher": 600, "student": 150, "envs": 256, "steps": 96},
    "smoke": {"teacher": 2, "student": 1, "envs": 32, "steps": 16},
}
REFERENCE_SEEDS = (7, 17, 27, 37, 47)
# name, seed, overrides (scale multiplies the update budget; net = network sizes)
REDUCED_RUNS = (
    [("better_2x_budget", s, {"scale": 2}) for s in (7, 17, 27)]
    + [("reference", s, {}) for s in REFERENCE_SEEDS]
    + [("weaker_open_loop", s, {"target_mode": "open_loop"}) for s in (7, 17)]
    + [("alt_mlp_estimator", s, {"estimator": "mlp"}) for s in (7, 17)]
    + [("weaker_third_budget", s, {"teacher": 200, "student": 50}) for s in REFERENCE_SEEDS]
    + [("weaker_open_loop_third_budget", s, {"target_mode": "open_loop", "teacher": 200, "student": 50}) for s in (7, 17)]
    + [("alt_mlp_estimator_third_budget", s, {"estimator": "mlp", "teacher": 200, "student": 50}) for s in (7, 17)]
    + [("weaker_untrained", 7, {"teacher": 0, "student": 0})]
)
# Sized to 8 cores: 3 single-threaded reference seeds, two 2x-budget runs, one
# open-loop run and an untrained export. The 2x-budget runs used 2 threads until
# student update 170/200 of 300 (first job, preempted on a contended node) and
# finished single-threaded after an exact resume; threads change speed only.
PAPER_SEEDS = (7, 17, 27)
PAPER_RUNS = (
    [("paper_nets_2x_budget", s, {"net": "paper", "scale": 2}) for s in (7, 17)]
    + [("paper_nets", s, {"net": "paper"}) for s in PAPER_SEEDS]
    + [("paper_nets_open_loop", 7, {"net": "paper", "target_mode": "open_loop"})]
    + [("paper_nets_untrained", 7, {"net": "paper", "teacher": 0, "student": 0})]
)
CONFIG_SEEDS = {"reduced": REFERENCE_SEEDS, "paper": PAPER_SEEDS}
TRAIN_RUNS = REDUCED_RUNS + PAPER_RUNS
CONFIG_GROUPS = {"reduced": "reference", "paper": "paper_nets"}
GRID_KX = (0.25, 0.5, 1.0, 2.0, 4.0, 5.5)
GRID_KY = (0.0, 0.1, 0.3, 1.0, 4.0)
TRAINED_PREFIXES = ("reference_", "better_", "weaker_", "alt_", "paper_nets")
HONEST_PROBES = ("simple_baseline_robust", "shortcut_passive", "shortcut_zero_lateral", "shortcut_analytic_impedance",
                 "shortcut_analytic_clipped", "brute_force_grid_search")


def out_dir(mode: str) -> Path:
    return ROOT / "outputs" / ("hardening" if mode == "full" else "hardening_smoke")


def cases_dir(mode: str) -> Path:
    return HD / (f"cases_{TAG}" if mode == "full" else f"cases_{TAG}_smoke")


def detail_dir(mode: str) -> Path:
    return out_dir(mode) / f"cases_{TAG}"


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
        column = [cases[n][key] for n in names if n in cases and key in cases[n]]
        sd = statistics.pstdev(column) if len(column) > 1 else 0.0
        out[key] = sd if sd > 1e-12 else 1.0
    return out


def rank_of(values: dict[str, float], base: dict[str, float], scales: dict[str, float]) -> float:
    return sum(
        w * d * (values[f"{e}/{m}"] - base[f"{e}/{m}"]) / scales[f"{e}/{m}"]
        for e, m, w, d in metric_specs()
        if f"{e}/{m}" in values and f"{e}/{m}" in base
    )


def group_members(names, group: str) -> list[str]:
    pattern = re.compile(rf"^{re.escape(group)}_s\d+$")
    return sorted((n for n in names if pattern.match(n)), key=lambda n: int(n.rsplit("_s", 1)[1]))


def is_trained(name: str) -> bool:
    return name.startswith(TRAINED_PREFIXES) and "__" not in name


def load_cases(mode: str, split: str) -> dict[str, dict[str, float]]:
    cases = {}
    for path in sorted(cases_dir(mode).glob(f"*/scores_{split}.json")):
        if path.parent.name.startswith("brute_force_grid_chunk"):
            continue  # grid candidates live under their own method names; 'facet' there is an empty slot
        values = flat(json.loads(path.read_text()))
        if values:
            cases[path.parent.name] = values
    return cases


def genuine_names(cases) -> list[str]:
    return [c for c in cases if is_trained(c)] + [c for c in HONEST_PROBES if c in cases]


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
    report["wall_virtual_mass"] = float(test_eval.wall_scenario(1).virtual_mass[0])
    report["episodes_test"] = {"impulse": int(test_eval.impulse_scenario(1)[0].n), "wall": int(test_eval.wall_scenario(1).n),
                               "tracking": int(test_eval.tracking_scenario(1).n)}
    report["episodes_train"] = {"impulse": int(train_eval.impulse_scenario(1)[0].n), "wall": int(train_eval.wall_scenario(1).n),
                                "tracking": int(train_eval.tracking_scenario(1).n)}

    # Training reward targets (method) vs evaluator tracking metric on one trajectory.
    sc = test_eval.tracking_scenario(321).take(np.arange(64))
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

    # Wall operating point: commanded-impedance and passive impulse relative to the robust baseline.
    sc = test_eval.wall_scenario(test_eval.SEEDS["soft_wall_collision"])
    base = test_eval.wall_episodes(test_eval.rollout(test_eval.VelocityBaseline(5.5, 4.0), sc, test_eval.WALL_STEPS), sc)

    class Zero:
        def make(self, n):
            return lambda obs: np.zeros((n, 2))

    passive = test_eval.wall_episodes(test_eval.rollout(Zero(), sc, test_eval.WALL_STEPS), sc)
    report["wall_commanded_reduction_on_passive_states"] = float(1 - passive["commanded_impulse"].sum() / base["measured_impulse"].sum())
    report["wall_passive_measured_reduction"] = float(1 - passive["measured_impulse"].sum() / base["measured_impulse"].sum())
    report["ok"] = all(v < 1e-9 for key, v in report.items() if key.endswith("diff"))
    write_json(HD / f"checks_{TAG}.json", report)
    print(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------
# training and scoring helpers
# ---------------------------------------------------------------------------
def run_budget(mode: str, overrides: dict) -> tuple[int, int]:
    budget = BUDGETS[mode]
    scale = overrides.get("scale", 1)
    return overrides.get("teacher", budget["teacher"] * scale), overrides.get("student", budget["student"] * scale)


def train_job(mode: str, name: str, seed: int, overrides: dict) -> Path:
    budget = BUDGETS[mode]
    teacher, student = run_budget(mode, overrides)
    target = out_dir(mode) / "train" / f"{name}_s{seed}"
    if (target / "policy.pt").exists():
        return target
    # Same arguments as scripts/method.sh (single thread), so a run equals
    # `SEED=<seed> bash scripts/method.sh` with the matching flags.
    cmd = [
        sys.executable, "-u", "-m", "method.run_facet",
        "--output-dir", str(target),
        "--seed", str(seed),
        "--teacher-updates", str(teacher),
        "--student-updates", str(student),
        "--n-envs", str(budget["envs"]),
        "--rollout-steps", str(budget["steps"]),
        "--estimator", overrides.get("estimator", "gru"),
        "--target-mode", overrides.get("target_mode", "smoothed"),
        "--net-size", overrides.get("net", "reduced"),
        "--threads", str(overrides.get("threads", 1)),
    ]
    target.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with open(target / "train.log", "a") as log:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"[harness] trained {name}_s{seed} rc={rc} in {time.time() - started:.0f}s", flush=True)
    return target


def evaluate(root: Path, case: str, split: str, mode: str, output: Path | None = None, legacy: bool = False) -> dict:
    tag = f"{split}_v2" if legacy else split
    cdir = cases_dir(mode) / case
    output = output or (cdir / f"scores_{tag}.json")
    details = detail_dir(mode) / case / f"details_{tag}.json"
    status_path = cdir / f"status_{tag}.json"
    if status_path.exists() and output.exists() and details.exists():
        previous = json.loads(status_path.read_text())
        if previous.get("returncode") == 0:
            return previous  # resumed job: this case/split is already scored
    output.parent.mkdir(parents=True, exist_ok=True)
    details.parent.mkdir(parents=True, exist_ok=True)
    if legacy:
        cmd = [sys.executable, "-u", str(HD / "legacy_v2_eval.py"), f"eval/{split}/evaluate.py", str(root), str(output), str(details)]
    else:
        cmd = [sys.executable, "-u", f"eval/{split}/evaluate.py", str(root), "--output", str(output), "--details", str(details)]
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    (details.parent / f"eval_{tag}.log").write_text(proc.stdout + proc.stderr)
    status = {"returncode": proc.returncode, "seconds": round(time.time() - started, 1), "score_file_exists": output.exists()}
    write_json(status_path, status)
    print(f"[harness] {case}/{tag}: rc={proc.returncode} {status['seconds']}s", flush=True)
    return status


def install(mode: str, case: str, policy: Path | None) -> Path:
    root = detail_dir(mode) / case / "root"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    if policy is not None:
        (root / "facet").mkdir()
        shutil.copy2(policy, root / "facet" / "policy.pt")
    return root


def run_parallel(tasks: list, workers: int) -> list:
    """tasks: list of zero-argument callables; exceptions are reported, not raised."""

    def guarded(fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            print(f"[harness] task failed: {type(exc).__name__}: {exc}", flush=True)
            return None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(guarded, tasks))


def score_runs(mode: str, runs, splits=("test", "train"), legacy: bool = False) -> list:
    tasks = []
    for name, seed, _ in runs:
        case = f"{name}_s{seed}"
        target = out_dir(mode) / "train" / case
        if not (target / "policy.pt").exists():
            print(f"[harness] {case} has no policy", flush=True)
            continue
        log = json.loads((target / "train_log.json").read_text())
        write_json(cases_dir(mode) / case / "train_summary.json",
                   {"config": log["config"], "train_seconds": log["train_seconds"],
                    "final": log["history"][-1] if log["history"] else None})
        for split in splits:
            root_name = f"{case}__{split}" + ("_v2" if legacy else "")
            tasks.append(lambda c=case, s=split, r=root_name, p=target / "policy.pt": evaluate(install(mode, r, p), c, s, mode, legacy=legacy))
    return tasks


# ---------------------------------------------------------------------------
# score-existing
# ---------------------------------------------------------------------------
def cmd_score_existing(args) -> None:
    import torch  # noqa: F401  (probes need torch)

    sys.path.insert(0, str(HD))
    import probes  # noqa: E402

    mode = args.mode
    out = out_dir(mode)
    workers = cpu_count()
    if mode == "smoke":
        run_parallel([lambda r=r: train_job(mode, *r) for r in REDUCED_RUNS], workers)
    tasks = score_runs(mode, REDUCED_RUNS)

    probe_dir = out / f"probes_{TAG}"
    builders = {
        "shortcut_passive": lambda: probes.Passive(),
        "shortcut_overspeed": lambda: probes.Overspeed(),
        "shortcut_zero_lateral": lambda: probes.LinearGains(0.5, 0.0),
        "shortcut_analytic_impedance": lambda: probes.AnalyticImpedance(),
        "shortcut_analytic_clipped": lambda: probes.AnalyticClipped(2.0),
        "shortcut_nan_policy": lambda: probes.NaNPolicy(),
        "shortcut_wrong_shape": lambda: probes.WrongShape(),
        "simple_baseline_robust": lambda: probes.LinearGains(5.5, 4.0),
        "probe_dob_impedance": lambda: probes.DOBImpedance(),
    }
    for case, build in builders.items():
        try:
            probes.save(build(), probe_dir / case / "policy.pt")
        except Exception as exc:  # noqa: BLE001
            print(f"[harness] could not build {case}: {type(exc).__name__}: {exc}", flush=True)
            continue
        tasks.append(lambda c=case: evaluate(install(mode, c, probe_dir / c / "policy.pt"), c, "test", mode))
    # The out-of-bounds model-based probe is also scored on train (headroom check on both slices).
    tasks.append(lambda: evaluate(install(mode, "probe_dob_impedance__train", probe_dir / "probe_dob_impedance" / "policy.pt"),
                                  "probe_dob_impedance", "train", mode))

    def missing():
        return evaluate(install(mode, "shortcut_missing_policy", None), "shortcut_missing_policy", "test", mode)

    def broken():
        root = install(mode, "shortcut_broken_policy", None)
        (root / "facet").mkdir()
        (root / "facet" / "policy.pt").write_bytes(b"this is not a TorchScript archive\n")
        return evaluate(root, "shortcut_broken_policy", "test", mode)

    def forged():
        # Saved rollout traces with perfect summaries and impossible flags, no policy.
        root = install(mode, "shortcut_forged_traces", None)
        (root / "facet").mkdir()
        payload = {
            "method": "facet", "mode": "impulse", "episodes": 96, "success_rate": 2.0,
            "mean_final_forward_speed": 1.5,
            "records": [{"t": [0.02] * 96, "pos": [[0, 0]] * 96, "vel": [[1.5, 0]] * 96, "action": [[0, 0]] * 96,
                         "force_ext": [[0, 0]] * 96, "wall_force": [[-1e-4, 0]] * 96, "failed": [-1] * 96}],
        }
        for name in ("facet_smoothed_impulse.json", "facet_low_kp_wall.json"):
            (root / "facet" / name).write_text(json.dumps(payload))
        return evaluate(root, "shortcut_forged_traces", "test", mode)

    tasks += [missing, broken, forged]

    # Brute force: enumerate velocity-feedback gains, select on train only.
    grid_names = []
    chunks = max(1, min(workers, 6))
    grid_roots = [out / f"grid_{TAG}_{i}" for i in range(chunks)]
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
        path = cases_dir(mode) / f"brute_force_grid_chunk{j}" / "scores_train.json"
        if path.exists():
            payload = json.loads(path.read_text())
            for name in grid_names:
                values = flat(payload, name)
                if values:
                    grid[name] = values
    # argmax of the rank is independent of the base; the spread is the grid's own.
    scales = cohort_scales(grid, list(grid))
    any_base = next(iter(grid.values()))
    objective = {name: rank_of(values, any_base, scales) for name, values in grid.items()}
    best = max(objective, key=objective.get)
    write_json(cases_dir(mode) / "brute_force_grid_search" / "selection.json",
               {"objective_train": objective, "selected": best, "candidates_train": grid})
    best_policy = next(root / best / "policy.pt" for root in grid_roots if (root / best / "policy.pt").exists())
    run_parallel([lambda: evaluate(install(mode, "brute_force_grid_search", best_policy), "brute_force_grid_search", "test", mode),
                  lambda: evaluate(install(mode, "brute_force_grid_search__train", best_policy), "brute_force_grid_search", "train", mode)],
                 workers)
    print("[harness] score-existing complete", flush=True)


# ---------------------------------------------------------------------------
# train-paper
# ---------------------------------------------------------------------------
def cmd_train_paper(args) -> None:
    mode = args.mode

    def cost(run):
        _, _, over = run
        teacher, student = run_budget(mode, over)
        return (teacher + student) * (3.0 if over.get("net") == "paper" else 1.0)

    runs = sorted(PAPER_RUNS, key=cost, reverse=True)
    workers = len(runs)  # 8 threads in total (see PAPER_RUNS); the untrained export takes seconds
    print(f"[harness] training {len(runs)} paper-size runs with {workers} workers", flush=True)
    run_parallel([lambda r=r: train_job(mode, *r) for r in runs], workers)
    print("[harness] train-paper complete", flush=True)


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------
def choose_configuration(mode: str) -> dict:
    """Pre-registered rule (train slice, genuine cohort scales): the paper-size
    configuration becomes the reference if all its seeds trained and their mean
    rank is at least the reduced configuration's mean rank."""
    cases = load_cases(mode, "train")
    scales = cohort_scales(cases, genuine_names(cases))
    groups = {cfg: group_members(cases, prefix) for cfg, prefix in CONFIG_GROUPS.items()}
    base = cases[groups["reduced"][0]]
    ranks = {cfg: {n: rank_of(cases[n], base, scales) for n in names} for cfg, names in groups.items()}
    means = {cfg: (statistics.mean(r.values()) if r else float("-inf")) for cfg, r in ranks.items()}
    complete = len(groups["paper"]) == len(CONFIG_SEEDS["paper"])
    chosen = "paper" if complete and means["paper"] >= means["reduced"] else "reduced"
    decision = {"rule": "paper if all paper seeds trained and mean train rank(paper) >= mean train rank(reduced); "
                        "genuine-cohort scales on the train slice",
                "ranks_train": ranks, "mean_rank_train": means, "chosen": chosen}
    write_json(HD / f"decision_{TAG}.json", decision)
    print(f"[harness] configuration: {chosen} {means}", flush=True)
    return decision


def select_canonical(mode: str, config: str) -> str:
    """Median seed of the chosen configuration by rank (genuine cohort, train slice)."""
    cases = load_cases(mode, "train")
    refs = group_members(cases, CONFIG_GROUPS[config])
    if not refs:
        raise SystemExit("no reference seed was scored")
    scales = cohort_scales(cases, genuine_names(cases))
    base = cases[refs[0]]
    ranks = {name: rank_of(cases[name], base, scales) for name in refs}
    ordered = sorted(refs, key=lambda n: ranks[n])
    chosen = ordered[(len(ordered) - 1) // 2]
    write_json(HD / (f"canonical_{TAG}.json" if mode == "full" else f"canonical_{TAG}_smoke.json"), {
        "rule": "median seed of the chosen configuration by rank (genuine cohort scales, train slice)",
        "configuration": config, "case": chosen, "seed": int(chosen.rsplit("_s", 1)[1]),
        "ranks_train_relative_to_first_seed": ranks, "order": ordered,
    })
    print(f"[harness] canonical reference: {chosen} (order {ordered})", flush=True)
    return chosen


def install_canonical(mode: str, case: str) -> Path:
    src = out_dir(mode) / "train" / case
    dst = ROOT / "checkpoints" / "facet" if mode == "full" else out_dir(mode) / "canonical" / "facet"
    dst.mkdir(parents=True, exist_ok=True)
    for item in dst.iterdir():
        if item.is_file():
            item.unlink()
    for name in ("policy.pt", "train_log.json", "agent_state.pt"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    return dst / "policy.pt"


def cmd_finalize(args) -> None:
    import torch

    sys.path.insert(0, str(HD))
    import probes  # noqa: E402

    mode = args.mode
    out = out_dir(mode)
    workers = cpu_count()
    if mode == "smoke":
        run_parallel([lambda r=r: train_job(mode, *r) for r in PAPER_RUNS], workers)
    run_parallel(score_runs(mode, PAPER_RUNS), workers)

    decision = choose_configuration(mode)
    canonical_case = select_canonical(mode, decision["chosen"])
    canonical = install_canonical(mode, canonical_case)

    probe_dir = out / f"probes_{TAG}"
    inner = lambda: torch.jit.load(str(canonical))  # noqa: E731
    builders = {
        "shortcut_wall_dodger": lambda: probes.WallDodger(inner()),
        "shortcut_retreat_full": lambda: probes.RetreatOnContact(inner(), 16.0),
        "shortcut_retreat_mild": lambda: probes.RetreatOnContact(inner(), 2.5),
        "shortcut_reference_lateral_clip": lambda: probes.LateralClip(inner(), 2.0),
    }
    tasks = []
    for case, build in builders.items():
        probes.save(build(), probe_dir / case / "policy.pt")
        tasks.append(lambda c=case: evaluate(install(mode, c, probe_dir / c / "policy.pt"), c, "test", mode))

    def tamper():
        root = install(mode, "shortcut_baseline_tamper", canonical)
        probes.save(probes.Overspeed(), root / "velocity_robust" / "policy.pt")
        probes.save(probes.Passive(), root / "velocity_vanilla" / "policy.pt")
        return evaluate(root, "shortcut_baseline_tamper", "test", mode)

    def score_file():
        fake = cases_dir(mode) / "shortcut_score_file" / "scores_test.json"
        status = cases_dir(mode) / "shortcut_score_file" / "status_test.json"
        if not status.exists():
            write_json(fake, {"experiments": {exp: {"results": {"facet": {"impulse_recovery_score": 1.0}}}
                                              for exp in ("large_lateral_impulse", "soft_wall_collision", "impedance_tracking")}})
        return evaluate(install(mode, "shortcut_score_file", canonical), "shortcut_score_file", "test", mode, output=fake)

    tasks += [tamper, score_file]
    # Previous round's protocol (wall virtual mass 12 kg, 32 pushes per level, 128 wall and
    # 256 tracking episodes) for the paper-size runs and the new probes: before/after values.
    tasks += score_runs(mode, PAPER_RUNS, splits=("test",), legacy=True)
    # Consistency: the v2 wrapper must reproduce the previous round's stored scores of its canonical.
    tasks += score_runs(mode, [("reference", 27, {})], splits=("test",), legacy=True)
    for case in ("probe_dob_impedance", "shortcut_passive", "shortcut_analytic_clipped"):
        tasks.append(lambda c=case: evaluate(install(mode, f"{c}__v2", probe_dir / c / "policy.pt"), c, "test", mode, legacy=True))
    run_parallel(tasks, workers)
    print(f"[harness] finalize complete; canonical {canonical_case}", flush=True)


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
    specs = metric_specs()
    canonical_path = HD / (f"canonical_{TAG}.json" if mode == "full" else f"canonical_{TAG}_smoke.json")
    if canonical_path.exists():
        canonical = json.loads(canonical_path.read_text())
    else:  # interim analysis before finalize: median reduced-network seed on train
        train_cases = load_cases(mode, "train")
        refs = group_members(train_cases, "reference")
        scales = cohort_scales(train_cases, genuine_names(train_cases))
        ranks = {n: rank_of(train_cases[n], train_cases[refs[0]], scales) for n in refs}
        ordered = sorted(refs, key=lambda n: ranks[n])
        canonical = {"case": ordered[(len(ordered) - 1) // 2], "interim": True, "order": ordered}
    base_case = canonical["case"]
    cases = load_cases(mode, "test")
    base = cases[base_case]
    trained = [c for c in cases if is_trained(c)]
    genuine = genuine_names(cases)
    cohorts = {"all": list(cases), "genuine": genuine, "trained": trained}
    scales = {name: cohort_scales(cases, members) for name, members in cohorts.items()}
    summary = {"canonical": canonical,
               "rank_rule": "sum weight*|coef|*direction*(value - canonical reference)/cohort population std",
               "cohorts": cohorts, "scales": scales, "cases": {}}
    legacy_cases = {}
    for path in sorted(cdir.glob("*/scores_test_v2.json")):
        values = flat(json.loads(path.read_text()))
        if values:
            legacy_cases[path.parent.name] = values
    for case, values in cases.items():
        entry = {"metrics": values}
        for name in cohorts:
            entry[f"rank_{name}"] = rank_of(values, base, scales[name])
        status = cdir / case / "status_test.json"
        if status.exists():
            entry["status"] = json.loads(status.read_text())
        meta = json.loads((cdir / case / "scores_test.json").read_text()).get("metadata", {})
        entry["evaluator_status"] = meta.get("methods", {}).get("facet")
        if case in legacy_cases:
            entry["metrics_v2_protocol"] = legacy_cases[case]
        summary["cases"][case] = entry

    groups = {}
    for group in ("reference", "better_2x_budget", "weaker_third_budget", "weaker_open_loop", "alt_mlp_estimator",
                  "weaker_open_loop_third_budget", "alt_mlp_estimator_third_budget", "weaker_untrained",
                  "paper_nets", "paper_nets_2x_budget", "paper_nets_open_loop", "paper_nets_untrained"):
        names = group_members(cases, group)
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
        groups[group] = entry
    summary["groups"] = groups

    rng = np.random.default_rng(0)

    def details(case: str):
        path = detail_dir(mode) / case / "details_test.json"
        return json.loads(path.read_text()).get("facet") if path.exists() else None

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
        pairs += [(f"better_2x_budget_s{s}", f"reference_s{s}"), (f"weaker_third_budget_s{s}", f"reference_s{s}"),
                  (f"weaker_open_loop_s{s}", f"reference_s{s}"), (f"alt_mlp_estimator_s{s}", f"reference_s{s}"),
                  (f"weaker_untrained_s{s}", f"reference_s{s}"),
                  (f"paper_nets_s{s}", f"reference_s{s}"),
                  (f"paper_nets_2x_budget_s{s}", f"paper_nets_s{s}"), (f"paper_nets_open_loop_s{s}", f"paper_nets_s{s}"),
                  (f"paper_nets_untrained_s{s}", f"paper_nets_s{s}")]
    for case in ("shortcut_retreat_full", "shortcut_retreat_mild", "shortcut_wall_dodger", "shortcut_reference_lateral_clip",
                 "probe_dob_impedance", "shortcut_analytic_clipped", "shortcut_passive"):
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

    levels, wall_stats = {}, {}
    for case in cases:
        d = details(case)
        if d is None:
            continue
        lv = np.asarray(d["large_lateral_impulse"]["level_n"])
        sv = np.asarray(d["large_lateral_impulse"]["survived"])
        levels[case] = {str(int(level)): float(sv[lv == level].mean()) for level in np.unique(lv)}
        wall = d["soft_wall_collision"]
        base_imp = np.asarray(wall["baseline_impulse"]).sum()
        stats = {"retreat_rms_max": float(np.nanmax(np.asarray(wall["retreat_rms"]))),
                 "valid_rate": float(np.mean(wall["valid"]))}
        if "measured_impulse" in wall:
            measured = np.asarray(wall["measured_impulse"])
            commanded = np.asarray(wall["commanded_impulse"])
            stats["measured_reduction"] = float(1 - measured.sum() / base_imp)
            stats["commanded_reduction"] = float(1 - commanded.sum() / base_imp)
            stats["measured_over_commanded_mean"] = float(np.mean(measured / np.maximum(commanded, 1e-9)))
        wall_stats[case] = stats
    summary["impulse_survival_by_peak_force_n"] = levels
    summary["wall_measured_vs_commanded"] = wall_stats

    agreement = {}
    for case in cases:
        train_path = cdir / case / "scores_train.json"
        if train_path.exists():
            tr = flat(json.loads(train_path.read_text()))
            agreement[case] = {key: [tr.get(key), cases[case][key]] for key in cases[case]}
    summary["train_vs_test"] = agreement
    decision_path = HD / f"decision_{TAG}.json"
    if decision_path.exists():
        summary["configuration_decision"] = json.loads(decision_path.read_text())
    write_json(HD / (f"summary_{TAG}.json" if mode == "full" else f"summary_{TAG}_smoke.json"), summary)

    lines = [f"base = {base_case}",
             "",
             "| case | rank (all) | rank (genuine) | rank (trained) | " + " | ".join(f"{e.split('_')[0]}:{m}" for e, m, _, _ in specs) + " |",
             "|---|---:|---:|---:|" + "---:|" * len(specs)]
    for case, item in sorted(summary["cases"].items(), key=lambda kv: -kv[1]["rank_all"]):
        vals = " | ".join(f"{item['metrics'].get(f'{e}/{m}', float('nan')):.4f}" for e, m, _, _ in specs)
        lines.append(f"| {case} | {item['rank_all']:+.3f} | {item['rank_genuine']:+.3f} | {item['rank_trained']:+.3f} | {vals} |")
    (HD / (f"summary_{TAG}.md" if mode == "full" else f"summary_{TAG}_smoke.md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({"canonical": canonical, "groups": {g: {k: v for k, v in e.items() if k.startswith("rank_")} for g, e in groups.items()},
                      "wall": wall_stats.get(base_case)}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("checks")
    for name in ("score-existing", "train-paper", "finalize", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--mode", choices=list(BUDGETS), default="full")
    args = parser.parse_args()
    {"checks": cmd_checks, "score-existing": cmd_score_existing, "train-paper": cmd_train_paper,
     "finalize": cmd_finalize, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
