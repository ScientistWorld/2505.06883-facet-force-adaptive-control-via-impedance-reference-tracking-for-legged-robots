#!/usr/bin/env python3
"""Score the visible train slice of saved force-interaction rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SLICE_NAME = "train"


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    if not values:
        return 0.02
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def artifact_dirs(root: Path) -> list[Path]:
    if list(root.glob("*.json")):
        return [root]
    dirs = [path for path in root.iterdir() if path.is_dir() and list(path.glob("*.json"))]
    if not dirs:
        raise FileNotFoundError(f"no rollout JSON artifacts found under {root}")
    return sorted(dirs)


def load_rollouts(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for directory in artifact_dirs(root):
        for path in sorted(directory.glob("*.json")):
            payload = json.loads(path.read_text())
            if "records" not in payload or "method" not in payload or "mode" not in payload:
                continue
            grouped.setdefault(str(payload["mode"]), {})[str(payload["method"])] = payload
    return grouped


def method_index(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    experiments = payload.get("experiments", payload)
    index: dict[str, set[str]] = {}
    for exp_name, exp in experiments.items():
        if not isinstance(exp, dict):
            continue
        results = exp.get("results", exp)
        if isinstance(results, dict):
            index[exp_name] = {name for name in results if not name.startswith("_")}
    return index


def slice_indices(episode_count: int) -> list[int]:
    return [idx for idx in range(episode_count) if idx % 2 == 0]


def take(values: list[Any], indices: list[int]) -> list[Any]:
    return [values[idx] for idx in indices if idx < len(values)]


def summarize(run: dict[str, Any]) -> dict[str, float]:
    records = run.get("records")
    if not records:
        raise ValueError(f"{run.get('method', '<unknown>')} has no rollout records")

    episode_count = int(run.get("episodes") or len(records[-1]["failed"]))
    indices = slice_indices(episode_count)
    if not indices:
        raise ValueError(f"{run.get('method', '<unknown>')} has no {SLICE_NAME} episodes")

    final = records[-1]
    failed = [float(value) for value in take(final["failed"], indices)]
    final_vel = [[float(component) for component in row] for row in take(final["vel"], indices)]
    times = [mean([float(value) for value in take(step["t"], indices)]) for step in records]
    dt = median([times[i] - times[i - 1] for i in range(1, len(times))]) if len(times) > 1 else 0.02
    wall_impulse = [0.0 for _ in indices]
    post_contact_wall_impulse = [0.0 for _ in indices]
    contact = [False for _ in indices]
    contact_start = [None for _ in indices]
    contact_sample_count = [0 for _ in indices]
    entry_speed = [0.0 for _ in indices]

    for step in records:
        for local_idx, episode_idx in enumerate(indices):
            force = step["wall_force"][episode_idx]
            force_x = abs(float(force[0]))
            time = float(step["t"][episode_idx])
            wall_impulse[local_idx] += force_x * max(dt, 1e-9)
            if force_x > 1.0e-5:
                if not contact[local_idx]:
                    contact_start[local_idx] = time
                    entry_speed[local_idx] = float(step["vel"][episode_idx][0])
                contact[local_idx] = True
            if contact_start[local_idx] is not None:
                tau = time - float(contact_start[local_idx])
                if 0.0 <= tau < 0.50:
                    if contact_sample_count[local_idx] % 2 == 0:
                        post_contact_wall_impulse[local_idx] += 2.0 * force_x * max(dt, 1e-9)
                    contact_sample_count[local_idx] += 1

    return {
        "success_rate": 1.0 - mean(failed),
        "mean_final_forward_speed_mps": mean([row[0] for row in final_vel]),
        "impulse_recovery_score": (1.0 - mean(failed)) * max(mean([row[0] for row in final_vel]), 0.0) / 1.5,
        "mean_wall_impulse": mean(wall_impulse),
        "mean_post_contact_wall_impulse": mean(post_contact_wall_impulse),
        "wall_contact_rate": mean([1.0 if value else 0.0 for value in contact]),
        "mean_wall_entry_speed_mps": mean(entry_speed),
        "episode_count": float(len(indices)),
    }


def build_scores(grouped: dict[str, dict[str, dict[str, Any]]], expected: dict[str, set[str]]) -> dict[str, Any]:
    scores: dict[str, Any] = {
        "metadata": {
            "slice": SLICE_NAME,
            "split_rule": "even rollout episode indices; post-contact wall impulse uses even samples in the first 0.50s",
        },
        "experiments": {},
    }

    impulse_runs = grouped.get("impulse", {})
    if expected.get("large_lateral_impulse"):
        impulse_runs = {
            name: run for name, run in impulse_runs.items() if name in expected["large_lateral_impulse"]
        }
    if impulse_runs:
        impulse_summary = {name: summarize(run) for name, run in impulse_runs.items()}
        scores["experiments"]["large_lateral_impulse"] = {
            "results": {
                name: {
                    "impulse_recovery_score": metrics["impulse_recovery_score"],
                    "success_rate": metrics["success_rate"],
                    "mean_final_forward_speed_mps": metrics["mean_final_forward_speed_mps"],
                }
                for name, metrics in sorted(impulse_summary.items())
            }
        }

    wall_runs = grouped.get("wall", {})
    if expected.get("soft_wall_collision"):
        wall_runs = {name: run for name, run in wall_runs.items() if name in expected["soft_wall_collision"]}
    if wall_runs:
        wall_summary = {name: summarize(run) for name, run in wall_runs.items()}
        robust_impulse = wall_summary.get("velocity_robust", {}).get("mean_wall_impulse")
        robust_post_contact_impulse = wall_summary.get("velocity_robust", {}).get(
            "mean_post_contact_wall_impulse"
        )
        if robust_impulse is None:
            raise ValueError("soft_wall_collision requires a velocity_robust wall baseline artifact")
        if robust_post_contact_impulse is None:
            raise ValueError("soft_wall_collision requires post-contact wall impulse for velocity_robust")
        scores["experiments"]["soft_wall_collision"] = {
            "results": {
                name: {
                    "post_contact_impulse_reduction": 1.0
                    - metrics["mean_post_contact_wall_impulse"] / max(robust_post_contact_impulse, 1e-9),
                    "collision_impulse_reduction": 1.0
                    - metrics["mean_wall_impulse"] / max(robust_impulse, 1e-9),
                    "wall_contact_rate": metrics["wall_contact_rate"],
                    "mean_wall_entry_speed_mps": metrics["mean_wall_entry_speed_mps"],
                }
                for name, metrics in sorted(wall_summary.items())
            }
        }

    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", nargs="?", type=Path, default=Path("/home/user/checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("/home/user/scoring/scores_train.json"))
    parser.add_argument("--method-index", type=Path, default=Path("/home/user/scoring/scores.json"))
    args = parser.parse_args()

    expected = method_index(args.method_index)
    scores = build_scores(load_rollouts(args.input_dir), expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
