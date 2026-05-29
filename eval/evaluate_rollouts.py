#!/usr/bin/env python3
"""Evaluate saved rollout artifacts without importing method code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing rollout artifact: {path}")
    return json.loads(path.read_text())


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def median(values: list[float]) -> float:
    if not values:
        return 0.02
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def summarize(run: dict) -> dict[str, float]:
    records = run.get("records")
    if not records:
        raise ValueError(f"{run.get('method', '<unknown>')} has no rollout records")
    final = records[-1]
    failed = [float(value) for value in final["failed"]]
    final_vel = [[float(component) for component in row] for row in final["vel"]]
    times = [mean([float(value) for value in step["t"]]) for step in records]
    dt = median([times[i] - times[i - 1] for i in range(1, len(times))]) if len(times) > 1 else 0.02
    episode_count = len(failed)
    wall_impulse = [0.0 for _ in range(episode_count)]
    contact = [False for _ in range(episode_count)]
    for step in records:
        for idx, force in enumerate(step["wall_force"]):
            force_x = abs(float(force[0]))
            wall_impulse[idx] += force_x * max(dt, 1e-9)
            contact[idx] = contact[idx] or force_x > 1.0e-5
    return {
        "success_rate": 1.0 - mean(failed),
        "mean_final_forward_speed_mps": mean([row[0] for row in final_vel]),
        "mean_wall_impulse": mean(wall_impulse),
        "wall_contact_rate": mean([1.0 if value else 0.0 for value in contact]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("checkpoints/facet_planar"))
    parser.add_argument("--output", type=Path, default=Path("scoring/scores.json"))
    args = parser.parse_args()

    impulse = {
        "facet_smoothed": load(args.input_dir / "facet_smoothed_impulse.json"),
        "velocity_robust": load(args.input_dir / "velocity_robust_impulse.json"),
        "velocity_vanilla": load(args.input_dir / "velocity_vanilla_impulse.json"),
    }
    wall = {
        "facet_low_kp": load(args.input_dir / "facet_low_kp_wall.json"),
        "velocity_robust": load(args.input_dir / "velocity_robust_wall.json"),
    }
    impulse_summary = {name: summarize(run) for name, run in impulse.items()}
    wall_summary = {name: summarize(run) for name, run in wall.items()}

    scores = {
        "experiments": {
            "large_lateral_impulse": {
                "results": {
                    name: {
                        "success_rate": metrics["success_rate"],
                        "mean_final_forward_speed_mps": metrics["mean_final_forward_speed_mps"],
                    }
                    for name, metrics in impulse_summary.items()
                }
            },
            "soft_wall_collision": {
                "results": {
                    "velocity_robust": {
                        "collision_impulse_reduction": 0.0,
                        "wall_contact_rate": wall_summary["velocity_robust"]["wall_contact_rate"],
                    },
                    "facet_low_kp": {
                        "collision_impulse_reduction": 1.0
                        - wall_summary["facet_low_kp"]["mean_wall_impulse"]
                        / max(wall_summary["velocity_robust"]["mean_wall_impulse"], 1e-9),
                        "wall_contact_rate": wall_summary["facet_low_kp"]["wall_contact_rate"],
                    },
                }
            },
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2) + "\n")


if __name__ == "__main__":
    main()
