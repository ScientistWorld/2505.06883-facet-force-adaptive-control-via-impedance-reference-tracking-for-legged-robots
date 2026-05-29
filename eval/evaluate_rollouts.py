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
        "facet_high_kp": load(args.input_dir / "facet_high_kp_wall.json"),
        "velocity_robust": load(args.input_dir / "velocity_robust_wall.json"),
    }

    scores = {
        "experiments": {
            "large_lateral_impulse": {
                "results": {
                    name: {
                        "success_rate": run["success_rate"],
                        "mean_abs_lateral_displacement_m": run["mean_abs_lateral_displacement"],
                        "mean_final_forward_speed_mps": run["mean_final_forward_speed"],
                    }
                    for name, run in impulse.items()
                }
            },
            "soft_wall_collision": {
                "results": {
                    name: {
                        "mean_collision_impulse_Ns": run["mean_wall_impulse"],
                    }
                    for name, run in wall.items()
                }
            },
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2) + "\n")


if __name__ == "__main__":
    main()
