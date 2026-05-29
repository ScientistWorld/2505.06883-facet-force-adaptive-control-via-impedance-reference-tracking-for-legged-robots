#!/usr/bin/env python3
"""Train and evaluate the reduced FACET PPO policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from method.ppo_facet import PPOConfig, policy_rollout, train_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=96)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/facet_planar"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--teacher-updates", type=int, default=60)
    parser.add_argument("--student-updates", type=int, default=30)
    parser.add_argument("--n-envs", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=96)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or (args.output_dir / "facet_student.pt")
    if not checkpoint.exists():
        config = PPOConfig(
            n_envs=args.n_envs,
            rollout_steps=args.rollout_steps,
            teacher_updates=args.teacher_updates,
            student_updates=args.student_updates,
            seed=args.seed,
        )
        checkpoint = train_policy(args.output_dir, config)
    runs = [
        policy_rollout("impulse", args.episodes, args.seed, kp=4.0, name="facet_smoothed", checkpoint=checkpoint),
        policy_rollout("wall", args.episodes, args.seed + 1, kp=4.0, name="facet_low_kp", checkpoint=checkpoint),
    ]
    for run in runs:
        path = args.output_dir / f"{run['method']}_{run['mode']}.json"
        path.write_text(json.dumps(run))


if __name__ == "__main__":
    main()
