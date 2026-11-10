#!/usr/bin/env python3
"""Train the reduced FACET teacher/student policy and export ``policy.pt``.

The exported TorchScript student is what the evaluators score; they roll it
out themselves in their own simulator copy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from method.ppo_facet import PPOConfig, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/facet"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--teacher-updates", type=int, default=PPOConfig.teacher_updates)
    parser.add_argument("--student-updates", type=int, default=PPOConfig.student_updates)
    parser.add_argument("--n-envs", type=int, default=PPOConfig.n_envs)
    parser.add_argument("--rollout-steps", type=int, default=PPOConfig.rollout_steps)
    parser.add_argument("--estimator", choices=["gru", "mlp"], default=PPOConfig.estimator)
    parser.add_argument("--target-mode", choices=["smoothed", "open_loop"], default=PPOConfig.target_mode)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="retrain even if policy.pt exists")
    args = parser.parse_args()

    policy = args.output_dir / "policy.pt"
    if policy.exists() and not args.force:
        print(f"[facet] {policy} exists; skipping training (use --force to retrain)", flush=True)
        return
    config = PPOConfig(
        n_envs=args.n_envs,
        rollout_steps=args.rollout_steps,
        teacher_updates=args.teacher_updates,
        student_updates=args.student_updates,
        seed=args.seed,
        estimator=args.estimator,
        target_mode=args.target_mode,
        threads=args.threads,
    )
    summary = train(args.output_dir, config)
    print(f"[facet] exported {policy} after {summary['train_seconds']} s", flush=True)


if __name__ == "__main__":
    main()
