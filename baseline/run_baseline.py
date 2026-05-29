#!/usr/bin/env python3
"""Velocity-tracking baselines from the FACET paper, at benchmark scale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.facet_planar_env import PlanarForceEnv


class VelocityController:
    def __init__(self, target_vx: float, lateral_gain: float, forward_gain: float):
        self.target_vx = target_vx
        self.lateral_gain = lateral_gain
        self.forward_gain = forward_gain

    def act(self, vel: np.ndarray) -> np.ndarray:
        action = np.zeros_like(vel)
        action[:, 0] = self.forward_gain * (self.target_vx - vel[:, 0])
        action[:, 1] = -self.lateral_gain * vel[:, 1]
        return action.astype(np.float32)


def rollout(mode: str, episodes: int, seed: int, name: str, lateral_gain: float) -> dict:
    env = PlanarForceEnv(episodes, mode=mode, seed=seed)
    controller = VelocityController(env.cfg.target_vx, lateral_gain=lateral_gain, forward_gain=5.5)
    env.reset()
    wall_impulse = np.zeros(episodes, dtype=np.float32)
    records = []
    for _ in range(env.cfg.episode_steps):
        action = controller.act(env.vel)
        obs, reward, done, info = env.step(action)
        wall_impulse += np.abs(info["wall_force"][:, 0]) * env.cfg.dt
        records.append(
            {
                "t": info["time"].tolist(),
                "pos": env.pos.tolist(),
                "vel": env.vel.tolist(),
                "action": action.tolist(),
                "force_ext": info["force_ext"].tolist(),
                "wall_force": info["wall_force"].tolist(),
                "failed": info["failed"].astype(int).tolist(),
            }
        )
        if done.all():
            break
    return {
        "method": name,
        "mode": mode,
        "episodes": episodes,
        "success_rate": float(1.0 - env.failed.mean()),
        "mean_wall_impulse": float(wall_impulse.mean()),
        "mean_abs_lateral_displacement": float(np.abs(env.pos[:, 1]).mean()),
        "mean_final_forward_speed": float(env.vel[:, 0].mean()),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=96)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/facet_planar"))
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        rollout("impulse", args.episodes, args.seed, "velocity_vanilla", lateral_gain=8.0),
        rollout("impulse", args.episodes, args.seed + 1, "velocity_robust", lateral_gain=4.0),
        rollout("wall", args.episodes, args.seed + 2, "velocity_robust", lateral_gain=4.0),
    ]
    for run in runs:
        path = args.output_dir / f"{run['method']}_{run['mode']}.json"
        path.write_text(json.dumps(run))


if __name__ == "__main__":
    main()

