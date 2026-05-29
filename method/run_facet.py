#!/usr/bin/env python3
"""Run the FACET impedance-reference method in the benchmark simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.facet_planar_env import PlanarForceEnv
from method.facet_reference import FACETController


def rollout(mode: str, episodes: int, seed: int, kp: float, name: str) -> dict:
    env = PlanarForceEnv(episodes, mode=mode, seed=seed)
    controller = FACETController(episodes, env.cfg.dt, kp=kp, virtual_mass=env.cfg.mass)
    obs = env.reset()
    controller.reset(env.pos, env.vel)
    records = []
    wall_impulse = np.zeros(episodes, dtype=np.float32)

    for _ in range(env.cfg.episode_steps):
        force_ext, _ = env.external_forces()
        action, ref_info = controller.act(env.pos, env.vel, force_ext)
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
                "ref_pos": ref_info["ref_pos"].tolist(),
                "ref_vel": ref_info["ref_vel"].tolist(),
            }
        )
        if done.all():
            break

    success = 1.0 - env.failed.mean()
    return {
        "method": name,
        "mode": mode,
        "episodes": episodes,
        "kp": kp,
        "success_rate": float(success),
        "mean_wall_impulse": float(wall_impulse.mean()),
        "mean_abs_lateral_displacement": float(np.abs(env.pos[:, 1]).mean()),
        "mean_final_forward_speed": float(env.vel[:, 0].mean()),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=96)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/facet_planar"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        rollout("impulse", args.episodes, args.seed, kp=16.0, name="facet_smoothed"),
        rollout("wall", args.episodes, args.seed + 1, kp=4.0, name="facet_low_kp"),
        rollout("wall", args.episodes, args.seed + 2, kp=36.0, name="facet_high_kp"),
    ]
    for run in runs:
        path = args.output_dir / f"{run['method']}_{run['mode']}.json"
        path.write_text(json.dumps(run))


if __name__ == "__main__":
    main()

