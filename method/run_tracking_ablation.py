#!/usr/bin/env python3
"""Generate reduced FACET tracking-target ablation artifacts.

The paper's Table 2 compares policies trained with different reference targets:
smoothed integrated targets, open-loop integrated targets, closed-loop
integrated targets, and raw acceleration targets. This reduced benchmark keeps
the same target definitions in the planar force-interaction setting and writes
per-episode tracking errors for method-independent evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.facet_planar_env import PlanarForceEnv


def desired_position(pos: np.ndarray, kp: float, kd: float, target_velocity: np.ndarray) -> np.ndarray:
    return pos + (kd / max(kp, 1e-6)) * target_velocity[None, :]


def integrate_reference(
    start_pos: np.ndarray,
    start_vel: np.ndarray,
    current_pos: np.ndarray,
    force_sequence: list[np.ndarray],
    kp: float,
    kd: float,
    mass: float,
    target_velocity: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref_pos = start_pos.copy()
    ref_vel = start_vel.copy()
    ref_acc = np.zeros_like(ref_pos)
    x_des = desired_position(current_pos, kp, kd, target_velocity)
    for force in force_sequence:
        spring = kp * (x_des - ref_pos) - kd * ref_vel
        ref_acc = (spring + force) / mass
        ref_vel = ref_vel + ref_acc * dt
        ref_pos = ref_pos + ref_vel * dt
    return ref_pos, ref_vel, ref_acc


def lag_target(
    pos_history: list[np.ndarray],
    vel_history: list[np.ndarray],
    force_history: list[np.ndarray],
    current_pos: np.ndarray,
    lag: int | str,
    kp: float,
    kd: float,
    mass: float,
    target_velocity: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if lag == "open":
        start_idx = 0
    else:
        start_idx = max(0, len(pos_history) - 1 - int(lag))
    forces = force_history[start_idx:]
    if not forces:
        forces = [np.zeros_like(current_pos)]
    return integrate_reference(
        pos_history[start_idx],
        vel_history[start_idx],
        current_pos,
        forces,
        kp,
        kd,
        mass,
        target_velocity,
        dt,
    )


def mixed_target(
    pos_history: list[np.ndarray],
    vel_history: list[np.ndarray],
    force_history: list[np.ndarray],
    current_pos: np.ndarray,
    lags: tuple[int | str, ...],
    kp: float,
    kd: float,
    mass: float,
    target_velocity: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = [
        lag_target(pos_history, vel_history, force_history, current_pos, lag, kp, kd, mass, target_velocity, dt)
        for lag in lags
    ]
    pos_targets = np.stack([target[0] for target in targets])
    vel_targets = np.stack([target[1] for target in targets])
    acc_targets = np.stack([target[2] for target in targets])
    return pos_targets.mean(axis=0), vel_targets.mean(axis=0), acc_targets.mean(axis=0)


def rollout_variant(name: str, episodes: int, seed: int, kp: float) -> dict:
    env = PlanarForceEnv(episodes, mode="impulse", seed=seed)
    env.reset()
    kd = 2.0 * np.sqrt(kp)
    target_velocity = np.array([env.cfg.target_vx, 0.0], dtype=np.float32)
    pos_history = [env.pos.copy()]
    vel_history = [env.vel.copy()]
    force_history: list[np.ndarray] = []
    pos_error = np.zeros(episodes, dtype=np.float64)
    vel_error = np.zeros(episodes, dtype=np.float64)
    acc_error = np.zeros(episodes, dtype=np.float64)
    counts = np.zeros(episodes, dtype=np.float64)

    if name == "smoothed_teacher":
        control_lags: tuple[int | str, ...] = (8, 16, 32)
        mode = "integrated"
    elif name == "open_loop_teacher":
        control_lags = ("open",)
        mode = "integrated"
    elif name == "closed_loop_teacher":
        control_lags = (1,)
        mode = "integrated"
    elif name == "acceleration_teacher":
        control_lags = (1,)
        mode = "acceleration"
    else:
        raise ValueError(f"unknown ablation variant: {name}")

    for _ in range(env.cfg.episode_steps):
        force_ext, _ = env.external_forces()
        ref_pos, ref_vel, ref_acc = mixed_target(
            pos_history,
            vel_history,
            force_history + [force_ext],
            env.pos,
            control_lags,
            kp,
            kd,
            env.cfg.mass,
            target_velocity,
            env.cfg.dt,
        )
        spring_acc = (kp * (desired_position(env.pos, kp, kd, target_velocity) - ref_pos) - kd * ref_vel) / env.cfg.mass
        if mode == "acceleration":
            action = ref_acc - force_ext / env.cfg.mass + env.cfg.damping * env.vel
        else:
            action = spring_acc + 1.0 * (ref_pos - env.pos) + 0.5 * (ref_vel - env.vel)

        prev_vel = env.vel.copy()
        _, _, _, info = env.step(action.astype(np.float32))
        actual_acc = (env.vel - prev_vel) / env.cfg.dt
        force_history.append(info["force_ext"])
        pos_history.append(env.pos.copy())
        vel_history.append(env.vel.copy())

        metric_pos, metric_vel, _ = mixed_target(
            pos_history,
            vel_history,
            force_history,
            env.pos,
            (8, 16, 32, "open"),
            kp,
            kd,
            env.cfg.mass,
            target_velocity,
            env.cfg.dt,
        )
        _, _, metric_acc = lag_target(
            pos_history,
            vel_history,
            force_history,
            env.pos,
            1,
            kp,
            kd,
            env.cfg.mass,
            target_velocity,
            env.cfg.dt,
        )
        active = ~env.failed
        pos_error[active] += np.sum((env.pos[active] - metric_pos[active]) ** 2, axis=1)
        vel_error[active] += np.sum((env.vel[active] - metric_vel[active]) ** 2, axis=1)
        acc_error[active] += np.sum((actual_acc[active] - metric_acc[active]) ** 2, axis=1)
        counts[active] += 1.0

    denom = np.maximum(counts, 1.0)
    return {
        "method": name,
        "success": (1.0 - env.failed.astype(np.float32)).tolist(),
        "pos_error": (pos_error / denom).tolist(),
        "vel_error": (vel_error / denom).tolist(),
        "acc_error": (acc_error / denom).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=96)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/facet_planar"))
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--kp", type=float, default=4.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = [
        "smoothed_teacher",
        "open_loop_teacher",
        "closed_loop_teacher",
        "acceleration_teacher",
    ]
    artifact = {
        "experiment": "tracking_target_ablation",
        "mode": "tracking_ablation",
        "episodes": args.episodes,
        "kp": args.kp,
        "variants": {
            name: rollout_variant(name, args.episodes, args.seed + idx, args.kp)
            for idx, name in enumerate(variants)
        },
    }
    (args.output_dir / "tracking_target_ablation.json").write_text(json.dumps(artifact))


if __name__ == "__main__":
    main()
