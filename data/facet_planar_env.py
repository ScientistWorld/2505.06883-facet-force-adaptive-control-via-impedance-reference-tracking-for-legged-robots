"""Lightweight force-interaction environment used by the FACET benchmark.

The paper evaluates legged robots in Isaac Lab. This module keeps the same
force-interaction problem structure at smaller scale: a planar body must move
forward while lateral pushes and wall contacts perturb it. Controllers choose
ground-reaction accelerations; excessive lateral reaction represents the robot
trying to resist a push so hard that non-foot contact/failure occurs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EnvConfig:
    dt: float = 0.02
    mass: float = 12.0
    action_limit: float = 8.0
    episode_steps: int = 350
    target_vx: float = 1.5
    push_start: float = 2.0
    wall_x: float = 3.0
    wall_kp: float = 260.0
    wall_kd: float = 28.0
    fall_reaction_threshold: float = 5.2
    fall_tilt_threshold: float = 1.7
    damping: float = 0.05


def ramp_force(t: float, start: float, duration: float, peak: float, axis: int) -> np.ndarray:
    """Triangular impulse force in x/y; impulse is 0.5 * duration * peak."""
    f = np.zeros(2, dtype=np.float32)
    tau = t - start
    if 0.0 <= tau <= duration:
        half = 0.5 * duration
        scale = tau / half if tau <= half else (duration - tau) / half
        f[axis] = peak * max(0.0, scale)
    return f


class PlanarForceEnv:
    """Vectorized point-body dynamics with a legged-stability proxy."""

    def __init__(self, n_envs: int, mode: str, seed: int = 0, config: EnvConfig | None = None):
        self.n_envs = n_envs
        self.mode = mode
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = np.zeros(self.n_envs, dtype=np.float32)
        self.step_count = np.zeros(self.n_envs, dtype=np.int32)
        self.pos = np.zeros((self.n_envs, 2), dtype=np.float32)
        self.vel = np.zeros((self.n_envs, 2), dtype=np.float32)
        self.tilt = np.zeros(self.n_envs, dtype=np.float32)
        self.failed = np.zeros(self.n_envs, dtype=bool)
        self.done = np.zeros(self.n_envs, dtype=bool)
        self.push_peak = self.rng.uniform(320.0, 430.0, size=self.n_envs).astype(np.float32)
        self.push_duration = self.rng.uniform(0.45, 0.55, size=self.n_envs).astype(np.float32)
        return self.obs()

    def obs(self) -> np.ndarray:
        return np.concatenate(
            [
                self.pos,
                self.vel,
                self.tilt[:, None],
                self.t[:, None],
                self.push_peak[:, None] / 400.0,
            ],
            axis=1,
        ).astype(np.float32)

    def external_forces(self) -> tuple[np.ndarray, np.ndarray]:
        forces = np.zeros((self.n_envs, 2), dtype=np.float32)
        wall_forces = np.zeros((self.n_envs, 2), dtype=np.float32)
        for i in range(self.n_envs):
            if self.mode == "impulse":
                forces[i] += ramp_force(
                    float(self.t[i]),
                    self.cfg.push_start,
                    float(self.push_duration[i]),
                    float(self.push_peak[i]),
                    axis=1,
                )
            elif self.mode == "wall":
                penetration = self.pos[i, 0] - self.cfg.wall_x
                if penetration > 0:
                    wall_forces[i, 0] = -self.cfg.wall_kp * penetration - self.cfg.wall_kd * max(
                        self.vel[i, 0], 0.0
                    )
        return forces + wall_forces, wall_forces

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        action = np.clip(action.astype(np.float32), -self.cfg.action_limit, self.cfg.action_limit)
        force_ext, wall_force = self.external_forces()
        acc = action + force_ext / self.cfg.mass - self.cfg.damping * self.vel
        self.vel += acc * self.cfg.dt
        self.pos += self.vel * self.cfg.dt

        lateral_reaction = np.abs(action[:, 1])
        self.tilt = 0.93 * self.tilt + 0.02 * lateral_reaction + 0.002 * np.abs(force_ext[:, 1])
        new_fail = (lateral_reaction > self.cfg.fall_reaction_threshold) & (
            np.abs(force_ext[:, 1]) > 80.0
        )
        new_fail |= self.tilt > self.cfg.fall_tilt_threshold
        self.failed |= new_fail

        self.step_count += 1
        self.t += self.cfg.dt
        timeout = self.step_count >= self.cfg.episode_steps
        self.done = timeout | self.failed

        reward = -0.1 * np.sum(action * action, axis=1) - 0.1 * self.failed.astype(np.float32)
        info = {
            "force_ext": force_ext.copy(),
            "wall_force": wall_force.copy(),
            "failed": self.failed.copy(),
            "time": self.t.copy(),
        }
        return self.obs(), reward.astype(np.float32), self.done.copy(), info

