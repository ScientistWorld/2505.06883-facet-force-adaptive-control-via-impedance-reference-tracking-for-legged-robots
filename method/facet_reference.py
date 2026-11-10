"""FACET impedance reference model and temporally smoothed tracking targets.

Implements paper Sec. 4.1-4.2: the virtual mass-spring-damper

    m x_ref'' = Kp (x_des - x_ref) + Kd (xdot_des - xdot_ref) + f_ext       (Eq. 2)

is integrated from the robot's own states ``L`` control steps in the past
(Eq. 6). With the fixed-target-velocity command x_des = x_sim + (Kd/Kp) v_target
and xdot_des = 0 the spring term becomes Kp (x_sim - x_ref) + Kd (v_target - v_ref).
Rewards (Eq. 7) average per-target exponentials over the starts
t' in {t - 8dt, t - 16dt, t - 32dt}. ``mode="open_loop"`` keeps a single target
integrated from the episode start (the paper's open-loop ablation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ImpedanceCommand:
    kp: np.ndarray                # [n]
    kd: np.ndarray                # [n]
    virtual_mass: np.ndarray      # [n]
    target_velocity: np.ndarray   # [n, 2]


def reference_substep(
    x_ref: np.ndarray,
    v_ref: np.ndarray,
    x_sim: np.ndarray,
    force: np.ndarray,
    command: ImpedanceCommand,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One semi-implicit Euler step of Eq. (2) in fixed-target-velocity mode."""
    spring = command.kp[:, None] * (x_sim - x_ref) + command.kd[:, None] * (command.target_velocity - v_ref)
    acc = (spring + force) / command.virtual_mass[:, None]
    v_new = v_ref + acc * dt
    x_new = x_ref + v_new * dt
    return x_new, v_new


class ReferenceTargets:
    """Per-environment history of robot states and external forces.

    ``pos[h]``/``vel[h]`` hold states with ``h = H - 1`` the latest; ``force[h]``
    is the force applied while moving from state ``h`` to ``h + 1``.
    """

    def __init__(self, n_envs: int, dt: float, lags: tuple[int, ...] = (8, 16, 32), mode: str = "smoothed"):
        if mode not in ("smoothed", "open_loop"):
            raise ValueError(f"unknown target mode {mode}")
        self.n = n_envs
        self.dt = dt
        self.lags = tuple(lags)
        self.mode = mode
        self.H = max(self.lags) + 1
        self.pos = np.zeros((self.H, n_envs, 2))
        self.vel = np.zeros((self.H, n_envs, 2))
        self.force = np.zeros((self.H, n_envs, 2))
        self.age = np.zeros(n_envs, dtype=np.int64)
        self.ol_pos = np.zeros((n_envs, 2))
        self.ol_vel = np.zeros((n_envs, 2))

    @property
    def n_targets(self) -> int:
        return 1 if self.mode == "open_loop" else len(self.lags)

    def reset(self, idx: np.ndarray, pos: np.ndarray, vel: np.ndarray) -> None:
        self.pos[:, idx] = pos[idx]
        self.vel[:, idx] = vel[idx]
        self.force[:, idx] = 0.0
        self.age[idx] = 0
        self.ol_pos[idx] = pos[idx]
        self.ol_vel[idx] = vel[idx]

    def push(self, prev_pos: np.ndarray, force: np.ndarray, pos: np.ndarray, vel: np.ndarray, command: ImpedanceCommand) -> None:
        """Record one simulator transition (force applied from the previous state)."""
        if self.mode == "open_loop":
            self.ol_pos, self.ol_vel = reference_substep(self.ol_pos, self.ol_vel, prev_pos, force, command, self.dt)
        self.pos = np.roll(self.pos, -1, axis=0)
        self.vel = np.roll(self.vel, -1, axis=0)
        self.force = np.roll(self.force, -1, axis=0)
        self.pos[-1] = pos
        self.vel[-1] = vel
        self.force[-2] = force
        self.force[-1] = 0.0
        self.age += 1

    def _integrate(
        self,
        pos_w: np.ndarray,
        vel_w: np.ndarray,
        force_w: np.ndarray,
        effective: np.ndarray,
        command: ImpedanceCommand,
    ) -> tuple[np.ndarray, np.ndarray]:
        steps = force_w.shape[0]
        begin = steps - effective
        x = pos_w[0].copy()
        v = vel_w[0].copy()
        for j in range(steps):
            started = (j >= begin)[:, None]
            x_new, v_new = reference_substep(x, v, pos_w[j], force_w[j], command, self.dt)
            nxt = min(j + 1, pos_w.shape[0] - 1)
            x = np.where(started, x_new, pos_w[nxt])
            v = np.where(started, v_new, vel_w[nxt])
        return x, v

    def current_targets(self, command: ImpedanceCommand) -> tuple[np.ndarray, np.ndarray]:
        """Targets for the latest state: arrays [n_targets, n, 2]."""
        if self.mode == "open_loop":
            return self.ol_pos[None].copy(), self.ol_vel[None].copy()
        H = self.H
        xs, vs = [], []
        for lag in self.lags:
            effective = np.minimum(lag, self.age)
            x, v = self._integrate(
                self.pos[H - 1 - lag :], self.vel[H - 1 - lag :], self.force[H - 1 - lag : H - 1], effective, command
            )
            xs.append(x)
            vs.append(v)
        return np.stack(xs), np.stack(vs)

    def predicted_targets(self, force_now: np.ndarray, command: ImpedanceCommand) -> tuple[np.ndarray, np.ndarray]:
        """Targets for the next state given the force acting now: [n_targets, n, 2]."""
        if self.mode == "open_loop":
            x, v = reference_substep(self.ol_pos, self.ol_vel, self.pos[-1], force_now, command, self.dt)
            return x[None], v[None]
        H = self.H
        xs, vs = [], []
        for lag in self.lags:
            effective = np.minimum(lag, self.age + 1)
            pos_w = np.concatenate([self.pos[H - lag :], self.pos[H - 1 :]], axis=0)
            vel_w = np.concatenate([self.vel[H - lag :], self.vel[H - 1 :]], axis=0)
            force_w = np.concatenate([self.force[H - lag : H - 1], force_now[None]], axis=0)
            x, v = self._integrate(pos_w, vel_w, force_w, effective, command)
            xs.append(x)
            vs.append(v)
        return np.stack(xs), np.stack(vs)
