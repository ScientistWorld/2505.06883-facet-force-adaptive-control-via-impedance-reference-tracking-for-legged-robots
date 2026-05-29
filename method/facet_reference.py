"""FACET impedance reference model and smoothed target controller."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class ImpedanceCommand:
    kp: float
    kd: float
    virtual_mass: float
    target_velocity: np.ndarray


class SmoothedImpedanceReference:
    """Integrates Eq. (2) from multiple starts, matching FACET Sec. 4.2."""

    def __init__(self, n_envs: int, dt: float, lags: tuple[int, ...] = (8, 16, 32)):
        self.n_envs = n_envs
        self.dt = dt
        self.lags = lags
        self.max_lag = max(lags)
        self.history: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=self.max_lag + 1)

    def reset(self, pos: np.ndarray, vel: np.ndarray) -> None:
        self.history.clear()
        for _ in range(self.max_lag + 1):
            self.history.append((pos.copy(), vel.copy()))

    def push_state(self, pos: np.ndarray, vel: np.ndarray) -> None:
        self.history.append((pos.copy(), vel.copy()))

    def desired_position(self, pos: np.ndarray, command: ImpedanceCommand) -> np.ndarray:
        # Paper's fixed target velocity mode: x_des = Kd/Kp * v_target + x_sim.
        return pos + (command.kd / max(command.kp, 1e-6)) * command.target_velocity[None, :]

    def integrate_from(
        self,
        start_pos: np.ndarray,
        start_vel: np.ndarray,
        current_pos: np.ndarray,
        force_ext: np.ndarray,
        command: ImpedanceCommand,
        steps: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ref_pos = start_pos.copy()
        ref_vel = start_vel.copy()
        x_des = self.desired_position(current_pos, command)
        acc = np.zeros_like(ref_pos)
        for _ in range(max(1, steps)):
            spring = command.kp * (x_des - ref_pos) - command.kd * ref_vel
            acc = (spring + force_ext) / command.virtual_mass
            ref_vel = ref_vel + acc * self.dt
            ref_pos = ref_pos + ref_vel * self.dt
        return ref_pos, ref_vel, acc

    def targets(
        self, current_pos: np.ndarray, force_ext: np.ndarray, command: ImpedanceCommand
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos_targets = []
        vel_targets = []
        acc_targets = []
        hist = list(self.history)
        for lag in self.lags:
            idx = max(0, len(hist) - 1 - lag)
            start_pos, start_vel = hist[idx]
            ref_pos, ref_vel, ref_acc = self.integrate_from(
                start_pos, start_vel, current_pos, force_ext, command, lag
            )
            pos_targets.append(ref_pos)
            vel_targets.append(ref_vel)
            acc_targets.append(ref_acc)
        return np.stack(pos_targets), np.stack(vel_targets), np.stack(acc_targets)


class FACETController:
    """Teacher-style policy that tracks smoothed impedance reference targets."""

    def __init__(self, n_envs: int, dt: float, kp: float = 16.0, virtual_mass: float = 12.0):
        kd = 2.0 * np.sqrt(kp)
        self.command = ImpedanceCommand(
            kp=kp,
            kd=kd,
            virtual_mass=virtual_mass,
            target_velocity=np.array([1.5, 0.0], dtype=np.float32),
        )
        self.ref = SmoothedImpedanceReference(n_envs=n_envs, dt=dt)

    def reset(self, pos: np.ndarray, vel: np.ndarray) -> None:
        self.ref.reset(pos, vel)

    def act(self, pos: np.ndarray, vel: np.ndarray, force_ext: np.ndarray) -> tuple[np.ndarray, dict]:
        pos_targets, vel_targets, acc_targets = self.ref.targets(pos, force_ext, self.command)
        ref_pos = pos_targets.mean(axis=0)
        ref_vel = vel_targets.mean(axis=0)
        # PD tracking of the smoothed reference plus the reference acceleration.
        action = acc_targets.mean(axis=0) + 5.0 * (ref_pos - pos) + 2.0 * (ref_vel - vel)
        self.ref.push_state(pos, vel)
        return action.astype(np.float32), {
            "ref_pos": ref_pos,
            "ref_vel": ref_vel,
            "ref_acc": acc_targets.mean(axis=0),
        }

