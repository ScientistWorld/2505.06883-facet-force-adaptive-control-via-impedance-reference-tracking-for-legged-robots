"""Planar force-interaction simulator for the FACET benchmark (training side).

The paper evaluates legged robots in Isaac Lab. This benchmark keeps the same
force-interaction problem at reduced scale: a planar body with randomized mass,
drag and actuator response is commanded through an impedance-style interface
(target velocity, stiffness, damping, virtual mass) while lateral pushes,
steady forces and a soft wall act on it. The controller outputs desired planar
ground-reaction accelerations; producing a large lateral reaction while being
pushed, or accumulating too much tilt, counts as a fall (the reduced analogue
of non-foot bodies touching the ground).

Policy interface (identical in eval/train and eval/test):

    observation, float32, OBS_DIM = 16, built by ``PlanarForceSim.observation``
        [0:2]   commanded target velocity (m/s, world frame)
        [2]     commanded stiffness Kp / 40
        [3]     commanded damping Kd / 20
        [4]     commanded virtual mass / 20
        [5:7]   body planar velocity (m/s)
        [7]     body tilt indicator (a fall happens above FALL_TILT)
        [8:10]  gait oscillator sin/cos at GAIT_FREQUENCY_HZ
        [10:16] previous three commanded actions, oldest first (m/s^2)
    action, 2 floats: desired planar acceleration (m/s^2); the simulator clips
        each axis to +-ACTION_LIMIT.

Commands follow the paper's fixed-target-velocity mode: the setpoint is
x_des = x + (Kd / Kp) * v_target, with critical damping Kd = 2 sqrt(Kp).

Evaluators keep their own frozen copy of these dynamics; editing this file
changes training only.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

DT = 0.02
NOMINAL_MASS = 12.0
ACTION_LIMIT = 8.0
FALL_REACTION = 5.2          # m/s^2 of lateral reaction while being pushed
FALL_PUSH_FORCE = 80.0       # N of lateral external force that makes a reaction dangerous
FALL_TILT = 2.4
TILT_DECAY = 0.93
TILT_REACTION_GAIN = 0.03
TILT_FORCE_GAIN = 0.0002
WALL_KP = 260.0
WALL_KD = 28.0
GAIT_FREQUENCY_HZ = 1.5
OBS_DIM = 16
ACTION_DIM = 2
ACTION_HISTORY = 3

# Domain randomization applied to every training and evaluation episode.
BODY_MASS_RANGE = (10.0, 14.0)
DAMPING_RANGE = (0.03, 0.07)
ACTUATOR_GAIN_RANGE = (0.85, 1.15)
ACTUATOR_TAU_RANGE = (0.02, 0.08)


@dataclass
class Scenario:
    """Per-episode command, body randomization and external-force schedule."""

    kp: np.ndarray                # [n] virtual stiffness (N/m)
    kd: np.ndarray                # [n] virtual damping (N s/m)
    virtual_mass: np.ndarray      # [n] (kg)
    target_velocity: np.ndarray   # [n, 2] (m/s)
    init_pos: np.ndarray          # [n, 2] (m)
    init_vel: np.ndarray          # [n, 2] (m/s)
    body_mass: np.ndarray         # [n] (kg)
    damping: np.ndarray           # [n] linear drag (1/s)
    actuator_gain: np.ndarray     # [n]
    actuator_tau: np.ndarray      # [n] first-order actuator lag (s)
    const_force: np.ndarray       # [n, 2] (N)
    const_start: np.ndarray       # [n] (s)
    const_duration: np.ndarray    # [n] (s)
    impulse_peak: np.ndarray      # [n, 2] peak of a triangular ramp push (N)
    impulse_start: np.ndarray     # [n] (s)
    impulse_duration: np.ndarray  # [n] (s)
    wall_x: np.ndarray            # [n] wall plane x position (m); inf = no wall

    @property
    def n(self) -> int:
        return int(self.kp.shape[0])

    def assign(self, idx: np.ndarray, other: "Scenario") -> None:
        """Overwrite episodes ``idx`` with the episodes of ``other`` (same length as idx)."""
        for item in fields(self):
            getattr(self, item.name)[idx] = getattr(other, item.name)

    def take(self, idx: np.ndarray) -> "Scenario":
        return Scenario(**{item.name: np.array(getattr(self, item.name)[idx]) for item in fields(self)})


def blank_scenario(n: int) -> Scenario:
    """Nominal body, zero command, no external forces, no wall."""
    zeros = np.zeros(n)
    return Scenario(
        kp=np.full(n, 4.0),
        kd=np.full(n, 4.0),
        virtual_mass=np.full(n, NOMINAL_MASS),
        target_velocity=np.zeros((n, 2)),
        init_pos=np.zeros((n, 2)),
        init_vel=np.zeros((n, 2)),
        body_mass=np.full(n, NOMINAL_MASS),
        damping=np.full(n, 0.05),
        actuator_gain=np.ones(n),
        actuator_tau=np.full(n, DT),
        const_force=np.zeros((n, 2)),
        const_start=zeros.copy(),
        const_duration=zeros.copy(),
        impulse_peak=np.zeros((n, 2)),
        impulse_start=zeros.copy(),
        impulse_duration=np.full(n, 0.5),
        wall_x=np.full(n, np.inf),
    )


def randomize_body(rng: np.random.Generator, scenario: Scenario) -> None:
    n = scenario.n
    scenario.body_mass[:] = rng.uniform(*BODY_MASS_RANGE, size=n)
    scenario.damping[:] = rng.uniform(*DAMPING_RANGE, size=n)
    scenario.actuator_gain[:] = rng.uniform(*ACTUATOR_GAIN_RANGE, size=n)
    scenario.actuator_tau[:] = rng.uniform(*ACTUATOR_TAU_RANGE, size=n)


def set_stiffness(scenario: Scenario, kp: np.ndarray | float) -> None:
    """Set Kp and the critically damped Kd = 2 sqrt(Kp) used throughout the paper."""
    scenario.kp[:] = kp
    scenario.kd[:] = 2.0 * np.sqrt(scenario.kp)


def ramp_profile(t: np.ndarray, start: np.ndarray, duration: np.ndarray) -> np.ndarray:
    """Triangular push profile in [0, 1]; the impulse is 0.5 * duration * peak."""
    tau = t - start
    half = np.maximum(0.5 * duration, 1e-6)
    rising = tau / half
    falling = (duration - tau) / half
    scale = np.where(tau <= 0.5 * duration, rising, falling)
    active = (tau >= 0.0) & (tau <= duration)
    return np.where(active, np.maximum(scale, 0.0), 0.0)


class PlanarForceSim:
    """Vectorized planar body with an actuator lag and a fall model.

    State after ``k`` steps is (pos, vel). ``step`` applies the external force
    evaluated at the current state, so the force returned by ``step`` is the
    one acting during the transition. Fallen episodes are frozen.
    """

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.n = scenario.n
        self.reset()

    def reset(self, idx: np.ndarray | None = None) -> None:
        sc = self.scenario
        if idx is None:
            n = self.n
            self.t = np.zeros(n)
            self.k = np.zeros(n, dtype=np.int64)
            self.pos = np.array(sc.init_pos, dtype=np.float64)
            self.vel = np.array(sc.init_vel, dtype=np.float64)
            self.acc_eff = np.zeros((n, 2))
            self.tilt = np.zeros(n)
            self.failed = np.zeros(n, dtype=bool)
            self.last_actions = np.zeros((n, ACTION_HISTORY, 2))
            return
        self.t[idx] = 0.0
        self.k[idx] = 0
        self.pos[idx] = sc.init_pos[idx]
        self.vel[idx] = sc.init_vel[idx]
        self.acc_eff[idx] = 0.0
        self.tilt[idx] = 0.0
        self.failed[idx] = False
        self.last_actions[idx] = 0.0

    def external_forces(self) -> tuple[np.ndarray, np.ndarray]:
        """Total external force and its wall component at the current state."""
        sc = self.scenario
        const_on = (self.t >= sc.const_start) & (self.t < sc.const_start + sc.const_duration)
        force = sc.const_force * const_on[:, None]
        force = force + sc.impulse_peak * ramp_profile(self.t, sc.impulse_start, sc.impulse_duration)[:, None]
        wall = np.zeros((self.n, 2))
        penetration = self.pos[:, 0] - sc.wall_x
        inside = penetration > 0.0
        wall[:, 0] = np.where(
            inside, -WALL_KP * np.where(inside, penetration, 0.0) - WALL_KD * np.maximum(self.vel[:, 0], 0.0), 0.0
        )
        return force + wall, wall

    def observation(self) -> np.ndarray:
        sc = self.scenario
        phase = 2.0 * np.pi * GAIT_FREQUENCY_HZ * self.t
        obs = np.concatenate(
            [
                sc.target_velocity,
                (sc.kp / 40.0)[:, None],
                (sc.kd / 20.0)[:, None],
                (sc.virtual_mass / 20.0)[:, None],
                self.vel,
                self.tilt[:, None],
                np.sin(phase)[:, None],
                np.cos(phase)[:, None],
                self.last_actions.reshape(self.n, -1),
            ],
            axis=1,
        )
        return obs.astype(np.float32)

    def step(self, action: np.ndarray) -> dict[str, np.ndarray]:
        sc = self.scenario
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(self.n, ACTION_DIM), -ACTION_LIMIT, ACTION_LIMIT)
        force, wall = self.external_forces()
        alive = ~self.failed
        alpha = np.clip(DT / np.maximum(sc.actuator_tau, 1e-6), 0.0, 1.0)[:, None]
        acc_eff = self.acc_eff + alpha * (sc.actuator_gain[:, None] * action - self.acc_eff)
        acc = acc_eff + force / sc.body_mass[:, None] - sc.damping[:, None] * self.vel
        vel = self.vel + acc * DT
        pos = self.pos + vel * DT
        reaction = np.abs(acc_eff[:, 1])
        tilt = TILT_DECAY * self.tilt + TILT_REACTION_GAIN * reaction + TILT_FORCE_GAIN * np.abs(force[:, 1])
        new_fail = ((reaction > FALL_REACTION) & (np.abs(force[:, 1]) > FALL_PUSH_FORCE)) | (tilt > FALL_TILT)

        keep = alive[:, None]
        self.acc_eff = np.where(keep, acc_eff, self.acc_eff)
        self.vel = np.where(keep, vel, self.vel)
        self.pos = np.where(keep, pos, self.pos)
        self.tilt = np.where(alive, tilt, self.tilt)
        newly_failed = alive & new_fail
        self.failed = self.failed | newly_failed
        self.last_actions = np.roll(self.last_actions, shift=-1, axis=1)
        self.last_actions[:, -1, :] = action
        self.t = self.t + DT
        self.k = self.k + 1
        return {
            "force": force,
            "wall_force": wall,
            "alive": alive,
            "newly_failed": newly_failed,
        }
