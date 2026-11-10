#!/usr/bin/env python3
"""Held-out test evaluator for the FACET force-interaction benchmark.

Every method is a TorchScript policy at ``<root>/<method>/policy.pt``. This
evaluator owns everything on the other side of the comparison: the simulator
(a frozen copy of the benchmark dynamics), the scenarios and their seeds, the commanded operating points, the velocity-tracking baselines, the
episode counts and every threshold. It rolls each policy out itself and
recomputes all metrics from its own rollouts; files written by a method other
than ``policy.pt`` are ignored.

A method whose policy is missing, fails to load, raises, returns a wrong shape
or a non-finite action is a failed run and receives the worst value of every
metric in every experiment. Proposed methods listed in scoring/reference.json
are always reported, so deleting a policy cannot hide a result.

Usage: python eval/test/evaluate.py [checkpoint_root] [--output path]
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import traceback
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable

import numpy as np
import torch

SLICE = "test"
SEEDS = {
    "large_lateral_impulse": 640_271_903,
    "soft_wall_collision": 640_271_917,
    "impedance_tracking": 640_271_931,
}

# ---------------------------------------------------------------------------
# Frozen simulator copy (must match data/facet_planar_env.py dynamics).
# ---------------------------------------------------------------------------
DT = 0.02
NOMINAL_MASS = 12.0
ACTION_LIMIT = 8.0
FALL_REACTION = 5.2
FALL_PUSH_FORCE = 80.0
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
BODY_MASS_RANGE = (10.0, 14.0)
DAMPING_RANGE = (0.03, 0.07)
ACTUATOR_GAIN_RANGE = (0.85, 1.15)
ACTUATOR_TAU_RANGE = (0.02, 0.08)


@dataclass
class Scenario:
    kp: np.ndarray
    kd: np.ndarray
    virtual_mass: np.ndarray
    target_velocity: np.ndarray
    init_pos: np.ndarray
    init_vel: np.ndarray
    body_mass: np.ndarray
    damping: np.ndarray
    actuator_gain: np.ndarray
    actuator_tau: np.ndarray
    const_force: np.ndarray
    const_start: np.ndarray
    const_duration: np.ndarray
    impulse_peak: np.ndarray
    impulse_start: np.ndarray
    impulse_duration: np.ndarray
    wall_x: np.ndarray

    @property
    def n(self) -> int:
        return int(self.kp.shape[0])

    def take(self, idx: np.ndarray) -> "Scenario":
        return Scenario(**{item.name: np.array(getattr(self, item.name)[idx]) for item in fields(self)})


def blank_scenario(n: int) -> Scenario:
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
    scenario.kp[:] = kp
    scenario.kd[:] = 2.0 * np.sqrt(scenario.kp)


def ramp_profile(t: np.ndarray, start: np.ndarray, duration: np.ndarray) -> np.ndarray:
    tau = t - start
    half = np.maximum(0.5 * duration, 1e-6)
    rising = tau / half
    falling = (duration - tau) / half
    scale = np.where(tau <= 0.5 * duration, rising, falling)
    active = (tau >= 0.0) & (tau <= duration)
    return np.where(active, np.maximum(scale, 0.0), 0.0)


class PlanarForceSim:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.n = scenario.n
        sc = scenario
        n = self.n
        self.t = np.zeros(n)
        self.k = np.zeros(n, dtype=np.int64)
        self.pos = np.array(sc.init_pos, dtype=np.float64)
        self.vel = np.array(sc.init_vel, dtype=np.float64)
        self.acc_eff = np.zeros((n, 2))
        self.tilt = np.zeros(n)
        self.failed = np.zeros(n, dtype=bool)
        self.last_actions = np.zeros((n, ACTION_HISTORY, 2))

    def external_forces(self) -> tuple[np.ndarray, np.ndarray]:
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
        return {"force": force, "wall_force": wall, "alive": alive, "newly_failed": newly_failed}


# ---------------------------------------------------------------------------
# Benchmark protocol (fixed by the evaluator).
# ---------------------------------------------------------------------------
CHUNK = 64
COMMAND_SPEED = 1.5
IMPULSE_COMMAND_KP = 4.0
IMPULSE_LEVELS_N = (100.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1000.0, 1200.0)
IMPULSE_PER_LEVEL = 32
IMPULSE_STEPS = 350
WALL_COMMAND_KP = 0.5
WALL_POSITION = 3.0
WALL_EPISODES = 128
WALL_STEPS = 200
TRACK_EPISODES = 256
TRACK_STEPS = 350
VIRTUAL_MASS_CHOICES = (6.0, 12.0, 24.0)   # inertia shaping: 0.5x, 1x, 2x the nominal body
LAGS = (8, 16, 32)
VEL_ERR_CAP = 5.0
POS_ERR_CAP = 2.5
RECOVERY_SCALE = 0.5
WALL_WINDOW_STEPS = 25          # 0.5 s after first contact (paper Sec. 10.2)
CONTACT_TRACK_STEPS = 50        # 1.0 s after first contact
ENTRY_SPEED_TOLERANCE = 0.3
RETREAT_TOLERANCE = 0.5         # m/s RMS the body may fall behind the commanded response after contact
INVALID_CONTACT_PENALTY = 2.0   # invalid episode counts as twice the baseline impulse
BASELINES = {
    "velocity_vanilla": (5.5, 8.0),
    "velocity_robust": (5.5, 4.0),
}
COMPARISON_BASELINE = "velocity_robust"
WORST = {
    "large_lateral_impulse": {
        "impulse_recovery_score": 0.0,
        "success_rate": 0.0,
        "velocity_tracking_rmse_mps": VEL_ERR_CAP,
    },
    "soft_wall_collision": {
        "post_contact_impulse_reduction": -1.0,
        "wall_contact_rate": 0.0,
        "entry_speed_error_mps": COMMAND_SPEED,
        "contact_tracking_rmse_mps": VEL_ERR_CAP,
    },
    "impedance_tracking": {
        "velocity_tracking_rmse_mps": VEL_ERR_CAP,
        "position_tracking_rmse_m": POS_ERR_CAP,
        "success_rate": 0.0,
    },
}
BOUNDS = {
    "impulse_recovery_score": (0.0, 1.0),
    "success_rate": (0.0, 1.0),
    "velocity_tracking_rmse_mps": (0.0, VEL_ERR_CAP),
    "post_contact_impulse_reduction": (-1.0, 1.0),
    "wall_contact_rate": (0.0, 1.0),
    "entry_speed_error_mps": (0.0, COMMAND_SPEED),
    "contact_tracking_rmse_mps": (0.0, VEL_ERR_CAP),
    "position_tracking_rmse_m": (0.0, POS_ERR_CAP),
}


def impulse_scenario(seed: int) -> tuple[Scenario, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(IMPULSE_LEVELS_N) * IMPULSE_PER_LEVEL
    sc = blank_scenario(n)
    randomize_body(rng, sc)
    set_stiffness(sc, IMPULSE_COMMAND_KP)
    sc.target_velocity[:, 0] = COMMAND_SPEED
    sc.init_vel[:, 0] = COMMAND_SPEED
    levels = np.repeat(np.asarray(IMPULSE_LEVELS_N), IMPULSE_PER_LEVEL)
    sc.impulse_peak[:, 1] = levels * rng.choice([-1.0, 1.0], size=n)
    sc.impulse_start[:] = rng.uniform(1.5, 2.5, size=n)
    sc.impulse_duration[:] = rng.uniform(0.4, 0.6, size=n)
    order = rng.permutation(n)
    return sc.take(order), levels[order]


def wall_scenario(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    n = WALL_EPISODES
    sc = blank_scenario(n)
    randomize_body(rng, sc)
    set_stiffness(sc, WALL_COMMAND_KP)
    sc.target_velocity[:, 0] = COMMAND_SPEED
    sc.init_vel[:, 0] = COMMAND_SPEED
    sc.wall_x[:] = WALL_POSITION
    sc.init_pos[:, 0] = WALL_POSITION - rng.uniform(0.75, 1.5, size=n)
    return sc


def tracking_scenario(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    n = TRACK_EPISODES
    sc = blank_scenario(n)
    randomize_body(rng, sc)
    set_stiffness(sc, np.exp(rng.uniform(math.log(0.5), math.log(32.0), size=n)))
    sc.virtual_mass[:] = rng.choice(np.asarray(VIRTUAL_MASS_CHOICES), size=n)
    sc.target_velocity[:, 0] = rng.uniform(-0.5, 2.0, size=n)
    sc.target_velocity[:, 1] = rng.uniform(-0.5, 0.5, size=n)
    sc.init_vel[:, 0] = rng.uniform(-0.5, 2.0, size=n)
    sc.init_vel[:, 1] = rng.uniform(-0.5, 0.5, size=n)
    has_const = rng.random(n) < 0.7
    sc.const_force[:] = rng.uniform(-40.0, 40.0, size=(n, 2)) * has_const[:, None]
    sc.const_start[:] = rng.uniform(0.0, 4.0, size=n)
    sc.const_duration[:] = rng.uniform(1.0, 4.0, size=n)
    has_push = rng.random(n) < 0.8
    peak = rng.uniform(80.0, 200.0, size=(n, 2)) * rng.choice([-1.0, 1.0], size=(n, 2))
    sc.impulse_peak[:] = peak * has_push[:, None]
    sc.impulse_start[:] = rng.uniform(0.5, 5.5, size=n)
    sc.impulse_duration[:] = rng.uniform(0.4, 0.6, size=n)
    return sc


# ---------------------------------------------------------------------------
# Policies.
# ---------------------------------------------------------------------------
class PolicyError(RuntimeError):
    pass


ActFn = Callable[[np.ndarray], np.ndarray]


class ScriptedPolicy:
    """A TorchScript policy loaded fresh (and reset) for every chunk of episodes."""

    def __init__(self, path: Path):
        if not path.is_file():
            raise PolicyError(f"missing policy file {path}")
        self.blob = path.read_bytes()
        torch.jit.load(io.BytesIO(self.blob), map_location="cpu")

    def make(self, n: int) -> ActFn:
        torch.manual_seed(0)  # stochastic policies are scored reproducibly
        module = torch.jit.load(io.BytesIO(self.blob), map_location="cpu")
        module.eval()
        if hasattr(module, "reset"):
            module.reset(n)

        def act(obs: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                out = module(torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32)))
            if not isinstance(out, torch.Tensor):
                raise PolicyError("policy must return a tensor")
            arr = out.detach().to(torch.float64).cpu().numpy()
            if arr.shape != (n, ACTION_DIM):
                raise PolicyError(f"policy returned shape {tuple(arr.shape)}, expected {(n, ACTION_DIM)}")
            if not np.all(np.isfinite(arr)):
                raise PolicyError("policy returned a non-finite action")
            return arr

        return act


class VelocityBaseline:
    """Stiff velocity-tracking controller with fixed feedback gains."""

    def __init__(self, forward_gain: float, lateral_gain: float):
        self.forward_gain = forward_gain
        self.lateral_gain = lateral_gain

    def make(self, n: int) -> ActFn:
        gains = np.asarray([self.forward_gain, self.lateral_gain], dtype=np.float32)

        def act(obs: np.ndarray) -> np.ndarray:
            obs = obs.astype(np.float32)
            return (gains * (obs[:, 0:2] - obs[:, 5:7])).astype(np.float64)

        return act


@dataclass
class Trajectory:
    pos: np.ndarray        # [T + 1, n, 2]
    vel: np.ndarray        # [T + 1, n, 2]
    force: np.ndarray      # [T, n, 2] external force during step j
    wall: np.ndarray       # [T, n, 2]
    failed: np.ndarray     # [n]
    fail_state: np.ndarray # [n] index of the state at which the fall registered (T + 1 = never)


def rollout(policy, scenario: Scenario, steps: int) -> Trajectory:
    n = scenario.n
    pos = np.zeros((steps + 1, n, 2))
    vel = np.zeros((steps + 1, n, 2))
    force = np.zeros((steps, n, 2))
    wall = np.zeros((steps, n, 2))
    fail_state = np.full(n, steps + 1, dtype=np.int64)
    for start in range(0, n, CHUNK):
        idx = np.arange(start, min(start + CHUNK, n))
        sim = PlanarForceSim(scenario.take(idx))
        act = policy.make(len(idx))
        pos[0, idx] = sim.pos
        vel[0, idx] = sim.vel
        for k in range(steps):
            action = act(sim.observation())
            info = sim.step(action)
            pos[k + 1, idx] = sim.pos
            vel[k + 1, idx] = sim.vel
            force[k, idx] = info["force"]
            wall[k, idx] = info["wall_force"]
            fail_state[idx[info["newly_failed"]]] = k + 1
    return Trajectory(pos, vel, force, wall, fail_state <= steps, fail_state)


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------
def tracking_errors(traj: Trajectory, sc: Scenario, with_signed: bool = False):
    """Squared velocity/position errors against the commanded impedance response.

    For every state k and window L in LAGS, the commanded virtual
    mass-spring-damper is integrated over the last L control steps from the
    body's own state, under the same external forces, and compared with the
    body's state at k. Rows are states k = 1..T; states after a fall get the cap.
    With ``with_signed`` also returns the window-averaged signed x-velocity
    shortfall (commanded response minus body), zero after a fall.
    """
    pos, vel, force = traj.pos, traj.vel, traj.force
    steps = force.shape[0]
    kp = sc.kp[None, :, None]
    kd = sc.kd[None, :, None]
    mass = sc.virtual_mass[None, :, None]
    vt = sc.target_velocity[None, :, :]
    k = np.arange(1, steps + 1)
    ev = np.zeros((steps, sc.n))
    ep = np.zeros((steps, sc.n))
    sx = np.zeros((steps, sc.n))
    for lag in LAGS:
        s = np.maximum(k - lag, 0)
        x = pos[s].copy()
        v = vel[s].copy()
        for j in range(lag):
            idx = s + j
            active = (idx < k)[:, None, None]
            idc = np.minimum(idx, steps - 1)
            spring = kp * (pos[idc] - x) + kd * (vt - v)
            acc = (spring + force[idc]) / mass
            v_new = v + acc * DT
            x_new = x + v_new * DT
            v = np.where(active, v_new, v)
            x = np.where(active, x_new, x)
        ev += np.sum((vel[k] - v) ** 2, axis=-1)
        ep += np.sum((pos[k] - x) ** 2, axis=-1)
        sx += v[..., 0] - vel[k][..., 0]
    ev /= len(LAGS)
    ep /= len(LAGS)
    sx /= len(LAGS)
    after_fall = k[:, None] > traj.fail_state[None, :]
    ev = np.where(after_fall, VEL_ERR_CAP**2, np.minimum(ev, VEL_ERR_CAP**2))
    ep = np.where(after_fall, POS_ERR_CAP**2, np.minimum(ep, POS_ERR_CAP**2))
    if with_signed:
        return ev, ep, np.where(after_fall, 0.0, np.clip(sx, -VEL_ERR_CAP, VEL_ERR_CAP))
    return ev, ep


def impulse_metrics(traj: Trajectory, sc: Scenario) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    ev, _ = tracking_errors(traj, sc)
    steps = ev.shape[0]
    k = np.arange(1, steps + 1)[:, None]
    window = k >= np.floor(sc.impulse_start / DT)[None, :]
    rmse = np.sqrt((ev * window).sum(0) / np.maximum(window.sum(0), 1))
    survived = ~traj.failed
    score = survived * np.exp(-rmse / RECOVERY_SCALE)
    episodes = {"score": score, "survived": survived.astype(float), "rmse": rmse}
    return {
        "impulse_recovery_score": float(score.mean()),
        "success_rate": float(survived.mean()),
        "velocity_tracking_rmse_mps": float(rmse.mean()),
    }, episodes


def wall_episodes(traj: Trajectory, sc: Scenario) -> dict[str, np.ndarray]:
    ev, _, shortfall = tracking_errors(traj, sc, with_signed=True)
    steps = traj.force.shape[0]
    n = sc.n
    wall = np.abs(traj.wall[:, :, 0])
    touching = wall > 0.0
    has_contact = touching.any(axis=0)
    first = np.where(has_contact, touching.argmax(axis=0), steps)
    entry_speed = traj.vel[np.minimum(first, steps), np.arange(n), 0]
    j = np.arange(steps)[:, None]
    in_window = (j >= first[None, :]) & (j < first[None, :] + WALL_WINDOW_STEPS)
    impulse = (wall * in_window).sum(axis=0) * DT
    entry_error = np.where(has_contact, np.abs(entry_speed - sc.target_velocity[:, 0]), COMMAND_SPEED)
    entry_error = np.minimum(entry_error, COMMAND_SPEED)
    k = np.arange(1, steps + 1)[:, None]
    contact_window = (k > first[None, :]) & (k <= first[None, :] + CONTACT_TRACK_STEPS)
    count = np.maximum(contact_window.sum(0), 1)
    contact_rmse = np.sqrt((ev * contact_window).sum(0) / count)
    # Falling behind the commanded compliant response after contact (braking or
    # retreating from the wall instead of following the commanded impedance).
    retreat_rms = np.sqrt((np.maximum(shortfall, 0.0) ** 2 * contact_window).sum(0) / count)
    valid = (
        has_contact
        & (first + WALL_WINDOW_STEPS <= steps)
        & (entry_error <= ENTRY_SPEED_TOLERANCE)
        & (traj.fail_state > first + WALL_WINDOW_STEPS)
        & (retreat_rms <= RETREAT_TOLERANCE)
    )
    contact_rmse = np.where(valid, contact_rmse, VEL_ERR_CAP)
    # Impulse per unit approach speed (scaled to the commanded 1.5 m/s), so that
    # entering slower within the tolerance does not lower the score.
    normalized = impulse * COMMAND_SPEED / np.maximum(np.abs(entry_speed), 0.5)
    return {
        "impulse": normalized,
        "raw_impulse": impulse,
        "valid": valid,
        "entry_error": entry_error,
        "contact_rmse": contact_rmse,
        "retreat_rms": np.where(has_contact, retreat_rms, VEL_ERR_CAP),
    }


def wall_metrics(ep: dict[str, np.ndarray], base: dict[str, np.ndarray]) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    reference = base["impulse"]
    charged = np.where(ep["valid"], ep["impulse"], INVALID_CONTACT_PENALTY * reference)
    reduction = 1.0 - charged.sum() / max(reference.sum(), 1e-9)
    per_episode = 1.0 - charged / np.maximum(reference, 1e-9)
    episodes = {
        "reduction": np.clip(per_episode, -1.0, 1.0),
        "charged": charged,
        "baseline_impulse": reference,
        "valid": ep["valid"].astype(float),
        "entry_error": ep["entry_error"],
        "contact_rmse": ep["contact_rmse"],
        "retreat_rms": ep["retreat_rms"],
    }
    return {
        "post_contact_impulse_reduction": float(np.clip(reduction, -1.0, 1.0)),
        "wall_contact_rate": float(ep["valid"].mean()),
        "entry_speed_error_mps": float(ep["entry_error"].mean()),
        "contact_tracking_rmse_mps": float(ep["contact_rmse"].mean()),
    }, episodes


def tracking_metrics(traj: Trajectory, sc: Scenario) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    ev, ep = tracking_errors(traj, sc)
    vel_rmse = np.sqrt(ev.mean(axis=0))
    pos_rmse = np.sqrt(ep.mean(axis=0))
    survived = ~traj.failed
    episodes = {"vel_rmse": vel_rmse, "pos_rmse": pos_rmse, "survived": survived.astype(float)}
    return {
        "velocity_tracking_rmse_mps": float(vel_rmse.mean()),
        "position_tracking_rmse_m": float(pos_rmse.mean()),
        "success_rate": float(survived.mean()),
    }, episodes


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def proposed_names(reference_path: Path) -> set[str]:
    try:
        payload = json.loads(reference_path.read_text())
    except (OSError, ValueError):
        return set()
    names: set[str] = set()
    for exp in (payload.get("experiments") or {}).values():
        if not isinstance(exp, dict):
            continue
        for name, entry in (exp.get("results") or {}).items():
            if isinstance(entry, dict) and entry.get("type") == "proposed":
                names.add(str(name))
    return names


def discover_methods(root: Path, reference_path: Path) -> list[str]:
    names: set[str] = set()
    if root.is_dir():
        for directory in root.iterdir():
            if directory.is_dir() and not directory.name.startswith((".", "_")) and (directory / "policy.pt").is_file():
                names.add(directory.name)
    names |= proposed_names(reference_path)
    return sorted(name for name in names if name not in BASELINES)


class Benchmark:
    def __init__(self) -> None:
        self.impulse_sc, self.impulse_levels = impulse_scenario(SEEDS["large_lateral_impulse"])
        self.wall_sc = wall_scenario(SEEDS["soft_wall_collision"])
        self.track_sc = tracking_scenario(SEEDS["impedance_tracking"])
        robust = VelocityBaseline(*BASELINES[COMPARISON_BASELINE])
        self.wall_reference = wall_episodes(rollout(robust, self.wall_sc, WALL_STEPS), self.wall_sc)

    def score(self, policy) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, list[float]]]]:
        results: dict[str, dict[str, float]] = {}
        details: dict[str, dict[str, list[float]]] = {}
        metrics, episodes = impulse_metrics(rollout(policy, self.impulse_sc, IMPULSE_STEPS), self.impulse_sc)
        episodes["level_n"] = self.impulse_levels
        results["large_lateral_impulse"], details["large_lateral_impulse"] = metrics, episodes
        wall_ep = wall_episodes(rollout(policy, self.wall_sc, WALL_STEPS), self.wall_sc)
        results["soft_wall_collision"], details["soft_wall_collision"] = wall_metrics(wall_ep, self.wall_reference)
        results["impedance_tracking"], details["impedance_tracking"] = tracking_metrics(
            rollout(policy, self.track_sc, TRACK_STEPS), self.track_sc
        )
        for exp, metrics in results.items():
            for name, value in metrics.items():
                low, high = BOUNDS[name]
                if not (math.isfinite(value) and low - 1e-9 <= value <= high + 1e-9):
                    raise PolicyError(f"{exp}.{name}={value} outside [{low}, {high}]")
        as_lists = {exp: {key: np.asarray(val, dtype=float).tolist() for key, val in eps.items()} for exp, eps in details.items()}
        return results, as_lists


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_root", nargs="?", type=Path, default=Path("/home/user/checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("/home/user/scoring/scores_test.json"))
    parser.add_argument("--reference", type=Path, default=Path("/home/user/scoring/reference.json"))
    parser.add_argument("--details", type=Path, default=None, help="optional per-episode dump")
    args = parser.parse_args()

    args.output.unlink(missing_ok=True)
    torch.set_num_threads(1)
    benchmark = Benchmark()
    scores: dict = {"experiments": {exp: {"results": {}} for exp in WORST}, "metadata": {}}
    details: dict = {}
    status: dict[str, dict[str, str]] = {}

    entries: list[tuple[str, object | None, str | None]] = []
    for name, gains in BASELINES.items():
        entries.append((name, VelocityBaseline(*gains), None))
    for name in discover_methods(args.checkpoint_root, args.reference):
        try:
            entries.append((name, ScriptedPolicy(args.checkpoint_root / name / "policy.pt"), None))
        except Exception as exc:  # noqa: BLE001 - any load failure is a failed run
            entries.append((name, None, f"{type(exc).__name__}: {exc}"))

    for name, policy, error in entries:
        result = None
        if policy is not None:
            try:
                result, details[name] = benchmark.score(policy)
            except Exception as exc:  # noqa: BLE001 - any rollout failure is a failed run
                error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
        if result is None:
            result = {exp: dict(values) for exp, values in WORST.items()}
            status[name] = {"status": "failed", "reason": error or "unknown failure"}
        else:
            status[name] = {"status": "ok"}
        for exp, metrics in result.items():
            scores["experiments"][exp]["results"][name] = metrics
        print(f"[evaluate:{SLICE}] {name}: {status[name]['status']} {json.dumps(result, sort_keys=True)}", flush=True)

    scores["metadata"] = {
        "slice": SLICE,
        "episodes": {
            "large_lateral_impulse": int(benchmark.impulse_sc.n),
            "soft_wall_collision": int(benchmark.wall_sc.n),
            "impedance_tracking": int(benchmark.track_sc.n),
        },
        "methods": status,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n")
    if args.details is not None:
        args.details.parent.mkdir(parents=True, exist_ok=True)
        args.details.write_text(json.dumps(details))
    return 0


if __name__ == "__main__":
    sys.exit(main())
