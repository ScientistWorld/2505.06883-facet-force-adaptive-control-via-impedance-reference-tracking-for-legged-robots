"""TorchScript probe policies used to re-run shortcuts through the hardened test.

None of these read eval/test; they only use the public policy interface.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class Passive(nn.Module):
    """Zero action: never resists, never tracks (the 'do nothing' shortcut)."""

    @torch.jit.export
    def reset(self, n: int) -> None:
        pass

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(obs.shape[0], 2, dtype=obs.dtype)


class Overspeed(nn.Module):
    """Maximum forward acceleration, zero lateral action."""

    @torch.jit.export
    def reset(self, n: int) -> None:
        pass

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(obs.shape[0], 2, dtype=obs.dtype)
        out[:, 0] = 8.0
        return out


class LinearGains(nn.Module):
    """Velocity feedback a = K (v_target - v); K_y = 0 is pure lateral yielding."""

    def __init__(self, kx: float, ky: float):
        super().__init__()
        self.kx = float(kx)
        self.ky = float(ky)

    @torch.jit.export
    def reset(self, n: int) -> None:
        pass

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        gains = torch.tensor([self.kx, self.ky], dtype=obs.dtype)
        return gains * (obs[:, 0:2] - obs[:, 5:7])


class AnalyticImpedance(nn.Module):
    """Hand-written impedance law for a nominal body: a = Kd (v_t - v) / m + c v."""

    @torch.jit.export
    def reset(self, n: int) -> None:
        pass

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        kd = obs[:, 3:4] * 20.0
        mass = obs[:, 4:5] * 20.0
        return kd * (obs[:, 0:2] - obs[:, 5:7]) / mass + 0.05 * obs[:, 5:7]


class AnalyticClipped(nn.Module):
    """Analytic impedance law with the lateral action clipped below the fall threshold."""

    def __init__(self, clip: float):
        super().__init__()
        self.clip = float(clip)

    @torch.jit.export
    def reset(self, n: int) -> None:
        pass

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        kd = obs[:, 3:4] * 20.0
        mass = obs[:, 4:5] * 20.0
        acc = kd * (obs[:, 0:2] - obs[:, 5:7]) / mass + 0.05 * obs[:, 5:7]
        lateral = torch.clamp(acc[:, 1:2], -self.clip, self.clip)
        return torch.cat([acc[:, 0:1], lateral], dim=1)


class RetreatOnContact(nn.Module):
    """Wraps a policy; on the wall-test command, brakes once an unexplained
    deceleration (wall contact) is seen, instead of following the compliance."""

    def __init__(self, inner: torch.jit.ScriptModule, brake: float):
        super().__init__()
        self.inner = inner
        self.brake = float(brake)
        self.register_buffer("prev_vx", torch.zeros(1))
        self.register_buffer("latched", torch.zeros(1, dtype=torch.bool))
        self.steps = 0

    @torch.jit.export
    def reset(self, n: int) -> None:
        self.inner.reset(n)
        self.prev_vx = torch.zeros(n)
        self.latched = torch.zeros(n, dtype=torch.bool)
        self.steps = 0

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.inner(obs)
        vx = obs[:, 5]
        if self.prev_vx.shape[0] != obs.shape[0]:
            self.prev_vx = vx.clone()
            self.latched = torch.zeros(obs.shape[0], dtype=torch.bool)
            self.steps = 0
        wall_cmd = (torch.abs(obs[:, 2] * 40.0 - 0.5) < 0.01) & (torch.abs(obs[:, 0] - 1.5) < 0.01) & (torch.abs(obs[:, 1]) < 0.01)
        if self.steps > 0:
            self.latched = self.latched | (wall_cmd & ((vx - self.prev_vx) < -0.04))
        self.steps = self.steps + 1
        self.prev_vx = vx.clone()
        braked = torch.cat([torch.clamp(out[:, 0:1] - self.brake, -8.0, 8.0), out[:, 1:2]], dim=1)
        return torch.where(self.latched[:, None], braked, out)


class LateralClip(nn.Module):
    """Wraps a policy and clips its lateral action (a hand-added safety filter)."""

    def __init__(self, inner: torch.jit.ScriptModule, clip: float):
        super().__init__()
        self.inner = inner
        self.clip = float(clip)

    @torch.jit.export
    def reset(self, n: int) -> None:
        self.inner.reset(n)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.inner(obs)
        return torch.cat([out[:, 0:1], torch.clamp(out[:, 1:2], -self.clip, self.clip)], dim=1)


class WallDodger(nn.Module):
    """Wraps a policy but brakes hard whenever a very soft stiffness is commanded."""

    def __init__(self, inner: torch.jit.ScriptModule):
        super().__init__()
        self.inner = inner

    @torch.jit.export
    def reset(self, n: int) -> None:
        self.inner.reset(n)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.inner(obs)
        soft = (obs[:, 2] * 40.0) < 1.0
        braked = out.clone()
        braked[:, 0] = -8.0
        return torch.where(soft[:, None], braked, out)


class DOBImpedance(nn.Module):
    """Hand-written disturbance-observer impedance controller (headroom probe).

    Estimates the external force per unit body mass from the last velocity
    change, the previous action (nominal first-order actuator lag) and nominal
    drag, then commands the acceleration that makes a nominal 12 kg body follow
    the commanded virtual mass-spring-damper (fixed-target-velocity mode). The
    lateral action is limited below the fall threshold. A hand-engineered
    model-based controller is out of bounds as a submission (DIRECTION.md); it
    is scored only to measure how far above the learned reference the test reaches.
    """

    def __init__(self, alpha: float = 0.4, lateral_clip: float = 4.0):
        super().__init__()
        self.alpha = float(alpha)
        self.lateral_clip = float(lateral_clip)
        self.mass = 12.0
        self.drag = 0.05
        self.dt = 0.02
        self.register_buffer("prev_v", torch.zeros(1, 2))
        self.register_buffer("acc_hat", torch.zeros(1, 2))
        self.register_buffer("f_hat", torch.zeros(1, 2))
        self.steps = 0

    @torch.jit.export
    def reset(self, n: int) -> None:
        self.prev_v = torch.zeros(n, 2)
        self.acc_hat = torch.zeros(n, 2)
        self.f_hat = torch.zeros(n, 2)
        self.steps = 0

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        n = obs.shape[0]
        if self.prev_v.shape[0] != n:
            self.reset(n)
        v = obs[:, 5:7]
        last_action = obs[:, 14:16]
        acc_now = self.acc_hat + self.alpha * (last_action - self.acc_hat)
        if self.steps > 0:
            self.f_hat = (v - self.prev_v) / self.dt - acc_now + self.drag * self.prev_v
        self.acc_hat = acc_now
        self.prev_v = v.clone()
        self.steps = self.steps + 1
        kd = obs[:, 3:4] * 20.0
        virtual_mass = obs[:, 4:5] * 20.0
        desired = (kd * (obs[:, 0:2] - v) + self.mass * self.f_hat) / virtual_mass
        action = desired - self.f_hat + self.drag * v
        lateral = torch.clamp(action[:, 1:2], -self.lateral_clip, self.lateral_clip)
        return torch.clamp(torch.cat([action[:, 0:1], lateral], dim=1), -8.0, 8.0)


class NaNPolicy(nn.Module):
    @torch.jit.export
    def reset(self, n: int) -> None:
        pass

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.full((obs.shape[0], 2), float("nan"), dtype=obs.dtype)


class WrongShape(nn.Module):
    @torch.jit.export
    def reset(self, n: int) -> None:
        pass

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(obs.shape[0], 3, dtype=obs.dtype)


def save(module: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(module).save(str(path))
