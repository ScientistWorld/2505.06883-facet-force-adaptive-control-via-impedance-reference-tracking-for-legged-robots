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
