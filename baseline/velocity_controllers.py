#!/usr/bin/env python3
"""Velocity-tracking baselines from the FACET paper, at benchmark scale.

The evaluators own and run their own copies of these controllers (so the
comparison side cannot be changed by a submission). This module exports the
same controllers as TorchScript policies; they double as a minimal example of
the policy interface (see data/facet_planar_env.py). Run with
``python -m baseline.velocity_controllers --output-root outputs/baselines``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

# name -> (forward gain, lateral gain) on velocity error; actions in m/s^2.
VELOCITY_BASELINES = {
    "velocity_vanilla": (5.5, 8.0),  # stiff velocity tracking
    "velocity_robust": (5.5, 4.0),   # softer lateral gain, tolerates medium pushes
}


class VelocityPolicy(nn.Module):
    def __init__(self, forward_gain: float, lateral_gain: float):
        super().__init__()
        self.forward_gain = float(forward_gain)
        self.lateral_gain = float(lateral_gain)

    @torch.jit.export
    def reset(self, n: int) -> None:
        pass

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        target = obs[:, 0:2]
        velocity = obs[:, 5:7]
        gains = torch.tensor([self.forward_gain, self.lateral_gain], dtype=obs.dtype)
        return gains * (target - velocity)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("outputs/baselines"))
    args = parser.parse_args()
    for name, gains in VELOCITY_BASELINES.items():
        out = args.output_root / f"export_{name}" / "policy.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.jit.script(VelocityPolicy(*gains)).save(str(out))
        print(f"exported {out}")


if __name__ == "__main__":
    main()
