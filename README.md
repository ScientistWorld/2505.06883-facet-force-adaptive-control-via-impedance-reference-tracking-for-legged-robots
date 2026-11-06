# FACET: Force-Adaptive Control via Impedance Reference Tracking for Legged Robots

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2505.06883"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2505.06883-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2505.06883](https://arxiv.org/abs/2505.06883) |

## Summary

FACET addresses force-aware control for legged robots. The paper argues that standard velocity or position tracking produces stiff behavior under external forces, while an impedance-style command interface can make the same controller soft, robust to pushes, and capable of forceful interaction.

## Problem

How can a legged robot be controlled so that it remains useful for commanded
locomotion while also responding appropriately to external physical forces? The
core task is not only to move to or along a desired planar command, but to do so
with tunable interaction behavior: the same controller should tolerate large
pushes, reduce collision severity when compliant behavior is requested, and
generate meaningful sustained forces when forceful interaction is requested.

This benchmark treats the problem as a legged control task under contact-rich
disturbances. A candidate method receives the allowed robot observations and
high-level control commands, then outputs low-level actions for the simulated
robot. The environment measures whether the resulting behavior is stable,
trackable, force-aware, and controllable across command settings.

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.6-sol-xhigh` | +100.9 |
| `gpt-5.5-xhigh` | +100.8 |
| `gpt-5.6-terra-xhigh` | +100.6 |
| `gpt-5.6-luna-xhigh` | +97.7 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install sciworld
sw research 2505.06883 --mode innovate --agent claude-code
sw evaluate 2505.06883
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

