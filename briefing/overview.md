# Overview

FACET addresses force-aware control for legged robots. The paper argues that
standard velocity or position tracking produces stiff behavior under external
forces, while an impedance-style command interface can make the same controller
soft, robust to pushes, and capable of forceful interaction.

The reproduced environment focuses on the core simulation claims: surviving a
large lateral impulse and reducing collision impulse against a wall. It provides
a reduced-scale planar benchmark, FACET's smoothed impedance reference
implementation, velocity-tracking baselines, and reusable evaluation scripts
that score saved rollout artifacts.
