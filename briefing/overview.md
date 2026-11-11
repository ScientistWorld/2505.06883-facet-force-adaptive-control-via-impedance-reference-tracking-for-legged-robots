# Overview

FACET addresses force-aware control for legged robots. The paper argues that
standard velocity or position tracking produces stiff behavior under external
forces, while an impedance-style command interface can make the same controller
soft, robust to pushes, and capable of forceful interaction.

The environment reproduces the core simulation claims at reduced scale on a
planar force-interaction benchmark: surviving large lateral impulses by
yielding, reducing collision impulse against a wall when compliance is
commanded, and following commanded interaction behaviour under random forces.
It provides the benchmark simulator, the reproduced FACET training pipeline,
fixed velocity-tracking baselines, and evaluators that roll out each submitted
policy themselves and score it on visible (train) and held-out (test)
scenarios.
