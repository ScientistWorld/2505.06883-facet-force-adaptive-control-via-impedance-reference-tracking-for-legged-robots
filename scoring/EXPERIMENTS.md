# Experiment Overview

Every submitted method is a single deployable controller (a TorchScript policy,
see `data/facet_planar_env.py` for the interface). The evaluator rolls it out
itself in its own copy of the simulator, on scenarios it generates from its own
seeds, next to its own velocity-tracking baselines, and recomputes every number
from those rollouts. A controller that is missing, fails to load, crashes,
returns a wrong shape or a non-finite action, or produces an out-of-range
metric receives the worst value of every metric in every experiment.

The three experiments share one question: does the controller behave like the
interaction behaviour it is commanded to have (stiffness, damping, virtual
inertia, target velocity) while external forces act on it, without falling?

## large_lateral_impulse

A controller walking forward at 1.5 m/s is struck by a large sideways ramp
push, swept from moderate to far beyond the forces seen in training. The
experiment establishes whether the controller survives large pushes by
yielding in the commanded way rather than resisting stiffly, and whether it
still follows the commanded behaviour afterwards. Survival alone is not
enough: an episode is credited by how closely the body's velocity follows the
commanded response from the push onward, so freezing, drifting or
overspeeding to stay upright scores poorly. Velocity-tracking baselines fail
at most push levels.

## soft_wall_collision

The controller walks into a soft wall at 1.5 m/s while commanded to be
compliant (low stiffness, light virtual inertia). The experiment establishes
the safety side of the claim: the collision impulse in the first 0.5 s after
contact should be much lower than that of the robust velocity-tracking
baseline on the identical episode. Impulse is only credited for realising the
commanded compliance: being softer than commanded (braking, backing off) is
charged like being stiffer, collisions must happen at the commanded approach
speed without a fall, and the contact response itself is scored for how
closely it follows the commanded behaviour. A body that does nothing special
on contact gets only a small reduction; reaching the commanded reduction
requires reacting to the contact force.

## impedance_tracking

Randomised commands (stiffness, critical damping, virtual inertia, target
velocity) under random steady forces and pushes in both planar axes. The
experiment establishes how faithfully the controller realises the commanded
interaction behaviour across the whole command range, and how often it falls,
which is what lets the same controller be soft in one setting and firm in
another.
