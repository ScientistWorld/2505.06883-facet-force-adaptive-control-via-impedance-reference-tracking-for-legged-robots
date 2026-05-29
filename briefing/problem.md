# Problem Definition

## Research Question

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

## Why It Matters

Legged robots that rigidly reject all disturbances can be unsafe around people
and unreliable during physical interaction with objects. A useful controller
should keep the robot balanced while allowing soft guidance, safer collisions,
and strong pulling or pushing when the task requires it.

## Success Criteria

A successful solution should:

- Maintain locomotion stability under large horizontal impulses.
- Reduce impact impulse in collision tests when configured for compliant
  behavior.
- Preserve command-following accuracy under randomized commands and external
  forces.
- Show that interaction behavior is controllable across command settings, rather
  than being fixed after training.
- Avoid solving the task by ignoring locomotion quality, falling safely, or
  exploiting simulator-specific shortcuts.
