# Research Direction

## Research Problem

This benchmark studies force-adaptive control for legged robots. The scientific
question is whether a learned controller can combine stable locomotion, tunable
compliance, and forceful interaction within one deployable policy, instead of
behaving like a fixed stiff velocity tracker or a purely passive disturbance
response.

## Core Contribution

The paper's contribution is a learning-based control formulation that exposes a
continuous interaction-control interface and trains the robot to match a
force-responsive task-space behavior under external perturbations. A scientist
should work within that contribution area: improving how the robot learns,
represents, estimates, or follows the intended force-adaptive behavior while
preserving the same kind of high-level interaction interface.

## Approach Scope

In-bounds work includes changes to the learned controller, estimator,
representation of command history, reward shaping, curriculum, training
schedule, disturbance sampling, policy architecture, adaptation mechanism, or
tracking target construction, as long as the result remains a single legged
controller for the benchmark's force-adaptive locomotion and interaction tasks.

Methods may improve robustness, tracking accuracy, compliance modulation, force
estimation, or sample efficiency, but they must still solve the same problem:
controllable physical interaction under the fixed benchmark interface, with a
learned controller as the deployed policy.

## Out of Bounds

- Replacing the task with standard velocity tracking, pure position tracking, or
  direct scripted force control that does not provide tunable interaction
  behavior.
- Using a hand-engineered model-predictive controller or trajectory optimizer as
  the deployed controller unless the learned policy remains the primary
  force-adaptive controller being evaluated.
- Changing the benchmark to a different robot, fixed-base manipulator, wheeled
  platform, or non-contact locomotion problem.
- Adding privileged deployment sensors or simulator-only state to avoid the
  estimation and interaction challenge.
- Beating the scores by hard-coding the evaluator scenarios, detecting wall or
  impulse tests directly, or specializing separate policies for each metric.
- Treating falling, sitting, freezing, or excessive yielding as success when the
  command-tracking and forceful-interaction constraints are violated.
