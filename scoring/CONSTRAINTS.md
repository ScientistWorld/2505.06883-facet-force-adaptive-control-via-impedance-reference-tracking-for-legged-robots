# Scientist Constraints

## Simulator and Interface

Use the provided planar force-interaction simulator (`data/facet_planar_env.py`)
with its dynamics, actuator model, action limit, control interval, wall model
and fall rule. The evaluators keep their own frozen copy of these dynamics, so
editing `data/` changes only your training, never the score. A submission is a
single TorchScript policy at `checkpoints/<method>/policy.pt` that maps the
16-dimensional observation to the 2-dimensional action; it may keep internal
state (for example a recurrent estimator) and may export a `reset(n)` method.

## Observations and Commands

The deployed policy may use only the observation the evaluator gives it. Do
not smuggle external forces, wall penetration, scenario parameters, future
information or termination labels into the policy by any other route, and do
not detect the evaluator's test scenarios (for example by their exact command
values) to switch to scenario-specific behaviour. The command interface
(target velocity, stiffness, damping, virtual mass) stays the one provided.

## Training Data and Randomization

Use only data generated inside the provided simulator. Do not import
demonstrations, logs or pretrained controllers. Privileged simulator
information may be used during training (as teachers, critics or estimator
targets), not by the deployed policy.

## Baselines

The velocity-tracking baselines and every comparison quantity are owned by the
evaluator and cannot be changed by a submission.

## Compute and Scale

The reference was trained with 600 teacher + 150 student PPO updates of
256 environments x 96 steps (about 18.4M environment steps). Improvements
should come from the method, not from a larger training budget, larger
simulator batch, or ensembles of independently trained policies; report the
environment steps you used.

## Compliance Checklist

- [ ] Uses the provided simulator, action limit, control interval, wall model
  and fall rule for training and submits a single TorchScript policy.
- [ ] The deployed policy receives nothing but the evaluator's observation and
  does not detect test scenarios.
- [ ] No external demonstrations, logs or pretrained controllers.
- [ ] Training budget no larger than the reference's (about 18.4M env steps).
- [ ] The deployed controller is learned; a hand-engineered model-based
  controller or trajectory optimiser is not a valid submission (see
  `DIRECTION.md`).
