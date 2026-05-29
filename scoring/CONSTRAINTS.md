# Scientist Constraints

## Robot and Simulator

Use the provided legged-robot environments, robot assets, actuator model, action
rate, contact model, and termination rules. The paper's claims are about control
under physical interaction, so changing the robot morphology, simulator physics,
wall model, or fall definition would change the benchmark rather than improve
the method.

## Observations and Commands

Use only the observation channels exposed by the benchmark for the policy at
deployment time. Do not add external force sensors, privileged simulator state,
future trajectory information, wall penetration state, or termination labels to
the deployed policy input. The high-level command interface must remain the one
provided by the environment so that compliance and forceful behavior are tested
under the same task formulation.

## Training Data and Randomization

Use only data generated inside the provided simulation environments plus the
paper-specified robot assets. Do not import motion-capture datasets, offline
robot logs, additional hardware demonstrations, external terrain/contact
datasets, or pretrained controllers. Keep the benchmark command ranges,
disturbance distributions, domain randomization families, and evaluation seeds
fixed when reporting scores.

## Baselines

Keep the provided velocity-tracking and robust velocity-tracking baselines
unchanged when computing relative metrics. A scientist may add new comparison
methods, but the official scores must still include the fixed baselines so that
collision reduction, robustness gain, and payload capacity remain comparable.

## Compute and Scale

Do not increase the robot count per experiment, number of evaluation trials, or
evaluation horizon beyond the benchmark settings when producing official scores.
Training may be made more sample-efficient or better optimized, but improvements
must not come from using a larger simulator batch, longer training budget, or
ensembling multiple independently trained policies at evaluation time.

## Hardware Claims

Only report hardware payload or compliant-following scores if the corresponding
real-robot procedure is actually run. Simulation-only reproductions must label
payload tests as simulated or equivalent opposing-force tests and must not claim
real-world deployment.

## Compliance Checklist

- [ ] Uses the provided robot assets, simulator, action rate, contact model, and
  termination rules.
- [ ] Deployed policy receives no privileged simulator state, external force
  sensor readings, future information, or evaluator-only signals.
- [ ] Official scores use the fixed command ranges, disturbance distributions,
  domain randomization families, and evaluation seeds.
- [ ] No external demonstrations, robot logs, motion-capture data, pretrained
  controllers, or additional contact datasets are used.
- [ ] Velocity-tracking and robust velocity-tracking baselines are unchanged for
  relative metrics.
- [ ] Official results are from a single deployed policy per method, not an
  ensemble or hand-picked checkpoint collection.
- [ ] Hardware results are only claimed when the hardware procedure was run.
