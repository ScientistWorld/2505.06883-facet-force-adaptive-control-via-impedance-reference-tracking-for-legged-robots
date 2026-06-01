# Evaluation

## Official Reduced-Gym Metrics

- **Perturbation success rate**: fraction of randomized trials in which the
  robot remains upright after a lateral ramp impulse while walking forward.
  Higher is better. Failure occurs when non-foot body parts contact the ground.
- **Final forward speed**: forward speed at the end of the perturbation
  episode. This is a constraint metric for the reduced benchmark: a controller
  should survive the impulse while still making locomotion progress.
- **Collision impulse**: time integral of the wall contact force during the
  first 0.5 seconds after a wall collision. Lower is better for compliant
  settings.
- **Collision impulse reduction**: relative reduction in collision impulse
  compared with a robust velocity-tracking controller under the same wall test.
  Higher is better.
- **Task success rate**: fraction of randomized command-and-force trials that
  avoid falling or invalid contacts. Higher is better.

The paper also reports maximum survived impulse over a sweep, tracking-error
ablations, and real-robot pulling capacity. Those are useful context but are
not official scored targets in this reduced gym unless a future agent adds
validated artifacts and matching entries in `scoring/scores.json`.

## Baselines and Targets

The main simulation baselines are:

- A standard velocity-tracking locomotion controller.
- A velocity-tracking controller trained with the same random impulse
  perturbation distribution used for the reproduced system.

The paper's headline targets are:

- Survival under substantially larger lateral disturbances than both velocity
  baselines, with reported robustness up to about 200 Ns impulse and trajectory
  examples at 400 N peak force.
- About 80% lower collision impulse than the robust velocity baseline in the
  wall-collision setting when compliant behavior is requested.
- On hardware, reliable pulling of a 10 kg payload, compared with reported
  baseline limits of 2.5 kg for the built-in controller and failure beyond 5 kg
  for the robust velocity controller.

Hardware numbers are useful context, but the reusable benchmark should rely on
simulation metrics unless a real robot is explicitly available.

## Evaluation Protocol

Evaluate all methods in the same simulator, robot model, action rate, episode
length, termination rules, command ranges, and domain randomization settings.
Do not tune a method on the final evaluation seeds.

For the perturbation benchmark, command forward locomotion at 1.5 m/s. Apply a
horizontal ramp impulse after the robot has reached stable locomotion, with the
force lasting 0.4 to 0.6 seconds. Report success rate over 64 randomized
parallel trials per disturbance level.

For the collision benchmark, command the robot into a soft virtual wall at
1.5 m/s. Integrate the wall contact force over the first 0.5 seconds after
contact and report the impulse distribution for each tested control setting.

For any future expanded tracking evaluation, sample high-level commands and
external forces from the benchmark distributions, then report errors and
success rate using the same evaluator for every submitted method.
