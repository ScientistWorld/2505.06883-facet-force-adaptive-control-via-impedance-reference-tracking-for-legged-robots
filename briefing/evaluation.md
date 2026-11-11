# Evaluation

## How a method is scored

A method is one deployable controller exported as a TorchScript policy at
`checkpoints/<method>/policy.pt` (interface in `data/facet_planar_env.py`:
16-dimensional observation of the command, body velocity, tilt, gait phase and
previous actions; 2-dimensional planar acceleration action). The evaluators
roll every policy on disk out themselves, in their own frozen copy of the
simulator, on scenarios generated from their own seeds, and compute every
metric from those rollouts. Files a method writes besides `policy.pt` are
ignored. The comparison side (velocity-tracking baselines, commands, wall,
force sweeps, episode counts, thresholds) is fixed inside the evaluator.

- `scripts/evaluate_train.sh` - visible scenarios, `scoring/scores_train.json`.
- `scripts/evaluate_test.sh` - held-out scenarios (same protocol, hidden
  seeds), `scoring/scores_test.json`. Both run with no arguments.

A policy that is missing, cannot be loaded, raises, returns a wrong shape or a
non-finite action, or yields a metric outside its valid range is a failed run
and receives the worst value of every metric in every experiment.

## Experiments and metrics

- **large_lateral_impulse** (weight 0.4). Walking forward at 1.5 m/s, a
  lateral ramp push of 0.4-0.6 s with peak force swept over 100-1200 N
  (64 trials per level). `impulse_recovery_score` (primary): survival times
  exp(-RMS velocity error / 0.5 m/s) against the commanded response from push
  onset; `success_rate`; `velocity_tracking_rmse_mps`.
- **soft_wall_collision** (weight 0.3). Walking into a soft virtual wall at
  1.5 m/s under a compliant command (low stiffness, light virtual inertia).
  `post_contact_impulse_reduction` (primary): reduction of the wall impulse in
  the first 0.5 s after contact relative to the robust velocity baseline on the
  same episode, where only realising the commanded compliance lowers the
  charged impulse (braking or backing off is charged like being stiff);
  `wall_contact_rate` and `entry_speed_error_mps` (constraints: the wall must be
  reached at the commanded speed without a fall or retreat);
  `contact_tracking_rmse_mps` (how closely the contact response follows the
  commanded behaviour).
- **impedance_tracking** (weight 0.3). Random commands (stiffness, damping,
  virtual inertia, target velocity) under random steady forces and pushes.
  `velocity_tracking_rmse_mps` (primary), `position_tracking_rmse_m`,
  `success_rate`.

Tracking errors compare the body's state with the commanded mass-spring-damper
response integrated over the preceding 0.16, 0.32 and 0.64 s from the body's
own state under the same external forces (the paper's Sec. 10.3 tracking
error); states after a fall count as the error caps.

## Ranking

A method's rank is the sum over all scored metrics of experiment weight x
|coefficient| x its improvement over the reference row in
`scoring/reference.json` (taken in each metric's better direction), each
divided by the metric's spread across the ranked attempts.

## Baselines and paper targets

The baselines are fixed-gain velocity-tracking controllers: `velocity_vanilla`
(stiff in both axes) and `velocity_robust` (softer lateral gain, the
comparison for the wall impulse). The paper's simulation results (Go2
quadruped in Isaac Lab) are: survival under much larger lateral impulses than
both velocity baselines (up to about 200 N s), about 80% lower collision
impulse than the robust baseline when compliance is commanded, and lower
tracking errors for its full method than for its ablations (Table 2). They are
recorded under `_paper_reported` in `scoring/reference.json`; this benchmark is
a reduced planar version and its numbers are on its own scale. Hardware
results (payload pulling) are context only.

## Protocol rules

Evaluate every method with the same simulator, action rate, episode lengths,
termination rules, command ranges and domain randomisation. Do not tune on the
held-out scenarios; the train evaluator exists for development.
