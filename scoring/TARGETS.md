# Evaluation Targets

All three experiments are scored by `scripts/evaluate_train.sh` (visible
scenarios, `scoring/scores_train.json`) and the held-out
`scripts/evaluate_test.sh` (`scoring/scores_test.json`). Both run with no
arguments and score every policy at `checkpoints/<method>/policy.pt` together
with the evaluator-owned baselines `velocity_vanilla` and `velocity_robust`.
`scripts/evaluate.sh` writes the train-slice scores to `scoring/scores.json`.
Experiment weights and metric coefficients are in `scoring/reference.json`.

## large_lateral_impulse (weight 0.4)

Forward walking at 1.5 m/s, lateral ramp pushes of 100-1200 N peak lasting
0.4-0.6 s, 64 randomised episodes per force level (512 episodes).

- **Primary**: `impulse_recovery_score` (higher is better): mean over episodes
  of survival x exp(-velocity tracking RMSE / 0.5 m/s), from push onset.
- `success_rate` (higher is better): fraction of episodes without a fall.
- `velocity_tracking_rmse_mps` (lower is better): RMS error between the body's
  velocity and the commanded response from push onset; falls count as 5 m/s.

## soft_wall_collision (weight 0.3)

Walking into a soft wall (260 N/m, 28 N s/m) at 1.5 m/s with the compliant
command (stiffness 0.5, critical damping, virtual mass 6 kg); 256 randomised
episodes, each paired with the robust velocity baseline on the identical episode.

- **Primary**: `post_contact_impulse_reduction` (higher is better): 1 - charged
  0.5 s post-contact wall impulse / robust baseline impulse. Charged impulse is
  commanded + |measured - commanded|; invalid collisions are charged twice the
  baseline. Its best attainable value is the commanded compliance's own
  reduction (about 0.6).
- `wall_contact_rate` (higher is better, constraint): fraction of valid
  collisions (contact in time, entry speed 1.5 +- 0.3 m/s, no fall, no retreat
  behind the commanded response).
- `entry_speed_error_mps` (lower is better, constraint): |entry speed - 1.5|.
- `contact_tracking_rmse_mps` (lower is better): RMS velocity error versus the
  commanded response during the first 1 s after contact.

## impedance_tracking (weight 0.3)

Randomised stiffness (0.5-32, critical damping), virtual mass (6/12/24 kg),
target velocity and initial velocity, steady forces up to 40 N per axis and
ramp pushes of 80-200 N per axis; 512 randomised 7 s episodes.

- **Primary**: `velocity_tracking_rmse_mps` (lower is better).
- `position_tracking_rmse_m` (lower is better).
- `success_rate` (higher is better).

Errors are measured against the commanded mass-spring-damper integrated over
the preceding 0.16/0.32/0.64 s from the body's own state under the same
external forces; states after a fall count as the caps (5 m/s, 2.5 m).
