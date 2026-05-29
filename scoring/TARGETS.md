# Evaluation Targets

## large_lateral_impulse

Tests whether the controller remains balanced when a forward-moving body is
struck by a large lateral ramp impulse.

- **Metric**: `success_rate` (higher is better)
- **Metric**: `mean_abs_lateral_displacement_m` (higher is better for this
  benchmark because compliant survival requires moving with the push instead of
  rigidly rejecting it)
- **How to run evaluation**: run `scripts/reproduce.sh` or generate rollout
  artifacts with `scripts/method.sh` and `scripts/baseline.sh`, then run
  `scripts/evaluate.sh`.

## soft_wall_collision

Tests whether improved disturbance handling is achieved while preserving the
paper's safety constraint: lower collision impulse under compliant settings.

- **Metric**: `mean_collision_impulse_Ns` (lower is better)
- **How to run evaluation**: run `scripts/reproduce.sh` or generate rollout
  artifacts with `scripts/method.sh` and `scripts/baseline.sh`, then run
  `scripts/evaluate.sh`.

