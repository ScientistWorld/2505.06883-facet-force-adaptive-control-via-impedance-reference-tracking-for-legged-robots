# Evaluation Targets

## large_lateral_impulse

Tests whether the controller remains balanced when a forward-moving body is
struck by a large lateral ramp impulse.

- **Metric**: `success_rate` (higher is better)
- **Metric**: `mean_final_forward_speed_mps` (higher is better as a locomotion
  constraint; the controller should survive without simply giving up forward
  motion)
- **How to run evaluation**: run `scripts/reproduce.sh` or generate rollout
  artifacts with `scripts/method.sh` and `scripts/baseline.sh`, then run
  `scripts/evaluate.sh`.

## soft_wall_collision

Tests whether improved disturbance handling is achieved while preserving the
paper's safety constraint: lower collision impulse under compliant settings.

- **Metric**: `collision_impulse_reduction` (higher is better, relative to the
  robust velocity baseline)
- **Metric**: `wall_contact_rate` (higher is better as a task-validity
  constraint; the controller must actually reach/contact the wall rather than
  lowering impulse by stopping early)
- **How to run evaluation**: run `scripts/reproduce.sh` or generate rollout
  artifacts with `scripts/method.sh` and `scripts/baseline.sh`, then run
  `scripts/evaluate.sh`.
