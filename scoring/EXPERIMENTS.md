# Experiment Overview

## large_lateral_impulse

This experiment checks whether a controller can keep moving after a large
sideways push. A useful result has a higher survival rate than the velocity
tracking baselines and keeps meaningful forward motion rather than simply
freezing in place.

## soft_wall_collision

This experiment checks the safety side of the claim. The controller is driven
into a soft wall, and the evaluator measures the reduction in integrated
contact impulse relative to the robust velocity baseline. A useful result lowers
the impulse while still preserving locomotion robustness.

## ablation_tracking_targets

This paper-reported reference target records the Table 2 tracking-target
ablation. The current reduced benchmark does not run this experiment yet, but it
defines the next expansion target for future agents.
