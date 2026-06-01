# Experiment Overview

## large_lateral_impulse

This experiment checks whether a controller can keep moving after a large
sideways push. A useful result has a higher survival rate than the velocity
tracking baselines and keeps meaningful forward motion rather than simply
freezing in place.

## soft_wall_collision

This experiment checks the safety side of the claim. The controller is driven
into a soft wall, and the evaluator measures the reduction in integrated
contact impulse relative to the robust velocity baseline. The contact-rate
and entry-speed constraints prevent a method from improving this metric by
simply stopping before it reaches the wall or approaching it much more slowly.

## tracking_target_ablation

This experiment checks a secondary training-target claim from the paper. It
compares several ways of defining the reference target and reports normalized
tracking errors plus a stability constraint. A useful result should show that
the target choice changes tracking behavior in the same qualitative direction
as the paper's ablation, rather than only improving the main force-interaction
score.
