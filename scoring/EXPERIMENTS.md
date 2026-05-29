# Experiment Overview

## large_lateral_impulse

This experiment checks whether a controller can keep moving after a large
sideways push. A useful result has a higher survival rate than the velocity
tracking baselines and keeps meaningful forward motion rather than simply
freezing in place.

## soft_wall_collision

This experiment checks the safety side of the claim. The controller is driven
into a soft wall, and the evaluator measures the integrated contact impulse. A
useful result lowers the impulse compared with the robust velocity baseline,
showing that robustness was not obtained by becoming rigid.
