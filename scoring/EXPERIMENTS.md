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
constraint prevents a method from improving this metric by simply stopping
before it reaches the wall. The impulse window follows the paper's 0.5 s
post-contact protocol.

## impedance_stiffness_modulation

This experiment checks whether changing the virtual stiffness command changes
the resulting contact impulse for the same learned controller. A useful result
has lower normalized wall impulse under the low-stiffness command while still
making contact with the wall. The impulse window follows the same 0.5 s
post-contact protocol as the soft-wall experiment.
