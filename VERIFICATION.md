# Verification Report

## Initial Assessment

The workspace already had a validated reduced FACET reproduction at
`core_claim`: the method implements the paper's impedance-reference tracking
algorithm at planar benchmark scale, the main evaluator reads saved rollout
artifacts, and `python validate.py --compare` passed. The claimed milestone is
justified, but I found a scoring-documentation mismatch that needed cleanup.

## Issues Found

- **Minor**: `reference.json` assigned a nonzero coefficient to
  `mean_wall_entry_speed_mps`, but the main `scripts/evaluate.sh` path did not
  emit that metric into `scores.json`.
- **Minor**: several reduced-benchmark reference values were not direct paper
  measurements and lacked explicit notes explaining the scaling/calibration.
- **Cosmetic**: tracked trajectory logs remained in the repository even though
  trajectory output is ignored and not part of the reusable gym.

## Fixes Applied

- Updated `eval/evaluate_rollouts.py` so the main evaluator computes and writes
  `mean_wall_entry_speed_mps` for the wall-collision experiment.
- Regenerated `scoring/scores.json` from existing rollout artifacts with
  `scripts/evaluate.sh`.
- Added metric descriptions and experiment notes to `scoring/reference.json` to
  distinguish direct paper claims from reduced-benchmark calibrated quantities.
- Updated `scoring/TARGETS.md` and `scoring/EXPERIMENTS.md` to document the
  wall entry-speed and post-contact impulse metrics without leaking method
  details.
- Removed obsolete tracked `trajectory/*.jsonl` logs from version control.

## Final Judgment

The workspace is ready to serve as a research gym at `core_claim`. From-scratch
agents can use the method-agnostic problem, data, and evaluation files without
depending on `method/`; improve-mode agents can inspect the reduced PPO
implementation and compare against the saved baseline scores. The main
limitation is that the full Isaac Lab robot setup is replaced by a procedural
planar force-interaction benchmark, so higher milestones would require a more
complete simulator reproduction or additional validated secondary claims.

- Milestone: core_claim
- Ready for gym use: yes
- Confidence: medium
- Key limitation: reduced procedural planar simulation rather than full robot
  Isaac Lab experiments.
