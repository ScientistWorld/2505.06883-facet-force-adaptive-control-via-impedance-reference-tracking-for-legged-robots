# Verification Report

## Initial Assessment

The workspace entered final verification at `core_claim`. The existing reduced
FACET reproduction was runnable and `python validate.py --compare` passed: the
method implements the paper's impedance-reference tracking idea at planar
benchmark scale, evaluation reads saved rollout artifacts rather than importing
`method/`, and train/test split evaluators were present.

The claimed milestone remains justified. The reproduction is not a full Isaac
Lab robot reproduction, but the reduced setup preserves the central algorithmic
computation and demonstrates the core claims on two scored force-interaction
experiments.

## Issues Found

- **Critical**: An unverified tracking-target ablation had been added to
  `reference.json`, `TARGETS.md`, and `EXPERIMENTS.md`, but no matching
  `scores.json` entry or saved artifact existed. This created a phantom target
  and would make the official gym score include an experiment that the current
  reproduction had not actually run.
- **Minor**: The default method script attempted to run the unverified ablation
  producer, so a fresh reproduction could emit an experiment no longer aligned
  with the official scoring schema.
- **Minor**: Generated job logs were tracked in git even though `jobs/` is
  ignored and not part of the reusable research environment.
- **Cosmetic**: `CHECKLIST.md` still marked `core_claim_plus` complete even
  though the current accepted milestone is `core_claim`.

## Fixes Applied

- Removed the unverified tracking-target ablation from `scoring/reference.json`,
  `scoring/TARGETS.md`, `scoring/EXPERIMENTS.md`, and the method-agnostic
  evaluation briefing.
- Removed the ablation producer from the default method path and pruned optional
  ablation scoring hooks from main/train/test evaluators.
- Deleted the unused ablation source file and tracked `jobs/` log artifacts.
- Updated `PROGRESS.md`, `MILESTONES.md`, and `CHECKLIST.md` so the documented
  state matches the verified `core_claim` milestone.
- Re-ran `scripts/evaluate.sh`, `scripts/evaluate_train.sh`,
  `scripts/evaluate_test.sh`, and `python validate.py --compare`; all checks
  pass.

## Final Judgment

The workspace is ready to serve as a research gym at `core_claim`. From-scratch
agents can use the method-agnostic problem definition, procedural data loader,
and artifact-based evaluators without seeing `method/`. Improve-mode agents can
inspect the reduced PPO/FACET implementation and compare against the saved
velocity-baseline scores.

The scoring now covers only experiments present in `scores.json`: large lateral
impulse recovery and compliant wall-collision impulse reduction. The 60/40
weight split is appropriate for the paper's core claim because disturbance
robustness is the primary benefit and compliant collision reduction is the main
safety constraint.

- Milestone: core_claim
- Ready for gym use: yes
- Confidence: medium
- Key limitation: reduced procedural planar simulation rather than full Isaac
  Lab robot experiments.

## Superseded by the hardening rounds

The scoring described above (saved rollout artifacts) was replaced in the
hardening iteration: the evaluators now roll out each submitted TorchScript
policy themselves, and the final round recalibrated the wall experiment, the
test size and the shipped reference seed. See `HARDENING.md` for the current
gym, its shortcut audit and its remaining limitations.
