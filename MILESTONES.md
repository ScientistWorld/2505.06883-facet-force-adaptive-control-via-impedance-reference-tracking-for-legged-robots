# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-05-29 08:44] - method_runs
- Implemented the FACET impedance reference model with temporally smoothed targets at lags 8, 16, and 32 control steps.
- Implemented a reduced planar force-interaction environment, velocity-tracking baselines, artifact-based evaluation, and an end-to-end reproduction script.
- Verified the method and baselines execute end-to-end and write `scoring/scores.json`.

### [2026-05-29 08:45] - core_claim
- Reproduced the reduced-scale central claim: the FACET controller survives the large lateral impulse setting while velocity-tracking baselines fail.
- Reproduced the safety constraint: low-stiffness FACET substantially reduces wall collision impulse relative to the robust velocity baseline, with a higher-stiffness FACET setting producing larger impulse as expected.

### [2026-05-29 08:47] - core_claim
- Audited the continuation workspace from a reset `method_runs` state.
- Fixed the official impulse metric by replacing unbounded lateral displacement with final forward speed as the locomotion constraint.
- Regenerated `scoring/scores.json` from existing rollout artifacts and verified `scripts/evaluate.sh` plus `python validate.py --compare` pass locally.

### [2026-05-29 08:48] - core_claim_plus
- CPU container job `8a1779a2-03f` reran the packaged reproduction and passed `python validate.py --compare`.
- Verified the corrected scoring contract inside the container: FACET survived the large impulse trials while both velocity baselines failed.
- Verified the additional wall-compliance setting: low-stiffness FACET reduced collision impulse versus the robust velocity baseline, and high-stiffness FACET produced a larger impulse as expected.
