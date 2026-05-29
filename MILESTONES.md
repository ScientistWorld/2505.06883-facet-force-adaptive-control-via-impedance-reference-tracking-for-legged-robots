# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-05-29 08:57] - none
- Re-audited the previous reduced FACET implementation and found it used an analytic controller rather than the paper's PPO-trained impedance-tracking policy.
- Added a lightweight PPO teacher/student training path with temporally smoothed impedance-reference rewards; submitted a compute job is required before claiming `method_runs`.

### [2026-05-29 09:08] - core_claim
- Continuation starts from the protocol reset milestone `core_claim`.
- Audited the PPO replacement path and found the current committed scores still come from stale pre-PPO rollout artifacts, so a fresh compute job is being submitted before restoring any higher milestone.
- Tightened the benchmark before submission: evaluation now recomputes metrics from rollout records, wall scoring includes a contact-rate constraint, and PPO training samples stiffness/velocity commands.

### [2026-05-29 09:24] - core_claim_plus
- CPU job `24368869-42e` trained the reduced PPO teacher/student FACET policy end-to-end and regenerated all rollout artifacts.
- The PPO policy reached `1.0` impulse success while both velocity baselines remained at `0.0`, and low-stiffness wall contact reduced collision impulse by `0.901` with `1.0` contact rate.
- Added a follow-up stiffness-modulation experiment to test the secondary claim that the same policy changes contact impulse when the impedance stiffness command changes.
