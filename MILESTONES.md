# Reproduction Milestones

**Current: method_runs**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

### [2026-05-29 08:40] - none
- Read the paper and identified the central FACET computation.
- Confirmed the public code link is a website repository rather than a complete
  training release.
- Implemented reduced-scale method, baselines, and evaluator; execution is
  pending compute-node smoke test.

### [2026-05-29 08:55] - none
- CPU smoke job executed and wrote rollout artifacts, but validation failed due
  to an extra evaluator metric.
- Diagnosed failed FACET impulse behavior as an incorrect reduced controller
  action that included the external-force term; patched it to reproduce only the
  virtual spring component.

### [2026-05-29 09:05] - method_runs
- CPU job `4850b244-e2c` ran the method, baselines, evaluator, and comparison
  successfully.
- The wall-collision compliance constraint is reproduced at reduced scale, but
  the impulse robustness metric still fails, so `core_claim` is not claimed.
- Patched the reduced fall proxy to penalize excessive resisting reaction more
  than passive exposure to external force.

## Stop Justification
- Not stopped; next action is a CPU smoke job.
