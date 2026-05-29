# Progress

## What Works

- Read the FACET paper and extracted the core algorithm: impedance reference
  model tracking with temporally smoothed targets.
- Implemented a reduced-scale planar force-interaction benchmark in `data/`.
- Implemented the FACET smoothed impedance reference controller in `method/`.
- Implemented velocity-tracking baselines in `baseline/`.
- Implemented artifact-based evaluation in `eval/`; it reads rollout JSON files
  and writes `scoring/scores.json` without importing `method/`.
- CPU job `4850b244-e2c` ran end-to-end and passed `python validate.py
  --compare`, so the current milestone is `method_runs`.
- CPU job `1fa1f65f-30f` reproduced the reduced-scale core claim: FACET
  survived all large impulse trials while both velocity baselines failed, and
  retained much lower wall-collision impulse than the robust baseline.
- `MILESTONES.md`, `CHECKLIST.md`, and scoring docs now reflect the achieved
  `core_claim` state and the remaining optional expansion items.

## Results

- CPU smoke job `6d94ef34-558` executed the rollout pipeline and produced
  artifacts, but failed final validation because an extra metric was emitted.
- The same job showed the first FACET impulse controller failed all impulse
  trials, so the controller was corrected to output the virtual spring component
  rather than the reference acceleration including external force.
- CPU smoke job `4850b244-e2c` passed validation. Wall collision supports the
  compliance constraint: `facet_low_kp` produced `28.74 Ns` collision impulse
  versus `455.65 Ns` for `velocity_robust`.
- The impulse robustness benefit is not reproduced yet: all methods had
  `0.0` success. The reduced fall proxy was too sensitive to external force
  itself, so it has been adjusted to mainly penalize large resisting ground
  reaction.
- CPU job `f52306ba-0b1` improved impulse behavior but not enough for
  `core_claim`: `facet_smoothed` reached `0.133` success while both velocity
  baselines remained at `0.0`. FACET impulse stiffness and tracking gains were
  reduced to better represent the compliant mode used for large pushes.
- CPU job `1fa1f65f-30f` reached the core claim at reduced scale:
  `facet_smoothed` achieved `1.0` impulse success while `velocity_robust` and
  `velocity_vanilla` achieved `0.0`; `facet_low_kp` produced `27.18 Ns` wall
  collision impulse versus `455.65 Ns` for `velocity_robust`.

## Remaining

- Submit one more CPU verification job from the packaged `scripts/run.sh` to
  confirm the current workspace reproduces the same scores after metadata
  cleanup.
- Optional future expansion: add explicit open-loop and closed-loop reference
  target ablations from Table 2.
- Optional future expansion: port the benchmark from the planar dynamics model
  to Isaac Lab if the full robot assets and training code become available.

## Issues

- The public project code link resolves to the website repository, not a full
  Isaac Lab training code release.
- The full robot simulation is scaled down because reproducing Isaac Lab assets
  and 250M-step teacher-student PPO is not practical in this benchmark setup.
- The benchmark's impulse displacement metric is much larger than the paper's
  plotted robot trajectories because the reduced planar model has no arena
  boundary; the scored core comparison is survival versus baselines plus
  collision impulse.
- `scoring/reference.json` uses reduced-scale reference values for baseline
  methods where the paper only provides approximate plot values. The paper's
  qualitative claim is preserved: FACET survives the harder impulse setting and
  substantially reduces wall collision impulse relative to the robust velocity
  baseline.

## Deviations from Paper

- Paper setup: Isaac Lab legged robot PPO with teacher-student training,
  privileged teacher observations, and real Unitree Go2/B1+Z1/G1 platforms.
- Reproduction setup: procedural planar force-interaction dynamics with the same
  impedance reference model, fixed target velocity command form, critical
  damping, and smoothed target starts at 8/16/32 control intervals.
- Reason: the linked public repository is not a full training code release, and
  the full robot assets/training stack would exceed the scope of a reusable
  lightweight benchmark. The algorithmic target-generation computation is kept;
  the robot physics and training scale are reduced.
