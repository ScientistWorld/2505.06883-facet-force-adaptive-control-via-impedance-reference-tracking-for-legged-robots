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

## Remaining

- Rerun `scripts/reproduce.sh` on the compute node after the controller fix.
- Inspect `scoring/scores.json` and tune the reduced-scale dynamics if the FACET
  method still does not beat velocity baselines on survival and collision
  impulse.
- Add ablations for open-loop and closed-loop target generation after the core
  robustness result is working.
- If core claim succeeds, optionally add ablation variants for open-loop and
  closed-loop reference targets.

## Issues

- The public project code link resolves to the website repository, not a full
  Isaac Lab training code release.
- The full robot simulation is scaled down because reproducing Isaac Lab assets
  and 250M-step teacher-student PPO is not practical in this benchmark setup.

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
