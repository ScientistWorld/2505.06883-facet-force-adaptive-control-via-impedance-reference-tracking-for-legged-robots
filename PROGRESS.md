# Progress

## What Works

- Read the FACET paper and extracted the core algorithm: impedance reference
  model tracking with temporally smoothed targets.
- Implemented a reduced-scale planar force-interaction benchmark in `data/`.
- Implemented the FACET smoothed impedance reference controller in `method/`.
- Implemented velocity-tracking baselines in `baseline/`.
- Implemented artifact-based evaluation in `eval/`; it reads rollout JSON files
  and writes `scoring/scores.json` without importing `method/`.

## Results

- Pending CPU smoke job. Local validation of `scoring/reference.json` and import
  separation passes, but rollout execution requires the container's numpy
  install.

## Remaining

- Run `scripts/reproduce.sh` on the compute node.
- Inspect `scoring/scores.json` and tune the reduced-scale dynamics if the FACET
  method does not beat velocity baselines on survival and collision impulse.
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
