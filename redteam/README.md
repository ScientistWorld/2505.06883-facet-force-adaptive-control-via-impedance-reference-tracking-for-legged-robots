This audit keeps candidate implementations, per-candidate hidden-test results,
and an auditor-only normalization calculation here. Each candidate has a
`run.sh` which installs its on-disk submission and invokes the original
`bash scripts/evaluate_test.sh` with **no arguments**, in a separate process.
No evaluator, interpreter, installed package, or surviving process is modified.

`audit.py batch` runs the complete bundle and restores the original method,
scores, and checkpoint directory afterwards. Training and simulation must run
as scheduler jobs. Large generated artifacts and logs stay local and ignored;
the source and small result JSONs are committed. `analyze.py` alone reads the
auditor's reference specification and test scores; candidate preparation and
controller selection never read either. Selection for the brute force shortcut
uses only the visible train evaluator.

`reference_stored` verifies the supplied saved rollout tapes. The three
`reference_seed*` cases fully retrain with the original 60/30 updates, 128
environments and 96 rollout steps, and hold evaluation seeds fixed at 7/8.
`better_extended_training` triples training for a diagnostic explicitly outside
the gym's fixed training budget. `better_reward_fix` uses the original budget
and corrects the mean-reward-versus-mean-state discrepancy with paper Eq. 7,
and uses each sampled command in the auxiliary forward-tracking reward.
`weaker_untrained_shortcut` keeps the architecture but performs zero updates.

Ranks in `summary.json` use the actual stored test reference, absolute
coefficients, the declared higher/lower directions, experiment weights, and
population standard deviations over two explicitly listed audit cohorts.
Production rank magnitudes depend on future submitted attempts; dominance and
exact ties are reported separately from those illustrative magnitudes.
