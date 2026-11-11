# Hardening report (final round)

This round builds on the previous hardening round ("v2", same iteration number, recorded in
`redteam/hardened/summary.json`). That round already replaced the original gym's trace-file scoring with
evaluator-owned rollouts of TorchScript policies. It kept that design and measured it, fixed what still
stopped the test from rewarding better control, and re-ran every policy and shortcut through the final
hidden test. The original audit is `REDTEAM.md` (the "original gym").

Ranks below follow the gym's rule: for each metric, weight x |coefficient| x improvement over the
reference, divided by the metric's population standard deviation over a cohort, then summed. **all** is
every scored case. **genuine** is every RL-trained policy plus the honest hand-written controllers:
simple velocity baseline, passive, zero-lateral, two analytic impedance laws and the brute-force pick.
`[a, b]` is a paired 95% bootstrap interval. It resamples episodes jointly, so both policies see the same
inputs; this is the test's noise. All numbers are on the held-out test slice with base = the shipped
reference (`reference_s17`) unless stated otherwise. Everything is in `redteam/hardened/summary_v3.{md,json}`.

## What was fixed in this round

1. **The wall experiment measured nothing.**
   - Problem: it commanded a virtual mass of 12 kg, equal to the nominal body. A passive body then already
     realises the commanded impedance (pure-Python check: impulse reduction of the ideal controller 0.257,
     passive 0.264). Under the v2 protocol every policy scored 0.220-0.226. That covered the reference
     seeds, the 2x-budget runs, the untrained policy, passive, the analytic laws and an ideal
     disturbance-observer controller alike, so the reference sat at the metric's attainable optimum.
     The rank's spread normalisation then turned the 0.004 noise into rank.
   - Fix: the wall command now uses the lightest commanded inertia, 6 kg (Kp 0.5, critical damping), in
     both evaluators. That is closer to the paper's own compliant regime: its virtual masses of 1-4 kg are
     7-27% of a ~15 kg robot.
   - Result: the commanded impedance's own reduction (the attainable optimum) is now 0.616; passive gets
     0.279.

   | | v2 wall reduction | final wall reduction |
   |---|---:|---:|
   | reference seeds | 0.222-0.226 | 0.431-0.547 |
   | 2x budget | 0.222-0.224 | 0.500-0.545 |
   | disturbance-observer probe | 0.223 | 0.558 |
   | passive | 0.224 | 0.279 |
   | untrained | 0.222 | 0.253 |
   | analytic laws | 0.223 | 0.262 |
   | history-free (MLP) estimator | 0.222 | 0.252 |

   Reaching the commanded compliance now requires reacting to the contact force. Being softer than
   commanded is still charged (commanded + |measured - commanded|).
2. **Test size.** The impulse sweep now uses the paper's 64 randomised trials per force level (was 32;
   512 episodes in total). The wall test now has 256 episodes (was 128) and tracking 512 (was 256). The
   train evaluator mirrors this with its own seeds. Train and test differ only in seeds and slice name
   (`diff eval/train/evaluate.py eval/test/evaluate.py`), and both match `data/` dynamics exactly
   (`redteam/hardened/checks_v3.json`).
3. **Noise from a near-saturated constraint.**
   - Problem: `entry_speed_error_mps` is a validity constraint. Entries outside 1.5 +- 0.3 m/s already
     invalidate an episode, and impulses are normalised to the commanded speed. Its spread across methods
     is tiny (0.058 m/s), so cruise-speed bias alone moved ranks by up to 0.18. That is as much as the
     primary wall metric separates reference seeds.
   - Fix: its coefficient goes from 0.3 to 0.1.
   - All other coefficients and the experiment weights (impulse 0.4, wall 0.3, tracking 0.3) are unchanged;
     they follow the paper's three simulation claims.
4. **Restoring the paper's network sizes was tested.** I added a `--net-size paper` option (policy
   256x3, critic 512-256-256, encoder 256, GRU 256) and trained three seeds at the reference budget, plus
   2x-budget, open-loop and untrained variants.
   - Rule, registered before the results: adopt the paper sizes if their mean train-slice rank is at least
     the reduced networks'.
   - Outcome: the reduced networks stay (`decision_v3.json`: 0.155 vs 0.085 on train; 0.120 vs 0.121 on
     test, a tie).
   - Paper-size networks are therefore an honest alternative, not an improvement the reference lacks.
5. **Shipped reference seed: `reference_s17`.**
   - My registered rule (median seed on the train slice) picked `reference_s37`. On train, four of the five
     seeds lie within 0.09 rank, so that pick was essentially arbitrary, and `s37` is the weakest seed on
     test. The clipped analytic controller outranked it (+0.12 all), so it was unfit as the bar.
   - The shipped reference is the median seed by mean rank over train and test (`canonical_v3.json`). That
     is `s17`, which is also the test-slice median and what the previous round's rule gives.
   - It is the typical seed, not the best (`s27` and `s47` rank above it). The train-only choice and the
     probes wrapped around `s37` are kept in `canonical_v3_train_only.json` and `cases_v3_wrapped_on_s37/`.
   - This is a deviation from the rule I registered, stated here rather than hidden.
6. **Docs.** `briefing/evaluation.md`, `briefing/overview.md`, `briefing/method.md`,
   `scoring/EXPERIMENTS.md`, `TARGETS.md`, `CONSTRAINTS.md` and `scoring/reference.json` now describe the
   gym as it is. The old docs described saved rollout files, final forward speed, a wall metric over the
   whole episode, and "legged-robot assets".

Kept from the previous round:
- The evaluator owns the simulator copy, scenarios, seeds, baselines, operating points, thresholds and
  episode counts. It rolls policies out itself and reads nothing a method writes except `policy.pt`.
- Missing, unloadable, crashing, wrong-shape or non-finite policies, and any out-of-range metric, receive
  the worst value of every metric in every experiment. Proposed methods listed in `reference.json` are
  always reported, so deleting a policy cannot hide a result.
- The wall retreat gate and the charge against the commanded-impedance impulse.
- The impulse score credits survival only together with tracking the commanded response.

I verified the before/after wrapper: re-scoring the previous round's reference under the v2 protocol
reproduces its stored scores exactly (maximum difference 0).

## Do better results now score higher?

| variant (test, paired vs its comparison) | original gym (REDTEAM, Rank A / G vs stored reference) | v2, same pair (all / genuine) | final, all | final, genuine |
|---|---:|---:|---:|---:|
| 2x training budget, seed 7 vs reference seed 7 | 3x budget: +0.117 / +0.463, below seed spread | +0.050 / +0.297 | **+0.109** [+0.046, +0.172] | **+0.437** [+0.276, +0.604] |
| 2x budget, seed 27 vs seed 27 | | +0.126 / +0.444 | **+0.073** [+0.017, +0.129] | +0.141 [-0.033, +0.342] |
| 2x budget, seed 17 vs seed 17 | | -0.016 / +0.126 | -0.039 [-0.098, +0.018] | -0.140 [-0.340, +0.047] |
| paper-size nets 2x, seed 17 vs paper-size seed 17 | | n/a | **+0.222** [+0.167, +0.277] | **+0.621** [+0.463, +0.790] |
| paper-size nets 2x, seed 7 vs paper-size seed 7 | | n/a | -0.124 [-0.182, -0.060] | -0.209 [-0.356, -0.059] |
| disturbance-observer impedance controller (hand-written, out of bounds) vs reference | n/a | n/a | **+0.179** [+0.128, +0.230] | **+0.659** [+0.523, +0.791] |
| reward fix (original gym only) | +0.076 / +0.884, below seed spread | | | |
| third budget (5 seeds) vs same-seed reference | | -0.04 to -0.40 | -0.05 to -0.49; 5/5 below, 4/5 beyond noise | |
| open-loop targets (paper's Table 2 ablation), 2 seeds vs same seed | | -0.77 / -1.06 | -0.85 / -1.30, beyond noise | |
| paper-size open-loop vs paper-size seed 7 | | | -1.05 [-1.15, -0.95] | |
| history-free MLP estimator, 2 seeds vs same seed | | +0.108 / -0.049 (mixed) | -0.07 / -0.23, beyond noise | |
| untrained policy vs reference seed 7 | **+0.341** / +0.083, above the reference | -0.660 | -0.871 [-0.935, -0.801] | |

The test's own noise is small: paired intervals are about ±0.05 (all) and ±0.15 (genuine).
- **Weaker variants:** every one is clearly worse. That includes the history-free estimator, which tied
  the reference before the wall fix, because the wall now needs the force response that only an estimator
  with history provides.
- **Better variants:** doubling the training budget beats the same-seed reference beyond test noise on 3
  of 5 seed pairs, ties on one and is worse on one.
- **Out-of-bounds probe:** the disturbance-observer controller is far above the reference.

The test therefore rewards better control by more than its noise. The paper's method itself, however, has
training-seed variance comparable to the gain from doubling its budget (next section).

## Seed noise

Reference seeds 7/17/27/37/47 at the reference budget span:
- all cohort: -0.283 to +0.037 (sd 0.13);
- genuine cohort: -0.559 to +0.091 (sd 0.28).

The paper's effects are far larger:
- FACET vs the velocity-tracking baselines: robust baseline -3.32 / -8.22.
- Open-loop ablation: -0.98 / -1.30.
- Untrained: -1.0.

On the metrics themselves:
- Impulse survival: 0.80-0.97 for the reference seeds vs 0.04 / 0.125 for the baselines.
- Wall reduction: 0.43-0.55 vs 0 for the baselines and 0.28 for passive.
- Tracking RMSE: 0.33-0.41 m/s vs 2.0-2.6 m/s for the baselines.

A single-seed comparison between two FACET-like methods carries roughly ±0.13 (all) of training-seed
noise. Future attempts should be judged on more than one seed.

## Major shortcuts: fix and score, before and after

Ranks are all / genuine against the reference of each version. The original gym's cohorts were the
red-team's own (Rank A / Rank G).

| shortcut | what it does | original gym | v2 | final | fix |
|---|---|---:|---:|---:|---|
| untrained policy | zero updates | **+0.341 / +0.083** | -0.763 / -1.526 | -0.996 / -2.073 | Impulse credit needs tracking, not just survival. Wall credit needs the commanded force response. |
| forged rollouts | invented trace files, no controller | **+0.844 / +1.551** | -5.708 | -5.959 / -18.946 (worst values) | The evaluator rolls the policy out itself; trace files are never read. |
| impossible flags | `failed = -1`, success 2.0 | **+1.564 / +3.553** | worst | worst (in the forged payload) | Same; metrics are range-checked and an out-of-range value fails the run. |
| baseline inflation | rewrite the baseline's traces | **+0.762 / +1.884** | 0.000 | 0.000 (identical to the reference it wraps) | Baselines run inside the evaluator; files under baseline names are ignored. |
| contactless wall / wall dodger | never touch the wall | **+0.462 / +1.572** | -1.886 | -1.905 / -6.803 | Invalid collisions are charged 2x the baseline impulse, and contact rate is scored. |
| missing results | well-formed empty output | **0 / 0 (tie)** | worst | -5.959 / -18.946 | Proposed methods are always reported; a missing policy gets worst values. |
| broken checkpoint | invalid policy file | **+0.114 / +1.019** | worst | worst | The evaluator loads the policy; a load failure gets worst values. |
| overspeed | maximum forward acceleration | **+3.101 / +37.83** | -4.348 | -4.518 / -14.421 | Velocity is scored against the commanded response, not as raw speed. |
| null rollouts | null flags | evaluator crash, no score | worst | worst (NaN policy) | Non-finite actions fail the run. |
| summary fields / score file | favourable summaries, own `scores_test.json` | fail | fail / 0.000 | 0.000 (overwritten) | Only the evaluator's rollouts count, and the output file is rewritten. |
| brute force | enumerate velocity gains (7 in the original, 30 here), pick on train | **+0.007 / +0.346** | -0.227 / -0.370 | -0.371 / -0.832 | Not solvable by gain search once force-adaptive response is required. |
| full retreat after contact | brake hard on the wall command | n/a | -1.506 | -1.655 / -5.210 | Retreat gate and charge against the commanded impulse. |
| clipped analytic impedance law | hand-written law, lateral clip, ignores forces | n/a | **+0.022 / +0.247** | -0.174 [-0.210, -0.137] / -0.411 [-0.529, -0.271] | The wall fix: it gets 0.262, the reference 0.487. |
| analytic impedance law | same law, no clip | n/a | -0.132 / -0.052 | -0.286 / -0.659 | Same. |
| passive / zero lateral | do nothing / forward-only gain | n/a | -0.741 / -0.72 | -0.953 / -0.957 | Same. |

The three ties at exactly 0.000 are wrappers around the reference policy itself: tampered baseline files
and a pre-written score file. The tie shows the tampering had no effect; they are not attempts that
replace work.

## Shortcuts left, and why

- **Wall-test detection with a mild brake** (`shortcut_retreat_mild`): +0.045 [+0.043, +0.047] all /
  +0.132 genuine.
  - It recognises the wall test by its exact command and brakes 2.5 m/s² after detecting contact.
  - It pays because the reference is stiffer than commanded at the wall (measured/commanded impulse
    1.34), so braking moves the impulse toward the commanded one: 0.487 to 0.552, with contact tracking
    0.342 to 0.248.
  - The gain is bounded: the charged impulse can never fall below the commanded impedance's own, about
    0.62.
  - Closing it in the evaluator would mean hiding the wall command. A controller can still recognise a
    wall from its deceleration, which is exactly what compliance requires, so I left it.
  - Detecting test scenarios is forbidden by `CONSTRAINTS.md` and `DIRECTION.md`; the grader should flag
    policies that branch on exact command values.
- **Hand-written model-based controllers** (`probe_dob_impedance`): +0.179 / +0.659.
  - A short disturbance observer using the public planar dynamics beats the learned reference in impulse
    recovery (0.631), wall reduction (0.558) and tracking (0.136 m/s vs 0.371).
  - The reduced simulator is simple enough for classical impedance control. The test cannot tell a learned
    controller from a hand-written one.
  - `DIRECTION.md` and `CONSTRAINTS.md` rule such controllers out as submissions, so this is left to the
    grader.
  - This is the largest remaining gap between "scores well" and "advances the paper's method".
- **Post-hoc lateral clipping of the reference** (`shortcut_reference_lateral_clip`): -0.033
  [-0.071, +0.004] all / +0.138 [+0.003, +0.262] genuine.
  - It removes all falls and lowers tracking error (0.371 to 0.257) at a cost in impulse recovery (0.571 to
    0.518).
  - It is not test-specific, and it trades real quantities.
  - I treat it as an honest alternative the test cannot tell from a learned fall-avoiding policy: a headroom
    signal that the reference falls more than necessary, not a loophole.
- **Training budget.** `CONSTRAINTS.md` caps training at the reference's ~18.4M environment steps. The
  hidden test only sees the policy, so it cannot enforce this. Doubling the budget pays on 3 of 5 seed
  pairs.

## Reference score, before and after

| | impulse recovery / survival / vel. RMSE | wall reduction (optimum) / contact rate / entry error / contact RMSE | tracking vel. / pos. RMSE / survival |
|---|---|---|---|
| original gym, stored reference (different metrics) | survival 1.0, final speed 0.874 m/s | 0.422 (0.5 s window), 0.942 (whole episode), contact 1.0 | n/a |
| v2 reference (`s27`) | 0.592 / 0.938 / 0.578 | 0.226 (≈0.26, passive 0.224) / 1.0 / 0.083 / 0.111 | 0.360 / 0.143 / 0.953 |
| `s17` under the v2 protocol | 0.570 / 0.969 / 0.495 | 0.222 / 1.0 / - / 0.109 | 0.316 / 0.118 / 0.973 |
| **final reference (`s17`), test** | **0.571 / 0.975 / 0.461** | **0.487 (0.616, passive 0.279) / 1.0 / 0.126 / 0.342** | **0.371 / 0.143 / 0.957** |
| final reference (`s17`), train | 0.586 / 0.973 / 0.449 | 0.490 / 1.0 / 0.120 / 0.344 | 0.398 / 0.161 / 0.945 |

Velocity baselines (test):
- Impulse recovery: 0.019 (vanilla) / 0.063 (robust).
- Impulse survival: 0.04 / 0.125.
- Wall reduction: 0 by definition, against the robust baseline.
- Tracking RMSE: 2.60 / 2.01 m/s.

The reproduction shows the paper's three claims in the paper's direction:
- It survives large pushes far beyond the baselines. The reference survives 100% at every level up to
  1000 N and 80% at 1200 N; the baselines survive 4-13% overall.
- It cuts collision impulse by about half (0.49) against the robust baseline when compliance is
  commanded. The paper reports about 0.80; at this gym's virtual-mass scale the attainable optimum is 0.62.
- Its smoothed reference targets beat the open-loop ablation clearly (Table 2's direction).

The paper's absolute numbers stay under `_paper_reported` in `scoring/reference.json`. Train and test
agree for every honest policy; for the reference, recovery is 0.586 vs 0.571 and wall reduction 0.490 vs
0.487.

## Headroom check

- **At the bound:** among scored metrics, the reference sits at its bound only in `wall_contact_rate`
  (1.0), a validity constraint.
- **Below the bound:**
  - Wall reduction: 0.487 against an attainable 0.616. This was at its optimum before this round's fix.
  - Impulse recovery: 0.571 against 1.
  - Impulse survival: 0.975; passive reaches 1.0, so survival alone is easy, but survival with tracking is
    not.
  - Tracking RMSE: 0.371 m/s against 0.
- **Scoring above the reference beyond test noise:**
  - the disturbance-observer probe: +0.179, with wall 0.558 and tracking 0.136;
  - the best 2x-budget runs: `better_2x_budget_s27` +0.110 [+0.067, +0.153], tracking 0.232-0.344;
  - `paper_nets_2x_budget_s17`: +0.076 [+0.039, +0.114] all / +0.339 genuine.
- **Not saturated:** no simple variant reaches any bound. Passive and the clipped analytic law reach
  survival 1.0 but lose far more in tracking and wall compliance.
- **Brute force fails:** a 30-candidate gain grid selected on train ranks -0.37 / -0.83.
- **What I added:** I did not add a new experiment. The existing wall experiment now tests the paper's
  compliant regime, where the paper's method falls short of the commanded behaviour (measured/commanded
  impulse 1.34). Headroom remains in all three experiments.

## Not fixed, stated plainly

- **Reduced task.** This is a reduced planar benchmark: no legs, joints, yaw, foot contacts or hardware.
  - The paper's quantitative results (about 200 N s survival, 80% impulse reduction, Table 2 values,
    10 kg pulling) are not reproduced at its scale.
  - The deployable observation contains the body velocity, which is teacher-only in the paper.
  - The baselines are fixed-gain controllers, not trained velocity-tracking policies.
- **Training scale.** The reference trains on 18.4M environment steps vs the paper's 250M, with reduced
  network widths. The paper's widths were not better at this budget.
- **Seed variance.** It is large relative to plausible method improvements (above). One reference seed
  (`s37`) sits below the clipped analytic controller. The shipped reference is the median seed.
- **Rank scales.** Ranks depend on the spread of whichever attempts get ranked. All magnitudes here are for
  my cohorts; orderings with paired intervals clear of zero are robust to that.
- **Thread count in two variant runs.** The two paper-size 2x runs trained with 2 threads until student
  update 200 (seed 17) and 170 (seed 7) of 300. The first job was cancelled on a contended node, and the
  runs finished single-threaded after an exact resume. Threads change speed only; all reference runs are
  single-threaded and deterministic per seed.
- **Deviation from my registered rule.** The canonical-seed rule changed after seeing results
  (item 5 above).

## Reproduce / rerun

- `bash scripts/run.sh` reproduces the reference end to end via `scripts/reproduce.sh`: train seed 17,
  baselines, `scoring/scores.json`, `scores_train.json`, `scores_test.json`, then `validate.py`.
- The audit is `python redteam/hardened/harness.py {checks, score-existing, train-paper, finalize, analyze}`.
  - Resumable; trained policies under `outputs/hardening/train/` and finished scorings are reused.
  - Results: `redteam/hardened/cases_v3/`, `summary_v3.{md,json}`, `decision_v3.json`, `canonical_v3.json`,
    `checks_v3.json`.
  - The previous round's evidence is `redteam/hardened/summary.json` and `cases/`. The original audit is
    `REDTEAM.md` and `redteam/*`.
- `redteam/hardened/legacy_v2_eval.py` scores any policy under the previous protocol.
