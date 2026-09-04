# Battle mode

## Implementation plan

Battle mode is a separate extension of the existing C++ runtime. It keeps the
single-player `CCTetrisEnv`, 254-feature observation, and 3,200 stable
`(hold, rotation, y, x)` actions unchanged.

The implementation proceeds in five layers:

1. Add an engine-owned garbage-row primitive and a native two-player match
   coordinator. The coordinator validates both actions against one pre-step
   state and resolves placements, attacks, cancellation, delayed garbage, and
   terminal state atomically.
2. Expose a separate battle C API and a Python joint environment. Every policy
   receives its own 254 features, the opponent's public visible board, public
   garbage/board statistics, and its own legal-action mask. Opponent previews
   and randomizer state remain private.
3. Extend the masked Double-DQN path with a battle-aware placement network,
   compact replay storage, full resume state, randomized seats, and a bounded
   pool of recent and older frozen policies.
4. Train through fixed-seed curriculum gates against legal-random, Cold Clear,
   frozen checkpoints, and a final mixed pool. All episode and reward data is
   written to JSONL without an external tracking service.
5. Evaluate paired, seat-swapped matches and checkpoint matrices, then publish
   the measured results. Win-rate targets are reported as passed only when the
   corresponding 500-match fixed-seed report exists.

## Implementation map

| Area | Important files | Purpose |
| --- | --- | --- |
| Native battle core | `include/tetris_v2/battle.hpp`, `src/battle.cpp` | Atomic joint match, attack, FIFO cancellation, delayed garbage, terminal state, canonical observation |
| Stable placements and garbage primitive | `include/tetris_v2/decision.hpp`, `src/decision.cpp`, `include/tetris_v2/env.hpp`, `src/env.cpp`, `include/tetris_v2/cc_env.hpp`, `src/cc_env.cpp` | Preserve the 3,200 complete-placement contract and add engine-owned garbage insertion |
| C ABI and deterministic heuristic | `include/tetris_v2/c_api.h`, `src/c_api.cpp`, `include/tetris_v2/cold_clear_bot.hpp`, `src/cold_clear_bot.cpp`, `include/tetris_v2/cc2_dag.hpp`, `src/cc2_dag.cpp`, `include/tetris_v2/cc2_sync.hpp`, `src/cc2_sync.cpp` | Stable battle bindings and fixed-work Cold Clear |
| Python environment and learning | `tetris_v2/rl/battle/{config,stats,reward,runtime,env,dqn,replay,opponents,policies,checkpoint,curriculum,metrics,evaluation,train,cli}.py` | Joint environment, shared masked Double-DQN, replay, pool, exact resume, curriculum, metrics, and evaluation |
| Commands and configuration | `scripts/{battle_smoke,eval_battle,eval_battle_pool,battle_matrix}.py`, `tetris_v2/conf/trainer/battle_dqn.yaml`, `tetris_v2/conf/experiment/{battle_smoke,battle_selfplay}.yaml`, `tetris_v2/conf/battle_curriculum.yaml`, `pyproject.toml` | Installed CLIs, CPU smoke/long presets, and configurable rules/gates |
| Existing API integration | `tetris_v2/train.py`, `tetris_v2/rl/policy.py`, `scripts/{eval_rl,play_rl_cli,play_rl_pygame}.py` | Hydra dispatch plus battle checkpoint evaluation/playback without changing single-player APIs |
| Regression coverage | `tests/cpp/battle_tests.cpp`, `tests/cpp/c_api_tests.cpp`, `tests/python/test_battle_*.py`, `tests/python/test_hydra_training.py`, `tests/python/test_policy.py` | Mechanics, symmetry, schemas, DQN math, resume, logging, CLI, metrics, and single-player compatibility |
| Build and documentation | `CMakeLists.txt`, `README.md`, `docs/TRAINING.md`, `docs/BATTLE.md` | Native targets, user entry points, full contract, commands, and measured results |

## Rule decisions

- Both seats have independent boards, holds, queues, randomizer state, pending
  garbage, and statistics. By default their piece streams start from the same
  match seed; this is configurable.
- The default line-clear attack table is `[0, 0, 1, 2, 4]`. Optional combo,
  back-to-back, spin, and perfect-clear bonuses default to disabled.
- A garbage packet stores its hole columns when it is sent. Dedicated seeded
  streams for the two recipients prevent player-processing order from changing
  the holes.
- Outgoing attack cancels the sender's oldest pending rows first. Any remainder
  becomes a packet for the opponent. Simultaneous outgoing attacks cross; they
  do not cancel one another before becoming incoming packets.
- Garbage delay is measured in joint placement steps and defaults to one. Thus
  a player has one placement in which to cancel a newly received packet.
- Garbage is applied only after both placements, attacks, and cancellations are
  known. A block pushed above the 40-row board or collision with the newly
  spawned active piece is a top-out.
- If both players top out in one joint step, the result is a draw. Reaching the
  configurable match-step limit (500 by default) is also a draw.
- Invalid actions are rejected before either board changes. Evaluation counts
  them as policy failures; it never silently substitutes a legal action.

## Public policy contract

Battle mode keeps the single-player action ID exactly intact. Each player
chooses one of 3,200 complete placements, flattened as
`(use_hold, rotation, landing_y, x)` with shape `[2, 4, 40, 10]`. Both masks
are computed before a joint step. An invalid action rejects the whole step;
there is no evaluator or trainer fallback.

The native C ABI and Python environment expose the same 470 values under
schema `tetris_v2_battle_470_v1`; Python does not replace or reinterpret a
native-only tail:

| Slice | Values | Visibility |
| --- | --- | --- |
| `0:254` | Own visible board, active piece, hold, five previews, hold availability, combo, and back-to-back | Private to the controlled player |
| `254:454` | Opponent's 20 x 10 locked visible board, in the same bottom-up board order used by the existing network | Public |
| `454:458` | Own/opponent queued garbage and next-arrival delay | Public battle state |
| `458:463` | Own aggregate/max height, holes, bumpiness, and wells | Derived from the public board |
| `463:468` | Opponent aggregate/max height, holes, bumpiness, and wells | Derived from the public board |
| `468:470` | Mapped height and hole advantages | Derived and normalized |

The legal-action mask remains separate and has 3,200 entries. Values with
different natural ranges are mapped to `[0, 1]`. The opponent's active piece,
hold, preview queue, bag order, and randomizer state are never copied into the
controlled player's observation.

## Joint-step order

For each step the native coordinator:

1. enumerates and validates both stable placement IDs against the pre-step
   state;
2. applies both locks, line clears, and next-piece spawns;
3. computes attack from the configured line-clear table;
4. cancels each sender's queued garbage FIFO;
5. pre-samples every remaining outgoing row's hole from the recipient's seeded
   garbage stream and enqueues the packet;
6. applies every packet whose delay is due;
7. checks both top-outs together, then the match-length draw; and
8. publishes both successor observations, masks, events, and cumulative stats.

This order means processing Player 1's C++ object first cannot change Player
2's submitted decision or attack. A max-step draw is exposed as a Gymnasium
time-limit truncation with valid successor masks so DQN bootstraps it. A real
top-out is a termination and has empty successor masks.

Each player's public match stats include placements, cumulative native raw
score, total lines, attack, cancellation, garbage sent/received/applied, and
top-out state. The raw score is also retained in episode JSONL; it is separate
from the antisymmetric learning return below.

## Reward

Rewards are exactly antisymmetric. Player 0 receives the following value and
Player 1 receives its negative:

```text
20.00 * terminal_result
+ 0.05 * (garbage_sent_0 - garbage_sent_1)
+ 0.03 * (garbage_cancelled_0 - garbage_cancelled_1)
+ 0.01 * (lines_0 - lines_1)
+ 0.02 * (board_quality_change_0 - board_quality_change_1)
+ 0.02 * (height_change_1 - height_change_0)
+ 0.03 * (hole_change_1 - hole_change_0)
+ 0.04 * (garbage_applied_1 - garbage_applied_0)
```

`terminal_result` is `+1`, `-1`, or `0` for a win, loss, or draw. Board terms
are potential differences, not persistent per-step bonuses, and there is no
reward for merely extending a match. All eight weights are configurable in
[`battle_dqn.yaml`](../tetris_v2/conf/trainer/battle_dqn.yaml). Training JSONL
and evaluation JSON retain every component plus the total.

## Learner, replay, and opponent pool

`BattlePlacementQNet` keeps the existing 254-input placement-convolution DQN
as its own-state branch and adds an opponent-board/context residual over the
same 3,200-value Q map. The residual output starts at zero. Consequently,
`--init-checkpoint` can import an existing 254/3,200 placement-convolution DQN
without changing any initial Q value.

Training uses a shared masked Double-DQN, a target network, Huber loss,
gradient clipping, linear epsilon and learning-rate schedules, and batched
3,200-action scoring. Replay stores both player-perspective observations and
bit-packs current and successor masks. True terminals do not bootstrap;
time-limit truncations do.

The learner seat is randomized. Frozen-opponent matches add the learner's
transition; current-policy mirror matches add valid transitions from both
perspectives. The bounded pool preserves the initial and newest policy and a
spread of older policies, then samples from both recent and older buckets.
Snapshots are added at fixed intervals and whenever a curriculum gate passes.

The default curriculum is:

| Stage | Training mix | Fixed-seed promotion gate |
| --- | --- | --- |
| Random | 100% random | 90% wins over at least 100 matches |
| Heuristic | 30% random, 70% deterministic Cold Clear | 95% random and 55% Cold Clear |
| Frozen self-play | 10% random, 20% Cold Clear, 60% frozen, 10% current | 95% random, 60% Cold Clear, 52% frozen |
| Mixed | 20% random, 30% Cold Clear, 50% frozen | Final stage |

Promotion never uses training return. `think_ms=0` makes Cold Clear run exactly
eight seeded DAG work units, which is the default for reproducible training
and evaluation. A positive value deliberately restores the older wall-clock
budget and may vary with machine load. Automatic curriculum promotion and the
Cold Clear win-rate gate therefore require `think_ms=0`; positive-budget
benchmark reports are explicitly marked nondeterministic.

The stage mixes and every promotion threshold live in
[`battle_curriculum.yaml`](../tetris_v2/conf/battle_curriculum.yaml). Copy that
file for an experiment and pass `--curriculum-config path/to/copy.yaml`; the
trainer validates it into typed stages, stores the resolved definitions in
full checkpoints, and rejects a different curriculum on resume. This keeps
promotion settings configurable without allowing an accidental mid-run rule
change.

## Checkpoints and exact continuation

Compact `algo="battle_dqn"` policy files contain the network, observation and
action schemas, architecture config, battle rules, reward weights, and
measurement metadata. Full
`algo="battle_dqn_training"` files additionally contain online/target weights,
optimizer and schedules, counters, every RNG stream, curriculum, bounded pool,
configuration, and a checksum-protected compressed replay sidecar.

Full checkpoints are written only after a match finishes. If a requested step
budget is crossed mid-match, training completes that match and records the
small overshoot. This is what makes resume exact: the native match itself does
not need to be serialized. Changing a trajectory-affecting rule, reward,
schedule, mix, buffer, or optimizer setting on resume is rejected. The total
timesteps argument is an absolute target, not an amount to add.

Each episode JSONL row names the learner state as
`learner_step_<global-step>` as well as the opponent type, identifier, and
checkpoint path, so later analysis can associate both sides with the exact
training boundary. It records generated attack and post-cancellation garbage
sent separately; `loss`/`episode_mean_loss` are the mean of updates performed
in that episode, while `last_loss` and `optimizer_updates` make a zero-update
episode unambiguous.

Each full trainer checkpoint also embeds the compact policies for every
currently retained frozen opponent. Consequently, an older trainer checkpoint
can still resume or evaluate its bounded pool after later training evicts the
original files, or after the checkpoint and replay sidecar are moved together.

Standalone evaluation inherits the learner checkpoint's attack table, garbage
delay, match limit, piece-stream mode, and reward weights. Passing
`--attack-table`, `--garbage-delay`, `--max-steps`,
`--independent-piece-seeds`, or `--mirrored-piece-seeds` is an explicit
evaluation override and the resolved contract is written into the JSON report.
Older compact policies that predate stored battle metadata use the documented
defaults.

## Commands

Build and run every native and Python test first:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
python -m pytest -q
python -m pytest -q tests/python/test_battle_*.py
```

Exercise the native ABI, paired seats, deterministic replay, and two CPU
backpropagation updates:

```bash
tetris-battle-smoke \
  --matches 2 \
  --max-steps 8 \
  --opponent cold_clear \
  --repeat-determinism \
  --train-updates 2 \
  --json-output runs/battle_smoke/report.json
```

The Hydra development and long-run presets are:

```bash
tetris-train experiment=battle_smoke runtime=cpu
tetris-train experiment=battle_selfplay runtime=cpu
```

The equivalent direct long run, including the documented 20/30/50 manual mix,
is:

```bash
tetris-train-battle \
  --total-timesteps 1000000 \
  --device cpu \
  --disable-curriculum \
  --attack-table 0 0 1 2 4 \
  --garbage-delay 1 \
  --max-match-steps 500 \
  --opponent-pool-size 20 \
  --checkpoint-frequency 100000 \
  --pool-checkpoint-frequency 100000 \
  --eval-frequency 50000 \
  --eval-matches 100 \
  --log-dir runs/battle_selfplay
```

Curriculum mode is the default. Add `--disable-curriculum` to use the four
manual opponent weights. To tune automatic stage gates, copy
`tetris_v2/conf/battle_curriculum.yaml` and add
`--curriculum-config configs/my_battle_curriculum.yaml`. Warm-start and resume
are intentionally different:

```bash
# Copy only compatible single-player DQN weights into a fresh battle run.
tetris-train-battle \
  --init-checkpoint runs/dqn/dqn_final.pt \
  --total-timesteps 1000000 \
  --log-dir runs/battle_warm_start

# Continue the manual --disable-curriculum run to a larger absolute target.
tetris-train-battle \
  --resume-checkpoint runs/battle_selfplay/battle_training_final.pt \
  --total-timesteps 1500000 \
  --disable-curriculum \
  --log-dir runs/battle_selfplay_resume
```

Resume must preserve the original run's resolved curriculum/manual-mix mode
and trajectory-affecting arguments; use the saved checkpoint configuration as
the source of truth.

Run the two required 500-match gates on disjoint fixed seed blocks. These
commands exit nonzero when the win, legality, determinism, or seat-fairness
gate fails:

```bash
tetris-eval-battle runs/battle_selfplay/battle_dqn_final.pt \
  --opponent random \
  --matches 500 \
  --seed 2000000 \
  --repeat-determinism \
  --json-output runs/battle_selfplay/eval_random_500.json

tetris-eval-battle runs/battle_selfplay/battle_dqn_final.pt \
  --opponent cold_clear \
  --cold-clear-think-ms 0 \
  --matches 500 \
  --seed 2100000 \
  --repeat-determinism \
  --json-output runs/battle_selfplay/eval_cold_clear_500.json
```

Scripted garbage pressure, retained-pool evaluation, and a raw matrix are
separate reproducible reports:

```bash
tetris-eval-battle runs/battle_selfplay/battle_dqn_final.pt \
  --opponent cold_clear --matches 100 --seed 2200000 \
  --pressure-rows 4 --pressure-interval 20 --pressure-hole 4 \
  --no-default-win-gate --json-output runs/battle_selfplay/pressure.json

tetris-eval-battle-pool \
  runs/battle_selfplay/battle_training_final.pt \
  --matches-per-snapshot 20 \
  --json-output runs/battle_selfplay/pool_eval.json

tetris-battle-matrix \
  runs/battle_selfplay/opponent_pool/battle_policy_step_000000000000.pt \
  runs/battle_selfplay/battle_dqn_final.pt \
  --labels initial final \
  --matches-per-pair 100 \
  --json-output runs/battle_selfplay/matrix.json \
  --csv-output runs/battle_selfplay/matrix.csv
```

A frozen battle policy can also run the existing no-opponent survival gate.
The adapter uses its own 254 features, a blank public opponent board, no queued
garbage, and correctly derived own board statistics:

```bash
tetris-eval runs/battle_selfplay/battle_dqn_final.pt \
  --algo battle_dqn \
  --episodes 100 \
  --seed 2300000 \
  --max-steps 2000 \
  --min-placements 2000 \
  --json-output runs/battle_selfplay/survival_2000.json
```

Every evaluation JSON includes raw per-match seeds, learner seat, placements,
lines, attack generated, post-cancellation garbage sent, cancellation, garbage
received/applied, top-outs, returns, board statistics, inference latency,
reward components, illegal actions, and a deterministic trace hash. Attack per
100 pieces uses generated attack, while separate garbage-sent rates retain the
post-cancellation value. Inference is weighted per actual policy decision and
height/hole summaries are weighted per sampled board state. Summaries retain
physical-seat win rates and raw match counts for fairness, plus separate
learner/opponent averages for returns, lines, attack, garbage, top-outs,
height, holes, legality, and latency. Matrix
JSON and CSV retain raw win rate, draw-adjusted score rate, W/L/D totals, and
`n` for every directed pairing.

## Current measured status and known limits

The audit separates a deliberately tiny from-scratch pipeline checkpoint from
a stronger warm start. The tiny run requested 20 online steps and completed at
the next exact episode boundary, step 25. The warm start imported
`runs/v3_hybrid_finetune/dqn_hybrid_final.pt`, requested one step, and finished
its first match at step 22 with warmup set to 1,000, so it performed no battle
optimizer update. This isolates whether the compatible single-player branch
transfers before self-play. Local raw reports are under
`runs/battle_audit_20260827/` and are ignored by Git.

| Measurement | Measured result |
| --- | --- |
| Strict native build and C++ tests | Passed with `-Wall -Wextra -Wpedantic -Werror`; all 6 CTest targets passed five consecutive runs |
| Canonical observation contract | Native C ABI and Python policy both expose the same 470 normalized values; exact-size drift, end-to-end equality, privacy, and perspective tests passed |
| Full Python regression, including the pre-existing single-player algorithms | 153 passed; one third-party pygame/setuptools deprecation warning |
| Focused Python battle regression | 50 passed |
| Existing single-player DQN smoke | Two 10-placement episodes, zero illegal actions, both reached the requested time limit without top-out |
| Native/Python battle smoke, two fixed-work Cold Clear matches at four steps | Passed; two draws, zero illegal actions, exact deterministic repeat, and two finite CPU updates |
| Battle smoke wall time and peak RSS | 1.63 seconds; 774,400 KiB |
| Tiny train/save run | 20 requested / 25 episode-boundary steps; 1.85 seconds process wall time, 777,972 KiB peak RSS, about 0.300 seconds trainer time in the checkpoint |
| Resume from that full checkpoint | Continued to the next boundary at step 38 with replay and the one-entry frozen pool restored |
| Current episode/evaluation logging contract | A three-update episode records generated and sent attack separately, mean loss 0.758448, last loss 0.709261, update count 3, both policy identifiers, raw score, board stats, and all reward parts. A curriculum evaluation records step 1 and 0.0588 seconds training time in JSON and JSONL |
| Warm start vs random, 500 paired matches, seeds 3,400,000 onward | **Passed target:** 500 wins / 0 losses / 0 draws; 100% win and score rates, 0% seat gap, zero illegal actions, deterministic repeat passed |
| Warm-start random detail | Minimum 13, median 22, mean 21.572 placements; 5.91 lines, 2.756 generated/sent attack rows, and 12.776 attack per 100 pieces per learner match |
| Warm-start random evaluation cost | 70.46 seconds process wall time; 762,308 KiB peak RSS, including the repeated 500-match determinism pass |
| Warm start vs fixed-work Cold Clear, 20 development matches, seeds 3,300,000 onward | **Failed heuristic target:** 0 wins / 20 losses; deterministic repeat and legality passed. Minimum 74, median 120.5, mean 130.95 placements; 58.9 lines, 18.5 generated attack, 16.05 sent, 2.45 cancelled, and 27.65 applied rows per match |
| Warm start vs itself, 100 paired matches, seeds 3,700,000 onward | 44 wins / 44 losses / 12 draws; median 265.5 and mean 273.1 placements, 130.46 lines, 43.55 generated attack, 39.83 sent, 3.72 cancelled, and 35.94 applied rows. Determinism and legality passed; the strict seat gate **failed** at a 20-point physical-seat gap |
| Warm-start no-opponent survival development, 20 seeds | 16/20 reached 2,000 placements; minimum 1,460, median 2,000, mean 1,919.15, zero illegal actions. This demonstrates survival on a meaningful share; the disjoint 100-seed result is reported below |
| Warm-start no-opponent survival final, 100 seeds beginning at 3,900,000 | **74/100 reached 2,000 placements**; minimum 45, fifth percentile 667.15, median 2,000, mean 1,745.51; mean 694.27 lines and zero illegal actions |
| Warm-start survival evaluation cost | 275.46 seconds process wall time; 761,448 KiB peak RSS |
| Fresh untrained Battle-DQN vs random baseline, 100 paired matches | 26 wins / 72 losses / 2 draws; median 18.5 and mean 18.58 placements, zero attack and illegal actions |
| Fresh untrained Battle-DQN vs Cold Clear baseline, 100 paired matches | 0 wins / 100 losses; median 21 and mean 20.68 placements; Cold Clear averaged 3.91 lines and 2.74 attack rows, zero illegal actions |
| 500 paired matches vs random, seeds 2,000,000 onward | **Failed target:** 20 wins / 464 losses / 16 draws; 4.0% win rate, 5.6% score rate, 0% seat gap, zero illegal actions, deterministic repeat passed |
| Random-gate match detail | Minimum 11, median 14, mean 14.996 learner placements; learner/opponent lines 0.004/0.024, learner mean return -17.766, no attack generated |
| Random-gate evaluation cost | 51.05 seconds process wall time; 761,632 KiB peak RSS, including the repeated 500-match determinism pass |
| 500 paired matches vs fixed-work Cold Clear, seeds 2,100,000 onward | **Failed target:** 0 wins / 500 losses / 0 draws; 0% win rate, 0% seat gap, zero illegal actions, deterministic repeat passed |
| Cold-Clear-gate match and attack result | Minimum 9, median 14, mean 14.512 placements; learner/opponent lines 0.004/1.88, attack 0/1.16 rows, learner mean return -20.136 |
| Cold-Clear-gate evaluation cost | 69.09 seconds process wall time; 761,960 KiB peak RSS, including the repeated 500-match determinism pass |
| Scripted four-row pressure, 100 matches vs Cold Clear | Route passed legality/fairness checks; learner lost 100/100, averaged 12.34 placements, applied 4.4 garbage rows, returned -20.252, and generated no attack |
| Tiny learner vs itself, 100 paired matches | 100 draws, median 14 and mean 14.7 placements, zero attack and illegal actions. The warm start above supersedes this passive result for the equal-strength attack/defense target |
| Initial vs step-25 matrix, 100 paired matches | Initial-to-step25 W/L/D was 18/12/70 (18% raw wins, 53% score); reverse was 12/18/70 (12%, 47%); zero illegal actions. **No clear checkpoint gain.** |
| Retained-pool route, 20 paired matches at the stored 20-step limit | Step 25 vs initial: 6 wins / 0 losses / 14 draws, 65% score rate, zero seat gap and illegal actions |
| Existing 2,000-placement single-player survival gate, 100 seeds | **Failed target:** 0/100 reached 2,000; minimum 11, median 14, mean 14.7 placements; all topped out, zero illegal actions |

The requested success targets resolve as follows:

| Target | Status |
| --- | --- |
| Existing single-player and new battle regressions | **Passed** |
| Zero illegal evaluation actions and fixed-seed replay | **Passed** on every cited repeated battle report; survival reports also had zero illegal actions |
| Seat fairness | Native real-policy swap regression passed, but the warm self 100-match block failed the 5% gate at 20%; **not marked fully passed** |
| At least 95% of 500 vs random | **Passed**, 500/500 |
| At least 65% of 500 vs heuristic | **Not reached**; 0/20 development, so no final block was spent |
| Clear gain over initial checkpoint | **Not reached** by the step-25 matrix |
| Retention after self-play begins | **Not established**; no long self-play run was performed |
| Meaningful 2,000-placement survival share | **Passed**, 74/100 |
| Equal-strength offense and defense | **Passed behaviorally**: nonzero generation, sending, cancellation, and applied garbage; its separate seat gate failed as noted above |
| Save/stop/resume with opponent-pool state | **Passed** by exact split-versus-uninterrupted regression and the measured resume artifact |

`final_contract_training/episodes.jsonl` was produced before the final
reporting-only logger enrichment. The current schema proof is
`logging_contract_final/episodes.jsonl`; the older checkpoint weights, replay,
and evaluation reports are unaffected. Curriculum timing proof is under
`curriculum_contract_final/evaluation/` and `evaluations.jsonl`.
The disposable `curriculum_contract.yaml` used there has an `audit` stage with
`random: 1.0`, a two-match zero-win-rate promotion check, and a terminal
`complete` stage with `random: 1.0`; the regression test constructs the same
configuration.

Timed CPU evaluations set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
`OPENBLAS_NUM_THREADS=1`; some earlier gate jobs were co-scheduled. Their
process wall times are reproducibility observations, not isolated throughput
benchmarks.

The principal audit commands, including their exact seed blocks, were:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DCMAKE_CXX_FLAGS='-Wall -Wextra -Wpedantic -Werror'
cmake --build build --parallel
ctest --test-dir build --repeat-until-fail 5 --output-on-failure
python -m pytest -q
python -m compileall -q tetris_v2 scripts tests/python

# Used for the timed CPU evaluation commands below.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

tetris-eval runs/dqn_pretrain/dqn_expert_pretrain.pt \
  --algo dqn --episodes 2 --seed 2700000 --max-steps 10 \
  --json-output runs/battle_audit_20260827/single_player_regression_smoke.json

tetris-battle-smoke --matches 2 --max-steps 4 --opponent cold_clear \
  --repeat-determinism --train-updates 2 \
  --json-output runs/battle_audit_20260827/battle_smoke_final.json

tetris-train-battle --total-timesteps 20 --buffer-size 256 \
  --warmup-steps 10 --batch-size 8 --train-frequency 1 --gradient-steps 1 \
  --learning-rate-decay-steps 20 --epsilon-start 0.2 --epsilon-end 0.02 \
  --epsilon-decay-steps 20 --max-match-steps 20 --opponent-pool-size 20 \
  --pool-checkpoint-frequency 1000 --random-opponent-weight 1 \
  --heuristic-opponent-weight 0 --frozen-opponent-weight 0 \
  --current-opponent-weight 0 --disable-curriculum --log-frequency 10 \
  --eval-frequency 1000 --checkpoint-frequency 1000 --device cpu --seed 123 \
  --log-dir runs/battle_audit_20260827/final_contract_training

tetris-eval-battle runs/battle_audit_20260827/final_contract_training/battle_dqn_final.pt \
  --opponent random --matches 500 --seed 2000000 --max-steps 500 \
  --repeat-determinism \
  --json-output runs/battle_audit_20260827/eval_random_500_final.json
tetris-eval-battle runs/battle_audit_20260827/final_contract_training/battle_dqn_final.pt \
  --opponent cold_clear --matches 500 --seed 2100000 --max-steps 500 \
  --repeat-determinism \
  --json-output runs/battle_audit_20260827/eval_cold_clear_500_final.json

tetris-battle-smoke --matches 100 --seed 3000000 --max-steps 500 \
  --opponent random --train-updates 0 \
  --json-output runs/battle_audit_20260827/baseline_fresh_vs_random_100.json
tetris-battle-smoke --matches 100 --seed 3100000 --max-steps 500 \
  --opponent cold_clear --cold-clear-think-ms 0 --train-updates 0 \
  --json-output runs/battle_audit_20260827/baseline_fresh_vs_cold_clear_100.json
tetris-eval-battle runs/battle_audit_20260827/final_contract_training/battle_dqn_final.pt \
  --opponent self --matches 100 --seed 2900000 --max-steps 500 \
  --no-default-win-gate \
  --json-output runs/battle_audit_20260827/eval_self_100.json

tetris-eval-battle runs/battle_audit_20260827/final_contract_training/battle_dqn_final.pt \
  --opponent cold_clear --matches 100 --seed 2600000 --max-steps 500 \
  --pressure-rows 4 --pressure-interval 20 --pressure-hole 4 \
  --no-default-win-gate --no-seat-fairness-gate \
  --json-output runs/battle_audit_20260827/pressure_100_final.json

tetris-battle-matrix \
  runs/battle_audit_20260827/final_contract_training/opponent_pool/battle_policy_step_000000000000.pt \
  runs/battle_audit_20260827/final_contract_training/battle_dqn_final.pt \
  --labels initial step25 --matches-per-pair 100 --seed 2200000 \
  --json-output runs/battle_audit_20260827/matrix_final.json \
  --csv-output runs/battle_audit_20260827/matrix_final.csv
tetris-eval-battle-pool \
  runs/battle_audit_20260827/final_contract_training/battle_training_final.pt \
  --matches-per-snapshot 20 --seed 2300000 --no-seat-fairness-gate \
  --json-output runs/battle_audit_20260827/pool_eval_final.json

tetris-eval runs/battle_audit_20260827/final_contract_training/battle_dqn_final.pt \
  --algo battle_dqn --episodes 100 --seed 2400000 --max-steps 2000 \
  --min-placements 2000 \
  --json-output runs/battle_audit_20260827/survival_2000.json

# Current three-update episode logging proof.
tetris-train-battle --total-timesteps 1 --buffer-size 4 --warmup-steps 0 \
  --batch-size 1 --train-frequency 1 --gradient-steps 3 \
  --learning-rate-decay-steps 10 --epsilon-decay-steps 10 \
  --target-sync-interval 2 --max-match-steps 1 --opponent-pool-size 2 \
  --random-opponent-weight 1 --heuristic-opponent-weight 0 \
  --frozen-opponent-weight 0 --current-opponent-weight 0 \
  --disable-curriculum --log-frequency 100 --eval-frequency 100 \
  --checkpoint-frequency 100 --pool-checkpoint-frequency 100 \
  --log-dir runs/battle_audit_20260827/logging_contract_final \
  --device cpu --seed 811

# Two-match disposable gate used only to prove curriculum timing output.
tetris-train-battle --total-timesteps 1 --buffer-size 4 --warmup-steps 0 \
  --batch-size 1 --train-frequency 1 --gradient-steps 1 \
  --learning-rate-decay-steps 10 --epsilon-decay-steps 10 \
  --target-sync-interval 2 --max-match-steps 1 --opponent-pool-size 2 \
  --random-opponent-weight 1 --heuristic-opponent-weight 0 \
  --frozen-opponent-weight 0 --current-opponent-weight 0 \
  --curriculum-config runs/battle_audit_20260827/curriculum_contract.yaml \
  --log-frequency 100 --eval-frequency 1 --eval-matches 2 \
  --checkpoint-frequency 100 --pool-checkpoint-frequency 100 \
  --log-dir runs/battle_audit_20260827/curriculum_contract_final \
  --device cpu --seed 812

# Compatible single-player warm start; warmup prevents a battle update.
tetris-train-battle \
  --init-checkpoint runs/v3_hybrid_finetune/dqn_hybrid_final.pt \
  --total-timesteps 1 --buffer-size 256 --warmup-steps 1000 --batch-size 8 \
  --max-match-steps 500 --opponent-pool-size 4 \
  --pool-checkpoint-frequency 1000 --random-opponent-weight 1 \
  --heuristic-opponent-weight 0 --frozen-opponent-weight 0 \
  --current-opponent-weight 0 --disable-curriculum --log-frequency 1000 \
  --eval-frequency 1000 --checkpoint-frequency 1000 --device cpu --seed 3201 \
  --log-dir runs/battle_audit_20260827/warmstart_probe

tetris-eval-battle runs/battle_audit_20260827/warmstart_probe/battle_dqn_final.pt \
  --opponent random --matches 500 --seed 3400000 --max-steps 500 \
  --repeat-determinism \
  --json-output runs/battle_audit_20260827/warmstart_probe/eval_random_final500_seed3400000.json
tetris-eval-battle runs/battle_audit_20260827/warmstart_probe/battle_dqn_final.pt \
  --opponent cold_clear --matches 20 --seed 3300000 --max-steps 500 \
  --repeat-determinism --no-default-win-gate \
  --json-output runs/battle_audit_20260827/warmstart_probe/eval_cold_clear_dev20.json
tetris-eval-battle runs/battle_audit_20260827/warmstart_probe/battle_dqn_final.pt \
  --opponent self --matches 100 --seed 3700000 --max-steps 500 \
  --repeat-determinism --no-default-win-gate \
  --json-output runs/battle_audit_20260827/warmstart_probe/eval_self_final100_seed3700000.json
tetris-eval runs/battle_audit_20260827/warmstart_probe/battle_dqn_final.pt \
  --algo battle_dqn --episodes 100 --seed 3900000 --max-steps 2000 \
  --json-output runs/battle_audit_20260827/warmstart_probe/survival_final100_seed3900000.json
```

The compatible warm start reached the 95%-of-500 random target and turned the
previously passive equal-policy match into long attack/cancellation/garbage
play. Its separate survival run also established the requested 2,000-piece
behavior on a meaningful share of seeds. Those results come from transferred
single-player weights with zero battle optimizer updates, not from claimed
self-play improvement.

The 65%-of-500 Cold Clear target, clear gain over the initial checkpoint,
post-self-play retention, and the strict physical-seat gate on the self block
remain **unreached**. A 500-match Cold Clear final block was not spent after
the disjoint 20-match development block lost 20/20. The from-scratch step-25
learner also remained below the fresh baselines and produced no attack; it is
a pipeline audit, not meaningful training. A million-step CPU self-play run
was not performed inside this implementation pass.

The next useful algorithm run is the staged curriculum initialized from the
now-validated warm checkpoint. The zero-initialized battle residual must learn
opponent pressure and incoming-garbage response while weak/frozen opponents
prevent loss of the 100% random result. Rerun the random gate, Cold Clear
development set, checkpoint matrix, retained-pool checks, self-seat audit, and
survival benchmark on disjoint seeds at every promotion; reserve a new
500-match Cold Clear block for a checkpoint that first clears development.

At the default 50,000-transition replay capacity, observations use about 188
MB and packed masks about 40 MB before models, optimizer, native state, and
framework overhead. The measured tiny-update smoke includes PyTorch process
overhead and is intentionally a conservative peak, not a promise for every
platform.

Combo, back-to-back, spin, and perfect-clear attack bonuses are deliberately
rejected when enabled because battle-specific detection has not been
implemented and tested. The public opponent plane contains locked cells, not
the falling piece. There is no rendering rewrite, online multiplayer, or
low-level movement action mode. Generated run artifacts remain ignored under
`runs/`; copy the JSON reports and exact command/config into experiment notes
before deleting a run directory.
