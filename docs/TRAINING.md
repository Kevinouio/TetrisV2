# Training guide

TetrisV2 provides three masked placement learners:

- DQN, including the existing expert/DAgger hybrid path.
- Structured PPO with a placement-convolution actor and a separate value
  network.
- Discrete Flow-DQN, a full-map PyTorch adaptation inspired by
  [Flow Q-Learning](https://arxiv.org/abs/2502.02538).

All three use the same 254-feature observation and the same `8 x 40 x 10`
action map, flattened in `(hold, rotation, y, x)` order. PPO and DQN can use
schema-v2 label datasets for behavioral cloning. Flow-DQN needs schema-v3
transitions because it also trains Bellman critics.

One environment step is one locked piece, including decisions that use Hold.
Therefore `--max-steps 150` means at most 150 placements.

## Prerequisites

From the repository root:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DTETRIS_STRICT_WARNINGS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure

python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

The commands below auto-select CUDA when PyTorch can use it. Pass
`--device cpu` to force CPU, or `--device cuda` to require a GPU.

## Hydra training presets

Use `tetris-train` for normal training. The preset YAML files hold the full
argument sets, so a run only needs an experiment name and, optionally, a
runtime choice. `experiment` is required; `tetris-train --help` lists every
available preset:

```bash
# Flow-DQN's validated two-stage path
tetris-train experiment=flow_offline runtime=cuda
tetris-train experiment=flow_online runtime=cuda

# Structured PPO
tetris-train experiment=ppo_pretrain runtime=cuda
tetris-train experiment=ppo_dagger_pretrain runtime=cuda
tetris-train experiment=ppo_finetune runtime=cuda

# DQN
tetris-train experiment=dqn_pretrain runtime=cuda
tetris-train experiment=dqn_dagger_pretrain runtime=cuda
tetris-train experiment=dqn_hybrid runtime=cuda
```

The DAgger presets expect the learner-state datasets described below to exist.
The pure-RL presets are `dqn_from_scratch`, `ppo_from_scratch`, and
`flow_from_scratch`.

Configuration lives under [`tetris_v2/conf`](../tetris_v2/conf). Each trainer
group contains every argument accepted by its compatibility CLI, while the
experiment files contain the values for a particular training stage. Inspect
a fully composed job without running it:

```bash
tetris-train experiment=flow_offline runtime=cpu dry_run=true
tetris-train experiment=flow_offline --cfg job
```

To add a reusable project experiment, copy the closest file in
`tetris_v2/conf/experiment/`, change `preset.name`, and edit only its
`trainer.args` overrides. The new filename becomes its `experiment=<name>`.

Override only the values that make the experiment different:

```bash
tetris-train experiment=flow_offline \
  runtime=cuda \
  trainer.args.offline_updates=25000 \
  trainer.args.log_dir=runs/flow_debug
```

Hydra also makes the four Flow pilot coefficients a concise multirun. Each job
gets a distinct log directory through config interpolation:

```bash
tetris-train -m \
  experiment=flow_pilot \
  'trainer.args.distill_q_coef=0.1,0.3,1.0,3.0' \
  runtime=cuda
```

`runtime=auto` leaves device selection to PyTorch, and `runtime=cpu|cuda`
forces it. Relative dataset, checkpoint, library, and output paths are resolved
from the directory where `tetris-train` was invoked. Hydra keeps its composed
config snapshots under ignored `runs/hydra/`; model artifacts still go to the
preset's `trainer.args.log_dir`. The existing algorithm-specific commands
remain supported and are shown below as manual equivalents and CLI reference.

## Shared expert transition dataset

```bash
tetris-generate-expert-data \
  --output-dir runs/v4_expert_transitions \
  --episodes 500 \
  --max-steps 300 \
  --think-ms 1 \
  --random-action-prob 0.05 \
  --seed 100000
```

The small amount of off-teacher behavior exposes the model to recoverable
mistakes. Increase `--think-ms` when stronger expert search matters more than
generation speed. New shards use schema v3 and contain the teacher action,
executed action, shaped/raw reward, next observation/mask, and termination
flags.

## DQN expert/DAgger path

### 1. Warm-start DQN

```bash
tetris-pretrain-dqn \
  --dataset-dir runs/v4_expert_transitions \
  --updates 20000 \
  --batch-size 256 \
  --checkpoint-frequency 5000 \
  --log-dir runs/dqn_pretrain
```

This automatically selects the placement-aware convolutional network for the
current 254-feature observation and 3,200-action schema.

### 2. Collect learner-state DAgger data

```bash
tetris-generate-expert-data \
  --output-dir runs/dagger_1 \
  --episodes 300 \
  --max-steps 150 \
  --think-ms 1 \
  --behavior-checkpoint runs/dqn_pretrain/dqn_expert_pretrain.pt \
  --behavior-algo dqn \
  --teacher-action-prob 0.05 \
  --random-action-prob 0.02 \
  --seed 40000
```

The learner chooses the visited states while Cold Clear supplies the training
label for every state. Then continue supervised training on the base and
DAgger datasets:

```bash
tetris-pretrain-dqn \
  --dataset-dir runs/v4_expert_transitions \
  --extra-dataset-dir runs/dagger_1 \
  --extra-dataset-dir runs/dagger_1 \
  --init-checkpoint runs/dqn_pretrain/dqn_expert_pretrain.pt \
  --updates 15000 \
  --batch-size 256 \
  --learning-rate 0.001 \
  --checkpoint-frequency 3000 \
  --log-dir runs/dqn_dagger_1
```

Repeating `--extra-dataset-dir` deliberately gives the learner-state data more
sampling weight. Evaluate the saved checkpoints instead of assuming the last
one is best; supervised performance can regress after too many updates.

### 3. Select a warm-start checkpoint

For example, test a saved checkpoint on seeds that were not used to generate
the data:

```bash
tetris-eval runs/dqn_dagger_1/dqn_expert_pretrain_step_12000.pt \
  --algo dqn \
  --episodes 50 \
  --seed 50000 \
  --max-steps 300 \
  --min-placements 101 \
  --min-lines 20 \
  --json-output runs/dqn_dagger_1/eval.json
```

The command exits nonzero if any episode fails either threshold. Select the
checkpoint with the strongest held-out minimum, not merely the highest mean.

### 4. Hybrid DQN fine-tune

Replace the checkpoint below if a different saved step won validation:

```bash
tetris-train-dqn-hybrid \
  --offline-dataset-dir runs/dagger_1 \
  --init-checkpoint runs/dqn_dagger_1/dqn_expert_pretrain_step_12000.pt \
  --total-timesteps 10000 \
  --buffer-size 20000 \
  --warmup-steps 512 \
  --batch-size 128 \
  --train-frequency 4 \
  --learning-rate 0.0001 \
  --epsilon-start 0.02 \
  --epsilon-end 0.01 \
  --epsilon-decay-steps 10000 \
  --max-steps 300 \
  --expert-think-ms 1 \
  --online-expert-interval 1 \
  --lambda-bc-start 1.0 \
  --lambda-bc-end 1.0 \
  --lambda-pair-start 0 \
  --lambda-pair-end 0 \
  --pairs-per-sample 0 \
  --checkpoint-frequency 2500 \
  --eval-frequency 2500 \
  --eval-episodes 10 \
  --log-dir runs/dqn_hybrid
```

Run the strict gate on independent seeds:

```bash
tetris-eval runs/dqn_hybrid/dqn_hybrid_final.pt \
  --algo dqn \
  --episodes 100 \
  --seed 70000 \
  --max-steps 150 \
  --min-placements 101 \
  --min-lines 20 \
  --json-output runs/dqn_hybrid/eval_100_seeds.json
```

## Structured PPO path

PPO automatically selects its structured placement actor for the current
schema. The value network has separate parameters, while old flat-MLP PPO
checkpoints continue to load as legacy networks.

### 1. Pretrain the actor

```bash
tetris-pretrain-ppo \
  --dataset-dir runs/v4_expert_transitions \
  --updates 20000 \
  --batch-size 512 \
  --checkpoint-frequency 5000 \
  --log-dir runs/ppo_pretrain
```

### 2. Collect PPO learner states and continue pretraining

```bash
tetris-generate-expert-data \
  --output-dir runs/ppo_dagger_1 \
  --episodes 300 \
  --max-steps 300 \
  --think-ms 1 \
  --behavior-checkpoint runs/ppo_pretrain/ppo_expert_pretrain.pt \
  --behavior-algo ppo \
  --teacher-action-prob 0.05 \
  --random-action-prob 0.02 \
  --seed 150000

tetris-pretrain-ppo \
  --dataset-dir runs/v4_expert_transitions \
  --extra-dataset-dir runs/ppo_dagger_1 \
  --extra-dataset-dir runs/ppo_dagger_1 \
  --init-checkpoint runs/ppo_pretrain/ppo_expert_pretrain.pt \
  --updates 15000 \
  --batch-size 512 \
  --checkpoint-frequency 3000 \
  --log-dir runs/ppo_dagger_pretrain
```

### 3. Fine-tune on-policy with annealed imitation

```bash
tetris-train-ppo \
  --init-checkpoint runs/ppo_dagger_pretrain/ppo_expert_pretrain.pt \
  --expert-dataset-dir runs/v4_expert_transitions \
  --extra-expert-dataset-dir runs/ppo_dagger_1 \
  --extra-expert-dataset-dir runs/ppo_dagger_1 \
  --total-timesteps 1000000 \
  --num-envs 8 \
  --n-steps 1024 \
  --minibatch-size 1024 \
  --bc-coef-start 1.0 \
  --bc-coef-end 0.1 \
  --bc-anneal-timesteps 1000000 \
  --max-steps 300 \
  --seed 300000 \
  --log-dir runs/ppo_finetune
```

Training records approximate KL, clip fraction, explained variance, and BC
agreement. It writes `ppo_best.pt` using minimum placements, minimum lines,
mean lines, then mean return as the checkpoint ordering. Use
`--resume-checkpoint` to continue optimizer/counter state; use
`--init-checkpoint` to copy weights into a fresh run. A resumed annealed run
continues from its restored learning rate and anneals that value to zero over
the newly requested remaining timesteps; rollout sampling streams restart.

## Discrete Flow-DQN path

Flow-DQN learns twin structured Q maps, a time-conditioned flow over the full
3,200-value placement map, and a one-step actor distilled from ten Euler flow
steps. Deterministic inference uses zero source noise and masked argmax;
stochastic inference scales the Gaussian latent by temperature and then uses
the same masked argmax, without a second categorical draw. This discrete design
is inspired by the paper and the authors'
[official MIT implementation](https://github.com/seohongpark/fql), but the
continuous-action FQL algorithm itself cannot directly select Tetris actions.

The uniform-time flow-matching objective is retained. Two discrete auxiliaries
prevent a one-hot placement map from collapsing to a nearly uniform policy: a
masked executed-action loss trains the flow field at `t=0`, and a second masked
executed-action loss trains the distilled one-step actor directly. Their
defaults are controlled by `--flow-t0-ce-coef 1.0` and `--actor-bc-coef 1.0`.
`--action-logit-scale 10` converts the learned map values into useful masked
categorical training logits for behavioral cloning and Q guidance. These
labels are simply replay-buffer actions in pure-RL mode, so expert data remains
optional. Training logs `t0_acc`, `bc_acc`, and masked entropy; use them to
confirm that the flow and actor are learning the behavior distribution before
starting a long run.

### 1. Select the normalized-Q distillation coefficient

Run a 25,000-update offline pilot with the one-step distillation coefficient
set to each of `0.1`, `0.3`, `1.0`, and `3.0`. Q guidance remains normalized
and its multiplier remains `1.0`:

```bash
tetris-train-flow-dqn \
  --offline-dataset-dir runs/v4_expert_transitions \
  --offline-updates 25000 \
  --online-timesteps 0 \
  --distill-q-coef 0.3 \
  --q-guidance-coef 1.0 \
  --flow-t0-ce-coef 1.0 \
  --actor-bc-coef 1.0 \
  --action-logit-scale 10 \
  --normalized-q \
  --checkpoint-frequency 25000 \
  --log-dir runs/flow_pilot_v2_0.3

tetris-eval runs/flow_pilot_v2_0.3/flow_dqn_final.pt \
  --algo flow_dqn \
  --episodes 50 \
  --seed 200000 \
  --max-steps 150 \
  --json-output runs/flow_pilot_v2_0.3/eval_seed200000_50.json
```

Repeat with the other three values and select by minimum placements, minimum
lines, then mean lines. The corrected workspace pilots produced:

| Coefficient | Minimum placements | Minimum lines | Mean lines |
|---:|---:|---:|---:|
| 0.1 | 150 | 49 | 55.88 |
| 0.3 | 78 | 24 | 54.86 |
| 1.0 | 71 | 19 | 54.42 |
| 3.0 | 150 | 50 | 55.12 |

Coefficient `3.0` wins the stated lexicographic ordering. The earlier
`flow_pilot_*` artifacts predate the two discrete executed-action losses and
are retained only as diagnostics; use the `flow_pilot_v2_*` results.

### 2. Train the selected configuration offline-to-online

The workspace run continued the winning pilot for another 175,000 offline
updates, bringing its counter to 200,000:

```bash
tetris-train-flow-dqn \
  --offline-dataset-dir runs/v4_expert_transitions \
  --resume-checkpoint runs/flow_pilot_v2_3.0/flow_dqn_final.pt \
  --offline-updates 175000 \
  --online-timesteps 0 \
  --distill-q-coef 3.0 \
  --checkpoint-frequency 25000 \
  --log-dir runs/flow_dqn_full
```

A fresh run can omit `--resume-checkpoint` and use `--offline-updates 200000`.
The online phase resumes the complete network, optimizer, schema, and counter
state:

```bash
tetris-train-flow-dqn \
  --offline-dataset-dir runs/v4_expert_transitions \
  --resume-checkpoint runs/flow_dqn_full/flow_dqn_final.pt \
  --offline-updates 0 \
  --online-timesteps 500000 \
  --online-offline-fraction 0.9 \
  --exploration-temperature 0.25 \
  --warmup-steps 512 \
  --buffer-size 100000 \
  --batch-size 256 \
  --train-frequency 1 \
  --gradient-steps 1 \
  --max-steps 300 \
  --seed 360000 \
  --device cuda \
  --checkpoint-frequency 25000 \
  --eval-frequency 25000 \
  --eval-episodes 5 \
  --log-interval 5000 \
  --log-dir runs/flow_dqn_online_safe
```

The 90% expert-transition update mix and `0.25` Gaussian-latent temperature
were the stable online configuration. The completed checkpoint records 200,000
offline updates, 500,000 environment steps, and 499,489 online gradient updates
after the 512-step replay warmup. Keep periodic checkpoints: held-out selection
may prefer an earlier boundary even when the full schedule completes.

If the final gate misses, run one learner-state remediation round:

```bash
tetris-generate-expert-data \
  --output-dir runs/flow_dagger_retry_1 \
  --episodes 300 \
  --max-steps 300 \
  --think-ms 1 \
  --behavior-checkpoint runs/flow_dqn_full/flow_dqn_final.pt \
  --behavior-algo flow_dqn \
  --teacher-action-prob 0.05 \
  --random-action-prob 0.02 \
  --seed 600000

tetris-train-flow-dqn \
  --offline-dataset-dir runs/flow_dagger_retry_1 \
  --offline-updates 100000 \
  --online-timesteps 250000 \
  --init-checkpoint runs/flow_dqn_full/flow_dqn_final.pt \
  --distill-q-coef 3.0 \
  --max-steps 300 \
  --seed 650000 \
  --log-dir runs/flow_dqn_retry_1
```

Use the coefficient selected by the pilot, then run the final gate on a new
100-seed block rather than reusing the failed block.

`--resume-checkpoint` restores all learned networks, optimizers, and counters;
the in-memory online replay buffer is intentionally rebuilt and must warm up
again after a restarted process.

## PPO remediation round

If PPO's first million-step fine-tune misses the gate, collect and upweight one
new 300-episode learner-state dataset, then perform another supervised refresh
and 250,000 online timesteps:

```bash
tetris-generate-expert-data \
  --output-dir runs/ppo_dagger_retry_1 \
  --episodes 300 \
  --max-steps 300 \
  --think-ms 1 \
  --behavior-checkpoint runs/ppo_finetune/ppo_best.pt \
  --behavior-algo ppo \
  --teacher-action-prob 0.05 \
  --random-action-prob 0.02 \
  --seed 700000

tetris-pretrain-ppo \
  --dataset-dir runs/v4_expert_transitions \
  --extra-dataset-dir runs/ppo_dagger_retry_1 \
  --extra-dataset-dir runs/ppo_dagger_retry_1 \
  --init-checkpoint runs/ppo_finetune/ppo_best.pt \
  --updates 15000 \
  --log-dir runs/ppo_retry_pretrain_1

tetris-train-ppo \
  --init-checkpoint runs/ppo_retry_pretrain_1/ppo_expert_pretrain.pt \
  --expert-dataset-dir runs/v4_expert_transitions \
  --extra-expert-dataset-dir runs/ppo_dagger_retry_1 \
  --extra-expert-dataset-dir runs/ppo_dagger_retry_1 \
  --total-timesteps 250000 \
  --bc-coef-start 1.0 \
  --bc-coef-end 0.1 \
  --bc-anneal-timesteps 250000 \
  --max-steps 300 \
  --seed 750000 \
  --log-dir runs/ppo_retry_1
```

## Pure-RL experiments

Expert datasets are optional for all online trainers:

```bash
tetris-train-ppo --total-timesteps 1000000 --log-dir runs/ppo_from_scratch
tetris-train-dqn --total-timesteps 1000000 --log-dir runs/dqn_from_scratch
tetris-train-flow-dqn \
  --offline-updates 0 \
  --online-timesteps 500000 \
  --log-dir runs/flow_dqn_from_scratch
```

The expert-assisted configurations are the intended reliability paths.

## Final acceptance gate

Use seeds that were not used for generation, training, development evaluation,
or an earlier final retry. For PPO, for example:

```bash
tetris-eval runs/ppo_finetune/ppo_best.pt \
  --algo ppo \
  --episodes 100 \
  --seed 400000 \
  --max-steps 150 \
  --min-placements 101 \
  --min-lines 20 \
  --json-output runs/ppo_finetune/eval_100_seeds.json
```

For Flow-DQN, change the checkpoint and use `--algo flow_dqn` with a disjoint
100-seed block:

```bash
tetris-eval runs/flow_dqn_full/flow_dqn_final.pt \
  --algo flow_dqn \
  --episodes 100 \
  --seed 500100 \
  --max-steps 150 \
  --min-placements 101 \
  --min-lines 20 \
  --deterministic \
  --json-output runs/flow_dqn_full/eval_final_seed500100_100.json
```

The command exits nonzero if any seed misses either threshold or produces an
illegal action. Evaluation report schema v2 records each episode's index, seed,
placements, lines, return, top-out, truncation, and illegal-action count. Its
summary includes the placement/line distributions, mean return, top-out and
truncation rates, and total illegal actions; the gate records failed episodes
and `max_illegal_actions=0`.

## Play a checkpoint

```bash
tetris-play-rl runs/dqn_hybrid/dqn_hybrid_final.pt --algo dqn

tetris-play-rl-cli runs/dqn_hybrid/dqn_hybrid_final.pt \
  --algo dqn \
  --render-board \
  --delay-ms 50

tetris-play-rl runs/ppo_finetune/ppo_best.pt --algo ppo
tetris-play-rl runs/flow_dqn_full/flow_dqn_final.pt --algo flow_dqn
```

Use `--stochastic` for exploratory playback. Deterministic greedy playback is
the default and is the mode used by the quality gate.

## Proven workspace result

The structured PPO checkpoint at `runs/ppo_finetune/ppo_best.pt` completed the
full expert-assisted schedule and passed all 100 final seeds beginning at
`400000` with a 150-placement cap. Every episode reached 150 placements; the
minimum was 51 lines, the mean was 56.35 lines, mean return was 114.21, and
there were zero top-outs and zero illegal actions.
The full per-seed report is
`runs/ppo_finetune/eval_final_seed400000_100.json`.

Flow-DQN completed the 200,000-offline-update and 500,000-online-step schedule.
The 200,000-update offline boundary at
`runs/flow_dqn_full/flow_dqn_final.pt` won a separate 100-seed development
rerank over the strongest online snapshots, so it was retained as the best
checkpoint. It passed the disjoint final seeds `500100` through `500199`:
minimum 143 placements, minimum 40 lines, mean 149.93 placements and 54.91
lines, mean return 160.70, and zero illegal actions. One episode topped out
after 143 placements and 40 lines; the other 99 reached the 150-placement cap.
The schema-v2 report is
`runs/flow_dqn_full/eval_final_seed500100_100.json`.

For transparency, the 500,000-step online endpoint missed the preceding final
block because one seed topped out at 25 placements and 4 lines. That failed
block was preserved at
`runs/flow_dqn_online_safe/eval_final_seed500000_100.json` and was not reused.

The prepared checkpoint at
`runs/v3_hybrid_finetune/dqn_hybrid_final.pt` passed all 100 independent seeds
at a 150-placement cap: minimum 150 placements, minimum 50 lines, mean 56.84
lines, and zero top-outs. At a 300-placement cap across another 50 seeds, its
minimum was 211 placements and 79 lines.

The strongest supervised checkpoint is
`runs/v3_conv_dagger_pretrain_round1/dqn_expert_pretrain_step_12000.pt`. It
reached the 300-placement cap on all 50 independent seeds, with at least 111
lines and zero top-outs. The JSON reports beside those checkpoints contain the
full per-seed results.

Generated datasets and checkpoints under `runs/` are intentionally ignored by
Git. New clones must train them again or copy the desired artifacts separately.

## Compatibility

Current checkpoints use 254 observation features and 3,200 stable placement
actions. Legacy MLP PPO checkpoints still load when their matching environment
schema is available. Schema-v2 current-action datasets remain valid for PPO
and DQN cloning, but Flow-DQN rejects them with a regeneration message because
they do not contain Bellman transitions. Older eight-action and
451-feature/97-action artifacts do not match the current environment.
