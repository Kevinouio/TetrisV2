# Training guide

The recommended quality path is:

1. Generate Cold Clear demonstrations.
2. Warm-start the placement-aware DQN network.
3. Collect DAgger states with that network and label them with Cold Clear.
4. Warm-start again on both datasets.
5. Apply a short hybrid TD-learning fine-tune and accept a checkpoint only if
   it passes the multi-seed evaluation gate.

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

## 1. Generate the base expert dataset

```bash
tetris-generate-expert-data \
  --output-dir runs/expert_base \
  --episodes 200 \
  --max-steps 300 \
  --think-ms 1 \
  --random-action-prob 0.05 \
  --seed 20000
```

The small amount of off-teacher behavior exposes the model to recoverable
mistakes. Increase `--think-ms` when stronger expert search matters more than
generation speed.

## 2. Warm-start DQN

```bash
tetris-pretrain-dqn \
  --dataset-dir runs/expert_base \
  --updates 20000 \
  --batch-size 256 \
  --checkpoint-frequency 5000 \
  --log-dir runs/dqn_pretrain
```

This automatically selects the placement-aware convolutional network for the
current 254-feature observation and 3,200-action schema.

## 3. Collect learner-state DAgger data

```bash
tetris-generate-expert-data \
  --output-dir runs/dagger_1 \
  --episodes 300 \
  --max-steps 150 \
  --think-ms 1 \
  --behavior-checkpoint runs/dqn_pretrain/dqn_expert_pretrain.pt \
  --teacher-action-prob 0.05 \
  --random-action-prob 0.02 \
  --seed 40000
```

The learner chooses the visited states while Cold Clear supplies the training
label for every state. Then continue supervised training on the base and
DAgger datasets:

```bash
tetris-pretrain-dqn \
  --dataset-dir runs/expert_base \
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

## 4. Select a warm-start checkpoint

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

## 5. Hybrid DQN fine-tune

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

## PPO and DQN from scratch

Both algorithms use the same observation, stable action IDs, legal-action
masks, shaped reward, and one-placement timestep:

```bash
tetris-train-ppo \
  --total-timesteps 1000000 \
  --num-envs 8 \
  --max-steps 300 \
  --log-dir runs/ppo

tetris-train-dqn \
  --total-timesteps 1000000 \
  --max-steps 300 \
  --log-dir runs/dqn_from_scratch
```

These are useful experiment entry points, but the repository's demonstrated
quality gate is for expert-assisted hybrid DQN. Evaluate either algorithm with
`tetris-eval` before using its checkpoint.

## Play a checkpoint

```bash
tetris-play-rl runs/dqn_hybrid/dqn_hybrid_final.pt --algo dqn

tetris-play-rl-cli runs/dqn_hybrid/dqn_hybrid_final.pt \
  --algo dqn \
  --render-board \
  --delay-ms 50
```

Use `--stochastic` for exploratory playback. Deterministic greedy playback is
the default and is the mode used by the quality gate.

## Proven workspace result

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
actions. Older eight-action and 451-feature/97-action datasets and checkpoints
cannot be loaded into the current environment; regenerate data and retrain
instead.
