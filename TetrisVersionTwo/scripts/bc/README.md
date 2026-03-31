# Top-1 Behavioral Cloning (Cold Clear Expert)

This module implements a reproducible supervised imitation baseline for Tetris:
predict Cold Clear's top-1 action from the current state.

The implementation lives in `TetrisVersionTwo/scripts/bc/`, with root shims so
you can run commands as `python -m bc.<tool>`.

## What This Baseline Does

- Collects expert demonstrations from Cold Clear.
- Encodes state with:
  - board occupancy `[1, 20, 10]`
  - current piece one-hot
  - hold one-hot (with empty-hold class)
  - next-queue one-hots (default length 5)
  - optional scalars (`lines`, `combo`, `b2b`)
- Uses canonical action tuples:
  - `(use_hold, piece, rotation, x, y)`
- Builds an observed action vocabulary and maps:
  - `action_to_id`, `id_to_action`
- Trains `BCPolicyNet` (CNN + MLP) with cross entropy.
- Evaluates:
  - offline classification (top-1/top-5, confusion summary)
  - online gameplay with legality-masked argmax
- Provides `BCAgent` inference wrapper.

## Requirements

- Python 3.10+ recommended
- PyTorch, NumPy
- Built `tetris_v2_c_api` shared library (`.so`/`.dll`/`.dylib`)

Install (example):

```bash
pip install -r requirements.txt
```

## Quick Start (WSL/Linux example)

Collect data:

```bash
python -m bc.collect_data \
  --lib build-wsl/TetrisVersionTwo/libtetris_v2_c_api.so \
  --num_episodes 5000 \
  --out_dir data/bc_top1
```

Train:

```bash
python -m bc.train --data_dir data/bc_top1 --out_dir runs/bc_top1
```

Offline eval:

```bash
python -m bc.evaluate \
  --checkpoint runs/bc_top1/best.pt \
  --data_dir data/bc_top1 \
  --split test
```

Online eval:

```bash
python -m bc.evaluate \
  --checkpoint runs/bc_top1/best.pt \
  --play_games 100 \
  --lib build-wsl/TetrisVersionTwo/libtetris_v2_c_api.so
```

## Data Collection Details

Episode definition:

- One episode = one game until top-out/game-over or `--max_steps_per_episode`
  (default `2000`).

Parallel collection:

- `--collect_workers N` controls process count (default `1`).
- `--worker_chunksize` controls episode scheduling granularity (default `1`).

Live progress:

- `--progress_mode {console,json,both}` (default `both`)
- `--progress_every_sec` (default `2.0`)
- `--progress_path` (default `<out_dir>/progress.json`)

Manual early stop while running:

- `--stop_file` (default `<out_dir>/STOP`)
- Create the file to stop cleanly and still finalize shards/metadata:

```bash
touch data/bc_top1/STOP
```

Outputs:

- `data/bc_top1/shards/*.pt`
- `data/bc_top1/metadata.json`
- `data/bc_top1/progress.json`

## BC Autoplay in Pygame Viewer

`play_pygame.py` can now run BC autoplay directly.

Cold Clear autoplay:

```bash
python TetrisVersionTwo/scripts/play_pygame.py \
  --lib build-wsl/TetrisVersionTwo/libtetris_v2_c_api.so \
  --ai
```

BC autoplay:

```bash
python TetrisVersionTwo/scripts/play_pygame.py \
  --lib build-wsl/TetrisVersionTwo/libtetris_v2_c_api.so \
  --ai \
  --bc-checkpoint runs/bc_top1/best.pt \
  --bc-device cuda
```

Notes:

- If `--bc-checkpoint` is provided, autoplay backend is BC.
- Otherwise autoplay backend is Cold Clear.
- `A` toggles autoplay in both modes.

## Troubleshooting

- Collection is CPU-only by design (Cold Clear/env rollouts + encoding).
- GPU is used in training/inference (`bc.train`, BC eval/inference paths).
- If collection is interrupted with `Ctrl+C`, outputs may be partial.
  Prefer stop-file based shutdown for clean finalization.

## Module Map

- `collect_data.py`: rollout collection + vocab build + sharding
- `dataset.py`: shard loading for train/val/test
- `encoders.py`: deterministic state encoding
- `model.py`: `BCPolicyNet`
- `train.py`: supervised training + checkpoints + metrics
- `evaluate.py`: offline and online evaluation
- `inference_agent.py`: legality-masked inference wrapper (`BCAgent`)
- `utils.py`: C API adapter, action codec, split utilities
