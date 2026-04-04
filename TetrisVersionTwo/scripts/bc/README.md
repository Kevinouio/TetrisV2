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

Loader/runtime tuning knobs:
- `--pin_memory` / `--no-pin_memory`
- `--persistent_workers` / `--no-persistent_workers`
- `--prefetch_factor`
- `--torch_num_threads`, `--torch_num_interop_threads`, `--omp_num_threads`, `--mkl_num_threads`, `--openblas_num_threads`

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
- Thread caps for worker CPU usage:
  - `--torch_num_threads`
  - `--torch_num_interop_threads`
  - `--omp_num_threads`
  - `--mkl_num_threads`
  - `--openblas_num_threads`

Live progress:

- `--progress_mode {console,json,both}` (default `both`)
- `--progress_every_sec` (default `2.0`)
- `--progress_path` (default `<out_dir>/progress.json`)
- `--rss_warn_mb` (warn once on main-process RSS threshold, default disabled)
- `--worker_rss_warn_mb` (warn once on max worker RSS threshold, default disabled)
- When `--viewer` is enabled, progress payloads also include additive viewer health fields:
  - `viewer_events_processed`
  - `viewer_frames_rendered`
  - `viewer_restart_count`
  - `viewer_last_frame_age_sec`

Live pygame viewer (disabled by default):

- `--viewer`
- `--viewer_fullscreen` / `--no-viewer_fullscreen` (default: windowed / no-fullscreen)
- `--viewer_fps`
- `--viewer_publish_every_steps` (auto defaults to `1` when `--viewer` is enabled and flag is unset)
- `--viewer_compact_telemetry` / `--no-viewer_compact_telemetry`
- `--viewer_board_every_steps` (auto defaults to `1` when `--viewer` is enabled and flag is unset)
- `--viewer_max_queue`
- `--viewer_grid_padding`
- `--viewer_min_tile_px`
- `--viewer_agent`
- `--viewer_reopen_file` (default sibling `VIEWER_OPEN` near progress output)

Viewer controls:
- `Click`/`Arrow keys`/`Tab` to select worker card
- `PageUp`/`PageDown` or `[`/`]` to change pages when workers overflow one screen
- `F11` fullscreen toggle, `R` reset focus, `Q` close viewer

Worker identity:
- Worker cards are PID-based (`PID <pid>`).
- If a worker respawns, it appears as a new PID card.
- Stale PID cards are pruned after inactivity.

Example:

```bash
python -m bc.collect_data \
  --lib build-wsl-rel/TetrisVersionTwo/libtetris_v2_c_api.so \
  --num_episodes 2000 \
  --collect_workers 4 \
  --viewer
```

Viewer overhead note:
- With `--viewer` enabled, the default is per-step publishing for full live playback.
- For lower overhead, explicitly increase `--viewer_publish_every_steps` and `--viewer_board_every_steps`.
- Keep `--viewer_compact_telemetry` enabled when tuning for throughput.
- `--viewer_max_queue` bounds IPC memory; overflow drops telemetry events (collection continues).
- With `--viewer` off, collection no longer imports `pygame`, reducing headless startup/memory overhead.
- Collection now spools episode records to temporary files and streams shard builds to avoid unbounded in-memory growth on long runs.

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

## DAgger Collection Viewer

`bc.dagger` supports the same live viewer flags during each round collection step:

```bash
python -m bc.dagger \
  --base_data_dir data/bc_top1 \
  --run_dir runs/bc_dagger \
  --lib build-wsl-rel/TetrisVersionTwo/libtetris_v2_c_api.so \
  --collect_workers 4 \
  --viewer
```

The DAgger viewer additionally surfaces round telemetry (`round_id`, `beta`, expert/learner step split, learner fallback counters).

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

## Storage Cleanup

Use the cleanup utility to reclaim space from old DAgger artifacts while keeping
latest rounds and runnable checkpoints.

Dry-run (default, no deletion):

```bash
python -m bc.cleanup_data
```

Apply deletions:

```bash
python -m bc.cleanup_data --apply
```

Optional aggressive mode (also prune old round `aggregated_data` files):

```bash
python -m bc.cleanup_data --apply --prune_old_aggregated_data
```

Defaults:

- dry-run mode unless `--apply` is passed
- keep latest 2 rounds per DAgger run
- preserve base dataset path (`data/bc_top1`)
- prune old `round_XX/dagger_train/shards/*.pt` first
- optional JSON report via `--json_report <path>`

## Module Map

- `collect_data.py`: rollout collection + vocab build + sharding
- `dataset.py`: shard loading for train/val/test
- `encoders.py`: deterministic state encoding
- `model.py`: `BCPolicyNet`
- `train.py`: supervised training + checkpoints + metrics
- `evaluate.py`: offline and online evaluation
- `inference_agent.py`: legality-masked inference wrapper (`BCAgent`)
- `cleanup_data.py`: safe disk cleanup for generated BC/DAgger artifacts
- `utils.py`: C API adapter, action codec, split utilities
