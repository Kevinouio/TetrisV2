# DQN Ref Baseline (`dqn_ref`)

Faithful `Tetris-A.I/Version2`-style baseline ported to `TetrisVersionTwo` indexed placement actions.

## What It Is
- Afterstate candidate scorer (not global-action Q head).
- 6 handcrafted afterstate features.
- MLP `6 -> 32 -> 32 -> 32 -> 1`.
- Prioritized replay + target network.
- Version2 reward shaping and GA-style training loop.

## Main Command
```bash
python -m dqn_ref.train \
  --lib build-wsl/TetrisVersionTwo/libtetris_v2_c_api.so \
  --run_dir runs/dqn_ref
```

## Parallel Agent Evaluation
```bash
python -m dqn_ref.train \
  --lib build-wsl/TetrisVersionTwo/libtetris_v2_c_api.so \
  --run_dir runs/dqn_ref_parallel \
  --agent_workers 8
```

If your machine has CUDA libraries but you want pure CPU parallel workers, you can also pin:

```bash
CUDA_VISIBLE_DEVICES="" python -m dqn_ref.train \
  --lib build-wsl/TetrisVersionTwo/libtetris_v2_c_api.so \
  --run_dir runs/dqn_ref_parallel \
  --device cpu \
  --agent_workers 8
```

## Fullscreen Live Viewer (Esports Dashboard)
```bash
CUDA_VISIBLE_DEVICES="" python -m dqn_ref.train \
  --lib build-wsl/TetrisVersionTwo/libtetris_v2_c_api.so \
  --run_dir runs/dqn_ref_viewer \
  --device cpu \
  --agent_workers 8 \
  --viewer \
  --viewer_fullscreen \
  --viewer_fps 20 \
  --viewer_publish_every_steps 10
```

## Pygame Playback in Shared Env Viewer
Run DQN in the same `play_pygame.py` runtime path used by BC:

```bash
python TetrisVersionTwo/scripts/play_pygame.py \
  --lib build-wsl/TetrisVersionTwo/libtetris_v2_c_api.so \
  --ai \
  --dqn-checkpoint runs/dqn_ref_viewer/model/best_model.pt \
  --dqn-device cpu
```

`--bc-checkpoint` and `--dqn-checkpoint` are mutually exclusive.

Viewer notes:
- Boards are rendered with per-piece colors (`I/O/T/L/J/S/Z`) from live `board_piece_ids` telemetry.
- Press `Q` to close the viewer while training keeps running.
- Reopen during the same run by creating the trigger file (default `VIEWER_OPEN` under `run_dir`):

```bash
touch runs/dqn_ref_viewer/VIEWER_OPEN
```

- You can override the trigger path with `--viewer_reopen_file <path>`.

## WSL + Conda (`Tetris`) Command
```bash
wsl bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate Tetris && cd /mnt/c/Users/kevin/Desktop/GithubProjects/TetrisV2 && python -m dqn_ref.train --lib build-wsl/TetrisVersionTwo/libtetris_v2_c_api.so --run_dir runs/dqn_ref"
```

## Smoke Run
```bash
python -m dqn_ref.train \
  --lib build-wsl/TetrisVersionTwo/libtetris_v2_c_api.so \
  --run_dir runs/dqn_ref_smoke \
  --smoke
```

## Key Outputs
- `runs/dqn_ref/config.json`
- `runs/dqn_ref/episode_metrics.csv`
- `runs/dqn_ref/generation_metrics.csv`
- `runs/dqn_ref/summary.json`
- `runs/dqn_ref/model/best_model.pt`

## Performance Benchmarks
```bash
python -m dqn_ref.bench_candidates \
  --lib build-wsl/TetrisVersionTwo/libtetris_v2_c_api.so \
  --mode compare
```

```bash
python -m dqn_ref.bench_throughput \
  --lib build-wsl/TetrisVersionTwo/libtetris_v2_c_api.so \
  --mode auto
```

See `OPTIMIZATION_NOTES.md` for the benchmark workflow and before/after table template.
