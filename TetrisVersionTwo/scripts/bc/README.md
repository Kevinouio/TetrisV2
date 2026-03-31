# Top-1 Behavioral Cloning Baseline

This package implements a clean supervised imitation baseline using the existing
`tetris_v2_c_api` environment + Cold Clear expert.

## What it does

- Collects `(state, expert_action)` pairs from Cold Clear rollouts.
- Uses a canonical action tuple: `(use_hold, piece, rotation, x, y)`.
- Trains a PyTorch classifier to predict the expert's top-1 action class.
- Evaluates:
  - offline top-1/top-5 action prediction accuracy
  - online gameplay with legality-masked argmax policy
- Provides inference wrapper (`BCAgent`) for environment loops.

## Assumptions

- C++ env and bot are already implemented and exposed via `tetris_cc_*` C API.
- Legal placements can be enumerated through `tetris_cc_env_placement_*`.
- Cold Clear action can be queried through `tetris_cc_bot_choose_and_apply*`.

## Commands

```bash
python -m bc.collect_data --num_episodes 5000 --out_dir data/bc_top1 --lib <path-to-lib>
python -m bc.train --data_dir data/bc_top1 --out_dir runs/bc_top1
python -m bc.evaluate --checkpoint runs/bc_top1/best.pt --data_dir data/bc_top1 --split test
python -m bc.evaluate --checkpoint runs/bc_top1/best.pt --play_games 100 --lib <path-to-lib>
```

## Files

- `collect_data.py`: rollout collection + vocab build + dataset sharding
- `dataset.py`: `.pt` shard loading for train/val/test
- `encoders.py`: deterministic state encoding
- `model.py`: small CNN + MLP action classifier (`BCPolicyNet`)
- `train.py`: supervised training loop with early stopping/checkpointing
- `evaluate.py`: offline + online evaluation
- `inference_agent.py`: legality-masked inference wrapper
- `utils.py`: C API adapter, action codec, split helpers

