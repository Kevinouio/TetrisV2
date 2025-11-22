# Tetris Version One

Stable baseline containing the working NES/Modern Tetris environments, a native PPO training pipeline, and a minimal set of scripts/tests. The legacy `tetris_v2` tree and experimental code have been archived under `archive/`.

## What’s included (v1)
- Environments (`TetrisVersionOne/env/`)
  - NES ruleset: gravity table, scoring, no hold/kicks, single preview.
  - Modern SRS: 7‑bag, hold, 5-piece preview, kicks, ghost, lock delay, soft‑drop factor.
- PPO agent (`TetrisVersionOne/agents/ppo/`) with rollout buffer, actor‑critic, training/eval CLIs.
- Scripts (`TetrisVersionOne/scripts/`): `train`, `eval`, `play_human`.
- Tests (`TetrisVersionOne/tests/`): env smoke tests + placement planner.
- Presets (`TetrisVersionOne/presets/`): board presets for curriculum stages.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Play (human)
```bash
python -m TetrisVersionOne.scripts.play_human --env modern --fps 60
```
Use `--env nes` for the classic ruleset.

## Train (PPO)
```bash
python -m TetrisVersionOne.scripts.train --env modern \
  --total-timesteps 1500000 --num-envs 8 --log-dir runs/ppo_modern_v1
```
Common flags: `--env nes|modern`, `--reward-scale`, `--placement-actions`, curriculum options, and reward shaping overrides (e.g. `--agent-reward-weight hole_penalty=0.5`).

## Evaluate
```bash
python -m TetrisVersionOne.scripts.eval runs/ppo_modern_v1/final_model.pt \
  --env modern --render
```

## Tests
```bash
pytest
```

## Roadmap toward “v2”
The next major line (“v2”) will focus on imitation + flow-based methods:
- Imitation learning with flow matching.
- FlowQ / flow-based Q-learning, trained on trajectories from a strong Cold Clear agent.
- Integrations will build atop the Version One envs and PPO baselines.

## License
See `LICENSE`.
