# Tetris Version One

Stable, cleaned-up baseline containing the working NES/Modern environments and a native PPO agent.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Play (Human)
```bash
python -m TetrisVersionOne.scripts.play_human --env modern --fps 60
```
Use `--env nes` for the classic ruleset. Modern supports hold, SRS kicks, and 5-piece previews.

## Train (PPO)
```bash
python -m TetrisVersionOne.scripts.train --env modern \
  --total-timesteps 1500000 --num-envs 8 --log-dir runs/ppo_modern_v1
```
Common flags: `--env nes|modern`, `--reward-scale`, `--placement-actions` (macro moves), curriculum options, and reward shaping overrides (e.g. `--agent-reward-weight hole_penalty=0.5`).

## Evaluate
```bash
python -m TetrisVersionOne.scripts.eval runs/ppo_modern_v1/final_model.pt \
  --env modern --render
```

## Tests
```bash
pytest
```

## Layout
- `TetrisVersionOne/env/` – NES + Modern environments, wrappers, curriculum, presets.
- `TetrisVersionOne/agents/ppo/` – PPO implementation, training/eval CLIs.
- `TetrisVersionOne/scripts/` – convenience wrappers (`train`, `eval`, `play_human`).
- `TetrisVersionOne/tests/` – environment and planner tests.
- `TetrisVersionOne/presets/` – sample board presets for curricula.
