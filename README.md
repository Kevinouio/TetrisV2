# TetrisV2

Scaffolding for a reinforcement-learning project that starts with an NES-style Tetris
environment, then extends to modern Tetris rules, and finally to an adversarial setup.

## Layout
- `tetris_v2/` Python package
  - `envs/`: Gymnasium environments (`NES` first, then modern)
  - `agents/`: training scripts and model definitions (DQN/PPO/etc.)
  - `adversarial/`: two-player/self-play logic and versus envs
  - `configs/`: YAML configs for experiments
  - `scripts/`: CLI utilities for training, eval, and human play
  - `tests/`: unit tests
  - `notebooks/`: exploration and analysis

## Quickstart (after you implement stubs)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Human Play / UI
```bash
python -m tetris_v2.scripts.play_human --env modern --fps 60
```
- NES rules: `--env nes`
- Modern rules: `--env modern`
- Versus (random opponent): `--env versus`

Controls (real-time, DAS/ARR aware):
- Modern / Versus:
  - Move Left `a`, Move Right `d`
  - Soft Drop `Shift`, Hard Drop `Space`
  - Rotate Left `←`, Rotate Right `→`, Rotate 180 `↑`
  - Hold `c`
- NES keeps the classic bindings (`←/→` move, `↑` rotate, `Space` hard drop, `S`/`↓` soft drop)
- `R`: reset, `Esc`/`Q`: quit

Want different keys? Edit the `KEY_BINDINGS` dictionary near the top of `tetris_v2/scripts/play_human.py` to remap any action (bindings are grouped per ruleset profile). Save the file and rerun the script to pick up the new layout.

## Milestones
1. Implement `NesTetrisEnv` under `tetris_v2/envs/nes_tetris_env.py` (Gymnasium)
2. Add human play loop in `tetris_v2/scripts/play_human.py`
3. Write simple SB3 trainer in `tetris_v2/scripts/train_nes.py`
4. Extend to modern Tetris in `tetris_v2/envs/modern_tetris_env.py`
5. Build adversarial env in `tetris_v2/adversarial/`
