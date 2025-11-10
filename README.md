# TetrisV2 

TetrisV2 is a practical playground for building state‑of‑the‑art Tetris agents.
It provides faithful environments (NES, modern/SRS, and versus), a real‑time
human UI, and a clear roadmap to implement a broad family of reinforcement‑
learning algorithms—from value‑based baselines to planning methods like
AlphaZero and MuZero.

The goal is twofold:
- Ship strong, reproducible RL baselines that play excellent Tetris.
- Serve as a teaching/reference repo for implementing algorithms “from scratch”
  and adapting them to a non‑trivial control task.

## What’s In The Box
- Environments (`tetris_v2/envs/`)
  - NES ruleset: gravity table, scoring, no hold/kicks, single next-piece preview.
  - Modern ruleset: SRS kicks, 7‑bag, hold, **5-piece preview queue**, combos/B2B,
    ghost piece, lock delay, soft-drop factor, rgb/console rendering.
  - Versus: two modern boards with garbage exchange and a default random
    sparring partner.
- Real‑time play (`tetris_v2/scripts/play_human.py`)
  - Pygame window with HUD, ghost, hold and next queue.
  - Customizable DAS/ARR key repeat and bindings; rotate 180 included.
- Agents (`tetris_v2/agents/`) — organised by use case
  - `single_agent/`: score/survival learners (DQN→Rainbow, PPO/A2C, SAC‑discrete,
    distributional Q, R2D2, MuZero/AlphaZero, Dreamer, evolutionary baselines).
  - `exploration/`: RND, ICM, count‑based bonuses, NoisyNets, recurrent heads.
  - `adversarial_supplier/`: agents that choose pieces (minimise player score).
  - `pvp/`: self‑play for battle Tetris (NFSP, PSRO/league, AZ‑style self‑play).
  - `heuristics/`: linear/beam‑search baselines; random agent for smoke tests.
  - `offline/`: CQL/IQL/BCQ‑discrete pipelines for log‑only learning.
  - `ablations/`: scripts/harnesses for fair comparisons.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
If you prefer an editable install: `pip install -e .`.

## Play (Human)
```bash
python -m tetris_v2.scripts.play_human --env modern --fps 60
```
- Envs: `--env nes | modern | versus`
- Controls (modern/versus):
  - Move Left `a`, Move Right `d`
  - Soft Drop `Shift` (soft‑drop factor), Hard Drop `Space`
  - Rotate Left `←`, Rotate Right `→`, Rotate 180 `↑`
  - Hold `c`
- NES uses classic bindings (`←/→`, `↑`, `Space`, `S/↓`).
- Remap keys: edit `KEY_BINDINGS` in `tetris_v2/scripts/play_human.py`.
- Tuning: `--fps`, `--das`, `--arr`. The modern env also exposes
  `soft_drop_factor` (see code) and score‑blitz timers.

## Train (quick examples)
Scratch DQN trainer (fully native PyTorch implementation):
```bash
python -m tetris_v2.agents.dqn.train --env modern --total-timesteps 500000 \
  --learning-rate 2.5e-4 --seed 42 --log-dir runs/dqn_modern
```
Evaluate a saved policy:
```bash
python -m tetris_v2.agents.dqn.eval runs/dqn_modern/final_model.pt --env modern --render
```

## Repository Layout
- `envs/`  environments and utilities
- `scripts/` CLI tools (`play_human`, training/eval helpers)
- `agents/` grouped algorithm families (see above)
- `adversarial/` versus wrappers
- `configs/` experiment YAMLs
- `tests/` unit tests and sanity checks

## Roadmap (Algorithms To Implement)
- Single‑agent: DQN/Double/Dueling, PER, Rainbow, C51/QR/IQN, n‑step Q, R2D2,
  A2C/A3C, PPO, IMPALA/V‑trace, ACER/ACKTR, SAC‑discrete, MuZero/Gumbel, AZ‑MCTS,
  DreamerV3, BC/DAgger/BC+PPO, CMA‑ES/NES/NEAT
- Exploration add‑ons: RND, ICM, ensemble disagreement, count‑based, NoisyNets,
  LSTM/GRU heads
- Adversarial supplier: scripted minimax(1), PPO with KL to 7‑bag, RARL/PSRO
- PvP: self‑play PPO/A2C, NFSP, PSRO/league, AZ‑style self‑play
- Heuristics/search: linear evals, beam search (for baselines + BC datasets)
- Offline RL: CQL, IQL, BCQ‑discrete
- Ablations: Double vs plain DQN, PER vs uniform, NoisyNets vs ε‑greedy,
  PPO clip vs KL, distributional vs scalar Q

## Troubleshooting
- `ModuleNotFoundError: gymnasium` — install deps with `pip install -r requirements.txt`.
- Pygame “video system not initialized” — ensure the script starts pygame (we
  do) and you’re not running in a headless shell.
- Font warnings on macOS — pygame falls back to built‑ins; safe to ignore.

## Contributing
- Each subfolder under `agents/` contains a README to document design choices,
  algorithm derivations, and environment‑specific adaptations.
- Keep implementations modular so add‑ons (e.g., RND, LSTM) plug into multiple
  baselines.

## License
See `LICENSE`.
