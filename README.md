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
Nes-specific shaping options are available, e.g.:
```bash
python -m tetris_v2.agents.dqn.train --env nes --rotation-penalty 5 \
  --line-clear-reward 0 25 50 150 --step-penalty 1 \
  --prioritized-replay --per-alpha 0.6 --per-beta 0.4
```
Evaluate a saved policy:
```bash
python -m tetris_v2.agents.dqn.eval runs/dqn_modern/final_model.pt --env modern --render
```
Prefer not to retype dozens of flags? Use the helper wrappers:
```bash
./scripts/train_agent.sh --algo ppo --env nes --num-envs 8 --log-dir runs/ppo_nes_fast
./scripts/eval_agent.sh  --algo ppo --env nes --checkpoint runs/ppo_nes_fast/best_model.pt --render
```
On Windows PowerShell, call the `.ps1` equivalents:
```powershell
.\scripts\train_agent.ps1 --algo dqn --env modern --num-envs 4 --log-dir runs\dqn_modern
.\scripts\eval_agent.ps1  --algo dqn --env modern --checkpoint runs\dqn_modern\final_model.pt
```

Need more directed exploration or richer shaping? Toggle Boltzmann sampling and the
new hole/survival reward wrapper:
```bash
python -m tetris_v2.agents.dqn.train --env modern --exploration-strategy boltzmann \
  --boltzmann-temp-start 2.0 --boltzmann-temp-end 0.2 \
  --advanced-reward-weight hole_penalty=1.2 --log-dir runs/dqn_adv_reward
```
Advanced reward shaping (holes, survival, idle penalties) is always enabled; use `--advanced-reward-weight KEY=VALUE`
to tweak individual scalars.
Scale out experience collection with asynchronous env workers:
```bash
python -m tetris_v2.agents.dqn.train --env modern --num-envs 8 --total-timesteps 2000000 \
  --prioritized-replay --log-dir runs/dqn_vector
```
Switch between vanilla DQN, Double DQN, and Dueling heads on the same CLI:
```bash
python -m tetris_v2.agents.dqn.train --env nes --no-double-dqn --no-dueling  # classic DQN
python -m tetris_v2.agents.dqn.train --env modern --double-dqn --dueling    # Rainbow-style core
python -m tetris_v2.agents.dqn.train --env modern --prioritized-replay      # enables PER buffer
```

Native PPO baseline (same observation encoder, actor-critic head, and reward extras):
```bash
python -m tetris_v2.agents.ppo.train --env modern --total-timesteps 1500000 \
  --n-steps 4096 --minibatch-size 1024 --policy-temperature-start 1.25 \
  --log-dir runs/ppo_modern
```
Need more throughput? Append `--num-envs N` (e.g., `--num-envs 8`) to launch a pool of AsyncVectorEnv workers feeding PPO.
Evaluate PPO checkpoints via:
```bash
python -m tetris_v2.agents.ppo.eval runs/ppo_modern/final_model.pt --env modern --render
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
