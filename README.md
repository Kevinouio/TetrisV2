# TetrisV2 Workspace

This repository contains two tracks:
- `TetrisVersionOne/`: original Python env + PPO workflow.
- `TetrisVersionTwo/`: C++ single-player runtime + Cold Clear bot + Python RL stack (PPO + DQN).

## VersionTwo quick start

Build C++ runtime:
```bash
cmake -S TetrisVersionTwo -B build/TetrisVersionTwo
cmake --build build/TetrisVersionTwo -j8
```

Install Python deps:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Three runtime lanes

1. Human lane (manual placement explorer):
```bash
python3 TetrisVersionTwo/scripts/play_human.py
```

2. Cold Clear lane (C++ bot autoplay):
```bash
python3 TetrisVersionTwo/scripts/play_cold_clear.py --think-ms 20 --auto-reset
```

3. RL lane (PPO/DQN):
- Train PPO:
```bash
python3 TetrisVersionTwo/scripts/train_rl_ppo.py --total-timesteps 500000 --log-dir runs/v2_ppo
```
- Train DQN:
```bash
python3 TetrisVersionTwo/scripts/train_rl_dqn.py --total-timesteps 500000 --log-dir runs/v2_dqn
```
- Evaluate:
```bash
python3 TetrisVersionTwo/scripts/eval_rl.py runs/v2_ppo/ppo_final.pt --algo ppo --episodes 10
python3 TetrisVersionTwo/scripts/eval_rl.py runs/v2_dqn/dqn_final.pt --algo dqn --episodes 10
```
- Play in pygame:
```bash
python3 TetrisVersionTwo/scripts/play_rl_pygame.py runs/v2_ppo/ppo_final.pt --algo ppo
python3 TetrisVersionTwo/scripts/play_rl_pygame.py runs/v2_dqn/dqn_final.pt --algo dqn
```
- Play in CLI:
```bash
python3 TetrisVersionTwo/scripts/play_rl_cli.py runs/v2_ppo/ppo_final.pt --algo ppo --render-board
python3 TetrisVersionTwo/scripts/play_rl_cli.py runs/v2_dqn/dqn_final.pt --algo dqn --render-board
```

### VersionTwo RL contracts
- Runtime source of truth: C++ `tetris_cc_*` API only.
- RL action space is fixed 8 discrete actions:
  - `None, Left, Right, SoftDrop, HardDrop, RotateCW, RotateCCW, Hold`
  - Rotate-180 is intentionally excluded.
- RL reward is raw C++ env reward (no Python reward shaping).
- PPO checkpoints use `algo=ppo`; DQN checkpoints use `algo=dqn`.

### Checkpoint layout
- PPO:
  - periodic: `runs/v2_ppo/ppo_checkpoint_step_<N>.pt`
  - final: `runs/v2_ppo/ppo_final.pt`
- DQN:
  - periodic: `runs/v2_dqn/dqn_checkpoint_step_<N>.pt`
  - final: `runs/v2_dqn/dqn_final.pt`

### Verify
```bash
ctest --test-dir build/TetrisVersionTwo --output-on-failure
python3 TetrisVersionTwo/tests/python_ctypes_smoke.py
python3 TetrisVersionTwo/tests/python_rl_env_tests.py
python3 TetrisVersionTwo/tests/python_rl_smoke.py
```

## License
See `LICENSE`.
