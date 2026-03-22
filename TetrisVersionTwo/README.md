# TetrisVersionTwo

`TetrisVersionTwo` is the C++ runtime track with three Python-facing lanes:
- Human manual placement explorer,
- Cold Clear autoplay,
- RL (PPO + DQN) train/eval/play.

## Runtime surface
- C API: `tetris_cc_*` (`include/tetris_v2/c_api.h`)
- Shared library target: `tetris_v2_c_api`
- Python ctypes binding: `TetrisVersionTwo/rl/runtime.py`
- Gym wrapper for RL: `TetrisVersionTwo/rl/env.py`

## Lanes

Human lane:
```bash
python3 TetrisVersionTwo/scripts/play_human.py
```

Cold Clear lane:
```bash
python3 TetrisVersionTwo/scripts/play_cold_clear.py --think-ms 20 --auto-reset
```

RL lane:
```bash
python3 TetrisVersionTwo/scripts/train_rl_ppo.py --total-timesteps 500000 --log-dir runs/v2_ppo
python3 TetrisVersionTwo/scripts/train_rl_dqn.py --total-timesteps 500000 --log-dir runs/v2_dqn
python3 TetrisVersionTwo/scripts/eval_rl.py runs/v2_ppo/ppo_final.pt --algo ppo --episodes 10
python3 TetrisVersionTwo/scripts/eval_rl.py runs/v2_dqn/dqn_final.pt --algo dqn --episodes 10
python3 TetrisVersionTwo/scripts/play_rl_pygame.py runs/v2_ppo/ppo_final.pt --algo ppo
python3 TetrisVersionTwo/scripts/play_rl_cli.py runs/v2_dqn/dqn_final.pt --algo dqn --render-board
```

## RL policy interface
- Action space is fixed and discrete (8 actions):
  - `None, Left, Right, SoftDrop, HardDrop, RotateCW, RotateCCW, Hold`
- Rotate-180 is intentionally excluded.
- Reward is raw C++ env reward (no Python shaping in this pass).

## Build and test
```bash
cmake -S TetrisVersionTwo -B build/TetrisVersionTwo
cmake --build build/TetrisVersionTwo -j8
ctest --test-dir build/TetrisVersionTwo --output-on-failure
python3 TetrisVersionTwo/tests/python_ctypes_smoke.py
python3 TetrisVersionTwo/tests/python_rl_env_tests.py
python3 TetrisVersionTwo/tests/python_rl_smoke.py
```
