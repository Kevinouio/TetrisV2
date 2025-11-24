# TetrisV2 Workspace

This repo now contains two tracks:
- **Version One (Python)** — NES and modern SRS environments with PPO training (`TetrisVersionOne/`), plus scripts/tests/presets.
- **Version Two (C++/Rust interop)** — A cold-clear-compatible environment (`TetrisVersionTwo/`) with a C++ port of Cold Clear 2 (`cold_clear_cpp`) and a pygame viewer.

## Version Two (Cold Clear)
- C++ core: `TetrisVersionTwo/cold_clear_cpp` mirrors the Rust reference (`cold-clear-2/`).
- Environment/wrapper: `TetrisVersionTwo/include/tetris_v2/`, `TetrisVersionTwo/src/`.
- CLI viewer: `TetrisVersionTwo/cli/cc_play.cpp` (outputs JSON; now includes a `ghost` overlay of the chosen move).
- Pygame viewer: `TetrisVersionTwo/scripts/play_pygame.py` renders the board, queue/hold, and ghost outline.

### Build
```bash
cmake -S . -B build
cmake --build build
```

### Run Cold Clear in CLI/pygame
```bash
# CLI (prints JSON or text)
build/TetrisVersionTwo/cc_play --steps 200 --delay-ms 120 --json

# Pygame viewer (expects cc_play at build/TetrisVersionTwo/cc_play)
python TetrisVersionTwo/scripts/play_pygame.py --steps 500 --delay-ms 80
```
Notes:
- The bot considers the full visible queue (8 by default) and speculates with the 7‑bag. Increase search time by raising the iteration count in `cc_play` (argument to `choose_move`).
- Default heuristics live in `cold_clear_cpp/include/cold_clear_cpp/eval.hpp` (C++) and `cold-clear-2/src/default.json` (Rust). Adjust these to trade off survival vs. T‑spins.
- Combo tracking bug is fixed; soft-drop distance is included in the eval reward.

### Demo (Cold Clear playing to survive)
Default survival-oriented eval (human-set weights; not tuned for score chasing):
<a href="Recordings/ColdClear.gif">


  <img src="Recordings/ColdClear.gif" width="360" alt="Cold Clear survival loop">
</a>

### Logging for imitation
To collect demonstrations from Cold Clear, tap into `cc_play` to emit per-move state/action JSONL (board, hold, queue, bag, chosen `Placement`). A future `scripts/` helper can batch games over seeds to produce a dataset.

## Version One (Python/PPO)
- Envs: `TetrisVersionOne/env/` (NES + modern SRS with ghost/hold/kicks).
- Agent: `TetrisVersionOne/agents/ppo/`.
- Scripts: `TetrisVersionOne/scripts/` (`train`, `eval`, `play_human`).
- Tests: `TetrisVersionOne/tests/`.

### Install (Python toolchain)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Quick starts
```bash
# Human play (modern)
python -m TetrisVersionOne.scripts.play_human --env modern --fps 60

# Train PPO (modern)
python -m TetrisVersionOne.scripts.train --env modern \
  --total-timesteps 1500000 --num-envs 8 --log-dir runs/ppo_modern_v1

# Eval PPO
python -m TetrisVersionOne.scripts.eval runs/ppo_modern_v1/final_model.pt \
  --env modern --render
```

## Roadmap
- Collect Cold Clear trajectories for imitation learning.
- Flow-based / RL fine-tuning on top of the learned policy.
- Tighten integration between Python envs and the C++/Rust Cold Clear core.

## License
See `LICENSE`.
