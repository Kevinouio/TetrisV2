# DQN Ref Porting Notes

This baseline ports `Tetris-A.I/Version2` training style into `TetrisVersionTwo`.

## Directly Ported
- 6-feature afterstate representation:
  - `total_height`, `bumpiness`, `lines_removed`, `holes`, `y_pos`, `pillar`
- MLP architecture: `6 -> 32 -> 32 -> 32 -> 1`
- Epsilon-greedy over candidate afterstate scores
- Prioritized replay structure and update flow
- Online/target network training with SmoothL1 and gradient clipping
- Step/game cadence:
  - train every 200 steps
  - target sync every 1000 steps
  - fallback train at least once per game
- Reward logic from `Version2/Agent.py`, including branch constants and thresholds
- GA-style orchestration (population, tournament selection, crossover, mutation)

## Necessary Adaptations
- Uses `TetrisVersionTwo` indexed placement actions from C API (`NativeAction(use_hold, placement_index)`), not low-level controls.
- Uses C API placement afterstate boards to evaluate candidates efficiently.
- Training loop is headless (no pygame rendering in the training path).
- Adds `max_steps_per_episode` safety cap for robustness.
- Logs are written as CSV/JSON artifacts under `runs/dqn_ref/*`.

## Performance Update
- Added placement-option caching in `ModernTetrisEnv` to avoid recomputing full placement BFS on repeated queries.
- Added batch candidate C API exports for reduced FFI overhead:
  - `tetris_cc_env_candidate_count`
  - `tetris_cc_env_candidate_get`
  - `tetris_cc_env_candidate_features_write`
- `dqn_ref` now prefers the batch path with automatic fallback to the legacy per-placement APIs.
- Benchmark commands and reporting template: `OPTIMIZATION_NOTES.md`.
