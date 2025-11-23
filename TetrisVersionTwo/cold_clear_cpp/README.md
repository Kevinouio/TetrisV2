# cold_clear_cpp

A standalone C++17 reimplementation of the Cold Clear 2 bot. The project mirrors the structure of the Rust reference in `cold-clear-2/` without modifying that source.

## Layout

- `include/cold_clear_cpp/` — public headers.
  - `types.hpp` — pieces, rotations, placements.
  - `board.hpp` — bitboard-based playfield.
  - `state.hpp` — full game state (bag, hold, combo/back-to-back).
  - `movegen.hpp` — move generation and SRS kicks.
  - `eval.hpp` — heuristic weights and evaluation.
  - `state_map.hpp` — hashed game-state map used by the search DAG.
  - `dag.hpp` — layered DAG search (known/speculated layers).
  - `bot.hpp` — Freestyle bot wrapper around the DAG.
- `src/` — implementations that closely follow the Rust modules (`data.rs`, `movegen.rs`, `dag/*`, `bot/freestyle.rs`).
- `tests/test_all.cpp` — basic smoke tests for board operations, movegen, evaluation, and bot wiring.
- `LICENSE-MIT`, `LICENSE-APACHE` — copied from the Rust project.

## Building

```
cmake -S TetrisVersionTwo/cold_clear_cpp -B build/cold_clear_cpp
cmake --build build/cold_clear_cpp
ctest --test-dir build/cold_clear_cpp
```

## Mapping to Rust

- `data.rs` → `types.hpp`, `board.hpp`, `state.hpp`
- `movegen.rs` → `movegen.hpp`/`src/movegen.cpp`
- `dag.rs`, `dag/known.rs`, `dag/speculated.rs`, `map.rs` → `dag.hpp`, `state_map.hpp`, `src/dag.cpp`
- `bot.rs`, `bot/freestyle.rs`, `default.json` → `bot.hpp`, `eval.hpp`, `src/bot.cpp`, `src/eval.cpp`
- `tbp.rs`, `sync.rs`, `main.rs` are not reimplemented yet; the C++ project focuses on the core engine and search.
