---
title: Version Two
permalink: /version-two/
---

**Site Navigation:** [Home]({{ '/' | relative_url }}) | [Overview]({{ '/overview/' | relative_url }}) | [Version One]({{ '/version-one/' | relative_url }}) | [Version Two]({{ '/version-two/' | relative_url }}) | [Timeline]({{ '/timeline/' | relative_url }}) | [Videos]({{ '/videos/' | relative_url }}) | [Experiments]({{ '/experiments/' | relative_url }}) | [Results]({{ '/results/' | relative_url }})

## Scope

Version Two is the C++/interop track designed for stronger systems-level control and Cold Clear compatibility.

Primary directories:

- `TetrisVersionTwo/include/tetris_v2/`
- `TetrisVersionTwo/src/`
- `TetrisVersionTwo/apps/cli_bot_play.cpp`
- `TetrisVersionTwo/scripts/play_pygame.py`
- `TetrisVersionTwo/scripts/bc/`

## C++ Core and C API

- Board/state logic and move application are implemented in C++.
- A C API exposes environment lifecycle, state readout, and expert interactions.
- Python tooling (viewer + BC/DAgger scripts) binds to the shared library (`libtetris_v2_c_api.so` in WSL builds).

Build:

```bash
cmake -S . -B build-wsl
cmake --build build-wsl -j
```

## Cold Clear Integration

- Version Two includes a C++ Cold Clear 2 integration layer.
- The CLI app and pygame viewer can drive the expert in live gameplay.

Pygame viewer:

```bash
python TetrisVersionTwo/scripts/play_pygame.py \
  --lib build-wsl/TetrisVersionTwo/libtetris_v2_c_api.so \
  --ai
```

## BC and DAgger in Version Two

BC/DAgger pipeline modules:

- `collect_data.py`
- `train.py`
- `evaluate.py`
- `inference_agent.py`
- `dagger.py`

These scripts operate directly on Version Two state/action semantics to avoid training-serving mismatch.

## Interoperability Notes

- Version Two is the authoritative path for Cold Clear-based data generation.
- Policy checkpoints produced here can be run in the pygame viewer using `--bc-checkpoint`.
- Tooling now supports long-running data jobs, progress tracking, and cleanup workflows.

