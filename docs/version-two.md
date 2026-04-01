---
title: Version Two (Detail)
permalink: /version-two/
---

**Navigation:** [Home]({{ '/' | relative_url }}) | [System / Implementation]({{ '/system/' | relative_url }}) | [Algorithms]({{ '/algorithms/' | relative_url }}) | [Results]({{ '/results/' | relative_url }}) | [Media]({{ '/media/' | relative_url }}) | [Timeline]({{ '/timeline/' | relative_url }})

## Version Two Deep Dive

Version Two is the C++ systems track and the primary expert-compatible runtime layer.

## Implementation Surface

- C++ interfaces and core logic:
  - `TetrisVersionTwo/include/tetris_v2`
  - `TetrisVersionTwo/src`
- Runtime apps/tooling:
  - `TetrisVersionTwo/apps/cli_bot_play.cpp`
  - `TetrisVersionTwo/scripts/play_pygame.py`
- BC/DAgger orchestration:
  - `TetrisVersionTwo/scripts/bc`

## Build and Run

```bash
cmake -S . -B build-wsl
cmake --build build-wsl -j

python TetrisVersionTwo/scripts/play_pygame.py \
  --lib build-wsl/TetrisVersionTwo/libtetris_v2_c_api.so \
  --ai
```

## Engineering Role in the Full Project

- Authoritative state/action semantics for expert imitation.
- Cold Clear-compatible expert interface for supervision.
- Shared runtime for data collection, evaluation, and viewer-based debugging.

## BC/DAgger Connection

- Data collection and aggregation through `python -m bc.collect_data` and `python -m bc.dagger`.
- Supervised training via `python -m bc.train`.
- Offline and online evaluation via `python -m bc.evaluate`.
