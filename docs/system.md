---
title: System / Implementation
permalink: /system/
---

**Navigation:** [Home]({{ '/' | relative_url }}) | [System / Implementation]({{ '/system/' | relative_url }}) | [Algorithms]({{ '/algorithms/' | relative_url }}) | [Results]({{ '/results/' | relative_url }}) | [Media]({{ '/media/' | relative_url }}) | [Timeline]({{ '/timeline/' | relative_url }})

## System Overview

`TetrisV2` is implemented as a two-track architecture with shared evaluation goals:

```text
Version One (Python RL) --------------------\
                                             -> policy evaluation + comparison
Version Two (C++ env + Cold Clear + C API) -/
                         |
                         -> BC/DAgger data + model training + viewer runtime
```

## Version One Implementation

Version One is the rapid-iteration RL environment and training layer.

- Environment families: NES-like and modern SRS.
- PPO implementation and training/evaluation scripts live under:
  - `TetrisVersionOne/env`
  - `TetrisVersionOne/agents/ppo`
  - `TetrisVersionOne/scripts`
- Best for reward shaping, curriculum iteration, and RL baseline iteration.

Deep dive: [Version One Detail]({{ '/version-one/' | relative_url }})

## Version Two Implementation

Version Two is the systems/runtime and expert-compatible stack.

- C++ state and rules implementation:
  - `TetrisVersionTwo/include/tetris_v2`
  - `TetrisVersionTwo/src`
- App/tooling surface:
  - `TetrisVersionTwo/apps/cli_bot_play.cpp`
  - `TetrisVersionTwo/scripts/play_pygame.py`
- Shared library interface supports Python orchestration via C API.

Deep dive: [Version Two Detail]({{ '/version-two/' | relative_url }})

## Data and Logging Pipeline

BC/DAgger pipeline lives in `TetrisVersionTwo/scripts/bc`.

Core flow:
1. Collect expert or DAgger data (`python -m bc.collect_data`, `python -m bc.dagger`).
2. Train supervised policy (`python -m bc.train`).
3. Evaluate offline/online (`python -m bc.evaluate`).
4. Run policy in pygame viewer (`--bc-checkpoint` in `play_pygame.py`).

Artifacts are organized under `data/*` and `runs/*` with metadata and summary files for reproducibility.

## Viewer and Tooling Stack

- CLI gameplay/testing via Version Two app.
- Pygame viewer for qualitative policy behavior inspection.
- BC/DAgger utilities for progress reporting, run summaries, and cleanup.

## How Components Connect

- Version Two produces production-aligned state/action semantics.
- BC/DAgger learns from Version Two expert signals and learner-state aggregation.
- Results are fed back into evaluation and visualization for iterative improvement.
- Version One remains the experimentation lane for alternative RL methods.
