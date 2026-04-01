---
title: Overview
permalink: /overview/
---

**Site Navigation:** [Home]({{ '/' | relative_url }}) | [Overview]({{ '/overview/' | relative_url }}) | [Version One]({{ '/version-one/' | relative_url }}) | [Version Two]({{ '/version-two/' | relative_url }}) | [Timeline]({{ '/timeline/' | relative_url }}) | [Videos]({{ '/videos/' | relative_url }}) | [Experiments]({{ '/experiments/' | relative_url }}) | [Results]({{ '/results/' | relative_url }})

## Project Architecture

`TetrisV2` is intentionally split into two tracks:

```text
TetrisV2/
  TetrisVersionOne/
    env/
    agents/ppo/
    scripts/
    tests/
  TetrisVersionTwo/
    include/tetris_v2/
    src/
    apps/cli_bot_play.cpp
    scripts/play_pygame.py
    scripts/bc/
```

- **Version One** is the rapid iteration layer for environment variants and RL training loops.
- **Version Two** is the systems/performance layer for Cold Clear-compatible logic exposed through a C API.

## BC and DAgger Pipeline Position

Behavioral cloning and DAgger live in `TetrisVersionTwo/scripts/bc/` and reuse the Version Two C API.

Core commands:

```bash
python -m bc.collect_data ...
python -m bc.train ...
python -m bc.evaluate ...
python -m bc.dagger ...
```

This keeps dataset collection and policy learning close to the production game state representation.

## Integration Roadmap

1. Maintain strong parity between the Python and C++ state/action semantics.
2. Improve imitation-data diversity (expert demonstrations and DAgger states).
3. Improve evaluation rigor (offline metrics plus online gameplay metrics).
4. Keep tooling reproducible and practical for iterative experimentation.

## Design Principles

- Reproducible runs (explicit seeds, structured metadata).
- Minimal hidden magic (command-line first, inspectable artifacts).
- Fast manual debugging loops (pygame viewer + run summaries).
- Compatibility over rewrites (reuse existing env/expert APIs whenever possible).

