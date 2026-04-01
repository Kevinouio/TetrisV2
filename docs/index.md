---
title: Home
permalink: /
---

**Site Navigation:** [Home]({{ '/' | relative_url }}) | [Overview]({{ '/overview/' | relative_url }}) | [Version One]({{ '/version-one/' | relative_url }}) | [Version Two]({{ '/version-two/' | relative_url }}) | [Timeline]({{ '/timeline/' | relative_url }}) | [Videos]({{ '/videos/' | relative_url }}) | [Experiments]({{ '/experiments/' | relative_url }}) | [Results]({{ '/results/' | relative_url }})

## Mission

`TetrisV2` is a long-running engineering effort to build a high-quality Tetris research stack across two implementations:

- `TetrisVersionOne/`: Python-first environments and PPO training workflows.
- `TetrisVersionTwo/`: C++ environment + Cold Clear compatibility layer + C API + Python viewers/tools.

This site tracks the technical decisions, experiments, failures, and milestones over time.

## Current Snapshot

- Version One is stable for PPO training and evaluation on NES and modern SRS rulesets.
- Version Two has a working C++ Cold Clear path, C API surface, and a pygame viewer.
- BC and DAgger tooling is implemented under `TetrisVersionTwo/scripts/bc/` with command-line workflows for data collection, training, and evaluation.
- Current focus is improving imitation-learning quality, dataset curation, and tighter integration between Python and C++ pipelines.

## Quick Links

- Repo root: [`TetrisV2`](https://github.com/kevinouio/TetrisV2)
- Version One docs: [Version One]({{ '/version-one/' | relative_url }})
- Version Two docs: [Version Two]({{ '/version-two/' | relative_url }})
- Devlog entries: [Timeline]({{ '/timeline/' | relative_url }})
- BC/DAgger runs and metrics notes: [Experiments]({{ '/experiments/' | relative_url }}) and [Results]({{ '/results/' | relative_url }})

## Latest Milestone

### 2026-04

- DAgger pipeline integrated over the BC baseline in `TetrisVersionTwo/scripts/bc/dagger.py`.
- Random-board state generation and rollout variants were added to broaden learner-state coverage.
- Run artifact management and cleanup utilities were added to control storage growth.

## How This Site Is Organized

- **Overview:** system architecture and roadmap.
- **Version One:** Python environments + PPO training stack.
- **Version Two:** C++ Cold Clear-compatible environment and tooling.
- **Timeline:** dated development updates.
- **Videos:** local media and external recordings.
- **Experiments:** run configurations and notes.
- **Results:** consolidated metrics and status.

