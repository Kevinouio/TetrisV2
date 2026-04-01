---
title: Home
permalink: /
---

**Navigation:** [Home]({{ '/' | relative_url }}) | [System / Implementation]({{ '/system/' | relative_url }}) | [Algorithms]({{ '/algorithms/' | relative_url }}) | [Results]({{ '/results/' | relative_url }}) | [Media]({{ '/media/' | relative_url }}) | [Timeline]({{ '/timeline/' | relative_url }})

## TetrisV2: Search, RL, and Imitation in One Stack

`TetrisV2` is a long-term technical project to build and evaluate Tetris agents across two complementary tracks:
- `TetrisVersionOne/`: Python-first environments and PPO workflows for rapid RL iteration.
- `TetrisVersionTwo/`: C++ environment with Cold Clear-compatible search, C API, and viewer/tooling integration.

## Why This Project Exists

The project is designed to close the gap between:
- fast experimentation in Python RL pipelines, and
- production-leaning systems integration around a C++ game core and expert search policy.

The goal is not only to train agents, but to build a reproducible engineering pipeline for collecting data, evaluating behavior, and iterating on algorithmic choices.

## Two Development Tracks

| Track | Primary Role | Core Stack | Key Paths |
|---|---|---|---|
| Version One | RL prototyping and environment iteration | Python, PyTorch, Gym-style env APIs | `TetrisVersionOne/env`, `TetrisVersionOne/agents/ppo`, `TetrisVersionOne/scripts` |
| Version Two | Systems integration and expert-compatible runtime | C++, C API, Python tooling | `TetrisVersionTwo/include`, `TetrisVersionTwo/src`, `TetrisVersionTwo/apps`, `TetrisVersionTwo/scripts` |

## Architecture at a Glance

1. Version Two environment + expert interface produce high-quality supervised labels.
2. BC/DAgger pipeline (`python -m bc.*`) builds datasets and trains policy models.
3. Policies are evaluated offline and online, then visualized in the pygame viewer.
4. Version One remains the flexible RL sandbox for alternative learning strategies.

## Featured Media

![Cold Clear Survival Demo]({{ '/assets/gifs/ColdClear.gif' | relative_url }})

<video controls muted playsinline width="760">
  <source src="{{ '/assets/videos/version-two/project_highlight.mp4' | relative_url }}" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Key Contributions

- Built dual-track Tetris architecture (Python RL + C++ expert-compatible runtime).
- Integrated Cold Clear-compatible search and C API bindings.
- Implemented top-1 behavioral cloning and DAgger data aggregation workflows.
- Added online/offline evaluation scripts and interactive viewer support.
- Added storage/cleanup workflows for large-scale run artifacts.

## Explore the Project

- [System / Implementation]({{ '/system/' | relative_url }})
- [Algorithms]({{ '/algorithms/' | relative_url }})
- [Results]({{ '/results/' | relative_url }})
- [Media]({{ '/media/' | relative_url }})
- [Timeline]({{ '/timeline/' | relative_url }})
