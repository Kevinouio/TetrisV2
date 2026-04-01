---
title: Version One
permalink: /version-one/
---

**Site Navigation:** [Home]({{ '/' | relative_url }}) | [Overview]({{ '/overview/' | relative_url }}) | [Version One]({{ '/version-one/' | relative_url }}) | [Version Two]({{ '/version-two/' | relative_url }}) | [Timeline]({{ '/timeline/' | relative_url }}) | [Videos]({{ '/videos/' | relative_url }}) | [Experiments]({{ '/experiments/' | relative_url }}) | [Results]({{ '/results/' | relative_url }})

## Scope

Version One is the Python training playground with two environment families:

- NES-like rules
- Modern SRS rules (hold, kicks, previews)

Primary directories:

- `TetrisVersionOne/env/`
- `TetrisVersionOne/agents/ppo/`
- `TetrisVersionOne/scripts/`
- `TetrisVersionOne/tests/`
- `TetrisVersionOne/presets/`

## Training Workflow

Install:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Train PPO (modern example):

```bash
python -m TetrisVersionOne.scripts.train --env modern \
  --total-timesteps 1500000 --num-envs 8 --log-dir runs/ppo_modern_v1
```

Evaluate:

```bash
python -m TetrisVersionOne.scripts.eval runs/ppo_modern_v1/final_model.pt \
  --env modern --render
```

Human play:

```bash
python -m TetrisVersionOne.scripts.play_human --env modern --fps 60
```

## What This Track Is Good For

- Fast reward-shaping experiments.
- Rapid policy architecture iteration in Python.
- Controlled curriculum and preset testing.

## Known Limitations

- Python stack is slower than the C++ path for large-scale simulation.
- Cross-track parity needs continuous validation when rules/features evolve.
- Final deployment behavior should always be checked against Version Two.


