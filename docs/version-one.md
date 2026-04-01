---
title: Version One (Detail)
permalink: /version-one/
---

**Navigation:** [Home]({{ '/' | relative_url }}) | [System / Implementation]({{ '/system/' | relative_url }}) | [Algorithms]({{ '/algorithms/' | relative_url }}) | [Results]({{ '/results/' | relative_url }}) | [Media]({{ '/media/' | relative_url }}) | [Timeline]({{ '/timeline/' | relative_url }})

## Version One Deep Dive

Version One is the Python-first experimentation track focused on RL training speed and environment iteration.

## Implementation Surface

- Environments: `TetrisVersionOne/env`
- PPO: `TetrisVersionOne/agents/ppo`
- Scripts: `TetrisVersionOne/scripts`
- Tests: `TetrisVersionOne/tests`
- Presets: `TetrisVersionOne/presets`

## Workflow

```bash
python -m TetrisVersionOne.scripts.train --env modern \
  --total-timesteps 1500000 --num-envs 8 --log-dir runs/ppo_modern_v1

python -m TetrisVersionOne.scripts.eval runs/ppo_modern_v1/final_model.pt \
  --env modern --render
```

## Engineering Role in the Full Project

- Fast lane for reward and curriculum exploration.
- RL baseline line of evidence alongside BC/DAgger.
- Useful for stress-testing design choices before deeper systems integration.

## Limits and Next Work

- Slower at large-scale simulation than C++ path.
- Needs ongoing semantic parity checks with Version Two.
- Continue using it as the RL prototyping lane.
