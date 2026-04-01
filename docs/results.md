---
title: Results
permalink: /results/
---

**Site Navigation:** [Home]({{ '/' | relative_url }}) | [Overview]({{ '/overview/' | relative_url }}) | [Version One]({{ '/version-one/' | relative_url }}) | [Version Two]({{ '/version-two/' | relative_url }}) | [Timeline]({{ '/timeline/' | relative_url }}) | [Videos]({{ '/videos/' | relative_url }}) | [Experiments]({{ '/experiments/' | relative_url }}) | [Results]({{ '/results/' | relative_url }})

## Results Summary

This page consolidates key metrics across PPO, expert gameplay, BC, and DAgger.

## Current Status

- Metrics below are a living summary.
- Mark entries as **provisional** until backed by repeat runs and fixed seeds.

## Version One PPO (Provisional)

| Environment | Model | Metric Focus | Latest Status | Artifacts |
|---|---|---|---|---|
| Modern SRS | PPO baseline | Survival and line clears | In active tuning | `runs/ppo_modern_v1` |
| NES-like | PPO baseline | Stability under classic constraints | In active tuning | `runs/ppo_nes_*` |

## Version Two Cold Clear Expert

| Component | Metric Focus | Latest Status | Artifacts |
|---|---|---|---|
| C++ Cold Clear autoplay | Survival behavior sanity | Stable demo path | `TetrisVersionTwo/apps/cli_bot_play.cpp`, `Recordings/ColdClear.gif` |
| Pygame viewer integration | Runtime visual validation | Stable | `TetrisVersionTwo/scripts/play_pygame.py` |

## Behavioral Cloning and DAgger

| Model | Dataset | Primary Metrics | Status | Artifacts |
|---|---|---|---|---|
| BC top-1 | Expert rollout dataset | Top-1/top-5 offline + online lines/survival | Baseline established | `runs/bc_top1` |
| DAgger fine-tuned | Aggregated BC + DAgger rounds | Round-wise policy quality and gameplay | Ongoing | `runs/dagger_top1`, `runs/dagger_random` |

## Recommended Reporting Block Per Run

```markdown
### <run_name> (YYYY-MM-DD)
- Data: transitions=<n>, vocab=<n>
- Train: best_val_loss=<x>, top1=<x>, top5=<x>
- Online: avg_lines=<x>, avg_pieces=<x>, topout_rate=<x>
- Notes: failure modes and next change
```

## Interpretation Notes

- Offline classification improvements do not always translate to better gameplay.
- Online evaluation should include enough games to reduce noise.
- DAgger rounds should be analyzed for both data quality and distribution shift effects.

