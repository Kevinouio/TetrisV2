---
title: Results
permalink: /results/
---

**Navigation:** [Home]({{ '/' | relative_url }}) | [System / Implementation]({{ '/system/' | relative_url }}) | [Algorithms]({{ '/algorithms/' | relative_url }}) | [Results]({{ '/results/' | relative_url }}) | [Media]({{ '/media/' | relative_url }}) | [Timeline]({{ '/timeline/' | relative_url }})

## Current Working Baselines

| Area | Baseline | Status | Main Artifact |
|---|---|---|---|
| Expert Search | Cold Clear-compatible policy in Version Two | Working | Version Two CLI/pygame paths |
| RL | PPO baseline in Version One | Working, iterating | `runs/ppo_*` |
| Imitation | Top-1 BC | Working | `runs/bc_top1` |
| Imitation | DAgger on top of BC | Working, iterating | `runs/dagger_top1`, `runs/dagger_random` |

## Qualitative Observations

- BC improves immediate expert-likeness but can drift under novel board distributions.
- DAgger improves learner-state coverage and reduces brittle behavior in off-distribution positions.
- Offline classification gains do not always map directly to better long-run gameplay.
- Viewer-based inspection remains critical for spotting policy failure modes.

## Quantitative Placeholders

### Offline Classification Metrics

| Run | Split | Top-1 | Top-5 | Val Loss | Notes |
|---|---|---|---|---|---|
| `bc_top1` | test | `TBD` | `TBD` | `TBD` | Fill from `runs/bc_top1/summary.json` |
| `dagger_top1` | test | `TBD` | `TBD` | `TBD` | Fill from latest round summary |

### Online Gameplay Metrics

| Run | Avg Lines | Avg Pieces | Topout Rate | Invalid Raw | Notes |
|---|---|---|---|---|---|
| `bc_top1` | `TBD` | `TBD` | `TBD` | `TBD` | |
| `dagger_top1` | `TBD` | `TBD` | `TBD` | `TBD` | |
| Cold Clear Expert | `TBD` | `TBD` | `TBD` | n/a | Reference baseline |

## Completed Experiments (Merged from Previous Experiments Page)

| Date | Run | Objective | Dataset | Outcome |
|---|---|---|---|---|
| 2026-03-31 | `bc_top1` | Top-1 BC baseline | `data/bc_top1` | Baseline established |
| 2026-04-01 | `dagger_top1` | Learner-state aggregation | Aggregated BC+DAgger | Round-wise fine-tuning active |
| Ongoing | `dagger_random` | Broader state coverage | Random-board rounds | In progress |

## Planned Comparison Areas

- BC vs DAgger under identical online eval budgets.
- Expert imitation quality vs gameplay stability tradeoff.
- Random-board DAgger vs rollout-only DAgger.
- PPO (Version One) vs imitation-based pipelines in common evaluation protocol.

## Plot and Figure Placeholders

```markdown
![Offline Accuracy Curve]({{ '/assets/screenshots/offline_accuracy_curve.png' | relative_url }})
![Online Lines Cleared Comparison]({{ '/assets/screenshots/online_lines_comparison.png' | relative_url }})
```

Use this section for stable figure names once plotting scripts are finalized.
