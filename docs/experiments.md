---
title: Experiments
permalink: /experiments/
---

**Site Navigation:** [Home]({{ '/' | relative_url }}) | [Overview]({{ '/overview/' | relative_url }}) | [Version One]({{ '/version-one/' | relative_url }}) | [Version Two]({{ '/version-two/' | relative_url }}) | [Timeline]({{ '/timeline/' | relative_url }}) | [Videos]({{ '/videos/' | relative_url }}) | [Experiments]({{ '/experiments/' | relative_url }}) | [Results]({{ '/results/' | relative_url }})

## Experiment Ledger

Use this page as the central record of experiment intent, configuration, and artifacts.

## Active and Recent Tracks

| Track | Objective | Primary Script(s) | Status |
|---|---|---|---|
| Version One PPO | Improve stability and sample efficiency on modern rules | `TetrisVersionOne/scripts/train.py`, `TetrisVersionOne/scripts/eval.py` | Ongoing |
| Version Two Cold Clear | Validate C++ environment parity and expert quality | `TetrisVersionTwo/apps/cli_bot_play.cpp`, `TetrisVersionTwo/scripts/play_pygame.py` | Ongoing |
| BC Baseline | Top-1 imitation of Cold Clear actions | `python -m bc.collect_data`, `python -m bc.train`, `python -m bc.evaluate` | Implemented |
| DAgger | Improve learner-state coverage beyond pure expert rollouts | `python -m bc.dagger` | Implemented, iterating |

## BC and DAgger Experiment Matrix

| Date | Run | Dataset | Train Config | Eval Scope | Artifacts | Notes |
|---|---|---|---|---|---|---|
| 2026-03-31 | `bc_top1` | `data/bc_top1` | CNN+MLP CE baseline | Offline + online | `runs/bc_top1` | Initial supervised baseline. |
| 2026-04-01 | `dagger_top1` | Aggregated BC+DAgger rounds | Fine-tune per round | Round-wise eval | `runs/dagger_top1` | Added learner-state labeling loop. |
| TBD | `dagger_random` | Random-board round data | Configure per round | Online gameplay stress | `runs/dagger_random` | Track recovery from board perturbations. |

## Suggested Entry Format

```markdown
### Run: <run_name>
- Date:
- Goal:
- Command:
- Dataset:
- Key hyperparameters:
- Outcome:
- Follow-up:
```

## Notes on Reproducibility

- Keep command lines in run logs.
- Preserve `metadata.json`, training summaries, and round metrics.
- Record seed values and any non-default flags.

