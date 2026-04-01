---
title: Algorithms
permalink: /algorithms/
---

**Navigation:** [Home]({{ '/' | relative_url }}) | [System / Implementation]({{ '/system/' | relative_url }}) | [Algorithms]({{ '/algorithms/' | relative_url }}) | [Results]({{ '/results/' | relative_url }}) | [Media]({{ '/media/' | relative_url }}) | [Timeline]({{ '/timeline/' | relative_url }})

## Algorithm Portfolio

This page documents algorithm choices, their current status, and why each method exists in the stack.

## Summary Table

| Method | Category | Status | Role |
|---|---|---|---|
| Cold Clear-compatible search | Expert / Search | Implemented | Expert action source and strong policy reference |
| PPO baseline | Reinforcement Learning | Implemented | RL baseline in Python-first environments |
| Top-1 Behavioral Cloning | Imitation Learning | Implemented | Supervised expert imitation baseline |
| Ranking-aware Behavioral Cloning | Imitation Learning | Planned | Improve supervision quality beyond top-1 labels |
| DAgger | Imitation Learning | Implemented | Aggregate learner-state labels from expert |
| Offline RL / sequence-model baselines | Future Direction | Planned | Improve robustness beyond supervised imitation |
| Flow-based / Flow-Q-learning direction | Future Direction | Planned | Explore alternative policy/value learning formulations |

## Expert / Search Methods

### Cold Clear-compatible Expert

- **Status:** Implemented  
- **Purpose:** Provide high-quality action labels and strong online baseline behavior.  
- **Inputs:** Current board state, active piece, hold/queue context.  
- **Outputs:** Chosen placement/action and resulting trajectory behavior.  
- **Why Included:** Enables reproducible supervision and a known-strong comparison policy.  
- **Notes:** Integrated through Version Two C++ + C API tooling and used directly by BC/DAgger data collection.

## Reinforcement Learning

### PPO Baseline (Version One)

- **Status:** Implemented  
- **Purpose:** Establish RL baseline and allow reward/curriculum experiments.  
- **Inputs:** Environment observations from NES/modern SRS variants.  
- **Outputs:** Trained policy checkpoints and evaluation trajectories.  
- **Why Included:** Provides a flexible RL reference line independent of expert imitation.  
- **Notes:** Lives in `TetrisVersionOne/agents/ppo` and scripts under `TetrisVersionOne/scripts`.

## Imitation Learning

### Top-1 Behavioral Cloning

- **Status:** Implemented  
- **Purpose:** Learn to predict expert’s selected action from state.  
- **Inputs:** Expert-labeled state-action pairs from Version Two.  
- **Outputs:** Supervised policy network and offline/online metrics.  
- **Why Included:** Fast, reproducible baseline for expert imitation quality.  
- **Notes:** Commands: `python -m bc.collect_data`, `python -m bc.train`, `python -m bc.evaluate`.

### Ranking-aware Behavioral Cloning

- **Status:** Planned  
- **Purpose:** Use richer supervision than single top-1 action when available.  
- **Inputs:** Candidate actions with relative preference/score structure.  
- **Outputs:** Policy better aligned with expert ranking preferences.  
- **Why Included:** Addresses ambiguity where multiple legal actions are strong.  
- **Notes:** Not yet implemented in current codebase.

### DAgger

- **Status:** Implemented  
- **Purpose:** Reduce compounding errors by labeling learner-visited states with expert actions.  
- **Inputs:** Learner rollouts + expert query on visited states.  
- **Outputs:** Aggregated dataset across rounds and updated checkpoints.  
- **Why Included:** Improves coverage over pure expert-trajectory data.  
- **Notes:** Orchestrated by `python -m bc.dagger` with per-round metrics and summaries.

## Future / Planned Directions

### Offline RL / Sequence-Model Baselines

- **Status:** Planned  
- **Purpose:** Evaluate alternatives that can leverage logged trajectories more effectively.  
- **Inputs:** Curated datasets from expert + DAgger runs.  
- **Outputs:** Additional policy baselines for robust comparison.  
- **Why Included:** BC/DAgger may plateau without richer objective structure.  
- **Notes:** Candidate area after current BC/DAgger baselines are stabilized.

### Flow-based / Flow-Q-learning Direction

- **Status:** Planned  
- **Purpose:** Explore flow-based or flow-Q-learning-inspired formulations for policy improvement.  
- **Inputs:** State-action datasets and/or environment interaction budget.  
- **Outputs:** Experimental policy/value models for comparison against PPO/BC/DAgger.  
- **Why Included:** Potential path for improved sample usage and decision quality.  
- **Notes:** Explicitly future-facing; no production implementation yet.
