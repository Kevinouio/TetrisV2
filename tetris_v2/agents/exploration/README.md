# Exploration & Representation Add-ons

Modules in this directory provide plug-in components that can augment any
single-agent learner (e.g., curiosity bonuses, auxiliary networks, recurrent
policies).  They should be implemented in a way that keeps them reusable across
value-based and actor-critic stacks.

Planned components:

- Random Network Distillation (RND)
- Intrinsic Curiosity Module (ICM)
- Ensemble / Disagreement exploration bonuses
- Count-based / CTS style novelty bonuses on board hashes
- NoisyNets and parameter-space noise wrappers
- Recurrent policy/value heads (LSTM / GRU) for long-term context
