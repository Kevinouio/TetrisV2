# Single-Agent Tetris Agents

This tier covers algorithms that control a lone player whose objective is to
score as many points (or survive as long) as possible in the NES/modern Tetris
environments.  Implementations will live under this package, grouped by
algorithm family.

Planned implementations:

- **Value-Based**
  - DQN (baseline)
  - Double DQN
  - Dueling DQN
  - Prioritized Experience Replay (PER)
  - Rainbow (C51 + PER + n-step + noisy + dueling + double)
  - Distributional: C51, QR-DQN, IQN
  - n-step Q-learning
  - Recurrent Q-learning (e.g., R2D2)
- **Actor-Critic**
  - A2C / A3C
  - PPO
  - IMPALA / V-trace
  - ACER / ACKTR
  - Soft Actor-Critic (discrete)
- **Hybrid / Model-Based**
  - MuZero / Gumbel MuZero
  - AlphaZero-style policy+value with MCTS
  - DreamerV3 (discrete head)
- **Imitation + Fine-Tuning**
  - Behaviour Cloning (BC)
  - DAgger
  - BC warm-start with PPO fine-tuning
- **Evolutionary Baselines**
  - CMA-ES / NES / NEAT over compact evaluation networks

Existing modules:

- `dqn/` – baseline DQN trainer, models, and evaluation scripts.
