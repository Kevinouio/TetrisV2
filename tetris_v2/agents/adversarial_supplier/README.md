# Adversarial Piece Supplier

This package houses agents that control the piece generator instead of the
player.  The objective is to minimise the player’s score while respecting Tetris
distributional constraints (e.g., 7-bag).

Planned implementations:

- Scripted minimax/heuristic supplier (depth-1 search)
- Robust Adversarial RL (RARL) with PPO/A2C alternating updates
- PPO supplier with KL penalty to a 7-bag prior
- Population-based methods (PSRO / fictitious play) for supplier-player games
- Supplier-side MCTS with short lookahead
