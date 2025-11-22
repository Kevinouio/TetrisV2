"""Env registration helpers for Gymnasium."""
from gymnasium.envs.registration import register

def register_envs():
    # NES env
    try:
        register(
            id="KevinNES/Tetris-v0",
            entry_point="TetrisVersionOne.env.nes_tetris_env:NesTetrisEnv",
        )
    except Exception:
        pass

    # Modern env
    try:
        register(
            id="KevinModern/Tetris-v0",
            entry_point="TetrisVersionOne.env.modern_tetris_env:ModernTetrisEnv",
        )
    except Exception:
        pass
