"""Env registration helpers for Gymnasium."""
from gymnasium.envs.registration import register

def register_envs():
    # NES env
    try:
        register(
            id="KevinNES/Tetris-v0",
            entry_point="tetris_v2.envs.nes_tetris_env:NesTetrisEnv",
        )
    except Exception:
        pass

    # Modern env
    try:
        register(
            id="KevinModern/Tetris-v0",
            entry_point="tetris_v2.envs.modern_tetris_env:ModernTetrisEnv",
        )
    except Exception:
        pass

    # Versus env
    try:
        register(
            id="KevinVersus/Tetris-v0",
            entry_point="tetris_v2.adversarial.versus_env:VersusEnv",
        )
    except Exception:
        pass
