import pytest

pytest.importorskip("gymnasium")

from tetris_v2.envs.modern_tetris_env import ModernTetrisEnv


def test_modern_env_step_cycle():
    env = ModernTetrisEnv()
    obs, _ = env.reset(seed=0)
    assert obs["board"].shape == (20, 10)
    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert "board" in next_obs
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "score" in info
    env.close()
