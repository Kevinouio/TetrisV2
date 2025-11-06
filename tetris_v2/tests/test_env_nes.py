import pytest

pytest.importorskip("gymnasium")

from tetris_v2.envs.nes_tetris_env import NesTetrisEnv


def test_nes_env_step():
    env = NesTetrisEnv()
    obs, _ = env.reset(seed=0)
    assert obs["board"].shape == (20, 10)
    next_obs, reward, terminated, truncated, info = env.step(0)
    assert "board" in next_obs
    assert isinstance(reward, float)
    assert not isinstance(info, list)
    env.close()
