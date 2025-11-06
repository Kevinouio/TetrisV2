import pytest

pytest.importorskip("gymnasium")

from tetris_v2.adversarial.versus_env import VersusEnv


def test_versus_env_basic_step():
    env = VersusEnv()
    obs, _ = env.reset(seed=0)
    assert "board" in obs and "opponent_board" in obs
    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert isinstance(reward, float)
    assert "player_attack" in info
    assert next_obs["opponent_board"].shape == (20, 10)
    env.close()
