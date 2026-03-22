"""RL env wrapper contract tests for VersionTwo."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TetrisVersionTwo.rl.actions import RL_ACTION_MAP
from TetrisVersionTwo.rl.env import CCTetrisEnv
from TetrisVersionTwo.rl.runtime import (
    ACTION_CCW,
    ACTION_CW,
    ACTION_HARD_DROP,
    ACTION_HOLD,
    ACTION_LEFT,
    ACTION_NONE,
    ACTION_RIGHT,
    ACTION_SOFT_DROP,
)


def assert_action_mapping() -> None:
    expected = (
        ACTION_NONE,
        ACTION_LEFT,
        ACTION_RIGHT,
        ACTION_SOFT_DROP,
        ACTION_HARD_DROP,
        ACTION_CW,
        ACTION_CCW,
        ACTION_HOLD,
    )
    if tuple(RL_ACTION_MAP) != expected:
        raise AssertionError(f"RL_ACTION_MAP mismatch: got={RL_ACTION_MAP} expected={expected}")


def assert_observation_contract() -> None:
    env = CCTetrisEnv(seed=11, max_steps=200)
    try:
        obs, info = env.reset(seed=11)
        if obs.dtype != np.float32:
            raise AssertionError(f"Observation dtype must be float32, got {obs.dtype}")
        expected = int(env.runtime.observation_size(include_hidden_rows=False))
        if obs.shape != (expected,):
            raise AssertionError(f"Observation shape mismatch: got {obs.shape}, expected {(expected,)}")
        if int(env.observation_space.shape[0]) != expected:
            raise AssertionError("Observation space size mismatch with runtime observation_size")
        if not isinstance(info, dict):
            raise AssertionError("Reset info must be dict.")
    finally:
        env.close()


def assert_step_determinism() -> None:
    action_indices = [0, 5, 4, 1, 4, 6, 2, 3, 4, 7, 0, 4, 5, 4, 0, 4]
    env_a = CCTetrisEnv(seed=37, max_steps=500)
    env_b = CCTetrisEnv(seed=37, max_steps=500)
    try:
        obs_a, _ = env_a.reset(seed=37)
        obs_b, _ = env_b.reset(seed=37)
        if not np.allclose(obs_a, obs_b, atol=1e-6):
            raise AssertionError("Initial observations diverged for identical seeds.")
        for action in action_indices:
            n_obs_a, rew_a, term_a, trunc_a, info_a = env_a.step(action)
            n_obs_b, rew_b, term_b, trunc_b, info_b = env_b.step(action)
            if not np.allclose(n_obs_a, n_obs_b, atol=1e-6):
                raise AssertionError(f"Observation diverged on action={action}.")
            if abs(float(rew_a) - float(rew_b)) > 1e-6:
                raise AssertionError(f"Reward diverged on action={action}: {rew_a} vs {rew_b}")
            if bool(term_a) != bool(term_b) or bool(trunc_a) != bool(trunc_b):
                raise AssertionError("Termination flags diverged.")
            if int(info_a.get("lines", 0)) != int(info_b.get("lines", 0)):
                raise AssertionError("Line counters diverged.")
            if term_a or trunc_a:
                break
    finally:
        env_a.close()
        env_b.close()


def assert_termination_meta_consistency() -> None:
    env = CCTetrisEnv(seed=101, max_steps=2000)
    try:
        _, _ = env.reset(seed=101)
        prev_lines = 0
        terminated = False
        truncated = False
        for _ in range(2500):
            _, reward, terminated, truncated, info = env.step(4)  # hard drop action index
            if not math.isfinite(float(reward)):
                raise AssertionError("Non-finite reward returned.")
            lines = int(info.get("lines", 0))
            if lines < prev_lines:
                raise AssertionError(f"Lines must be monotonic: prev={prev_lines} now={lines}")
            prev_lines = lines
            if terminated or truncated:
                if terminated and not bool(info.get("game_over", False)):
                    raise AssertionError("terminated=True but info['game_over']=False")
                break
    finally:
        env.close()


def main() -> int:
    try:
        assert_action_mapping()
        assert_observation_contract()
        assert_step_determinism()
        assert_termination_meta_consistency()
    except ModuleNotFoundError as exc:
        print(f"python_rl_env_tests: SKIP ({exc})")
        return 0
    print("python_rl_env_tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
