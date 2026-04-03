from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from TetrisVersionTwo.scripts.bc.utils import find_library
from TetrisVersionTwo.scripts.dqn_ref.env_bridge import DQNRefEnvBridge


def _try_find_lib() -> Path | None:
    preferred = Path("build-wsl/TetrisVersionTwo/libtetris_v2_c_api.so")
    candidates = [preferred]
    try:
        candidates.append(find_library(None))
    except Exception:
        pass

    for cand in candidates:
        if cand is None:
            continue
        p = Path(cand)
        if not p.exists():
            continue
        try:
            with DQNRefEnvBridge(lib_path=p, seed=1) as env:
                env.reset(1)
            return p
        except Exception:
            continue
    return None


@pytest.mark.skipif(_try_find_lib() is None, reason="tetris_v2 shared library not found")
def test_batch_candidate_parity_with_legacy() -> None:
    lib = _try_find_lib()
    assert lib is not None
    with DQNRefEnvBridge(lib_path=lib, seed=123) as env:
        if not getattr(env, "_supports_candidate_batch", False):
            pytest.skip("batch candidate API not available in this library")

        env.reset(123)
        for step in range(5):
            env._has_candidate_batch = True
            batch = env.enumerate_candidates()

            env._has_candidate_batch = False
            legacy = env.enumerate_candidates()

            assert len(batch) == len(legacy), f"mismatch at step={step}"
            for b, l in zip(batch, legacy):
                assert b.native_action == l.native_action
                assert b.action_tuple == l.action_tuple
                assert np.allclose(b.feature_vector, l.feature_vector, atol=1e-5, rtol=1e-5)

            env._has_candidate_batch = True
            if not batch:
                break
            env.step(batch[0].native_action)

