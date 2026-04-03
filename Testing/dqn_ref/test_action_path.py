from __future__ import annotations

from pathlib import Path

import pytest
import torch

from TetrisVersionTwo.scripts.bc.utils import find_library
from TetrisVersionTwo.scripts.dqn_ref.agent import DQNRefAgent
from TetrisVersionTwo.scripts.dqn_ref.config import DQNRefConfig, SEEDED_GENOME
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
        if not Path(cand).exists():
            continue
        try:
            with DQNRefEnvBridge(lib_path=Path(cand), seed=1) as env:
                env.reset(1)
            return Path(cand)
        except Exception:
            continue
    return None


@pytest.mark.skipif(_try_find_lib() is None, reason="tetris_v2 shared library not found")
def test_chosen_action_is_legal() -> None:
    lib = _try_find_lib()
    assert lib is not None
    agent = DQNRefAgent(
        genome=SEEDED_GENOME,
        config=DQNRefConfig(),
        device=torch.device("cpu"),
    )
    with DQNRefEnvBridge(lib_path=lib, seed=42) as env:
        env.reset(42)
        candidates = env.enumerate_candidates()
        assert len(candidates) > 0
        chosen = agent.get_action(candidates)
        assert chosen is not None
        legal_keys = {c.native_action.key() for c in candidates}
        assert chosen.native_action.key() in legal_keys
