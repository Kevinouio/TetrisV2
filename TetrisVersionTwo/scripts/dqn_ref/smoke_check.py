from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..bc.utils import find_library
from .config import DQNRefConfig, SEEDED_GENOME
from .env_bridge import DQNRefEnvBridge
from .agent import DQNRefAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check for dqn_ref action path.")
    parser.add_argument("--lib", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lib_path = find_library(args.lib)
    device = torch.device(args.device) if args.device else torch.device("cpu")
    cfg = DQNRefConfig()
    agent = DQNRefAgent(
        genome=dict(SEEDED_GENOME),
        config=cfg,
        device=device,
    )

    with DQNRefEnvBridge(lib_path=lib_path, seed=int(args.seed)) as env:
        for ep in range(int(args.episodes)):
            env.reset(int(args.seed + ep))
            old = [0.0] * 6
            for _ in range(int(args.max_steps)):
                candidates = env.enumerate_candidates()
                if not candidates:
                    break
                chosen = agent.get_action(candidates)
                if chosen is None:
                    break
                legal_keys = {c.native_action.key() for c in candidates}
                assert chosen.native_action.key() in legal_keys, "Chosen action is not legal."
                step = env.step(chosen.native_action)
                reward_terms = agent.calculate_reward(chosen.feature_vector, finished=bool(step["game_over"]))
                agent.remember(
                    state=np.asarray(old, dtype=np.float32),
                    next_state=chosen.feature_vector,
                    reward=reward_terms.total,
                    finished=bool(step["game_over"]),
                )
                old = chosen.feature_vector.tolist()
                if bool(step["game_over"]):
                    break

    print("[dqn_ref][smoke_check] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
