from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict

from ..bc.utils import find_library
from .env_bridge import DQNRefEnvBridge


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark end-to-end episode throughput for dqn_ref candidate generation."
    )
    parser.add_argument("--lib", type=Path, default=None, help="Path to tetris_v2_c_api shared library.")
    parser.add_argument("--seed", type=int, default=1337, help="Base seed.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of benchmark episodes.")
    parser.add_argument("--max_steps", type=int, default=2000, help="Max steps per episode.")
    parser.add_argument(
        "--mode",
        choices=("auto", "batch", "legacy"),
        default="auto",
        help="Candidate API mode.",
    )
    parser.add_argument(
        "--json_out",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def _set_mode(bridge: DQNRefEnvBridge, mode: str) -> str:
    if mode == "batch":
        if not getattr(bridge, "_supports_candidate_batch", False):
            raise RuntimeError("Batch candidate API is not available in this shared library.")
        bridge._has_candidate_batch = True
    elif mode == "legacy":
        bridge._has_candidate_batch = False
    elif mode == "auto":
        bridge._has_candidate_batch = bool(getattr(bridge, "_supports_candidate_batch", False))
    return "batch" if bool(getattr(bridge, "_has_candidate_batch", False)) else "legacy"


def main() -> None:
    args = _parse_args()
    lib = find_library(args.lib)
    rng = random.Random(int(args.seed))

    with DQNRefEnvBridge(lib_path=lib, seed=int(args.seed)) as bridge:
        selected_mode = _set_mode(bridge, args.mode)

        total_steps = 0
        total_decisions = 0
        total_candidates = 0
        topouts = 0
        t0 = time.perf_counter()
        for ep in range(max(1, int(args.episodes))):
            bridge.reset(int(args.seed) + ep)
            ep_steps = 0
            for _ in range(max(1, int(args.max_steps))):
                candidates = bridge.enumerate_candidates()
                total_candidates += len(candidates)
                total_decisions += 1
                if not candidates:
                    break
                picked = candidates[rng.randrange(len(candidates))]
                step = bridge.step(picked.native_action)
                ep_steps += 1
                if bool(step.get("game_over", False)):
                    break

            total_steps += ep_steps
            state = bridge.state()
            if bool(state.get("top_out", False)):
                topouts += 1

        elapsed = max(1e-9, time.perf_counter() - t0)
        payload: Dict[str, Any] = {
            "library": str(lib),
            "mode": selected_mode,
            "episodes": int(max(1, int(args.episodes))),
            "max_steps": int(max(1, int(args.max_steps))),
            "elapsed_sec": elapsed,
            "episodes_per_sec": float(max(1, int(args.episodes)) / elapsed),
            "steps_per_sec": float(total_steps / elapsed),
            "avg_steps_per_episode": float(total_steps / max(1, int(args.episodes))),
            "avg_candidates_per_decision": float(total_candidates / max(1, total_decisions)),
            "topout_count": int(topouts),
        }

    out = json.dumps(payload, indent=2, sort_keys=True)
    print(out)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(out + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
