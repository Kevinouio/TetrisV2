from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..bc.utils import find_library
from .env_bridge import DQNRefEnvBridge


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark candidate generation latency for dqn_ref (batch API vs legacy path)."
    )
    parser.add_argument("--lib", type=Path, default=None, help="Path to tetris_v2_c_api shared library.")
    parser.add_argument("--seed", type=int, default=1337, help="Base seed.")
    parser.add_argument("--samples", type=int, default=500, help="Timed candidate-enumeration samples.")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup samples not included in timing.")
    parser.add_argument(
        "--mode",
        choices=("auto", "batch", "legacy", "compare"),
        default="compare",
        help="Benchmark mode.",
    )
    parser.add_argument(
        "--parity_states",
        type=int,
        default=50,
        help="Number of states for batch-vs-legacy parity check in compare mode.",
    )
    parser.add_argument(
        "--json_out",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def _set_mode(bridge: DQNRefEnvBridge, mode: str) -> bool:
    if mode == "batch":
        if not getattr(bridge, "_supports_candidate_batch", False):
            raise RuntimeError("Batch candidate API is not available in this shared library.")
        bridge._has_candidate_batch = True
    elif mode == "legacy":
        bridge._has_candidate_batch = False
    return bool(getattr(bridge, "_has_candidate_batch", False))


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def _benchmark_mode(
    bridge: DQNRefEnvBridge,
    mode: str,
    seed: int,
    samples: int,
    warmup: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    using_batch = _set_mode(bridge, mode)
    timed_latencies_ms: List[float] = []
    candidate_counts: List[int] = []
    decisions = 0

    bridge.reset(seed)
    total = max(0, warmup) + max(1, samples)
    for i in range(total):
        t0 = time.perf_counter()
        candidates = bridge.enumerate_candidates()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if i >= warmup:
            timed_latencies_ms.append(latency_ms)
            candidate_counts.append(len(candidates))

        if candidates:
            picked = candidates[rng.randrange(len(candidates))]
            bridge.step(picked.native_action)
            decisions += 1

        if bool(bridge.state().get("game_over", False)):
            bridge.reset(seed + i + 1)

    return {
        "mode": "batch" if using_batch else "legacy",
        "samples": len(timed_latencies_ms),
        "avg_latency_ms": float(statistics.fmean(timed_latencies_ms)) if timed_latencies_ms else 0.0,
        "p50_latency_ms": _percentile(timed_latencies_ms, 50.0),
        "p95_latency_ms": _percentile(timed_latencies_ms, 95.0),
        "avg_candidate_count": float(statistics.fmean(candidate_counts)) if candidate_counts else 0.0,
        "decisions_executed": int(decisions),
    }


def _compare_parity(bridge: DQNRefEnvBridge, seed: int, states: int) -> Dict[str, Any]:
    if not getattr(bridge, "_supports_candidate_batch", False):
        return {
            "checked_states": 0,
            "mismatched_states": 0,
            "status": "skipped_no_batch_api",
        }

    rng = random.Random(seed)
    bridge.reset(seed)
    mismatches = 0

    for idx in range(max(0, states)):
        bridge._has_candidate_batch = True
        batch_candidates = bridge.enumerate_candidates()

        bridge._has_candidate_batch = False
        legacy_candidates = bridge.enumerate_candidates()

        if len(batch_candidates) != len(legacy_candidates):
            mismatches += 1
        else:
            for b, l in zip(batch_candidates, legacy_candidates):
                if b.native_action != l.native_action or b.action_tuple != l.action_tuple:
                    mismatches += 1
                    break
                if not np.allclose(b.feature_vector, l.feature_vector, atol=1e-5, rtol=1e-5):
                    mismatches += 1
                    break

        bridge._has_candidate_batch = True
        if batch_candidates:
            picked = batch_candidates[rng.randrange(len(batch_candidates))]
            bridge.step(picked.native_action)
        if bool(bridge.state().get("game_over", False)):
            bridge.reset(seed + idx + 1)

    bridge._has_candidate_batch = True
    return {
        "checked_states": int(max(0, states)),
        "mismatched_states": int(mismatches),
        "status": "ok" if mismatches == 0 else "mismatch_detected",
    }


def main() -> None:
    args = _parse_args()
    lib = find_library(args.lib)
    summary: Dict[str, Any] = {
        "library": str(lib),
        "seed": int(args.seed),
    }

    with DQNRefEnvBridge(lib_path=lib, seed=int(args.seed)) as bridge:
        if args.mode == "auto":
            selected = "batch" if getattr(bridge, "_has_candidate_batch", False) else "legacy"
            summary["result"] = _benchmark_mode(
                bridge=bridge,
                mode=selected,
                seed=int(args.seed),
                samples=int(args.samples),
                warmup=int(args.warmup),
            )
        elif args.mode in ("batch", "legacy"):
            summary["result"] = _benchmark_mode(
                bridge=bridge,
                mode=args.mode,
                seed=int(args.seed),
                samples=int(args.samples),
                warmup=int(args.warmup),
            )
        else:
            batch_res = _benchmark_mode(
                bridge=bridge,
                mode="batch",
                seed=int(args.seed),
                samples=int(args.samples),
                warmup=int(args.warmup),
            )
            legacy_res = _benchmark_mode(
                bridge=bridge,
                mode="legacy",
                seed=int(args.seed) + 10_000,
                samples=int(args.samples),
                warmup=int(args.warmup),
            )
            parity = _compare_parity(
                bridge=bridge,
                seed=int(args.seed) + 20_000,
                states=int(args.parity_states),
            )
            speedup = 0.0
            if float(batch_res["avg_latency_ms"]) > 0.0:
                speedup = float(legacy_res["avg_latency_ms"]) / float(batch_res["avg_latency_ms"])
            summary["result"] = {
                "batch": batch_res,
                "legacy": legacy_res,
                "speedup_legacy_over_batch": speedup,
                "parity": parity,
            }

    payload = json.dumps(summary, indent=2, sort_keys=True)
    print(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
