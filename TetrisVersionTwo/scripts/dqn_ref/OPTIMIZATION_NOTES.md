# Candidate Generation Optimization Notes

This note tracks the move-generation optimization that reduces repeated placement recomputation and Python<->C API overhead.

## What Changed
- `ModernTetrisEnv` now caches active-piece placement options by state revision.
- C API adds batch candidate exports:
  - `tetris_cc_env_candidate_count`
  - `tetris_cc_env_candidate_get`
  - `tetris_cc_env_candidate_features_write`
- `dqn_ref` bridge now prefers the batch path and falls back to legacy per-placement calls if symbols are unavailable.

## Benchmark Commands

```bash
python -m dqn_ref.bench_candidates \
  --lib build-wsl/TetrisVersionTwo/libtetris_v2_c_api.so \
  --mode compare \
  --samples 500 \
  --warmup 50 \
  --parity_states 50 \
  --json_out runs/dqn_ref_bench/candidate_compare.json
```

```bash
python -m dqn_ref.bench_throughput \
  --lib build-wsl/TetrisVersionTwo/libtetris_v2_c_api.so \
  --mode auto \
  --episodes 100 \
  --max_steps 2000 \
  --json_out runs/dqn_ref_bench/throughput_auto.json
```

## Before/After Summary Template

| Metric | Before (legacy) | After (batch/cache) | Speedup |
|---|---:|---:|---:|
| Avg candidate latency (ms) | `TBD` | `TBD` | `TBD` |
| P95 candidate latency (ms) | `TBD` | `TBD` | `TBD` |
| Steps/sec (throughput bench) | `TBD` | `TBD` | `TBD` |
| Episodes/sec (throughput bench) | `TBD` | `TBD` | `TBD` |

## Notes
- Candidate parity check compares batch vs legacy tuples and feature vectors on sampled states.
- Benchmark in the same runtime configuration (WSL + same CPU governor / conda env) for fair comparison.

