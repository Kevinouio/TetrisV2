# Project Plan (TetrisVersionTwo focus)

## 1) Reproducible env (Docker)
- Add root Dockerfile that builds cold_clear_cpp + tetris_v2 targets and installs Python viewer deps.
- CI hook: build image, run `ctest` in container (optional for speed).
- Publish image tag (e.g., ghcr.io/<org>/tetrisv2:latest).

## 2) Expert data generation (Cold Clear)
- Extend `cc_play` to emit JSONL with full state + chosen Placement (board bits, hold, queue, bag mask, move, reward info, softdrop).
- Script `scripts/gen_expert_runs.py`: run N seeds, call `cc_play --json`, collect episodes, shard output, store metadata (seed, weights hash, build hash).
- Dataset schema doc + small sample in `Recordings/` or `data/samples/`.

## 3) FlowQ / IL training pipeline
- Define observation encoding for NN (board bitplanes, queue, hold, b2b/combo, bag).
- Model: flow-based Q or policy head; config in `configs/flowq.yaml`.
- Training script `scripts/train_flowq.py`: load dataset, supervised Q/policy loss on expert actions, eval hooks.
- Evaluation script `scripts/eval_flowq.py`: run trained agent in env, report lines survived / score vs. expert baseline.
- Logging/checkpointing layout (`runs/flowq/...`).

## 4) Env/agent glue for training
- Add lightweight Python wrapper to call C++ env and Cold Clear agent (ctypes/pybind or CLI pipe) for online rollout tests.
- Multithreading for training and optimize the memory usage and etc when training.

## 5) Quality gates
- C++: keep smoke tests (`tetris_v2_tests`, `tetris_cc_tests`); add test for JSONL logging correctness.
- Python: unit test dataset parsing + FlowQ batch shapes.
- Lint/format: clang-format + black/ruff (optional).

## 6) Documentation
- README section on Docker usage and how to generate expert data + train FlowQ.
- Notes on weight/version hashing to keep datasets traceable.
