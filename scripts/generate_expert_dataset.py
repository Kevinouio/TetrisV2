"""Generate top-1 expert placement labels for TetrisV2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from tetris_v2.rl.actions import PLACEMENT_ACTION_DIM
from tetris_v2.rl.env import CCTetrisEnv
from tetris_v2.rl.expert_dataset import write_manifest, write_shard
from tetris_v2.rl.policy import load_policy


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate top-1 expert dataset shards for DQN.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/expert_dataset"))
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--think-ms", type=int, default=10)
    parser.add_argument("--random-action-prob", type=float, default=0.15)
    parser.add_argument("--behavior-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--teacher-action-prob",
        type=float,
        default=0.1,
        help="With --behavior-checkpoint, execute the teacher this fraction of the time.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lib", type=Path, default=None)
    return parser.parse_args(argv)


def _top1_label(runtime, info: dict, think_ms: int) -> dict:
    choice = runtime.bot_choose(think_ms=think_ms)
    if not choice["success"]:
        raise RuntimeError("Cold Clear failed to choose an action for the current state.")

    action_mask = np.asarray(info["action_mask"], dtype=np.uint8)
    best_action = runtime.decision_for_choice(
        use_hold=bool(choice["use_hold"]),
        placement_index=int(choice["placement_index"]),
    )
    if best_action >= PLACEMENT_ACTION_DIM or action_mask[best_action] == 0:
        raise RuntimeError(f"Cold Clear chose illegal action {best_action}.")

    legal_count = int(np.count_nonzero(action_mask))
    return {
        "action_mask": action_mask,
        "teacher_best_action": best_action,
        "placement_count_raw": int(info["placement_count_raw"]),
        "placement_overflow": bool(info["placement_overflow"]),
        "nodes": int(choice["nodes"]),
        "think_ms": float(choice["think_ms"]),
        "budget_miss": int(choice["budget_miss"]),
        "unexpanded_count": legal_count - 1,
    }


def _select_action(
    rng: np.random.Generator,
    legal: np.ndarray,
    teacher_action: int,
    random_action_prob: float,
    policy_action: Optional[int] = None,
    teacher_action_prob: float = 0.0,
) -> int:
    if rng.random() < random_action_prob:
        alternatives = legal[legal != teacher_action]
        if alternatives.size:
            return int(rng.choice(alternatives))
    if policy_action is None or rng.random() < teacher_action_prob:
        return teacher_action
    return int(policy_action)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.episodes <= 0:
        raise SystemExit("--episodes must be >= 1")
    if args.shard_size <= 0:
        raise SystemExit("--shard-size must be >= 1")
    if not (0.0 <= float(args.random_action_prob) <= 1.0):
        raise SystemExit("--random-action-prob must be in [0, 1]")
    if not (0.0 <= float(args.teacher_action_prob) <= 1.0):
        raise SystemExit("--teacher-action-prob must be in [0, 1]")

    rng = np.random.default_rng(args.seed)
    env = CCTetrisEnv(seed=args.seed, max_steps=args.max_steps, lib_path=args.lib)
    print(f"[expert dataset] using library: {env.runtime.lib._name}")
    behavior = (
        load_policy("dqn", args.behavior_checkpoint, device=args.device)
        if args.behavior_checkpoint is not None
        else None
    )
    if behavior is not None and (
        behavior.obs_dim != env.observation_space.shape[0]
        or behavior.action_dim != env.action_space.n
    ):
        env.close()
        raise SystemExit("Behavior checkpoint does not match the current environment schema.")

    records = []
    shard_paths: list[Path] = []
    shard_idx = 0
    total = 0

    def flush_shard() -> None:
        nonlocal records, shard_idx, total
        if not records:
            return
        out = args.output_dir / f"expert_shard_{shard_idx:05d}.npz"
        write_shard(out, records)
        shard_paths.append(out)
        total += len(records)
        records = []
        shard_idx += 1

    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            terminated = False
            truncated = False
            step = 0
            while not (terminated or truncated):
                label = _top1_label(env.runtime, info, args.think_ms)

                legal = np.flatnonzero(label["action_mask"] > 0)
                teacher_action = int(label["teacher_best_action"])
                policy_action = (
                    behavior.act(
                        obs,
                        deterministic=True,
                        action_mask=np.asarray(label["action_mask"], dtype=np.float32),
                    )
                    if behavior is not None
                    else None
                )
                action = _select_action(
                    rng,
                    legal,
                    teacher_action,
                    float(args.random_action_prob),
                    policy_action=policy_action,
                    teacher_action_prob=float(args.teacher_action_prob),
                )

                records.append(
                    {
                        "obs": np.asarray(obs, dtype=np.float32),
                        "action_mask": np.asarray(label["action_mask"], dtype=np.uint8),
                        "teacher_best_action": teacher_action,
                        "seed": int(args.seed + ep),
                        "episode": int(ep),
                        "step": int(step),
                        "legal_action_count": int(legal.size),
                        "placement_count_raw": int(label["placement_count_raw"]),
                        "placement_overflow": int(label["placement_overflow"]),
                        "nodes": int(label["nodes"]),
                        "think_ms": float(label["think_ms"]),
                        "budget_miss": int(label["budget_miss"]),
                        "unexpanded_count": int(label["unexpanded_count"]),
                    }
                )

                obs, _, terminated, truncated, info = env.step(action)
                step += 1

                if len(records) >= args.shard_size:
                    flush_shard()

            if (ep + 1) % 10 == 0:
                print(f"[expert dataset] episodes={ep + 1}/{args.episodes} samples={total + len(records)}")
    finally:
        env.close()

    flush_shard()
    write_manifest(args.output_dir / "manifest.json", shards=shard_paths, total_samples=total)
    summary = {
        "episodes": int(args.episodes),
        "samples": int(total),
        "shards": [p.name for p in shard_paths],
        "label_mode": "top1",
        "think_ms": int(args.think_ms),
        "seed": int(args.seed),
        "behavior_checkpoint": (
            str(args.behavior_checkpoint) if args.behavior_checkpoint is not None else None
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {total} expert samples to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
