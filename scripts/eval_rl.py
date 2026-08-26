"""Evaluate a TetrisV2 PPO or DQN checkpoint on the C++ environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from tetris_v2.rl.env import CCTetrisEnv
from tetris_v2.rl.evaluation import EpisodeMetrics, evaluate_gate, summarize_episodes
from tetris_v2.rl.policy import load_policy


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a TetrisV2 PPO or DQN checkpoint.")
    parser.add_argument("checkpoint", type=Path, help="Checkpoint path (.pt)")
    parser.add_argument("--algo", choices=("ppo", "dqn"), required=True)
    parser.add_argument("--episodes", type=_positive_int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=_nonnegative_int, default=4000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--temperature", type=float, default=1.0, help="PPO only")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Exploration rate when --stochastic",
    )
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=True,
        help="Use greedy policy actions (default).",
    )
    parser.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Sample/explore instead of greedy actions.",
    )
    parser.add_argument("--lib", type=Path, default=None)
    parser.add_argument(
        "--min-placements",
        type=_nonnegative_int,
        default=None,
        help="Fail unless every episode reaches this many placements.",
    )
    parser.add_argument(
        "--min-lines",
        type=_nonnegative_int,
        default=None,
        help="Fail unless every episode clears this many lines.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print only the machine-readable JSON report.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write the JSON report to PATH.",
    )
    return parser.parse_args(argv)


def _print_human_report(report: dict[str, Any]) -> None:
    episodes = report["episodes"]
    for episode in episodes:
        print(
            f"Episode {episode['episode']} seed={episode['seed']}: "
            f"placements={episode['placements']} lines={episode['lines']} "
            f"return={float(episode['return']):.2f} "
            f"topout={episode['topout']} truncated={episode['truncated']}"
        )

    summary = report["summary"]
    print(f"Summary over {summary['episode_count']} episodes:")
    for name in ("placements", "lines"):
        values = summary[name]
        print(
            f"  {name}: min={values['min']} p5={float(values['p5']):.2f} "
            f"median={float(values['median']):.2f} mean={float(values['mean']):.2f}"
        )
    print(f"  return: mean={float(summary['mean_return']):.2f}")
    print(f"  topout rate={float(summary['topout_rate']):.2%}")
    print(f"  truncation rate={float(summary['truncation_rate']):.2%}")

    gate = report["gate"]
    if gate["enabled"]:
        requirements = []
        if gate["min_placements"] is not None:
            requirements.append(f"placements>={gate['min_placements']}")
        if gate["min_lines"] is not None:
            requirements.append(f"lines>={gate['min_lines']}")
        status = "PASS" if gate["passed"] else "FAIL"
        detail = f"; failed episodes={gate['failed_episodes']}" if not gate["passed"] else ""
        print(f"Gate: {status} ({', '.join(requirements)}){detail}")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy = load_policy(args.algo, args.checkpoint, device=args.device)
    env = CCTetrisEnv(seed=args.seed, max_steps=args.max_steps, lib_path=args.lib)

    try:
        if policy.action_dim != int(env.action_space.n):
            raise SystemExit(
                f"Checkpoint action_dim={policy.action_dim} incompatible with "
                f"env action space={env.action_space.n}."
            )
        env_obs_dim = int(np.prod(env.observation_space.shape))
        if policy.obs_dim != env_obs_dim:
            raise SystemExit(
                f"Checkpoint obs_dim={policy.obs_dim} incompatible with "
                f"env observation size={env_obs_dim}."
            )

        episodes = []
        for ep in range(args.episodes):
            episode_seed = args.seed + ep
            obs, info = env.reset(seed=episode_seed)
            action_mask = np.asarray(info["action_mask"], dtype=np.float32)
            terminated = False
            truncated = False
            total_reward = 0.0
            placements = int(info["placements"])
            last_lines = int(info["lines"])
            while not (terminated or truncated):
                action = policy.act(
                    obs,
                    deterministic=args.deterministic,
                    temperature=args.temperature,
                    epsilon=args.epsilon,
                    action_mask=action_mask,
                )
                obs, reward, terminated, truncated, info = env.step(action)
                action_mask = np.asarray(info["action_mask"], dtype=np.float32)
                total_reward += float(reward)
                placements = int(info["placements"])
                last_lines = int(info["lines"])

            episodes.append(
                EpisodeMetrics(
                    episode=ep + 1,
                    seed=episode_seed,
                    placements=placements,
                    lines=last_lines,
                    episode_return=total_reward,
                    topout=bool(info["top_out"]),
                    truncated=bool(truncated),
                )
            )
    finally:
        env.close()

    gate = evaluate_gate(
        episodes,
        min_placements=args.min_placements,
        min_lines=args.min_lines,
    )
    report = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "algorithm": args.algo,
        "configuration": {
            "episodes": args.episodes,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "deterministic": args.deterministic,
            "temperature": args.temperature,
            "epsilon": args.epsilon,
        },
        "episodes": [episode.to_dict() for episode in episodes],
        "summary": summarize_episodes(episodes),
        "gate": gate,
    }
    json_report = json.dumps(report, indent=2, allow_nan=False)
    if args.json_output is not None:
        args.json_output.write_text(f"{json_report}\n", encoding="utf-8")
    if args.json:
        print(json_report)
    else:
        _print_human_report(report)
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
