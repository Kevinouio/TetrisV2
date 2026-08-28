"""Behavior-cloning warm start for the structured TetrisV2 PPO policy."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from tetris_v2.rl.expert_dataset import (
    discover_shards,
    load_dataset,
    load_dataset_directory,
)
from tetris_v2.rl.ppo.core import PPOAgent, PPOConfig


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain PPO with top-1 expert actions.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--extra-dataset-dir", type=Path, action="append", default=[])
    parser.add_argument("--updates", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[512, 256],
        help="Hidden layers for legacy MLP PPO; structured PPO uses 32 channels.",
    )
    parser.add_argument(
        "--network-type",
        choices=("auto", "placement_conv", "mlp"),
        default="auto",
    )
    parser.add_argument("--bc-coef", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--checkpoint-frequency", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/ppo_pretrain"))
    return parser.parse_args(argv)


def _load_expert_data(dataset_dir: Path, extra_dirs: list[Path]):
    if not extra_dirs:
        return load_dataset_directory(dataset_dir)
    shards = discover_shards(dataset_dir)
    for directory in extra_dirs:
        shards.extend(discover_shards(directory))
    return load_dataset(shards)


def run(args: argparse.Namespace) -> int:
    if args.updates <= 0:
        raise SystemExit("--updates must be >= 1")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be >= 1")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dataset = _load_expert_data(args.dataset_dir, args.extra_dataset_dir)
    obs_dim = int(dataset.obs.shape[-1])
    action_dim = int(dataset.action_mask.shape[-1])

    if args.init_checkpoint is not None:
        agent, _ = PPOAgent.load(
            str(args.init_checkpoint),
            device=args.device,
            restore_optimizer=False,
        )
        if agent.config.obs_dim != obs_dim or agent.config.action_dim != action_dim:
            raise SystemExit(
                f"Init checkpoint shape mismatch: ckpt=({agent.config.obs_dim},{agent.config.action_dim}) "
                f"dataset=({obs_dim},{action_dim})"
            )
        if args.network_type != "auto" and agent.network_type != args.network_type:
            raise SystemExit(
                f"Init checkpoint network_type={agent.network_type}, requested={args.network_type}"
            )
        agent.config.max_grad_norm = float(args.max_grad_norm)
        agent.reset_optimizer(args.learning_rate)
    else:
        agent = PPOAgent(
            PPOConfig(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_sizes=tuple(args.hidden_sizes),
                learning_rate=args.learning_rate,
                max_grad_norm=args.max_grad_norm,
                network_type=args.network_type,
                device=args.device,
            )
        )

    args.log_dir.mkdir(parents=True, exist_ok=True)
    running_bc = 0.0
    running_agreement = 0.0
    last_metrics = {"bc_loss": 0.0, "teacher_top1_agreement": 0.0}
    decay = 0.98

    for step in range(1, args.updates + 1):
        last_metrics = agent.pretrain_expert_batch(
            dataset.sample(args.batch_size, rng),
            coefficient=args.bc_coef,
        )
        running_bc = decay * running_bc + (1.0 - decay) * last_metrics["bc_loss"]
        running_agreement = (
            decay * running_agreement
            + (1.0 - decay) * last_metrics["teacher_top1_agreement"]
        )

        if args.log_interval > 0 and step % args.log_interval == 0:
            print(
                f"[ppo pretrain step {step:,}] bc={running_bc:.5f} "
                f"agree={running_agreement:.3f}"
            )
        if args.checkpoint_frequency > 0 and step % args.checkpoint_frequency == 0:
            agent.save(
                str(args.log_dir / f"ppo_expert_pretrain_step_{step}.pt"),
                metadata={
                    "global_step": step,
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    **last_metrics,
                },
            )

    final_path = args.log_dir / "ppo_expert_pretrain.pt"
    agent.save(
        str(final_path),
        metadata={
            "global_step": args.updates,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "bc_loss": running_bc,
            "teacher_top1_agreement": running_agreement,
        },
    )
    print(f"Saved PPO expert pretrain checkpoint to {final_path}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
