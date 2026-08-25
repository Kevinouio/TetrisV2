"""Supervised expert warm-start for the TetrisV2 DQN agent."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tetris_v2.rl.dqn.core import DQNAgent, DQNConfig
from tetris_v2.rl.dqn.expert_losses import expert_aux_losses
from tetris_v2.rl.expert_dataset import discover_shards, load_dataset, load_dataset_directory


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain DQN with top-1 expert labels.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--extra-dataset-dir", type=Path, action="append", default=[])
    parser.add_argument("--updates", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--lambda-bc", type=float, default=1.0)
    parser.add_argument("--lambda-pair", type=float, default=0.0)
    parser.add_argument("--pairs-per-sample", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--checkpoint-frequency", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/dqn_pretrain"))
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.updates <= 0:
        raise SystemExit("--updates must be >= 1")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be >= 1")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.extra_dataset_dir:
        shards = discover_shards(args.dataset_dir)
        for directory in args.extra_dataset_dir:
            shards.extend(discover_shards(directory))
        dataset = load_dataset(shards)
    else:
        dataset = load_dataset_directory(args.dataset_dir)
    obs_dim = int(dataset.obs.shape[-1])
    action_dim = int(dataset.action_mask.shape[-1])

    if args.init_checkpoint is not None:
        agent, _ = DQNAgent.load(str(args.init_checkpoint), device=args.device)
        if int(agent.config.obs_dim) != obs_dim or int(agent.config.action_dim) != action_dim:
            raise SystemExit(
                f"Init checkpoint shape mismatch: ckpt=({agent.config.obs_dim},{agent.config.action_dim}) "
                f"dataset=({obs_dim},{action_dim})"
            )
        agent.config.learning_rate = float(args.learning_rate)
        for group in agent.optimizer.param_groups:
            group["lr"] = float(args.learning_rate)
    else:
        cfg = DQNConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=tuple(args.hidden_sizes),
            learning_rate=args.learning_rate,
            device=args.device,
        )
        agent = DQNAgent(cfg)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    next_ckpt = args.checkpoint_frequency

    avg_bc = 0.0
    avg_pair = 0.0
    avg_agree = 0.0
    decay = 0.98

    for step in range(1, args.updates + 1):
        batch = dataset.sample(args.batch_size, rng)
        obs = torch.from_numpy(batch["obs"]).to(agent.device)
        teacher_best = torch.from_numpy(batch["teacher_best_action"]).to(agent.device)
        action_mask = torch.from_numpy(batch["action_mask"]).to(agent.device)
        q_values = agent.online(obs)
        bc_loss, pair_loss, agreement = expert_aux_losses(
            q_values,
            teacher_best,
            action_mask,
            rng=rng,
            pairs_per_sample=max(0, int(args.pairs_per_sample)),
        )
        total = float(args.lambda_bc) * bc_loss + float(args.lambda_pair) * pair_loss

        agent.optimizer.zero_grad()
        total.backward()
        if agent.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(agent.online.parameters(), agent.config.max_grad_norm)
        agent.optimizer.step()

        if step % 100 == 0:
            agent.sync_target()

        bc_v = float(bc_loss.detach().item())
        pair_v = float(pair_loss.detach().item())
        agree_v = float(agreement.detach().item())
        avg_bc = decay * avg_bc + (1.0 - decay) * bc_v
        avg_pair = decay * avg_pair + (1.0 - decay) * pair_v
        avg_agree = decay * avg_agree + (1.0 - decay) * agree_v

        if step % max(1, args.log_interval) == 0:
            print(
                f"[pretrain step {step:,}] bc={avg_bc:.5f} pair={avg_pair:.5f} "
                f"agree={avg_agree:.3f}"
            )

        if step >= next_ckpt:
            ckpt = args.log_dir / f"dqn_expert_pretrain_step_{step}.pt"
            agent.save(
                str(ckpt),
                metadata={
                    "global_step": float(step),
                    "obs_dim": float(obs_dim),
                    "action_dim": float(action_dim),
                    "bc_loss": float(bc_v),
                    "pair_loss": float(pair_v),
                    "teacher_top1_agreement": float(agree_v),
                },
            )
            next_ckpt += args.checkpoint_frequency

    final_path = args.log_dir / "dqn_expert_pretrain.pt"
    agent.save(
        str(final_path),
        metadata={
            "global_step": float(args.updates),
            "obs_dim": float(obs_dim),
            "action_dim": float(action_dim),
            "bc_loss": float(avg_bc),
            "pair_loss": float(avg_pair),
            "teacher_top1_agreement": float(avg_agree),
        },
    )
    print(f"Saved expert pretrain checkpoint to {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
