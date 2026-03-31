from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import BCDataset
from .inference_agent import BCAgent
from .utils import BCEnvAdapter, find_library


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate top-1 behavioral cloning policy.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=None, help="Dataset directory for offline evaluation.")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--play_games", type=int, default=0, help="Run online gameplay for this many games.")
    parser.add_argument("--lib", type=Path, default=None, help="Path to tetris_v2_c_api shared library.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_steps_per_episode", type=int, default=2_000)
    parser.add_argument("--expert_think_ms", type=int, default=20)
    parser.add_argument("--out_json", type=Path, default=None)
    return parser.parse_args()


def topk_correct(logits: torch.Tensor, target: torch.Tensor, k: int) -> int:
    k = min(k, int(logits.shape[1]))
    topk = logits.topk(k, dim=1).indices
    correct = topk.eq(target.unsqueeze(1)).any(dim=1)
    return int(correct.sum().item())


def offline_evaluate(agent: BCAgent, data_dir: Path, split: str, batch_size: int) -> Dict[str, object]:
    dataset = BCDataset(data_dir, split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    total = 0
    top1 = 0
    top5 = 0
    true_labels: List[int] = []
    pred_labels: List[int] = []

    model = agent.model
    device = agent.device
    model.eval()

    with torch.no_grad():
        for batch in loader:
            board = batch["board"].to(device)
            aux = batch["aux"].to(device)
            target = batch["action_id"].to(device)
            logits = model(board, aux)
            pred = logits.argmax(dim=1)

            batch_size_i = int(target.shape[0])
            total += batch_size_i
            top1 += topk_correct(logits, target, k=1)
            top5 += topk_correct(logits, target, k=5)
            true_labels.extend(target.cpu().tolist())
            pred_labels.extend(pred.cpu().tolist())

    if total == 0:
        raise RuntimeError(f"Offline split '{split}' is empty.")

    per_true_wrong = defaultdict(Counter)
    true_counts = Counter(true_labels)
    for t, p in zip(true_labels, pred_labels):
        if t != p:
            per_true_wrong[t][p] += 1

    confusion_summary = []
    for class_id, count in true_counts.most_common(20):
        top_wrong = per_true_wrong[class_id].most_common(3)
        confusion_summary.append(
            {
                "class_id": int(class_id),
                "count": int(count),
                "top_wrong": [[int(pid), int(c)] for pid, c in top_wrong],
            }
        )

    return {
        "split": split,
        "num_samples": int(total),
        "top1_accuracy": float(top1 / total),
        "top5_accuracy": float(top5 / total),
        "confusion_summary": confusion_summary,
    }


def online_policy_evaluate(
    checkpoint: Path,
    lib_path: Path,
    play_games: int,
    seed: int,
    max_steps_per_episode: int,
    device: str | None,
) -> Dict[str, object]:
    lines_list = []
    pieces_list = []
    reward_list = []
    topouts = 0
    invalid_unmasked = 0
    unseen_fallback = 0
    failed_steps = 0

    with BCEnvAdapter(lib_path=lib_path, seed=seed) as env:
        agent = BCAgent(checkpoint, device=device, env_adapter=env)
        for game_idx in range(play_games):
            env.reset(seed + game_idx)
            game_reward = 0.0
            game_pieces = 0

            for _ in range(max_steps_per_episode):
                state = env.get_state()
                if bool(state["game_over"]):
                    break

                legal_actions = env.enumerate_legal_actions()
                if not legal_actions:
                    break

                action, diag = agent.predict_action_with_diagnostics(state, legal_actions=legal_actions)
                invalid_unmasked += int(bool(diag["raw_argmax_invalid"]))
                unseen_fallback += int(bool(diag["used_fallback_unseen_legal"]))

                result = env.step_native_action(action)
                if not bool(result["success"]):
                    failed_steps += 1
                    break
                game_reward += float(result["reward"])
                game_pieces += 1
                if bool(result["game_over"]):
                    break

            final_state = env.get_state()
            lines_list.append(int(final_state["lines"]))
            pieces_list.append(int(game_pieces))
            reward_list.append(float(game_reward))
            topouts += int(bool(final_state["top_out"]))

    return {
        "games": int(play_games),
        "avg_lines": float(np.mean(lines_list)) if lines_list else 0.0,
        "avg_pieces": float(np.mean(pieces_list)) if pieces_list else 0.0,
        "avg_reward": float(np.mean(reward_list)) if reward_list else 0.0,
        "topouts": int(topouts),
        "invalid_unmasked_predictions": int(invalid_unmasked),
        "unseen_legal_fallbacks": int(unseen_fallback),
        "failed_steps": int(failed_steps),
    }


def online_expert_evaluate(
    lib_path: Path,
    play_games: int,
    seed: int,
    max_steps_per_episode: int,
    think_ms: int,
) -> Dict[str, object]:
    lines_list = []
    pieces_list = []
    reward_list = []
    topouts = 0
    failed_steps = 0

    with BCEnvAdapter(lib_path=lib_path, seed=seed) as env:
        for game_idx in range(play_games):
            env.reset(seed + game_idx)
            game_reward = 0.0
            game_pieces = 0

            for _ in range(max_steps_per_episode):
                state = env.get_state()
                if bool(state["game_over"]):
                    break

                legal_actions = env.enumerate_legal_actions()
                if not legal_actions:
                    break

                step = env.expert_choose_and_apply(think_ms=think_ms)
                if not bool(step["success"]):
                    failed_steps += 1
                    break
                game_reward += float(step["reward"])
                game_pieces += 1
                if bool(step["game_over"]):
                    break

            final_state = env.get_state()
            lines_list.append(int(final_state["lines"]))
            pieces_list.append(int(game_pieces))
            reward_list.append(float(game_reward))
            topouts += int(bool(final_state["top_out"]))

    return {
        "games": int(play_games),
        "avg_lines": float(np.mean(lines_list)) if lines_list else 0.0,
        "avg_pieces": float(np.mean(pieces_list)) if pieces_list else 0.0,
        "avg_reward": float(np.mean(reward_list)) if reward_list else 0.0,
        "topouts": int(topouts),
        "failed_steps": int(failed_steps),
        "expert_think_ms": int(think_ms),
    }


def main() -> int:
    args = parse_args()
    results: Dict[str, object] = {}
    agent = BCAgent(args.checkpoint, device=args.device)

    if args.data_dir is not None:
        offline = offline_evaluate(
            agent=agent,
            data_dir=args.data_dir,
            split=args.split,
            batch_size=int(args.batch_size),
        )
        results["offline"] = offline
        print(
            "[evaluate][offline] "
            f"split={offline['split']} samples={offline['num_samples']} "
            f"top1={offline['top1_accuracy']:.4f} top5={offline['top5_accuracy']:.4f}"
        )
        print("[evaluate][offline] confusion summary (top classes):")
        for row in offline["confusion_summary"][:10]:
            print(
                f"  class={row['class_id']:4d} count={row['count']:8d} "
                f"top_wrong={row['top_wrong']}"
            )

    if args.play_games > 0:
        lib_path = find_library(args.lib)
        policy = online_policy_evaluate(
            checkpoint=args.checkpoint,
            lib_path=lib_path,
            play_games=int(args.play_games),
            seed=int(args.seed),
            max_steps_per_episode=int(args.max_steps_per_episode),
            device=args.device,
        )
        expert = online_expert_evaluate(
            lib_path=lib_path,
            play_games=int(args.play_games),
            seed=int(args.seed),
            max_steps_per_episode=int(args.max_steps_per_episode),
            think_ms=int(args.expert_think_ms),
        )
        results["online_policy"] = policy
        results["online_expert"] = expert
        print(
            "[evaluate][online][policy] "
            f"games={policy['games']} avg_lines={policy['avg_lines']:.2f} "
            f"avg_pieces={policy['avg_pieces']:.2f} avg_reward={policy['avg_reward']:.2f} "
            f"topouts={policy['topouts']} invalid_unmasked={policy['invalid_unmasked_predictions']} "
            f"fallbacks={policy['unseen_legal_fallbacks']}"
        )
        print(
            "[evaluate][online][expert] "
            f"games={expert['games']} avg_lines={expert['avg_lines']:.2f} "
            f"avg_pieces={expert['avg_pieces']:.2f} avg_reward={expert['avg_reward']:.2f} "
            f"topouts={expert['topouts']}"
        )

    if not results:
        raise SystemExit("No evaluation requested. Provide --data_dir and/or --play_games > 0.")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[evaluate] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

