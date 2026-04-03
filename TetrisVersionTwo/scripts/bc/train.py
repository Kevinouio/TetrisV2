from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from .config import ModelConfig, TrainConfig, parse_int_tuple
from .dataset import BCDataset, class_histogram, load_metadata
from .model import BCPolicyNet
from .utils import configure_cpu_runtime, ensure_dir, set_global_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train top-1 behavioral cloning policy.")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("runs/bc_top1"))
    parser.add_argument(
        "--init_checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to warm-start from (supports vocab growth by partial head copy).",
    )
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--learning_rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pin host memory in DataLoader (default: auto on CUDA).",
    )
    parser.add_argument(
        "--persistent_workers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep DataLoader workers alive across epochs (default: enabled when num_workers > 0).",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor when num_workers > 0.",
    )
    parser.add_argument("--torch_num_threads", type=int, default=0)
    parser.add_argument("--torch_num_interop_threads", type=int, default=0)
    parser.add_argument("--omp_num_threads", type=int, default=0)
    parser.add_argument("--mkl_num_threads", type=int, default=0)
    parser.add_argument("--openblas_num_threads", type=int, default=0)
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=0,
        help="If > 0, cap train samples to this exact count via deterministic sampling.",
    )
    parser.add_argument("--conv_channels", type=str, default="32,64,64")
    parser.add_argument("--mlp_hidden", type=str, default="256,256")
    parser.add_argument(
        "--overfit_samples",
        type=int,
        default=0,
        help="If > 0, overfit this many samples for sanity checking.",
    )
    return parser.parse_args()


def select_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def topk_correct(logits: torch.Tensor, target: torch.Tensor, k: int) -> int:
    k = min(k, int(logits.shape[1]))
    topk = logits.topk(k, dim=1).indices
    correct = topk.eq(target.unsqueeze(1)).any(dim=1)
    return int(correct.sum().item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    total_count = 0

    for batch in loader:
        board = batch["board"].to(device)
        aux = batch["aux"].to(device)
        target = batch["action_id"].to(device)

        logits = model(board, aux)
        loss = criterion(logits, target)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = int(target.shape[0])
        total_count += batch_size
        total_loss += float(loss.item()) * batch_size
        total_top1 += topk_correct(logits, target, k=1)
        total_top5 += topk_correct(logits, target, k=5)

    if total_count == 0:
        return {"loss": 0.0, "top1": 0.0, "top5": 0.0}
    return {
        "loss": total_loss / total_count,
        "top1": total_top1 / total_count,
        "top5": total_top5 / total_count,
    }


def make_subset(dataset: Dataset, limit: int) -> Dataset:
    size = min(len(dataset), max(1, int(limit)))
    return Subset(dataset, list(range(size)))


def save_checkpoint(
    path: Path,
    model: BCPolicyNet,
    model_config: Dict[str, object],
    encoder_config: Dict[str, object],
    id_to_action: Sequence[Sequence[int]],
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "encoder_config": encoder_config,
        "id_to_action": [list(v) for v in id_to_action],
        "epoch": int(epoch),
        "metrics": metrics,
    }
    torch.save(payload, path)


def serialization_sanity(
    checkpoint_path: Path,
    board_batch: torch.Tensor,
    aux_batch: torch.Tensor,
    device: torch.device,
) -> bool:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = checkpoint["model_config"]
    kwargs = dict(
        action_vocab_size=int(model_cfg["action_vocab_size"]),
        aux_dim=int(model_cfg["aux_dim"]),
        board_height=int(model_cfg["board_height"]),
        board_width=int(model_cfg["board_width"]),
        conv_channels=tuple(int(v) for v in model_cfg["conv_channels"]),
        mlp_hidden=tuple(int(v) for v in model_cfg["mlp_hidden"]),
    )
    model_a = BCPolicyNet(**kwargs).to(device)
    model_b = BCPolicyNet(**kwargs).to(device)
    model_a.load_state_dict(checkpoint["model_state_dict"])
    model_b.load_state_dict(checkpoint["model_state_dict"])
    model_a.eval()
    model_b.eval()

    with torch.no_grad():
        out1 = model_a(board_batch.to(device), aux_batch.to(device)).cpu()
        out2 = model_b(board_batch.to(device), aux_batch.to(device)).cpu()
    return torch.allclose(out1, out2, atol=1e-7, rtol=1e-6)


def load_warm_start_checkpoint(model: BCPolicyNet, checkpoint_path: Path) -> Dict[str, int]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"init checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", None)
    if not isinstance(state, dict):
        raise ValueError(f"Invalid checkpoint model_state_dict: {checkpoint_path}")

    dst_state = model.state_dict()
    loaded_exact = 0
    loaded_partial = 0
    skipped = 0

    for key, src_val in state.items():
        if key not in dst_state:
            skipped += 1
            continue
        dst_val = dst_state[key]
        if src_val.shape == dst_val.shape:
            dst_val.copy_(src_val)
            loaded_exact += 1
            continue

        can_partial = (
            key.startswith("head.")
            and src_val.ndim == dst_val.ndim
            and src_val.ndim in (1, 2)
        )
        if not can_partial:
            skipped += 1
            continue

        if src_val.ndim == 2:
            if src_val.shape[1] != dst_val.shape[1]:
                skipped += 1
                continue
            rows = min(int(src_val.shape[0]), int(dst_val.shape[0]))
            dst_val[:rows, :].copy_(src_val[:rows, :])
            loaded_partial += 1
        else:
            rows = min(int(src_val.shape[0]), int(dst_val.shape[0]))
            dst_val[:rows].copy_(src_val[:rows])
            loaded_partial += 1

    model.load_state_dict(dst_state)
    return {
        "loaded_exact_tensors": int(loaded_exact),
        "loaded_partial_tensors": int(loaded_partial),
        "skipped_tensors": int(skipped),
    }


def main() -> int:
    args = parse_args()
    if int(args.max_train_samples) < 0:
        raise ValueError("--max_train_samples must be >= 0.")
    if int(args.prefetch_factor) <= 0:
        raise ValueError("--prefetch_factor must be > 0.")
    configure_cpu_runtime(
        torch_num_threads=max(0, int(args.torch_num_threads)),
        torch_num_interop_threads=max(0, int(args.torch_num_interop_threads)),
        omp_num_threads=max(0, int(args.omp_num_threads)),
        mkl_num_threads=max(0, int(args.mkl_num_threads)),
        openblas_num_threads=max(0, int(args.openblas_num_threads)),
    )
    set_global_seeds(args.seed)
    device = select_device(args.device)

    conv_channels = parse_int_tuple(args.conv_channels)
    mlp_hidden = parse_int_tuple(args.mlp_hidden)
    model_hparams = ModelConfig(
        conv_channels=tuple(int(v) for v in conv_channels),
        mlp_hidden=tuple(int(v) for v in mlp_hidden),
    )

    metadata = load_metadata(Path(args.data_dir))
    encoder_config = metadata.get("encoder_config", {})
    if not isinstance(encoder_config, dict):
        raise ValueError("metadata['encoder_config'] must be a dictionary.")
    id_to_action = metadata.get("id_to_action", [])
    if not isinstance(id_to_action, list):
        raise ValueError("metadata['id_to_action'] must be a list.")
    board_shape = metadata.get("board_shape", [20, 10])
    if not isinstance(board_shape, list) or len(board_shape) != 2:
        raise ValueError("metadata['board_shape'] must be [height, width].")

    train_ds: Dataset = BCDataset(args.data_dir, split="train")
    val_ds: Dataset = BCDataset(args.data_dir, split="val")
    num_train_available = int(len(train_ds))
    train_cap_applied = False
    if args.overfit_samples > 0:
        train_ds = make_subset(train_ds, args.overfit_samples)
        val_ds = make_subset(val_ds, args.overfit_samples)
        print(f"[train] overfit mode enabled with {len(train_ds)} samples.")
    elif int(args.max_train_samples) > 0 and len(train_ds) > int(args.max_train_samples):
        max_samples = int(args.max_train_samples)
        sample_gen = torch.Generator()
        sample_gen.manual_seed(int(args.seed))
        sampled_indices = torch.randperm(len(train_ds), generator=sample_gen)[:max_samples].tolist()
        train_ds = Subset(train_ds, sampled_indices)
        train_cap_applied = True
        print(
            f"[train] applying deterministic train cap: "
            f"available={num_train_available} used={len(train_ds)} max_train_samples={max_samples}"
        )

    pin_memory = bool(args.pin_memory) if args.pin_memory is not None else (device.type != "cpu")
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "drop_last": False,
        "pin_memory": pin_memory,
    }
    if int(args.num_workers) > 0:
        loader_kwargs["persistent_workers"] = (
            bool(args.persistent_workers) if args.persistent_workers is not None else True
        )
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_kwargs,
    )

    train_dataset_ref = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
    if not isinstance(train_dataset_ref, BCDataset):
        raise RuntimeError("Unexpected training dataset type.")

    model = BCPolicyNet(
        action_vocab_size=int(train_dataset_ref.action_vocab_size),
        aux_dim=int(train_dataset_ref.aux_dim),
        board_height=int(board_shape[0]),
        board_width=int(board_shape[1]),
        conv_channels=model_hparams.conv_channels,
        mlp_hidden=model_hparams.mlp_hidden,
    ).to(device)
    warm_start_stats: Optional[Dict[str, int]] = None
    if args.init_checkpoint is not None:
        warm_start_stats = load_warm_start_checkpoint(model, args.init_checkpoint)
        print(
            "[train] warm_start "
            f"checkpoint={args.init_checkpoint} "
            f"exact={warm_start_stats['loaded_exact_tensors']} "
            f"partial={warm_start_stats['loaded_partial_tensors']} "
            f"skipped={warm_start_stats['skipped_tensors']}"
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    patience_count = 0

    print(f"[train] device={device} train_samples={len(train_ds)} val_samples={len(val_ds)}")
    print("[train] top train classes:")
    for class_id, count in class_histogram(train_dataset_ref.action_id, top_k=10):
        print(f"  class={class_id:4d} count={count:8d}")

    model_config_dict = {
        "action_vocab_size": int(train_dataset_ref.action_vocab_size),
        "aux_dim": int(train_dataset_ref.aux_dim),
        "board_height": int(board_shape[0]),
        "board_width": int(board_shape[1]),
        "conv_channels": [int(v) for v in model_hparams.conv_channels],
        "mlp_hidden": [int(v) for v in model_hparams.mlp_hidden],
    }

    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_metrics["loss"]),
            "train_top1": float(train_metrics["top1"]),
            "train_top5": float(train_metrics["top5"]),
            "val_loss": float(val_metrics["loss"]),
            "val_top1": float(val_metrics["top1"]),
            "val_top5": float(val_metrics["top5"]),
        }
        history.append(row)
        print(
            f"[train] epoch={epoch:03d} "
            f"train_loss={row['train_loss']:.4f} train_top1={row['train_top1']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_top1={row['val_top1']:.4f}"
        )

        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            best_epoch = epoch
            patience_count = 0
            save_checkpoint(
                best_path,
                model=model,
                model_config=model_config_dict,
                encoder_config=encoder_config,
                id_to_action=id_to_action,
                epoch=epoch,
                metrics={
                    "val_loss": row["val_loss"],
                    "val_top1": row["val_top1"],
                    "val_top5": row["val_top5"],
                },
            )
        else:
            patience_count += 1
            if patience_count >= int(args.patience):
                print(
                    f"[train] early stopping at epoch={epoch} "
                    f"(best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f})."
                )
                break

    final_metrics = history[-1] if history else {}
    save_checkpoint(
        last_path,
        model=model,
        model_config=model_config_dict,
        encoder_config=encoder_config,
        id_to_action=id_to_action,
        epoch=int(final_metrics.get("epoch", 0)),
        metrics={
            "val_loss": float(final_metrics.get("val_loss", 0.0)),
            "val_top1": float(final_metrics.get("val_top1", 0.0)),
            "val_top5": float(final_metrics.get("val_top5", 0.0)),
        },
    )

    metrics_json_path = out_dir / "metrics_history.json"
    metrics_json_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    metrics_csv_path = out_dir / "metrics_history.csv"
    with metrics_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_top1",
                "train_top5",
                "val_loss",
                "val_top1",
                "val_top5",
            ],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(row)

    sanity_loader = DataLoader(val_ds, batch_size=min(64, max(1, len(val_ds))), shuffle=False)
    sanity_batch = next(iter(sanity_loader))
    serialization_ok = serialization_sanity(
        checkpoint_path=best_path,
        board_batch=sanity_batch["board"],
        aux_batch=sanity_batch["aux"],
        device=device,
    )
    print(f"[train] serialization_sanity={'PASS' if serialization_ok else 'FAIL'}")

    summary = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "serialization_sanity": bool(serialization_ok),
        "num_train_samples": int(len(train_ds)),
        "num_train_samples_available": int(num_train_available),
        "num_train_samples_used": int(len(train_ds)),
        "max_train_samples": int(args.max_train_samples),
        "train_cap_applied": bool(train_cap_applied),
        "num_val_samples": int(len(val_ds)),
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
        "warm_start": warm_start_stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[train] wrote checkpoints to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
