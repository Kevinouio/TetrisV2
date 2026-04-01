from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


ROUND_RE = re.compile(r"^round_(\d+)$")
WSL_PATH_RE = re.compile(r"^/mnt/([a-zA-Z])/(.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prune heavy BC/DAgger generated artifacts with safe defaults."
    )
    parser.add_argument("--runs_root", type=Path, default=Path("runs"))
    parser.add_argument("--keep_rounds", type=int, default=2)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply deletions. Without this flag, only a dry-run preview is produced.",
    )
    parser.add_argument(
        "--prune_old_dagger_shards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete old round dagger_train shard files (*.pt) outside kept rounds.",
    )
    parser.add_argument(
        "--prune_old_aggregated_data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete old round aggregated_data files outside kept rounds.",
    )
    parser.add_argument("--base_data_dir", type=Path, default=Path("data/bc_top1"))
    parser.add_argument(
        "--json_report",
        type=Path,
        default=None,
        help="Optional output path for machine-readable cleanup report JSON.",
    )
    return parser.parse_args()


def to_abs(path: Path) -> Path:
    return path.expanduser().resolve()


def to_abs_soft(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def normalize_maybe_wsl_path(raw: str) -> Path:
    value = raw.strip()
    if os.name == "nt":
        match = WSL_PATH_RE.match(value)
        if match:
            drive = match.group(1).upper()
            suffix = match.group(2).replace("/", "\\")
            return Path(f"{drive}:\\{suffix}")
    return Path(value)


def format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024.0:.2f} KiB"
    if num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024.0 * 1024.0):.2f} MiB"
    return f"{num_bytes / (1024.0 * 1024.0 * 1024.0):.2f} GiB"


def find_candidate_run_dirs(runs_root: Path) -> Tuple[List[Path], int]:
    scanned = 0
    out: List[Path] = []
    if not runs_root.exists():
        return out, scanned
    for child in sorted(runs_root.iterdir()):
        if not child.is_dir():
            continue
        scanned += 1
        has_summary = (child / "dagger_summary.json").is_file()
        has_rounds = any(
            p.is_dir() and ROUND_RE.match(p.name) is not None for p in child.iterdir()
        )
        if has_summary or has_rounds:
            out.append(child)
    return out, scanned


def discover_round_dirs(run_dir: Path) -> List[Tuple[int, Path]]:
    rounds: List[Tuple[int, Path]] = []
    for item in run_dir.iterdir():
        if not item.is_dir():
            continue
        match = ROUND_RE.match(item.name)
        if not match:
            continue
        rounds.append((int(match.group(1)), item))
    rounds.sort(key=lambda x: x[0])
    return rounds


def choose_keep_round_ids(rounds: Sequence[Tuple[int, Path]], keep_rounds: int) -> Set[int]:
    if keep_rounds <= 0:
        return set()
    return {rid for rid, _ in rounds[-keep_rounds:]}


def load_latest_checkpoint_from_summary(run_dir: Path, errors: List[Dict[str, str]]) -> Optional[Path]:
    summary_path = run_dir / "dagger_summary.json"
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        raw = payload.get("latest_checkpoint")
        if not isinstance(raw, str) or not raw.strip():
            return None
        return to_abs_soft(normalize_maybe_wsl_path(raw))
    except Exception as exc:
        errors.append(
            {
                "path": str(summary_path),
                "error": f"Failed reading latest_checkpoint: {type(exc).__name__}: {exc}",
            }
        )
        return None


def is_protected(path: Path, protected_paths: Sequence[Path]) -> bool:
    for protected in protected_paths:
        if protected.is_dir():
            if path_within(path, protected):
                return True
        else:
            if path == protected:
                return True
    return False


def collect_file_candidates(
    *,
    run_dir: Path,
    rounds: Sequence[Tuple[int, Path]],
    keep_round_ids: Set[int],
    runs_root: Path,
    protected_paths: Sequence[Path],
    prune_old_dagger_shards: bool,
    prune_old_aggregated_data: bool,
    skipped_paths: List[Dict[str, str]],
) -> List[Path]:
    candidates: List[Path] = []
    seen: Set[Path] = set()
    for rid, round_dir in rounds:
        if rid in keep_round_ids:
            continue
        roots: List[Path] = []
        if prune_old_dagger_shards:
            roots.append(round_dir / "dagger_train" / "shards")
        if prune_old_aggregated_data:
            roots.append(round_dir / "aggregated_data")
        for root in roots:
            if not root.exists() or not root.is_dir():
                continue
            pattern_files: List[Path]
            if root.name == "shards":
                pattern_files = sorted(root.rglob("*.pt"))
            else:
                pattern_files = sorted(p for p in root.rglob("*") if p.is_file())
            for file_path in pattern_files:
                resolved = to_abs_soft(file_path)
                if file_path.is_symlink():
                    skipped_paths.append(
                        {"path": str(resolved), "reason": "symlink_file_skipped"}
                    )
                    continue
                if not path_within(resolved, runs_root):
                    skipped_paths.append(
                        {"path": str(resolved), "reason": "outside_runs_root_skipped"}
                    )
                    continue
                if is_protected(resolved, protected_paths):
                    skipped_paths.append(
                        {"path": str(resolved), "reason": "protected_path_skipped"}
                    )
                    continue
                if resolved in seen:
                    continue
                seen.add(resolved)
                candidates.append(resolved)
    return sorted(candidates)


def remove_empty_dirs_under(root: Path, skipped_paths: List[Dict[str, str]]) -> int:
    if not root.exists() or not root.is_dir():
        return 0
    removed = 0
    dirs = [p for p in root.rglob("*") if p.is_dir()]
    dirs.append(root)
    dirs.sort(key=lambda p: len(p.parts), reverse=True)
    for d in dirs:
        if d.is_symlink():
            skipped_paths.append(
                {"path": str(to_abs_soft(d)), "reason": "symlink_dir_skipped"}
            )
            continue
        try:
            if any(d.iterdir()):
                continue
            d.rmdir()
            removed += 1
        except OSError:
            pass
    return removed


def build_run_report(
    *,
    run_dir: Path,
    rounds: Sequence[Tuple[int, Path]],
    keep_round_ids: Set[int],
    candidates: Sequence[Path],
) -> Dict[str, object]:
    reclaimable = 0
    for p in candidates:
        try:
            reclaimable += int(p.stat().st_size)
        except OSError:
            pass
    return {
        "run_dir": str(run_dir),
        "round_ids": [int(rid) for rid, _ in rounds],
        "kept_round_ids": sorted(int(v) for v in keep_round_ids),
        "planned_delete_count": int(len(candidates)),
        "bytes_reclaimable": int(reclaimable),
    }


def main() -> int:
    args = parse_args()
    if int(args.keep_rounds) < 0:
        raise ValueError("--keep_rounds must be >= 0.")

    runs_root = to_abs(args.runs_root)
    base_data_dir = to_abs_soft(args.base_data_dir)
    if not runs_root.exists():
        raise FileNotFoundError(f"--runs_root does not exist: {runs_root}")

    apply_mode = bool(args.apply)
    run_dirs, runs_scanned = find_candidate_run_dirs(runs_root)
    skipped_paths: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []
    run_reports: List[Dict[str, object]] = []
    planned_deletions: List[str] = []
    executed_deletions: List[str] = []
    protected_paths: List[Path] = [base_data_dir]
    planned_paths: List[Path] = []

    print(
        f"[cleanup] mode={'APPLY' if apply_mode else 'DRY-RUN'} "
        f"runs_root={runs_root} keep_rounds={int(args.keep_rounds)}"
    )

    for run_dir in run_dirs:
        rounds = discover_round_dirs(run_dir)
        keep_round_ids = choose_keep_round_ids(rounds, int(args.keep_rounds))
        latest_checkpoint = load_latest_checkpoint_from_summary(run_dir, errors)
        run_protected = list(protected_paths)
        if latest_checkpoint is not None:
            run_protected.append(latest_checkpoint)
        candidates = collect_file_candidates(
            run_dir=run_dir,
            rounds=rounds,
            keep_round_ids=keep_round_ids,
            runs_root=runs_root,
            protected_paths=run_protected,
            prune_old_dagger_shards=bool(args.prune_old_dagger_shards),
            prune_old_aggregated_data=bool(args.prune_old_aggregated_data),
            skipped_paths=skipped_paths,
        )
        planned_paths.extend(candidates)
        run_report = build_run_report(
            run_dir=run_dir,
            rounds=rounds,
            keep_round_ids=keep_round_ids,
            candidates=candidates,
        )
        run_reports.append(run_report)
        print(
            f"[cleanup][run] name={run_dir.name} rounds={len(rounds)} "
            f"keep={run_report['kept_round_ids']} planned_files={run_report['planned_delete_count']} "
            f"reclaimable={format_bytes(int(run_report['bytes_reclaimable']))}"
        )

    planned_paths = sorted(set(planned_paths))
    bytes_reclaimable = 0
    for p in planned_paths:
        try:
            bytes_reclaimable += int(p.stat().st_size)
        except OSError:
            pass
    planned_deletions = [str(p) for p in planned_paths]

    bytes_reclaimed = 0
    removed_empty_dirs = 0
    if apply_mode:
        for p in planned_paths:
            try:
                if not p.exists() or not p.is_file():
                    skipped_paths.append({"path": str(p), "reason": "missing_at_apply_time"})
                    continue
                size = int(p.stat().st_size)
                p.unlink()
                bytes_reclaimed += size
                executed_deletions.append(str(p))
            except Exception as exc:
                errors.append(
                    {
                        "path": str(p),
                        "error": f"Failed deleting file: {type(exc).__name__}: {exc}",
                    }
                )

        if bool(args.prune_old_dagger_shards):
            for run_dir in run_dirs:
                rounds = discover_round_dirs(run_dir)
                keep_round_ids = choose_keep_round_ids(rounds, int(args.keep_rounds))
                for rid, round_dir in rounds:
                    if rid in keep_round_ids:
                        continue
                    removed_empty_dirs += remove_empty_dirs_under(
                        round_dir / "dagger_train",
                        skipped_paths=skipped_paths,
                    )

    protected_paths_str = [str(base_data_dir)]
    for run_dir in run_dirs:
        latest_checkpoint = load_latest_checkpoint_from_summary(run_dir, errors=[])
        if latest_checkpoint is not None:
            protected_paths_str.append(str(latest_checkpoint))
    protected_paths_str = sorted(set(protected_paths_str))

    report: Dict[str, object] = {
        "runs_scanned": int(runs_scanned),
        "runs_processed": int(len(run_dirs)),
        "keep_rounds": int(args.keep_rounds),
        "protected_paths": protected_paths_str,
        "planned_deletions": planned_deletions,
        "executed_deletions": executed_deletions,
        "bytes_reclaimable": int(bytes_reclaimable),
        "bytes_reclaimed": int(bytes_reclaimed),
        "removed_empty_dirs": int(removed_empty_dirs),
        "skipped_paths": skipped_paths,
        "errors": errors,
        "runs": run_reports,
        "mode": "apply" if apply_mode else "dry_run",
        "prune_old_dagger_shards": bool(args.prune_old_dagger_shards),
        "prune_old_aggregated_data": bool(args.prune_old_aggregated_data),
    }

    print(
        f"[cleanup] runs_processed={len(run_dirs)} "
        f"planned_files={len(planned_deletions)} reclaimable={format_bytes(bytes_reclaimable)} "
        f"deleted_files={len(executed_deletions)} reclaimed={format_bytes(bytes_reclaimed)}"
    )
    if errors:
        print(f"[cleanup] errors={len(errors)} (see report for details)")

    if args.json_report is not None:
        out_path = to_abs_soft(args.json_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[cleanup] wrote report: {out_path}")

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

