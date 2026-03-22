"""Short PPO/DQN train+eval smoke for VersionTwo RL stack."""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _assert_checkpoint(path: Path, algo: str) -> None:
    from TetrisVersionTwo.rl.policy import load_policy

    if not path.exists():
        raise AssertionError(f"Missing {algo} checkpoint: {path}")
    policy = load_policy(algo, path)
    if policy.obs_dim <= 0 or policy.action_dim <= 0:
        raise AssertionError(f"Invalid checkpoint dimensions for {algo}: {policy.obs_dim} / {policy.action_dim}")
    for key in ("policy_loss", "value_loss", "entropy", "avg_loss"):
        if key in policy.metadata and not math.isfinite(float(policy.metadata[key])):
            raise AssertionError(f"Non-finite metadata field {key} in {algo} checkpoint.")


def main() -> int:
    try:
        from TetrisVersionTwo.rl.dqn.train import main as dqn_train_main
        from TetrisVersionTwo.rl.ppo.train import main as ppo_train_main
        from TetrisVersionTwo.scripts.eval_rl import main as eval_main

        with tempfile.TemporaryDirectory(prefix="tetris_v2_rl_smoke_") as tmp:
            root = Path(tmp)

            ppo_dir = root / "ppo"
            rc = ppo_train_main(
                [
                    "--total-timesteps",
                    "512",
                    "--num-envs",
                    "2",
                    "--n-steps",
                    "64",
                    "--minibatch-size",
                    "128",
                    "--update-epochs",
                    "2",
                    "--seed",
                    "13",
                    "--log-interval",
                    "256",
                    "--eval-frequency",
                    "100000000",
                    "--checkpoint-frequency",
                    "100000000",
                    "--log-dir",
                    str(ppo_dir),
                ]
            )
            if rc != 0:
                raise AssertionError(f"PPO smoke training returned non-zero exit: {rc}")
            ppo_ckpt = ppo_dir / "ppo_final.pt"
            _assert_checkpoint(ppo_ckpt, "ppo")
            rc = eval_main(
                [
                    str(ppo_ckpt),
                    "--algo",
                    "ppo",
                    "--episodes",
                    "2",
                    "--seed",
                    "401",
                    "--max-steps",
                    "300",
                ]
            )
            if rc != 0:
                raise AssertionError(f"PPO smoke evaluation returned non-zero exit: {rc}")

            dqn_dir = root / "dqn"
            rc = dqn_train_main(
                [
                    "--total-timesteps",
                    "1500",
                    "--buffer-size",
                    "10000",
                    "--warmup-steps",
                    "100",
                    "--batch-size",
                    "64",
                    "--train-frequency",
                    "1",
                    "--target-sync-interval",
                    "200",
                    "--seed",
                    "17",
                    "--log-interval",
                    "500",
                    "--eval-frequency",
                    "100000000",
                    "--checkpoint-frequency",
                    "100000000",
                    "--log-dir",
                    str(dqn_dir),
                ]
            )
            if rc != 0:
                raise AssertionError(f"DQN smoke training returned non-zero exit: {rc}")
            dqn_ckpt = dqn_dir / "dqn_final.pt"
            _assert_checkpoint(dqn_ckpt, "dqn")
            rc = eval_main(
                [
                    str(dqn_ckpt),
                    "--algo",
                    "dqn",
                    "--episodes",
                    "2",
                    "--seed",
                    "501",
                    "--max-steps",
                    "300",
                ]
            )
            if rc != 0:
                raise AssertionError(f"DQN smoke evaluation returned non-zero exit: {rc}")
    except ModuleNotFoundError as exc:
        print(f"python_rl_smoke: SKIP ({exc})")
        return 0

    print("python_rl_smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
