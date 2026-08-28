from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import generate_expert_dataset
from tetris_v2.rl.actions import PLACEMENT_ACTION_DIM
from tetris_v2.rl.env import CCTetrisEnv
from tetris_v2.rl.expert import ExpertRanker
from tetris_v2.rl.expert_dataset import (
    discover_shards,
    load_dataset,
    load_dataset_directory,
    load_transition_dataset_directory,
    write_manifest,
    write_shard,
)


def expert_record(obs_dim: int, *, step: int = 0, transition: bool = False) -> dict:
    mask = np.zeros(PLACEMENT_ACTION_DIM, dtype=np.uint8)
    mask[:3] = 1
    record = {
        "obs": np.arange(obs_dim, dtype=np.float32),
        "action_mask": mask,
        "teacher_best_action": 1,
        "seed": 123,
        "episode": 0,
        "step": step,
        "legal_action_count": 3,
        "placement_count_raw": 3,
        "placement_overflow": 0,
        "nodes": 12,
        "think_ms": 1.0,
        "budget_miss": 0,
        "unexpanded_count": 2,
    }
    if transition:
        record.update(
            {
                "executed_action": 2,
                "reward": 0.25,
                "raw_reward": 100.0,
                "next_obs": np.arange(obs_dim, dtype=np.float32) + 1.0,
                "next_action_mask": mask.copy(),
                "terminated": False,
                "truncated": True,
            }
        )
    return record


def test_expert_ranking_matches_environment_mask() -> None:
    env = CCTetrisEnv(seed=123, max_steps=20)
    try:
        _, info = env.reset(seed=123)
        ranker = ExpertRanker(env.runtime, think_ms=1)

        for _ in range(4):
            rank = ranker.rank_current_state()
            legal = rank.action_mask > 0

            assert rank.action_mask.shape == (PLACEMENT_ACTION_DIM,)
            assert np.count_nonzero(legal) == info["legal_action_count"]
            assert rank.teacher_best_action in np.flatnonzero(legal)
            assert rank.nodes >= 0
            assert rank.think_ms >= 0.0
            assert rank.nps >= 0.0
            assert rank.unexpanded_count >= 0
            assert rank.unexpanded_count > 0

            _, _, terminated, truncated, info = env.step(rank.teacher_best_action)
            if terminated or truncated:
                break
    finally:
        env.close()


def test_manifest_and_shards_round_trip(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    shard = dataset_dir / "shards" / "expert_shard_00000.npz"
    manifest = dataset_dir / "manifest.json"
    records = [expert_record(5, step=0), expert_record(5, step=1)]

    write_shard(shard, records)
    write_manifest(manifest, shards=[shard], total_samples=len(records))

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["total_samples"] == 2
    assert payload["shards"] == ["shards/expert_shard_00000.npz"]
    assert discover_shards(dataset_dir) == [shard]

    dataset = load_dataset(discover_shards(dataset_dir))
    assert len(dataset) == 2
    np.testing.assert_array_equal(dataset.teacher_best_action, [1, 1])
    np.testing.assert_allclose(dataset.obs[1], records[1]["obs"])
    np.testing.assert_array_equal(dataset.unexpanded_count, [2, 2])

    with np.load(shard, allow_pickle=False) as stored:
        legacy_fields = {key: stored[key] for key in stored.files if key != "unexpanded_count"}
    np.savez_compressed(shard, **legacy_fields)
    legacy_dataset = load_dataset_directory(dataset_dir)
    np.testing.assert_array_equal(legacy_dataset.unexpanded_count, [0, 0])


def test_transition_schema_round_trip_and_bc_compatibility(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "transitions"
    shard = dataset_dir / "expert_shard_00000.npz"
    record = expert_record(5, transition=True)
    write_shard(shard, [record])
    write_manifest(
        dataset_dir / "manifest.json",
        shards=[shard],
        total_samples=1,
        version=3,
    )

    labels = load_dataset_directory(dataset_dir)
    transitions = load_transition_dataset_directory(dataset_dir)

    assert len(labels) == len(transitions) == 1
    assert transitions.executed_action.tolist() == [2]
    assert transitions.reward.tolist() == pytest.approx([0.25])
    assert transitions.raw_reward.tolist() == pytest.approx([100.0])
    assert transitions.terminated.tolist() == [0]
    assert transitions.truncated.tolist() == [1]
    np.testing.assert_allclose(transitions.next_obs[0], record["next_obs"])
    np.testing.assert_array_equal(transitions.next_action_mask[0], record["next_action_mask"])

    batch = transitions.sample(2, np.random.default_rng(3))
    assert set(("executed_action", "reward", "next_obs", "terminated", "truncated")) <= set(batch)
    assert batch["executed_action"].tolist() == [2, 2]


def test_transition_loader_rejects_schema_v2_with_regeneration_guidance(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "labels"
    shard = dataset_dir / "expert_shard_00000.npz"
    write_shard(shard, [expert_record(5)])
    write_manifest(dataset_dir / "manifest.json", shards=[shard], total_samples=1)

    with pytest.raises(ValueError, match="schema v3.*Regenerate"):
        load_transition_dataset_directory(dataset_dir)


def test_manifest_rejects_missing_shards_and_wrong_sample_count(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    shard = dataset_dir / "expert_shard_00000.npz"
    manifest = dataset_dir / "manifest.json"
    write_shard(shard, [expert_record(5)])
    write_manifest(manifest, shards=[shard], total_samples=1)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["shards"].append("missing.npz")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing.npz"):
        discover_shards(dataset_dir)

    payload["shards"].pop()
    payload["total_samples"] = 2
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="expects 2 samples"):
        load_dataset_directory(dataset_dir)


def test_generator_defaults_to_non_mutating_top1_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_envs = []

    class Top1Runtime:
        lib = SimpleNamespace(_name="top1-test")

        def __init__(self):
            self.choose_calls = 0

        def bot_choose(self, think_ms: int) -> dict:
            self.choose_calls += 1
            return {
                "success": True,
                "use_hold": False,
                "placement_index": 1,
                "score": 42.0,
                "nodes": 17,
                "think_ms": float(think_ms),
                "nps": 100.0,
                "budget_miss": 0,
            }

        def decision_for_choice(self, *, use_hold: bool, placement_index: int) -> int:
            assert not use_hold
            return placement_index

        def bot_rank_actions(self, **_: object) -> dict:
            raise AssertionError("Top-1 mode must not request a full ranking.")

    class Top1Env:
        def __init__(self, **_: object):
            self.runtime = Top1Runtime()
            self.actions = []
            created_envs.append(self)

        def reset(self, **_: object):
            mask = np.zeros(PLACEMENT_ACTION_DIM, dtype=np.float32)
            mask[:3] = 1.0
            return np.arange(4, dtype=np.float32), {
                "action_mask": mask,
                "placement_count_raw": 3,
                "placement_overflow": False,
            }

        def step(self, action: int):
            self.actions.append(action)
            obs, info = self.reset()
            info["raw_reward"] = 100.0
            return obs + 1.0, 1.25, False, True, info

        def close(self) -> None:
            pass

    monkeypatch.setattr(generate_expert_dataset, "CCTetrisEnv", Top1Env)
    output_dir = tmp_path / "top1"

    assert (
        generate_expert_dataset.main(
            [
                "--output-dir",
                str(output_dir),
                "--episodes",
                "1",
                "--max-steps",
                "1",
                "--random-action-prob",
                "0",
            ]
        )
        == 0
    )

    env = created_envs[0]
    dataset = load_dataset_directory(output_dir)
    transitions = load_transition_dataset_directory(output_dir)
    assert env.runtime.choose_calls == 1
    assert env.actions == [1]
    assert dataset.teacher_best_action.tolist() == [1]
    np.testing.assert_array_equal(dataset.action_mask[0, :4], [1.0, 1.0, 1.0, 0.0])
    assert dataset.unexpanded_count.tolist() == [2]
    assert transitions.executed_action.tolist() == [1]
    assert transitions.reward.tolist() == pytest.approx([1.25])
    assert transitions.raw_reward.tolist() == pytest.approx([100.0])
    assert transitions.terminated.tolist() == [0]
    assert transitions.truncated.tolist() == [1]
    np.testing.assert_allclose(transitions.next_obs[0], np.arange(4, dtype=np.float32) + 1.0)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["label_mode"] == "top1"
    assert summary["schema_version"] == 3
    assert summary["max_steps"] == 1
    assert summary["random_action_prob"] == 0.0
    assert summary["teacher_action_prob"] is None
    assert summary["shard_size"] == 4096
    assert summary["library"] == "top1-test"
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == 3


def test_generator_routes_generic_dagger_behavior_algorithm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = []
    executed = []

    class Runtime:
        lib = SimpleNamespace(_name="dagger-test")

        def bot_choose(self, think_ms: int) -> dict:
            return {
                "success": True,
                "use_hold": False,
                "placement_index": 1,
                "nodes": 1,
                "think_ms": float(think_ms),
                "budget_miss": 0,
            }

        def decision_for_choice(self, *, use_hold: bool, placement_index: int) -> int:
            return placement_index

    class Env:
        def __init__(self, **_: object):
            self.runtime = Runtime()
            self.observation_space = SimpleNamespace(shape=(4,))
            self.action_space = SimpleNamespace(n=PLACEMENT_ACTION_DIM)

        def reset(self, **_: object):
            mask = np.zeros(PLACEMENT_ACTION_DIM, dtype=np.float32)
            mask[:3] = 1.0
            return np.zeros(4, dtype=np.float32), {
                "action_mask": mask,
                "placement_count_raw": 3,
                "placement_overflow": False,
            }

        def step(self, action: int):
            executed.append(action)
            obs, info = self.reset()
            info["raw_reward"] = 0.0
            return obs, 0.0, False, True, info

        def close(self) -> None:
            pass

    class Behavior:
        obs_dim = 4
        action_dim = PLACEMENT_ACTION_DIM

        def act(self, *_: object, **__: object) -> int:
            return 2

    def fake_load_policy(algo: str, checkpoint: Path, *, device: object = None) -> Behavior:
        loaded.append((algo, checkpoint, device))
        return Behavior()

    monkeypatch.setattr(generate_expert_dataset, "CCTetrisEnv", Env)
    monkeypatch.setattr(generate_expert_dataset, "load_policy", fake_load_policy)
    checkpoint = tmp_path / "behavior.pt"
    output_dir = tmp_path / "dagger"

    assert generate_expert_dataset.main(
        [
            "--output-dir",
            str(output_dir),
            "--episodes",
            "1",
            "--behavior-checkpoint",
            str(checkpoint),
            "--behavior-algo",
            "flow_dqn",
            "--teacher-action-prob",
            "0",
            "--random-action-prob",
            "0",
        ]
    ) == 0

    assert loaded == [("flow_dqn", checkpoint, None)]
    assert executed == [2]
    transitions = load_transition_dataset_directory(output_dir)
    assert transitions.teacher_best_action.tolist() == [1]
    assert transitions.executed_action.tolist() == [2]
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["behavior_algo"] == "flow_dqn"
    assert summary["teacher_action_prob"] == 0.0
    assert summary["random_action_prob"] == 0.0


def test_generator_exploration_uses_off_teacher_action() -> None:
    legal = np.asarray([0, 1], dtype=np.int64)
    action = generate_expert_dataset._select_action(
        np.random.default_rng(9),
        legal,
        teacher_action=1,
        random_action_prob=1.0,
    )
    assert action == 0


def test_failed_expert_choice_is_not_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_envs = []

    class FailedRuntime:
        lib = SimpleNamespace(_name="failed-rank-test")

        def bot_choose(self, **_: object) -> dict:
            return {
                "success": False,
            }

        def bot_sync(self) -> bool:
            raise AssertionError("A failed rank must not enter recovery or storage.")

    class FailedRankEnv:
        def __init__(self, **_: object):
            self.runtime = FailedRuntime()
            self.include_hidden_rows = False
            self.closed = False
            created_envs.append(self)

        def reset(self, **_: object):
            return np.zeros(3, dtype=np.float32), {}

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(generate_expert_dataset, "CCTetrisEnv", FailedRankEnv)
    output_dir = tmp_path / "failed"

    with pytest.raises(RuntimeError):
        generate_expert_dataset.main(
            [
                "--output-dir",
                str(output_dir),
                "--episodes",
                "1",
                "--max-steps",
                "1",
                "--think-ms",
                "1",
            ]
        )

    assert created_envs and created_envs[0].closed
    assert not list(output_dir.glob("*.npz"))
    assert not (output_dir / "manifest.json").exists()
