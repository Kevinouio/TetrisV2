from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import pretrain_rl_dqn_expert, train_rl_dqn_hybrid
from tetris_v2 import train as hydra_train
from tetris_v2.rl.dqn import train as dqn_train
from tetris_v2.rl.flow_dqn import train as flow_train
from tetris_v2.rl.battle import train as battle_train
from tetris_v2.rl.ppo import pretrain as ppo_pretrain
from tetris_v2.rl.ppo import train as ppo_train


EXPERIMENTS = {
    "dqn_from_scratch": "dqn",
    "dqn_pretrain": "dqn_pretrain",
    "dqn_dagger_pretrain": "dqn_pretrain",
    "dqn_hybrid": "dqn_hybrid",
    "ppo_from_scratch": "ppo",
    "ppo_pretrain": "ppo_pretrain",
    "ppo_dagger_pretrain": "ppo_pretrain",
    "ppo_finetune": "ppo",
    "flow_pilot": "flow_dqn",
    "flow_offline": "flow_dqn",
    "flow_online": "flow_dqn",
    "flow_from_scratch": "flow_dqn",
    "battle_smoke": "battle_dqn",
    "battle_selfplay": "battle_dqn",
}

PARSERS = {
    "dqn": (dqn_train.parse_args, []),
    "dqn_pretrain": (pretrain_rl_dqn_expert.parse_args, ["--dataset-dir", "unused"]),
    "dqn_hybrid": (
        train_rl_dqn_hybrid.parse_args,
        ["--offline-dataset-dir", "unused"],
    ),
    "ppo": (ppo_train.parse_args, []),
    "ppo_pretrain": (ppo_pretrain.parse_args, ["--dataset-dir", "unused"]),
    "flow_dqn": (flow_train.parse_args, []),
    "battle_dqn": (battle_train.parse_args, []),
}


@pytest.mark.parametrize(("experiment", "trainer"), EXPERIMENTS.items())
def test_every_hydra_preset_composes_with_the_native_parser_contract(
    experiment: str,
    trainer: str,
) -> None:
    config = hydra_train.compose_training_config(
        [f"experiment={experiment}", "runtime=cpu"]
    )
    args = hydra_train.build_namespace(config)
    parser, parser_argv = PARSERS[trainer]

    assert config.trainer.name == trainer
    assert args.device == "cpu"
    assert set(vars(args)) == set(vars(parser(parser_argv)))


def test_dagger_paths_preserve_order_and_duplicate_weighting() -> None:
    config = hydra_train.compose_training_config(["experiment=ppo_dagger_pretrain"])
    args = hydra_train.build_namespace(config)

    assert len(args.extra_dataset_dir) == 2
    assert all(isinstance(path, Path) for path in args.extra_dataset_dir)
    assert args.extra_dataset_dir[0] == args.extra_dataset_dir[1]
    assert args.extra_dataset_dir[0].is_absolute()


def test_flow_presets_capture_validated_schedule_and_support_overrides() -> None:
    online = hydra_train.build_namespace(
        hydra_train.compose_training_config(["experiment=flow_online", "runtime=cuda"])
    )
    assert online.offline_updates == 0
    assert online.online_timesteps == 500_000
    assert online.online_offline_fraction == 0.9
    assert online.exploration_temperature == 0.25
    assert online.warmup_steps == 512
    assert online.device == "cuda"

    pilot = hydra_train.build_namespace(
        hydra_train.compose_training_config(
            ["experiment=flow_pilot", "trainer.args.distill_q_coef=0.3"]
        )
    )
    assert pilot.distill_q_coef == 0.3
    assert pilot.log_dir.name == "flow_pilot_0.3"

    unnormalized = hydra_train.build_namespace(
        hydra_train.compose_training_config(
            ["experiment=flow_from_scratch", "trainer.args.normalized_q=false"]
        )
    )
    assert unnormalized.normalized_q is False


def test_battle_presets_route_attack_table_boolean_flags_and_paths() -> None:
    smoke = hydra_train.build_namespace(
        hydra_train.compose_training_config(["experiment=battle_smoke", "runtime=cpu"])
    )
    assert smoke.total_timesteps == 20
    assert smoke.attack_table == [0, 0, 1, 2, 4]
    assert smoke.disable_curriculum is True
    assert smoke.independent_piece_seeds is False
    assert smoke.cold_clear_think_ms == 0
    assert smoke.curriculum_config is None
    assert smoke.device == "cpu"
    assert smoke.log_dir.is_absolute()

    custom = hydra_train.build_namespace(
        hydra_train.compose_training_config(
            [
                "experiment=battle_selfplay",
                "trainer.args.attack_table=[0,0,2,3,5]",
                "trainer.args.independent_piece_seeds=true",
                "trainer.args.disable_curriculum=true",
            ]
        )
    )
    assert custom.attack_table == [0, 0, 2, 3, 5]
    assert custom.independent_piece_seeds is True
    assert custom.disable_curriculum is True


def test_native_parser_coerces_string_overrides() -> None:
    config = hydra_train.compose_training_config(
        ["experiment=ppo_from_scratch", "trainer.args.minibatch_size='128'"]
    )
    args = hydra_train.build_namespace(config)
    assert args.minibatch_size == 128
    assert isinstance(args.minibatch_size, int)


def test_run_config_routes_resolved_namespace_without_argparse_translation() -> None:
    config = hydra_train.compose_training_config(["experiment=flow_offline"])
    captured: list[argparse.Namespace] = []

    status = hydra_train.run_config(
        config,
        runners={"flow_dqn": lambda args: captured.append(args) or 7},
    )

    assert status == 7
    assert len(captured) == 1
    assert captured[0].offline_updates == 200_000
    assert captured[0].offline_dataset_dir.is_absolute()


def test_run_config_routes_battle_namespace() -> None:
    config = hydra_train.compose_training_config(["experiment=battle_smoke"])
    captured: list[argparse.Namespace] = []

    status = hydra_train.run_config(
        config,
        runners={"battle_dqn": lambda args: captured.append(args) or 3},
    )

    assert status == 3
    assert len(captured) == 1
    assert captured[0].attack_table == [0, 0, 1, 2, 4]
    assert captured[0].disable_curriculum is True


def test_hydra_validation_rejects_missing_inputs_and_conflicting_checkpoints() -> None:
    missing = hydra_train.compose_training_config(
        ["experiment=ppo_pretrain", "trainer.args.dataset_dir=null"]
    )
    with pytest.raises(ValueError, match="dataset-dir"):
        hydra_train.build_namespace(missing)

    conflicting = hydra_train.compose_training_config(
        [
            "experiment=flow_from_scratch",
            "trainer.args.init_checkpoint=init.pt",
            "trainer.args.resume_checkpoint=resume.pt",
        ]
    )
    with pytest.raises(ValueError, match="only one"):
        hydra_train.build_namespace(conflicting)

    battle_conflict = hydra_train.compose_training_config(
        [
            "experiment=battle_selfplay",
            "trainer.args.init_checkpoint=init.pt",
            "trainer.args.resume_checkpoint=resume.pt",
        ]
    )
    with pytest.raises(ValueError, match="only one"):
        hydra_train.build_namespace(battle_conflict)


@pytest.mark.parametrize(
    "overrides",
    [
        ["experiment=flow_from_scratch", "trainer.args.normalized_q=banana"],
        ["experiment=ppo_from_scratch", "trainer.args.network_type=garbage"],
        ["experiment=dqn_from_scratch", "trainer.args.batch_size=oops"],
        ["experiment=battle_selfplay", "trainer.args.attack_table=[0,1]"],
        ["experiment=battle_selfplay", "trainer.args.disable_curriculum=banana"],
    ],
)
def test_hydra_overrides_keep_native_type_and_choice_validation(
    overrides: list[str],
) -> None:
    config = hydra_train.compose_training_config(overrides)
    with pytest.raises(ValueError, match="Hydra configuration|true or false"):
        hydra_train.build_namespace(config)


@pytest.mark.parametrize("value", ["dagger", "null"])
def test_path_list_overrides_reject_scalars(value: str) -> None:
    config = hydra_train.compose_training_config(
        ["experiment=ppo_pretrain", f"trainer.args.extra_dataset_dir={value}"]
    )
    with pytest.raises(ValueError, match="extra_dataset_dir.*list"):
        hydra_train.build_namespace(config)


def test_preset_cannot_be_combined_with_a_different_trainer() -> None:
    config = hydra_train.compose_training_config(
        ["experiment=flow_online", "trainer=ppo"]
    )
    with pytest.raises(ValueError, match="requires trainer=flow_dqn"):
        hydra_train.build_namespace(config)


def test_relative_paths_resolve_from_invocation_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config = hydra_train.compose_training_config(["experiment=flow_offline"])
    args = hydra_train.build_namespace(config)

    assert args.offline_dataset_dir == tmp_path / "runs/v4_expert_transitions"
    assert args.log_dir == tmp_path / "runs/flow_dqn_offline"

    battle = hydra_train.build_namespace(
        hydra_train.compose_training_config(
            [
                "experiment=battle_selfplay",
                "trainer.args.curriculum_config=configs/custom_battle.yaml",
            ]
        )
    )
    assert battle.curriculum_config == tmp_path / "configs/custom_battle.yaml"


def test_flow_hydra_preset_trains_and_saves_a_checkpoint(tmp_path: Path) -> None:
    output = tmp_path / "flow_hydra_smoke"
    config = hydra_train.compose_training_config(
        [
            "experiment=flow_from_scratch",
            "runtime=cpu",
            "trainer.args.online_timesteps=2",
            "trainer.args.buffer_size=4",
            "trainer.args.warmup_steps=1",
            "trainer.args.batch_size=1",
            "trainer.args.channels=2",
            "trainer.args.flow_steps=1",
            "trainer.args.max_steps=2",
            "trainer.args.log_interval=0",
            "trainer.args.eval_frequency=0",
            "trainer.args.checkpoint_frequency=0",
            f"trainer.args.log_dir={output}",
        ]
    )

    assert hydra_train.run_config(config) == 0
    assert (output / "flow_dqn_final.pt").is_file()


def test_packaged_hydra_cli_dry_run(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tetris_v2.train",
            "experiment=flow_offline",
            "runtime=cpu",
            "dry_run=true",
            f"hydra.run.dir={tmp_path / 'hydra'}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "trainer: flow_dqn" in result.stdout
    assert "offline_updates: 200000" in result.stdout
    assert "device: cpu" in result.stdout


def test_battle_console_scripts_are_published() -> None:
    project = Path("pyproject.toml").read_text(encoding="utf-8")
    expected = (
        'tetris-battle-smoke = "scripts.battle_smoke:main"',
        'tetris-train-battle = "tetris_v2.rl.battle.train:main"',
        'tetris-eval-battle = "scripts.eval_battle:main"',
        'tetris-eval-battle-pool = "scripts.eval_battle_pool:main"',
        'tetris-battle-matrix = "scripts.battle_matrix:main"',
    )
    assert all(entry in project for entry in expected)
    for target in (
        "scripts.battle_smoke",
        "tetris_v2.rl.battle.train",
        "scripts.eval_battle",
        "scripts.eval_battle_pool",
        "scripts.battle_matrix",
    ):
        assert callable(getattr(import_module(target), "main"))
