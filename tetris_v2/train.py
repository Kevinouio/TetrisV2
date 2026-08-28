"""Hydra launcher for every TetrisV2 optimizer."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stderr
from importlib import import_module
from io import StringIO
from pathlib import Path
from typing import Any

import hydra
from hydra import compose, initialize_config_module
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


Runner = Callable[[argparse.Namespace], int]

RUNNER_TARGETS = {
    "dqn": "tetris_v2.rl.dqn.train:run",
    "dqn_pretrain": "scripts.pretrain_rl_dqn_expert:run",
    "dqn_hybrid": "scripts.train_rl_dqn_hybrid:run",
    "ppo": "tetris_v2.rl.ppo.train:run",
    "ppo_pretrain": "tetris_v2.rl.ppo.pretrain:run",
    "flow_dqn": "tetris_v2.rl.flow_dqn.train:run",
}

PARSER_TARGETS = {
    name: f"{target.split(':', maxsplit=1)[0]}:parse_args"
    for name, target in RUNNER_TARGETS.items()
}

PATH_ARGS = {
    "dataset_dir",
    "expert_dataset_dir",
    "init_checkpoint",
    "lib",
    "log_dir",
    "offline_dataset_dir",
    "resume_checkpoint",
}

PATH_LIST_ARGS = {
    "extra_dataset_dir",
    "extra_expert_dataset_dir",
}

BOOLEAN_OPTIONAL_ARGS = {"lr_anneal", "normalized_q"}
MULTI_VALUE_ARGS = {"hidden_sizes"}


def _load_runner(name: str) -> Runner:
    target = RUNNER_TARGETS[name]
    module_name, attribute = target.split(":", maxsplit=1)
    return getattr(import_module(module_name), attribute)


def _load_parser(name: str) -> Callable[[list[str]], argparse.Namespace]:
    target = PARSER_TARGETS[name]
    module_name, attribute = target.split(":", maxsplit=1)
    return getattr(import_module(module_name), attribute)


def _absolute_path(value: object) -> Path:
    return Path(to_absolute_path(str(value)))


def _native_argv(values: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in values.items():
        if value is None:
            continue
        option = f"--{key.replace('_', '-')}"
        if key in BOOLEAN_OPTIONAL_ARGS:
            if not isinstance(value, bool):
                raise ValueError(f"trainer.args.{key} must be true or false")
            argv.append(option if value else f"--no-{key.replace('_', '-')}")
        elif key in PATH_LIST_ARGS:
            if not isinstance(value, list):
                raise ValueError(f"trainer.args.{key} must be a list")
            for item in value:
                argv.extend((option, str(item)))
        elif key in MULTI_VALUE_ARGS:
            if not isinstance(value, list) or not value:
                raise ValueError(f"trainer.args.{key} must be a non-empty list")
            argv.append(option)
            argv.extend(str(item) for item in value)
        else:
            if isinstance(value, (dict, list)):
                raise ValueError(f"trainer.args.{key} must be a scalar")
            argv.extend((option, str(value)))
    return argv


def _parse_native(name: str, values: dict[str, Any]) -> argparse.Namespace:
    stderr = StringIO()
    try:
        with redirect_stderr(stderr):
            return _load_parser(name)(_native_argv(values))
    except SystemExit as exc:
        detail = stderr.getvalue().strip().splitlines()
        message = detail[-1] if detail else "invalid trainer arguments"
        raise ValueError(f"Invalid Hydra configuration for {name}: {message}") from exc


def _display_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_display_value(item) for item in value]
    return value


def build_namespace(config: DictConfig) -> argparse.Namespace:
    """Resolve one Hydra trainer config into the native trainer namespace."""

    name = str(config.trainer.name)
    if name not in RUNNER_TARGETS:
        choices = ", ".join(sorted(RUNNER_TARGETS))
        raise ValueError(f"Unknown trainer {name!r}; choose one of: {choices}")

    expected_trainer = str(config.preset.trainer)
    if name != expected_trainer:
        raise ValueError(
            f"Preset {config.preset.name!r} requires trainer={expected_trainer}, "
            f"not trainer={name}"
        )

    values = OmegaConf.to_container(config.trainer.args, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("trainer.args must be a mapping")

    device = config.runtime.get("device")
    if device is not None:
        values["device"] = str(device)

    for key in PATH_ARGS:
        if values.get(key) is not None:
            values[key] = _absolute_path(values[key])
    for key in PATH_LIST_ARGS:
        if key in values:
            if not isinstance(values[key], list):
                raise ValueError(f"trainer.args.{key} must be a list")
            values[key] = [_absolute_path(item) for item in values[key]]

    if name in {"ppo", "flow_dqn"}:
        if (
            values.get("init_checkpoint") is not None
            and values.get("resume_checkpoint") is not None
        ):
            raise ValueError("Set only one of trainer.args.init_checkpoint and resume_checkpoint")

    return _parse_native(name, values)


def run_config(
    config: DictConfig,
    *,
    runners: Mapping[str, Runner] | None = None,
) -> int:
    """Run a composed config; ``runners`` is injectable for fast routing tests."""

    name = str(config.trainer.name)
    args = build_namespace(config)
    if bool(config.get("dry_run", False)):
        resolved_args = {key: _display_value(value) for key, value in vars(args).items()}
        print(f"trainer: {name}")
        print(OmegaConf.to_yaml(OmegaConf.create(resolved_args)).rstrip())
        return 0

    runner = runners[name] if runners is not None else _load_runner(name)
    result = runner(args)
    return int(result) if result is not None else 0


def compose_training_config(overrides: Sequence[str] = ()) -> DictConfig:
    """Compose packaged training configs without starting Hydra's CLI runtime."""

    with initialize_config_module(
        config_module="tetris_v2.conf",
        job_name="tetris-train",
        version_base="1.3",
    ):
        return compose(config_name="train", overrides=list(overrides))


@hydra.main(version_base="1.3", config_path="conf", config_name="train")
def _hydra_main(config: DictConfig) -> int:
    return run_config(config)


def main() -> int:
    result = _hydra_main()
    return int(result) if result is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_namespace", "compose_training_config", "main", "run_config"]
