from __future__ import annotations

import argparse
from collections import deque
import concurrent.futures
import csv
import json
import multiprocessing as mp
import os
import queue as queue_mod
import random
import time
from dataclasses import asdict, replace
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..bc.utils import find_library
from .agent import DQNRefAgent
from .config import DQNRefConfig, GAConfig, RuntimeConfig, TrainingConfig
from .env_bridge import DQNRefEnvBridge
from .genetic import GeneticPopulation, PopulationEntry


def parse_args() -> argparse.Namespace:
    defaults = DQNRefConfig()
    parser = argparse.ArgumentParser(description="Faithful Tetris-A.I Version2 DQN/GA baseline on TetrisVersionTwo.")
    parser.add_argument("--lib", type=Path, default=None, help="Path to libtetris_v2_c_api shared library.")
    parser.add_argument("--run_dir", type=Path, default=Path("runs/dqn_ref"))
    parser.add_argument("--seed", type=int, default=defaults.runtime.seed)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--torch_compile", action="store_true", default=defaults.runtime.torch_compile)
    parser.add_argument("--channels_last", action="store_true", default=defaults.runtime.channels_last)
    parser.add_argument("--log_every_episodes", type=int, default=defaults.runtime.log_every_episodes)
    parser.add_argument(
        "--agent_workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for per-agent evaluation in each generation.",
    )

    parser.add_argument("--population_size", type=int, default=defaults.ga.population_size)
    parser.add_argument("--generations", type=int, default=defaults.ga.generations)
    parser.add_argument("--games_per_agent", type=int, default=defaults.ga.total_games_per_agent)
    parser.add_argument("--size_pick", type=int, default=defaults.ga.size_pick)
    parser.add_argument("--generation_rate", type=int, default=defaults.ga.generation_rate)
    parser.add_argument("--elite_count", type=int, default=defaults.ga.elite_count)
    parser.add_argument("--max_steps_per_episode", type=int, default=defaults.training.max_steps_per_episode)
    parser.add_argument("--viewer", action="store_true", help="Enable live pygame viewer dashboard.")
    parser.add_argument(
        "--viewer_fullscreen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start viewer in fullscreen mode (default: true).",
    )
    parser.add_argument("--viewer_fps", type=int, default=20, help="Viewer redraw FPS.")
    parser.add_argument(
        "--viewer_publish_every_steps",
        type=int,
        default=10,
        help="Publish telemetry snapshots every N steps per episode.",
    )
    parser.add_argument(
        "--viewer_max_queue",
        type=int,
        default=4096,
        help="Maximum queued viewer events before dropping telemetry.",
    )
    parser.add_argument("--viewer_grid_padding", type=int, default=8, help="Padding between mini-cards.")
    parser.add_argument("--viewer_min_tile_px", type=int, default=6, help="Minimum mini-board tile size.")
    parser.add_argument("--viewer_agent", type=int, default=1, help="Initial selected agent id.")
    parser.add_argument(
        "--viewer_reopen_file",
        type=Path,
        default=None,
        help="Optional file trigger path. Creating this file reopens the live viewer.",
    )

    parser.add_argument("--smoke", action="store_true", help="Run a tiny smoke config (pop=4, gen=1, games=2).")
    return parser.parse_args()


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def episode_seed(base_seed: int, generation: int, agent_idx: int, episode_idx: int) -> int:
    return int(base_seed + generation * 1_000_003 + agent_idx * 10_007 + episode_idx * 97)


def compute_diversity(population: List[PopulationEntry]) -> float:
    if len(population) < 2:
        return 0.0
    genomes = [entry.genome for entry in population]
    diffs: List[float] = []
    keys = list(genomes[0].keys())
    for g1, g2 in combinations(genomes, 2):
        a = np.asarray([float(g1[k]) for k in keys], dtype=np.float32)
        b = np.asarray([float(g2[k]) for k in keys], dtype=np.float32)
        diffs.append(float(np.sum(np.abs(a - b))))
    return float(np.mean(diffs)) if diffs else 0.0


def _to_cpu_serializable(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_serializable(v) for v in obj)
    return obj


def _agent_checkpoint_payload(agent: DQNRefAgent) -> Dict[str, Any]:
    model_state = {k: v.detach().cpu() for k, v in agent.model1.state_dict().items()}
    optimizer_state = _to_cpu_serializable(agent.trainer.optimizer_state())
    return {
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
    }


def _queue_put_best_effort(event_queue: Any, event: Dict[str, object]) -> bool:
    if event_queue is None:
        return False
    try:
        event_queue.put_nowait(event)
        return True
    except Exception:
        return False


def _board_for_event(board: object) -> List[int]:
    arr = np.asarray(board, dtype=np.uint8)
    if arr.size != 200:
        return [0] * 200
    return arr.reshape(-1).astype(np.uint8, copy=False).tolist()


def _piece_ids_for_event(piece_ids: object) -> List[int]:
    arr = np.asarray(piece_ids, dtype=np.uint8)
    if arr.size != 200:
        return [255] * 200
    return arr.reshape(-1).astype(np.uint8, copy=False).tolist()


def run_episode(
    *,
    env: DQNRefEnvBridge,
    agent: DQNRefAgent,
    seed: int,
    max_steps_per_episode: int,
    generation: int,
    agent_index: int,
    episode_index: int,
    games_per_agent: int,
    total_lines_before_episode: float,
    publish_every_steps: int,
    publish_event: Optional[Callable[[Dict[str, object]], None]] = None,
) -> Dict[str, float]:
    env.reset(seed)
    old_state = np.zeros((6,), dtype=np.float32)

    ep_return = 0.0
    steps = 0
    reward_breakdown: Dict[str, float] = {
        "game_over_term": 0.0,
        "survival_term": 0.0,
        "y_pos_term": 0.0,
        "total_height_term": 0.0,
        "lines_term": 0.0,
        "holes_term": 0.0,
        "bumpiness_term": 0.0,
        "pillar_term": 0.0,
        "high_placement_penalty_term": 0.0,
    }

    publish_step_interval = max(1, int(publish_every_steps))

    for step_idx in range(int(max_steps_per_episode)):
        candidates = env.enumerate_candidates()
        if not candidates:
            break
        chosen = agent.get_action(candidates)
        if chosen is None:
            break

        step = env.step(chosen.native_action)
        done = bool(step["game_over"])
        reward_terms = agent.calculate_reward(chosen.feature_vector, finished=done)
        reward_value = float(reward_terms.total)

        agent.remember(old_state, chosen.feature_vector, reward_value, done)
        old_state = chosen.feature_vector
        agent.check_steps()

        ep_return += reward_value
        steps += 1

        as_map = reward_terms.to_dict()
        for key in reward_breakdown.keys():
            reward_breakdown[key] += float(as_map[key])

        if publish_event is not None and ((step_idx + 1) % publish_step_interval == 0 or done):
            state_now = env.state()
            lines_total = int(state_now.get("lines", 0))
            fitness_prov = (float(total_lines_before_episode) + float(lines_total)) / float(
                max(1, games_per_agent)
            )
            publish_event(
                {
                    "type": "step_snapshot",
                    "generation": int(generation),
                    "agent_index": int(agent_index),
                    "episode_index": int(episode_index),
                    "games_per_agent": int(games_per_agent),
                    "step_in_episode": int(step_idx + 1),
                    "board": _board_for_event(state_now.get("board")),
                    "board_piece_ids": _piece_ids_for_event(state_now.get("board_piece_ids")),
                    "lines_total": int(lines_total),
                    "episode_return_running": float(ep_return),
                    "epsilon": float(agent.epsilon),
                    "loss_last": float(agent.last_loss),
                    "fitness_provisional": float(fitness_prov),
                    "status": "active",
                    "timestamp": float(time.time()),
                }
            )

        if done:
            break

    agent.check_training()
    agent.n_games += 1
    epsilon = agent.decay_epsilon(agent.n_games)
    lr = agent.calculate_lr(agent.n_games)

    final_state = env.state()
    lines = int(final_state["lines"])

    out = {
        "episode_return": float(ep_return),
        "lines_cleared": float(lines),
        "survival_length": float(steps),
        "epsilon": float(epsilon),
        "learning_rate": float(lr),
        "loss": float(agent.last_loss),
    }
    out.update(reward_breakdown)

    if publish_event is not None:
        fitness_prov = (float(total_lines_before_episode) + float(lines)) / float(max(1, games_per_agent))
        publish_event(
            {
                "type": "episode_done",
                "generation": int(generation),
                "agent_index": int(agent_index),
                "episode_index": int(episode_index),
                "games_per_agent": int(games_per_agent),
                "board": _board_for_event(final_state.get("board")),
                "board_piece_ids": _piece_ids_for_event(final_state.get("board_piece_ids")),
                "episode_return": float(ep_return),
                "lines_cleared": float(lines),
                "lines_total": int(lines),
                "survival_length": float(steps),
                "epsilon": float(epsilon),
                "loss": float(agent.last_loss),
                "fitness_provisional": float(fitness_prov),
                "status": "active",
                "timestamp": float(time.time()),
            }
        )

    return out


def _evaluate_agent_worker(
    *,
    lib_path: str,
    config: DQNRefConfig,
    genome: Dict[str, float],
    generation: int,
    agent_index: int,
    base_seed: int,
    device_str: str,
    checkpoint_path: str | None,
    viewer_queue: Any = None,
    viewer_publish_every_steps: int = 10,
) -> Dict[str, Any]:
    if str(device_str).lower().startswith("cpu"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    worker_seed = int(base_seed + generation * 1_000_003 + agent_index * 10_007 + 7_919)
    set_global_seeds(worker_seed)
    if str(device_str).lower().startswith("cpu"):
        try:
            torch.cuda.is_available = lambda: False  # type: ignore[assignment]
        except Exception:
            pass
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    if checkpoint is not None and not checkpoint.exists():
        checkpoint = None
    agent = DQNRefAgent(
        genome=dict(genome),
        config=config,
        device=torch.device(device_str),
        checkpoint_path=checkpoint,
    )

    episode_rows: List[Dict[str, float | int]] = []
    total_lines = 0.0
    total_return = 0.0
    total_survival = 0.0
    last_metrics: Dict[str, float] = {}

    with DQNRefEnvBridge(
        lib_path=Path(lib_path),
        seed=episode_seed(base_seed, generation, agent_index, 0),
    ) as env:
        _queue_put_best_effort(
            viewer_queue,
            {
                "type": "agent_started",
                "generation": int(generation),
                "agent_index": int(agent_index),
                "games_per_agent": int(config.ga.total_games_per_agent),
                "status": "active",
                "timestamp": float(time.time()),
            },
        )
        for episode_index in range(1, int(config.ga.total_games_per_agent) + 1):
            metrics = run_episode(
                env=env,
                agent=agent,
                seed=episode_seed(base_seed, generation, agent_index, episode_index),
                max_steps_per_episode=int(config.training.max_steps_per_episode),
                generation=int(generation),
                agent_index=int(agent_index),
                episode_index=int(episode_index),
                games_per_agent=int(config.ga.total_games_per_agent),
                total_lines_before_episode=float(total_lines),
                publish_every_steps=int(viewer_publish_every_steps),
                publish_event=lambda e: _queue_put_best_effort(viewer_queue, e),
            )
            total_lines += float(metrics["lines_cleared"])
            total_return += float(metrics["episode_return"])
            total_survival += float(metrics["survival_length"])
            row: Dict[str, float | int] = {
                "generation": int(generation),
                "agent_index": int(agent_index),
                "episode_index": int(episode_index),
            }
            row.update(metrics)
            episode_rows.append(row)
            last_metrics = metrics

    fitness = float(total_lines / float(config.ga.total_games_per_agent))
    _queue_put_best_effort(
        viewer_queue,
        {
            "type": "agent_done",
            "generation": int(generation),
            "agent_index": int(agent_index),
            "episodes_completed": int(config.ga.total_games_per_agent),
            "fitness": float(fitness),
            "avg_return": float(total_return / float(config.ga.total_games_per_agent)),
            "avg_survival": float(total_survival / float(config.ga.total_games_per_agent)),
            "status": "done",
            "timestamp": float(time.time()),
        },
    )
    return {
        "agent_index": int(agent_index),
        "fitness": fitness,
        "avg_return": float(total_return / float(config.ga.total_games_per_agent)),
        "avg_survival": float(total_survival / float(config.ga.total_games_per_agent)),
        "episode_rows": episode_rows,
        "last_metrics": last_metrics,
        "checkpoint_payload": _agent_checkpoint_payload(agent),
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_global_seeds(int(args.seed))
    device = resolve_device(args.device)

    base_cfg = DQNRefConfig()
    ga_cfg = replace(
        base_cfg.ga,
        population_size=int(args.population_size),
        generations=int(args.generations),
        total_games_per_agent=int(args.games_per_agent),
        size_pick=int(args.size_pick),
        generation_rate=int(args.generation_rate),
        elite_count=int(args.elite_count),
    )
    training_cfg = replace(
        base_cfg.training,
        max_steps_per_episode=int(args.max_steps_per_episode),
    )
    runtime_cfg = RuntimeConfig(
        seed=int(args.seed),
        device=str(device),
        torch_compile=bool(args.torch_compile),
        channels_last=bool(args.channels_last),
        log_every_episodes=max(1, int(args.log_every_episodes)),
    )
    cfg = DQNRefConfig(
        model=base_cfg.model,
        replay=base_cfg.replay,
        training=training_cfg,
        ga=ga_cfg,
        runtime=runtime_cfg,
    )

    if bool(args.smoke):
        cfg = replace(
            cfg,
            ga=replace(cfg.ga, population_size=4, generations=1, total_games_per_agent=2, elite_count=2),
            training=replace(cfg.training, max_steps_per_episode=min(300, cfg.training.max_steps_per_episode)),
        )

    if cfg.ga.population_size <= 0 or cfg.ga.generations <= 0 or cfg.ga.total_games_per_agent <= 0:
        raise ValueError("population_size, generations, and games_per_agent must all be > 0.")
    agent_workers = max(1, int(args.agent_workers))
    if device.type != "cpu" and agent_workers > 1:
        print(
            "[dqn_ref] warning: parallel agent workers with non-CPU device can be unstable; "
            "falling back to --agent_workers 1."
        )
        agent_workers = 1

    viewer_enabled = bool(args.viewer)
    viewer: Any = None
    viewer_queue: Any = None
    viewer_manager: Any = None
    live_viewer_cls: Any = None

    lib_path = find_library(args.lib)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_checkpoint = model_dir / "best_model.pt"
    hiscore_path = run_dir / "hiscore.txt"
    viewer_max_events = max(1, int(args.viewer_max_queue))
    viewer_event_buffer: deque[Dict[str, object]] = deque(maxlen=viewer_max_events)
    viewer_reopen_file = (
        Path(args.viewer_reopen_file) if args.viewer_reopen_file is not None else (run_dir / "VIEWER_OPEN")
    )

    hiscore = 0.0
    if hiscore_path.exists():
        try:
            hiscore = float(hiscore_path.read_text(encoding="utf-8").strip())
        except Exception:
            hiscore = 0.0

    def create_viewer_instance() -> Any:
        if live_viewer_cls is None:
            return None
        try:
            next_viewer = live_viewer_cls(
                total_agents=int(cfg.ga.population_size),
                total_generations=int(cfg.ga.generations),
                games_per_agent=int(cfg.ga.total_games_per_agent),
                fullscreen=bool(args.viewer_fullscreen),
                fps=int(args.viewer_fps),
                grid_padding=int(args.viewer_grid_padding),
                min_tile_px=int(args.viewer_min_tile_px),
                initial_selected_agent=int(args.viewer_agent),
                run_dir=str(run_dir),
            )
            if not next_viewer.ready:
                return None
            return next_viewer
        except Exception:
            return None

    if viewer_enabled:
        try:
            from .viewer_live import LiveTrainingViewer

            live_viewer_cls = LiveTrainingViewer
            viewer = create_viewer_instance()
            if viewer is None:
                print(
                    "[dqn_ref] warning: viewer init failed; starting headless. "
                    f"Create '{viewer_reopen_file}' to retry open."
                )
            if agent_workers > 1:
                viewer_manager = mp.Manager()
                viewer_queue = viewer_manager.Queue(max(1, int(args.viewer_max_queue)))
        except Exception as exc:
            print(f"[dqn_ref] warning: viewer unavailable ({exc}); continuing headless.")
            viewer_enabled = False
            viewer = None

    cfg_payload = {
        "config": {
            "model": asdict(cfg.model),
            "replay": asdict(cfg.replay),
            "training": asdict(cfg.training),
            "ga": asdict(cfg.ga),
            "runtime": asdict(cfg.runtime),
        },
        "viewer": {
            "enabled": bool(viewer_enabled),
            "fullscreen": bool(args.viewer_fullscreen),
            "fps": int(args.viewer_fps),
            "publish_every_steps": int(args.viewer_publish_every_steps),
            "max_queue": int(args.viewer_max_queue),
            "grid_padding": int(args.viewer_grid_padding),
            "min_tile_px": int(args.viewer_min_tile_px),
            "initial_agent": int(args.viewer_agent),
            "reopen_file": str(viewer_reopen_file),
        },
        "lib_path": str(lib_path),
    }
    write_json(run_dir / "config.json", cfg_payload)

    def emit_viewer_event(event: Dict[str, object]) -> None:
        if not viewer_enabled:
            return
        event_payload = dict(event)
        if viewer_queue is not None:
            if not _queue_put_best_effort(viewer_queue, event_payload):
                viewer_event_buffer.append(event_payload)
            return
        viewer_event_buffer.append(event_payload)
        if viewer is not None and not viewer.closed:
            viewer.process_event(event_payload)

    def pump_viewer() -> None:
        nonlocal viewer
        if not viewer_enabled:
            return
        if viewer_queue is not None:
            for _ in range(2048):
                try:
                    ev = viewer_queue.get_nowait()
                except queue_mod.Empty:
                    break
                except Exception:
                    break
                viewer_event_buffer.append(ev)
                if viewer is not None and not viewer.closed:
                    viewer.process_event(ev)

        if viewer is not None and viewer.closed:
            viewer = None

        if viewer is None and live_viewer_cls is not None and viewer_reopen_file.exists():
            try:
                viewer_reopen_file.unlink(missing_ok=True)
            except Exception:
                pass
            viewer = create_viewer_instance()
            if viewer is not None:
                for ev in viewer_event_buffer:
                    viewer.process_event(ev)
                print(f"[dqn_ref] viewer reopened via trigger: {viewer_reopen_file}")
            else:
                print(
                    "[dqn_ref] warning: viewer reopen failed. "
                    f"Create '{viewer_reopen_file}' again to retry."
                )

        if viewer is not None and not viewer.closed:
            viewer.tick()

    episode_csv = run_dir / "episode_metrics.csv"
    generation_csv = run_dir / "generation_metrics.csv"
    summary_json = run_dir / "summary.json"
    generation_history: List[Dict[str, object]] = []

    episode_fields = [
        "generation",
        "agent_index",
        "episode_index",
        "episode_return",
        "lines_cleared",
        "survival_length",
        "epsilon",
        "learning_rate",
        "loss",
        "game_over_term",
        "survival_term",
        "y_pos_term",
        "total_height_term",
        "lines_term",
        "holes_term",
        "bumpiness_term",
        "pillar_term",
        "high_placement_penalty_term",
    ]
    generation_fields = [
        "generation",
        "best_fitness",
        "mean_fitness",
        "std_fitness",
        "diversity",
        "uniform_rate",
        "alpha_rate",
        "mutate_rate",
        "checkpoint_saved",
        "checkpoint_path",
    ]

    with episode_csv.open("w", newline="", encoding="utf-8") as ep_fp, generation_csv.open(
        "w", newline="", encoding="utf-8"
    ) as gen_fp:
        episode_writer = csv.DictWriter(ep_fp, fieldnames=episode_fields)
        generation_writer = csv.DictWriter(gen_fp, fieldnames=generation_fields)
        episode_writer.writeheader()
        generation_writer.writeheader()

        gp = GeneticPopulation(
            config=cfg,
            device=device,
            seed=int(args.seed),
            model_checkpoint=model_checkpoint,
        )
        pop = gp.create_population(cfg.ga.population_size)

        for generation in range(1, int(cfg.ga.generations) + 1):
            print(f"[dqn_ref] generation {generation}/{cfg.ga.generations}")
            emit_viewer_event(
                {
                    "type": "generation_started",
                    "generation": int(generation),
                    "population_size": int(len(pop)),
                    "games_per_agent": int(cfg.ga.total_games_per_agent),
                    "timestamp": float(time.time()),
                }
            )
            fitness_values: List[float] = []
            worker_results: Dict[int, Dict[str, Any]] = {}

            if agent_workers <= 1:
                for agent_index, entry in enumerate(pop, start=1):
                    total_lines = 0.0
                    total_return = 0.0
                    total_survival = 0.0
                    last_metrics: Dict[str, float] = {}
                    emit_viewer_event(
                        {
                            "type": "agent_started",
                            "generation": int(generation),
                            "agent_index": int(agent_index),
                            "games_per_agent": int(cfg.ga.total_games_per_agent),
                            "status": "active",
                            "timestamp": float(time.time()),
                        }
                    )

                    with DQNRefEnvBridge(
                        lib_path=lib_path,
                        seed=episode_seed(args.seed, generation, agent_index, 0),
                    ) as env:
                        def publish_and_pump(event: Dict[str, object]) -> None:
                            emit_viewer_event(event)
                            pump_viewer()

                        for episode_index in range(1, int(cfg.ga.total_games_per_agent) + 1):
                            metrics = run_episode(
                                env=env,
                                agent=entry.agent,
                                seed=episode_seed(args.seed, generation, agent_index, episode_index),
                                max_steps_per_episode=int(cfg.training.max_steps_per_episode),
                                generation=int(generation),
                                agent_index=int(agent_index),
                                episode_index=int(episode_index),
                                games_per_agent=int(cfg.ga.total_games_per_agent),
                                total_lines_before_episode=float(total_lines),
                                publish_every_steps=int(args.viewer_publish_every_steps),
                                publish_event=publish_and_pump if viewer is not None else None,
                            )
                            total_lines += float(metrics["lines_cleared"])
                            total_return += float(metrics["episode_return"])
                            total_survival += float(metrics["survival_length"])
                            last_metrics = metrics

                            row = {
                                "generation": generation,
                                "agent_index": agent_index,
                                "episode_index": episode_index,
                            }
                            row.update(metrics)
                            episode_writer.writerow(row)

                            if episode_index % int(cfg.runtime.log_every_episodes) == 0:
                                print(
                                    "[dqn_ref] "
                                    f"gen={generation} agent={agent_index}/{len(pop)} "
                                    f"ep={episode_index}/{cfg.ga.total_games_per_agent} "
                                    f"lines={metrics['lines_cleared']:.1f} "
                                    f"ret={metrics['episode_return']:.2f} "
                                    f"eps={metrics['epsilon']:.5f} "
                                    f"loss={metrics['loss']:.4f}"
                                )
                            pump_viewer()

                    entry.fitness = float(total_lines / float(cfg.ga.total_games_per_agent))
                    fitness_values.append(entry.fitness)
                    worker_results[int(agent_index)] = {
                        "agent_index": int(agent_index),
                        "fitness": float(entry.fitness),
                        "avg_return": float(total_return / cfg.ga.total_games_per_agent),
                        "avg_survival": float(total_survival / cfg.ga.total_games_per_agent),
                        "last_metrics": dict(last_metrics),
                        "checkpoint_payload": _agent_checkpoint_payload(entry.agent),
                    }
                    print(
                        "[dqn_ref] "
                        f"gen={generation} agent={agent_index} fitness={entry.fitness:.4f} "
                        f"avg_return={total_return / cfg.ga.total_games_per_agent:.2f} "
                        f"avg_survival={total_survival / cfg.ga.total_games_per_agent:.1f}"
                    )
                    emit_viewer_event(
                        {
                            "type": "agent_done",
                            "generation": int(generation),
                            "agent_index": int(agent_index),
                            "episodes_completed": int(cfg.ga.total_games_per_agent),
                            "fitness": float(entry.fitness),
                            "avg_return": float(total_return / cfg.ga.total_games_per_agent),
                            "avg_survival": float(total_survival / cfg.ga.total_games_per_agent),
                            "status": "done",
                            "timestamp": float(time.time()),
                        }
                    )
                    pump_viewer()
            else:
                print(
                    "[dqn_ref] "
                    f"parallel agent evaluation enabled (workers={agent_workers}, "
                    f"population={len(pop)}, games_per_agent={cfg.ga.total_games_per_agent})"
                )
                futures: List[concurrent.futures.Future] = []
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=agent_workers,
                    mp_context=mp.get_context("spawn"),
                ) as executor:
                    for agent_index, entry in enumerate(pop, start=1):
                        futures.append(
                            executor.submit(
                                _evaluate_agent_worker,
                                lib_path=str(lib_path),
                                config=cfg,
                                genome=dict(entry.genome),
                                generation=int(generation),
                                agent_index=int(agent_index),
                                base_seed=int(args.seed),
                                device_str=str(device),
                                checkpoint_path=(
                                    str(model_checkpoint) if model_checkpoint.exists() else None
                                ),
                                viewer_queue=viewer_queue,
                                viewer_publish_every_steps=int(args.viewer_publish_every_steps),
                            )
                        )
                    done_agents = 0
                    pending = set(futures)
                    while pending:
                        done, pending = concurrent.futures.wait(
                            pending,
                            timeout=max(0.01, 1.0 / float(max(5, int(args.viewer_fps)))),
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        pump_viewer()
                        for future in done:
                            result = future.result()
                            agent_index = int(result["agent_index"])
                            worker_results[agent_index] = result
                            done_agents += 1
                            last_metrics = result.get("last_metrics", {})
                            print(
                                "[dqn_ref] "
                                f"gen={generation} agent={agent_index}/{len(pop)} completed "
                                f"({done_agents}/{len(pop)}) "
                                f"fitness={float(result['fitness']):.4f} "
                                f"last_eps={float(last_metrics.get('epsilon', 0.0)):.5f} "
                                f"last_loss={float(last_metrics.get('loss', 0.0)):.4f}"
                            )

                for agent_index, entry in enumerate(pop, start=1):
                    result = worker_results.get(agent_index)
                    if result is None:
                        raise RuntimeError(f"Missing worker result for agent_index={agent_index}.")
                    rows = result.get("episode_rows", [])
                    for row in rows:
                        episode_writer.writerow(row)
                    entry.fitness = float(result["fitness"])
                    fitness_values.append(entry.fitness)
                    print(
                        "[dqn_ref] "
                        f"gen={generation} agent={agent_index} fitness={entry.fitness:.4f} "
                        f"avg_return={float(result['avg_return']):.2f} "
                        f"avg_survival={float(result['avg_survival']):.1f}"
                    )

            best_entry_index = max(range(len(pop)), key=lambda idx: float(pop[idx].fitness))
            best_entry = pop[best_entry_index]
            best_agent_index = int(best_entry_index + 1)
            best_fitness = float(best_entry.fitness)
            mean_fitness = float(np.mean(fitness_values))
            std_fitness = float(np.std(fitness_values))
            diversity = compute_diversity(pop)
            uniform_rate, alpha_rate = gp.get_crossover_rates(generation)
            mutate_rate = gp.get_mutate_rate(generation)

            checkpoint_saved = False
            if (hiscore <= 0.0 and best_fitness > 0.0) or (best_fitness > hiscore * 1.10):
                best_result = worker_results.get(best_agent_index)
                if best_result is None:
                    raise RuntimeError(
                        f"Missing best agent result for checkpoint save (agent={best_agent_index})."
                    )
                torch.save(best_result["checkpoint_payload"], model_checkpoint)
                hiscore = best_fitness
                hiscore_path.write_text(str(hiscore), encoding="utf-8")
                checkpoint_saved = True
                print(f"[dqn_ref] checkpoint updated: {model_checkpoint} (hiscore={hiscore:.4f})")

            gen_row = {
                "generation": generation,
                "best_fitness": best_fitness,
                "mean_fitness": mean_fitness,
                "std_fitness": std_fitness,
                "diversity": diversity,
                "uniform_rate": float(uniform_rate),
                "alpha_rate": float(alpha_rate),
                "mutate_rate": float(mutate_rate),
                "checkpoint_saved": int(checkpoint_saved),
                "checkpoint_path": str(model_checkpoint if checkpoint_saved else ""),
            }
            generation_writer.writerow(gen_row)
            generation_history.append(
                {
                    **gen_row,
                    "best_genome": {k: float(v) for k, v in best_entry.genome.items()},
                }
            )

            write_json(
                summary_json,
                {
                    "hiscore": float(hiscore),
                    "latest_generation": int(generation),
                    "generation_history": generation_history,
                    "checkpoint_path": str(model_checkpoint) if model_checkpoint.exists() else "",
                },
            )

            print(
                "[dqn_ref] "
                f"generation={generation} best={best_fitness:.4f} "
                f"mean={mean_fitness:.4f} std={std_fitness:.4f} diversity={diversity:.4f}"
            )
            emit_viewer_event(
                {
                    "type": "generation_done",
                    "generation": int(generation),
                    "best_fitness": float(best_fitness),
                    "mean_fitness": float(mean_fitness),
                    "std_fitness": float(std_fitness),
                    "timestamp": float(time.time()),
                }
            )
            pump_viewer()

            if generation < int(cfg.ga.generations):
                elites = gp.best_elites(pop)
                pop = gp.next_population(elites, generation_number=generation)

    print(f"[dqn_ref] done. artifacts: {run_dir}")
    if viewer is not None:
        viewer.close()
    if viewer_manager is not None:
        try:
            viewer_manager.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
