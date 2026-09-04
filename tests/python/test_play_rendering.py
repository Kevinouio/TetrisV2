from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

from tetris_v2.play.rendering import (
    EMPTY_CELL_ID,
    FeedbackLayer,
    PiecePose,
    PoseTween,
    Renderer,
    landing_y,
    piece_cells,
    visible_cells,
)


@pytest.fixture(scope="module", autouse=True)
def pygame_runtime():
    pygame.init()
    yield
    pygame.quit()


def empty_board() -> list[list[int]]:
    return [[EMPTY_CELL_ID for _ in range(10)] for _ in range(20)]


def test_piece_geometry_matches_engine_rotation_convention() -> None:
    assert set(piece_cells(0, 0)) == {(-1, 0), (0, 0), (1, 0), (2, 0)}
    assert set(piece_cells(2, 1)) == {(0, 1), (0, 0), (0, -1), (1, 0)}
    assert piece_cells(7, 0) == ()


def test_ghost_lands_on_floor_and_visible_cells_use_top_down_rows() -> None:
    board = empty_board()
    assert landing_y(board, piece=2, rotation=0, x=4, y=19) == 0
    assert set(visible_cells(PiecePose(2, 0, 4, 0))) == {
        (3, 19),
        (4, 19),
        (5, 19),
        (4, 18),
    }

    board[19][4] = 1
    assert landing_y(board, piece=2, rotation=0, x=4, y=19) == 1


def test_pose_tween_snaps_geometry_and_eases_position() -> None:
    tween = PoseTween(duration_ms=45)
    tween.reset(PiecePose(2, 0, 4, 10))
    tween.set_target(PiecePose(2, 1, 5, 9))
    pose = tween.update(8)
    assert pose is not None
    assert pose.rotation == 1
    assert 4 < pose.x < 5
    assert 9 < pose.y < 10


def test_feedback_is_bounded() -> None:
    feedback = FeedbackLayer(max_particles=24)
    for _ in range(20):
        feedback.emit_line_clear([18, 19], (255, 255, 255))
    assert len(feedback.particles) == 24
    for _ in range(6):
        feedback.update(0.1)
    assert not feedback.particles


def test_renderer_draws_headlessly_and_accepts_mapping_step_events() -> None:
    board = empty_board()
    snapshot = {
        "board": board,
        "active": {"piece": 2, "rotation": 0, "x": 4, "y": 8},
        "hold": {"has_hold": True, "hold_piece": 0, "hold_available": False},
        "queue": [1, 3, 4, 5, 6],
        "meta": {"lines": 12, "combo": 1, "b2b": True},
    }
    renderer = Renderer(enable_audio=False, max_particles=32)
    logical = renderer.render(
        None,
        snapshot,
        {"score": 12345, "elapsed_s": 62.0},
        "playing",
        1000,
    )
    assert logical.get_size() == (960, 900)
    assert logical.get_at(renderer.layout.board_rect.center) != pygame.Color(0, 0, 0, 255)

    renderer.on_step(
        snapshot,
        {
            "action_succeeded": True,
            "piece_locked": True,
            "lines_cleared": 1,
            "combo": 1,
        },
        "hard_drop",
        {**snapshot, "active": {"piece": 1, "rotation": 0, "x": 4, "y": 19}},
        1010,
    )
    assert renderer.effects.particles
    assert renderer.effects.line_flashes
    assert renderer.effects.callouts

    destination = pygame.Surface((1280, 720))
    fitted = renderer.present(destination)
    assert fitted.size == (768, 720)
    assert fitted.center == (640, 360)


def test_renderer_owned_tween_tracks_steps_and_resets_at_piece_boundaries() -> None:
    renderer = Renderer(enable_audio=False)
    board = empty_board()
    before = {
        "board": board,
        "active": {"piece": 2, "rotation": 0, "x": 4, "y": 10},
        "queue": [],
        "meta": {},
    }
    renderer.render(None, before)
    after_move = {
        **before,
        "active": {"piece": 2, "rotation": 1, "x": 5, "y": 10},
    }
    renderer.on_step(
        before,
        {"action_succeeded": True, "piece_locked": False},
        "rotate_cw",
        after_move,
    )
    renderer.update(8)
    pose = renderer.pose_tween.pose
    assert pose is not None
    assert pose.rotation == 1
    assert 4 < pose.x < 5

    spawned = {
        **before,
        "active": {"piece": 2, "rotation": 0, "x": 4, "y": 19},
    }
    renderer.on_step(
        after_move,
        {"action_succeeded": True, "piece_locked": True},
        "hard_drop",
        spawned,
    )
    assert renderer.pose_tween.pose == PiecePose(2, 0, 4, 19)
