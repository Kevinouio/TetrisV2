from __future__ import annotations

import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tetris_v2.envs import utils
from tetris_v2.envs.modern_ruleset import (
    HARD_DROP as MODERN_HARD_DROP,
    HOLD as MODERN_HOLD,
    MOVE_LEFT as MODERN_MOVE_LEFT,
    MOVE_NONE as MODERN_MOVE_NONE,
    MOVE_RIGHT as MODERN_MOVE_RIGHT,
    SOFT_DROP as MODERN_SOFT_DROP,
    ROTATE_CCW as MODERN_ROTATE_CCW,
    ROTATE_CW as MODERN_ROTATE_CW,
    ROTATE_180 as MODERN_ROTATE_180,
)
from tetris_v2.envs.nes_tetris_env import NesTetrisEnv
from tetris_v2.envs.placement_planner import (
    ActionCodes,
    HoldContext,
    build_action_specs,
    compute_action_plans,
    simulate_action_sequence,
)


MODERN_CODES = ActionCodes(
    move_left=MODERN_MOVE_LEFT,
    move_right=MODERN_MOVE_RIGHT,
    soft_drop=MODERN_SOFT_DROP,
    hard_drop=MODERN_HARD_DROP,
    rotate_cw=MODERN_ROTATE_CW,
    rotate_ccw=MODERN_ROTATE_CCW,
    rotate_180=MODERN_ROTATE_180,
    hold=MODERN_HOLD,
    neutral=MODERN_MOVE_NONE,
)

NES_CODES = ActionCodes(
    move_left=NesTetrisEnv.MOVE_LEFT,
    move_right=NesTetrisEnv.MOVE_RIGHT,
    soft_drop=NesTetrisEnv.SOFT_DROP,
    hard_drop=NesTetrisEnv.HARD_DROP,
    rotate_cw=NesTetrisEnv.ROTATE_CW,
    rotate_ccw=None,
    rotate_180=None,
    hold=None,
    neutral=NesTetrisEnv.MOVE_NONE,
)


def test_modern_empty_board_has_all_columns():
    board = utils.create_board()
    piece = utils.spawn_piece(utils.NAME_TO_ID["T"])
    specs = build_action_specs(allow_hold=False)
    plans = compute_action_plans(
        board=board,
        specs=specs,
        current_piece=piece,
        action_codes=MODERN_CODES,
        hold_context=HoldContext(False, False, None, None),
        is_modern=True,
    )
    cols_with_rotation0 = {plan.spec.column for plan in plans.values() if plan.spec.rotation == 0}
    assert cols_with_rotation0, "Expected some reachable columns for rotation 0"
    assert 4 in cols_with_rotation0
    assert min(cols_with_rotation0) == 0
    assert max(cols_with_rotation0) >= 7


def test_hold_plan_includes_hold_action():
    board = utils.create_board()
    current_piece = utils.spawn_piece(utils.NAME_TO_ID["J"])
    specs = build_action_specs(allow_hold=True)
    hold_ctx = HoldContext(True, True, utils.NAME_TO_ID["I"], None)
    plans = compute_action_plans(
        board=board,
        specs=specs,
        current_piece=current_piece,
        action_codes=MODERN_CODES,
        hold_context=hold_ctx,
        is_modern=True,
    )
    hold_plans = [plan for plan in plans.values() if plan.spec.use_hold]
    assert hold_plans, "Expected at least one hold-enabled plan."
    assert all(plan.actions[0] == MODERN_HOLD for plan in hold_plans)


def test_sequence_reaches_requested_column():
    board = utils.create_board()
    # Build a simple stack to force interesting movement.
    board[-1, :3] = 1
    piece = utils.spawn_piece(utils.NAME_TO_ID["L"])
    specs = build_action_specs(allow_hold=False)
    plans = compute_action_plans(
        board=board,
        specs=specs,
        current_piece=piece,
        action_codes=NES_CODES,
        hold_context=HoldContext(False, False, None, None),
        is_modern=False,
    )
    assert plans, "Expected at least one valid plan."
    for plan in plans.values():
        # Skip hold plans (not present for NES) and ensure we can replay actions.
        final_piece = simulate_action_sequence(
            board,
            piece,
            plan.actions,
            NES_CODES,
            is_modern=False,
        )
        assert final_piece.col == plan.spec.column
        assert final_piece.rotation % 4 == plan.spec.rotation % 4
        assert final_piece.row == plan.landing_row
