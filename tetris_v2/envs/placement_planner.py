"""Precompute action sequences for discrete placement actions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import utils
from .modern_ruleset import ModernTetrisKicks


@dataclass(frozen=True)
class PlacementActionSpec:
    """Describe a macro placement choice."""

    use_hold: bool
    rotation: int
    column: int


@dataclass(frozen=True)
class PlacementPlan:
    """Concrete set of environment actions to realise a placement."""

    spec: PlacementActionSpec
    actions: Tuple[int, ...]
    landing_row: int


@dataclass(frozen=True)
class ActionCodes:
    """Container for low-level action identifiers."""

    move_left: int
    move_right: int
    soft_drop: int
    hard_drop: int
    rotate_cw: int
    rotate_ccw: Optional[int]
    rotate_180: Optional[int]
    hold: Optional[int]
    neutral: Optional[int]


@dataclass(frozen=True)
class HoldContext:
    """Describe whether hold can be used this placement."""

    supports_hold: bool
    can_hold: bool
    hold_piece_id: Optional[int]
    queue_head_id: Optional[int]


def build_action_specs(*, allow_hold: bool, board_width: int = utils.BOARD_WIDTH) -> List[PlacementActionSpec]:
    """Return the catalog of rotation/column combinations for macro actions."""

    specs: List[PlacementActionSpec] = []
    hold_options = (False, True) if allow_hold else (False,)
    for use_hold in hold_options:
        for rotation in range(4):
            for column in range(board_width):
                specs.append(PlacementActionSpec(use_hold=use_hold, rotation=rotation, column=column))
    return specs


def compute_action_plans(
    *,
    board: np.ndarray,
    specs: Sequence[PlacementActionSpec],
    current_piece: Optional[utils.PieceState],
    action_codes: ActionCodes,
    hold_context: HoldContext,
    is_modern: bool,
) -> Dict[PlacementActionSpec, PlacementPlan]:
    """Return executable plans for every reachable placement spec."""

    plans: Dict[PlacementActionSpec, PlacementPlan] = {}

    base_specs = [spec for spec in specs if not spec.use_hold]
    if current_piece is not None and base_specs:
        plans.update(
            _plans_for_piece(
                board=board,
                piece=current_piece,
                specs=base_specs,
                action_codes=action_codes,
                is_modern=is_modern,
                prefix=(),
            )
        )

    if hold_context.supports_hold and hold_context.can_hold and action_codes.hold is not None:
        hold_piece_id = hold_context.hold_piece_id
        if hold_piece_id is None:
            hold_piece_id = hold_context.queue_head_id
        if hold_piece_id is not None:
            hold_piece = utils.spawn_piece(hold_piece_id)
            hold_specs = [spec for spec in specs if spec.use_hold]
            prefix = (action_codes.hold,)
            plans.update(
                _plans_for_piece(
                    board=board,
                    piece=hold_piece,
                    specs=hold_specs,
                    action_codes=action_codes,
                    is_modern=is_modern,
                    prefix=prefix,
                )
            )

    return plans


def _plans_for_piece(
    *,
    board: np.ndarray,
    piece: utils.PieceState,
    specs: Sequence[PlacementActionSpec],
    action_codes: ActionCodes,
    is_modern: bool,
    prefix: Tuple[int, ...],
) -> Dict[PlacementActionSpec, PlacementPlan]:
    if not specs:
        return {}
    alignments, parents, state_lookup = _enumerate_reachable_states(board, piece, action_codes, is_modern)
    plans: Dict[PlacementActionSpec, PlacementPlan] = {}
    start_key = _state_key(piece)
    for spec in specs:
        target_key = alignments.get((spec.rotation % 4, spec.column))
        if target_key is None:
            continue
        actions = _reconstruct_actions(parents, start_key, target_key)
        expanded = _apply_tap_expansion(actions, action_codes)
        sequence = prefix + tuple(expanded) + (action_codes.hard_drop,)
        landing_row = _landing_row(board, state_lookup[target_key])
        plans[spec] = PlacementPlan(spec=spec, actions=sequence, landing_row=landing_row)
    return plans


def _enumerate_reachable_states(
    board: np.ndarray,
    start_piece: utils.PieceState,
    action_codes: ActionCodes,
    is_modern: bool,
) -> Tuple[Dict[Tuple[int, int], Tuple[int, int, int]], Dict[Tuple[int, int, int], Tuple[Tuple[int, int, int], int]], Dict[Tuple[int, int, int], utils.PieceState]]:
    queue: Deque[utils.PieceState] = deque([start_piece])
    start_key = _state_key(start_piece)
    parents: Dict[Tuple[int, int, int], Tuple[Tuple[int, int, int], int]] = {}
    alignments: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    state_lookup: Dict[Tuple[int, int, int], utils.PieceState] = {start_key: start_piece}
    visited = {start_key}
    actions = _available_actions(action_codes)

    while queue:
        state = queue.popleft()
        key = _state_key(state)
        align_key = (state.rotation % 4, state.col)
        alignments.setdefault(align_key, key)

        for action in actions:
            next_state = _simulate_action(board, state, action, action_codes, is_modern)
            if next_state is None:
                continue
            next_key = _state_key(next_state)
            if next_key in visited:
                continue
            visited.add(next_key)
            parents[next_key] = (key, action)
            state_lookup[next_key] = next_state
            queue.append(next_state)

    return alignments, parents, state_lookup


def _state_key(piece: utils.PieceState) -> Tuple[int, int, int]:
    return (int(piece.row), int(piece.col), int(piece.rotation % 4))


def _reconstruct_actions(
    parents: Dict[Tuple[int, int, int], Tuple[Tuple[int, int, int], int]],
    start_key: Tuple[int, int, int],
    end_key: Tuple[int, int, int],
) -> List[int]:
    actions: List[int] = []
    key = end_key
    while key != start_key:
        parent, action = parents[key]
        actions.append(action)
        key = parent
    actions.reverse()
    return actions


def _available_actions(action_codes: ActionCodes) -> List[int]:
    actions = [action_codes.move_left, action_codes.move_right, action_codes.soft_drop]
    actions.append(action_codes.rotate_cw)
    if action_codes.rotate_ccw is not None:
        actions.append(action_codes.rotate_ccw)
    if action_codes.rotate_180 is not None:
        actions.append(action_codes.rotate_180)
    return actions


def _apply_tap_expansion(actions: Sequence[int], action_codes: ActionCodes) -> List[int]:
    expanded: List[int] = []
    for idx, code in enumerate(actions):
        expanded.append(code)
        next_code = actions[idx + 1] if idx + 1 < len(actions) else None
        if (
            action_codes.neutral is not None
            and code in (action_codes.move_left, action_codes.move_right)
            and next_code == code
        ):
            expanded.append(action_codes.neutral)
    return expanded


def _simulate_action(
    board: np.ndarray,
    piece: utils.PieceState,
    action: int,
    action_codes: ActionCodes,
    is_modern: bool,
) -> Optional[utils.PieceState]:
    if action == action_codes.move_left:
        candidate = piece.moved(d_col=-1)
    elif action == action_codes.move_right:
        candidate = piece.moved(d_col=1)
    elif action == action_codes.soft_drop:
        candidate = piece.moved(d_row=1)
    else:
        candidate = _rotate_with_kicks(piece, action, action_codes, board, is_modern)
    if candidate is None:
        return None
    min_row = -(utils.HIDDEN_ROWS + 4)
    if candidate.row < min_row:
        return None
    if _collides(board, candidate):
        return None
    return candidate


def _rotate_with_kicks(
    piece: utils.PieceState,
    action: int,
    action_codes: ActionCodes,
    board: np.ndarray,
    is_modern: bool,
) -> Optional[utils.PieceState]:
    if action == action_codes.rotate_cw:
        delta = 1
    elif action_codes.rotate_ccw is not None and action == action_codes.rotate_ccw:
        delta = -1
    elif action_codes.rotate_180 is not None and action == action_codes.rotate_180:
        delta = 2
    else:
        return None
    rotated = piece.rotated(delta)
    kicks: Iterable[Tuple[int, int]]
    if not is_modern:
        kicks = ((0, 0),)
    else:
        from_rotation = piece.rotation % 4
        to_rotation = rotated.rotation % 4
        if delta % 4 == 2:
            kicks = ModernTetrisKicks.kicks_180_for(piece.piece_id, from_rotation, to_rotation)
        else:
            kicks = ModernTetrisKicks.kicks_for(piece.piece_id, from_rotation, to_rotation)
    for dr, dc in kicks:
        candidate = utils.PieceState(
            piece_id=rotated.piece_id,
            rotation=rotated.rotation,
            row=rotated.row + dr,
            col=rotated.col + dc,
        )
        if not _collides(board, candidate):
            return candidate
    return None


def _collides(board: np.ndarray, piece: utils.PieceState) -> bool:
    return utils.collides(board, piece)


def _landing_row(board: np.ndarray, piece: utils.PieceState) -> int:
    final_piece = utils.project_lock_position(board, piece)
    return int(final_piece.row)


def simulate_action_sequence(
    board: np.ndarray,
    start_piece: utils.PieceState,
    actions: Sequence[int],
    action_codes: ActionCodes,
    is_modern: bool,
) -> utils.PieceState:
    """Replay a macro action sequence and return the resulting piece state."""

    piece = start_piece
    for action in actions:
        if action_codes.neutral is not None and action == action_codes.neutral:
            continue
        if action_codes.hold is not None and action == action_codes.hold:
            raise ValueError("simulate_action_sequence does not support hold transitions.")
        if action == action_codes.hard_drop:
            piece = utils.project_lock_position(board, piece)
            break
        next_piece = _simulate_action(board, piece, action, action_codes, is_modern)
        if next_piece is None:
            raise ValueError("Action sequence led to an invalid state.")
        piece = next_piece
    return piece
