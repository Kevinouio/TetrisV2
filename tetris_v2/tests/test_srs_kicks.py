import pytest

pytest.importorskip("gymnasium")

from tetris_v2.envs import utils
from tetris_v2.envs.modern_ruleset import ModernRuleset


def _fresh_rules(piece_name: str, col: int) -> ModernRuleset:
    rules = ModernRuleset()
    rules._board = utils.create_board()
    piece_id = utils.NAME_TO_ID[piece_name]
    rules._current = utils.PieceState(piece_id=piece_id, rotation=0, row=utils.NES_SPAWN_ROW, col=col)
    return rules


def test_s_piece_wall_kick():
    rules = _fresh_rules("S", col=0)
    success = rules._attempt_rotate(1)
    assert success
    assert not utils.collides(rules._board, rules._current)
    assert rules._current.col >= 0


def test_z_piece_wall_kick():
    rules = _fresh_rules("Z", col=utils.BOARD_WIDTH - 3)
    success = rules._attempt_rotate(-1)
    assert success
    assert not utils.collides(rules._board, rules._current)
    assert rules._current.col <= utils.BOARD_WIDTH - 1


def test_i_piece_floor_kick():
    rules = ModernRuleset()
    rules._board = utils.create_board()
    piece_id = utils.NAME_TO_ID["I"]
    rules._current = utils.PieceState(piece_id=piece_id, rotation=0, row=utils.NES_SPAWN_ROW, col=2)
    # Drop piece near the floor to force upward kick
    for _ in range(18):
        rules._current = rules._current.moved(d_row=1)
    success = rules._attempt_rotate(1)
    assert success
    assert not utils.collides(rules._board, rules._current)
