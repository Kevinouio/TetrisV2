from __future__ import annotations

import numpy as np

from TetrisVersionTwo.scripts.dqn_ref.features import compute_features_from_board
from TetrisVersionTwo.scripts.dqn_ref.reward import ReferenceReward
from TetrisVersionTwo.scripts.dqn_ref.config import SEEDED_GENOME


def test_features_empty_board() -> None:
    board = np.zeros((20, 10), dtype=np.uint8)
    feat = compute_features_from_board(board, y_pos=0, lines_removed=0)
    arr = feat.as_array()
    assert arr.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_features_simple_columns() -> None:
    board = np.zeros((20, 10), dtype=np.uint8)
    board[19, 0] = 1
    board[18, 0] = 1
    board[19, 1] = 1
    feat = compute_features_from_board(board, y_pos=19, lines_removed=1)
    arr = feat.as_array()
    # heights: [2,1,0,...] => bumpiness=|2-1|+|1-0|=2.
    assert arr[0] == 3.0
    assert arr[1] == 2.0
    assert arr[2] == 1.0
    assert arr[3] == 0.0
    assert arr[5] == 0.0


def test_reward_returns_float_terms() -> None:
    reward = ReferenceReward(SEEDED_GENOME)
    features = np.asarray([40.0, 6.0, 2.0, 1.0, 10.0, 0.0], dtype=np.float32)
    terms = reward.compute(features, finished=False)
    as_map = terms.to_dict()
    assert isinstance(as_map["total"], float)
    assert "lines_term" in as_map
    assert "holes_term" in as_map
