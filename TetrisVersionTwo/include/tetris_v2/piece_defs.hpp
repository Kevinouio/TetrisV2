#pragma once

#include <array>

#include "tetris_v2/types.hpp"

namespace tetris_v2 {

std::array<Cell, 4> piece_cells(Piece piece, Rotation rotation);
ActivePiece spawn_piece(Piece piece);

}  // namespace tetris_v2
