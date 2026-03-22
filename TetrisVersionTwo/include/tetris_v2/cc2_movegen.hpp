#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "tetris_v2/cc2_data.hpp"

namespace tetris_v2::cc2 {

std::vector<std::pair<Placement, std::uint32_t>> find_moves(const Board& board, Piece piece);

}  // namespace tetris_v2::cc2

