#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "cold_clear_cpp/state.hpp"

namespace cold_clear_cpp {

std::vector<std::pair<Placement, std::uint32_t>> find_moves(const Board& board, Piece piece);

}  // namespace cold_clear_cpp
