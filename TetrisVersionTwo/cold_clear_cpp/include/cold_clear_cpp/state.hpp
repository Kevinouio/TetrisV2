#pragma once

#include <cstdint>

#include "cold_clear_cpp/board.hpp"

namespace cold_clear_cpp {

struct GameState {
    Board board{};
    PieceSet bag{all_pieces()};
    Piece reserve{Piece::I};
    bool back_to_back{false};
    std::uint8_t combo{0};

    PlacementInfo advance(Piece next, const Placement& placement);
    bool operator==(const GameState& rhs) const {
        return board == rhs.board && bag == rhs.bag && reserve == rhs.reserve &&
               back_to_back == rhs.back_to_back && combo == rhs.combo;
    }
};

struct GameStateHash {
    std::size_t operator()(const GameState& state) const noexcept;
};

}  // namespace cold_clear_cpp
