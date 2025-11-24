#pragma once

#include <optional>
#include <vector>

#include "cold_clear_cpp/types.hpp"
#include "tetris_v2/tetris_env.hpp"

namespace tetris_v2 {

class Agent {
public:
    virtual ~Agent() = default;
    virtual std::optional<cold_clear_cpp::Placement> choose_move(std::size_t iterations = 1024) = 0;
    virtual void on_new_pieces(const std::vector<cold_clear_cpp::Piece>& pieces) = 0;
    virtual void on_advance(const cold_clear_cpp::Placement& mv) = 0;
};

}  // namespace tetris_v2
