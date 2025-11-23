#pragma once

#include <array>
#include <cstdint>

#include "cold_clear_cpp/types.hpp"

namespace cold_clear_cpp {

struct Board {
    std::array<std::uint64_t, 10> cols{};

    bool occupied(std::pair<std::int8_t, std::int8_t> cell) const;
    std::int8_t distance_to_ground(std::int8_t x, std::int8_t y) const;
    void place(const PieceLocation& location);
    std::uint64_t line_clears() const;
    void remove_lines(std::uint64_t mask);

    bool operator==(const Board& rhs) const { return cols == rhs.cols; }
};

}  // namespace cold_clear_cpp
