#include "cold_clear_cpp/board.hpp"

#include <algorithm>

namespace cold_clear_cpp {

bool Board::occupied(std::pair<std::int8_t, std::int8_t> cell) const {
    auto [x, y] = cell;
    if (x < 0 || x >= 10 || y < 0 || y >= 40) {
        return true;
    }
    return (cols[static_cast<std::size_t>(x)] & (1ull << y)) != 0;
}

std::int8_t Board::distance_to_ground(std::int8_t x, std::int8_t y) const {
    if (x < 0 || x >= 10 || y < 0 || y >= 40) {
        return 0;
    }

    std::int8_t dist = 0;
    while (y - dist - 1 >= 0) {
        auto next_y = y - dist - 1;
        if (cols[static_cast<std::size_t>(x)] & (1ull << next_y)) {
            break;
        }
        dist++;
    }
    return dist;
}

void Board::place(const PieceLocation& location) {
    for (auto cell : location.cells()) {
        auto [cx, cy] = cell;
        if (cx < 0 || cx >= 10 || cy < 0 || cy >= 40) {
            continue;
        }
        cols[static_cast<std::size_t>(cx)] |= (1ull << cy);
    }
}

std::uint64_t Board::line_clears() const {
    std::uint64_t mask = ~0ull;
    for (auto c : cols) {
        mask &= c;
    }
    return mask;
}

void Board::remove_lines(std::uint64_t mask) {
    for (auto& column : cols) {
        std::uint64_t new_col = 0;
        std::int8_t dst = 0;
        for (std::int8_t src = 0; src < 40; ++src) {
            if (mask & (1ull << src)) {
                continue;
            }
            if (column & (1ull << src)) {
                new_col |= (1ull << dst);
            }
            dst++;
        }
        column = new_col;
    }
}

}  // namespace cold_clear_cpp
