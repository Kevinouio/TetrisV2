#include "cold_clear_cpp/board.hpp"

#include <algorithm>
#include <cassert>
#if defined(__BMI2__)
#include <immintrin.h>
#endif

namespace cold_clear_cpp {

bool Board::occupied(std::pair<std::int8_t, std::int8_t> cell) const {
    auto [x, y] = cell;
    if (x < 0 || x >= 10 || y < 0 || y >= 40) {
        return true;
    }
    return (cols[static_cast<std::size_t>(x)] & (1ull << y)) != 0;
}

std::int8_t Board::distance_to_ground(std::int8_t x, std::int8_t y) const {
    // Match the Rust implementation: inputs are expected to be in range, and the
    // distance is computed via a leading-ones count on the inverted, shifted
    // column bitboard.
    assert(x >= 0 && x < 10);
    assert(y >= 0 && y < 40);
    if (y == 0) {
        return 0;
    }

    auto column = cols[static_cast<std::size_t>(x)];
    std::uint64_t shifted = (~column) << static_cast<unsigned>(64 - y);

#if defined(__GNUG__)
    // leading_ones(v) == leading_zeros(~v)
    return static_cast<std::int8_t>(__builtin_clzll(~shifted));
#else
    std::uint64_t v = ~shifted;
    int leading = 0;
    while ((v & (1ull << 63)) == 0 && leading < 64) {
        v <<= 1;
        ++leading;
    }
    return static_cast<std::int8_t>(leading);
#endif
}

void Board::place(const PieceLocation& location) {
    for (auto cell : location.cells()) {
        auto [cx, cy] = cell;
        assert(cx >= 0 && cx < 10 && cy >= 0 && cy < 40);
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
#if defined(__BMI2__)
        column = _pext_u64(column, ~mask);
#else
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
#endif
    }
}

}  // namespace cold_clear_cpp
