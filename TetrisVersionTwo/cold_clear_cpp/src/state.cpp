#include "cold_clear_cpp/state.hpp"

#include <algorithm>
#include <numeric>

namespace cold_clear_cpp {

namespace {
inline int column_height(std::uint64_t col) {
    if (col == 0) {
        return 0;
    }
#if defined(__GNUG__)
    return 64 - __builtin_clzll(col);
#else
    int height = 0;
    while (col) {
        col >>= 1;
        ++height;
    }
    return height;
#endif
}

inline std::size_t hash_combine(std::size_t seed, std::size_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
    return seed;
}
}  // namespace

bool PieceLocation::obstructed(const Board& board) const {
    for (auto cell : cells()) {
        if (board.occupied(cell)) {
            return true;
        }
    }
    return false;
}

std::int8_t PieceLocation::drop_distance(const Board& board) const {
    std::int8_t dist = 40;
    for (auto cell : cells()) {
        auto [cx, cy] = cell;
        dist = std::min<std::int8_t>(dist, board.distance_to_ground(cx, cy));
    }
    return dist;
}

bool PieceLocation::above_stack(const Board& board) const {
    for (auto cell : cells()) {
        auto [cx, cy] = cell;
        if (cx < 0 || cx >= 10) {
            return false;
        }
        if (cy < column_height(board.cols[static_cast<std::size_t>(cx)])) {
            return false;
        }
    }
    return true;
}

PlacementInfo GameState::advance(Piece next, const Placement& placement) {
    bag.reset(piece_index(next));
    if (bag.none()) {
        bag = all_pieces();
    }
    if (placement.location.piece != next) {
        reserve = next;
    }

    board.place(placement.location);
    auto cleared_mask = board.line_clears();
    bool b2b = false;
    if (cleared_mask != 0) {
        board.remove_lines(cleared_mask);
        auto cleared = static_cast<std::uint32_t>(__builtin_popcountll(cleared_mask));
        bool hard = (cleared == 4) || (placement.spin != Spin::None);
        b2b = hard && back_to_back;
        back_to_back = hard;
        combo = static_cast<std::uint8_t>(std::min<std::uint32_t>(combo + 1, 255));
    } else {
        combo = 0;
    }

    PlacementInfo info;
    info.placement = placement;
    info.lines_cleared = static_cast<std::uint32_t>(__builtin_popcountll(cleared_mask));
    info.combo = combo;
    info.back_to_back = b2b;
    info.perfect_clear = std::all_of(
        board.cols.begin(), board.cols.end(), [](std::uint64_t c) { return c == 0; });

    return info;
}

std::size_t GameStateHash::operator()(const GameState& state) const noexcept {
    std::size_t h = 1469598103934665603ull;
    for (auto col : state.board.cols) {
        h = hash_combine(h, std::hash<std::uint64_t>{}(col));
    }
    h = hash_combine(h, static_cast<std::size_t>(state.reserve));
    h = hash_combine(h, static_cast<std::size_t>(state.back_to_back));
    h = hash_combine(h, static_cast<std::size_t>(state.combo));
    h = hash_combine(h, state.bag.to_ullong());
    return h;
}

}  // namespace cold_clear_cpp
