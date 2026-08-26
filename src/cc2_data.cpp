#include "tetris_v2/cc2_data.hpp"

#include <algorithm>

namespace tetris_v2::cc2 {

namespace {

int highest_filled_row_plus_one(std::uint64_t col) {
    return static_cast<int>(64u - bit_ops::countl_zero_u64(col));
}

std::array<Vec2, 4> base_cells(Piece piece) {
    switch (piece) {
        case Piece::I: return {Vec2{-1, 0}, Vec2{0, 0}, Vec2{1, 0}, Vec2{2, 0}};
        case Piece::O: return {Vec2{0, 0}, Vec2{1, 0}, Vec2{0, 1}, Vec2{1, 1}};
        case Piece::T: return {Vec2{-1, 0}, Vec2{0, 0}, Vec2{1, 0}, Vec2{0, 1}};
        case Piece::L: return {Vec2{-1, 0}, Vec2{0, 0}, Vec2{1, 0}, Vec2{1, 1}};
        case Piece::J: return {Vec2{-1, 0}, Vec2{0, 0}, Vec2{1, 0}, Vec2{-1, 1}};
        case Piece::S: return {Vec2{-1, 0}, Vec2{0, 0}, Vec2{0, 1}, Vec2{1, 1}};
        case Piece::Z: return {Vec2{-1, 1}, Vec2{0, 1}, Vec2{0, 0}, Vec2{1, 0}};
        case Piece::None: break;
    }
    return {Vec2{0, 0}, Vec2{0, 0}, Vec2{0, 0}, Vec2{0, 0}};
}

void clear_lines_column(std::uint64_t* col, std::uint64_t lines) {
    while (lines != 0) {
        const unsigned int i = bit_ops::countr_zero_u64(lines);
        const std::uint64_t mask = (1ull << i) - 1ull;
        *col = (*col & mask) | ((*col >> 1u) & ~mask);
        lines &= ~(1ull << i);
        lines >>= 1u;
    }
}

}  // namespace

Rotation rotate_cw(Rotation r) {
    switch (r) {
        case Rotation::North: return Rotation::East;
        case Rotation::East: return Rotation::South;
        case Rotation::South: return Rotation::West;
        case Rotation::West: return Rotation::North;
    }
    return Rotation::North;
}

Rotation rotate_ccw(Rotation r) {
    switch (r) {
        case Rotation::North: return Rotation::West;
        case Rotation::West: return Rotation::South;
        case Rotation::South: return Rotation::East;
        case Rotation::East: return Rotation::North;
    }
    return Rotation::North;
}

Rotation rotate_flip(Rotation r) {
    switch (r) {
        case Rotation::North: return Rotation::South;
        case Rotation::South: return Rotation::North;
        case Rotation::East: return Rotation::West;
        case Rotation::West: return Rotation::East;
    }
    return Rotation::North;
}

Vec2 rotate_cell(Rotation r, Vec2 v) {
    switch (r) {
        case Rotation::North: return v;
        case Rotation::East: return Vec2{v.y, static_cast<std::int8_t>(-v.x)};
        case Rotation::South:
            return Vec2{static_cast<std::int8_t>(-v.x), static_cast<std::int8_t>(-v.y)};
        case Rotation::West: return Vec2{static_cast<std::int8_t>(-v.y), v.x};
    }
    return v;
}

Rotation rotation_from_env(tetris_v2::Rotation r) {
    switch (r) {
        case tetris_v2::Rotation::North: return Rotation::North;
        case tetris_v2::Rotation::East: return Rotation::East;
        case tetris_v2::Rotation::South: return Rotation::South;
        case tetris_v2::Rotation::West: return Rotation::West;
    }
    return Rotation::North;
}

tetris_v2::Rotation rotation_to_env(Rotation r) {
    switch (r) {
        case Rotation::North: return tetris_v2::Rotation::North;
        case Rotation::East: return tetris_v2::Rotation::East;
        case Rotation::South: return tetris_v2::Rotation::South;
        case Rotation::West: return tetris_v2::Rotation::West;
    }
    return tetris_v2::Rotation::North;
}

Spin spin_from_env(tetris_v2::SpinType spin) {
    switch (spin) {
        case tetris_v2::SpinType::None: return Spin::None;
        case tetris_v2::SpinType::Mini: return Spin::Mini;
        case tetris_v2::SpinType::Full: return Spin::Full;
    }
    return Spin::None;
}

std::array<Vec2, 4> PieceLocation::cells() const {
    auto out = base_cells(piece);
    for (auto& c : out) {
        c = rotate_cell(rotation, c);
        c.x = static_cast<std::int8_t>(c.x + x);
        c.y = static_cast<std::int8_t>(c.y + y);
    }
    return out;
}

bool PieceLocation::obstructed(const Board& board) const {
    for (const auto& c : cells()) {
        if (board.occupied(c.x, c.y)) {
            return true;
        }
    }
    return false;
}

std::int8_t PieceLocation::drop_distance(const Board& board) const {
    std::int8_t out = 127;
    for (const auto& c : cells()) {
        out = std::min(out, board.distance_to_ground(c.x, c.y));
    }
    return out;
}

bool PieceLocation::above_stack(const Board& board) const {
    for (const auto& c : cells()) {
        if (c.x < 0 || c.x >= 10) {
            return false;
        }
        const int h = highest_filled_row_plus_one(board.cols[static_cast<std::size_t>(c.x)]);
        if (c.y < h) {
            return false;
        }
    }
    return true;
}

PieceLocation PieceLocation::canonical_form() const {
    switch (piece) {
        case Piece::T:
        case Piece::J:
        case Piece::L: return *this;
        case Piece::O:
            switch (rotation) {
                case Rotation::North: return *this;
                case Rotation::East:
                    return PieceLocation{piece, Rotation::North, x, static_cast<std::int8_t>(y - 1)};
                case Rotation::South:
                    return PieceLocation{
                        piece, Rotation::North, static_cast<std::int8_t>(x - 1), static_cast<std::int8_t>(y - 1)};
                case Rotation::West:
                    return PieceLocation{piece, Rotation::North, static_cast<std::int8_t>(x - 1), y};
            }
            break;
        case Piece::S:
        case Piece::Z:
            switch (rotation) {
                case Rotation::North:
                case Rotation::East: return *this;
                case Rotation::South:
                    return PieceLocation{piece, Rotation::North, x, static_cast<std::int8_t>(y - 1)};
                case Rotation::West:
                    return PieceLocation{piece, Rotation::East, static_cast<std::int8_t>(x - 1), y};
            }
            break;
        case Piece::I:
            switch (rotation) {
                case Rotation::North:
                case Rotation::East: return *this;
                case Rotation::South:
                    return PieceLocation{piece, Rotation::North, static_cast<std::int8_t>(x - 1), y};
                case Rotation::West:
                    return PieceLocation{piece, Rotation::East, x, static_cast<std::int8_t>(y + 1)};
            }
            break;
        case Piece::None: break;
    }
    return *this;
}

Board Board::from_env_board(const tetris_v2::Board& board) {
    Board out{};
    for (int y = 0; y < tetris_v2::Board::kRows; ++y) {
        const auto mask = board.row_mask(y);
        for (int x = 0; x < tetris_v2::Board::kWidth; ++x) {
            if ((mask & (1u << x)) != 0u) {
                out.cols[static_cast<std::size_t>(x)] |= (1ull << y);
            }
        }
    }
    return out;
}

bool Board::occupied(std::int8_t x, std::int8_t y) const {
    if (x < 0 || x >= 10 || y < 0 || y >= 40) {
        return true;
    }
    return (cols[static_cast<std::size_t>(x)] & (1ull << y)) != 0ull;
}

std::int8_t Board::distance_to_ground(std::int8_t x, std::int8_t y) const {
    if (x < 0 || x >= 10 || y <= 0) {
        return 0;
    }
    std::int8_t out = 0;
    while (y - out - 1 >= 0 && !occupied(x, static_cast<std::int8_t>(y - out - 1))) {
        ++out;
    }
    return out;
}

void Board::place(const PieceLocation& piece) {
    for (const auto& c : piece.cells()) {
        if (c.x < 0 || c.x >= 10 || c.y < 0 || c.y >= 40) {
            continue;
        }
        cols[static_cast<std::size_t>(c.x)] |= (1ull << c.y);
    }
}

std::uint64_t Board::line_clears() const {
    std::uint64_t out = ~0ull;
    for (auto c : cols) {
        out &= c;
    }
    return out;
}

void Board::remove_lines(std::uint64_t lines) {
    for (auto& c : cols) {
        clear_lines_column(&c, lines);
    }
}

bool Board::is_empty() const {
    for (auto c : cols) {
        if (c != 0) {
            return false;
        }
    }
    return true;
}

PlacementInfo GameState::advance(Piece next, const Placement& placement, bool use_hold) {
    auto consume_piece = [this](Piece piece) {
        if (piece == Piece::None) {
            return;
        }
        bag_mask = static_cast<PieceMask>(bag_mask & ~piece_bit(piece));
        if (bag_mask == 0) {
            bag_mask = kAllPiecesMask;
        }
    };

    consume_piece(next);
    if (use_hold) {
        if (reserve == Piece::None) {
            consume_piece(placement.location.piece);
        }
        reserve = next;
    }
    hold_available = true;

    board.place(placement.location);
    const std::uint64_t cleared_mask = board.line_clears();
    bool b2b_bonus = false;
    if (cleared_mask != 0) {
        board.remove_lines(cleared_mask);
        const bool hard =
            (bit_ops::popcount_u64(cleared_mask) == 4u) || (placement.spin != Spin::None);
        b2b_bonus = hard && back_to_back;
        back_to_back = hard;
        combo = static_cast<std::uint8_t>(std::min<unsigned int>(combo + 1u, 255u));
    } else {
        combo = 0;
    }

    PlacementInfo out{};
    out.placement = placement;
    out.lines_cleared = bit_ops::popcount_u64(cleared_mask);
    out.combo = combo;
    out.back_to_back = b2b_bonus;
    out.perfect_clear = board.is_empty();
    return out;
}

}  // namespace tetris_v2::cc2
