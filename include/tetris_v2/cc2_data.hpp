#pragma once

#include <array>
#include <cstdint>

#include "tetris_v2/bit_ops.hpp"
#include "tetris_v2/board.hpp"
#include "tetris_v2/types.hpp"

namespace tetris_v2::cc2 {

struct Board;

enum class Rotation : std::uint8_t {
    North = 0,
    West = 1,
    South = 2,
    East = 3,
};

enum class Spin : std::uint8_t {
    None = 0,
    Mini = 1,
    Full = 2,
};

struct Vec2 {
    std::int8_t x{0};
    std::int8_t y{0};
};

struct PieceLocation {
    Piece piece{Piece::None};
    Rotation rotation{Rotation::North};
    std::int8_t x{0};
    std::int8_t y{0};

    bool operator==(const PieceLocation& rhs) const {
        return piece == rhs.piece && rotation == rhs.rotation && x == rhs.x && y == rhs.y;
    }

    std::array<Vec2, 4> cells() const;
    bool obstructed(const Board& board) const;
    std::int8_t drop_distance(const Board& board) const;
    bool above_stack(const Board& board) const;
    PieceLocation canonical_form() const;
};

struct Placement {
    PieceLocation location{};
    Spin spin{Spin::None};

    bool operator==(const Placement& rhs) const {
        return location == rhs.location && spin == rhs.spin;
    }
};

struct PlacementInfo {
    Placement placement{};
    std::uint32_t lines_cleared{0};
    std::uint32_t combo{0};
    bool back_to_back{false};
    bool perfect_clear{false};
};

struct Board {
    std::array<std::uint64_t, 10> cols{};

    static Board from_env_board(const tetris_v2::Board& board);

    bool occupied(std::int8_t x, std::int8_t y) const;
    std::int8_t distance_to_ground(std::int8_t x, std::int8_t y) const;
    void place(const PieceLocation& piece);
    std::uint64_t line_clears() const;
    void remove_lines(std::uint64_t lines);
    bool is_empty() const;
};

using PieceMask = std::uint8_t;
constexpr PieceMask kAllPiecesMask = static_cast<PieceMask>(0x7F);

constexpr PieceMask piece_bit(Piece p) {
    return static_cast<PieceMask>(1u << static_cast<std::uint8_t>(p));
}

inline bool piece_mask_contains(PieceMask mask, Piece p) {
    if (p == Piece::None) {
        return false;
    }
    return (mask & piece_bit(p)) != 0;
}

inline std::uint32_t piece_mask_count(PieceMask mask) {
    return bit_ops::popcount_u64(mask);
}

struct GameState {
    Board board{};
    PieceMask bag_mask{kAllPiecesMask};
    Piece reserve{Piece::None};
    bool hold_available{true};
    bool back_to_back{false};
    std::uint8_t combo{0};

    PlacementInfo advance(Piece next, const Placement& placement, bool use_hold = false);
};

Rotation rotate_cw(Rotation r);
Rotation rotate_ccw(Rotation r);
Rotation rotate_flip(Rotation r);
Vec2 rotate_cell(Rotation r, Vec2 v);

Rotation rotation_from_env(tetris_v2::Rotation r);
tetris_v2::Rotation rotation_to_env(Rotation r);
Spin spin_from_env(tetris_v2::SpinType spin);

}  // namespace tetris_v2::cc2
