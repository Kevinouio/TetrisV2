#pragma once

#include <array>
#include <bitset>
#include <cstdint>
#include <functional>
#include <utility>

namespace cold_clear_cpp {

enum class Piece : std::uint8_t {
    I,
    O,
    T,
    L,
    J,
    S,
    Z,
};

enum class Rotation : std::uint8_t {
    North,
    West,
    South,
    East,
};

enum class Spin : std::uint8_t {
    None,
    Mini,
    Full,
};

using PieceSet = std::bitset<7>;

inline constexpr std::array<Piece, 7> kAllPieces{
    Piece::I, Piece::O, Piece::T, Piece::L, Piece::J, Piece::S, Piece::Z};

inline constexpr std::array<Rotation, 4> kAllRotations{
    Rotation::North, Rotation::West, Rotation::South, Rotation::East};

constexpr std::size_t piece_index(Piece p) {
    return static_cast<std::size_t>(p);
}

constexpr std::size_t rotation_index(Rotation r) {
    return static_cast<std::size_t>(r);
}

struct PieceLocation {
    Piece piece{};
    Rotation rotation{Rotation::North};
    std::int8_t x{0};
    std::int8_t y{0};

    std::array<std::pair<std::int8_t, std::int8_t>, 4> cells() const;
    PieceLocation canonical_form() const;
    bool obstructed(const struct Board& board) const;
    std::int8_t drop_distance(const struct Board& board) const;
    bool above_stack(const struct Board& board) const;
    bool operator==(const PieceLocation& rhs) const {
        return piece == rhs.piece && rotation == rhs.rotation && x == rhs.x && y == rhs.y;
    }
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

struct PlacementHash {
    std::size_t operator()(const Placement& p) const noexcept;
};

struct PieceLocationHash {
    std::size_t operator()(const PieceLocation& p) const noexcept;
};

PieceSet all_pieces();

}  // namespace cold_clear_cpp

#include "cold_clear_cpp/types.inl"
