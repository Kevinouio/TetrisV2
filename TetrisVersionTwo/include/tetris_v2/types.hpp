#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace tetris_v2 {

enum class Piece : std::uint8_t {
    I = 0,
    O = 1,
    T = 2,
    L = 3,
    J = 4,
    S = 5,
    Z = 6,
    None = 7,
};

enum class Rotation : std::uint8_t {
    North = 0,
    East = 1,
    South = 2,
    West = 3,
};

enum class Action : std::uint8_t {
    None = 0,
    Left = 1,
    Right = 2,
    SoftDrop = 3,
    HardDrop = 4,
    RotateCW = 5,
    RotateCCW = 6,
    Rotate180 = 7,
    Hold = 8,
};

enum class SpinType : std::uint8_t {
    None = 0,
    Mini = 1,
    Full = 2,
};

struct Cell {
    int x{0};
    int y{0};
};

struct ActivePiece {
    Piece piece{Piece::None};
    Rotation rotation{Rotation::North};
    int x{0};
    int y{0};

    bool operator==(const ActivePiece& rhs) const {
        return piece == rhs.piece && rotation == rhs.rotation && x == rhs.x && y == rhs.y;
    }
};

struct StepResult {
    bool game_over{false};
    bool top_out{false};
    bool piece_locked{false};
    bool hold_used{false};
    bool action_succeeded{false};
    int lines_cleared{0};
    bool spin_clear{false};
    SpinType spin_type{SpinType::None};
    bool difficult_clear{false};
    bool b2b_bonus_applied{false};
    int combo{-1};
    bool back_to_back{false};
    float reward{0.0f};
};

inline constexpr std::array<Piece, 7> kPlayablePieces{
    Piece::I, Piece::O, Piece::T, Piece::L, Piece::J, Piece::S, Piece::Z};

inline constexpr Rotation rotate_cw(Rotation r) {
    return static_cast<Rotation>((static_cast<std::uint8_t>(r) + 1u) & 3u);
}

inline constexpr Rotation rotate_ccw(Rotation r) {
    return static_cast<Rotation>((static_cast<std::uint8_t>(r) + 3u) & 3u);
}

inline constexpr Rotation rotate_180(Rotation r) {
    return static_cast<Rotation>((static_cast<std::uint8_t>(r) + 2u) & 3u);
}

inline constexpr std::size_t piece_index(Piece p) {
    return static_cast<std::size_t>(p);
}

inline constexpr char piece_to_char(Piece p) {
    switch (p) {
        case Piece::I: return 'I';
        case Piece::O: return 'O';
        case Piece::T: return 'T';
        case Piece::L: return 'L';
        case Piece::J: return 'J';
        case Piece::S: return 'S';
        case Piece::Z: return 'Z';
        case Piece::None: return '.';
    }
    return '?';
}

inline constexpr const char* piece_name(Piece p) {
    switch (p) {
        case Piece::I: return "I";
        case Piece::O: return "O";
        case Piece::T: return "T";
        case Piece::L: return "L";
        case Piece::J: return "J";
        case Piece::S: return "S";
        case Piece::Z: return "Z";
        case Piece::None: return "None";
    }
    return "Unknown";
}

}  // namespace tetris_v2
