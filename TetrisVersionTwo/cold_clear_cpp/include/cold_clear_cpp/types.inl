#pragma once

#include <cstddef>

namespace cold_clear_cpp {

constexpr std::pair<std::int8_t, std::int8_t> rotate_cell(
    Rotation r, std::pair<std::int8_t, std::int8_t> cell) {
    switch (r) {
        case Rotation::North:
            return cell;
        case Rotation::East:
            return {cell.second, static_cast<std::int8_t>(-cell.first)};
        case Rotation::South:
            return {static_cast<std::int8_t>(-cell.first),
                    static_cast<std::int8_t>(-cell.second)};
        case Rotation::West:
            return {static_cast<std::int8_t>(-cell.second), cell.first};
    }
    return cell;
}

constexpr std::array<std::pair<std::int8_t, std::int8_t>, 4> rotate_cells(
    Rotation r, std::array<std::pair<std::int8_t, std::int8_t>, 4> cells) {
    return {rotate_cell(r, cells[0]), rotate_cell(r, cells[1]),
            rotate_cell(r, cells[2]), rotate_cell(r, cells[3])};
}

constexpr Rotation cw(Rotation r) {
    switch (r) {
        case Rotation::North:
            return Rotation::East;
        case Rotation::East:
            return Rotation::South;
        case Rotation::South:
            return Rotation::West;
        case Rotation::West:
            return Rotation::North;
    }
    return r;
}

constexpr Rotation ccw(Rotation r) {
    switch (r) {
        case Rotation::North:
            return Rotation::West;
        case Rotation::West:
            return Rotation::South;
        case Rotation::South:
            return Rotation::East;
        case Rotation::East:
            return Rotation::North;
    }
    return r;
}

constexpr std::array<std::pair<std::int8_t, std::int8_t>, 4> base_cells(Piece p) {
    switch (p) {
        case Piece::I:
            return {{ {-1, 0}, {0, 0}, {1, 0}, {2, 0} }};
        case Piece::O:
            return {{ {0, 0}, {1, 0}, {0, 1}, {1, 1} }};
        case Piece::T:
            return {{ {-1, 0}, {0, 0}, {1, 0}, {0, 1} }};
        case Piece::L:
            return {{ {-1, 0}, {0, 0}, {1, 0}, {1, 1} }};
        case Piece::J:
            return {{ {-1, 0}, {0, 0}, {1, 0}, {-1, 1} }};
        case Piece::S:
            return {{ {-1, 0}, {0, 0}, {0, 1}, {1, 1} }};
        case Piece::Z:
            return {{ {-1, 1}, {0, 1}, {0, 0}, {1, 0} }};
    }
    return {};
}

inline std::array<std::pair<std::int8_t, std::int8_t>, 4> PieceLocation::cells() const {
    auto translate = [this](std::pair<std::int8_t, std::int8_t> cell) {
        return std::make_pair<std::int8_t, std::int8_t>(cell.first + x, cell.second + y);
    };

    auto rotated = rotate_cells(rotation, base_cells(piece));
    return {translate(rotated[0]), translate(rotated[1]), translate(rotated[2]),
            translate(rotated[3])};
}

inline PieceLocation PieceLocation::canonical_form() const {
    switch (piece) {
        case Piece::T:
        case Piece::J:
        case Piece::L:
            return *this;
        case Piece::O: {
            switch (rotation) {
                case Rotation::North:
                    return *this;
                case Rotation::East:
                    return PieceLocation{piece, Rotation::North, static_cast<std::int8_t>(x),
                                         static_cast<std::int8_t>(y - 1)};
                case Rotation::South:
                    return PieceLocation{piece, Rotation::North,
                                         static_cast<std::int8_t>(x - 1),
                                         static_cast<std::int8_t>(y - 1)};
                case Rotation::West:
                    return PieceLocation{piece, Rotation::North,
                                         static_cast<std::int8_t>(x - 1), y};
            }
            break;
        }
        case Piece::S:
        case Piece::Z: {
            switch (rotation) {
                case Rotation::North:
                case Rotation::East:
                    return *this;
                case Rotation::South:
                    return PieceLocation{piece, Rotation::North, x,
                                         static_cast<std::int8_t>(y - 1)};
                case Rotation::West:
                    return PieceLocation{piece, Rotation::East,
                                         static_cast<std::int8_t>(x - 1), y};
            }
            break;
        }
        case Piece::I: {
            switch (rotation) {
                case Rotation::North:
                case Rotation::East:
                    return *this;
                case Rotation::South:
                    return PieceLocation{piece, Rotation::North,
                                         static_cast<std::int8_t>(x - 1), y};
                case Rotation::West:
                    return PieceLocation{piece, Rotation::East, x,
                                         static_cast<std::int8_t>(y + 1)};
            }
            break;
        }
    }
    return *this;
}

inline std::size_t PlacementHash::operator()(const Placement& p) const noexcept {
    std::size_t h = 1469598103934665603ull;
    h ^= static_cast<std::size_t>(p.location.piece) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h ^= static_cast<std::size_t>(p.location.rotation) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h ^= static_cast<std::size_t>(p.location.x) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h ^= static_cast<std::size_t>(p.location.y) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h ^= static_cast<std::size_t>(p.spin) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h;
}

inline std::size_t PieceLocationHash::operator()(const PieceLocation& p) const noexcept {
    std::size_t h = 1469598103934665603ull;
    h ^= static_cast<std::size_t>(p.piece) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h ^= static_cast<std::size_t>(p.rotation) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h ^= static_cast<std::size_t>(p.x) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h ^= static_cast<std::size_t>(p.y) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h;
}

inline PieceSet all_pieces() {
    PieceSet set;
    for (auto p : kAllPieces) {
        set.set(piece_index(p));
    }
    return set;
}

}  // namespace cold_clear_cpp
