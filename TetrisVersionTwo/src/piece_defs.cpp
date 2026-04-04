#include "tetris_v2/piece_defs.hpp"

#include <cstddef>

namespace tetris_v2 {

namespace {

constexpr Cell rotate_cell(Rotation rotation, Cell c) {
    switch (rotation) {
        case Rotation::North:
            return c;
        case Rotation::East:
            return Cell{c.y, -c.x};
        case Rotation::South:
            return Cell{-c.x, -c.y};
        case Rotation::West:
            return Cell{-c.y, c.x};
    }
    return c;
}

constexpr std::array<Cell, 4> base_cells(Piece piece) {
    switch (piece) {
        case Piece::I:
            return {Cell{-1, 0}, Cell{0, 0}, Cell{1, 0}, Cell{2, 0}};
        case Piece::O:
            return {Cell{0, 0}, Cell{1, 0}, Cell{0, 1}, Cell{1, 1}};
        case Piece::T:
            return {Cell{-1, 0}, Cell{0, 0}, Cell{1, 0}, Cell{0, 1}};
        case Piece::L:
            return {Cell{-1, 0}, Cell{0, 0}, Cell{1, 0}, Cell{1, 1}};
        case Piece::J:
            return {Cell{-1, 0}, Cell{0, 0}, Cell{1, 0}, Cell{-1, 1}};
        case Piece::S:
            return {Cell{-1, 0}, Cell{0, 0}, Cell{0, 1}, Cell{1, 1}};
        case Piece::Z:
            return {Cell{-1, 1}, Cell{0, 1}, Cell{0, 0}, Cell{1, 0}};
        case Piece::None:
            return {Cell{0, 0}, Cell{0, 0}, Cell{0, 0}, Cell{0, 0}};
    }
    return {Cell{0, 0}, Cell{0, 0}, Cell{0, 0}, Cell{0, 0}};
}

constexpr std::array<std::array<std::array<Cell, 4>, 4>, 8> build_piece_cell_table() {
    std::array<std::array<std::array<Cell, 4>, 4>, 8> table{};
    for (std::size_t p = 0; p < table.size(); ++p) {
        const auto piece = static_cast<Piece>(p);
        const auto base = base_cells(piece);
        for (std::size_t r = 0; r < table[p].size(); ++r) {
            const auto rotation = static_cast<Rotation>(r);
            auto rotated = base;
            for (auto& c : rotated) {
                c = rotate_cell(rotation, c);
            }
            table[p][r] = rotated;
        }
    }
    return table;
}

constexpr auto kPieceCellTable = build_piece_cell_table();

}  // namespace

std::array<Cell, 4> piece_cells(Piece piece, Rotation rotation) {
    return kPieceCellTable[static_cast<std::size_t>(piece)][static_cast<std::size_t>(rotation)];
}

ActivePiece spawn_piece(Piece piece) {
    // Spawn position intentionally simple for scaffold: centered, just above visible stack.
    return ActivePiece{piece, Rotation::North, 4, 21};
}

}  // namespace tetris_v2
