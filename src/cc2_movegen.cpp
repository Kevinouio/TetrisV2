#include "tetris_v2/cc2_movegen.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <queue>
#include <unordered_map>

#include "tetris_v2/bit_ops.hpp"
#include "tetris_v2/piece_defs.hpp"

namespace tetris_v2::cc2 {

namespace {

struct PlacementHash {
    std::size_t operator()(const Placement& p) const noexcept {
        std::size_t h = 1469598103934665603ull;
        auto mix = [&h](std::size_t v) {
            h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        };
        mix(static_cast<std::size_t>(p.location.piece));
        mix(static_cast<std::size_t>(p.location.rotation));
        mix(static_cast<std::size_t>(static_cast<int>(p.location.x) + 128));
        mix(static_cast<std::size_t>(static_cast<int>(p.location.y) + 128));
        mix(static_cast<std::size_t>(p.spin));
        return h;
    }
};

struct Intermediate {
    Placement mv{};
    std::uint32_t soft_drops{0};

    bool operator<(const Intermediate& rhs) const { return soft_drops < rhs.soft_drops; }
};

using KickOffsets = std::array<Vec2, 5>;

KickOffsets offsets(Piece piece, Rotation rotation) {
    switch (piece) {
        case Piece::O:
            switch (rotation) {
                case Rotation::North: return {Vec2{0, 0}, Vec2{0, 0}, Vec2{0, 0}, Vec2{0, 0}, Vec2{0, 0}};
                case Rotation::East: return {Vec2{0, -1}, Vec2{0, -1}, Vec2{0, -1}, Vec2{0, -1}, Vec2{0, -1}};
                case Rotation::South:
                    return {Vec2{-1, -1}, Vec2{-1, -1}, Vec2{-1, -1}, Vec2{-1, -1}, Vec2{-1, -1}};
                case Rotation::West: return {Vec2{-1, 0}, Vec2{-1, 0}, Vec2{-1, 0}, Vec2{-1, 0}, Vec2{-1, 0}};
            }
            break;
        case Piece::I:
            switch (rotation) {
                case Rotation::North:
                    return {Vec2{0, 0}, Vec2{-1, 0}, Vec2{2, 0}, Vec2{-1, 0}, Vec2{2, 0}};
                case Rotation::East:
                    return {Vec2{-1, 0}, Vec2{0, 0}, Vec2{0, 0}, Vec2{0, 1}, Vec2{0, -2}};
                case Rotation::South:
                    return {Vec2{-1, 1}, Vec2{1, 1}, Vec2{-2, 1}, Vec2{1, 0}, Vec2{-2, 0}};
                case Rotation::West:
                    return {Vec2{0, 1}, Vec2{0, 1}, Vec2{0, 1}, Vec2{0, -1}, Vec2{0, 2}};
            }
            break;
        case Piece::None:
            break;
        default:
            switch (rotation) {
                case Rotation::North: return {Vec2{0, 0}, Vec2{0, 0}, Vec2{0, 0}, Vec2{0, 0}, Vec2{0, 0}};
                case Rotation::East:
                    return {Vec2{0, 0}, Vec2{1, 0}, Vec2{1, -1}, Vec2{0, 2}, Vec2{1, 2}};
                case Rotation::South: return {Vec2{0, 0}, Vec2{0, 0}, Vec2{0, 0}, Vec2{0, 0}, Vec2{0, 0}};
                case Rotation::West:
                    return {Vec2{0, 0}, Vec2{-1, 0}, Vec2{-1, -1}, Vec2{0, 2}, Vec2{-1, 2}};
            }
            break;
    }
    return {Vec2{0, 0}, Vec2{0, 0}, Vec2{0, 0}, Vec2{0, 0}, Vec2{0, 0}};
}

KickOffsets kicks(Piece piece, Rotation from, Rotation to) {
    auto from_offsets = offsets(piece, from);
    auto to_offsets = offsets(piece, to);
    KickOffsets out{};
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = Vec2{
            static_cast<std::int8_t>(from_offsets[i].x - to_offsets[i].x),
            static_cast<std::int8_t>(from_offsets[i].y - to_offsets[i].y)};
    }
    return out;
}

class CollisionMaps {
public:
    CollisionMaps(const Board& board, Piece piece) {
        constexpr std::array<Rotation, 4> kRotations{
            Rotation::North, Rotation::West, Rotation::South, Rotation::East};

        for (const auto rotation : kRotations) {
            const auto idx = static_cast<std::size_t>(rotation);
            PieceLocation rotated{piece, rotation, 0, 0};
            for (const auto& cell : rotated.cells()) {
                for (int x = 0; x < 10; ++x) {
                    const int src = x + cell.x;
                    std::uint64_t c =
                        (src >= 0 && src < 10) ? board.cols[static_cast<std::size_t>(src)] : ~0ull;
                    if (cell.y < 0) {
                        const unsigned int shift = static_cast<unsigned int>(-cell.y);
                        c = shift >= 64u ? ~0ull : ~((~c) << shift);
                    } else {
                        const unsigned int shift = static_cast<unsigned int>(cell.y);
                        c = shift >= 64u ? 0ull : (c >> shift);
                    }
                    boards_[idx][static_cast<std::size_t>(x)] |= c;
                }
            }
        }
    }

    bool obstructed(PieceLocation piece) const {
        if (piece.y < 0 || piece.y >= 64) {
            return true;
        }
        if (piece.x < 0 || piece.x >= 10) {
            return true;
        }
        const auto idx = static_cast<std::size_t>(piece.rotation);
        const auto col = static_cast<std::size_t>(piece.x);
        return (boards_[idx][col] & (1ull << static_cast<unsigned int>(piece.y))) != 0ull;
    }

private:
    std::array<std::array<std::uint64_t, 10>, 4> boards_{};
};

std::optional<Placement> shift(PieceLocation location, const CollisionMaps& collision_map, std::int8_t dx) {
    location.x = static_cast<std::int8_t>(location.x + dx);
    if (collision_map.obstructed(location)) {
        return std::nullopt;
    }
    return Placement{location, Spin::None};
}

std::optional<Placement> rotate(
    PieceLocation unkicked,
    const CollisionMaps& collision_map,
    const Board& board,
    const KickOffsets& kick_tests) {
    for (std::size_t i = 0; i < kick_tests.size(); ++i) {
        const auto& kick = kick_tests[i];
        PieceLocation target{
            unkicked.piece,
            unkicked.rotation,
            static_cast<std::int8_t>(unkicked.x + kick.x),
            static_cast<std::int8_t>(unkicked.y + kick.y)};

        if (collision_map.obstructed(target)) {
            continue;
        }

        Spin spin = Spin::None;
        if (target.piece == Piece::T) {
            int corners = 0;
            corners += board.occupied(static_cast<std::int8_t>(target.x - 1), static_cast<std::int8_t>(target.y - 1))
                ? 1
                : 0;
            corners += board.occupied(static_cast<std::int8_t>(target.x + 1), static_cast<std::int8_t>(target.y - 1))
                ? 1
                : 0;
            corners += board.occupied(static_cast<std::int8_t>(target.x - 1), static_cast<std::int8_t>(target.y + 1))
                ? 1
                : 0;
            corners += board.occupied(static_cast<std::int8_t>(target.x + 1), static_cast<std::int8_t>(target.y + 1))
                ? 1
                : 0;

            Vec2 a = rotate_cell(target.rotation, Vec2{-1, 1});
            Vec2 b = rotate_cell(target.rotation, Vec2{1, 1});
            int mini_corners = 0;
            mini_corners += board.occupied(
                                static_cast<std::int8_t>(target.x + a.x),
                                static_cast<std::int8_t>(target.y + a.y))
                ? 1
                : 0;
            mini_corners += board.occupied(
                                static_cast<std::int8_t>(target.x + b.x),
                                static_cast<std::int8_t>(target.y + b.y))
                ? 1
                : 0;

            if (corners < 3) {
                spin = Spin::None;
            } else if (mini_corners == 2 || i == 4) {
                spin = Spin::Full;
            } else {
                spin = Spin::Mini;
            }
        }

        return Placement{target, spin};
    }

    return std::nullopt;
}

std::optional<Placement> rotate_piece_cw(
    PieceLocation from,
    const CollisionMaps& collision_map,
    const Board& board) {
    if (from.piece == Piece::O) {
        return std::nullopt;
    }
    PieceLocation unkicked{
        from.piece, rotate_cw(from.rotation), from.x, from.y};
    auto kick_table = kicks(from.piece, from.rotation, unkicked.rotation);
    return rotate(unkicked, collision_map, board, kick_table);
}

std::optional<Placement> rotate_piece_ccw(
    PieceLocation from,
    const CollisionMaps& collision_map,
    const Board& board) {
    if (from.piece == Piece::O) {
        return std::nullopt;
    }
    PieceLocation unkicked{
        from.piece, rotate_ccw(from.rotation), from.x, from.y};
    auto kick_table = kicks(from.piece, from.rotation, unkicked.rotation);
    return rotate(unkicked, collision_map, board, kick_table);
}

}  // namespace

std::vector<std::pair<Placement, std::uint32_t>> find_moves(const Board& board, Piece piece) {
    if (piece == Piece::None) {
        return {};
    }

    std::priority_queue<Intermediate> queue;
    std::unordered_map<Placement, std::uint32_t, PlacementHash> values;
    std::unordered_map<Placement, std::uint32_t, PlacementHash> underground_locks;
    std::vector<std::pair<Placement, std::uint32_t>> locks;
    locks.reserve(64);

    const CollisionMaps collision_map(board, piece);
    const auto spawn = tetris_v2::spawn_piece(piece);
    const auto spawn_y = static_cast<std::int8_t>(spawn.y);
    const bool fast_mode = std::all_of(board.cols.begin(), board.cols.end(), [](std::uint64_t c) {
        return bit_ops::countl_zero_u64(c) > (64u - 16u);
    });

    auto update_position = [&](const Placement& target, std::uint32_t soft_drops) {
        if (fast_mode && target.location.above_stack(board)) {
            return;
        }
        auto it = values.find(target);
        if (it == values.end() || soft_drops < it->second) {
            values[target] = soft_drops;
            queue.push(Intermediate{target, soft_drops});
        }
    };

    if (fast_mode) {
        constexpr std::array<Rotation, 4> kRotations{
            Rotation::North, Rotation::East, Rotation::South, Rotation::West};
        for (const auto rotation : kRotations) {
            for (std::int8_t x = 0; x < 10; ++x) {
                PieceLocation location{piece, rotation, x, spawn_y};
                if (collision_map.obstructed(location)) {
                    continue;
                }
                const auto distance = location.drop_distance(board);
                location.y = static_cast<std::int8_t>(location.y - distance);
                const Placement mv{location, Spin::None};

                if (auto shifted = shift(location, collision_map, -1)) {
                    update_position(*shifted, static_cast<std::uint32_t>(distance));
                }
                if (auto shifted = shift(location, collision_map, 1)) {
                    update_position(*shifted, static_cast<std::uint32_t>(distance));
                }
                if (auto rotated = rotate_piece_cw(location, collision_map, board)) {
                    update_position(*rotated, static_cast<std::uint32_t>(distance));
                }
                if (auto rotated = rotate_piece_ccw(location, collision_map, board)) {
                    update_position(*rotated, static_cast<std::uint32_t>(distance));
                }

                if (location.canonical_form() == location) {
                    locks.emplace_back(mv, 0u);
                }
            }
        }
    } else {
        PieceLocation spawned{
            piece,
            rotation_from_env(spawn.rotation),
            static_cast<std::int8_t>(spawn.x),
            spawn_y};
        if (collision_map.obstructed(spawned)) {
            return {};
        }
        Placement start{spawned, Spin::None};
        queue.push(Intermediate{start, 0});
        values[start] = 0;
    }

    while (!queue.empty()) {
        const Intermediate expand = queue.top();
        queue.pop();

        auto it = values.find(expand.mv);
        if (it == values.end() || it->second != expand.soft_drops) {
            continue;
        }

        const auto drop_dist = expand.mv.location.drop_distance(board);
        Placement dropped{
            PieceLocation{
                expand.mv.location.piece,
                expand.mv.location.rotation,
                expand.mv.location.x,
                static_cast<std::int8_t>(expand.mv.location.y - drop_dist)},
            drop_dist == 0 ? expand.mv.spin : Spin::None};

        Placement canonical = dropped;
        canonical.location = canonical.location.canonical_form();
        auto lock_it = underground_locks.find(canonical);
        if (lock_it == underground_locks.end()) {
            underground_locks.emplace(canonical, expand.soft_drops);
        } else {
            lock_it->second = std::min(lock_it->second, expand.soft_drops);
        }

        update_position(dropped, expand.soft_drops + static_cast<std::uint32_t>(drop_dist));

        if (auto shifted = shift(expand.mv.location, collision_map, -1)) {
            update_position(*shifted, expand.soft_drops);
        }
        if (auto shifted = shift(expand.mv.location, collision_map, 1)) {
            update_position(*shifted, expand.soft_drops);
        }
        if (auto rotated = rotate_piece_cw(expand.mv.location, collision_map, board)) {
            update_position(*rotated, expand.soft_drops);
        }
        if (auto rotated = rotate_piece_ccw(expand.mv.location, collision_map, board)) {
            update_position(*rotated, expand.soft_drops);
        }
    }

    locks.reserve(locks.size() + underground_locks.size());
    for (const auto& entry : underground_locks) {
        locks.push_back(entry);
    }
    return locks;
}

}  // namespace tetris_v2::cc2
