#include "cold_clear_cpp/movegen.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <queue>
#include <unordered_map>

namespace cold_clear_cpp {

namespace {

struct Intermediate {
    Placement mv;
    std::uint32_t soft_drops{0};

    bool operator<(const Intermediate& rhs) const { return soft_drops < rhs.soft_drops; }
};

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

struct CollisionMaps {
    std::array<std::array<std::uint64_t, 10>, 4> boards{};

    explicit CollisionMaps(const Board& board, Piece piece) {
        for (auto rot : kAllRotations) {
            auto rotated_cells = rotate_cells(rot, base_cells(piece));
            for (auto [dx, dy] : rotated_cells) {
                for (int x = 0; x < 10; ++x) {
                    auto idx = x + dx;
                    std::uint64_t c = (idx >= 0 && idx < 10) ? board.cols[static_cast<std::size_t>(idx)]
                                                            : ~0ull;
                    if (dy < 0) {
                        c = ~(~c << (-dy));
                    } else {
                        c >>= dy;
                    }
                    boards[rotation_index(rot)][static_cast<std::size_t>(x)] |= c;
                }
            }
        }
    }

    bool obstructed(const PieceLocation& piece) const {
        if (piece.y < 0) {
            return true;
        }
        auto rot_idx = rotation_index(piece.rotation);
        if (piece.x < 0 || piece.x >= 10) {
            return true;
        }
        auto c = boards[rot_idx][static_cast<std::size_t>(piece.x)];
        return (c & (1ull << piece.y)) != 0;
    }
};

std::optional<Placement> shift(PieceLocation location, const CollisionMaps& collision_map, int dx) {
    location.x = static_cast<std::int8_t>(location.x + dx);
    if (collision_map.obstructed(location)) {
        return std::nullopt;
    }
    return Placement{location, Spin::None};
}

constexpr std::array<std::pair<std::int8_t, std::int8_t>, 5> offsets(Piece piece, Rotation rot) {
    switch (piece) {
        case Piece::O:
            switch (rot) {
                case Rotation::North:
                    return { { {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0} } };
                case Rotation::East:
                    return { { {0, -1}, {0, -1}, {0, -1}, {0, -1}, {0, -1} } };
                case Rotation::South:
                    return { { {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1} } };
                case Rotation::West:
                    return { { {-1, 0}, {-1, 0}, {-1, 0}, {-1, 0}, {-1, 0} } };
            }
            break;
        case Piece::I:
            switch (rot) {
                case Rotation::North:
                    return { { {0, 0}, {-1, 0}, {2, 0}, {-1, 0}, {2, 0} } };
                case Rotation::East:
                    return { { {-1, 0}, {0, 0}, {0, 0}, {0, 1}, {0, -2} } };
                case Rotation::South:
                    return { { {-1, 1}, {1, 1}, {-2, 1}, {1, 0}, {-2, 0} } };
                case Rotation::West:
                    return { { {0, 1}, {0, 1}, {0, 1}, {0, -1}, {0, 2} } };
            }
            break;
        default:
            switch (rot) {
                case Rotation::North:
                    return { { {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0} } };
                case Rotation::East:
                    return { { {0, 0}, {1, 0}, {1, -1}, {0, 2}, {1, 2} } };
                case Rotation::South:
                    return { { {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0} } };
                case Rotation::West:
                    return { { {0, 0}, {-1, 0}, {-1, -1}, {0, 2}, {-1, 2} } };
            }
            break;
    }
    return {};
}

constexpr std::array<std::pair<std::int8_t, std::int8_t>, 5> kicks(
    Piece piece, Rotation from, Rotation to) {
    std::array<std::pair<std::int8_t, std::int8_t>, 5> res{};
    auto from_offsets = offsets(piece, from);
    auto to_offsets = offsets(piece, to);
    for (std::size_t i = 0; i < res.size(); ++i) {
        res[i] = {static_cast<std::int8_t>(from_offsets[i].first - to_offsets[i].first),
                  static_cast<std::int8_t>(from_offsets[i].second - to_offsets[i].second)};
    }
    return res;
}

std::optional<Placement> rotate(
    const PieceLocation& unkicked,
    const CollisionMaps& collision_map,
    const Board& board,
    const std::array<std::pair<std::int8_t, std::int8_t>, 5>& kicks,
    bool cw_rotation) {
    for (std::size_t i = 0; i < kicks.size(); ++i) {
        auto [dx, dy] = kicks[i];
        PieceLocation target = unkicked;
        target.x = static_cast<std::int8_t>(target.x + dx);
        target.y = static_cast<std::int8_t>(target.y + dy);
        if (collision_map.obstructed(target)) {
            continue;
        }

        Spin spin = Spin::None;
        if (target.piece == Piece::T) {
            int corners = 0;
            for (auto [cx, cy] : {std::pair{-1, -1}, std::pair{1, -1}, std::pair{-1, 1}, std::pair{1, 1}}) {
                if (board.occupied({static_cast<std::int8_t>(cx + target.x),
                                    static_cast<std::int8_t>(cy + target.y)})) {
                    ++corners;
                }
            }

            int mini_corners = 0;
            for (auto [cx, cy] : {std::pair{-1, 1}, std::pair{1, 1}}) {
                auto rotated = rotate_cell(target.rotation, {cx, cy});
                if (board.occupied({static_cast<std::int8_t>(rotated.first + target.x),
                                    static_cast<std::int8_t>(rotated.second + target.y)})) {
                    ++mini_corners;
                }
            }

            if (corners >= 3) {
                if (mini_corners == 2 || i == 4) {
                    spin = Spin::Full;
                } else {
                    spin = Spin::Mini;
                }
            }
        }

        return Placement{target, spin};
    }
    return std::nullopt;
}

std::optional<Placement> rotate_cw(
    const PieceLocation& from, const CollisionMaps& collision_map, const Board& board) {
    if (from.piece == Piece::O) {
        return std::nullopt;
    }
    auto unkicked = from;
    unkicked.rotation = cw(from.rotation);
    static const auto lut = [] {
        std::array<std::array<std::array<std::pair<std::int8_t, std::int8_t>, 5>, 4>, 7> res{};
        for (auto p : kAllPieces) {
            for (auto r : kAllRotations) {
                res[piece_index(p)][rotation_index(r)] = kicks(p, r, cw(r));
            }
        }
        return res;
    }();
    return rotate(
        unkicked, collision_map, board, lut[piece_index(from.piece)][rotation_index(from.rotation)],
        true);
}

std::optional<Placement> rotate_ccw(
    const PieceLocation& from, const CollisionMaps& collision_map, const Board& board) {
    if (from.piece == Piece::O) {
        return std::nullopt;
    }
    auto unkicked = from;
    unkicked.rotation = ccw(from.rotation);
    static const auto lut = [] {
        std::array<std::array<std::array<std::pair<std::int8_t, std::int8_t>, 5>, 4>, 7> res{};
        for (auto p : kAllPieces) {
            for (auto r : kAllRotations) {
                res[piece_index(p)][rotation_index(r)] = kicks(p, r, ccw(r));
            }
        }
        return res;
    }();
    return rotate(
        unkicked, collision_map, board, lut[piece_index(from.piece)][rotation_index(from.rotation)],
        false);
}

}  // namespace

std::vector<std::pair<Placement, std::uint32_t>> find_moves(const Board& board, Piece piece) {
    std::priority_queue<Intermediate> queue;
    std::unordered_map<Placement, std::uint32_t, PlacementHash> values;
    std::unordered_map<Placement, std::uint32_t, PlacementHash> underground_locks;
    std::vector<std::pair<Placement, std::uint32_t>> locks;
    CollisionMaps collision_map(board, piece);

    bool fast_mode = std::all_of(
        board.cols.begin(), board.cols.end(), [](std::uint64_t c) { return column_height(c) < 16; });

    if (fast_mode) {
        for (auto rotation : kAllRotations) {
            for (int x = 0; x < 10; ++x) {
                PieceLocation location{piece, rotation, static_cast<std::int8_t>(x), 19};
                if (collision_map.obstructed(location)) {
                    continue;
                }
                auto distance = location.drop_distance(board);
                location.y = static_cast<std::int8_t>(location.y - distance);
                Placement mv{location, Spin::None};

                auto update_position = [&](const Placement& target, std::uint32_t soft_drops) {
                    if (fast_mode && target.location.above_stack(board)) {
                        return;
                    }
                    auto it = values.find(target);
                    auto prev = (it == values.end()) ? 40u : it->second;
                    if (soft_drops < prev) {
                        values[target] = soft_drops;
                        queue.push(Intermediate{target, soft_drops});
                    }
                };

                if (auto shifted = shift(location, collision_map, -1)) {
                    update_position(*shifted, static_cast<std::uint32_t>(distance));
                }
                if (auto shifted = shift(location, collision_map, 1)) {
                    update_position(*shifted, static_cast<std::uint32_t>(distance));
                }
                if (auto rotated = rotate_cw(location, collision_map, board)) {
                    update_position(*rotated, static_cast<std::uint32_t>(distance));
                }
                if (auto rotated = rotate_ccw(location, collision_map, board)) {
                    update_position(*rotated, static_cast<std::uint32_t>(distance));
                }

                if (location.canonical_form() == location) {
                    locks.emplace_back(mv, 0);
                }
            }
        }
    } else {
        PieceLocation spawned{piece, Rotation::North, 4, 19};
        if (collision_map.obstructed(spawned)) {
            spawned.y = static_cast<std::int8_t>(spawned.y + 1);
            if (collision_map.obstructed(spawned)) {
                return {};
            }
        }
        Placement spawned_move{spawned, Spin::None};
        queue.push(Intermediate{spawned_move, 0});
        values.emplace(spawned_move, 0);
    }

    while (!queue.empty()) {
        auto expand = queue.top();
        queue.pop();
        auto it = values.find(expand.mv);
        if (it == values.end() || expand.soft_drops != it->second) {
            continue;
        }

        auto drop_dist = expand.mv.location.drop_distance(board);
        PieceLocation dropped_loc = expand.mv.location;
        dropped_loc.y = static_cast<std::int8_t>(dropped_loc.y - drop_dist);
        Placement dropped{
            dropped_loc, drop_dist == 0 ? expand.mv.spin : Spin::None,
        };

        Placement canonical{dropped.location.canonical_form(), dropped.spin};
        auto found = underground_locks.find(canonical);
        if (found == underground_locks.end() || expand.soft_drops < found->second) {
            underground_locks[canonical] = expand.soft_drops;
        }

        auto update_position = [&](const Placement& target, std::uint32_t soft_drops) {
            if (fast_mode && target.location.above_stack(board)) {
                return;
            }
            auto it2 = values.find(target);
            auto prev = (it2 == values.end()) ? 40u : it2->second;
            if (soft_drops < prev) {
                values[target] = soft_drops;
                queue.push(Intermediate{target, soft_drops});
            }
        };

        update_position(dropped, expand.soft_drops + static_cast<std::uint32_t>(drop_dist));

        if (auto shifted = shift(expand.mv.location, collision_map, -1)) {
            update_position(*shifted, expand.soft_drops);
        }
        if (auto shifted = shift(expand.mv.location, collision_map, 1)) {
            update_position(*shifted, expand.soft_drops);
        }
        if (auto rotated = rotate_cw(expand.mv.location, collision_map, board)) {
            update_position(*rotated, expand.soft_drops);
        }
        if (auto rotated = rotate_ccw(expand.mv.location, collision_map, board)) {
            update_position(*rotated, expand.soft_drops);
        }
    }

    for (const auto& kv : underground_locks) {
        locks.emplace_back(kv.first, kv.second);
    }
    return locks;
}

}  // namespace cold_clear_cpp
