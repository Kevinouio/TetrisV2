#include "tetris_v2/env.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <queue>
#include <stdexcept>
#include <sstream>
#include <unordered_map>

#include "tetris_v2/piece_defs.hpp"

namespace tetris_v2 {

namespace {

SpinType classify_spin(
    const Board& locked_board,
    const ActivePiece& placement,
    bool spin_eligible,
    int last_rotate_kick_index) {
    if (placement.piece != Piece::T || !spin_eligible) {
        return SpinType::None;
    }

    auto occupied_corner = [&](int dx, int dy) {
        return locked_board.occupied(placement.x + dx, placement.y + dy);
    };
    const int corners =
        static_cast<int>(occupied_corner(-1, -1)) +
        static_cast<int>(occupied_corner(1, -1)) +
        static_cast<int>(occupied_corner(-1, 1)) +
        static_cast<int>(occupied_corner(1, 1));
    if (corners < 3) {
        return SpinType::None;
    }

    auto rotate = [&](int x, int y) -> Cell {
        switch (placement.rotation) {
            case Rotation::North: return Cell{x, y};
            case Rotation::East: return Cell{y, -x};
            case Rotation::South: return Cell{-x, -y};
            case Rotation::West: return Cell{-y, x};
        }
        return Cell{x, y};
    };
    const auto front_left = rotate(-1, 1);
    const auto front_right = rotate(1, 1);
    const bool both_front_corners =
        occupied_corner(front_left.x, front_left.y) &&
        occupied_corner(front_right.x, front_right.y);
    return (both_front_corners || last_rotate_kick_index == 4)
        ? SpinType::Full
        : SpinType::Mini;
}

struct PathMetadata {
    bool spin_eligible{false};
    bool last_rotate_used_kick{false};
    int last_rotate_kick_index{-1};
};

bool metadata_is_better(const PathMetadata& candidate, const PathMetadata& baseline) {
    if (candidate.spin_eligible != baseline.spin_eligible) {
        return candidate.spin_eligible && !baseline.spin_eligible;
    }
    if (candidate.spin_eligible) {
        const bool candidate_full_kick = candidate.last_rotate_kick_index == 4;
        const bool baseline_full_kick = baseline.last_rotate_kick_index == 4;
        if (candidate_full_kick != baseline_full_kick) {
            return candidate_full_kick;
        }
        if (candidate.last_rotate_used_kick != baseline.last_rotate_used_kick) {
            return candidate.last_rotate_used_kick && !baseline.last_rotate_used_kick;
        }
        if (candidate.last_rotate_kick_index != baseline.last_rotate_kick_index) {
            return candidate.last_rotate_kick_index > baseline.last_rotate_kick_index;
        }
    }
    return false;
}

bool placement_metadata_is_better(const PlacementOption& candidate, const PlacementOption& baseline) {
    return static_cast<int>(candidate.spin_type) > static_cast<int>(baseline.spin_type);
}

bool kick_passed_with_nonzero_offset(const std::vector<KickTest>& tests) {
    for (const auto& test : tests) {
        if (test.passed) {
            return test.kick_index != 0;
        }
    }
    return false;
}

int passing_kick_index(const std::vector<KickTest>& tests) {
    for (const auto& test : tests) {
        if (test.passed) {
            return test.kick_index;
        }
    }
    return -1;
}

using KickOffsets = std::array<Cell, 5>;

void compact_piece_ids(
    std::array<std::array<std::int8_t, Board::kWidth>, Board::kRows>& ids,
    const std::array<bool, Board::kRows>& cleared_rows) {
    std::array<std::array<std::int8_t, Board::kWidth>, Board::kRows> compacted{};
    for (auto& row : compacted) {
        row.fill(-1);
    }

    int dst = 0;
    for (int src = 0; src < Board::kRows; ++src) {
        if (cleared_rows[static_cast<std::size_t>(src)]) {
            continue;
        }
        compacted[static_cast<std::size_t>(dst++)] = ids[static_cast<std::size_t>(src)];
    }
    ids = compacted;
}

KickOffsets rotation_offsets(Piece piece, Rotation rotation) {
    switch (piece) {
        case Piece::O:
            switch (rotation) {
                case Rotation::North:
                    return {Cell{0, 0}, Cell{0, 0}, Cell{0, 0}, Cell{0, 0}, Cell{0, 0}};
                case Rotation::East:
                    return {Cell{0, -1}, Cell{0, -1}, Cell{0, -1}, Cell{0, -1}, Cell{0, -1}};
                case Rotation::South:
                    return {Cell{-1, -1}, Cell{-1, -1}, Cell{-1, -1}, Cell{-1, -1}, Cell{-1, -1}};
                case Rotation::West:
                    return {Cell{-1, 0}, Cell{-1, 0}, Cell{-1, 0}, Cell{-1, 0}, Cell{-1, 0}};
            }
            break;
        case Piece::I:
            switch (rotation) {
                case Rotation::North:
                    return {Cell{0, 0}, Cell{-1, 0}, Cell{2, 0}, Cell{-1, 0}, Cell{2, 0}};
                case Rotation::East:
                    return {Cell{-1, 0}, Cell{0, 0}, Cell{0, 0}, Cell{0, 1}, Cell{0, -2}};
                case Rotation::South:
                    return {Cell{-1, 1}, Cell{1, 1}, Cell{-2, 1}, Cell{1, 0}, Cell{-2, 0}};
                case Rotation::West:
                    return {Cell{0, 1}, Cell{0, 1}, Cell{0, 1}, Cell{0, -1}, Cell{0, 2}};
            }
            break;
        case Piece::None:
            break;
        default:
            switch (rotation) {
                case Rotation::North:
                    return {Cell{0, 0}, Cell{0, 0}, Cell{0, 0}, Cell{0, 0}, Cell{0, 0}};
                case Rotation::East:
                    return {Cell{0, 0}, Cell{1, 0}, Cell{1, -1}, Cell{0, 2}, Cell{1, 2}};
                case Rotation::South:
                    return {Cell{0, 0}, Cell{0, 0}, Cell{0, 0}, Cell{0, 0}, Cell{0, 0}};
                case Rotation::West:
                    return {Cell{0, 0}, Cell{-1, 0}, Cell{-1, -1}, Cell{0, 2}, Cell{-1, 2}};
            }
            break;
    }
    return {Cell{0, 0}, Cell{0, 0}, Cell{0, 0}, Cell{0, 0}, Cell{0, 0}};
}

KickOffsets srs_kicks(Piece piece, Rotation from, Rotation to) {
    KickOffsets kicks{};
    auto from_offsets = rotation_offsets(piece, from);
    auto to_offsets = rotation_offsets(piece, to);
    for (std::size_t i = 0; i < kicks.size(); ++i) {
        kicks[i] = Cell{
            from_offsets[i].x - to_offsets[i].x,
            from_offsets[i].y - to_offsets[i].y,
        };
    }
    return kicks;
}

}  // namespace

ModernTetrisEnv::ModernTetrisEnv(const EnvConfig& config)
    : config_(config), randomizer_(config.seed) {
    reset();
}

void ModernTetrisEnv::reset(std::optional<std::uint32_t> seed) {
    if (seed.has_value()) {
        randomizer_.reseed(*seed);
    } else {
        randomizer_.reseed(config_.seed);
    }

    state_ = EnvState{};
    state_.board.clear();
    for (auto& row : state_.piece_ids) {
        row.fill(-1);
    }
    ensure_queue(static_cast<std::size_t>(config_.queue_size) + 1u);
    spawn_next_piece(true);
}

StepResult ModernTetrisEnv::step(Action action) { return advance(action, true); }

StepResult ModernTetrisEnv::input(Action action) { return advance(action, false); }

StepResult ModernTetrisEnv::tick() { return advance(Action::None, true); }

StepResult ModernTetrisEnv::advance(Action action, bool advance_time) {
    StepResult result{};
    if (state_.game_over) {
        result.game_over = true;
        result.top_out = state_.top_out;
        result.combo = state_.combo;
        result.back_to_back = state_.back_to_back;
        return result;
    }

    bool skip_gravity = false;
    bool moved_while_grounded = false;
    bool action_succeeded = false;
    const bool was_grounded = touching_ground();

    switch (action) {
        case Action::Left:
            action_succeeded = try_move(-1, 0);
            if (action_succeeded) {
                state_.spin_eligible = false;
                state_.last_rotate_used_kick = false;
                state_.last_rotate_kick_index = -1;
            }
            moved_while_grounded = action_succeeded && (was_grounded || touching_ground());
            break;
        case Action::Right:
            action_succeeded = try_move(1, 0);
            if (action_succeeded) {
                state_.spin_eligible = false;
                state_.last_rotate_used_kick = false;
                state_.last_rotate_kick_index = -1;
            }
            moved_while_grounded = action_succeeded && (was_grounded || touching_ground());
            break;
        case Action::RotateCW: {
            bool used_kick = false;
            int kick_index = -1;
            action_succeeded = try_rotate(rotate_cw(state_.active.rotation), &used_kick, &kick_index);
            if (action_succeeded) {
                state_.spin_eligible = true;
                state_.last_rotate_used_kick = used_kick;
                state_.last_rotate_kick_index = kick_index;
            }
            moved_while_grounded = action_succeeded && (was_grounded || touching_ground());
            break;
        }
        case Action::RotateCCW: {
            bool used_kick = false;
            int kick_index = -1;
            action_succeeded = try_rotate(rotate_ccw(state_.active.rotation), &used_kick, &kick_index);
            if (action_succeeded) {
                state_.spin_eligible = true;
                state_.last_rotate_used_kick = used_kick;
                state_.last_rotate_kick_index = kick_index;
            }
            moved_while_grounded = action_succeeded && (was_grounded || touching_ground());
            break;
        }
        case Action::Rotate180:
            if (config_.allow_rotate_180) {
                auto rotated = kicked_rotate_180_with_kick(state_.active);
                if (rotated.has_value()) {
                    state_.active = rotated->first;
                    state_.spin_eligible = true;
                    state_.last_rotate_used_kick = rotated->second;
                    state_.last_rotate_kick_index = -1;
                    action_succeeded = true;
                }
                moved_while_grounded = action_succeeded && (was_grounded || touching_ground());
            }
            break;
        case Action::SoftDrop:
            if (try_move(0, -1)) {
                action_succeeded = true;
                state_.spin_eligible = false;
                state_.last_rotate_used_kick = false;
                state_.last_rotate_kick_index = -1;
                result.reward += 1.0f;
            }
            break;
        case Action::HardDrop: {
            int dropped = 0;
            while (try_move(0, -1)) {
                ++dropped;
            }
            result.reward += static_cast<float>(2 * dropped);
            action_succeeded = dropped > 0 || touching_ground();
            lock_active_piece(result);
            break;
        }
        case Action::Hold:
            action_succeeded = apply_hold();
            result.hold_used = action_succeeded;
            skip_gravity = action_succeeded;
            break;
        case Action::None:
            break;
    }

    if (moved_while_grounded) {
        if (state_.lock_resets_used < config_.max_lock_resets) {
            state_.lock_timer = 0;
            ++state_.lock_resets_used;
        }
    }

    if (advance_time && !state_.game_over && !result.piece_locked && !skip_gravity) {
        state_.gravity_accumulator += config_.gravity_per_step;
        while (state_.gravity_accumulator >= 1.0) {
            if (!try_move(0, -1)) {
                state_.gravity_accumulator = 0.0;
                break;
            }
            state_.gravity_accumulator -= 1.0;
        }
    }

    if (advance_time && !state_.game_over && !result.piece_locked) {
        if (touching_ground()) {
            ++state_.lock_timer;
            if (state_.lock_timer >= config_.lock_delay_steps) {
                lock_active_piece(result);
            }
        } else if (state_.lock_resets_used < config_.max_lock_resets) {
            state_.lock_timer = 0;
        }
    }

    result.action_succeeded = action_succeeded;
    result.combo = state_.combo;
    result.back_to_back = state_.back_to_back;
    result.game_over = state_.game_over;
    result.top_out = state_.top_out;
    return result;
}

std::vector<PlacementOption> ModernTetrisEnv::enumerate_active_piece_placements() const {
    std::vector<PlacementOption> options;
    if (state_.game_over || state_.active.piece == Piece::None || collides(state_.active)) {
        return options;
    }

    struct PoseKey {
        Piece piece{Piece::None};
        Rotation rotation{Rotation::North};
        int x{0};
        int y{0};

        bool operator==(const PoseKey& rhs) const {
            return piece == rhs.piece && rotation == rhs.rotation && x == rhs.x && y == rhs.y;
        }
    };

    struct PoseKeyHash {
        std::size_t operator()(const PoseKey& k) const noexcept {
            std::size_t h = 1469598103934665603ull;
            auto mix = [&h](std::size_t v) {
                h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
            };
            mix(static_cast<std::size_t>(k.piece));
            mix(static_cast<std::size_t>(k.rotation));
            mix(static_cast<std::size_t>(k.x + 128));
            mix(static_cast<std::size_t>(k.y + 128));
            return h;
        }
    };

    auto to_key = [](const ActivePiece& p) {
        return PoseKey{p.piece, p.rotation, p.x, p.y};
    };

    auto grounded = [this](const ActivePiece& p) {
        ActivePiece below = p;
        below.y -= 1;
        return collides(below);
    };

    struct SearchNode {
        ActivePiece piece{};
        PathMetadata metadata{};
    };

    std::queue<SearchNode> frontier;
    std::unordered_map<PoseKey, PathMetadata, PoseKeyHash> visited_best;
    std::unordered_map<PoseKey, std::size_t, PoseKeyHash> locked_best;

    PathMetadata start_metadata{
        state_.spin_eligible,
        state_.last_rotate_used_kick,
        state_.last_rotate_kick_index};
    frontier.push(SearchNode{state_.active, start_metadata});
    visited_best.emplace(to_key(state_.active), start_metadata);

    auto try_enqueue = [&](const ActivePiece& candidate, const PathMetadata& metadata) {
        if (collides(candidate)) {
            return;
        }
        auto key = to_key(candidate);
        auto it = visited_best.find(key);
        if (it == visited_best.end()) {
            visited_best.emplace(key, metadata);
            frontier.push(SearchNode{candidate, metadata});
            return;
        }
        if (metadata_is_better(metadata, it->second)) {
            it->second = metadata;
            frontier.push(SearchNode{candidate, metadata});
        }
    };

    while (!frontier.empty()) {
        SearchNode current = frontier.front();
        frontier.pop();

        auto key = to_key(current.piece);
        auto best_it = visited_best.find(key);
        if (best_it == visited_best.end()) {
            continue;
        }
        if (best_it->second.spin_eligible != current.metadata.spin_eligible ||
            best_it->second.last_rotate_used_kick != current.metadata.last_rotate_used_kick ||
            best_it->second.last_rotate_kick_index != current.metadata.last_rotate_kick_index) {
            continue;
        }

        if (grounded(current.piece)) {
            Board board_after = state_.board;
            auto cells = piece_cells(current.piece.piece, current.piece.rotation);
            for (const auto& cell : cells) {
                board_after.set_cell(current.piece.x + cell.x, current.piece.y + cell.y, true);
            }
            std::array<bool, Board::kRows> cleared_rows{};
            int cleared = 0;
            for (int y = 0; y < Board::kRows; ++y) {
                bool full = (board_after.row_mask(y) & Board::kFullRowMask) == Board::kFullRowMask;
                cleared_rows[static_cast<std::size_t>(y)] = full;
                if (full) {
                    ++cleared;
                }
            }
            const auto spin_type = classify_spin(
                board_after,
                current.piece,
                current.metadata.spin_eligible,
                current.metadata.last_rotate_kick_index);
            board_after.clear_filled_lines();
            PlacementOption candidate_option{
                current.piece,
                board_after,
                cleared_rows,
                cleared,
                current.metadata.spin_eligible,
                current.metadata.last_rotate_used_kick,
                current.metadata.last_rotate_kick_index,
                spin_type,
            };
            auto existing = locked_best.find(key);
            if (existing == locked_best.end()) {
                locked_best.emplace(key, options.size());
                options.push_back(std::move(candidate_option));
            } else {
                auto& baseline = options[existing->second];
                if (placement_metadata_is_better(candidate_option, baseline)) {
                    baseline = std::move(candidate_option);
                }
            }
        }

        ActivePiece left = current.piece;
        left.x -= 1;
        try_enqueue(left, PathMetadata{false, false, -1});

        ActivePiece right = current.piece;
        right.x += 1;
        try_enqueue(right, PathMetadata{false, false, -1});

        ActivePiece down = current.piece;
        down.y -= 1;
        try_enqueue(down, PathMetadata{false, false, -1});

        auto [cw, cw_tests] =
            kicked_rotation_with_tests(current.piece, rotate_cw(current.piece.rotation), 0, 0);
        if (cw.has_value()) {
            const int kick_index = passing_kick_index(cw_tests);
            try_enqueue(*cw, PathMetadata{true, kick_index > 0, kick_index});
        }

        auto [ccw, ccw_tests] =
            kicked_rotation_with_tests(current.piece, rotate_ccw(current.piece.rotation), 0, 0);
        if (ccw.has_value()) {
            const int kick_index = passing_kick_index(ccw_tests);
            try_enqueue(*ccw, PathMetadata{true, kick_index > 0, kick_index});
        }

        if (config_.allow_rotate_180) {
            if (auto r180 = kicked_rotate_180_with_kick(current.piece)) {
                try_enqueue(r180->first, PathMetadata{true, r180->second, -1});
            }
        }
    }

    std::sort(options.begin(), options.end(), [](const PlacementOption& a, const PlacementOption& b) {
        if (a.placement.x != b.placement.x) {
            return a.placement.x < b.placement.x;
        }
        if (a.placement.rotation != b.placement.rotation) {
            return static_cast<int>(a.placement.rotation) < static_cast<int>(b.placement.rotation);
        }
        return a.placement.y < b.placement.y;
    });

    return options;
}

std::optional<ActivePiece> ModernTetrisEnv::ghost_piece() const {
    if (state_.game_over || state_.active.piece == Piece::None || collides(state_.active)) {
        return std::nullopt;
    }

    ActivePiece ghost = state_.active;
    for (;;) {
        ActivePiece candidate = ghost;
        --candidate.y;
        if (collides(candidate)) {
            return ghost;
        }
        ghost = candidate;
    }
}

std::optional<PlacementOption> ModernTetrisEnv::placement_option_at(std::size_t index) const {
    auto options = enumerate_active_piece_placements();
    if (index >= options.size()) {
        return std::nullopt;
    }
    return options[index];
}

std::vector<std::uint8_t> ModernTetrisEnv::visible_board_piece_ids(bool include_active) const {
    constexpr std::uint8_t kEmpty = 255;
    std::vector<std::uint8_t> out(
        static_cast<std::size_t>(Board::kVisibleRows * Board::kWidth), kEmpty);

    for (int row = 0; row < Board::kVisibleRows; ++row) {
        int y = (Board::kVisibleRows - 1) - row;
        for (int x = 0; x < Board::kWidth; ++x) {
            auto id = state_.piece_ids[static_cast<std::size_t>(y)][static_cast<std::size_t>(x)];
            if (id >= 0 && id <= kGarbagePieceId) {
                out[static_cast<std::size_t>(row * Board::kWidth + x)] =
                    static_cast<std::uint8_t>(id);
            }
        }
    }

    if (include_active && state_.active.piece != Piece::None) {
        auto cells = piece_cells(state_.active.piece, state_.active.rotation);
        auto active_id = static_cast<std::uint8_t>(piece_index(state_.active.piece));
        for (const auto& c : cells) {
            int x = state_.active.x + c.x;
            int y = state_.active.y + c.y;
            if (x < 0 || x >= Board::kWidth || y < 0 || y >= Board::kVisibleRows) {
                continue;
            }
            int row = (Board::kVisibleRows - 1) - y;
            out[static_cast<std::size_t>(row * Board::kWidth + x)] = active_id;
        }
    }

    return out;
}

std::vector<std::uint8_t> ModernTetrisEnv::visible_placement_piece_ids(std::size_t index) const {
    constexpr std::uint8_t kEmpty = 255;
    auto option = placement_option_at(index);
    if (!option.has_value()) {
        return std::vector<std::uint8_t>(
            static_cast<std::size_t>(Board::kVisibleRows * Board::kWidth), kEmpty);
    }

    auto ids = state_.piece_ids;
    auto pid = static_cast<std::int8_t>(piece_index(option->placement.piece));
    auto cells = piece_cells(option->placement.piece, option->placement.rotation);
    for (const auto& c : cells) {
        int x = option->placement.x + c.x;
        int y = option->placement.y + c.y;
        if (x < 0 || x >= Board::kWidth || y < 0 || y >= Board::kRows) {
            continue;
        }
        ids[static_cast<std::size_t>(y)][static_cast<std::size_t>(x)] = pid;
    }

    if (option->lines_cleared > 0) {
        compact_piece_ids(ids, option->cleared_rows);
    }

    std::vector<std::uint8_t> out(
        static_cast<std::size_t>(Board::kVisibleRows * Board::kWidth), kEmpty);
    for (int row = 0; row < Board::kVisibleRows; ++row) {
        int y = (Board::kVisibleRows - 1) - row;
        for (int x = 0; x < Board::kWidth; ++x) {
            auto id = ids[static_cast<std::size_t>(y)][static_cast<std::size_t>(x)];
            if (id >= 0 && id <= kGarbagePieceId) {
                out[static_cast<std::size_t>(row * Board::kWidth + x)] =
                    static_cast<std::uint8_t>(id);
            }
        }
    }
    return out;
}

StepResult ModernTetrisEnv::apply_placement(const ActivePiece& placement) {
    StepResult result{};
    if (state_.game_over) {
        result.game_over = true;
        result.top_out = state_.top_out;
        result.combo = state_.combo;
        result.back_to_back = state_.back_to_back;
        return result;
    }

    auto options = enumerate_active_piece_placements();
    auto it = std::find_if(options.begin(), options.end(), [&](const PlacementOption& opt) {
        return opt.placement == placement;
    });
    if (it == options.end()) {
        result.action_succeeded = false;
        result.combo = state_.combo;
        result.back_to_back = state_.back_to_back;
        result.game_over = state_.game_over;
        result.top_out = state_.top_out;
        return result;
    }

    state_.active = it->placement;
    state_.spin_eligible = it->spin_eligible_path;
    state_.last_rotate_used_kick = it->last_rotate_used_kick_path;
    state_.last_rotate_kick_index = it->last_rotate_kick_index_path;
    lock_active_piece(result);
    result.action_succeeded = true;
    result.combo = state_.combo;
    result.back_to_back = state_.back_to_back;
    result.game_over = state_.game_over;
    result.top_out = state_.top_out;
    return result;
}

StepResult ModernTetrisEnv::apply_placement_index(std::size_t index) {
    StepResult result{};
    if (state_.game_over) {
        result.game_over = true;
        result.top_out = state_.top_out;
        result.combo = state_.combo;
        result.back_to_back = state_.back_to_back;
        return result;
    }

    auto option = placement_option_at(index);
    if (!option.has_value()) {
        result.action_succeeded = false;
        result.combo = state_.combo;
        result.back_to_back = state_.back_to_back;
        result.game_over = state_.game_over;
        result.top_out = state_.top_out;
        return result;
    }

    state_.active = option->placement;
    state_.spin_eligible = option->spin_eligible_path;
    state_.last_rotate_used_kick = option->last_rotate_used_kick_path;
    state_.last_rotate_kick_index = option->last_rotate_kick_index_path;
    lock_active_piece(result);
    result.action_succeeded = true;
    result.combo = state_.combo;
    result.back_to_back = state_.back_to_back;
    result.game_over = state_.game_over;
    result.top_out = state_.top_out;
    return result;
}

GarbageInsertResult ModernTetrisEnv::insert_garbage_rows(
    const std::vector<int>& hole_columns) {
    GarbageInsertResult result{};
    result.rows_requested = static_cast<int>(hole_columns.size());
    result.top_out = state_.top_out;
    if (hole_columns.empty() || state_.game_over) {
        return result;
    }
    for (const int hole : hole_columns) {
        if (hole < 0 || hole >= Board::kWidth) {
            throw std::invalid_argument("Garbage hole column is outside the board.");
        }
    }

    const int requested = static_cast<int>(hole_columns.size());
    const int inserted = std::min(requested, Board::kRows);
    result.rows_applied = inserted;
    result.overflow = requested > Board::kRows;

    const Board previous_board = state_.board;
    const auto previous_ids = state_.piece_ids;
    Board raised_board{};
    std::array<std::array<std::int8_t, Board::kWidth>, Board::kRows> raised_ids{};
    for (auto& row : raised_ids) {
        row.fill(kEmptyPieceId);
    }

    for (int y = 0; y < Board::kRows; ++y) {
        const auto mask = previous_board.row_mask(y);
        if (mask == 0) {
            continue;
        }
        const int destination_y = y + requested;
        if (destination_y >= Board::kRows) {
            result.overflow = true;
            continue;
        }
        for (int x = 0; x < Board::kWidth; ++x) {
            if ((mask & (Board::RowMask{1u} << x)) == 0) {
                continue;
            }
            raised_board.set_cell(x, destination_y);
            raised_ids[static_cast<std::size_t>(destination_y)][static_cast<std::size_t>(x)] =
                previous_ids[static_cast<std::size_t>(y)][static_cast<std::size_t>(x)];
        }
    }

    // Holes are supplied in arrival order. Applying one row at a time places the
    // newest row at the bottom, so a batch appears in reverse arrival order.
    for (int y = 0; y < inserted; ++y) {
        const auto hole_index = hole_columns.size() - 1u - static_cast<std::size_t>(y);
        const int hole = hole_columns[hole_index];
        for (int x = 0; x < Board::kWidth; ++x) {
            if (x == hole) {
                continue;
            }
            raised_board.set_cell(x, y);
            raised_ids[static_cast<std::size_t>(y)][static_cast<std::size_t>(x)] =
                kGarbagePieceId;
        }
    }

    state_.board = raised_board;
    state_.piece_ids = raised_ids;
    result.active_collision =
        state_.active.piece != Piece::None && collides(state_.active);
    if (result.overflow || result.active_collision) {
        state_.game_over = true;
        state_.top_out = true;
    }
    result.top_out = state_.top_out;
    return result;
}

RotationTrace ModernTetrisEnv::rotation_trace(Action rotate_action) const {
    RotationTrace trace{};
    trace.action = rotate_action;
    trace.from_rotation = state_.active.rotation;
    trace.target_rotation = state_.active.rotation;
    if (state_.active.piece == Piece::None) {
        return trace;
    }

    if (rotate_action == Action::RotateCW || rotate_action == Action::RotateCCW) {
        trace.target_rotation = (rotate_action == Action::RotateCW)
                                    ? rotate_cw(state_.active.rotation)
                                    : rotate_ccw(state_.active.rotation);
        auto [pose, tests] =
            kicked_rotation_with_tests(state_.active, trace.target_rotation, 0, 0);
        trace.tests = std::move(tests);
        trace.success = pose.has_value();
        trace.final_pose = pose;
        return trace;
    }

    if (rotate_action == Action::Rotate180) {
        trace.target_rotation = rotate_180(state_.active.rotation);
        int next_index = 0;
        auto [cw_first, tests_cw_first] =
            kicked_rotation_with_tests(state_.active, rotate_cw(state_.active.rotation), 0, next_index);
        next_index += static_cast<int>(tests_cw_first.size());
        trace.tests.insert(trace.tests.end(), tests_cw_first.begin(), tests_cw_first.end());

        if (cw_first.has_value()) {
            auto [cw_second, tests_cw_second] =
                kicked_rotation_with_tests(*cw_first, rotate_cw(cw_first->rotation), 1, next_index);
            next_index += static_cast<int>(tests_cw_second.size());
            trace.tests.insert(trace.tests.end(), tests_cw_second.begin(), tests_cw_second.end());
            if (cw_second.has_value()) {
                trace.success = true;
                trace.final_pose = cw_second;
                return trace;
            }
        }

        auto [ccw_first, tests_ccw_first] =
            kicked_rotation_with_tests(state_.active, rotate_ccw(state_.active.rotation), 2, next_index);
        next_index += static_cast<int>(tests_ccw_first.size());
        trace.tests.insert(trace.tests.end(), tests_ccw_first.begin(), tests_ccw_first.end());

        if (ccw_first.has_value()) {
            auto [ccw_second, tests_ccw_second] =
                kicked_rotation_with_tests(*ccw_first, rotate_ccw(ccw_first->rotation), 3, next_index);
            next_index += static_cast<int>(tests_ccw_second.size());
            trace.tests.insert(trace.tests.end(), tests_ccw_second.begin(), tests_ccw_second.end());
            if (ccw_second.has_value()) {
                trace.success = true;
                trace.final_pose = ccw_second;
                return trace;
            }
        }
    }

    return trace;
}

EnvSnapshot ModernTetrisEnv::snapshot() const {
    EnvSnapshot out{};
    out.state = state_;
    out.randomizer = randomizer_;
    return out;
}

void ModernTetrisEnv::restore(const EnvSnapshot& snapshot) {
    state_ = snapshot.state;
    randomizer_ = snapshot.randomizer;
}

void ModernTetrisEnv::restore(const EnvState& state) { state_ = state; }

std::vector<std::string> ModernTetrisEnv::render_rows(int visible_rows) const {
    auto rows = state_.board.render_rows(visible_rows);
    const int rendered_rows = static_cast<int>(rows.size());
    auto cells = piece_cells(state_.active.piece, state_.active.rotation);
    for (const auto& cell : cells) {
        int x = state_.active.x + cell.x;
        int y = state_.active.y + cell.y;
        if (x < 0 || x >= Board::kWidth || y < 0 || y >= rendered_rows) {
            continue;
        }
        int row_index = rendered_rows - 1 - y;
        rows[static_cast<std::size_t>(row_index)][static_cast<std::size_t>(x)] =
            piece_to_char(state_.active.piece);
    }
    return rows;
}

std::string ModernTetrisEnv::render_ascii(int visible_rows) const {
    std::ostringstream out;
    auto rows = render_rows(visible_rows);
    for (const auto& row : rows) {
        out << row << '\n';
    }
    return out.str();
}

void ModernTetrisEnv::ensure_queue(std::size_t minimum) {
    while (state_.queue.size() < minimum) {
        state_.queue.push_back(randomizer_.next_piece());
    }
}

void ModernTetrisEnv::spawn_next_piece(bool reset_hold_availability) {
    ensure_queue(static_cast<std::size_t>(config_.queue_size) + 1u);
    if (state_.queue.empty()) {
        state_.game_over = true;
        return;
    }

    auto next = state_.queue.front();
    state_.queue.pop_front();
    state_.active = spawn_piece(next);
    state_.active.y = config_.spawn_y;
    state_.lock_timer = 0;
    state_.lock_resets_used = 0;
    state_.gravity_accumulator = 0.0;
    state_.spin_eligible = false;
    state_.last_rotate_used_kick = false;
    state_.last_rotate_kick_index = -1;
    if (reset_hold_availability) {
        state_.hold_available = true;
    }

    if (collides(state_.active)) {
        state_.game_over = true;
        state_.top_out = true;
    }
}

bool ModernTetrisEnv::collides(const ActivePiece& piece) const {
    auto cells = piece_cells(piece.piece, piece.rotation);
    for (const auto& cell : cells) {
        int x = piece.x + cell.x;
        int y = piece.y + cell.y;
        if (state_.board.occupied(x, y)) {
            return true;
        }
    }
    return false;
}

bool ModernTetrisEnv::try_move(int dx, int dy) {
    ActivePiece candidate = state_.active;
    candidate.x += dx;
    candidate.y += dy;
    if (collides(candidate)) {
        return false;
    }
    state_.active = candidate;
    return true;
}

bool ModernTetrisEnv::try_rotate(Rotation target_rotation, bool* used_kick, int* kick_index) {
    auto [rotated, tests] = kicked_rotation_with_tests(state_.active, target_rotation, 0, 0);
    if (!rotated.has_value()) {
        if (used_kick) {
            *used_kick = false;
        }
        if (kick_index) {
            *kick_index = -1;
        }
        return false;
    }
    int passed = passing_kick_index(tests);
    if (used_kick) {
        *used_kick = passed > 0;
    }
    if (kick_index) {
        *kick_index = passed;
    }
    state_.active = *rotated;
    return true;
}

std::pair<std::optional<ActivePiece>, std::vector<KickTest>> ModernTetrisEnv::kicked_rotation_with_tests(
    const ActivePiece& from, Rotation target_rotation, int phase, int start_test_index) const {
    std::vector<KickTest> tests;
    if (from.piece == Piece::None) {
        return {std::nullopt, tests};
    }

    auto kicks = srs_kicks(from.piece, from.rotation, target_rotation);
    for (std::size_t i = 0; i < kicks.size(); ++i) {
        ActivePiece candidate = from;
        candidate.rotation = target_rotation;
        candidate.x += kicks[i].x;
        candidate.y += kicks[i].y;
        bool blocked = collides(candidate);

        KickTest kt{};
        kt.test_index = start_test_index + static_cast<int>(i);
        kt.phase = phase;
        kt.kick_index = static_cast<int>(i);
        kt.offset = kicks[i];
        kt.candidate = candidate;
        kt.collides = blocked;
        kt.passed = !blocked;
        tests.push_back(kt);

        if (!blocked) {
            return {candidate, tests};
        }
    }

    return {std::nullopt, tests};
}

std::optional<std::pair<ActivePiece, bool>> ModernTetrisEnv::kicked_rotate_180_with_kick(
    const ActivePiece& from) const {
    auto [cw_first, tests_cw_first] =
        kicked_rotation_with_tests(from, rotate_cw(from.rotation), 0, 0);
    if (cw_first.has_value()) {
        bool used_kick = kick_passed_with_nonzero_offset(tests_cw_first);
        auto [cw_second, tests_cw_second] =
            kicked_rotation_with_tests(*cw_first, rotate_cw(cw_first->rotation), 1, 0);
        if (cw_second.has_value()) {
            used_kick = used_kick || kick_passed_with_nonzero_offset(tests_cw_second);
            return std::make_pair(*cw_second, used_kick);
        }
    }

    auto [ccw_first, tests_ccw_first] =
        kicked_rotation_with_tests(from, rotate_ccw(from.rotation), 2, 0);
    if (ccw_first.has_value()) {
        bool used_kick = kick_passed_with_nonzero_offset(tests_ccw_first);
        auto [ccw_second, tests_ccw_second] =
            kicked_rotation_with_tests(*ccw_first, rotate_ccw(ccw_first->rotation), 3, 0);
        if (ccw_second.has_value()) {
            used_kick = used_kick || kick_passed_with_nonzero_offset(tests_ccw_second);
            return std::make_pair(*ccw_second, used_kick);
        }
    }

    return std::nullopt;
}

bool ModernTetrisEnv::touching_ground() const {
    ActivePiece candidate = state_.active;
    candidate.y -= 1;
    return collides(candidate);
}

bool ModernTetrisEnv::apply_hold() {
    if (!state_.hold_available || state_.active.piece == Piece::None) {
        return false;
    }

    Piece current = state_.active.piece;
    if (state_.hold.has_value()) {
        Piece swapped = *state_.hold;
        state_.hold = current;
        state_.active = spawn_piece(swapped);
        state_.active.y = config_.spawn_y;
        state_.lock_timer = 0;
        state_.lock_resets_used = 0;
        state_.gravity_accumulator = 0.0;
        state_.spin_eligible = false;
        state_.last_rotate_used_kick = false;
        state_.last_rotate_kick_index = -1;
        if (collides(state_.active)) {
            state_.game_over = true;
            state_.top_out = true;
        }
    } else {
        state_.hold = current;
        spawn_next_piece(false);
    }

    state_.hold_available = false;
    return true;
}

void ModernTetrisEnv::lock_active_piece(StepResult& result) {
    auto cells = piece_cells(state_.active.piece, state_.active.rotation);
    const bool locks_completely_above_visible =
        std::all_of(cells.begin(), cells.end(), [&](const Cell& cell) {
            return state_.active.y + cell.y >= Board::kVisibleRows;
        });
    auto pid = static_cast<std::int8_t>(piece_index(state_.active.piece));
    for (const auto& cell : cells) {
        int x = state_.active.x + cell.x;
        int y = state_.active.y + cell.y;
        state_.board.set_cell(x, y, true);
        if (x >= 0 && x < Board::kWidth && y >= 0 && y < Board::kRows) {
            state_.piece_ids[static_cast<std::size_t>(y)][static_cast<std::size_t>(x)] = pid;
        }
    }

    std::array<bool, Board::kRows> cleared_rows{};
    int lines = 0;
    for (int y = 0; y < Board::kRows; ++y) {
        bool full = (state_.board.row_mask(y) & Board::kFullRowMask) == Board::kFullRowMask;
        cleared_rows[static_cast<std::size_t>(y)] = full;
        if (full) {
            ++lines;
        }
    }

    const auto spin_type = classify_spin(
        state_.board,
        state_.active,
        state_.spin_eligible,
        state_.last_rotate_kick_index);

    state_.board.clear_filled_lines();
    if (lines > 0) {
        compact_piece_ids(state_.piece_ids, cleared_rows);
    }
    result.piece_locked = true;
    result.lines_cleared = lines;
    result.spin_type = spin_type;
    result.spin_clear = result.spin_type != SpinType::None;
    result.difficult_clear = lines > 0 && ((lines == 4) || result.spin_clear);
    result.b2b_bonus_applied = false;
    const float base_reward = action_reward(lines, result.spin_type);
    result.reward += base_reward;

    if (lines > 0) {
        state_.combo += 1;
        if (result.difficult_clear && state_.back_to_back) {
            result.reward += base_reward * config_.scoring.b2b_bonus_ratio;
            result.b2b_bonus_applied = true;
        }
        if (result.difficult_clear) {
            state_.back_to_back = true;
        } else {
            state_.back_to_back = false;
        }
        result.reward += combo_bonus(state_.combo);
    } else {
        state_.combo = -1;
    }

    state_.last_clear_spin = result.spin_clear;
    state_.last_clear_spin_type = result.spin_type;
    state_.last_clear_difficult = result.difficult_clear;
    state_.last_clear_b2b_bonus = result.b2b_bonus_applied;
    state_.total_lines_cleared += lines;
    if (locks_completely_above_visible) {
        state_.game_over = true;
        state_.top_out = true;
        state_.hold_available = true;
    } else {
        spawn_next_piece(true);
    }
}

float ModernTetrisEnv::action_reward(int lines, SpinType spin_type) const {
    switch (spin_type) {
        case SpinType::None:
            if (lines >= 0 && lines < static_cast<int>(config_.scoring.normal_clear_reward.size())) {
                return config_.scoring.normal_clear_reward[static_cast<std::size_t>(lines)];
            }
            break;
        case SpinType::Mini:
            if (lines >= 0 && lines < static_cast<int>(config_.scoring.mini_spin_reward.size())) {
                return config_.scoring.mini_spin_reward[static_cast<std::size_t>(lines)];
            }
            break;
        case SpinType::Full:
            if (lines >= 0 && lines < static_cast<int>(config_.scoring.full_spin_reward.size())) {
                return config_.scoring.full_spin_reward[static_cast<std::size_t>(lines)];
            }
            break;
    }
    return 0.0f;
}

float ModernTetrisEnv::combo_bonus(int combo) const {
    if (combo <= 0) {
        return 0.0f;
    }
    return static_cast<float>(combo) * config_.scoring.combo_unit_bonus;
}

}  // namespace tetris_v2
