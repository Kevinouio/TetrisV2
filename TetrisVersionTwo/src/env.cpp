#include "tetris_v2/env.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <optional>
#include <queue>
#include <sstream>
#include <unordered_map>

#include "tetris_v2/piece_defs.hpp"

namespace tetris_v2 {

namespace {

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
    if (candidate.spin_clear_candidate != baseline.spin_clear_candidate) {
        return candidate.spin_clear_candidate && !baseline.spin_clear_candidate;
    }
    if (candidate.spin_clear_candidate &&
        candidate.last_rotate_used_kick_path != baseline.last_rotate_used_kick_path) {
        return candidate.last_rotate_used_kick_path && !baseline.last_rotate_used_kick_path;
    }
    const bool candidate_full_kick = candidate.last_rotate_kick_index_path == 4;
    const bool baseline_full_kick = baseline.last_rotate_kick_index_path == 4;
    if (candidate_full_kick != baseline_full_kick) {
        return candidate_full_kick;
    }
    if (candidate.last_rotate_kick_index_path != baseline.last_rotate_kick_index_path) {
        return candidate.last_rotate_kick_index_path > baseline.last_rotate_kick_index_path;
    }
    return false;
}

constexpr std::array<int, 8> kB2BChainingUpperBounds{
    3, 8, 24, 67, 185, 504, 1370, 3725,
};

constexpr int kBlitzMaxLevel = 15;
constexpr double kBlitzTickRateHz = 60.0;

constexpr std::array<int, kBlitzMaxLevel> kBlitzLevelLineTotals{
    3, 8, 15, 24, 35, 48, 63, 80, 99, 120, 144, 170, 198, 228, 260,
};

constexpr std::array<double, kBlitzMaxLevel> kBlitzGravitySecondsPerRow{
    1.0, 0.643, 0.404, 0.249, 0.150, 0.0880, 0.0505, 0.0283,
    0.0155, 0.00827, 0.00431, 0.00219, 0.00108, 0.00052, 0.00024,
};

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
    : config_(config),
      randomizer_(config.seed),
      base_gravity_per_step_(config.gravity_per_step) {
    reset();
}

void ModernTetrisEnv::set_mode(GameMode mode) {
    if (config_.mode == mode) {
        return;
    }
    config_.mode = mode;
    if (is_mode_blitz()) {
        reset_blitz_state();
    } else {
        state_.blitz_timed_out = false;
        state_.blitz_time_remaining_ms = 0;
        state_.blitz_level = 1;
        state_.blitz_lines_to_next = 0;
        config_.gravity_per_step = base_gravity_per_step_;
        blitz_clock_started_ = false;
    }
}

void ModernTetrisEnv::set_blitz_time_limit_ms(int time_limit_ms) {
    blitz_time_limit_ms_ = time_limit_ms;
    if (!is_mode_blitz()) {
        return;
    }

    if (blitz_time_limit_ms_ <= 0) {
        state_.blitz_time_remaining_ms = 0;
        state_.blitz_timed_out = false;
        if (state_.game_over && !state_.top_out) {
            state_.game_over = false;
        }
        blitz_clock_started_ = false;
        return;
    }

    state_.blitz_time_remaining_ms = blitz_time_limit_ms_;
    state_.blitz_timed_out = false;
    if (state_.game_over && !state_.top_out) {
        state_.game_over = false;
    }
    blitz_last_wall_time_ = std::chrono::steady_clock::now();
    blitz_clock_started_ = true;
}

void ModernTetrisEnv::refresh_runtime_state() { refresh_blitz_timer(); }

void ModernTetrisEnv::reset(std::optional<std::uint32_t> seed) {
    if (seed.has_value()) {
        randomizer_.reseed(*seed);
        garbage_rng_state_ = (*seed == 0u ? 1u : *seed);
    } else {
        randomizer_.reseed(config_.seed);
        garbage_rng_state_ = (config_.seed == 0u ? 1u : config_.seed);
    }

    state_ = EnvState{};
    last_step_result_ = StepResult{};
    state_.board.clear();
    for (auto& row : state_.piece_ids) {
        row.fill(-1);
    }
    config_.gravity_per_step = base_gravity_per_step_;
    blitz_clock_started_ = false;
    if (is_mode_blitz()) {
        reset_blitz_state();
    }
    invalidate_placement_cache();
    ensure_queue(static_cast<std::size_t>(config_.queue_size) + 1u);
    spawn_next_piece(true);
}

StepResult ModernTetrisEnv::make_result_defaults() const {
    StepResult result{};
    result.combo = state_.combo;
    result.back_to_back = state_.back_to_back;
    result.b2b_streak = state_.b2b_streak;
    result.surge_charge = state_.b2b_surge_charge;
    result.blitz_mode = is_mode_blitz();
    result.timed_out = state_.blitz_timed_out;
    result.blitz_score_total = state_.blitz_score_total;
    result.blitz_level = state_.blitz_level;
    result.blitz_lines_to_next = state_.blitz_lines_to_next;
    result.blitz_time_remaining_ms = state_.blitz_time_remaining_ms;
    return result;
}

void ModernTetrisEnv::sync_state_to_result(StepResult& result) const {
    result.combo = state_.combo;
    result.back_to_back = state_.back_to_back;
    result.b2b_streak = state_.b2b_streak;
    result.game_over = state_.game_over;
    result.top_out = state_.top_out;
    result.timed_out = state_.blitz_timed_out;
    result.blitz_mode = is_mode_blitz();
    result.blitz_score_total = state_.blitz_score_total;
    result.blitz_level = state_.blitz_level;
    result.blitz_lines_to_next = state_.blitz_lines_to_next;
    result.blitz_time_remaining_ms = state_.blitz_time_remaining_ms;
}

StepResult ModernTetrisEnv::step(Action action) {
    refresh_runtime_state();
    StepResult result = make_result_defaults();
    if (state_.game_over) {
        sync_state_to_result(result);
        if (is_mode_legacy()) {
            result.legacy_reward = result.reward;
        }
        last_step_result_ = result;
        return result;
    }
    invalidate_placement_cache();

    bool skip_gravity = false;
    bool moved_while_grounded = false;
    bool action_succeeded = false;

    switch (action) {
        case Action::Left:
            action_succeeded = try_move(-1, 0);
            if (action_succeeded) {
                state_.spin_eligible = false;
                state_.last_rotate_used_kick = false;
                state_.last_rotate_kick_index = -1;
            }
            moved_while_grounded = action_succeeded && touching_ground();
            break;
        case Action::Right:
            action_succeeded = try_move(1, 0);
            if (action_succeeded) {
                state_.spin_eligible = false;
                state_.last_rotate_used_kick = false;
                state_.last_rotate_kick_index = -1;
            }
            moved_while_grounded = action_succeeded && touching_ground();
            break;
        case Action::RotateCW: {
            bool used_kick = false;
            int kick_index = -1;
            action_succeeded = try_rotate(rotate_cw(state_.active.rotation), &used_kick, &kick_index);
            if (action_succeeded) {
                state_.spin_eligible = true;
                state_.rotated_this_piece = true;
                state_.last_rotate_used_kick = used_kick;
                state_.last_rotate_kick_index = kick_index;
            }
            moved_while_grounded = action_succeeded && touching_ground();
            break;
        }
        case Action::RotateCCW: {
            bool used_kick = false;
            int kick_index = -1;
            action_succeeded = try_rotate(rotate_ccw(state_.active.rotation), &used_kick, &kick_index);
            if (action_succeeded) {
                state_.spin_eligible = true;
                state_.rotated_this_piece = true;
                state_.last_rotate_used_kick = used_kick;
                state_.last_rotate_kick_index = kick_index;
            }
            moved_while_grounded = action_succeeded && touching_ground();
            break;
        }
        case Action::Rotate180:
            if (config_.allow_rotate_180) {
                auto rotated = kicked_rotate_180_with_kick(state_.active);
                if (rotated.has_value()) {
                    state_.active = rotated->first;
                    state_.spin_eligible = true;
                    state_.rotated_this_piece = true;
                    state_.last_rotate_used_kick = rotated->second;
                    state_.last_rotate_kick_index = -1;
                    action_succeeded = true;
                }
                moved_while_grounded = action_succeeded && touching_ground();
            }
            break;
        case Action::SoftDrop:
            if (try_move(0, -1)) {
                action_succeeded = true;
                state_.spin_eligible = false;
                state_.last_rotate_used_kick = false;
                state_.last_rotate_kick_index = -1;
                if (is_mode_legacy()) {
                    result.reward += 1.0f;
                } else {
                    result.legacy_reward += 1.0f;
                }
            }
            break;
        case Action::HardDrop: {
            int dropped = 0;
            while (try_move(0, -1)) {
                ++dropped;
            }
            if (is_mode_legacy()) {
                result.reward += static_cast<float>(2 * dropped);
            } else {
                result.legacy_reward += static_cast<float>(2 * dropped);
            }
            action_succeeded = dropped > 0 || touching_ground();
            lock_active_piece(result);
            break;
        }
        case Action::Hold:
            action_succeeded = apply_hold();
            result.hold_used = action_succeeded;
            skip_gravity = true;
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

    if (!state_.game_over && !result.piece_locked && !skip_gravity) {
        state_.gravity_accumulator += config_.gravity_per_step;
        while (state_.gravity_accumulator >= 1.0f) {
            if (!try_move(0, -1)) {
                state_.gravity_accumulator = 0.0f;
                break;
            }
            state_.gravity_accumulator -= 1.0f;
        }
    }

    if (!state_.game_over && !result.piece_locked) {
        if (touching_ground()) {
            ++state_.lock_timer;
            if (state_.lock_timer >= config_.lock_delay_steps) {
                lock_active_piece(result);
            }
        } else {
            state_.lock_timer = 0;
            state_.lock_resets_used = 0;
        }
    }

    result.action_succeeded = action_succeeded;
    sync_state_to_result(result);
    if (is_mode_legacy()) {
        result.legacy_reward = result.reward;
    }
    last_step_result_ = result;
    return result;
}

void ModernTetrisEnv::invalidate_placement_cache() const {
    placement_cache_valid_ = false;
    ++placement_cache_epoch_;
}

const std::vector<PlacementOption>& ModernTetrisEnv::placement_options_cached() const {
    if (!placement_cache_valid_ || placement_cache_built_epoch_ != placement_cache_epoch_) {
        placement_cache_ = build_placement_options_uncached();
        placement_cache_built_epoch_ = placement_cache_epoch_;
        placement_cache_valid_ = true;
    }
    return placement_cache_;
}

std::vector<PlacementOption> ModernTetrisEnv::enumerate_active_piece_placements() const {
    return placement_options_cached();
}

const std::vector<PlacementOption>& ModernTetrisEnv::placement_options_view() const {
    return placement_options_cached();
}

std::vector<PlacementOption> ModernTetrisEnv::build_placement_options_uncached() const {
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
            board_after.clear_filled_lines();
            bool spin_candidate =
                (cleared > 0) && (current.piece.piece == Piece::T) && current.metadata.spin_eligible;
            bool difficult_candidate = (cleared == 4) || spin_candidate;
            PlacementOption candidate_option{
                current.piece,
                board_after,
                cleared_rows,
                cleared,
                current.metadata.spin_eligible,
                current.metadata.last_rotate_used_kick,
                current.metadata.last_rotate_kick_index,
                spin_candidate,
                difficult_candidate,
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

        auto cw = kicked_rotation_quick(current.piece, rotate_cw(current.piece.rotation));
        if (cw.has_value()) {
            const int kick_index = cw->second;
            try_enqueue(cw->first, PathMetadata{true, kick_index > 0, kick_index});
        }

        auto ccw = kicked_rotation_quick(current.piece, rotate_ccw(current.piece.rotation));
        if (ccw.has_value()) {
            const int kick_index = ccw->second;
            try_enqueue(ccw->first, PathMetadata{true, kick_index > 0, kick_index});
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

std::optional<PlacementOption> ModernTetrisEnv::placement_option_at(std::size_t index) const {
    const auto& options = placement_options_cached();
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
            if (id >= 0 && id <= 6) {
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
            if (id >= 0 && id <= 6) {
                out[static_cast<std::size_t>(row * Board::kWidth + x)] =
                    static_cast<std::uint8_t>(id);
            }
        }
    }
    return out;
}

StepResult ModernTetrisEnv::apply_option_impl(const PlacementOption& option) {
    refresh_runtime_state();
    StepResult result = make_result_defaults();
    if (state_.game_over) {
        sync_state_to_result(result);
        if (is_mode_legacy()) {
            result.legacy_reward = result.reward;
        }
        last_step_result_ = result;
        return result;
    }

    invalidate_placement_cache();
    state_.active = option.placement;
    state_.spin_eligible = option.spin_eligible_path;
    state_.rotated_this_piece = state_.rotated_this_piece || option.spin_eligible_path;
    state_.last_rotate_used_kick = option.last_rotate_used_kick_path;
    state_.last_rotate_kick_index = option.last_rotate_kick_index_path;
    lock_active_piece(result);
    result.action_succeeded = true;
    sync_state_to_result(result);
    if (is_mode_legacy()) {
        result.legacy_reward = result.reward;
    }
    last_step_result_ = result;
    return result;
}

StepResult ModernTetrisEnv::apply_placement(const ActivePiece& placement) {
    refresh_runtime_state();
    StepResult result = make_result_defaults();
    if (state_.game_over) {
        sync_state_to_result(result);
        if (is_mode_legacy()) {
            result.legacy_reward = result.reward;
        }
        last_step_result_ = result;
        return result;
    }

    const auto& options = placement_options_cached();
    auto it = std::find_if(options.begin(), options.end(), [&](const PlacementOption& opt) {
        return opt.placement == placement;
    });
    if (it == options.end()) {
        result.action_succeeded = false;
        sync_state_to_result(result);
        if (is_mode_legacy()) {
            result.legacy_reward = result.reward;
        }
        last_step_result_ = result;
        return result;
    }

    return apply_option_impl(*it);
}

StepResult ModernTetrisEnv::apply_placement_index(std::size_t index) {
    refresh_runtime_state();
    StepResult result = make_result_defaults();
    if (state_.game_over) {
        sync_state_to_result(result);
        if (is_mode_legacy()) {
            result.legacy_reward = result.reward;
        }
        last_step_result_ = result;
        return result;
    }

    auto option = placement_option_at(index);
    if (!option.has_value()) {
        result.action_succeeded = false;
        sync_state_to_result(result);
        if (is_mode_legacy()) {
            result.legacy_reward = result.reward;
        }
        last_step_result_ = result;
        return result;
    }

    return apply_option_impl(*option);
}

StepResult ModernTetrisEnv::apply_placement_option_fast(const PlacementOption& option) {
    return apply_option_impl(option);
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
    out.garbage_rng_state = garbage_rng_state_;
    out.mode = config_.mode;
    return out;
}

void ModernTetrisEnv::restore(const EnvSnapshot& snapshot) {
    state_ = snapshot.state;
    randomizer_ = snapshot.randomizer;
    garbage_rng_state_ = snapshot.garbage_rng_state;
    config_.mode = snapshot.mode;
    if (is_mode_blitz()) {
        if (blitz_time_limit_ms_ <= 0) {
            state_.blitz_timed_out = false;
            state_.blitz_time_remaining_ms = 0;
        }
        update_blitz_level_and_gravity();
        blitz_last_wall_time_ = std::chrono::steady_clock::now();
        blitz_clock_started_ = true;
    } else {
        config_.gravity_per_step = base_gravity_per_step_;
        blitz_clock_started_ = false;
    }
    last_step_result_ = make_result_defaults();
    invalidate_placement_cache();
}

void ModernTetrisEnv::restore(const EnvState& state) {
    state_ = state;
    if (is_mode_blitz()) {
        if (blitz_time_limit_ms_ <= 0) {
            state_.blitz_timed_out = false;
            state_.blitz_time_remaining_ms = 0;
        }
        update_blitz_level_and_gravity();
        blitz_last_wall_time_ = std::chrono::steady_clock::now();
        blitz_clock_started_ = true;
    } else {
        config_.gravity_per_step = base_gravity_per_step_;
        blitz_clock_started_ = false;
    }
    last_step_result_ = make_result_defaults();
    invalidate_placement_cache();
}

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
    invalidate_placement_cache();
    ensure_queue(static_cast<std::size_t>(config_.queue_size) + 1u);
    if (state_.queue.empty()) {
        state_.game_over = true;
        return;
    }

    auto next = state_.queue.front();
    state_.queue.pop_front();
    state_.active = spawn_piece(next);
    state_.lock_timer = 0;
    state_.lock_resets_used = 0;
    state_.gravity_accumulator = 0.0f;
    state_.spin_eligible = false;
    state_.rotated_this_piece = false;
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
    const auto cells = piece_cells(piece.piece, piece.rotation);
    const auto& rows = state_.board.rows();
    for (const auto& cell : cells) {
        const int x = piece.x + cell.x;
        const int y = piece.y + cell.y;
        if (x < 0 || x >= Board::kWidth || y < 0 || y >= Board::kRows) {
            return true;
        }
        if ((rows[static_cast<std::size_t>(y)] &
             (Board::RowMask{1u} << static_cast<unsigned>(x))) != 0) {
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
    invalidate_placement_cache();
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
    invalidate_placement_cache();
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

std::optional<ActivePiece> ModernTetrisEnv::kicked_rotation(
    const ActivePiece& from, Rotation target_rotation) const {
    return kicked_rotation_with_tests(from, target_rotation, 0, 0).first;
}

std::optional<std::pair<ActivePiece, int>> ModernTetrisEnv::kicked_rotation_quick(
    const ActivePiece& from, Rotation target_rotation) const {
    if (from.piece == Piece::None) {
        return std::nullopt;
    }

    const auto kicks = srs_kicks(from.piece, from.rotation, target_rotation);
    for (std::size_t i = 0; i < kicks.size(); ++i) {
        ActivePiece candidate = from;
        candidate.rotation = target_rotation;
        candidate.x += kicks[i].x;
        candidate.y += kicks[i].y;
        if (!collides(candidate)) {
            return std::make_pair(candidate, static_cast<int>(i));
        }
    }
    return std::nullopt;
}

std::optional<std::pair<ActivePiece, bool>> ModernTetrisEnv::kicked_rotate_180_with_kick(
    const ActivePiece& from) const {
    auto cw_first = kicked_rotation_quick(from, rotate_cw(from.rotation));
    if (cw_first.has_value()) {
        bool used_kick = cw_first->second > 0;
        auto cw_second = kicked_rotation_quick(cw_first->first, rotate_cw(cw_first->first.rotation));
        if (cw_second.has_value()) {
            used_kick = used_kick || (cw_second->second > 0);
            return std::make_pair(cw_second->first, used_kick);
        }
    }

    auto ccw_first = kicked_rotation_quick(from, rotate_ccw(from.rotation));
    if (ccw_first.has_value()) {
        bool used_kick = ccw_first->second > 0;
        auto ccw_second =
            kicked_rotation_quick(ccw_first->first, rotate_ccw(ccw_first->first.rotation));
        if (ccw_second.has_value()) {
            used_kick = used_kick || (ccw_second->second > 0);
            return std::make_pair(ccw_second->first, used_kick);
        }
    }

    return std::nullopt;
}

std::optional<ActivePiece> ModernTetrisEnv::kicked_rotate_180(const ActivePiece& from) const {
    auto rotated = kicked_rotate_180_with_kick(from);
    if (rotated.has_value()) {
        return rotated->first;
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

    invalidate_placement_cache();
    Piece current = state_.active.piece;
    if (state_.hold.has_value()) {
        Piece swapped = *state_.hold;
        state_.hold = current;
        state_.active = spawn_piece(swapped);
        state_.lock_timer = 0;
        state_.lock_resets_used = 0;
        state_.gravity_accumulator = 0.0f;
        state_.spin_eligible = false;
        state_.rotated_this_piece = false;
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
    return !state_.game_over;
}

bool ModernTetrisEnv::piece_immobile(const ActivePiece& piece) const {
    ActivePiece left = piece;
    left.x -= 1;
    ActivePiece right = piece;
    right.x += 1;
    ActivePiece down = piece;
    down.y -= 1;
    return collides(left) && collides(right) && collides(down);
}

int ModernTetrisEnv::apply_attack_rounding(float attack) const {
    if (!std::isfinite(attack) || attack <= 0.0f) {
        return 0;
    }
    switch (config_.attack.rounding_mode) {
        case AttackRoundingMode::Down:
        case AttackRoundingMode::Rng:
            return static_cast<int>(std::floor(attack));
    }
    return 0;
}

int ModernTetrisEnv::b2b_chaining_extra(int streak) const {
    if (streak <= 1) {
        return 0;
    }
    for (std::size_t i = 0; i < kB2BChainingUpperBounds.size(); ++i) {
        if (streak <= kB2BChainingUpperBounds[i]) {
            return static_cast<int>(i + 1);
        }
    }

    int extra = 8;
    int upper = kB2BChainingUpperBounds.back();
    int span = upper - kB2BChainingUpperBounds[kB2BChainingUpperBounds.size() - 2];
    while (streak > upper) {
        span = std::max(span + 1, static_cast<int>(std::llround(static_cast<double>(span) * 2.718281828)));
        upper += span;
        ++extra;
    }
    return extra;
}

int ModernTetrisEnv::versus_guideline_base_attack(int lines, SpinType spin_type) const {
    if (lines <= 0) {
        return 0;
    }

    if (spin_type == SpinType::Mini) {
        return 0;
    }
    if (spin_type == SpinType::Full) {
        switch (lines) {
            case 1: return 2;
            case 2: return 4;
            case 3: return 6;
            default: return 0;
        }
    }
    switch (lines) {
        case 1: return 0;
        case 2: return 1;
        case 3: return 2;
        case 4: return 4;
        default: return 0;
    }
}

int ModernTetrisEnv::versus_guideline_combo_bonus(int combo) const {
    if (combo <= 0) {
        return 0;
    }
    return std::max(0, combo / 2);
}

int ModernTetrisEnv::blitz_clear_points(int lines, SpinType spin_type) const {
    if (spin_type == SpinType::Full) {
        switch (lines) {
            case 0: return 400;
            case 1: return 800;
            case 2: return 1200;
            case 3: return 1600;
            case 4: return 2600;
            default: return 0;
        }
    }
    if (spin_type == SpinType::Mini) {
        switch (lines) {
            case 0: return 100;
            case 1: return 200;
            case 2: return 400;
            case 3: return 800;
            case 4: return 1600;
            default: return 0;
        }
    }

    switch (lines) {
        case 1: return 100;
        case 2: return 300;
        case 3: return 500;
        case 4: return 800;
        default: return 0;
    }
}

int ModernTetrisEnv::blitz_level_from_total_lines(int total_lines) const {
    int level = 1;
    for (std::size_t i = 0; i < kBlitzLevelLineTotals.size(); ++i) {
        if (total_lines >= kBlitzLevelLineTotals[i]) {
            level = static_cast<int>(i) + 2;
        } else {
            break;
        }
    }
    return std::min(level, kBlitzMaxLevel);
}

int ModernTetrisEnv::blitz_lines_to_next(int level, int total_lines) const {
    if (level >= kBlitzMaxLevel) {
        return 0;
    }
    const int goal = kBlitzLevelLineTotals[static_cast<std::size_t>(std::max(1, level) - 1)];
    return std::max(0, goal - total_lines);
}

void ModernTetrisEnv::update_blitz_level_and_gravity() {
    if (!is_mode_blitz()) {
        return;
    }
    state_.blitz_level = blitz_level_from_total_lines(state_.total_lines_cleared);
    state_.blitz_lines_to_next = blitz_lines_to_next(state_.blitz_level, state_.total_lines_cleared);

    const int idx = std::min(
        kBlitzMaxLevel - 1,
        std::max(0, state_.blitz_level - 1));
    const double sec_per_row = kBlitzGravitySecondsPerRow[static_cast<std::size_t>(idx)];
    double g = 0.0;
    if (std::isfinite(sec_per_row) && sec_per_row > 0.0) {
        g = 1.0 / sec_per_row;
    }
    config_.gravity_per_step = static_cast<float>(g / kBlitzTickRateHz);
}

void ModernTetrisEnv::reset_blitz_state() {
    state_.blitz_score_total = 0;
    state_.blitz_time_remaining_ms = (blitz_time_limit_ms_ > 0) ? blitz_time_limit_ms_ : 0;
    state_.blitz_timed_out = false;
    update_blitz_level_and_gravity();
    if (blitz_time_limit_ms_ > 0) {
        blitz_last_wall_time_ = std::chrono::steady_clock::now();
        blitz_clock_started_ = true;
    } else {
        blitz_clock_started_ = false;
    }
}

void ModernTetrisEnv::refresh_blitz_timer() {
    if (!is_mode_blitz()) {
        return;
    }
    if (blitz_time_limit_ms_ <= 0) {
        state_.blitz_timed_out = false;
        state_.blitz_time_remaining_ms = 0;
        return;
    }
    if (state_.blitz_timed_out) {
        state_.game_over = true;
        state_.top_out = false;
        return;
    }
    if (state_.game_over) {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    if (!blitz_clock_started_) {
        blitz_last_wall_time_ = now;
        blitz_clock_started_ = true;
    }

    const auto elapsed_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now - blitz_last_wall_time_).count());
    blitz_last_wall_time_ = now;
    if (elapsed_ms > 0) {
        state_.blitz_time_remaining_ms = std::max(0, state_.blitz_time_remaining_ms - elapsed_ms);
    }

    if (state_.blitz_time_remaining_ms <= 0) {
        state_.blitz_time_remaining_ms = 0;
        state_.blitz_timed_out = true;
        state_.game_over = true;
        state_.top_out = false;
    }
}

int ModernTetrisEnv::attack_base_for_clear(int lines, SpinType spin_type, bool b2b_active) const {
    if (lines <= 0) {
        return 0;
    }

    int base = 0;
    if (spin_type == SpinType::Full) {
        switch (lines) {
            case 1: base = 2; break;
            case 2: base = 4; break;
            case 3: base = 6; break;
            default: base = 0; break;
        }
    } else if (spin_type == SpinType::Mini) {
        switch (lines) {
            case 1: base = 0; break;
            case 2: base = 1; break;
            case 3: base = 2; break;
            default: base = 0; break;
        }
    } else {
        switch (lines) {
            case 1: base = 0; break;
            case 2: base = 1; break;
            case 3: base = 2; break;
            case 4: base = 4; break;
            default: base = 0; break;
        }
    }

    if (b2b_active && is_difficult_clear(lines, spin_type)) {
        base += 1;
    }
    return base;
}

bool ModernTetrisEnv::is_difficult_clear(int lines, SpinType spin_type) const {
    if (lines <= 0) {
        return false;
    }
    if (lines >= 4) {
        return true;
    }
    return spin_type != SpinType::None && lines >= 1;
}

bool ModernTetrisEnv::is_all_clear_after_line_clear() const { return state_.board.is_empty(); }

bool ModernTetrisEnv::apply_incoming_garbage(int lines, int* lines_applied) {
    refresh_runtime_state();
    if (lines_applied) {
        *lines_applied = 0;
    }
    if (lines <= 0 || state_.game_over) {
        return false;
    }

    invalidate_placement_cache();
    int applied = 0;
    for (int i = 0; i < lines; ++i) {
        garbage_rng_state_ = garbage_rng_state_ * 1664525u + 1013904223u;
        const int hole = static_cast<int>((garbage_rng_state_ >> 16u) % static_cast<std::uint32_t>(Board::kWidth));

        for (int y = Board::kRows - 1; y >= 1; --y) {
            state_.board.set_cell(0, y, false);
            auto src_mask = state_.board.row_mask(y - 1);
            for (int x = 0; x < Board::kWidth; ++x) {
                const bool filled = (src_mask & (Board::RowMask{1u} << static_cast<unsigned>(x))) != 0;
                state_.board.set_cell(x, y, filled);
                state_.piece_ids[static_cast<std::size_t>(y)][static_cast<std::size_t>(x)] =
                    state_.piece_ids[static_cast<std::size_t>(y - 1)][static_cast<std::size_t>(x)];
            }
        }

        for (int x = 0; x < Board::kWidth; ++x) {
            const bool filled = (x != hole);
            state_.board.set_cell(x, 0, filled);
            state_.piece_ids[0][static_cast<std::size_t>(x)] = -1;
        }
        ++applied;

        if (collides(state_.active)) {
            state_.game_over = true;
            state_.top_out = true;
            break;
        }
    }

    if (lines_applied) {
        *lines_applied = applied;
    }
    return state_.game_over;
}

void ModernTetrisEnv::lock_active_piece(StepResult& result) {
    invalidate_placement_cache();
    auto cells = piece_cells(state_.active.piece, state_.active.rotation);
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

    const bool immobile_lock = piece_immobile(state_.active);
    const bool rotated_before_lock = state_.rotated_this_piece;
    auto spin_type = SpinType::None;
    const bool allow_spin_detection = (lines > 0) || is_mode_blitz();
    if (allow_spin_detection && state_.active.piece == Piece::T && state_.spin_eligible) {
        auto occupied_corner = [&](int dx, int dy) {
            return state_.board.occupied(state_.active.x + dx, state_.active.y + dy);
        };
        const int corners =
            static_cast<int>(occupied_corner(-1, -1)) +
            static_cast<int>(occupied_corner(1, -1)) +
            static_cast<int>(occupied_corner(-1, 1)) +
            static_cast<int>(occupied_corner(1, 1));

        if (corners >= 3) {
            auto rotate = [&](int x, int y) -> Cell {
                switch (state_.active.rotation) {
                    case Rotation::North: return Cell{x, y};
                    case Rotation::East: return Cell{y, -x};
                    case Rotation::South: return Cell{-x, -y};
                    case Rotation::West: return Cell{-y, x};
                }
                return Cell{x, y};
            };
            auto c1 = rotate(-1, 1);
            auto c2 = rotate(1, 1);
            const int mini_corners =
                static_cast<int>(occupied_corner(c1.x, c1.y)) +
                static_cast<int>(occupied_corner(c2.x, c2.y));
            const bool full = (mini_corners == 2) || (state_.last_rotate_kick_index == 4);
            spin_type = full ? SpinType::Full : SpinType::Mini;
        }
    }

    if (lines > 0 && is_mode_tetrio_like() && config_.attack.all_mini_plus && rotated_before_lock &&
        immobile_lock) {
        if (state_.active.piece != Piece::T || spin_type == SpinType::None) {
            spin_type = SpinType::Mini;
        }
    }

    state_.board.clear_filled_lines();
    if (lines > 0) {
        compact_piece_ids(state_.piece_ids, cleared_rows);
    }

    const bool all_clear = (lines > 0) && is_all_clear_after_line_clear();
    const bool difficult = is_difficult_clear(lines, spin_type);

    result.piece_locked = true;
    result.lines_cleared = lines;
    result.spin_type = spin_type;
    result.spin_clear = result.spin_type != SpinType::None;
    result.difficult_clear = difficult;
    result.b2b_bonus_applied = false;
    result.all_clear = all_clear;
    result.rotated_before_lock = rotated_before_lock;
    result.immobile_lock = immobile_lock;
    result.attack_base = 0;
    result.attack_combo_scaled = 0.0f;
    result.combo_multiplier = 1.0f;
    result.attack_rounded = 0;
    result.attack_b2b_bonus = 0;
    result.attack_all_clear_bonus = 0;
    result.surge_release = 0;
    result.attack_total = 0;

    float legacy_gain = 0.0f;
    if (lines > 0) {
        legacy_gain += line_clear_reward(lines, result.spin_clear);
        if (difficult && state_.back_to_back) {
            legacy_gain += config_.scoring.b2b_bonus;
            result.b2b_bonus_applied = true;
        }
        state_.combo += 1;
        legacy_gain += combo_bonus(state_.combo);
    } else {
        state_.combo = -1;
    }

    if (is_mode_legacy()) {
        if (lines > 0) {
            if (difficult) {
                state_.back_to_back = true;
                state_.b2b_streak += 1;
            } else {
                state_.back_to_back = false;
                state_.b2b_streak = 0;
                state_.b2b_surge_charge = 0;
            }
        }
        result.reward += legacy_gain;
        result.legacy_reward = result.reward;
        result.surge_charge = state_.b2b_surge_charge;
    } else if (config_.mode == GameMode::Versus) {
        int next_streak = state_.b2b_streak;
        if (lines > 0) {
            if (difficult) {
                next_streak += 1;
            } else {
                next_streak = 0;
            }
        }

        if (lines > 0) {
            const bool b2b_active_before = state_.b2b_streak >= 1;
            result.attack_base = versus_guideline_base_attack(lines, spin_type);
            result.attack_combo_scaled = static_cast<float>(result.attack_base);
            result.attack_rounded = result.attack_base;

            if (difficult && b2b_active_before) {
                result.attack_b2b_bonus = 1;
            }
            if (all_clear) {
                result.attack_all_clear_bonus = 7;
            }
            const int combo_bonus = versus_guideline_combo_bonus(state_.combo);
            result.attack_total =
                result.attack_rounded +
                result.attack_b2b_bonus +
                result.attack_all_clear_bonus +
                combo_bonus;
        }

        state_.b2b_streak = next_streak;
        state_.b2b_surge_charge = 0;
        state_.back_to_back = state_.b2b_streak >= 1;
        result.surge_charge = 0;
        result.surge_release = 0;
        result.legacy_reward += legacy_gain;
        result.reward += static_cast<float>(result.attack_total);
        result.b2b_bonus_applied = result.attack_b2b_bonus > 0;
    } else if (is_mode_blitz()) {
        int next_streak = state_.b2b_streak;
        if (lines > 0) {
            if (difficult) {
                next_streak += 1;
            } else {
                next_streak = 0;
            }
        }

        int clear_points = blitz_clear_points(lines, spin_type);
        const bool b2b_active_before = state_.b2b_streak >= 1;
        if (lines > 0 && difficult && b2b_active_before && clear_points > 0) {
            clear_points = static_cast<int>(std::lround(static_cast<double>(clear_points) * 1.5));
            result.b2b_bonus_applied = true;
        }
        const int combo_points = (lines > 0) ? (std::max(0, state_.combo) * 50) : 0;
        const int all_clear_points = all_clear ? 3500 : 0;
        const int base_points = clear_points + combo_points + all_clear_points;
        const int level = std::max(1, state_.blitz_level);
        const int score_gain = base_points * level;

        state_.blitz_score_total += score_gain;
        state_.b2b_streak = next_streak;
        state_.back_to_back = state_.b2b_streak >= 1;
        state_.b2b_surge_charge = 0;
        result.surge_charge = 0;
        result.surge_release = 0;
        result.legacy_reward += legacy_gain;
        result.reward += static_cast<float>(score_gain);
    } else {
        int next_streak = state_.b2b_streak;
        int next_surge_charge = state_.b2b_surge_charge;
        bool streak_break = false;
        if (lines > 0) {
            if (all_clear) {
                next_streak += 2;
            } else if (difficult) {
                next_streak += 1;
            } else {
                next_streak = 0;
                streak_break = true;
            }
        }

        if (lines > 0) {
            const bool b2b_active_before = state_.b2b_streak >= 1;
            const int combo = std::max(0, state_.combo);
            result.combo_multiplier = 1.0f + 0.25f * static_cast<float>(combo);
            result.attack_base = attack_base_for_clear(lines, spin_type, b2b_active_before);
            if (result.attack_base > 0) {
                result.attack_combo_scaled = static_cast<float>(result.attack_base) * result.combo_multiplier;
            } else if (combo >= 2) {
                result.attack_combo_scaled =
                    std::log(1.0f + 1.25f * static_cast<float>(combo));
            } else {
                result.attack_combo_scaled = 0.0f;
            }
            result.attack_rounded = apply_attack_rounding(result.attack_combo_scaled);

            if (is_mode_charging()) {
                if (difficult) {
                    result.attack_b2b_bonus += 1;
                }
                if (streak_break && state_.b2b_surge_charge > 0) {
                    result.surge_release = state_.b2b_surge_charge;
                }
                if (streak_break) {
                    next_surge_charge = 0;
                } else if (next_streak >= config_.attack.b2b_charging_surge_start_streak) {
                    next_surge_charge = std::max(
                        config_.attack.b2b_charging_non_quickplay_base,
                        next_streak);
                } else {
                    next_surge_charge = 0;
                }
            } else if (is_mode_chaining()) {
                result.attack_b2b_bonus += b2b_chaining_extra(next_streak);
                next_surge_charge = 0;
            } else {
                next_surge_charge = 0;
            }

            if (all_clear) {
                result.attack_all_clear_bonus = config_.attack.all_clear_bonus;
            }
            result.attack_total =
                result.attack_rounded +
                result.attack_b2b_bonus +
                result.attack_all_clear_bonus +
                result.surge_release;
        }

        state_.b2b_streak = next_streak;
        state_.b2b_surge_charge = next_surge_charge;
        state_.back_to_back = state_.b2b_streak >= 1;
        result.surge_charge = state_.b2b_surge_charge;
        result.legacy_reward += legacy_gain;
        result.reward += static_cast<float>(result.attack_total);
        result.b2b_bonus_applied = result.attack_b2b_bonus > 0;
    }

    result.b2b_streak = state_.b2b_streak;
    state_.last_clear_spin = result.spin_clear;
    state_.last_clear_spin_type = result.spin_type;
    state_.last_clear_difficult = result.difficult_clear;
    state_.last_clear_b2b_bonus = result.b2b_bonus_applied;
    state_.last_clear_all_clear = result.all_clear;
    state_.last_attack_base = result.attack_base;
    state_.last_attack_combo_scaled = result.attack_combo_scaled;
    state_.last_attack_rounded = result.attack_rounded;
    state_.last_attack_b2b_bonus = result.attack_b2b_bonus;
    state_.last_attack_all_clear_bonus = result.attack_all_clear_bonus;
    state_.last_attack_surge_charge = result.surge_charge;
    state_.last_attack_surge_release = result.surge_release;
    state_.last_attack_total = result.attack_total;
    state_.total_lines_cleared += lines;
    if (is_mode_blitz()) {
        update_blitz_level_and_gravity();
    }
    spawn_next_piece(true);
    sync_state_to_result(result);
}

float ModernTetrisEnv::line_clear_reward(int lines, bool spin_clear) const {
    if (lines < 0 || lines > 4) {
        return 0.0f;
    }
    const auto& table = spin_clear ? config_.scoring.spin_clear_reward : config_.scoring.normal_clear_reward;
    return table[static_cast<std::size_t>(lines)];
}

float ModernTetrisEnv::combo_bonus(int combo) const {
    if (combo <= 0) {
        return 0.0f;
    }
    return static_cast<float>(combo) * config_.scoring.combo_unit_bonus;
}

}  // namespace tetris_v2
