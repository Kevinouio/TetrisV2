#include "tetris_v2/env.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <queue>
#include <sstream>
#include <unordered_set>

#include "tetris_v2/piece_defs.hpp"

namespace tetris_v2 {

namespace {

float combo_bonus(int combo) {
    if (combo <= 0) {
        return 0.0f;
    }
    return static_cast<float>(combo * 50);
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

StepResult ModernTetrisEnv::step(Action action) {
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

    switch (action) {
        case Action::Left:
            action_succeeded = try_move(-1, 0);
            moved_while_grounded = action_succeeded && touching_ground();
            break;
        case Action::Right:
            action_succeeded = try_move(1, 0);
            moved_while_grounded = action_succeeded && touching_ground();
            break;
        case Action::RotateCW:
            action_succeeded = try_rotate(rotate_cw(state_.active.rotation));
            moved_while_grounded = action_succeeded && touching_ground();
            break;
        case Action::RotateCCW:
            action_succeeded = try_rotate(rotate_ccw(state_.active.rotation));
            moved_while_grounded = action_succeeded && touching_ground();
            break;
        case Action::Rotate180:
            if (config_.allow_rotate_180) {
                auto rotated = kicked_rotate_180(state_.active);
                if (rotated.has_value()) {
                    state_.active = *rotated;
                    action_succeeded = true;
                }
                moved_while_grounded = action_succeeded && touching_ground();
            }
            break;
        case Action::SoftDrop:
            if (try_move(0, -1)) {
                action_succeeded = true;
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

    std::queue<ActivePiece> frontier;
    std::unordered_set<PoseKey, PoseKeyHash> visited;
    std::unordered_set<PoseKey, PoseKeyHash> locked_seen;

    frontier.push(state_.active);
    visited.insert(to_key(state_.active));

    while (!frontier.empty()) {
        ActivePiece current = frontier.front();
        frontier.pop();

        if (grounded(current)) {
            auto key = to_key(current);
            if (locked_seen.insert(key).second) {
                Board board_after = state_.board;
                auto cells = piece_cells(current.piece, current.rotation);
                for (const auto& cell : cells) {
                    board_after.set_cell(current.x + cell.x, current.y + cell.y, true);
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
                options.push_back(PlacementOption{current, board_after, cleared_rows, cleared});
            }
        }

        auto try_enqueue = [&](const ActivePiece& candidate) {
            if (collides(candidate)) {
                return;
            }
            auto [_, inserted] = visited.insert(to_key(candidate));
            if (inserted) {
                frontier.push(candidate);
            }
        };

        ActivePiece left = current;
        left.x -= 1;
        try_enqueue(left);

        ActivePiece right = current;
        right.x += 1;
        try_enqueue(right);

        ActivePiece down = current;
        down.y -= 1;
        try_enqueue(down);

        if (auto cw = kicked_rotation(current, rotate_cw(current.rotation))) {
            try_enqueue(*cw);
        }

        if (auto ccw = kicked_rotation(current, rotate_ccw(current.rotation))) {
            try_enqueue(*ccw);
        }

        if (config_.allow_rotate_180) {
            if (auto r180 = kicked_rotate_180(current)) {
                try_enqueue(*r180);
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

    state_.active = placement;
    lock_active_piece(result);
    result.action_succeeded = true;
    result.combo = state_.combo;
    result.back_to_back = state_.back_to_back;
    result.game_over = state_.game_over;
    result.top_out = state_.top_out;
    return result;
}

StepResult ModernTetrisEnv::apply_placement_index(std::size_t index) {
    auto option = placement_option_at(index);
    if (!option.has_value()) {
        StepResult result{};
        result.action_succeeded = false;
        result.combo = state_.combo;
        result.back_to_back = state_.back_to_back;
        result.game_over = state_.game_over;
        result.top_out = state_.top_out;
        return result;
    }
    return apply_placement(option->placement);
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

EnvState ModernTetrisEnv::snapshot() const { return state_; }

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
    state_.lock_timer = 0;
    state_.lock_resets_used = 0;
    state_.gravity_accumulator = 0.0f;
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

bool ModernTetrisEnv::try_rotate(Rotation target_rotation) {
    auto rotated = kicked_rotation(state_.active, target_rotation);
    if (!rotated.has_value()) {
        return false;
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

std::optional<ActivePiece> ModernTetrisEnv::kicked_rotation(
    const ActivePiece& from, Rotation target_rotation) const {
    return kicked_rotation_with_tests(from, target_rotation, 0, 0).first;
}

std::optional<ActivePiece> ModernTetrisEnv::kicked_rotate_180(const ActivePiece& from) const {
    if (auto first = kicked_rotation(from, rotate_cw(from.rotation))) {
        if (auto second = kicked_rotation(*first, rotate_cw(first->rotation))) {
            return second;
        }
    }
    if (auto first = kicked_rotation(from, rotate_ccw(from.rotation))) {
        if (auto second = kicked_rotation(*first, rotate_ccw(first->rotation))) {
            return second;
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
        state_.lock_timer = 0;
        state_.lock_resets_used = 0;
        state_.gravity_accumulator = 0.0f;
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

void ModernTetrisEnv::lock_active_piece(StepResult& result) {
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
    state_.board.clear_filled_lines();
    if (lines > 0) {
        compact_piece_ids(state_.piece_ids, cleared_rows);
    }
    result.piece_locked = true;
    result.lines_cleared = lines;
    result.reward += line_clear_reward(lines);

    if (lines > 0) {
        state_.combo += 1;
        bool hard_clear = (lines == 4);
        if (hard_clear && state_.back_to_back) {
            result.reward += 100.0f;
        }
        if (hard_clear) {
            state_.back_to_back = true;
        } else {
            state_.back_to_back = false;
        }
        result.reward += combo_bonus(state_.combo);
    } else {
        state_.combo = -1;
    }

    state_.total_lines_cleared += lines;
    spawn_next_piece(true);
}

float ModernTetrisEnv::line_clear_reward(int lines) const {
    switch (lines) {
        case 1: return 100.0f;
        case 2: return 300.0f;
        case 3: return 500.0f;
        case 4: return 800.0f;
        default: return 0.0f;
    }
}

}  // namespace tetris_v2
