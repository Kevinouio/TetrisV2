#include "tetris_v2/c_api.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <optional>
#include <vector>

#include "tetris_v2/cc_bot.hpp"
#include "tetris_v2/cc_env.hpp"
#include "tetris_v2/observation.hpp"
#include "tetris_v2/piece_defs.hpp"

namespace tetris_v2::depth {

class Bot;
Bot* create_default();
void destroy(Bot* bot);
bool sync_from_env(Bot* bot, const ModernTetrisEnv& env);
bool set_config(
    Bot* bot,
    int depth,
    double gamma,
    bool deduplicate_successors,
    bool use_transposition_table,
    bool collect_debug_info,
    std::uint64_t max_nodes);
bool choose(
    Bot* bot,
    int think_ms,
    bool* use_hold_out,
    std::size_t* placement_index_out,
    float* score_out,
    std::uint64_t* nodes_out,
    double* think_ms_out,
    double* nps_out,
    int* budget_miss_out);
bool apply_last_choice(
    Bot* bot,
    ModernTetrisEnv& env,
    StepResult* result_out,
    int* used_hold_out,
    std::size_t* placement_index_out);
bool choose_and_apply(
    Bot* bot,
    ModernTetrisEnv& env,
    int think_ms,
    StepResult* result_out,
    bool* use_hold_out,
    std::size_t* placement_index_out,
    float* score_out,
    std::uint64_t* nodes_out,
    double* think_ms_out,
    double* nps_out,
    int* budget_miss_out,
    int* used_hold_out);

}  // namespace tetris_v2::depth

namespace tetris_v2::beam {

class Bot;
Bot* create_default();
void destroy(Bot* bot);
bool sync_from_env(Bot* bot, const ModernTetrisEnv& env);
bool set_config(
    Bot* bot,
    int depth,
    int beam_width,
    double gamma,
    bool deduplicate_successors,
    bool use_transposition_table,
    bool collect_debug_info,
    std::uint64_t max_nodes);
bool choose(
    Bot* bot,
    int think_ms,
    bool* use_hold_out,
    std::size_t* placement_index_out,
    float* score_out,
    std::uint64_t* nodes_out,
    double* think_ms_out,
    double* nps_out,
    int* budget_miss_out);
bool apply_last_choice(
    Bot* bot,
    ModernTetrisEnv& env,
    StepResult* result_out,
    int* used_hold_out,
    std::size_t* placement_index_out);
bool choose_and_apply(
    Bot* bot,
    ModernTetrisEnv& env,
    int think_ms,
    StepResult* result_out,
    bool* use_hold_out,
    std::size_t* placement_index_out,
    float* score_out,
    std::uint64_t* nodes_out,
    double* think_ms_out,
    double* nps_out,
    int* budget_miss_out,
    int* used_hold_out);

}  // namespace tetris_v2::beam

struct tetris_cc_env_handle {
    explicit tetris_cc_env_handle(std::uint32_t seed) : env([seed] {
        tetris_v2::EnvConfig cfg{};
        cfg.seed = seed;
        return cfg;
    }()) {}

    struct CandidateCacheEntry {
        int use_hold{0};
        std::size_t placement_index{0};
        int piece{static_cast<int>(tetris_v2::Piece::None)};
        int rotation{0};
        int x{0};
        int y{0};
        int lines_cleared{0};
    };

    tetris_v2::cc::Env env;
    std::uint64_t mutation_epoch{1};
    mutable std::uint64_t candidate_cache_epoch{0};
    mutable bool candidate_cache_valid{false};
    mutable std::vector<CandidateCacheEntry> candidate_cache{};
    mutable std::vector<float> candidate_features_cache{};
};

struct tetris_cc_bot_handle {
    tetris_v2::cc::Bot cold_clear_bot;
    tetris_v2::depth::Bot* depth_bot{tetris_v2::depth::create_default()};
    tetris_v2::beam::Bot* beam_bot{tetris_v2::beam::create_default()};
    int backend{TETRIS_CC_BOT_BACKEND_COLD_CLEAR};

    ~tetris_cc_bot_handle() {
        tetris_v2::depth::destroy(depth_bot);
        tetris_v2::beam::destroy(beam_bot);
    }
};

struct tetris_cc_snapshot_handle {
    tetris_v2::EnvSnapshot snapshot{};
};

namespace {

constexpr std::size_t kCandidateFeatureDim = 6;

bool valid_bot_backend(int backend) {
    return backend == TETRIS_CC_BOT_BACKEND_COLD_CLEAR || backend == TETRIS_CC_BOT_BACKEND_DEPTH ||
           backend == TETRIS_CC_BOT_BACKEND_BEAM;
}

bool valid_env_mode(int mode) {
    return mode == TETRIS_CC_MODE_LEGACY || mode == TETRIS_CC_MODE_ZEN ||
           mode == TETRIS_CC_MODE_SCORING || mode == TETRIS_CC_MODE_VERSUS;
}

tetris_v2::GameMode parse_env_mode(int mode) {
    switch (mode) {
        case TETRIS_CC_MODE_ZEN: return tetris_v2::GameMode::Zen;
        case TETRIS_CC_MODE_SCORING: return tetris_v2::GameMode::Scoring;
        case TETRIS_CC_MODE_VERSUS: return tetris_v2::GameMode::Versus;
        case TETRIS_CC_MODE_LEGACY:
        default: return tetris_v2::GameMode::Legacy;
    }
}

int env_mode_to_c(tetris_v2::GameMode mode) {
    switch (mode) {
        case tetris_v2::GameMode::Legacy: return TETRIS_CC_MODE_LEGACY;
        case tetris_v2::GameMode::Zen: return TETRIS_CC_MODE_ZEN;
        case tetris_v2::GameMode::Scoring: return TETRIS_CC_MODE_SCORING;
        case tetris_v2::GameMode::Versus: return TETRIS_CC_MODE_VERSUS;
    }
    return TETRIS_CC_MODE_LEGACY;
}

tetris_v2::Action parse_action(int action) {
    if (action < static_cast<int>(tetris_v2::Action::None) ||
        action > static_cast<int>(tetris_v2::Action::Hold)) {
        return tetris_v2::Action::None;
    }
    return static_cast<tetris_v2::Action>(action);
}

std::optional<tetris_v2::Action> parse_rotate_action(int action) {
    auto parsed = parse_action(action);
    if (parsed == tetris_v2::Action::RotateCW || parsed == tetris_v2::Action::RotateCCW) {
        return parsed;
    }
    return std::nullopt;
}

size_t write_visible_board(
    const tetris_v2::Board& board,
    const std::optional<tetris_v2::ActivePiece>& active,
    uint8_t* out,
    size_t out_len) {
    if (!out) {
        return 0;
    }
    constexpr int kRows = tetris_v2::Board::kVisibleRows;
    constexpr int kCols = tetris_v2::Board::kWidth;
    constexpr size_t kTotal = static_cast<size_t>(kRows * kCols);

    std::array<uint8_t, kTotal> cells{};
    cells.fill(0);

    for (int row = 0; row < kRows; ++row) {
        int y = (kRows - 1) - row;
        auto mask = board.row_mask(y);
        for (int x = 0; x < kCols; ++x) {
            if (mask & (1u << x)) {
                cells[static_cast<size_t>(row * kCols + x)] = 1;
            }
        }
    }

    if (active.has_value()) {
        auto piece_cells = tetris_v2::piece_cells(active->piece, active->rotation);
        for (const auto& c : piece_cells) {
            int x = active->x + c.x;
            int y = active->y + c.y;
            if (x < 0 || x >= kCols || y < 0 || y >= kRows) {
                continue;
            }
            int row = (kRows - 1) - y;
            cells[static_cast<size_t>(row * kCols + x)] = 1;
        }
    }

    auto n = std::min(out_len, kTotal);
    std::copy(cells.begin(), cells.begin() + static_cast<std::ptrdiff_t>(n), out);
    return n;
}

void maybe_set_int(int* dst, int value) {
    if (dst) {
        *dst = value;
    }
}

void maybe_set_u64(std::uint64_t* dst, std::uint64_t value) {
    if (dst) {
        *dst = value;
    }
}

void maybe_set_float(float* dst, float value) {
    if (dst) {
        *dst = value;
    }
}

void maybe_set_double(double* dst, double value) {
    if (dst) {
        *dst = value;
    }
}

void refresh_env_runtime(tetris_cc_env_handle* handle) {
    if (!handle) {
        return;
    }
    handle->env.raw().refresh_runtime_state();
}

void refresh_env_runtime(const tetris_cc_env_handle* handle) {
    if (!handle) {
        return;
    }
    auto* mutable_handle = const_cast<tetris_cc_env_handle*>(handle);
    mutable_handle->env.raw().refresh_runtime_state();
}

void mark_env_mutated(tetris_cc_env_handle* handle) {
    if (!handle) {
        return;
    }
    if (handle->mutation_epoch == std::numeric_limits<std::uint64_t>::max()) {
        handle->mutation_epoch = 1;
    } else {
        ++handle->mutation_epoch;
    }
    handle->candidate_cache_valid = false;
}

bool active_piece_supported(const tetris_v2::EnvState& state) {
    auto piece = static_cast<int>(state.active.piece);
    return piece >= 0 && piece <= 6;
}

void append_candidate_features(
    const tetris_v2::Board& board_after_lock,
    int y_pos,
    int lines_removed,
    std::vector<float>& out) {
    std::array<int, tetris_v2::Board::kWidth> cols{};
    int holes = 0;
    int bumpiness = 0;

    for (int c = 0; c < tetris_v2::Board::kWidth; ++c) {
        bool block_seen = false;
        for (int r = 0; r < tetris_v2::Board::kVisibleRows; ++r) {
            const int y = (tetris_v2::Board::kVisibleRows - 1) - r;
            const bool filled = board_after_lock.occupied(c, y);
            if (filled && !block_seen) {
                block_seen = true;
                cols[static_cast<std::size_t>(c)] = tetris_v2::Board::kVisibleRows - r;
            }
            if (!filled && block_seen) {
                ++holes;
            }
        }
        if (c > 0) {
            bumpiness += std::abs(
                cols[static_cast<std::size_t>(c)] - cols[static_cast<std::size_t>(c - 1)]);
        }
    }

    int total_height = 0;
    for (int v : cols) {
        total_height += v;
    }

    int pillar = 0;
    for (int i = 1; i < tetris_v2::Board::kWidth - 1; ++i) {
        if ((cols[static_cast<std::size_t>(i - 1)] - cols[static_cast<std::size_t>(i)] >= 3) &&
            (cols[static_cast<std::size_t>(i + 1)] - cols[static_cast<std::size_t>(i)] >= 3)) {
            pillar = 1;
            break;
        }
    }
    if (pillar == 0) {
        if ((cols[1] - cols[0] >= 3) ||
            (cols[tetris_v2::Board::kWidth - 2] - cols[tetris_v2::Board::kWidth - 1] >= 3)) {
            pillar = 1;
        }
    }

    out.push_back(static_cast<float>(total_height));
    out.push_back(static_cast<float>(bumpiness));
    out.push_back(static_cast<float>(lines_removed));
    out.push_back(static_cast<float>(holes));
    out.push_back(static_cast<float>(y_pos));
    out.push_back(static_cast<float>(pillar));
}

void append_candidate_branch(tetris_cc_env_handle* handle, bool use_hold) {
    const auto& state = handle->env.state();
    if (state.game_over || !active_piece_supported(state)) {
        return;
    }

    const int piece = static_cast<int>(state.active.piece);
    auto placements = handle->env.enumerate_active_piece_placements();
    handle->candidate_cache.reserve(handle->candidate_cache.size() + placements.size());
    handle->candidate_features_cache.reserve(
        handle->candidate_features_cache.size() + placements.size() * kCandidateFeatureDim);
    for (std::size_t idx = 0; idx < placements.size(); ++idx) {
        const auto& option = placements[idx];
        tetris_cc_env_handle::CandidateCacheEntry row{};
        row.use_hold = use_hold ? 1 : 0;
        row.placement_index = idx;
        row.piece = piece;
        row.rotation = static_cast<int>(option.placement.rotation);
        row.x = option.placement.x;
        row.y = option.placement.y;
        row.lines_cleared = option.lines_cleared;
        handle->candidate_cache.push_back(row);
        append_candidate_features(
            option.board_after_lock,
            option.placement.y,
            option.lines_cleared,
            handle->candidate_features_cache);
    }
}

void ensure_candidate_cache(const tetris_cc_env_handle* handle) {
    if (!handle) {
        return;
    }
    auto* mutable_handle = const_cast<tetris_cc_env_handle*>(handle);
    refresh_env_runtime(mutable_handle);
    if (mutable_handle->candidate_cache_valid &&
        mutable_handle->candidate_cache_epoch == mutable_handle->mutation_epoch) {
        return;
    }

    mutable_handle->candidate_cache.clear();
    mutable_handle->candidate_features_cache.clear();

    const auto snapshot = mutable_handle->env.snapshot();
    append_candidate_branch(mutable_handle, false);

    const auto hold_available = mutable_handle->env.state().hold_available;
    if (hold_available) {
        auto hold_result = mutable_handle->env.step(tetris_v2::Action::Hold);
        if (hold_result.action_succeeded) {
            append_candidate_branch(mutable_handle, true);
        }
        mutable_handle->env.restore(snapshot);
    }

    mutable_handle->candidate_cache_epoch = mutable_handle->mutation_epoch;
    mutable_handle->candidate_cache_valid = true;
}

}  // namespace

extern "C" {

tetris_cc_env_handle* tetris_cc_env_create(uint32_t seed) {
    try {
        return new tetris_cc_env_handle(seed);
    } catch (...) {
        return nullptr;
    }
}

void tetris_cc_env_destroy(tetris_cc_env_handle* handle) { delete handle; }

void tetris_cc_env_reset(tetris_cc_env_handle* handle, uint32_t seed) {
    if (!handle) {
        return;
    }
    handle->env.reset(seed);
    mark_env_mutated(handle);
}

int tetris_cc_env_set_mode(tetris_cc_env_handle* handle, int mode) {
    if (!handle || !valid_env_mode(mode)) {
        return 0;
    }
    handle->env.raw().set_mode(parse_env_mode(mode));
    mark_env_mutated(handle);
    return 1;
}

int tetris_cc_env_get_mode(const tetris_cc_env_handle* handle, int* mode_out) {
    if (!handle || !mode_out) {
        return 0;
    }
    *mode_out = env_mode_to_c(handle->env.raw().mode());
    return 1;
}

int tetris_cc_env_step(tetris_cc_env_handle* handle, int action, float* reward_out) {
    if (!handle) {
        return 1;
    }
    auto result = handle->env.step(parse_action(action));
    mark_env_mutated(handle);
    if (reward_out) {
        *reward_out = result.reward;
    }
    return result.game_over ? 1 : 0;
}

int tetris_cc_env_hold(tetris_cc_env_handle* handle, float* reward_out) {
    if (!handle) {
        return 0;
    }
    auto result = handle->env.step(tetris_v2::Action::Hold);
    if (result.action_succeeded) {
        mark_env_mutated(handle);
    }
    if (reward_out) {
        *reward_out = result.reward;
    }
    return result.action_succeeded ? 1 : 0;
}

int tetris_cc_env_apply_incoming_garbage(
    tetris_cc_env_handle* handle, int lines, int* lines_applied_out, int* top_out_out) {
    if (!handle) {
        return 0;
    }
    int applied = 0;
    const bool top_out = handle->env.raw().apply_incoming_garbage(lines, &applied);
    mark_env_mutated(handle);
    maybe_set_int(lines_applied_out, applied);
    maybe_set_int(top_out_out, top_out ? 1 : 0);
    return 1;
}

tetris_cc_snapshot_handle* tetris_cc_env_snapshot_create(const tetris_cc_env_handle* handle) {
    if (!handle) {
        return nullptr;
    }
    try {
        auto* out = new tetris_cc_snapshot_handle{};
        out->snapshot = handle->env.snapshot();
        return out;
    } catch (...) {
        return nullptr;
    }
}

void tetris_cc_snapshot_destroy(tetris_cc_snapshot_handle* snapshot) { delete snapshot; }

int tetris_cc_env_restore_snapshot(tetris_cc_env_handle* handle, const tetris_cc_snapshot_handle* snapshot) {
    if (!handle || !snapshot) {
        return 0;
    }
    handle->env.restore(snapshot->snapshot);
    mark_env_mutated(handle);
    return 1;
}

size_t tetris_cc_env_observation_size(const tetris_cc_env_handle* handle, int include_hidden_rows) {
    if (!handle) {
        return 0;
    }
    refresh_env_runtime(handle);
    return tetris_v2::observation_size(
        handle->env.config().queue_size, include_hidden_rows != 0);
}

size_t tetris_cc_env_observation_write(
    const tetris_cc_env_handle* handle, int include_hidden_rows, float* out, size_t out_len) {
    if (!handle || !out) {
        return 0;
    }
    refresh_env_runtime(handle);
    auto obs = tetris_v2::encode_observation(
        handle->env.state(), handle->env.config().queue_size, include_hidden_rows != 0);
    auto n = std::min(out_len, obs.size());
    std::copy(obs.begin(), obs.begin() + static_cast<std::ptrdiff_t>(n), out);
    return n;
}

size_t tetris_cc_env_board_write(
    const tetris_cc_env_handle* handle, int include_active, uint8_t* out, size_t out_len) {
    if (!handle || !out) {
        return 0;
    }
    refresh_env_runtime(handle);
    std::optional<tetris_v2::ActivePiece> active{};
    if (include_active != 0) {
        active = handle->env.state().active;
    }
    return write_visible_board(handle->env.state().board, active, out, out_len);
}

size_t tetris_cc_env_board_piece_ids_write(
    const tetris_cc_env_handle* handle, int include_active, uint8_t* out, size_t out_len) {
    if (!handle || !out) {
        return 0;
    }
    refresh_env_runtime(handle);
    auto ids = handle->env.visible_board_piece_ids(include_active != 0);
    auto n = std::min(out_len, ids.size());
    std::copy(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(n), out);
    return n;
}

size_t tetris_cc_env_visible_garbage_count(const tetris_cc_env_handle* handle) {
    if (!handle) {
        return 0;
    }
    refresh_env_runtime(handle);
    constexpr int kRows = tetris_v2::Board::kVisibleRows;
    constexpr int kCols = tetris_v2::Board::kWidth;
    std::size_t count = 0;
    const auto& state = handle->env.state();
    for (int y = 0; y < kRows; ++y) {
        for (int x = 0; x < kCols; ++x) {
            if (!state.board.occupied(x, y)) {
                continue;
            }
            if (state.piece_ids[static_cast<std::size_t>(y)][static_cast<std::size_t>(x)] == -1) {
                ++count;
            }
        }
    }
    return count;
}

int tetris_cc_env_set_visible_board_mask(
    tetris_cc_env_handle* handle, const uint8_t* cells, size_t cells_len, int reset_meta) {
    if (!handle || !cells) {
        return 0;
    }
    constexpr int kRows = tetris_v2::Board::kVisibleRows;
    constexpr int kCols = tetris_v2::Board::kWidth;
    constexpr size_t kTotal = static_cast<size_t>(kRows * kCols);
    if (cells_len < kTotal) {
        return 0;
    }

    auto state = handle->env.state();
    for (int row = 0; row < kRows; ++row) {
        int y = (kRows - 1) - row;
        for (int x = 0; x < kCols; ++x) {
            auto idx = static_cast<size_t>(row * kCols + x);
            const bool filled = cells[idx] != 0;
            state.board.set_cell(x, y, filled);
            state.piece_ids[static_cast<size_t>(y)][static_cast<size_t>(x)] = -1;
        }
    }

    if (reset_meta != 0) {
        state.game_over = false;
        state.top_out = false;
        state.combo = -1;
        state.back_to_back = false;
        state.b2b_streak = 0;
        state.b2b_surge_charge = 0;
        state.total_lines_cleared = 0;
        state.lock_timer = 0;
        state.lock_resets_used = 0;
        state.gravity_accumulator = 0.0f;
        state.spin_eligible = false;
        state.rotated_this_piece = false;
        state.last_rotate_used_kick = false;
        state.last_rotate_kick_index = -1;
        state.last_clear_spin = false;
        state.last_clear_spin_type = tetris_v2::SpinType::None;
        state.last_clear_difficult = false;
        state.last_clear_b2b_bonus = false;
        state.last_clear_all_clear = false;
        state.last_attack_base = 0;
        state.last_attack_combo_scaled = 0.0f;
        state.last_attack_rounded = 0;
        state.last_attack_b2b_bonus = 0;
        state.last_attack_all_clear_bonus = 0;
        state.last_attack_surge_charge = 0;
        state.last_attack_surge_release = 0;
        state.last_attack_total = 0;
        state.blitz_score_total = 0;
        state.blitz_level = 1;
        state.blitz_lines_to_next =
            (handle->env.raw().mode() == tetris_v2::GameMode::Scoring) ? 3 : 0;
        state.blitz_time_remaining_ms =
            (handle->env.raw().mode() == tetris_v2::GameMode::Scoring) ? 120000 : 0;
        state.blitz_timed_out = false;
    }

    handle->env.restore(state);
    mark_env_mutated(handle);
    return 1;
}

int tetris_cc_env_active_piece(
    const tetris_cc_env_handle* handle, int* piece, int* rotation, int* x, int* y) {
    if (!handle) {
        return 0;
    }
    refresh_env_runtime(handle);
    const auto& a = handle->env.state().active;
    maybe_set_int(piece, static_cast<int>(a.piece));
    maybe_set_int(rotation, static_cast<int>(a.rotation));
    maybe_set_int(x, a.x);
    maybe_set_int(y, a.y);
    return 1;
}

int tetris_cc_env_hold_piece(
    const tetris_cc_env_handle* handle, int* has_hold, int* hold_piece, int* hold_available) {
    if (!handle) {
        return 0;
    }
    refresh_env_runtime(handle);
    const auto& state = handle->env.state();
    maybe_set_int(has_hold, state.hold.has_value() ? 1 : 0);
    maybe_set_int(
        hold_piece, static_cast<int>(state.hold.has_value() ? *state.hold : tetris_v2::Piece::None));
    maybe_set_int(hold_available, state.hold_available ? 1 : 0);
    return 1;
}

size_t tetris_cc_env_queue_count(const tetris_cc_env_handle* handle) {
    if (!handle) {
        return 0;
    }
    refresh_env_runtime(handle);
    return handle->env.state().queue.size();
}

int tetris_cc_env_queue_get(const tetris_cc_env_handle* handle, size_t index, int* piece) {
    if (!handle || !piece) {
        return 0;
    }
    refresh_env_runtime(handle);
    const auto& queue = handle->env.state().queue;
    if (index >= queue.size()) {
        return 0;
    }
    *piece = static_cast<int>(queue[index]);
    return 1;
}

int tetris_cc_env_meta(
    const tetris_cc_env_handle* handle,
    int* game_over,
    int* top_out,
    int* combo,
    int* back_to_back,
    int* total_lines_cleared,
    int* lock_timer,
    int* lock_resets_used) {
    if (!handle) {
        return 0;
    }
    refresh_env_runtime(handle);
    const auto& state = handle->env.state();
    maybe_set_int(game_over, state.game_over ? 1 : 0);
    maybe_set_int(top_out, state.top_out ? 1 : 0);
    maybe_set_int(combo, state.combo);
    maybe_set_int(back_to_back, state.back_to_back ? 1 : 0);
    maybe_set_int(total_lines_cleared, state.total_lines_cleared);
    maybe_set_int(lock_timer, state.lock_timer);
    maybe_set_int(lock_resets_used, state.lock_resets_used);
    return 1;
}

size_t tetris_cc_env_placement_count(const tetris_cc_env_handle* handle) {
    if (!handle) {
        return 0;
    }
    refresh_env_runtime(handle);
    return handle->env.enumerate_active_piece_placements().size();
}

int tetris_cc_env_placement_get(
    const tetris_cc_env_handle* handle, size_t index, int* x, int* y, int* rotation, int* lines_cleared) {
    if (!handle) {
        return 0;
    }
    refresh_env_runtime(handle);
    auto option = handle->env.placement_option_at(index);
    if (!option.has_value()) {
        return 0;
    }
    maybe_set_int(x, option->placement.x);
    maybe_set_int(y, option->placement.y);
    maybe_set_int(rotation, static_cast<int>(option->placement.rotation));
    maybe_set_int(lines_cleared, option->lines_cleared);
    return 1;
}

int tetris_cc_env_placement_get_ex(
    const tetris_cc_env_handle* handle,
    size_t index,
    int* x,
    int* y,
    int* rotation,
    int* lines_cleared,
    int* spin_candidate,
    int* difficult_candidate,
    int* last_rotate_used_kick) {
    if (!handle) {
        return 0;
    }
    refresh_env_runtime(handle);
    auto option = handle->env.placement_option_at(index);
    if (!option.has_value()) {
        return 0;
    }
    maybe_set_int(x, option->placement.x);
    maybe_set_int(y, option->placement.y);
    maybe_set_int(rotation, static_cast<int>(option->placement.rotation));
    maybe_set_int(lines_cleared, option->lines_cleared);
    maybe_set_int(spin_candidate, option->spin_clear_candidate ? 1 : 0);
    maybe_set_int(difficult_candidate, option->difficult_clear_candidate ? 1 : 0);
    maybe_set_int(last_rotate_used_kick, option->last_rotate_used_kick_path ? 1 : 0);
    return 1;
}

size_t tetris_cc_env_placement_board_write(
    const tetris_cc_env_handle* handle, size_t index, uint8_t* out, size_t out_len) {
    if (!handle || !out) {
        return 0;
    }
    refresh_env_runtime(handle);
    auto option = handle->env.placement_option_at(index);
    if (!option.has_value()) {
        return 0;
    }
    return write_visible_board(option->board_after_lock, std::nullopt, out, out_len);
}

size_t tetris_cc_env_placement_board_piece_ids_write(
    const tetris_cc_env_handle* handle, size_t index, uint8_t* out, size_t out_len) {
    if (!handle || !out) {
        return 0;
    }
    refresh_env_runtime(handle);
    auto option = handle->env.placement_option_at(index);
    if (!option.has_value()) {
        return 0;
    }
    auto ids = handle->env.visible_placement_piece_ids(index);
    auto n = std::min(out_len, ids.size());
    std::copy(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(n), out);
    return n;
}

size_t tetris_cc_env_candidate_count(const tetris_cc_env_handle* handle) {
    if (!handle) {
        return 0;
    }
    ensure_candidate_cache(handle);
    return handle->candidate_cache.size();
}

int tetris_cc_env_candidate_get(
    const tetris_cc_env_handle* handle,
    size_t index,
    int* use_hold,
    size_t* placement_index,
    int* piece,
    int* rotation,
    int* x,
    int* y,
    int* lines_cleared) {
    if (!handle) {
        return 0;
    }
    ensure_candidate_cache(handle);
    if (index >= handle->candidate_cache.size()) {
        return 0;
    }
    const auto& row = handle->candidate_cache[index];
    maybe_set_int(use_hold, row.use_hold);
    if (placement_index) {
        *placement_index = row.placement_index;
    }
    maybe_set_int(piece, row.piece);
    maybe_set_int(rotation, row.rotation);
    maybe_set_int(x, row.x);
    maybe_set_int(y, row.y);
    maybe_set_int(lines_cleared, row.lines_cleared);
    return 1;
}

size_t tetris_cc_env_candidate_features_write(
    const tetris_cc_env_handle* handle, float* out, size_t out_len) {
    if (!handle || !out) {
        return 0;
    }
    ensure_candidate_cache(handle);
    auto n = std::min(out_len, handle->candidate_features_cache.size());
    std::copy(
        handle->candidate_features_cache.begin(),
        handle->candidate_features_cache.begin() + static_cast<std::ptrdiff_t>(n),
        out);
    return n;
}

size_t tetris_cc_env_candidate_rows_write(
    const tetris_cc_env_handle* handle, tetris_cc_candidate_row* out, size_t out_len) {
    if (!handle || !out) {
        return 0;
    }
    ensure_candidate_cache(handle);
    const auto n = std::min(out_len, handle->candidate_cache.size());
    for (std::size_t i = 0; i < n; ++i) {
        const auto& cached = handle->candidate_cache[i];
        auto& row = out[i];
        row.use_hold = cached.use_hold;
        row.placement_index = cached.placement_index;
        row.piece = cached.piece;
        row.rotation = cached.rotation;
        row.x = cached.x;
        row.y = cached.y;
        row.lines_cleared = cached.lines_cleared;
        const auto feat_offset = static_cast<std::size_t>(i * kCandidateFeatureDim);
        for (std::size_t j = 0; j < kCandidateFeatureDim; ++j) {
            row.features[j] = handle->candidate_features_cache[feat_offset + j];
        }
    }
    return n;
}

int tetris_cc_env_apply_placement_index(
    tetris_cc_env_handle* handle, size_t index, float* reward_out, int* lines_cleared_out, int* game_over_out) {
    if (!handle) {
        return 0;
    }
    auto result = handle->env.apply_placement_index(index);
    if (!result.action_succeeded) {
        return 0;
    }
    mark_env_mutated(handle);
    if (reward_out) {
        *reward_out = result.reward;
    }
    maybe_set_int(lines_cleared_out, result.lines_cleared);
    maybe_set_int(game_over_out, result.game_over ? 1 : 0);
    return 1;
}

int tetris_cc_env_last_clear_meta(
    const tetris_cc_env_handle* handle,
    int* spin_clear,
    int* difficult_clear,
    int* b2b_bonus_applied) {
    if (!handle) {
        return 0;
    }
    refresh_env_runtime(handle);
    const auto& state = handle->env.state();
    maybe_set_int(spin_clear, state.last_clear_spin ? 1 : 0);
    maybe_set_int(difficult_clear, state.last_clear_difficult ? 1 : 0);
    maybe_set_int(b2b_bonus_applied, state.last_clear_b2b_bonus ? 1 : 0);
    return 1;
}

int tetris_cc_env_last_clear_spin_type(const tetris_cc_env_handle* handle, int* spin_type) {
    if (!handle) {
        return 0;
    }
    refresh_env_runtime(handle);
    maybe_set_int(
        spin_type,
        static_cast<int>(handle->env.state().last_clear_spin_type));
    return 1;
}

int tetris_cc_env_last_attack_meta(
    const tetris_cc_env_handle* handle,
    int* attack_base,
    float* attack_combo_scaled,
    int* attack_rounded,
    int* attack_b2b_bonus,
    int* attack_all_clear_bonus,
    int* attack_total,
    int* all_clear,
    int* b2b_streak,
    int* surge_charge,
    int* surge_release) {
    if (!handle) {
        return 0;
    }
    refresh_env_runtime(handle);
    const auto& state = handle->env.state();
    maybe_set_int(attack_base, state.last_attack_base);
    maybe_set_float(attack_combo_scaled, state.last_attack_combo_scaled);
    maybe_set_int(attack_rounded, state.last_attack_rounded);
    maybe_set_int(attack_b2b_bonus, state.last_attack_b2b_bonus);
    maybe_set_int(attack_all_clear_bonus, state.last_attack_all_clear_bonus);
    maybe_set_int(attack_total, state.last_attack_total);
    maybe_set_int(all_clear, state.last_clear_all_clear ? 1 : 0);
    maybe_set_int(b2b_streak, state.b2b_streak);
    maybe_set_int(surge_charge, state.last_attack_surge_charge);
    maybe_set_int(surge_release, state.last_attack_surge_release);
    return 1;
}

int tetris_cc_env_blitz_meta(
    const tetris_cc_env_handle* handle,
    int* score_total,
    int* level,
    int* lines_to_next,
    int* time_remaining_ms,
    int* timed_out) {
    if (!handle) {
        return 0;
    }
    refresh_env_runtime(handle);
    const auto& state = handle->env.state();
    maybe_set_int(score_total, state.blitz_score_total);
    maybe_set_int(level, state.blitz_level);
    maybe_set_int(lines_to_next, state.blitz_lines_to_next);
    maybe_set_int(time_remaining_ms, state.blitz_time_remaining_ms);
    maybe_set_int(timed_out, state.blitz_timed_out ? 1 : 0);
    return 1;
}

size_t tetris_cc_env_rotation_trace_count(const tetris_cc_env_handle* handle, int rotate_action) {
    if (!handle) {
        return 0;
    }
    refresh_env_runtime(handle);
    auto action = parse_rotate_action(rotate_action);
    if (!action.has_value()) {
        return 0;
    }
    auto trace = handle->env.rotation_trace(*action);
    return trace.tests.size();
}

int tetris_cc_env_rotation_trace_get(
    const tetris_cc_env_handle* handle,
    int rotate_action,
    size_t index,
    int* test_index,
    int* phase,
    int* kick_index,
    int* dx,
    int* dy,
    int* passed,
    int* candidate_x,
    int* candidate_y,
    int* candidate_rotation,
    int* candidate_collides) {
    if (!handle) {
        return 0;
    }
    refresh_env_runtime(handle);
    auto action = parse_rotate_action(rotate_action);
    if (!action.has_value()) {
        return 0;
    }
    auto trace = handle->env.rotation_trace(*action);
    if (index >= trace.tests.size()) {
        return 0;
    }
    const auto& test = trace.tests[index];
    maybe_set_int(test_index, test.test_index);
    maybe_set_int(phase, test.phase);
    maybe_set_int(kick_index, test.kick_index);
    maybe_set_int(dx, test.offset.x);
    maybe_set_int(dy, test.offset.y);
    maybe_set_int(passed, test.passed ? 1 : 0);
    maybe_set_int(candidate_x, test.candidate.x);
    maybe_set_int(candidate_y, test.candidate.y);
    maybe_set_int(candidate_rotation, static_cast<int>(test.candidate.rotation));
    maybe_set_int(candidate_collides, test.collides ? 1 : 0);
    return 1;
}

int tetris_cc_env_rotation_trace_meta(
    const tetris_cc_env_handle* handle,
    int rotate_action,
    int* success,
    int* final_x,
    int* final_y,
    int* final_rotation) {
    if (!handle) {
        return 0;
    }
    refresh_env_runtime(handle);
    auto action = parse_rotate_action(rotate_action);
    if (!action.has_value()) {
        return 0;
    }
    auto trace = handle->env.rotation_trace(*action);
    maybe_set_int(success, trace.success ? 1 : 0);
    if (trace.final_pose.has_value()) {
        maybe_set_int(final_x, trace.final_pose->x);
        maybe_set_int(final_y, trace.final_pose->y);
        maybe_set_int(final_rotation, static_cast<int>(trace.final_pose->rotation));
    } else {
        maybe_set_int(final_x, -1);
        maybe_set_int(final_y, -1);
        maybe_set_int(final_rotation, -1);
    }
    return 1;
}

tetris_cc_bot_handle* tetris_cc_bot_create_default(void) {
    try {
        auto* handle = new tetris_cc_bot_handle{};
        if (!handle->depth_bot || !handle->beam_bot) {
            delete handle;
            return nullptr;
        }
        return handle;
    } catch (...) {
        return nullptr;
    }
}

void tetris_cc_bot_destroy(tetris_cc_bot_handle* handle) { delete handle; }

int tetris_cc_bot_set_backend(tetris_cc_bot_handle* bot, int backend) {
    if (!bot || !valid_bot_backend(backend)) {
        return 0;
    }
    if (backend == TETRIS_CC_BOT_BACKEND_DEPTH && !bot->depth_bot) {
        return 0;
    }
    if (backend == TETRIS_CC_BOT_BACKEND_BEAM && !bot->beam_bot) {
        return 0;
    }
    bot->backend = backend;
    return 1;
}

int tetris_cc_bot_get_backend(const tetris_cc_bot_handle* bot, int* backend_out) {
    if (!bot || !backend_out) {
        return 0;
    }
    *backend_out = bot->backend;
    return 1;
}

int tetris_cc_bot_set_depth_config(
    tetris_cc_bot_handle* bot,
    int depth,
    double gamma,
    int deduplicate_successors,
    int use_transposition_table,
    int collect_debug_info,
    std::uint64_t max_nodes) {
    if (!bot || !bot->depth_bot) {
        return 0;
    }
    return tetris_v2::depth::set_config(
               bot->depth_bot,
               depth,
               gamma,
               deduplicate_successors != 0,
               use_transposition_table != 0,
               collect_debug_info != 0,
               max_nodes)
        ? 1
        : 0;
}

int tetris_cc_bot_set_beam_config(
    tetris_cc_bot_handle* bot,
    int depth,
    int beam_width,
    double gamma,
    int deduplicate_successors,
    int use_transposition_table,
    int collect_debug_info,
    std::uint64_t max_nodes) {
    if (!bot || !bot->beam_bot) {
        return 0;
    }
    return tetris_v2::beam::set_config(
               bot->beam_bot,
               depth,
               beam_width,
               gamma,
               deduplicate_successors != 0,
               use_transposition_table != 0,
               collect_debug_info != 0,
               max_nodes)
        ? 1
        : 0;
}

int tetris_cc_bot_sync_from_env(tetris_cc_bot_handle* bot, const tetris_cc_env_handle* env) {
    if (!bot || !env) {
        return 0;
    }
    if (bot->backend == TETRIS_CC_BOT_BACKEND_DEPTH) {
        if (!bot->depth_bot) {
            return 0;
        }
        return tetris_v2::depth::sync_from_env(bot->depth_bot, env->env.raw()) ? 1 : 0;
    }
    if (bot->backend == TETRIS_CC_BOT_BACKEND_BEAM) {
        if (!bot->beam_bot) {
            return 0;
        }
        return tetris_v2::beam::sync_from_env(bot->beam_bot, env->env.raw()) ? 1 : 0;
    }
    return bot->cold_clear_bot.sync_from_env(env->env) ? 1 : 0;
}

int tetris_cc_bot_choose(
    tetris_cc_bot_handle* bot,
    int think_ms,
    int* use_hold_out,
    size_t* placement_index_out,
    float* score_out,
    std::uint64_t* nodes_out,
    double* think_ms_out,
    double* nps_out,
    int* budget_miss_out) {
    if (!bot) {
        return 0;
    }
    if (bot->backend == TETRIS_CC_BOT_BACKEND_DEPTH) {
        if (!bot->depth_bot) {
            return 0;
        }
        bool use_hold = false;
        std::size_t placement_index = 0;
        float score = 0.0f;
        std::uint64_t nodes = 0;
        double think = 0.0;
        double nps = 0.0;
        int budget_miss = 0;
        if (!tetris_v2::depth::choose(
                bot->depth_bot,
                think_ms,
                &use_hold,
                &placement_index,
                &score,
                &nodes,
                &think,
                &nps,
                &budget_miss)) {
            return 0;
        }
        maybe_set_int(use_hold_out, use_hold ? 1 : 0);
        if (placement_index_out) {
            *placement_index_out = placement_index;
        }
        if (score_out) {
            *score_out = score;
        }
        maybe_set_u64(nodes_out, nodes);
        maybe_set_double(think_ms_out, think);
        maybe_set_double(nps_out, nps);
        maybe_set_int(budget_miss_out, budget_miss);
        return 1;
    }
    if (bot->backend == TETRIS_CC_BOT_BACKEND_BEAM) {
        if (!bot->beam_bot) {
            return 0;
        }
        bool use_hold = false;
        std::size_t placement_index = 0;
        float score = 0.0f;
        std::uint64_t nodes = 0;
        double think = 0.0;
        double nps = 0.0;
        int budget_miss = 0;
        if (!tetris_v2::beam::choose(
                bot->beam_bot,
                think_ms,
                &use_hold,
                &placement_index,
                &score,
                &nodes,
                &think,
                &nps,
                &budget_miss)) {
            return 0;
        }
        maybe_set_int(use_hold_out, use_hold ? 1 : 0);
        if (placement_index_out) {
            *placement_index_out = placement_index;
        }
        if (score_out) {
            *score_out = score;
        }
        maybe_set_u64(nodes_out, nodes);
        maybe_set_double(think_ms_out, think);
        maybe_set_double(nps_out, nps);
        maybe_set_int(budget_miss_out, budget_miss);
        return 1;
    }
    tetris_v2::cc::BotChoice choice{};
    tetris_v2::cc::BotThinkStats stats{};
    if (!bot->cold_clear_bot.choose(think_ms, &choice, &stats)) {
        return 0;
    }

    maybe_set_int(use_hold_out, choice.use_hold ? 1 : 0);
    if (placement_index_out) {
        *placement_index_out = choice.placement_index;
    }
    if (score_out) {
        *score_out = choice.score;
    }
    maybe_set_u64(nodes_out, stats.nodes);
    maybe_set_double(think_ms_out, stats.think_ms);
    maybe_set_double(nps_out, stats.nps);
    maybe_set_int(budget_miss_out, stats.budget_miss);
    return 1;
}

int tetris_cc_bot_choose_ex(
    tetris_cc_bot_handle* bot,
    int think_ms,
    int* use_hold_out,
    size_t* placement_index_out,
    float* score_out,
    std::uint64_t* nodes_out,
    double* think_ms_out,
    double* nps_out,
    int* budget_miss_out) {
    return tetris_cc_bot_choose(
        bot,
        think_ms,
        use_hold_out,
        placement_index_out,
        score_out,
        nodes_out,
        think_ms_out,
        nps_out,
        budget_miss_out);
}

int tetris_cc_bot_apply_choice(
    tetris_cc_bot_handle* bot,
    tetris_cc_env_handle* env,
    float* reward_out,
    int* lines_cleared_out,
    int* game_over_out,
    int* used_hold_out,
    size_t* placement_index_out) {
    if (!bot || !env) {
        return 0;
    }

    tetris_v2::StepResult result{};
    int used_hold = 0;
    std::size_t placement_index = 0;
    if (bot->backend == TETRIS_CC_BOT_BACKEND_DEPTH) {
        if (!bot->depth_bot) {
            return 0;
        }
        if (!tetris_v2::depth::apply_last_choice(
                bot->depth_bot, env->env.raw(), &result, &used_hold, &placement_index)) {
            return 0;
        }
    } else if (bot->backend == TETRIS_CC_BOT_BACKEND_BEAM) {
        if (!bot->beam_bot) {
            return 0;
        }
        if (!tetris_v2::beam::apply_last_choice(
                bot->beam_bot, env->env.raw(), &result, &used_hold, &placement_index)) {
            return 0;
        }
    } else if (!bot->cold_clear_bot.apply_last_choice(env->env, &result, &used_hold, &placement_index)) {
        return 0;
    }
    mark_env_mutated(env);

    if (reward_out) {
        *reward_out = result.reward;
    }
    maybe_set_int(lines_cleared_out, result.lines_cleared);
    maybe_set_int(game_over_out, result.game_over ? 1 : 0);
    maybe_set_int(used_hold_out, used_hold);
    if (placement_index_out) {
        *placement_index_out = placement_index;
    }
    return 1;
}

int tetris_cc_bot_choose_and_apply(
    tetris_cc_bot_handle* bot,
    tetris_cc_env_handle* env,
    int think_ms,
    float* reward_out,
    int* lines_cleared_out,
    int* game_over_out,
    int* used_hold_out,
    size_t* placement_index_out,
    float* score_out,
    std::uint64_t* nodes_out,
    double* think_ms_out,
    double* nps_out,
    int* budget_miss_out) {
    if (!bot || !env) {
        return 0;
    }
    tetris_v2::StepResult result{};
    int used_hold = 0;
    std::size_t placement_index = 0;

    if (bot->backend == TETRIS_CC_BOT_BACKEND_DEPTH) {
        if (!bot->depth_bot) {
            return 0;
        }
        float score = 0.0f;
        std::uint64_t nodes = 0;
        double think = 0.0;
        double nps = 0.0;
        int budget_miss = 0;
        bool use_hold_choice = false;
        if (!tetris_v2::depth::choose_and_apply(
                bot->depth_bot,
                env->env.raw(),
                think_ms,
                &result,
                &use_hold_choice,
                &placement_index,
                &score,
                &nodes,
                &think,
                &nps,
                &budget_miss,
                &used_hold)) {
            return 0;
        }
        mark_env_mutated(env);
        if (reward_out) {
            *reward_out = result.reward;
        }
        maybe_set_int(lines_cleared_out, result.lines_cleared);
        maybe_set_int(game_over_out, result.game_over ? 1 : 0);
        maybe_set_int(used_hold_out, used_hold);
        if (placement_index_out) {
            *placement_index_out = placement_index;
        }
        if (score_out) {
            *score_out = score;
        }
        maybe_set_u64(nodes_out, nodes);
        maybe_set_double(think_ms_out, think);
        maybe_set_double(nps_out, nps);
        maybe_set_int(budget_miss_out, budget_miss);
        return 1;
    }
    if (bot->backend == TETRIS_CC_BOT_BACKEND_BEAM) {
        if (!bot->beam_bot) {
            return 0;
        }
        float score = 0.0f;
        std::uint64_t nodes = 0;
        double think = 0.0;
        double nps = 0.0;
        int budget_miss = 0;
        bool use_hold_choice = false;
        if (!tetris_v2::beam::choose_and_apply(
                bot->beam_bot,
                env->env.raw(),
                think_ms,
                &result,
                &use_hold_choice,
                &placement_index,
                &score,
                &nodes,
                &think,
                &nps,
                &budget_miss,
                &used_hold)) {
            return 0;
        }
        mark_env_mutated(env);
        if (reward_out) {
            *reward_out = result.reward;
        }
        maybe_set_int(lines_cleared_out, result.lines_cleared);
        maybe_set_int(game_over_out, result.game_over ? 1 : 0);
        maybe_set_int(used_hold_out, used_hold);
        if (placement_index_out) {
            *placement_index_out = placement_index;
        }
        if (score_out) {
            *score_out = score;
        }
        maybe_set_u64(nodes_out, nodes);
        maybe_set_double(think_ms_out, think);
        maybe_set_double(nps_out, nps);
        maybe_set_int(budget_miss_out, budget_miss);
        return 1;
    }

    tetris_v2::cc::BotChoice choice{};
    tetris_v2::cc::BotThinkStats stats{};
    if (!bot->cold_clear_bot.choose_and_apply(
            env->env,
            think_ms,
            &result,
            &choice,
            &stats,
            &used_hold,
            &placement_index)) {
        return 0;
    }
    mark_env_mutated(env);

    if (reward_out) {
        *reward_out = result.reward;
    }
    maybe_set_int(lines_cleared_out, result.lines_cleared);
    maybe_set_int(game_over_out, result.game_over ? 1 : 0);
    maybe_set_int(used_hold_out, used_hold);
    if (placement_index_out) {
        *placement_index_out = placement_index;
    }
    if (score_out) {
        *score_out = choice.score;
    }
    maybe_set_u64(nodes_out, stats.nodes);
    maybe_set_double(think_ms_out, stats.think_ms);
    maybe_set_double(nps_out, stats.nps);
    maybe_set_int(budget_miss_out, stats.budget_miss);
    return 1;
}

int tetris_cc_bot_choose_and_apply_ex(
    tetris_cc_bot_handle* bot,
    tetris_cc_env_handle* env,
    int think_ms,
    float* reward_out,
    int* lines_cleared_out,
    int* game_over_out,
    int* used_hold_out,
    size_t* placement_index_out,
    float* score_out,
    std::uint64_t* nodes_out,
    double* think_ms_out,
    double* nps_out,
    int* budget_miss_out) {
    return tetris_cc_bot_choose_and_apply(
        bot,
        env,
        think_ms,
        reward_out,
        lines_cleared_out,
        game_over_out,
        used_hold_out,
        placement_index_out,
        score_out,
        nodes_out,
        think_ms_out,
        nps_out,
        budget_miss_out);
}

}  // extern "C"
