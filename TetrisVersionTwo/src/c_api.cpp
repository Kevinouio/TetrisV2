#include "tetris_v2/c_api.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <new>
#include <optional>
#include <vector>

#include "tetris_v2/cc_bot.hpp"
#include "tetris_v2/cc_env.hpp"
#include "tetris_v2/observation.hpp"
#include "tetris_v2/piece_defs.hpp"

struct tetris_cc_env_handle {
    explicit tetris_cc_env_handle(std::uint32_t seed) : env(tetris_v2::EnvConfig{seed}) {}
    tetris_v2::cc::Env env;
};

struct tetris_cc_bot_handle {
    tetris_v2::cc::Bot bot;
};

struct tetris_cc_snapshot_handle {
    tetris_v2::EnvSnapshot snapshot{};
};

namespace {

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

void maybe_set_double(double* dst, double value) {
    if (dst) {
        *dst = value;
    }
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
}

int tetris_cc_env_step(tetris_cc_env_handle* handle, int action, float* reward_out) {
    if (!handle) {
        return 1;
    }
    auto result = handle->env.step(parse_action(action));
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
    if (reward_out) {
        *reward_out = result.reward;
    }
    return result.action_succeeded ? 1 : 0;
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
    return 1;
}

size_t tetris_cc_env_observation_size(const tetris_cc_env_handle* handle, int include_hidden_rows) {
    if (!handle) {
        return 0;
    }
    return tetris_v2::observation_size(
        handle->env.config().queue_size, include_hidden_rows != 0);
}

size_t tetris_cc_env_observation_write(
    const tetris_cc_env_handle* handle, int include_hidden_rows, float* out, size_t out_len) {
    if (!handle || !out) {
        return 0;
    }
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
    auto ids = handle->env.visible_board_piece_ids(include_active != 0);
    auto n = std::min(out_len, ids.size());
    std::copy(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(n), out);
    return n;
}

int tetris_cc_env_active_piece(
    const tetris_cc_env_handle* handle, int* piece, int* rotation, int* x, int* y) {
    if (!handle) {
        return 0;
    }
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
    return handle->env.state().queue.size();
}

int tetris_cc_env_queue_get(const tetris_cc_env_handle* handle, size_t index, int* piece) {
    if (!handle || !piece) {
        return 0;
    }
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
    return handle->env.enumerate_active_piece_placements().size();
}

int tetris_cc_env_placement_get(
    const tetris_cc_env_handle* handle, size_t index, int* x, int* y, int* rotation, int* lines_cleared) {
    if (!handle) {
        return 0;
    }
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
    auto option = handle->env.placement_option_at(index);
    if (!option.has_value()) {
        return 0;
    }
    auto ids = handle->env.visible_placement_piece_ids(index);
    auto n = std::min(out_len, ids.size());
    std::copy(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(n), out);
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
    maybe_set_int(
        spin_type,
        static_cast<int>(handle->env.state().last_clear_spin_type));
    return 1;
}

size_t tetris_cc_env_rotation_trace_count(const tetris_cc_env_handle* handle, int rotate_action) {
    if (!handle) {
        return 0;
    }
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
        return new tetris_cc_bot_handle{};
    } catch (...) {
        return nullptr;
    }
}

void tetris_cc_bot_destroy(tetris_cc_bot_handle* handle) { delete handle; }

int tetris_cc_bot_sync_from_env(tetris_cc_bot_handle* bot, const tetris_cc_env_handle* env) {
    if (!bot || !env) {
        return 0;
    }
    return bot->bot.sync_from_env(env->env) ? 1 : 0;
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

    tetris_v2::cc::BotChoice choice{};
    tetris_v2::cc::BotThinkStats stats{};
    if (!bot->bot.choose(think_ms, &choice, &stats)) {
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
    if (!bot->bot.apply_last_choice(env->env, &result, &used_hold, &placement_index)) {
        return 0;
    }

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
    tetris_v2::cc::BotChoice choice{};
    tetris_v2::cc::BotThinkStats stats{};
    int used_hold = 0;
    std::size_t placement_index = 0;
    if (!bot->bot.choose_and_apply(
            env->env,
            think_ms,
            &result,
            &choice,
            &stats,
            &used_hold,
            &placement_index)) {
        return 0;
    }

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
