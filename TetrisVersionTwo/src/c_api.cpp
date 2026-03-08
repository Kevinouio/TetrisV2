#include "tetris_v2/c_api.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <new>
#include <optional>
#include <vector>

#include "tetris_v2/env.hpp"
#include "tetris_v2/observation.hpp"
#include "tetris_v2/piece_defs.hpp"

struct tetris_env_handle {
    explicit tetris_env_handle(std::uint32_t seed) : env(tetris_v2::EnvConfig{seed}) {}
    tetris_v2::ModernTetrisEnv env;
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
    if (parsed == tetris_v2::Action::RotateCW || parsed == tetris_v2::Action::RotateCCW ||
        parsed == tetris_v2::Action::Rotate180) {
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

}  // namespace

extern "C" {

tetris_env_handle* tetris_env_create(uint32_t seed) {
    try {
        return new tetris_env_handle(seed);
    } catch (...) {
        return nullptr;
    }
}

void tetris_env_destroy(tetris_env_handle* handle) { delete handle; }

void tetris_env_reset(tetris_env_handle* handle, uint32_t seed) {
    if (!handle) {
        return;
    }
    handle->env.reset(seed);
}

int tetris_env_step(tetris_env_handle* handle, int action, float* reward_out) {
    if (!handle) {
        return 1;
    }
    auto result = handle->env.step(parse_action(action));
    if (reward_out) {
        *reward_out = result.reward;
    }
    return result.game_over ? 1 : 0;
}

int tetris_env_hold(tetris_env_handle* handle, float* reward_out) {
    if (!handle) {
        return 0;
    }
    auto result = handle->env.step(tetris_v2::Action::Hold);
    if (reward_out) {
        *reward_out = result.reward;
    }
    return result.action_succeeded ? 1 : 0;
}

size_t tetris_env_observation_size(const tetris_env_handle* handle, int include_hidden_rows) {
    if (!handle) {
        return 0;
    }
    return tetris_v2::observation_size(
        handle->env.config().queue_size, include_hidden_rows != 0);
}

size_t tetris_env_observation_write(
    const tetris_env_handle* handle, int include_hidden_rows, float* out, size_t out_len) {
    if (!handle || !out) {
        return 0;
    }
    auto obs = tetris_v2::encode_observation(
        handle->env.state(), handle->env.config().queue_size, include_hidden_rows != 0);
    auto n = std::min(out_len, obs.size());
    std::copy(obs.begin(), obs.begin() + static_cast<std::ptrdiff_t>(n), out);
    return n;
}

size_t tetris_env_board_write(
    const tetris_env_handle* handle, int include_active, uint8_t* out, size_t out_len) {
    if (!handle || !out) {
        return 0;
    }
    std::optional<tetris_v2::ActivePiece> active{};
    if (include_active != 0) {
        active = handle->env.state().active;
    }
    return write_visible_board(handle->env.state().board, active, out, out_len);
}

int tetris_env_active_piece(
    const tetris_env_handle* handle, int* piece, int* rotation, int* x, int* y) {
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

int tetris_env_hold_piece(
    const tetris_env_handle* handle, int* has_hold, int* hold_piece, int* hold_available) {
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

size_t tetris_env_queue_count(const tetris_env_handle* handle) {
    if (!handle) {
        return 0;
    }
    return handle->env.state().queue.size();
}

int tetris_env_queue_get(const tetris_env_handle* handle, size_t index, int* piece) {
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

int tetris_env_meta(
    const tetris_env_handle* handle,
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

size_t tetris_env_placement_count(const tetris_env_handle* handle) {
    if (!handle) {
        return 0;
    }
    return handle->env.enumerate_active_piece_placements().size();
}

int tetris_env_placement_get(
    const tetris_env_handle* handle, size_t index, int* x, int* y, int* rotation, int* lines_cleared) {
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

size_t tetris_env_placement_board_write(
    const tetris_env_handle* handle, size_t index, uint8_t* out, size_t out_len) {
    if (!handle || !out) {
        return 0;
    }
    auto option = handle->env.placement_option_at(index);
    if (!option.has_value()) {
        return 0;
    }
    return write_visible_board(option->board_after_lock, std::nullopt, out, out_len);
}

int tetris_env_apply_placement_index(
    tetris_env_handle* handle, size_t index, float* reward_out, int* lines_cleared_out, int* game_over_out) {
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

size_t tetris_env_rotation_trace_count(const tetris_env_handle* handle, int rotate_action) {
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

int tetris_env_rotation_trace_get(
    const tetris_env_handle* handle,
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

int tetris_env_rotation_trace_meta(
    const tetris_env_handle* handle,
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

}  // extern "C"
