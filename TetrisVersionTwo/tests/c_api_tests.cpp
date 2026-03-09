#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

#include "tetris_v2/c_api.h"
#include "tetris_v2/env.hpp"
#include "tetris_v2/piece_defs.hpp"

namespace {

using namespace tetris_v2;

std::vector<std::uint8_t> board_from_cpp_state(const EnvState& state, bool include_active) {
    constexpr int kRows = Board::kVisibleRows;
    constexpr int kCols = Board::kWidth;
    std::vector<std::uint8_t> out(static_cast<std::size_t>(kRows * kCols), 0);

    for (int row = 0; row < kRows; ++row) {
        int y = (kRows - 1) - row;
        auto mask = state.board.row_mask(y);
        for (int x = 0; x < kCols; ++x) {
            if (mask & (1u << x)) {
                out[static_cast<std::size_t>(row * kCols + x)] = 1;
            }
        }
    }

    if (include_active) {
        auto cells = piece_cells(state.active.piece, state.active.rotation);
        for (const auto& c : cells) {
            int x = state.active.x + c.x;
            int y = state.active.y + c.y;
            if (x < 0 || x >= kCols || y < 0 || y >= kRows) {
                continue;
            }
            int row = (kRows - 1) - y;
            out[static_cast<std::size_t>(row * kCols + x)] = 1;
        }
    }

    return out;
}

std::vector<std::uint8_t> board_from_cpp_board(const Board& board) {
    EnvState fake{};
    fake.board = board;
    fake.active = ActivePiece{};
    return board_from_cpp_state(fake, false);
}

std::vector<std::uint8_t> board_from_c_api(tetris_env_handle* handle, bool include_active) {
    std::vector<std::uint8_t> out(200, 0);
    auto written = tetris_env_board_write(
        handle, include_active ? 1 : 0, out.data(), out.size());
    assert(written == out.size());
    return out;
}

std::vector<std::uint8_t> board_piece_ids_from_c_api(tetris_env_handle* handle, bool include_active) {
    std::vector<std::uint8_t> out(200, 255);
    auto written = tetris_env_board_piece_ids_write(
        handle, include_active ? 1 : 0, out.data(), out.size());
    assert(written == out.size());
    return out;
}

void test_smoke_and_state_exports() {
    auto* handle = tetris_env_create(1337);
    assert(handle != nullptr);

    auto obs_size = tetris_env_observation_size(handle, 1);
    assert(obs_size > 0);
    std::vector<float> obs(obs_size, 0.0f);
    auto obs_written = tetris_env_observation_write(handle, 1, obs.data(), obs.size());
    assert(obs_written == obs.size());

    int piece = -1;
    int rotation = -1;
    int x = 0;
    int y = 0;
    assert(tetris_env_active_piece(handle, &piece, &rotation, &x, &y) == 1);
    assert(piece >= 0 && piece <= 7);
    assert(rotation >= 0 && rotation <= 3);

    int has_hold = 0;
    int hold_piece = -1;
    int hold_available = 0;
    assert(tetris_env_hold_piece(handle, &has_hold, &hold_piece, &hold_available) == 1);
    assert(has_hold == 0 || has_hold == 1);
    assert(hold_available == 0 || hold_available == 1);

    auto q_count = tetris_env_queue_count(handle);
    assert(q_count > 0);
    for (std::size_t i = 0; i < q_count; ++i) {
        int q_piece = -1;
        assert(tetris_env_queue_get(handle, i, &q_piece) == 1);
        assert(q_piece >= 0 && q_piece <= 7);
    }

    int game_over = 0;
    int top_out = 0;
    int combo = 0;
    int b2b = 0;
    int total_lines = 0;
    int lock_timer = 0;
    int lock_resets = 0;
    assert(
        tetris_env_meta(handle, &game_over, &top_out, &combo, &b2b, &total_lines, &lock_timer, &lock_resets) ==
        1);

    auto board = board_from_c_api(handle, true);
    assert(board.size() == 200);
    auto ids = board_piece_ids_from_c_api(handle, true);
    assert(ids.size() == 200);

    tetris_env_destroy(handle);
}

void test_placement_parity_and_apply() {
    constexpr std::uint32_t kSeed = 2026;
    auto* handle = tetris_env_create(kSeed);
    assert(handle != nullptr);

    EnvConfig cfg;
    cfg.seed = kSeed;
    ModernTetrisEnv env_cpp(cfg);

    auto cpp_board = board_from_cpp_state(env_cpp.state(), true);
    auto c_board = board_from_c_api(handle, true);
    assert(cpp_board == c_board);
    auto cpp_ids = env_cpp.visible_board_piece_ids(true);
    auto c_ids = board_piece_ids_from_c_api(handle, true);
    assert(cpp_ids == c_ids);

    auto cpp_options = env_cpp.enumerate_active_piece_placements();
    auto c_count = tetris_env_placement_count(handle);
    assert(c_count == cpp_options.size());

    for (std::size_t i = 0; i < c_count; ++i) {
        int x = 0;
        int y = 0;
        int rotation = 0;
        int lines = 0;
        assert(tetris_env_placement_get(handle, i, &x, &y, &rotation, &lines) == 1);
        assert(x == cpp_options[i].placement.x);
        assert(y == cpp_options[i].placement.y);
        assert(rotation == static_cast<int>(cpp_options[i].placement.rotation));
        assert(lines == cpp_options[i].lines_cleared);

        int spin_candidate = 0;
        int difficult_candidate = 0;
        int used_kick = 0;
        assert(
            tetris_env_placement_get_ex(
                handle,
                i,
                &x,
                &y,
                &rotation,
                &lines,
                &spin_candidate,
                &difficult_candidate,
                &used_kick) == 1);
        assert(spin_candidate == (cpp_options[i].spin_clear_candidate ? 1 : 0));
        assert(difficult_candidate == (cpp_options[i].difficult_clear_candidate ? 1 : 0));
        assert(used_kick == (cpp_options[i].last_rotate_used_kick_path ? 1 : 0));

        std::vector<std::uint8_t> c_after(200, 0);
        auto written = tetris_env_placement_board_write(handle, i, c_after.data(), c_after.size());
        assert(written == c_after.size());
        auto cpp_after = board_from_cpp_board(cpp_options[i].board_after_lock);
        assert(c_after == cpp_after);

        std::vector<std::uint8_t> c_after_ids(200, 255);
        auto written_ids =
            tetris_env_placement_board_piece_ids_write(handle, i, c_after_ids.data(), c_after_ids.size());
        assert(written_ids == c_after_ids.size());
        auto cpp_after_ids = env_cpp.visible_placement_piece_ids(i);
        assert(c_after_ids == cpp_after_ids);
    }

    if (!cpp_options.empty()) {
        std::size_t pick = cpp_options.size() / 2;
        float c_reward = 0.0f;
        int c_lines = -1;
        int c_game_over = 0;
        assert(tetris_env_apply_placement_index(handle, pick, &c_reward, &c_lines, &c_game_over) == 1);

        auto cpp_step = env_cpp.apply_placement_index(pick);
        assert(cpp_step.action_succeeded);
        assert(std::fabs(cpp_step.reward - c_reward) < 1e-5f);
        assert(cpp_step.lines_cleared == c_lines);
        assert((cpp_step.game_over ? 1 : 0) == c_game_over);

        int c_spin_clear = 0;
        int c_difficult_clear = 0;
        int c_b2b_bonus = 0;
        assert(tetris_env_last_clear_meta(handle, &c_spin_clear, &c_difficult_clear, &c_b2b_bonus) == 1);
        assert(c_spin_clear == (cpp_step.spin_clear ? 1 : 0));
        assert(c_difficult_clear == (cpp_step.difficult_clear ? 1 : 0));
        assert(c_b2b_bonus == (cpp_step.b2b_bonus_applied ? 1 : 0));

        auto board_cpp_after = board_from_cpp_state(env_cpp.state(), true);
        auto board_c_after = board_from_c_api(handle, true);
        assert(board_cpp_after == board_c_after);
        auto board_cpp_after_ids = env_cpp.visible_board_piece_ids(true);
        auto board_c_after_ids = board_piece_ids_from_c_api(handle, true);
        assert(board_cpp_after_ids == board_c_after_ids);
    }

    tetris_env_destroy(handle);
}

void test_rotation_trace_parity() {
    constexpr std::uint32_t kSeed = 99;
    auto* handle = tetris_env_create(kSeed);
    assert(handle != nullptr);

    EnvConfig cfg;
    cfg.seed = kSeed;
    ModernTetrisEnv env_cpp(cfg);

    const std::array<Action, 3> actions{Action::RotateCW, Action::RotateCCW, Action::Rotate180};
    for (auto action : actions) {
        auto cpp_trace = env_cpp.rotation_trace(action);
        auto c_count = tetris_env_rotation_trace_count(handle, static_cast<int>(action));
        assert(c_count == cpp_trace.tests.size());
        for (std::size_t i = 0; i < cpp_trace.tests.size(); ++i) {
            assert(cpp_trace.tests[i].test_index == static_cast<int>(i));
        }

        int c_success = 0;
        int c_final_x = -2;
        int c_final_y = -2;
        int c_final_rotation = -2;
        assert(
            tetris_env_rotation_trace_meta(
                handle,
                static_cast<int>(action),
                &c_success,
                &c_final_x,
                &c_final_y,
                &c_final_rotation) == 1);

        assert((cpp_trace.success ? 1 : 0) == c_success);
        if (cpp_trace.final_pose.has_value()) {
            assert(c_final_x == cpp_trace.final_pose->x);
            assert(c_final_y == cpp_trace.final_pose->y);
            assert(c_final_rotation == static_cast<int>(cpp_trace.final_pose->rotation));
        } else {
            assert(c_final_x == -1 && c_final_y == -1 && c_final_rotation == -1);
        }

        for (std::size_t i = 0; i < c_count; ++i) {
            int test_index = -1;
            int phase = -1;
            int kick_index = -1;
            int dx = 0;
            int dy = 0;
            int passed = 0;
            int cx = 0;
            int cy = 0;
            int crotation = 0;
            int collides = 0;
            assert(
                tetris_env_rotation_trace_get(
                    handle,
                    static_cast<int>(action),
                    i,
                    &test_index,
                    &phase,
                    &kick_index,
                    &dx,
                    &dy,
                    &passed,
                    &cx,
                    &cy,
                    &crotation,
                    &collides) == 1);

            const auto& cpp = cpp_trace.tests[i];
            assert(test_index == cpp.test_index);
            assert(test_index == static_cast<int>(i));
            assert(phase == cpp.phase);
            assert(kick_index == cpp.kick_index);
            assert(dx == cpp.offset.x);
            assert(dy == cpp.offset.y);
            assert(passed == (cpp.passed ? 1 : 0));
            assert(cx == cpp.candidate.x);
            assert(cy == cpp.candidate.y);
            assert(crotation == static_cast<int>(cpp.candidate.rotation));
            assert(collides == (cpp.collides ? 1 : 0));
        }

        // Ensure each phase stops at first passing kick test.
        int current_phase = -1;
        bool seen_pass_in_phase = false;
        for (const auto& t : cpp_trace.tests) {
            if (t.phase != current_phase) {
                current_phase = t.phase;
                seen_pass_in_phase = false;
            }
            if (seen_pass_in_phase) {
                assert(false && "Phase continued after a passing kick");
            }
            if (t.passed) {
                seen_pass_in_phase = true;
            }
        }

        // When trace succeeds, final pose must match the last passing candidate in trace.
        if (cpp_trace.success) {
            const KickTest* last_pass = nullptr;
            for (const auto& t : cpp_trace.tests) {
                if (t.passed) {
                    last_pass = &t;
                }
            }
            assert(last_pass != nullptr);
            assert(cpp_trace.final_pose.has_value());
            assert(cpp_trace.final_pose->x == last_pass->candidate.x);
            assert(cpp_trace.final_pose->y == last_pass->candidate.y);
            assert(cpp_trace.final_pose->rotation == last_pass->candidate.rotation);
        }
    }

    tetris_env_destroy(handle);
}

void test_bounds_and_null_safety() {
    // Null handle safety.
    assert(tetris_env_observation_size(nullptr, 1) == 0);
    assert(tetris_env_queue_count(nullptr) == 0);
    assert(tetris_env_placement_count(nullptr) == 0);
    assert(tetris_env_rotation_trace_count(nullptr, static_cast<int>(Action::RotateCW)) == 0);
    assert(tetris_env_board_write(nullptr, 1, nullptr, 0) == 0);
    assert(tetris_env_active_piece(nullptr, nullptr, nullptr, nullptr, nullptr) == 0);
    assert(tetris_env_placement_get_ex(nullptr, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) == 0);
    assert(tetris_env_last_clear_meta(nullptr, nullptr, nullptr, nullptr) == 0);

    auto* handle = tetris_env_create(5);
    assert(handle != nullptr);

    int dummy = 0;
    float reward = 0.0f;
    std::array<std::uint8_t, 200> board{};

    assert(tetris_env_queue_get(handle, 9999, &dummy) == 0);
    assert(tetris_env_placement_get(handle, 9999, &dummy, &dummy, &dummy, &dummy) == 0);
    assert(
        tetris_env_placement_get_ex(
            handle,
            9999,
            &dummy,
            &dummy,
            &dummy,
            &dummy,
            &dummy,
            &dummy,
            &dummy) == 0);
    assert(tetris_env_placement_board_write(handle, 9999, board.data(), board.size()) == 0);
    assert(tetris_env_placement_board_piece_ids_write(handle, 9999, board.data(), board.size()) == 0);
    assert(tetris_env_apply_placement_index(handle, 9999, &reward, &dummy, &dummy) == 0);
    assert(tetris_env_rotation_trace_count(handle, static_cast<int>(Action::Left)) == 0);
    assert(
        tetris_env_rotation_trace_get(
            handle,
            static_cast<int>(Action::RotateCW),
            9999,
            &dummy,
            &dummy,
            &dummy,
            &dummy,
            &dummy,
            &dummy,
            &dummy,
            &dummy,
            &dummy,
            &dummy) == 0);

    tetris_env_destroy(handle);
}

}  // namespace

int main() {
    test_smoke_and_state_exports();
    test_placement_parity_and_apply();
    test_rotation_trace_parity();
    test_bounds_and_null_safety();
    return 0;
}
