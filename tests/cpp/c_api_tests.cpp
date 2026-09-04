#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

#include "tetris_v2/bit_ops.hpp"
#include "tetris_v2/c_api.h"
#include "tetris_v2/cc2_dag.hpp"
#include "tetris_v2/cc2_data.hpp"
#include "tetris_v2/cc2_movegen.hpp"
#include "tetris_v2/env.hpp"
#include "tetris_v2/piece_defs.hpp"

namespace {

using namespace tetris_v2;

void test_portable_bit_ops() {
    using namespace tetris_v2::bit_ops;

    assert(countl_zero_u64(0) == 64u);
    assert(countl_zero_u64(1) == 63u);
    assert(countl_zero_u64(1ull << 63u) == 0u);
    assert(countr_zero_u64(0) == 64u);
    assert(countr_zero_u64(1) == 0u);
    assert(countr_zero_u64(1ull << 63u) == 63u);
    assert(popcount_u64(0) == 0u);
    assert(popcount_u64(~0ull) == 64u);
    assert(popcount_u64(0xAAAAAAAAAAAAAAAAull) == 32u);
}

std::vector<std::uint8_t> board_from_cc_api(tetris_cc_env_handle* handle, bool include_active) {
    std::vector<std::uint8_t> out(200, 0);
    auto written = tetris_cc_env_board_write(
        handle, include_active ? 1 : 0, out.data(), out.size());
    assert(written == out.size());
    return out;
}

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

void test_cc_smoke_and_state_exports() {
    auto* env = tetris_cc_env_create(1337);
    assert(env != nullptr);

    auto obs_size = tetris_cc_env_observation_size(env, 1);
    assert(obs_size > 0);
    std::vector<float> obs(obs_size, 0.0f);
    auto obs_written = tetris_cc_env_observation_write(env, 1, obs.data(), obs.size());
    assert(obs_written == obs.size());

    int piece = -1;
    int rot = -1;
    int x = 0;
    int y = 0;
    assert(tetris_cc_env_active_piece(env, &piece, &rot, &x, &y) == 1);
    assert(piece >= 0 && piece <= 7);
    assert(rot >= 0 && rot <= 3);

    int has_hold = 0;
    int hold_piece = -1;
    int hold_available = 0;
    assert(tetris_cc_env_hold_piece(env, &has_hold, &hold_piece, &hold_available) == 1);
    assert(has_hold == 0 || has_hold == 1);
    assert(hold_available == 0 || hold_available == 1);

    const auto q_count = tetris_cc_env_queue_count(env);
    assert(q_count > 0);
    for (std::size_t i = 0; i < q_count; ++i) {
        int q_piece = -1;
        assert(tetris_cc_env_queue_get(env, i, &q_piece) == 1);
        assert(q_piece >= 0 && q_piece <= 7);
    }

    auto board = board_from_cc_api(env, true);
    assert(board.size() == 200);

    int game_over = 0;
    int top_out = 0;
    int combo = -1;
    int b2b = 0;
    int lines = 0;
    int lock_timer = 0;
    int lock_resets = 0;
    assert(
        tetris_cc_env_meta(
            env,
            &game_over,
            &top_out,
            &combo,
            &b2b,
            &lines,
            &lock_timer,
            &lock_resets) == 1);

    tetris_cc_env_destroy(env);
}

void test_cc_matches_cpp_env_for_basic_state() {
    constexpr std::uint32_t kSeed = 2026;
    auto* env = tetris_cc_env_create(kSeed);
    assert(env != nullptr);

    EnvConfig cfg;
    cfg.seed = kSeed;
    ModernTetrisEnv cpp_env(cfg);

    auto c_board = board_from_cc_api(env, true);
    auto cpp_board = board_from_cpp_state(cpp_env.state(), true);
    assert(c_board == cpp_board);

    auto count = tetris_cc_env_placement_count(env);
    auto options = cpp_env.enumerate_active_piece_placements();
    assert(count == options.size());

    for (std::size_t i = 0; i < count; ++i) {
        int x = 0;
        int y = 0;
        int rotation = 0;
        int lines = 0;
        assert(tetris_cc_env_placement_get(env, i, &x, &y, &rotation, &lines) == 1);
        assert(x == options[i].placement.x);
        assert(y == options[i].placement.y);
        assert(rotation == static_cast<int>(options[i].placement.rotation));
        assert(lines == options[i].lines_cleared);
    }

    tetris_cc_env_destroy(env);
}

void test_cc_snapshot_restore_keeps_future_deterministic() {
    auto* env_a = tetris_cc_env_create(777);
    auto* env_b = tetris_cc_env_create(1);
    assert(env_a != nullptr);
    assert(env_b != nullptr);

    for (int i = 0; i < 12; ++i) {
        const auto count = tetris_cc_env_placement_count(env_a);
        if (count == 0) {
            break;
        }
        float reward = 0.0f;
        int lines = 0;
        int game_over = 0;
        assert(tetris_cc_env_apply_placement_index(env_a, 0, &reward, &lines, &game_over) == 1);
        if (game_over != 0) {
            break;
        }
    }

    auto* snapshot = tetris_cc_env_snapshot_create(env_a);
    assert(snapshot != nullptr);
    assert(tetris_cc_env_restore_snapshot(env_b, snapshot) == 1);

    for (int i = 0; i < 16; ++i) {
        assert(board_from_cc_api(env_a, true) == board_from_cc_api(env_b, true));

        int piece_a = -1;
        int rot_a = -1;
        int x_a = 0;
        int y_a = 0;
        int piece_b = -1;
        int rot_b = -1;
        int x_b = 0;
        int y_b = 0;
        assert(tetris_cc_env_active_piece(env_a, &piece_a, &rot_a, &x_a, &y_a) == 1);
        assert(tetris_cc_env_active_piece(env_b, &piece_b, &rot_b, &x_b, &y_b) == 1);
        assert(piece_a == piece_b);
        assert(rot_a == rot_b);
        assert(x_a == x_b);
        assert(y_a == y_b);

        auto qa = tetris_cc_env_queue_count(env_a);
        auto qb = tetris_cc_env_queue_count(env_b);
        assert(qa == qb);
        for (std::size_t qi = 0; qi < qa; ++qi) {
            int pa = -1;
            int pb = -1;
            assert(tetris_cc_env_queue_get(env_a, qi, &pa) == 1);
            assert(tetris_cc_env_queue_get(env_b, qi, &pb) == 1);
            assert(pa == pb);
        }

        const auto count = tetris_cc_env_placement_count(env_a);
        if (count == 0) {
            break;
        }
        float reward_a = 0.0f;
        int lines_a = 0;
        int game_over_a = 0;
        float reward_b = 0.0f;
        int lines_b = 0;
        int game_over_b = 0;
        assert(tetris_cc_env_apply_placement_index(env_a, 0, &reward_a, &lines_a, &game_over_a) == 1);
        assert(tetris_cc_env_apply_placement_index(env_b, 0, &reward_b, &lines_b, &game_over_b) == 1);
        assert(lines_a == lines_b);
        assert(game_over_a == game_over_b);
        if (game_over_a != 0) {
            break;
        }
    }

    tetris_cc_snapshot_destroy(snapshot);
    tetris_cc_env_destroy(env_a);
    tetris_cc_env_destroy(env_b);
}

void test_cc_rotation_trace_disables_rotate180() {
    auto* env = tetris_cc_env_create(99);
    assert(env != nullptr);

    const auto cw_count = tetris_cc_env_rotation_trace_count(env, static_cast<int>(Action::RotateCW));
    const auto ccw_count = tetris_cc_env_rotation_trace_count(env, static_cast<int>(Action::RotateCCW));
    const auto r180_count = tetris_cc_env_rotation_trace_count(env, static_cast<int>(Action::Rotate180));
    assert(cw_count > 0);
    assert(ccw_count > 0);
    assert(r180_count == 0);

    tetris_cc_env_destroy(env);
}

void test_cc_play_mode_rich_steps_and_ghost() {
    auto* legacy = tetris_cc_env_create(99);
    auto* play = tetris_cc_env_create_play(99);
    assert(legacy != nullptr);
    assert(play != nullptr);

    int piece = -1;
    int rotation = -1;
    int x = 0;
    int y = 0;
    assert(tetris_cc_env_active_piece(legacy, &piece, &rotation, &x, &y) == 1);
    assert(y == 21);
    assert(tetris_cc_env_active_piece(play, &piece, &rotation, &x, &y) == 1);
    assert(y == 19);

    tetris_cc_env_step_result legacy_rotate{};
    assert(
        tetris_cc_env_step_ex(
            legacy, static_cast<int>(Action::Rotate180), &legacy_rotate) == 1);
    assert(legacy_rotate.action_succeeded == 0);
    assert(legacy_rotate.piece_locked == 0);

    const int rotation_before = rotation;
    tetris_cc_env_step_result play_rotate{};
    assert(
        tetris_cc_env_step_ex(
            play, static_cast<int>(Action::Rotate180), &play_rotate) == 1);
    assert(play_rotate.action_succeeded == 1);
    assert(play_rotate.piece_locked == 0);
    assert(play_rotate.game_over == 0);
    assert(tetris_cc_env_active_piece(play, &piece, &rotation, &x, &y) == 1);
    assert(rotation == (rotation_before + 2) % 4);

    int ghost_piece = -1;
    int ghost_rotation = -1;
    int ghost_x = 0;
    int ghost_y = 0;
    assert(
        tetris_cc_env_ghost_piece(
            play, &ghost_piece, &ghost_rotation, &ghost_x, &ghost_y) == 1);
    assert(ghost_piece == piece);
    assert(ghost_rotation == rotation);
    assert(ghost_x == x);
    assert(ghost_y <= y);

    tetris_cc_env_step_result hard_drop{};
    assert(
        tetris_cc_env_step_ex(
            play, static_cast<int>(Action::HardDrop), &hard_drop) == 1);
    assert(hard_drop.action_succeeded == 1);
    assert(hard_drop.piece_locked == 1);
    assert(hard_drop.hold_used == 0);
    assert(hard_drop.lines_cleared >= 0 && hard_drop.lines_cleared <= 4);
    assert(hard_drop.spin_type >= static_cast<int>(SpinType::None));
    assert(hard_drop.spin_type <= static_cast<int>(SpinType::Full));
    assert(std::isfinite(hard_drop.reward));
    assert(hard_drop.game_over == 0);
    assert(hard_drop.top_out == 0);

    assert(tetris_cc_env_step_ex(nullptr, static_cast<int>(Action::None), &hard_drop) == 0);
    assert(tetris_cc_env_step_ex(play, static_cast<int>(Action::None), nullptr) == 0);
    assert(tetris_cc_env_step_ex(play, -1, &hard_drop) == 0);
    assert(tetris_cc_env_ghost_piece(nullptr, nullptr, nullptr, nullptr, nullptr) == 0);

    tetris_cc_env_destroy(legacy);
    tetris_cc_env_destroy(play);
}

void test_cc_legacy_step_behavior_is_preserved() {
    auto* legacy = tetris_cc_env_create(314);
    auto* rich = tetris_cc_env_create(314);
    assert(legacy != nullptr);
    assert(rich != nullptr);

    float legacy_reward = -1.0f;
    const int legacy_return =
        tetris_cc_env_step(legacy, static_cast<int>(Action::HardDrop), &legacy_reward);
    tetris_cc_env_step_result rich_result{};
    assert(
        tetris_cc_env_step_ex(
            rich, static_cast<int>(Action::HardDrop), &rich_result) == 1);

    assert(legacy_return == rich_result.game_over);
    assert(legacy_reward == rich_result.reward);
    assert(board_from_cc_api(legacy, true) == board_from_cc_api(rich, true));

    tetris_cc_env_destroy(legacy);
    tetris_cc_env_destroy(rich);
}

void test_cc_zero_time_input_and_tick_bridge() {
    auto* env = tetris_cc_env_create_play(2718);
    assert(env != nullptr);

    int piece = -1;
    int rotation = -1;
    int x = 0;
    int y = 0;
    assert(tetris_cc_env_active_piece(env, &piece, &rotation, &x, &y) == 1);
    const int spawn_y = y;

    tetris_cc_env_step_result input_result{};
    assert(
        tetris_cc_env_input_ex(
            env, static_cast<int>(Action::Left), &input_result) == 1);
    assert(input_result.action_succeeded == 1);
    assert(
        tetris_cc_env_input_ex(
            env, static_cast<int>(Action::Right), &input_result) == 1);
    assert(input_result.action_succeeded == 1);
    assert(tetris_cc_env_active_piece(env, &piece, &rotation, &x, &y) == 1);
    assert(y == spawn_y);

    int soft_drop_count = 0;
    do {
        assert(
            tetris_cc_env_input_ex(
                env, static_cast<int>(Action::SoftDrop), &input_result) == 1);
        if (input_result.action_succeeded != 0) {
            ++soft_drop_count;
        }
    } while (input_result.action_succeeded != 0 && soft_drop_count < Board::kRows);
    assert(soft_drop_count > 0);
    assert(input_result.action_succeeded == 0);

    int lock_timer = -1;
    int lock_resets = -1;
    assert(
        tetris_cc_env_meta(
            env, nullptr, nullptr, nullptr, nullptr, nullptr, &lock_timer, &lock_resets) == 1);
    assert(lock_timer == 0);
    assert(lock_resets == 0);

    assert(
        tetris_cc_env_input_ex(
            env, static_cast<int>(Action::Left), &input_result) == 1);
    assert(input_result.action_succeeded == 1);
    assert(
        tetris_cc_env_meta(
            env, nullptr, nullptr, nullptr, nullptr, nullptr, &lock_timer, &lock_resets) == 1);
    assert(lock_timer == 0);
    assert(lock_resets == 1);

    tetris_cc_env_step_result tick_result{};
    assert(tetris_cc_env_tick_ex(env, &tick_result) == 1);
    assert(tick_result.piece_locked == 0);
    assert(
        tetris_cc_env_meta(
            env, nullptr, nullptr, nullptr, nullptr, nullptr, &lock_timer, &lock_resets) == 1);
    assert(lock_timer == 1);

    assert(
        tetris_cc_env_input_ex(
            env, static_cast<int>(Action::HardDrop), &input_result) == 1);
    assert(input_result.action_succeeded == 1);
    assert(input_result.piece_locked == 1);

    assert(tetris_cc_env_input_ex(nullptr, 0, &input_result) == 0);
    assert(tetris_cc_env_input_ex(env, -1, &input_result) == 0);
    assert(tetris_cc_env_input_ex(env, 0, nullptr) == 0);
    assert(tetris_cc_env_tick_ex(nullptr, &tick_result) == 0);
    assert(tetris_cc_env_tick_ex(env, nullptr) == 0);

    tetris_cc_env_destroy(env);
}

void test_cc_bot_null_safety() {
    assert(tetris_cc_env_step(nullptr, static_cast<int>(Action::None), nullptr) == 0);
    assert(tetris_cc_bot_sync_from_env(nullptr, nullptr) == 0);
    assert(
        tetris_cc_bot_choose(
            nullptr,
            5,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr) == 0);
    assert(
        tetris_cc_bot_choose_and_apply(
            nullptr,
            nullptr,
            5,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr) == 0);
    assert(
        tetris_cc_bot_rank_actions(
            nullptr,
            nullptr,
            5,
            nullptr,
            0,
            nullptr,
            0,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr) == 0);
}

void test_stable_combined_decisions() {
    constexpr std::size_t kPoseActions = 10u * 40u * 4u;
    constexpr std::size_t kActionCount = 2u * kPoseActions;
    assert(tetris_cc_env_decision_action_dim() == kActionCount);

    auto* env = tetris_cc_env_create(314159u);
    auto* bot = tetris_cc_bot_create_default();
    assert(env != nullptr);
    assert(bot != nullptr);

    std::vector<std::uint8_t> mask(kActionCount, 0);
    assert(tetris_cc_env_decision_mask_write(env, mask.data(), mask.size()) == mask.size());
    assert(std::count(mask.begin(), mask.end(), static_cast<std::uint8_t>(1)) > 0);
    assert(std::count(mask.begin() + static_cast<std::ptrdiff_t>(kPoseActions), mask.end(),
                      static_cast<std::uint8_t>(1)) > 0);

    assert(tetris_cc_bot_sync_from_env(bot, env) == 1);
    int use_hold = 0;
    std::size_t placement_index = 0;
    assert(tetris_cc_bot_choose(
               bot, 1, &use_hold, &placement_index, nullptr, nullptr, nullptr, nullptr, nullptr) ==
           1);

    std::size_t action = 0;
    assert(tetris_cc_env_decision_action_for_choice(
               env, use_hold, placement_index, &action) == 1);
    assert(action < mask.size());
    assert(mask[action] == 1);

    int decoded_hold = 0;
    std::size_t decoded_index = 0;
    int x = 0;
    int y = 0;
    int rotation = 0;
    assert(tetris_cc_env_decision_get(
               env, action, &decoded_hold, &decoded_index, &x, &y, &rotation) == 1);
    assert(decoded_hold == use_hold);
    assert(decoded_index == placement_index);
    const auto pose =
        (static_cast<std::size_t>(rotation) * 40u + static_cast<std::size_t>(y)) * 10u +
        static_cast<std::size_t>(x);
    assert(action == (use_hold != 0 ? kPoseActions : 0u) + pose);

    float reward = 0.0f;
    int lines = 0;
    int game_over = 0;
    int used_hold = 0;
    std::size_t applied_index = 0;
    assert(tetris_cc_env_apply_decision(
               env,
               action,
               &reward,
               &lines,
               &game_over,
               &used_hold,
               &applied_index) == 1);
    assert(std::isfinite(reward));
    assert(lines >= 0 && lines <= 4);
    assert(used_hold == use_hold);
    assert(applied_index == placement_index);

    tetris_cc_bot_destroy(bot);
    tetris_cc_env_destroy(env);
}

void test_cc_bot_rank_actions_mapping() {
    auto* env = tetris_cc_env_create(13579);
    auto* bot = tetris_cc_bot_create_default();
    assert(env != nullptr);
    assert(bot != nullptr);
    assert(tetris_cc_bot_sync_from_env(bot, env) == 1);

    std::array<float, 97> scores{};
    std::array<std::uint8_t, 97> legal{};
    std::uint64_t nodes = 0;
    double think_ms = 0.0;
    double nps = 0.0;
    int budget_miss = 0;
    int placement_count_raw = 0;
    int placement_overflow = 0;
    int unexpanded_count = 0;

    const int ok = tetris_cc_bot_rank_actions(
        bot,
        env,
        1,
        scores.data(),
        scores.size(),
        legal.data(),
        legal.size(),
        &nodes,
        &think_ms,
        &nps,
        &budget_miss,
        &placement_count_raw,
        &placement_overflow,
        &unexpanded_count);
    assert(ok == 1);
    assert(placement_count_raw >= 0);
    assert((placement_overflow != 0) == (placement_count_raw > 96));
    assert(think_ms >= 0.0);
    assert(nps >= 0.0);
    assert(unexpanded_count >= 0);

    int has_hold = 0;
    int hold_piece = -1;
    int hold_available = 0;
    assert(tetris_cc_env_hold_piece(env, &has_hold, &hold_piece, &hold_available) == 1);
    const int expected_legal =
        std::min<int>(placement_count_raw, 96) + (hold_available ? 1 : 0);
    int legal_count = 0;
    for (std::size_t i = 0; i < legal.size(); ++i) {
        if (legal[i] != 0) {
            ++legal_count;
            assert(std::isfinite(scores[i]));
        }
    }
    assert(legal_count == expected_legal);
    assert(expected_legal > 1);
    assert(unexpanded_count > 0);

    // Short-buffer safety.
    std::array<float, 96> short_scores{};
    std::array<std::uint8_t, 96> short_legal{};
    assert(
        tetris_cc_bot_rank_actions(
            bot,
            env,
            5,
            short_scores.data(),
            short_scores.size(),
            short_legal.data(),
            short_legal.size(),
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr) == 0);

    tetris_cc_bot_destroy(bot);
    tetris_cc_env_destroy(env);
}

void test_cc2_movegen_zero_safe_fast_mode() {
    using namespace tetris_v2::cc2;

    tetris_v2::cc2::Board empty{};
    for (const auto piece : tetris_v2::kPlayablePieces) {
        const auto moves = find_moves(empty, piece);
        assert(!moves.empty());
        for (const auto& move : moves) {
            assert(move.first.location.piece == piece);
        }
    }

    tetris_v2::cc2::Board near_empty{};
    near_empty.cols[0] = 0x1ull;
    near_empty.cols[4] = 0x3ull;
    near_empty.cols[9] = (1ull << 15);
    for (const auto piece : tetris_v2::kPlayablePieces) {
        const auto moves = find_moves(near_empty, piece);
        assert(!moves.empty());
        for (const auto& move : moves) {
            assert(move.first.location.piece == piece);
        }
    }
}

void test_cc2_combo_progression() {
    using namespace tetris_v2::cc2;

    GameState state{};
    const Placement line_clear{
        PieceLocation{Piece::I, tetris_v2::cc2::Rotation::North, 4, 0},
        Spin::None};

    auto prepare_single_line = [&] {
        state.board = tetris_v2::cc2::Board{};
        for (std::size_t x = 0; x < state.board.cols.size(); ++x) {
            if (x < 3 || x > 6) {
                state.board.cols[x] = 1ull;
            }
        }
    };

    prepare_single_line();
    const auto first = state.advance(Piece::I, line_clear);
    assert(first.lines_cleared == 1);
    assert(first.combo == 1);
    assert(state.combo == 1);

    prepare_single_line();
    const auto second = state.advance(Piece::I, line_clear);
    assert(second.lines_cleared == 1);
    assert(second.combo == 2);
    assert(state.combo == 2);

    state.combo = 255;
    prepare_single_line();
    const auto saturated = state.advance(Piece::I, line_clear);
    assert(saturated.combo == 255);
    assert(state.combo == 255);

    const Placement no_clear{
        PieceLocation{Piece::O, tetris_v2::cc2::Rotation::North, 4, 0},
        Spin::None};
    const auto reset = state.advance(Piece::O, no_clear);
    assert(reset.lines_cleared == 0);
    assert(reset.combo == 0);
    assert(state.combo == 0);
}

void test_cc2_hold_state_and_dag_edges() {
    using namespace tetris_v2::cc2;

    const auto i_moves = find_moves(tetris_v2::cc2::Board{}, Piece::I);
    assert(!i_moves.empty());

    GameState advanced{};
    advanced.reserve = Piece::None;
    advanced.hold_available = true;
    const auto held = advanced.advance(Piece::T, i_moves.front().first, true);
    assert(held.placement.location.piece == Piece::I);
    assert(advanced.reserve == Piece::T);
    assert(advanced.hold_available);
    assert(!piece_mask_contains(advanced.bag_mask, Piece::T));
    assert(!piece_mask_contains(advanced.bag_mask, Piece::I));

    FreestyleWeights weights{};
    weights.wasted_t = -10000.0f;

    GameState empty_hold{};
    empty_hold.reserve = Piece::None;
    empty_hold.hold_available = true;
    Dag empty_hold_dag;
    empty_hold_dag.reset(empty_hold, {Piece::T, Piece::I, Piece::O}, false);
    empty_hold_dag.do_work(weights, 0.0);
    const auto empty_hold_suggestion = empty_hold_dag.suggest();
    assert(empty_hold_suggestion.valid);
    assert(empty_hold_suggestion.use_hold);
    assert(empty_hold_suggestion.placement.location.piece == Piece::I);

    GameState unavailable{};
    unavailable.reserve = Piece::I;
    unavailable.hold_available = false;
    Dag unavailable_dag;
    unavailable_dag.reset(unavailable, {Piece::T, Piece::O}, false);
    unavailable_dag.do_work(weights, 0.0);
    const auto unavailable_suggestion = unavailable_dag.suggest();
    assert(unavailable_suggestion.valid);
    assert(!unavailable_suggestion.use_hold);
    assert(unavailable_suggestion.placement.location.piece == Piece::T);
}

void test_cc_bot_rank_actions_stress() {
    auto* env = tetris_cc_env_create(424242);
    auto* bot = tetris_cc_bot_create_default();
    assert(env != nullptr);
    assert(bot != nullptr);
    assert(tetris_cc_bot_sync_from_env(bot, env) == 1);

    std::array<float, 97> scores{};
    std::array<std::uint8_t, 97> legal{};

    auto run_budget = [&](int think_ms, int iterations) {
        for (int i = 0; i < iterations; ++i) {
            std::uint64_t nodes = 0;
            double elapsed_ms = 0.0;
            double nps = 0.0;
            int budget_miss = 0;
            int placement_count_raw = 0;
            int placement_overflow = 0;
            int unexpanded_count = 0;
            const int ok = tetris_cc_bot_rank_actions(
                bot,
                env,
                think_ms,
                scores.data(),
                scores.size(),
                legal.data(),
                legal.size(),
                &nodes,
                &elapsed_ms,
                &nps,
                &budget_miss,
                &placement_count_raw,
                &placement_overflow,
                &unexpanded_count);
            assert(ok == 1);
            assert(elapsed_ms >= 0.0);
            assert(nps >= 0.0);
            assert(unexpanded_count >= 0);

            std::size_t chosen = 96;
            int legal_count = 0;
            for (std::size_t idx = 0; idx < legal.size(); ++idx) {
                if (legal[idx] != 0) {
                    ++legal_count;
                    assert(std::isfinite(scores[idx]));
                    if (chosen == 96 && idx < 96) {
                        chosen = idx;
                    }
                }
            }
            assert(legal_count > 0);

            float reward = 0.0f;
            if (chosen < 96) {
                int lines = 0;
                int game_over = 0;
                const int ok_apply = tetris_cc_env_apply_placement_index(
                    env, chosen, &reward, &lines, &game_over);
                assert(ok_apply == 1);
                assert(lines >= 0 && lines <= 4);
                if (game_over != 0) {
                    tetris_cc_env_reset(env, 424242u + static_cast<std::uint32_t>(i + think_ms + 1));
                }
            } else {
                const int ok_hold = tetris_cc_env_hold(env, &reward);
                if (ok_hold == 0) {
                    int game_over = 0;
                    assert(tetris_cc_env_meta(env, &game_over, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) == 1);
                    assert(game_over == 1);
                    tetris_cc_env_reset(env, 424242u + static_cast<std::uint32_t>(i + think_ms + 1));
                }
            }
            assert(tetris_cc_bot_sync_from_env(bot, env) == 1);
        }
    };

    run_budget(1, 48);
    run_budget(10, 12);

    tetris_cc_bot_destroy(bot);
    tetris_cc_env_destroy(env);
}

void test_cc_bot_loop_and_budget_scaling() {
    auto* env = tetris_cc_env_create(24680);
    auto* bot = tetris_cc_bot_create_default();
    assert(env != nullptr);
    assert(bot != nullptr);
    assert(tetris_cc_bot_sync_from_env(bot, env) == 1);

    int topouts = 0;
    for (int i = 0; i < 300; ++i) {
        float reward = 0.0f;
        int lines = 0;
        int game_over = 0;
        int used_hold = 0;
        std::size_t placement_index = 0;
        float score = 0.0f;
        std::uint64_t nodes = 0;
        double think_ms = 0.0;
        double nps = 0.0;
        int budget_miss = 0;

        int ok = tetris_cc_bot_choose_and_apply(
            bot,
            env,
            0,
            &reward,
            &lines,
            &game_over,
            &used_hold,
            &placement_index,
            &score,
            &nodes,
            &think_ms,
            &nps,
            &budget_miss);
        assert(ok == 1);
        assert(lines >= 0 && lines <= 4);
        assert(used_hold == 0 || used_hold == 1);
        assert(std::isfinite(score));
        assert(nodes > 0);
        assert(think_ms >= 0.0);
        assert(nps >= 0.0);
        assert(budget_miss == 0);

        if (game_over != 0) {
            ++topouts;
            tetris_cc_env_reset(env, 24680u + static_cast<std::uint32_t>(i + 1));
            assert(tetris_cc_bot_sync_from_env(bot, env) == 1);
        }
    }
    assert(topouts >= 0);

    auto avg_think = [&](int budget_ms) {
        double sum = 0.0;
        int samples = 0;
        for (int i = 0; i < 6; ++i) {
            int use_hold = 0;
            std::size_t placement_index = 0;
            float score = 0.0f;
            std::uint64_t nodes = 0;
            double think_ms = 0.0;
            double nps = 0.0;
            int budget_miss = 0;
            int ok = tetris_cc_bot_choose(
                bot,
                budget_ms,
                &use_hold,
                &placement_index,
                &score,
                &nodes,
                &think_ms,
                &nps,
                &budget_miss);
            assert(ok == 1);
            assert(std::isfinite(score));
            assert(think_ms >= 0.0);
            assert(budget_miss == 0 || budget_miss == 1);
            if (i > 0) {
                sum += think_ms;
                ++samples;
            }
        }
        return sum / static_cast<double>(std::max(1, samples));
    };

    const double avg_1ms = avg_think(1);
    const double avg_20ms = avg_think(20);
    const double avg_50ms = avg_think(50);
    assert(avg_1ms >= 0.0);
    assert(avg_20ms >= avg_1ms + 2.0);
    assert(avg_50ms >= avg_20ms + 4.0);

    tetris_cc_bot_destroy(bot);
    tetris_cc_env_destroy(env);
}

}  // namespace

int main() {
    test_portable_bit_ops();
    test_cc_smoke_and_state_exports();
    test_cc_matches_cpp_env_for_basic_state();
    test_cc_snapshot_restore_keeps_future_deterministic();
    test_cc_rotation_trace_disables_rotate180();
    test_cc_play_mode_rich_steps_and_ghost();
    test_cc_legacy_step_behavior_is_preserved();
    test_cc_zero_time_input_and_tick_bridge();
    test_cc2_movegen_zero_safe_fast_mode();
    test_cc2_combo_progression();
    test_cc2_hold_state_and_dag_edges();
    test_cc_bot_null_safety();
    test_stable_combined_decisions();
    test_cc_bot_rank_actions_mapping();
    test_cc_bot_rank_actions_stress();
    test_cc_bot_loop_and_budget_scaling();
    return 0;
}
