#include <algorithm>
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

std::array<float, 6> features_from_visible_board(
    const std::vector<std::uint8_t>& board_after,
    int y_pos,
    int lines_removed) {
    assert(board_after.size() == 200);
    std::array<int, 10> cols{};
    int holes = 0;
    int bumpiness = 0;
    for (int c = 0; c < 10; ++c) {
        bool block_seen = false;
        for (int r = 0; r < 20; ++r) {
            auto v = board_after[static_cast<std::size_t>(r * 10 + c)];
            if (v > 0 && !block_seen) {
                block_seen = true;
                cols[static_cast<std::size_t>(c)] = 20 - r;
            }
            if (v == 0 && block_seen) {
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
    for (int i = 1; i < 9; ++i) {
        if ((cols[static_cast<std::size_t>(i - 1)] - cols[static_cast<std::size_t>(i)] >= 3) &&
            (cols[static_cast<std::size_t>(i + 1)] - cols[static_cast<std::size_t>(i)] >= 3)) {
            pillar = 1;
            break;
        }
    }
    if (pillar == 0 && ((cols[1] - cols[0] >= 3) || (cols[8] - cols[9] >= 3))) {
        pillar = 1;
    }
    return std::array<float, 6>{
        static_cast<float>(total_height),
        static_cast<float>(bumpiness),
        static_cast<float>(lines_removed),
        static_cast<float>(holes),
        static_cast<float>(y_pos),
        static_cast<float>(pillar),
    };
}

struct BeamTraceRow {
    int used_hold{0};
    std::size_t placement_index{0};
    float score{0.0f};
};

std::vector<BeamTraceRow> collect_beam_trace(std::uint32_t seed, int steps) {
    std::vector<BeamTraceRow> trace{};
    trace.reserve(static_cast<std::size_t>(std::max(0, steps)));

    auto* env = tetris_cc_env_create(seed);
    auto* bot = tetris_cc_bot_create_default();
    assert(env != nullptr);
    assert(bot != nullptr);

    tetris_cc_env_reset(env, seed);
    assert(tetris_cc_bot_set_backend(bot, TETRIS_CC_BOT_BACKEND_BEAM) == 1);
    assert(tetris_cc_bot_set_beam_config(bot, 2, 8, 1.0, 1, 0, 1, 0) == 1);
    assert(tetris_cc_bot_sync_from_env(bot, env) == 1);

    for (int i = 0; i < steps; ++i) {
        float reward = 0.0f;
        int lines_cleared = 0;
        int game_over = 0;
        int used_hold = 0;
        std::size_t placement_index = 0;
        float score = 0.0f;
        std::uint64_t nodes = 0;
        double think_ms = 0.0;
        double nps = 0.0;
        int budget_miss = 0;

        const int ok = tetris_cc_bot_choose_and_apply_ex(
            bot,
            env,
            20,
            &reward,
            &lines_cleared,
            &game_over,
            &used_hold,
            &placement_index,
            &score,
            &nodes,
            &think_ms,
            &nps,
            &budget_miss);
        assert(ok == 1);

        trace.push_back(BeamTraceRow{used_hold, placement_index, score});
        if (game_over) {
            break;
        }
    }

    tetris_cc_bot_destroy(bot);
    tetris_cc_env_destroy(env);
    return trace;
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

void test_cc_candidate_batch_api_matches_placement_api() {
    auto* env = tetris_cc_env_create(4242);
    assert(env != nullptr);

    const auto candidate_count = tetris_cc_env_candidate_count(env);
    const auto feature_count = candidate_count * 6;
    std::vector<float> features(feature_count, 0.0f);
    auto feature_written = tetris_cc_env_candidate_features_write(env, features.data(), features.size());
    assert(feature_written == feature_count);
    std::vector<tetris_cc_candidate_row> rows(candidate_count);
    auto row_written = tetris_cc_env_candidate_rows_write(env, rows.data(), rows.size());
    assert(row_written == candidate_count);

    for (std::size_t i = 0; i < candidate_count; ++i) {
        int use_hold = 0;
        std::size_t placement_index = 0;
        int piece = -1;
        int rotation = 0;
        int x = 0;
        int y = 0;
        int lines = 0;
        assert(
            tetris_cc_env_candidate_get(
                env,
                i,
                &use_hold,
                &placement_index,
                &piece,
                &rotation,
                &x,
                &y,
                &lines) == 1);
        assert(rows[i].use_hold == use_hold);
        assert(rows[i].placement_index == placement_index);
        assert(rows[i].piece == piece);
        assert(rows[i].rotation == rotation);
        assert(rows[i].x == x);
        assert(rows[i].y == y);
        assert(rows[i].lines_cleared == lines);
        assert(use_hold == 0 || use_hold == 1);
        assert(piece >= 0 && piece <= 6);
        assert(rotation >= 0 && rotation <= 3);
        assert(lines >= 0 && lines <= 4);

        if (use_hold == 0) {
            int px = 0;
            int py = 0;
            int prot = 0;
            int plines = 0;
            assert(
                tetris_cc_env_placement_get(
                    env,
                    placement_index,
                    &px,
                    &py,
                    &prot,
                    &plines) == 1);
            assert(px == x);
            assert(py == y);
            assert(prot == rotation);
            assert(plines == lines);

            std::vector<std::uint8_t> board_after(200, 0);
            auto wrote = tetris_cc_env_placement_board_write(
                env, placement_index, board_after.data(), board_after.size());
            assert(wrote == board_after.size());
            auto expected = features_from_visible_board(board_after, y, lines);
            for (std::size_t j = 0; j < expected.size(); ++j) {
                auto got = features[static_cast<std::size_t>(i * 6 + j)];
                auto got_row = rows[i].features[j];
                assert(std::fabs(got - expected[j]) <= 1e-4f);
                assert(std::fabs(got_row - expected[j]) <= 1e-4f);
            }
        }
    }

    tetris_cc_env_destroy(env);
}

void test_cc_rotation_trace_disables_rotate180() {
    auto* env = tetris_cc_env_create(99);
    assert(env != nullptr);

    const auto cw_count = tetris_cc_env_rotation_trace_count(env, static_cast<int>(Action::RotateCW));
    const auto ccw_count = tetris_cc_env_rotation_trace_count(env, static_cast<int>(Action::RotateCCW));
    const auto r180_count = tetris_cc_env_rotation_trace_count(env, static_cast<int>(Action::Rotate180));
    assert(cw_count >= 0);
    assert(ccw_count >= 0);
    assert(r180_count == 0);

    tetris_cc_env_destroy(env);
}

void test_cc_bot_null_safety() {
    assert(tetris_cc_bot_set_backend(nullptr, TETRIS_CC_BOT_BACKEND_DEPTH) == 0);
    assert(tetris_cc_bot_set_backend(nullptr, TETRIS_CC_BOT_BACKEND_BEAM) == 0);
    assert(tetris_cc_bot_get_backend(nullptr, nullptr) == 0);
    assert(tetris_cc_bot_set_depth_config(nullptr, 1, 1.0, 1, 0, 1, 0) == 0);
    assert(tetris_cc_bot_set_beam_config(nullptr, 2, 8, 1.0, 1, 0, 1, 0) == 0);
    assert(tetris_cc_bot_sync_from_env(nullptr, nullptr) == 0);
    assert(
        tetris_cc_bot_choose_ex(
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
        tetris_cc_bot_choose_and_apply_ex(
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
}

void test_cc_bot_sparse_board_loop_stability() {
    auto* env = tetris_cc_env_create(314159u);
    auto* bot = tetris_cc_bot_create_default();
    assert(env != nullptr);
    assert(bot != nullptr);

    std::array<std::uint8_t, 200> empty_board{};
    empty_board.fill(0);
    assert(
        tetris_cc_env_set_visible_board_mask(
            env, empty_board.data(), empty_board.size(), 1) == 1);
    assert(tetris_cc_bot_sync_from_env(bot, env) == 1);

    for (int i = 0; i < 16; ++i) {
        int used_hold = 0;
        std::size_t placement_index = 0;
        float score = 0.0f;
        std::uint64_t nodes = 0;
        double think_ms = 0.0;
        double nps = 0.0;
        int budget_miss = 0;

        int ok = tetris_cc_bot_choose_ex(
            bot,
            5,
            &used_hold,
            &placement_index,
            &score,
            &nodes,
            &think_ms,
            &nps,
            &budget_miss);
        assert(ok == 1);
        assert(used_hold == 0 || used_hold == 1);
        assert(std::isfinite(score));
        assert(nodes > 0);
        assert(think_ms >= 0.0);
        assert(nps >= 0.0);
        assert(budget_miss == 0 || budget_miss == 1);
    }

    tetris_cc_bot_destroy(bot);
    tetris_cc_env_destroy(env);
}

void test_cc_bot_loop_and_backend_switching() {
    auto* env = tetris_cc_env_create(24680);
    auto* bot = tetris_cc_bot_create_default();
    assert(env != nullptr);
    assert(bot != nullptr);
    int backend = -1;
    assert(tetris_cc_bot_get_backend(bot, &backend) == 1);
    assert(backend == TETRIS_CC_BOT_BACKEND_COLD_CLEAR);
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

        int ok = tetris_cc_bot_choose_and_apply_ex(
            bot,
            env,
            5,
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
        assert(budget_miss == 0 || budget_miss == 1);

        if (game_over != 0) {
            ++topouts;
            tetris_cc_env_reset(env, 24680u + static_cast<std::uint32_t>(i + 1));
            assert(tetris_cc_bot_sync_from_env(bot, env) == 1);
        }
    }
    assert(topouts >= 0);
    assert(tetris_cc_bot_set_backend(bot, TETRIS_CC_BOT_BACKEND_DEPTH) == 1);
    assert(tetris_cc_bot_get_backend(bot, &backend) == 1);
    assert(backend == TETRIS_CC_BOT_BACKEND_DEPTH);
    assert(tetris_cc_bot_set_depth_config(bot, 1, 1.0, 1, 0, 1, 0) == 1);
    assert(tetris_cc_bot_sync_from_env(bot, env) == 1);

    int use_hold_d1 = 0;
    std::size_t placement_d1 = 0;
    float score_d1 = 0.0f;
    std::uint64_t nodes_d1 = 0;
    double think_d1 = 0.0;
    double nps_d1 = 0.0;
    int budget_d1 = 0;
    assert(
        tetris_cc_bot_choose_ex(
            bot,
            1,
            &use_hold_d1,
            &placement_d1,
            &score_d1,
            &nodes_d1,
            &think_d1,
            &nps_d1,
            &budget_d1) == 1);
    assert(nodes_d1 > 0);
    assert(std::isfinite(score_d1));
    assert(think_d1 >= 0.0);
    assert(nps_d1 >= 0.0);
    assert(budget_d1 == 0 || budget_d1 == 1);

    assert(tetris_cc_bot_set_depth_config(bot, 2, 1.0, 1, 1, 1, 0) == 1);
    assert(tetris_cc_bot_sync_from_env(bot, env) == 1);
    int use_hold_d2 = 0;
    std::size_t placement_d2 = 0;
    float score_d2 = 0.0f;
    std::uint64_t nodes_d2 = 0;
    double think_d2 = 0.0;
    double nps_d2 = 0.0;
    int budget_d2 = 0;
    assert(
        tetris_cc_bot_choose_ex(
            bot,
            999,
            &use_hold_d2,
            &placement_d2,
            &score_d2,
            &nodes_d2,
            &think_d2,
            &nps_d2,
            &budget_d2) == 1);
    assert(nodes_d2 >= nodes_d1);
    assert(std::isfinite(score_d2));

    // Determinism check: repeated choose on unchanged snapshot/config should match.
    int use_hold_repeat = 0;
    std::size_t placement_repeat = 0;
    float score_repeat = 0.0f;
    std::uint64_t nodes_repeat = 0;
    double think_repeat = 0.0;
    double nps_repeat = 0.0;
    int budget_repeat = 0;
    assert(tetris_cc_bot_sync_from_env(bot, env) == 1);
    assert(
        tetris_cc_bot_choose_ex(
            bot,
            42,
            &use_hold_repeat,
            &placement_repeat,
            &score_repeat,
            &nodes_repeat,
            &think_repeat,
            &nps_repeat,
            &budget_repeat) == 1);
    assert(use_hold_repeat == use_hold_d2);
    assert(placement_repeat == placement_d2);
    assert(std::fabs(score_repeat - score_d2) <= 1e-6f);

    assert(tetris_cc_bot_set_backend(bot, TETRIS_CC_BOT_BACKEND_BEAM) == 1);
    assert(tetris_cc_bot_get_backend(bot, &backend) == 1);
    assert(backend == TETRIS_CC_BOT_BACKEND_BEAM);
    assert(tetris_cc_bot_set_beam_config(bot, 1, 8, 1.0, 1, 0, 1, 0) == 1);
    assert(tetris_cc_bot_sync_from_env(bot, env) == 1);

    int use_hold_b1 = 0;
    std::size_t placement_b1 = 0;
    float score_b1 = 0.0f;
    std::uint64_t nodes_b1 = 0;
    double think_b1 = 0.0;
    double nps_b1 = 0.0;
    int budget_b1 = 0;
    assert(
        tetris_cc_bot_choose_ex(
            bot,
            1,
            &use_hold_b1,
            &placement_b1,
            &score_b1,
            &nodes_b1,
            &think_b1,
            &nps_b1,
            &budget_b1) == 1);
    assert(nodes_b1 > 0);
    assert(std::isfinite(score_b1));

    assert(tetris_cc_bot_set_beam_config(bot, 2, 8, 1.0, 1, 1, 1, 0) == 1);
    assert(tetris_cc_bot_sync_from_env(bot, env) == 1);
    int use_hold_b2 = 0;
    std::size_t placement_b2 = 0;
    float score_b2 = 0.0f;
    std::uint64_t nodes_b2 = 0;
    double think_b2 = 0.0;
    double nps_b2 = 0.0;
    int budget_b2 = 0;
    assert(
        tetris_cc_bot_choose_ex(
            bot,
            999,
            &use_hold_b2,
            &placement_b2,
            &score_b2,
            &nodes_b2,
            &think_b2,
            &nps_b2,
            &budget_b2) == 1);
    assert(nodes_b2 >= nodes_b1);
    assert(std::isfinite(score_b2));

    int use_hold_beam_repeat = 0;
    std::size_t placement_beam_repeat = 0;
    float score_beam_repeat = 0.0f;
    std::uint64_t nodes_beam_repeat = 0;
    double think_beam_repeat = 0.0;
    double nps_beam_repeat = 0.0;
    int budget_beam_repeat = 0;
    assert(tetris_cc_bot_sync_from_env(bot, env) == 1);
    assert(
        tetris_cc_bot_choose_ex(
            bot,
            42,
            &use_hold_beam_repeat,
            &placement_beam_repeat,
            &score_beam_repeat,
            &nodes_beam_repeat,
            &think_beam_repeat,
            &nps_beam_repeat,
            &budget_beam_repeat) == 1);
    assert(use_hold_beam_repeat == use_hold_b2);
    assert(placement_beam_repeat == placement_b2);
    assert(std::fabs(score_beam_repeat - score_b2) <= 1e-6f);

    assert(tetris_cc_bot_set_backend(bot, TETRIS_CC_BOT_BACKEND_COLD_CLEAR) == 1);
    assert(tetris_cc_bot_get_backend(bot, &backend) == 1);
    assert(backend == TETRIS_CC_BOT_BACKEND_COLD_CLEAR);
    assert(tetris_cc_bot_sync_from_env(bot, env) == 1);
    tetris_cc_bot_destroy(bot);
    tetris_cc_env_destroy(env);
}

void test_beam_multistep_trace_determinism() {
    constexpr int kTraceSteps = 8;
    const std::array<std::uint32_t, 3> seeds{1234u, 5678u, 9012u};

    for (const auto seed : seeds) {
        const auto trace_a = collect_beam_trace(seed, kTraceSteps);
        const auto trace_b = collect_beam_trace(seed, kTraceSteps);
        assert(trace_a.size() == trace_b.size());
        for (std::size_t i = 0; i < trace_a.size(); ++i) {
            assert(trace_a[i].used_hold == trace_b[i].used_hold);
            assert(trace_a[i].placement_index == trace_b[i].placement_index);
            assert(std::fabs(trace_a[i].score - trace_b[i].score) <= 1e-6f);
        }
    }
}

}  // namespace

int main() {
    test_cc_smoke_and_state_exports();
    test_cc_matches_cpp_env_for_basic_state();
    test_cc_snapshot_restore_keeps_future_deterministic();
    test_cc_candidate_batch_api_matches_placement_api();
    test_cc_rotation_trace_disables_rotate180();
    test_cc_bot_null_safety();
    test_cc_bot_sparse_board_loop_stability();
    test_cc_bot_loop_and_backend_switching();
    test_beam_multistep_trace_determinism();
    return 0;
}
