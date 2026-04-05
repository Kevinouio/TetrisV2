#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <deque>

#include "tetris_v2/env.hpp"
#include "tetris_v2/piece_defs.hpp"

namespace {

using namespace tetris_v2;

EnvState make_clean_state(
    ModernTetrisEnv& env,
    Piece active_piece,
    const std::deque<Piece>& queue = std::deque<Piece>{}) {
    EnvState state = env.snapshot().state;
    state.board.clear();
    for (auto& row : state.piece_ids) {
        row.fill(-1);
    }
    state.active = spawn_piece(active_piece);
    state.hold.reset();
    state.hold_available = true;
    state.queue = queue;
    if (state.queue.empty()) {
        state.queue.push_back(Piece::I);
    }
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
    state.last_clear_spin_type = SpinType::None;
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
    state.blitz_lines_to_next = 3;
    state.blitz_time_remaining_ms = 120000;
    state.blitz_timed_out = false;
    return state;
}

void fill_row_with_gap(Board& board, int y, int gap_x) {
    for (int x = 0; x < Board::kWidth; ++x) {
        board.set_cell(x, y, x != gap_x);
    }
}

void fill_row_with_two_gaps(Board& board, int y, int gap_a, int gap_b) {
    for (int x = 0; x < Board::kWidth; ++x) {
        board.set_cell(x, y, x != gap_a && x != gap_b);
    }
}

const PlacementOption* find_placement_with_lines(
    const std::vector<PlacementOption>& options,
    int lines) {
    auto it = std::find_if(options.begin(), options.end(), [&](const PlacementOption& opt) {
        return opt.lines_cleared == lines;
    });
    if (it == options.end()) {
        return nullptr;
    }
    return &(*it);
}

const PlacementOption* require_lines(ModernTetrisEnv& env, int lines) {
    auto options = env.enumerate_active_piece_placements();
    auto* placement = find_placement_with_lines(options, lines);
    assert(placement != nullptr);
    return placement;
}

void test_combo_multiplier_rounding_for_double_combo_one() {
    EnvConfig cfg{};
    cfg.mode = GameMode::Zen;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;
    ModernTetrisEnv env(cfg);

    auto state = make_clean_state(env, Piece::O, {Piece::O, Piece::I, Piece::I});
    fill_row_with_two_gaps(state.board, 0, 4, 5);
    fill_row_with_two_gaps(state.board, 1, 4, 5);
    fill_row_with_two_gaps(state.board, 2, 4, 5);
    fill_row_with_two_gaps(state.board, 3, 4, 5);
    env.restore(state);

    auto options1 = env.enumerate_active_piece_placements();
    auto* first_double = find_placement_with_lines(options1, 2);
    assert(first_double != nullptr);
    auto s1 = env.apply_placement(first_double->placement);
    assert(s1.action_succeeded);
    assert(s1.lines_cleared == 2);
    assert(s1.attack_base == 1);
    assert(std::fabs(s1.attack_combo_scaled - 1.0f) <= 1e-6f);
    assert(s1.attack_rounded == 1);

    auto options2 = env.enumerate_active_piece_placements();
    auto* second_double = find_placement_with_lines(options2, 2);
    assert(second_double != nullptr);
    auto s2 = env.apply_placement(second_double->placement);
    assert(s2.action_succeeded);
    assert(s2.lines_cleared == 2);
    assert(s2.attack_base == 1);
    assert(std::fabs(s2.attack_combo_scaled - 1.25f) <= 1e-4f);
    assert(s2.attack_rounded == 1);
}

void test_blitz_scoring_b2b_combo_and_multiplier() {
    EnvConfig cfg{};
    cfg.mode = GameMode::Scoring;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;
    ModernTetrisEnv env(cfg);

    auto state = make_clean_state(env, Piece::I, {Piece::I, Piece::I, Piece::O});
    for (int y = 0; y < 8; ++y) {
        fill_row_with_gap(state.board, y, 4);
    }
    state.board.set_cell(0, 10, true);  // keep board non-empty to avoid all-clear bonus.
    env.restore(state);

    auto* quad1 = require_lines(env, 4);
    auto s1 = env.apply_placement(quad1->placement);
    assert(s1.action_succeeded);
    assert(s1.lines_cleared == 4);
    assert(static_cast<int>(std::lround(s1.reward)) == 800);

    auto* quad2 = require_lines(env, 4);
    auto s2 = env.apply_placement(quad2->placement);
    assert(s2.action_succeeded);
    assert(s2.lines_cleared == 4);
    assert(static_cast<int>(std::lround(s2.reward)) == 1250);  // 800*1.5 + 50 combo.
    assert(env.state().blitz_score_total == 2050);
}

void test_blitz_all_clear_bonus_additive() {
    EnvConfig cfg{};
    cfg.mode = GameMode::Scoring;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;
    ModernTetrisEnv env(cfg);

    auto state = make_clean_state(env, Piece::O, {Piece::I});
    fill_row_with_two_gaps(state.board, 0, 4, 5);
    fill_row_with_two_gaps(state.board, 1, 4, 5);
    env.restore(state);

    auto* dbl = require_lines(env, 2);
    auto step = env.apply_placement(dbl->placement);
    assert(step.action_succeeded);
    assert(step.all_clear);
    assert(static_cast<int>(std::lround(step.reward)) == 3800);  // (300 + 3500) * level 1.
    assert(env.state().blitz_score_total == 3800);
}

void test_blitz_spin_zero_scores() {
    EnvConfig cfg{};
    cfg.mode = GameMode::Scoring;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;
    ModernTetrisEnv env(cfg);

    auto state = make_clean_state(env, Piece::T, {Piece::I});
    state.active = ActivePiece{Piece::T, Rotation::North, 4, 1};
    state.spin_eligible = true;
    state.rotated_this_piece = true;
    state.last_rotate_kick_index = 0;
    state.board.set_cell(3, 0, true);
    state.board.set_cell(4, 0, true);
    state.board.set_cell(5, 0, true);
    state.board.set_cell(3, 2, true);
    state.board.set_cell(5, 2, true);
    env.restore(state);

    auto step = env.step(Action::HardDrop);
    assert(step.action_succeeded);
    assert(step.lines_cleared == 0);
    assert(step.spin_type == SpinType::Full);
    assert(static_cast<int>(std::lround(step.reward)) == 400);
    assert(env.state().blitz_score_total == 400);
}

void test_blitz_level_progression_and_clamp() {
    EnvConfig cfg{};
    cfg.mode = GameMode::Scoring;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;
    ModernTetrisEnv env(cfg);

    auto state = make_clean_state(env, Piece::O, {Piece::I});
    state.total_lines_cleared = 2;
    fill_row_with_two_gaps(state.board, 0, 4, 5);
    state.board.set_cell(0, 5, true);
    env.restore(state);

    auto* single = require_lines(env, 1);
    auto step = env.apply_placement(single->placement);
    assert(step.action_succeeded);
    assert(step.lines_cleared == 1);
    assert(step.blitz_level == 2);
    assert(step.blitz_lines_to_next == 5);

    state = make_clean_state(env, Piece::O, {Piece::I});
    state.total_lines_cleared = 500;
    env.restore(state);
    auto idle = env.step(Action::None);
    assert(idle.blitz_level == 15);
    assert(idle.blitz_lines_to_next == 0);
}

void test_blitz_gravity_table_lookup_and_clamp() {
    EnvConfig cfg{};
    cfg.mode = GameMode::Scoring;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;
    ModernTetrisEnv env(cfg);

    auto state = make_clean_state(env, Piece::I, {Piece::I});
    state.total_lines_cleared = 0;
    env.restore(state);
    const double level1_gravity = 1.0 / 1.0 / 60.0;
    assert(std::fabs(static_cast<double>(env.config().gravity_per_step) - level1_gravity) <= 1e-6);

    state = make_clean_state(env, Piece::I, {Piece::I});
    state.total_lines_cleared = 260;
    env.restore(state);
    const double level15_gravity = (1.0 / 0.00024) / 60.0;
    assert(std::fabs(static_cast<double>(env.config().gravity_per_step) - level15_gravity) <= 1e-5);

    state = make_clean_state(env, Piece::I, {Piece::I});
    state.total_lines_cleared = 500;
    env.restore(state);
    assert(std::fabs(static_cast<double>(env.config().gravity_per_step) - level15_gravity) <= 1e-5);
}

void test_blitz_timeout_sets_game_over() {
    EnvConfig cfg{};
    cfg.mode = GameMode::Scoring;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;
    ModernTetrisEnv env(cfg);

    auto state = make_clean_state(env, Piece::I, {Piece::I});
    state.blitz_time_remaining_ms = 0;
    env.restore(state);

    auto step = env.step(Action::None);
    assert(step.game_over);
    assert(!step.top_out);
    assert(step.timed_out);
}

void test_zen_mode_surge_release_on_break() {
    EnvConfig cfg{};
    cfg.mode = GameMode::Zen;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;
    ModernTetrisEnv env(cfg);

    auto state = make_clean_state(env, Piece::O, {Piece::O, Piece::I, Piece::I});
    fill_row_with_two_gaps(state.board, 0, 4, 5);
    fill_row_with_two_gaps(state.board, 1, 4, 5);
    state.board.set_cell(0, 3, true);  // prevent all-clear bonus path.
    state.back_to_back = true;
    state.b2b_streak = 4;
    state.b2b_surge_charge = 4;
    env.restore(state);

    auto options = env.enumerate_active_piece_placements();
    auto* double_clear = find_placement_with_lines(options, 2);
    assert(double_clear != nullptr);
    auto step = env.apply_placement(double_clear->placement);
    assert(step.action_succeeded);
    assert(step.lines_cleared == 2);
    assert(!step.difficult_clear);
    assert(step.surge_release == 4);
    assert(step.surge_charge == 0);
    assert(step.b2b_streak == 0);
    assert(step.attack_total == 5);
}

void test_all_clear_bonus_and_b2b_plus_two() {
    EnvConfig cfg{};
    cfg.mode = GameMode::Zen;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;
    ModernTetrisEnv env(cfg);

    auto state = make_clean_state(env, Piece::O, {Piece::I, Piece::I});
    fill_row_with_two_gaps(state.board, 0, 4, 5);
    fill_row_with_two_gaps(state.board, 1, 4, 5);
    env.restore(state);

    auto options = env.enumerate_active_piece_placements();
    auto* double_clear = find_placement_with_lines(options, 2);
    assert(double_clear != nullptr);
    auto step = env.apply_placement(double_clear->placement);
    assert(step.action_succeeded);
    assert(step.lines_cleared == 2);
    assert(step.all_clear);
    assert(step.attack_all_clear_bonus == 5);
    assert(step.b2b_streak == 2);
}

void test_all_mini_plus_non_t_with_rotation_and_immobility() {
    EnvConfig cfg{};
    cfg.mode = GameMode::Zen;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;
    ModernTetrisEnv env(cfg);

    auto state = make_clean_state(env, Piece::O, {Piece::I, Piece::I});
    fill_row_with_two_gaps(state.board, 0, 4, 5);
    fill_row_with_two_gaps(state.board, 1, 4, 5);
    state.board.set_cell(0, 3, true);  // keep board non-empty after clear.
    env.restore(state);

    auto rotate = env.step(Action::RotateCW);
    assert(rotate.action_succeeded);

    auto options = env.enumerate_active_piece_placements();
    auto* double_clear = find_placement_with_lines(options, 2);
    assert(double_clear != nullptr);
    auto step = env.apply_placement(double_clear->placement);
    assert(step.action_succeeded);
    assert(step.lines_cleared == 2);
    assert(step.spin_type == SpinType::Mini);
    assert(step.spin_clear);
    assert(step.difficult_clear);
    assert(step.rotated_before_lock);
    assert(step.immobile_lock);
}

void test_versus_base_attacks_for_line_clears() {
    auto run_case = [](Piece piece, int target_lines, int expected_attack, auto setup_board) {
        EnvConfig cfg{};
        cfg.mode = GameMode::Versus;
        cfg.gravity_per_step = 0.0f;
        cfg.lock_delay_steps = 999999;
        ModernTetrisEnv env(cfg);
        auto state = make_clean_state(env, piece, {piece, Piece::I, Piece::I});
        setup_board(state.board);
        env.restore(state);

        auto* placement = require_lines(env, target_lines);
        auto step = env.apply_placement(placement->placement);
        assert(step.action_succeeded);
        assert(step.lines_cleared == target_lines);
        assert(step.attack_total == expected_attack);
    };

    run_case(Piece::O, 1, 0, [](Board& b) {
        fill_row_with_two_gaps(b, 0, 4, 5);
        b.set_cell(0, 2, true);
    });
    run_case(Piece::O, 2, 1, [](Board& b) {
        fill_row_with_two_gaps(b, 0, 4, 5);
        fill_row_with_two_gaps(b, 1, 4, 5);
        b.set_cell(0, 3, true);
    });
    run_case(Piece::I, 3, 2, [](Board& b) {
        fill_row_with_gap(b, 0, 4);
        fill_row_with_gap(b, 1, 4);
        fill_row_with_gap(b, 2, 4);
        b.set_cell(0, 5, true);
    });
    run_case(Piece::I, 4, 4, [](Board& b) {
        fill_row_with_gap(b, 0, 4);
        fill_row_with_gap(b, 1, 4);
        fill_row_with_gap(b, 2, 4);
        fill_row_with_gap(b, 3, 4);
        b.set_cell(0, 6, true);
    });
}

void test_versus_combo_thresholds() {
    EnvConfig cfg{};
    cfg.mode = GameMode::Versus;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;
    ModernTetrisEnv env(cfg);

    auto state = make_clean_state(env, Piece::O, {Piece::O, Piece::O, Piece::O, Piece::O, Piece::O, Piece::I});
    for (int y = 0; y <= 9; ++y) {
        fill_row_with_two_gaps(state.board, y, 4, 5);
    }
    state.board.set_cell(0, 12, true);
    env.restore(state);

    const std::array<int, 5> expected_attack{1, 1, 2, 2, 3};
    for (int i = 0; i < static_cast<int>(expected_attack.size()); ++i) {
        auto* placement = require_lines(env, 2);
        auto step = env.apply_placement(placement->placement);
        assert(step.action_succeeded);
        assert(step.lines_cleared == 2);
        assert(step.attack_total == expected_attack[static_cast<std::size_t>(i)]);
    }
}

void test_versus_b2b_bonus_and_break_behavior() {
    EnvConfig cfg{};
    cfg.mode = GameMode::Versus;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;
    ModernTetrisEnv env(cfg);

    auto state = make_clean_state(env, Piece::I, {Piece::O, Piece::I});
    for (int y = 0; y <= 3; ++y) {
        fill_row_with_gap(state.board, y, 4);
    }
    state.back_to_back = true;
    state.b2b_streak = 1;
    state.board.set_cell(0, 6, true);
    env.restore(state);

    auto* tetris = require_lines(env, 4);
    auto step1 = env.apply_placement(tetris->placement);
    assert(step1.action_succeeded);
    assert(step1.lines_cleared == 4);
    assert(step1.attack_b2b_bonus == 1);
    assert(step1.attack_total == 5);
    assert(step1.b2b_streak >= 2);

    auto options2 = env.enumerate_active_piece_placements();
    auto single_it = std::find_if(options2.begin(), options2.end(), [](const PlacementOption& opt) {
        return opt.lines_cleared == 1 && !opt.difficult_clear_candidate;
    });
    assert(single_it != options2.end());
    auto step2 = env.apply_placement(single_it->placement);
    assert(step2.action_succeeded);
    assert(step2.lines_cleared == 1);
    assert(step2.attack_b2b_bonus == 0);
    assert(step2.b2b_streak == 0);
}

void test_versus_all_clear_bonus_plus_seven() {
    EnvConfig cfg{};
    cfg.mode = GameMode::Versus;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;
    ModernTetrisEnv env(cfg);

    auto state = make_clean_state(env, Piece::O, {Piece::I, Piece::I});
    fill_row_with_two_gaps(state.board, 0, 4, 5);
    fill_row_with_two_gaps(state.board, 1, 4, 5);
    env.restore(state);

    auto* placement = require_lines(env, 2);
    auto step = env.apply_placement(placement->placement);
    assert(step.action_succeeded);
    assert(step.lines_cleared == 2);
    assert(step.all_clear);
    assert(step.attack_base == 1);
    assert(step.attack_all_clear_bonus == 7);
    assert(step.attack_total == 8);
}

void test_versus_mini_base_zero_with_additives() {
    EnvConfig cfg{};
    cfg.mode = GameMode::Versus;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;
    ModernTetrisEnv env(cfg);

    auto state = make_clean_state(env, Piece::O, {Piece::I, Piece::I});
    fill_row_with_two_gaps(state.board, 0, 4, 5);
    fill_row_with_two_gaps(state.board, 1, 4, 5);
    state.board.set_cell(0, 3, true);  // keep board non-empty after clear.
    state.combo = 1;                    // next lock combo index becomes 2 -> +1.
    state.back_to_back = true;
    state.b2b_streak = 1;               // active b2b chain.
    env.restore(state);

    auto rotate = env.step(Action::RotateCW);
    assert(rotate.action_succeeded);

    auto* placement = require_lines(env, 2);
    auto step = env.apply_placement(placement->placement);
    assert(step.action_succeeded);
    assert(step.spin_type == SpinType::Mini);
    assert(step.attack_base == 0);
    assert(step.attack_b2b_bonus == 1);
    assert(step.attack_total == 2);  // 0 base + 1 combo + 1 b2b.
}

}  // namespace

int main() {
    test_combo_multiplier_rounding_for_double_combo_one();
    test_blitz_scoring_b2b_combo_and_multiplier();
    test_blitz_all_clear_bonus_additive();
    test_blitz_spin_zero_scores();
    test_blitz_level_progression_and_clamp();
    test_blitz_gravity_table_lookup_and_clamp();
    test_blitz_timeout_sets_game_over();
    test_zen_mode_surge_release_on_break();
    test_all_clear_bonus_and_b2b_plus_two();
    test_all_mini_plus_non_t_with_rotation_and_immobility();
    test_versus_base_attacks_for_line_clears();
    test_versus_combo_thresholds();
    test_versus_b2b_bonus_and_break_behavior();
    test_versus_all_clear_bonus_plus_seven();
    test_versus_mini_base_zero_with_additives();
    return 0;
}
