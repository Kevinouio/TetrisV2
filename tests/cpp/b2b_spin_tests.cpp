#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <optional>
#include <string>

#include "tetris_v2/env.hpp"
#include "tetris_v2/piece_defs.hpp"

namespace {

using namespace tetris_v2;

struct KickSpinScenario {
    EnvState state{};
    ActivePiece placement{};
    SpinType spin_type{SpinType::None};
};

bool twist_verbose() {
    const char* raw = std::getenv("TETRIS_TWIST_VERBOSE");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
}

void print_board_with_piece(const Board& board, const ActivePiece& piece, const char* label) {
    std::cout << label << '\n';
    std::array<std::string, Board::kVisibleRows> rows{};
    for (int row = Board::kVisibleRows - 1; row >= 0; --row) {
        std::string line;
        line.reserve(Board::kWidth);
        auto mask = board.row_mask(row);
        for (int x = 0; x < Board::kWidth; ++x) {
            line.push_back((mask & (1u << x)) ? '#' : '.');
        }
        rows[static_cast<std::size_t>((Board::kVisibleRows - 1) - row)] = std::move(line);
    }

    auto cells = piece_cells(piece.piece, piece.rotation);
    for (const auto& c : cells) {
        int x = piece.x + c.x;
        int y = piece.y + c.y;
        if (x < 0 || x >= Board::kWidth || y < 0 || y >= Board::kVisibleRows) {
            continue;
        }
        int row = (Board::kVisibleRows - 1) - y;
        rows[static_cast<std::size_t>(row)][static_cast<std::size_t>(x)] = piece_to_char(piece.piece);
    }

    for (const auto& row : rows) {
        std::cout << row << '\n';
    }

    std::cout << "Active piece="
              << piece_name(piece.piece)
              << " pos=(" << piece.x << "," << piece.y << ") rot="
              << static_cast<int>(piece.rotation) << '\n';
}

bool collides_local(const Board& board, const ActivePiece& piece) {
    auto cells = piece_cells(piece.piece, piece.rotation);
    for (const auto& c : cells) {
        if (board.occupied(piece.x + c.x, piece.y + c.y)) {
            return true;
        }
    }
    return false;
}

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
    state.total_lines_cleared = 0;
    state.lock_timer = 0;
    state.lock_resets_used = 0;
    state.gravity_accumulator = 0.0f;
    state.spin_eligible = false;
    state.last_rotate_used_kick = false;
    state.last_rotate_kick_index = -1;
    state.last_clear_spin = false;
    state.last_clear_spin_type = SpinType::None;
    state.last_clear_difficult = false;
    state.last_clear_b2b_bonus = false;
    return state;
}

void fill_row_with_gap(Board& board, int y, int gap_x) {
    for (int x = 0; x < Board::kWidth; ++x) {
        board.set_cell(x, y, x != gap_x);
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

float expected_spin_reward(SpinType spin_type, int lines) {
    constexpr std::array<float, 3> kMini{100.0f, 200.0f, 400.0f};
    constexpr std::array<float, 4> kFull{400.0f, 800.0f, 1200.0f, 1600.0f};
    if (spin_type == SpinType::Mini) {
        return kMini[static_cast<std::size_t>(lines)];
    }
    if (spin_type == SpinType::Full) {
        return kFull[static_cast<std::size_t>(lines)];
    }
    return 0.0f;
}

std::optional<KickSpinScenario> find_kick_spin_clear_scenario(
    ModernTetrisEnv& env,
    Piece piece) {
    const std::array<Rotation, 4> rotations{
        Rotation::North, Rotation::East, Rotation::South, Rotation::West};

    for (int pattern = 0; pattern < 3; ++pattern) {
        for (int gap = 0; gap < Board::kWidth; ++gap) {
            for (auto rotation : rotations) {
                for (int dx = -3; dx <= 3; ++dx) {
                    for (int dy = -3; dy <= 3; ++dy) {
                        auto state = make_clean_state(env, piece, {piece, Piece::T, Piece::I});
                        state.active.rotation = rotation;
                        fill_row_with_gap(state.board, 0, gap);
                        if (pattern == 1) {
                            fill_row_with_gap(state.board, 1, gap);
                        } else if (pattern == 2 && gap + 1 < Board::kWidth) {
                            fill_row_with_gap(state.board, 1, gap + 1);
                        }
                        state.board.set_cell(state.active.x + dx, state.active.y + dy, true);
                        if (collides_local(state.board, state.active)) {
                            continue;
                        }

                        env.restore(state);
                        auto options = env.enumerate_active_piece_placements();
                        auto it = std::find_if(options.begin(), options.end(), [](const PlacementOption& opt) {
                            return opt.lines_cleared > 0 &&
                                   opt.spin_type != SpinType::None &&
                                   opt.last_rotate_used_kick_path;
                        });
                        if (it != options.end()) {
                            if (twist_verbose()) {
                                std::cout << "\n[b2b_spin] Kick+spin candidate found piece="
                                          << piece_name(piece)
                                          << " pattern=" << pattern
                                          << " gap=" << gap
                                          << " blocker=(" << dx << "," << dy << ")"
                                          << " lines=" << it->lines_cleared
                                          << " spin_type=" << static_cast<int>(it->spin_type)
                                          << " kick=" << it->last_rotate_used_kick_path
                                          << '\n';
                                print_board_with_piece(state.board, state.active, "Board at discovery:");
                            }
                            return KickSpinScenario{state, it->placement, it->spin_type};
                        }
                    }
                }
            }
        }
    }
    return std::nullopt;
}

void test_tetris_to_tetris_gets_b2b_bonus() {
    EnvConfig cfg;
    cfg.seed = 1;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;

    ModernTetrisEnv env(cfg);
    auto state = make_clean_state(env, Piece::I, {Piece::I, Piece::T, Piece::T});
    for (int y = 0; y <= 7; ++y) {
        fill_row_with_gap(state.board, y, 4);
    }
    env.restore(state);

    auto options1 = env.enumerate_active_piece_placements();
    auto* first_tetris = find_placement_with_lines(options1, 4);
    assert(first_tetris != nullptr);
    auto step1 = env.apply_placement(first_tetris->placement);
    assert(step1.action_succeeded);
    assert(step1.lines_cleared == 4);
    assert(step1.difficult_clear);
    assert(!step1.b2b_bonus_applied);
    assert(step1.combo == 0);
    assert(step1.reward == 800.0f);
    assert(env.state().back_to_back);

    auto options2 = env.enumerate_active_piece_placements();
    auto* second_tetris = find_placement_with_lines(options2, 4);
    assert(second_tetris != nullptr);
    auto step2 = env.apply_placement(second_tetris->placement);
    assert(step2.action_succeeded);
    assert(step2.lines_cleared == 4);
    assert(step2.difficult_clear);
    assert(step2.b2b_bonus_applied);
    assert(step2.combo == 1);
    assert(step2.reward == 1250.0f);
    assert(env.state().back_to_back);
}

void test_tetris_then_single_breaks_b2b() {
    EnvConfig cfg;
    cfg.seed = 2;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;

    ModernTetrisEnv env(cfg);
    auto state = make_clean_state(env, Piece::I, {Piece::O, Piece::T, Piece::T});
    for (int y = 0; y <= 3; ++y) {
        fill_row_with_gap(state.board, y, 4);
    }
    for (int x = 0; x < Board::kWidth; ++x) {
        state.board.set_cell(x, 4, x != 4 && x != 5);
    }
    env.restore(state);

    auto options1 = env.enumerate_active_piece_placements();
    auto* first_tetris = find_placement_with_lines(options1, 4);
    assert(first_tetris != nullptr);
    auto step1 = env.apply_placement(first_tetris->placement);
    assert(step1.action_succeeded);
    assert(step1.lines_cleared == 4);
    assert(env.state().back_to_back);

    auto options2 = env.enumerate_active_piece_placements();
    auto single_it = std::find_if(options2.begin(), options2.end(), [](const PlacementOption& opt) {
        return opt.lines_cleared == 1 && opt.spin_type == SpinType::None;
    });
    if (single_it == options2.end()) {
        std::cerr << "[b2b_spin] No non-difficult single available after first tetris.\n";
        print_board_with_piece(env.state().board, env.state().active, "Board before expected non-difficult single:");
        assert(false && "No non-difficult single placement found");
    }
    auto step2 = env.apply_placement(single_it->placement);
    assert(step2.action_succeeded);
    assert(step2.lines_cleared == 1);
    if (step2.difficult_clear) {
        std::cerr << "[b2b_spin] Selected single cleared as difficult unexpectedly.\n";
        print_board_with_piece(env.state().board, env.state().active, "Board after unexpected difficult single:");
    }
    assert(!step2.difficult_clear);
    assert(step2.reward == 150.0f);
    assert(!env.state().back_to_back);
}

void test_tetris_zero_then_tetris_preserves_b2b() {
    EnvConfig cfg;
    cfg.seed = 3;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;

    ModernTetrisEnv env(cfg);
    auto state = make_clean_state(env, Piece::I, {Piece::O, Piece::I, Piece::T});
    for (int y = 0; y <= 7; ++y) {
        fill_row_with_gap(state.board, y, 4);
    }
    env.restore(state);

    auto options1 = env.enumerate_active_piece_placements();
    auto* first_tetris = find_placement_with_lines(options1, 4);
    assert(first_tetris != nullptr);
    auto step1 = env.apply_placement(first_tetris->placement);
    assert(step1.action_succeeded);
    assert(step1.lines_cleared == 4);
    assert(step1.combo == 0);
    assert(step1.reward == 800.0f);
    assert(env.state().back_to_back);

    auto options2 = env.enumerate_active_piece_placements();
    auto zero_it = std::find_if(options2.begin(), options2.end(), [](const PlacementOption& opt) {
        return opt.lines_cleared == 0 && opt.placement.x <= 1;
    });
    assert(zero_it != options2.end());
    auto step2 = env.apply_placement(zero_it->placement);
    assert(step2.action_succeeded);
    assert(step2.lines_cleared == 0);
    assert(step2.combo == -1);
    assert(step2.reward == 0.0f);
    assert(env.state().back_to_back);

    auto options3 = env.enumerate_active_piece_placements();
    auto* second_tetris = find_placement_with_lines(options3, 4);
    assert(second_tetris != nullptr);
    auto step3 = env.apply_placement(second_tetris->placement);
    assert(step3.action_succeeded);
    assert(step3.lines_cleared == 4);
    assert(step3.b2b_bonus_applied);
    assert(step3.combo == 0);
    assert(step3.reward == 1200.0f);
    assert(env.state().back_to_back);
}

void test_t_piece_kick_spin_clear() {
    EnvConfig cfg;
    cfg.seed = 4;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;

    ModernTetrisEnv env(cfg);

    auto scenario = find_kick_spin_clear_scenario(env, Piece::T);
    assert(scenario.has_value() && "No kick+spin clear scenario found for T");

    auto state_no_b2b = scenario->state;
    state_no_b2b.back_to_back = false;
    env.restore(state_no_b2b);
    auto clear_no_b2b = env.apply_placement(scenario->placement);
    assert(clear_no_b2b.action_succeeded);
    assert(clear_no_b2b.lines_cleared > 0);
    assert(clear_no_b2b.spin_clear);
    assert(clear_no_b2b.spin_type == scenario->spin_type);
    assert(clear_no_b2b.difficult_clear);
    assert(!clear_no_b2b.b2b_bonus_applied);
    const float base_reward = expected_spin_reward(
        scenario->spin_type,
        clear_no_b2b.lines_cleared);
    assert(clear_no_b2b.reward == base_reward);
    assert(env.state().back_to_back);

    auto state_with_b2b = scenario->state;
    state_with_b2b.back_to_back = true;
    env.restore(state_with_b2b);
    auto clear_with_b2b = env.apply_placement(scenario->placement);
    assert(clear_with_b2b.action_succeeded);
    assert(clear_with_b2b.lines_cleared > 0);
    assert(clear_with_b2b.spin_clear);
    assert(clear_with_b2b.spin_type == scenario->spin_type);
    assert(clear_with_b2b.difficult_clear);
    assert(clear_with_b2b.b2b_bonus_applied);
    assert(clear_with_b2b.reward == base_reward * 1.5f);
}

void test_enumerated_spin_type_matches_applied_result() {
    EnvConfig cfg;
    cfg.seed = 5;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;

    ModernTetrisEnv env(cfg);
    auto check = [&](int occupied_front_corners, SpinType expected) {
        auto state = make_clean_state(env, Piece::T, {Piece::I, Piece::O});
        state.active = ActivePiece{Piece::T, Rotation::North, 4, 1};
        state.spin_eligible = true;
        state.last_rotate_used_kick = false;
        state.last_rotate_kick_index = 0;

        for (int x = 0; x < Board::kWidth; ++x) {
            state.board.set_cell(x, 1, x < 3 || x > 5);
        }
        state.board.set_cell(3, 0, true);
        state.board.set_cell(5, 0, true);
        if (occupied_front_corners >= 1) {
            state.board.set_cell(3, 2, true);
        }
        if (occupied_front_corners >= 2) {
            state.board.set_cell(5, 2, true);
        }

        env.restore(state);
        const auto options = env.enumerate_active_piece_placements();
        const auto it = std::find_if(options.begin(), options.end(), [&](const PlacementOption& option) {
            return option.placement == state.active;
        });
        assert(it != options.end());
        assert(it->lines_cleared == 1);
        assert(it->last_rotate_kick_index_path != 4);
        assert(it->spin_type == expected);

        const auto result = env.apply_placement(it->placement);
        assert(result.action_succeeded);
        assert(result.lines_cleared == it->lines_cleared);
        assert(result.spin_type == it->spin_type);
        assert(result.spin_clear == (expected != SpinType::None));
        assert(result.difficult_clear == (expected != SpinType::None));
        const float expected_reward = expected == SpinType::None
            ? 100.0f
            : expected_spin_reward(expected, 1);
        assert(result.reward == expected_reward);
    };

    check(0, SpinType::None);
    check(1, SpinType::Mini);
    check(2, SpinType::Full);
}

void test_zero_line_spin_scoring_and_b2b_preservation() {
    EnvConfig cfg;
    cfg.seed = 6;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;

    ModernTetrisEnv env(cfg);
    auto check = [&](int occupied_front_corners, int kick_index, SpinType expected) {
        auto state = make_clean_state(env, Piece::T, {Piece::I, Piece::O});
        state.active = ActivePiece{Piece::T, Rotation::North, 4, 1};
        state.spin_eligible = true;
        state.last_rotate_used_kick = kick_index > 0;
        state.last_rotate_kick_index = kick_index;
        state.combo = 4;
        state.back_to_back = true;
        state.board.set_cell(3, 0, true);
        state.board.set_cell(5, 0, true);
        if (occupied_front_corners >= 1) {
            state.board.set_cell(3, 2, true);
        }
        if (occupied_front_corners >= 2) {
            state.board.set_cell(5, 2, true);
        }

        env.restore(state);
        const auto options = env.enumerate_active_piece_placements();
        const auto it = std::find_if(options.begin(), options.end(), [&](const PlacementOption& option) {
            return option.placement == state.active;
        });
        assert(it != options.end());
        assert(it->lines_cleared == 0);
        assert(it->spin_type == expected);

        const auto result = env.apply_placement(it->placement);
        assert(result.action_succeeded);
        assert(result.lines_cleared == 0);
        assert(result.spin_type == expected);
        assert(result.spin_clear);
        assert(!result.difficult_clear);
        assert(!result.b2b_bonus_applied);
        assert(result.combo == -1);
        assert(result.back_to_back);
        assert(result.reward == expected_spin_reward(expected, 0));
        assert(env.state().last_clear_spin_type == expected);
    };

    check(1, 0, SpinType::Mini);
    check(2, 0, SpinType::Full);
    check(1, 4, SpinType::Full);
}

void test_hold_lifecycle_and_rejected_hold_gravity() {
    EnvConfig cfg;
    cfg.seed = 7;
    cfg.gravity_per_step = 1.0f;
    cfg.lock_delay_steps = 999999;

    ModernTetrisEnv env(cfg);
    auto state = make_clean_state(env, Piece::T, {Piece::I, Piece::O});
    state.combo = 2;
    state.back_to_back = true;
    env.restore(state);

    const auto first_hold = env.step(Action::Hold);
    assert(first_hold.action_succeeded);
    assert(first_hold.hold_used);
    assert(!first_hold.game_over);
    assert(env.state().hold == Piece::T);
    assert(env.state().active.piece == Piece::I);
    assert(!env.state().hold_available);
    assert(env.state().combo == 2);
    assert(env.state().back_to_back);

    const int before_rejected_hold_y = env.state().active.y;
    const auto rejected_hold = env.step(Action::Hold);
    assert(!rejected_hold.action_succeeded);
    assert(!rejected_hold.hold_used);
    assert(env.state().active.y == before_rejected_hold_y - 1);
    assert(env.state().combo == 2);
    assert(env.state().back_to_back);

    const auto options = env.enumerate_active_piece_placements();
    assert(!options.empty());
    const auto lock = env.apply_placement(options.front().placement);
    assert(lock.action_succeeded);
    assert(lock.piece_locked);
    assert(env.state().hold_available);
}

void test_hold_topout_is_reported_as_used() {
    EnvConfig cfg;
    cfg.seed = 8;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;

    ModernTetrisEnv env(cfg);
    auto state = make_clean_state(env, Piece::T, {Piece::I, Piece::O});
    state.active = ActivePiece{Piece::T, Rotation::North, 4, 5};
    state.hold = Piece::O;
    state.combo = 2;
    state.back_to_back = true;
    state.board.set_cell(4, 21, true);
    env.restore(state);

    const auto result = env.step(Action::Hold);
    assert(result.action_succeeded);
    assert(result.hold_used);
    assert(result.game_over);
    assert(result.top_out);
    assert(env.state().active.piece == Piece::O);
    assert(env.state().hold == Piece::T);
    assert(!env.state().hold_available);
    assert(env.state().combo == 2);
    assert(env.state().back_to_back);
}

void test_complete_lockout_and_reset() {
    EnvConfig cfg;
    cfg.seed = 9;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;

    ModernTetrisEnv env(cfg);
    auto complete = make_clean_state(env, Piece::O, {Piece::I, Piece::T});
    complete.active = ActivePiece{Piece::O, Rotation::North, 0, Board::kVisibleRows};
    complete.hold_available = false;
    complete.back_to_back = true;
    complete.board.set_cell(0, Board::kVisibleRows - 1, true);
    complete.board.set_cell(1, Board::kVisibleRows - 1, true);
    env.restore(complete);

    const auto queue_before_lockout = env.state().queue;
    const auto lockout = env.step(Action::HardDrop);
    assert(lockout.action_succeeded);
    assert(lockout.piece_locked);
    assert(lockout.game_over);
    assert(lockout.top_out);
    assert(lockout.combo == -1);
    assert(lockout.back_to_back);
    assert(env.state().queue == queue_before_lockout);
    assert(env.state().hold_available);

    env.reset(10);
    assert(!env.state().game_over);
    assert(!env.state().top_out);
    assert(env.state().combo == -1);
    assert(!env.state().back_to_back);
    assert(!env.state().hold.has_value());
    assert(env.state().hold_available);

    auto partial = make_clean_state(env, Piece::O, {Piece::I, Piece::T});
    partial.active = ActivePiece{Piece::O, Rotation::North, 0, Board::kVisibleRows - 1};
    partial.board.set_cell(0, Board::kVisibleRows - 2, true);
    partial.board.set_cell(1, Board::kVisibleRows - 2, true);
    env.restore(partial);

    const auto partial_lock = env.step(Action::HardDrop);
    assert(partial_lock.action_succeeded);
    assert(partial_lock.piece_locked);
    assert(!partial_lock.game_over);
    assert(!partial_lock.top_out);
}

void test_drop_scoring_remains_additive() {
    EnvConfig cfg;
    cfg.seed = 10;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;

    ModernTetrisEnv env(cfg);
    auto state = make_clean_state(env, Piece::O, {Piece::I, Piece::T});
    state.active = ActivePiece{Piece::O, Rotation::North, 4, 3};
    for (int x = 0; x < Board::kWidth; ++x) {
        state.board.set_cell(x, 0, x != 4 && x != 5);
    }
    env.restore(state);

    const auto result = env.step(Action::HardDrop);
    assert(result.action_succeeded);
    assert(result.piece_locked);
    assert(result.lines_cleared == 1);
    assert(result.reward == 106.0f);  // 3 rows * 2 drop points + 100 single-clear points.
}

}  // namespace

int main() {
    test_tetris_to_tetris_gets_b2b_bonus();
    test_tetris_then_single_breaks_b2b();
    test_tetris_zero_then_tetris_preserves_b2b();
    test_t_piece_kick_spin_clear();
    test_enumerated_spin_type_matches_applied_result();
    test_zero_line_spin_scoring_and_b2b_preservation();
    test_hold_lifecycle_and_rejected_hold_gravity();
    test_hold_topout_is_reported_as_used();
    test_complete_lockout_and_reset();
    test_drop_scoring_remains_additive();
    return 0;
}
