#include <algorithm>
#include <array>
#include <cassert>
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
};

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
    EnvState state = env.snapshot();
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
    state.last_clear_spin = false;
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
                                   opt.spin_clear_candidate &&
                                   opt.last_rotate_used_kick_path;
                        });
                        if (it != options.end()) {
                            std::cout << "\n[b2b_spin] Kick+spin candidate found piece="
                                      << piece_name(piece)
                                      << " pattern=" << pattern
                                      << " gap=" << gap
                                      << " blocker=(" << dx << "," << dy << ")"
                                      << " lines=" << it->lines_cleared
                                      << " spin=" << it->spin_clear_candidate
                                      << " difficult=" << it->difficult_clear_candidate
                                      << " kick=" << it->last_rotate_used_kick_path
                                      << '\n';
                            print_board_with_piece(state.board, state.active, "Board at discovery:");
                            return KickSpinScenario{state, it->placement};
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
    assert(env.state().back_to_back);

    auto options2 = env.enumerate_active_piece_placements();
    auto* second_tetris = find_placement_with_lines(options2, 4);
    assert(second_tetris != nullptr);
    auto step2 = env.apply_placement(second_tetris->placement);
    assert(step2.action_succeeded);
    assert(step2.lines_cleared == 4);
    assert(step2.difficult_clear);
    assert(step2.b2b_bonus_applied);
    assert(env.state().back_to_back);
}

void test_tetris_then_single_breaks_b2b() {
    EnvConfig cfg;
    cfg.seed = 2;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;

    ModernTetrisEnv env(cfg);
    auto state = make_clean_state(env, Piece::I, {Piece::I, Piece::T, Piece::T});
    for (int y = 0; y <= 3; ++y) {
        fill_row_with_gap(state.board, y, 4);
    }
    fill_row_with_gap(state.board, 4, 4);
    env.restore(state);

    auto options1 = env.enumerate_active_piece_placements();
    auto* first_tetris = find_placement_with_lines(options1, 4);
    assert(first_tetris != nullptr);
    auto step1 = env.apply_placement(first_tetris->placement);
    assert(step1.action_succeeded);
    assert(step1.lines_cleared == 4);
    assert(env.state().back_to_back);

    auto options2 = env.enumerate_active_piece_placements();
    auto* single = find_placement_with_lines(options2, 1);
    assert(single != nullptr);
    auto step2 = env.apply_placement(single->placement);
    assert(step2.action_succeeded);
    assert(step2.lines_cleared == 1);
    assert(!step2.difficult_clear);
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
    assert(env.state().back_to_back);

    auto options2 = env.enumerate_active_piece_placements();
    auto zero_it = std::find_if(options2.begin(), options2.end(), [](const PlacementOption& opt) {
        return opt.lines_cleared == 0 && opt.placement.x <= 1;
    });
    assert(zero_it != options2.end());
    auto step2 = env.apply_placement(zero_it->placement);
    assert(step2.action_succeeded);
    assert(step2.lines_cleared == 0);
    assert(env.state().back_to_back);

    auto options3 = env.enumerate_active_piece_placements();
    auto* second_tetris = find_placement_with_lines(options3, 4);
    assert(second_tetris != nullptr);
    auto step3 = env.apply_placement(second_tetris->placement);
    assert(step3.action_succeeded);
    assert(step3.lines_cleared == 4);
    assert(step3.b2b_bonus_applied);
    assert(env.state().back_to_back);
}

void test_kick_spin_clear_for_all_non_o_pieces() {
    EnvConfig cfg;
    cfg.seed = 4;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;

    ModernTetrisEnv env(cfg);

    const std::array<Piece, 6> pieces{
        Piece::I, Piece::T, Piece::L, Piece::J, Piece::S, Piece::Z};

    for (auto piece : pieces) {
        auto scenario = find_kick_spin_clear_scenario(env, piece);
        assert(scenario.has_value() && "No kick+spin clear scenario found for piece");

        auto state_no_b2b = scenario->state;
        state_no_b2b.back_to_back = false;
        env.restore(state_no_b2b);
        auto clear_no_b2b = env.apply_placement(scenario->placement);
        assert(clear_no_b2b.action_succeeded);
        assert(clear_no_b2b.lines_cleared > 0);
        assert(clear_no_b2b.spin_clear);
        assert(clear_no_b2b.difficult_clear);
        assert(!clear_no_b2b.b2b_bonus_applied);
        assert(env.state().back_to_back);

        auto state_with_b2b = scenario->state;
        state_with_b2b.back_to_back = true;
        env.restore(state_with_b2b);
        auto clear_with_b2b = env.apply_placement(scenario->placement);
        assert(clear_with_b2b.action_succeeded);
        assert(clear_with_b2b.lines_cleared > 0);
        assert(clear_with_b2b.spin_clear);
        assert(clear_with_b2b.difficult_clear);
        assert(clear_with_b2b.b2b_bonus_applied);
    }
}

}  // namespace

int main() {
    test_tetris_to_tetris_gets_b2b_bonus();
    test_tetris_then_single_breaks_b2b();
    test_tetris_zero_then_tetris_preserves_b2b();
    test_kick_spin_clear_for_all_non_o_pieces();
    return 0;
}
