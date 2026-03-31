#include <array>
#include <cassert>
#include <cstddef>

#include "tetris_v2/env.hpp"
#include "tetris_v2/piece_defs.hpp"

namespace {

using namespace tetris_v2;

constexpr Cell kCell(int x, int y) { return Cell{x, y}; }

constexpr std::array<std::array<Cell, 5>, 4> kJLSTZCw{{
    std::array<Cell, 5>{kCell(0, 0), kCell(-1, 0), kCell(-1, 1), kCell(0, -2), kCell(-1, -2)},
    std::array<Cell, 5>{kCell(0, 0), kCell(1, 0), kCell(1, -1), kCell(0, 2), kCell(1, 2)},
    std::array<Cell, 5>{kCell(0, 0), kCell(1, 0), kCell(1, 1), kCell(0, -2), kCell(1, -2)},
    std::array<Cell, 5>{kCell(0, 0), kCell(-1, 0), kCell(-1, -1), kCell(0, 2), kCell(-1, 2)},
}};

constexpr std::array<std::array<Cell, 5>, 4> kJLSTZCcw{{
    std::array<Cell, 5>{kCell(0, 0), kCell(1, 0), kCell(1, 1), kCell(0, -2), kCell(1, -2)},
    std::array<Cell, 5>{kCell(0, 0), kCell(1, 0), kCell(1, -1), kCell(0, 2), kCell(1, 2)},
    std::array<Cell, 5>{kCell(0, 0), kCell(-1, 0), kCell(-1, 1), kCell(0, -2), kCell(-1, -2)},
    std::array<Cell, 5>{kCell(0, 0), kCell(-1, 0), kCell(-1, -1), kCell(0, 2), kCell(-1, 2)},
}};

constexpr std::array<std::array<Cell, 5>, 4> kICw{{
    std::array<Cell, 5>{kCell(1, 0), kCell(-1, 0), kCell(2, 0), kCell(-1, -1), kCell(2, 2)},
    std::array<Cell, 5>{kCell(0, -1), kCell(-1, -1), kCell(2, -1), kCell(-1, 1), kCell(2, -2)},
    std::array<Cell, 5>{kCell(-1, 0), kCell(1, 0), kCell(-2, 0), kCell(1, 1), kCell(-2, -2)},
    std::array<Cell, 5>{kCell(0, 1), kCell(1, 1), kCell(-2, 1), kCell(1, -1), kCell(-2, 2)},
}};

constexpr std::array<std::array<Cell, 5>, 4> kICcw{{
    std::array<Cell, 5>{kCell(0, -1), kCell(-1, -1), kCell(2, -1), kCell(-1, 1), kCell(2, -2)},
    std::array<Cell, 5>{kCell(-1, 0), kCell(1, 0), kCell(-2, 0), kCell(1, 1), kCell(-2, -2)},
    std::array<Cell, 5>{kCell(0, 1), kCell(1, 1), kCell(-2, 1), kCell(1, -1), kCell(-2, 2)},
    std::array<Cell, 5>{kCell(1, 0), kCell(-1, 0), kCell(2, 0), kCell(-1, -1), kCell(2, 2)},
}};

constexpr std::array<std::array<Cell, 5>, 4> kOCw{{
    std::array<Cell, 5>{kCell(0, 1), kCell(0, 1), kCell(0, 1), kCell(0, 1), kCell(0, 1)},
    std::array<Cell, 5>{kCell(1, 0), kCell(1, 0), kCell(1, 0), kCell(1, 0), kCell(1, 0)},
    std::array<Cell, 5>{kCell(0, -1), kCell(0, -1), kCell(0, -1), kCell(0, -1), kCell(0, -1)},
    std::array<Cell, 5>{kCell(-1, 0), kCell(-1, 0), kCell(-1, 0), kCell(-1, 0), kCell(-1, 0)},
}};

constexpr std::array<std::array<Cell, 5>, 4> kOCcw{{
    std::array<Cell, 5>{kCell(1, 0), kCell(1, 0), kCell(1, 0), kCell(1, 0), kCell(1, 0)},
    std::array<Cell, 5>{kCell(0, -1), kCell(0, -1), kCell(0, -1), kCell(0, -1), kCell(0, -1)},
    std::array<Cell, 5>{kCell(-1, 0), kCell(-1, 0), kCell(-1, 0), kCell(-1, 0), kCell(-1, 0)},
    std::array<Cell, 5>{kCell(0, 1), kCell(0, 1), kCell(0, 1), kCell(0, 1), kCell(0, 1)},
}};

bool same_cell(const Cell& lhs, const Cell& rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

EnvState make_fully_blocked_state(ModernTetrisEnv& env, Piece piece, Rotation from) {
    EnvState state = env.snapshot();
    state.board.clear();
    for (int y = 0; y < Board::kRows; ++y) {
        for (int x = 0; x < Board::kWidth; ++x) {
            state.board.set_cell(x, y, true);
        }
    }
    for (auto& row : state.piece_ids) {
        row.fill(-1);
    }
    state.active = ActivePiece{piece, from, 4, 20};
    state.hold.reset();
    state.hold_available = true;
    state.queue.clear();
    state.queue.push_back(Piece::I);
    state.queue.push_back(Piece::T);
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

void assert_trace_for_transition(
    ModernTetrisEnv& env,
    Piece piece,
    Rotation from,
    Action action,
    const std::array<Cell, 5>& expected,
    bool expect_no_alternative_kick_path) {
    env.restore(make_fully_blocked_state(env, piece, from));
    auto trace = env.rotation_trace(action);

    assert(trace.from_rotation == from);
    Rotation expected_target = (action == Action::RotateCW) ? rotate_cw(from) : rotate_ccw(from);
    assert(trace.target_rotation == expected_target);
    assert(!trace.success);
    assert(!trace.final_pose.has_value());
    assert(trace.tests.size() == expected.size());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto& test = trace.tests[i];
        assert(test.test_index == static_cast<int>(i));
        assert(test.phase == 0);
        assert(test.kick_index == static_cast<int>(i));
        assert(same_cell(test.offset, expected[i]));
        assert(test.collides);
        assert(!test.passed);
    }

    if (expect_no_alternative_kick_path) {
        for (std::size_t i = 1; i < trace.tests.size(); ++i) {
            assert(same_cell(trace.tests[i].offset, trace.tests[0].offset));
        }
    }
}

void assert_piece_tables(
    ModernTetrisEnv& env,
    Piece piece,
    const std::array<std::array<Cell, 5>, 4>& expected_cw,
    const std::array<std::array<Cell, 5>, 4>& expected_ccw,
    bool expect_no_alternative_kick_path) {
    for (int idx = 0; idx < 4; ++idx) {
        auto from = static_cast<Rotation>(idx);
        assert_trace_for_transition(
            env,
            piece,
            from,
            Action::RotateCW,
            expected_cw[static_cast<std::size_t>(idx)],
            expect_no_alternative_kick_path);
        assert_trace_for_transition(
            env,
            piece,
            from,
            Action::RotateCCW,
            expected_ccw[static_cast<std::size_t>(idx)],
            expect_no_alternative_kick_path);
    }
}

void test_kick_tables_for_jlstz_i_and_o() {
    EnvConfig cfg;
    cfg.seed = 1234;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;
    ModernTetrisEnv env(cfg);

    const std::array<Piece, 5> jlstz{
        Piece::J, Piece::L, Piece::S, Piece::T, Piece::Z};
    for (auto piece : jlstz) {
        assert_piece_tables(env, piece, kJLSTZCw, kJLSTZCcw, false);
    }

    assert_piece_tables(env, Piece::I, kICw, kICcw, false);
    assert_piece_tables(env, Piece::O, kOCw, kOCcw, true);
}

}  // namespace

int main() {
    test_kick_tables_for_jlstz_i_and_o();
    return 0;
}

