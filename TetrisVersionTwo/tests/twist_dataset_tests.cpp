#include <array>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "generated/twist_cases_generated.hpp"
#include "tetris_v2/env.hpp"
#include "tetris_v2/piece_defs.hpp"

namespace {

using namespace tetris_v2;
using namespace tetris_v2::twist_dataset;

constexpr const char* kColorGreen = "\x1b[32m";
constexpr const char* kColorRed = "\x1b[31m";
constexpr const char* kColorReset = "\x1b[0m";

bool twist_verbose() {
    const char* raw = std::getenv("TETRIS_TWIST_VERBOSE");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
}

Piece parse_piece(char piece) {
    switch (piece) {
        case 'I': return Piece::I;
        case 'O': return Piece::O;
        case 'T': return Piece::T;
        case 'L': return Piece::L;
        case 'J': return Piece::J;
        case 'S': return Piece::S;
        case 'Z': return Piece::Z;
        default: break;
    }
    return Piece::None;
}

Rotation parse_rotation(int rotation) {
    switch (rotation) {
        case 0: return Rotation::North;
        case 1: return Rotation::East;
        case 2: return Rotation::South;
        case 3: return Rotation::West;
        default: break;
    }
    return Rotation::North;
}

Action parse_action(std::string_view token) {
    if (token == "L") return Action::Left;
    if (token == "R") return Action::Right;
    if (token == "D") return Action::SoftDrop;
    if (token == "HD") return Action::HardDrop;
    if (token == "CW") return Action::RotateCW;
    if (token == "CCW") return Action::RotateCCW;
    if (token == "R180") return Action::Rotate180;
    if (token == "NONE") return Action::None;
    if (token == "HOLD") return Action::Hold;
    return Action::None;
}

bool is_rotate_action(Action action) {
    return action == Action::RotateCW || action == Action::RotateCCW || action == Action::Rotate180;
}

bool rotation_used_required_kick(const RotationTrace& trace) {
    if (!trace.success) {
        return false;
    }
    for (const auto& test : trace.tests) {
        if (test.passed) {
            return test.kick_index > 0;
        }
    }
    return false;
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

std::string render_fixture_board(const TwistCase& tc) {
    std::array<std::string, Board::kVisibleRows> rows{};
    for (int i = 0; i < Board::kVisibleRows; ++i) {
        rows[static_cast<std::size_t>(i)] =
            std::string(tc.board_top_to_bottom[static_cast<std::size_t>(i)]);
    }
    auto p = parse_piece(tc.active.piece);
    auto r = parse_rotation(tc.active.rotation);
    auto cells = piece_cells(p, r);
    for (const auto& c : cells) {
        int x = tc.active.x + c.x;
        int y = tc.active.y + c.y;
        if (x < 0 || x >= Board::kWidth || y < 0 || y >= Board::kVisibleRows) {
            continue;
        }
        int row = (Board::kVisibleRows - 1) - y;
        rows[static_cast<std::size_t>(row)][static_cast<std::size_t>(x)] = tc.active.piece;
    }
    std::ostringstream out;
    for (const auto& row : rows) {
        out << row << '\n';
    }
    return out.str();
}

void dump_case_failure(
    const TwistCase& tc,
    const ModernTetrisEnv& env,
    const std::optional<StepResult>& lock_result,
    bool observed_required_kick,
    const std::vector<std::string>& reasons) {
    std::cerr << "\n[twist_dataset] Case failed: " << tc.id << " family=" << tc.family << '\n';
    if (reasons.empty()) {
        std::cerr << "reason: <none>\n";
    } else {
        std::cerr << "reasons:\n";
        for (const auto& reason : reasons) {
            std::cerr << "  - " << reason << '\n';
        }
    }
    std::cerr << "source: " << tc.source_url << '\n';
    std::cerr << "note: " << tc.source_note << '\n';
    std::cerr << "actions:";
    for (const auto& action : tc.actions) {
        std::cerr << ' ' << action;
    }
    std::cerr << '\n';
    std::cerr << "fixture board + active:\n" << render_fixture_board(tc);
    std::cerr << "env board:\n" << env.render_ascii();
    if (lock_result.has_value()) {
        std::cerr << "lock result: lines=" << lock_result->lines_cleared
                  << " spin=" << lock_result->spin_clear
                  << " difficult=" << lock_result->difficult_clear
                  << " b2b_bonus=" << lock_result->b2b_bonus_applied
                  << " b2b_out=" << env.state().back_to_back
                  << " observed_kick=" << observed_required_kick << '\n';
    } else {
        std::cerr << "lock result: <none>\n";
    }
}

void setup_state_from_case(ModernTetrisEnv& env, const TwistCase& tc) {
    EnvState state = env.snapshot();
    state.board.clear();
    for (auto& row : state.piece_ids) {
        row.fill(-1);
    }

    for (int row = 0; row < Board::kVisibleRows; ++row) {
        int y = (Board::kVisibleRows - 1) - row;
        auto row_text = tc.board_top_to_bottom[static_cast<std::size_t>(row)];
        for (int x = 0; x < Board::kWidth; ++x) {
            if (row_text[static_cast<std::size_t>(x)] == '#') {
                state.board.set_cell(x, y, true);
            }
        }
    }

    state.active = ActivePiece{
        parse_piece(tc.active.piece),
        parse_rotation(tc.active.rotation),
        tc.active.x,
        tc.active.y,
    };
    state.hold.reset();
    state.hold_available = true;
    state.queue.clear();
    state.queue.push_back(Piece::I);
    state.queue.push_back(Piece::T);
    state.queue.push_back(Piece::O);
    state.game_over = false;
    state.top_out = false;
    state.combo = tc.state.combo;
    state.back_to_back = tc.state.back_to_back;
    state.total_lines_cleared = 0;
    state.lock_timer = 0;
    state.lock_resets_used = 0;
    state.gravity_accumulator = 0.0f;
    state.spin_eligible = tc.state.spin_eligible;
    state.last_rotate_used_kick = tc.state.last_rotate_used_kick;
    state.last_clear_spin = false;
    state.last_clear_difficult = false;
    state.last_clear_b2b_bonus = false;
    env.restore(state);
}

bool run_case(const TwistCase& tc) {
    EnvConfig cfg;
    cfg.seed = 1337;
    cfg.gravity_per_step = 0.0f;
    cfg.lock_delay_steps = 999999;
    ModernTetrisEnv env(cfg);
    setup_state_from_case(env, tc);

    std::optional<StepResult> lock_result;
    bool observed_required_kick = false;
    std::vector<std::string> failures;
    auto check = [&failures](bool condition, std::string_view reason) {
        if (!condition) {
            failures.emplace_back(reason);
        }
    };

    check(env.state().active.piece != Piece::None, "invalid active piece in fixture");
    check(!collides_local(env.state().board, env.state().active), "fixture active piece collides before actions");

    if (!failures.empty()) {
        dump_case_failure(tc, env, lock_result, observed_required_kick, failures);
        return false;
    }

    for (const auto& token : tc.actions) {
        Action action = parse_action(token);
        if (is_rotate_action(action)) {
            auto trace = env.rotation_trace(action);
            observed_required_kick = observed_required_kick || rotation_used_required_kick(trace);
        }
        auto result = env.step(action);
        if (result.piece_locked) {
            lock_result = result;
        }
    }

    check(lock_result.has_value(), "action script did not lock a piece");
    if (lock_result.has_value()) {
        check(lock_result->lines_cleared == tc.expect.lines_cleared, "lines_cleared mismatch");
        check(lock_result->spin_clear == tc.expect.spin_clear, "spin_clear mismatch");
        check(lock_result->difficult_clear == tc.expect.difficult_clear, "difficult_clear mismatch");
        check(lock_result->b2b_bonus_applied == tc.expect.b2b_bonus_applied, "b2b_bonus_applied mismatch");
        check(env.state().back_to_back == tc.expect.b2b_out, "back_to_back output mismatch");
        check(observed_required_kick == tc.expect.must_observe_kick, "must_observe_kick mismatch");
    }

    if (!failures.empty()) {
        dump_case_failure(tc, env, lock_result, observed_required_kick, failures);
        return false;
    }

    std::cout << "fixture board + active:\n" << render_fixture_board(tc);
    std::cout << "env board:\n" << env.render_ascii();
    if (lock_result.has_value()) {
        std::cout << "lock result: lines=" << lock_result->lines_cleared
                  << " spin=" << lock_result->spin_clear
                  << " difficult=" << lock_result->difficult_clear
                  << " b2b_bonus=" << lock_result->b2b_bonus_applied
                  << " b2b_out=" << env.state().back_to_back
                  << " observed_kick=" << observed_required_kick << '\n';
    } else {
        std::cout << "lock result: <none>\n";
    }

    if (twist_verbose()) {
        std::cout << "[twist_dataset] PASS " << tc.id << " lines=" << lock_result->lines_cleared
                  << " spin=" << lock_result->spin_clear
                  << " difficult=" << lock_result->difficult_clear
                  << " kick=" << observed_required_kick << '\n';
    }
    return true;
}

std::vector<std::string> validate_dataset_shape() {
    std::vector<std::string> failures;
    std::size_t t = 0;
    std::size_t i = 0;
    std::size_t j = 0;
    std::size_t l = 0;
    std::size_t s = 0;
    std::size_t z = 0;
    for (const auto& tc : kTwistCases) {
        if (tc.family == "T") ++t;
        if (tc.family == "I") ++i;
        if (tc.family == "J") ++j;
        if (tc.family == "L") ++l;
        if (tc.family == "S") ++s;
        if (tc.family == "Z") ++z;
    }
    if (kTwistCases.size() != 24) {
        failures.emplace_back("dataset size mismatch (expected 24)");
    }
    if (t != 6) failures.emplace_back("T family count mismatch (expected 6)");
    if (i != 6) failures.emplace_back("I family count mismatch (expected 6)");
    if (j != 3) failures.emplace_back("J family count mismatch (expected 3)");
    if (l != 3) failures.emplace_back("L family count mismatch (expected 3)");
    if (s != 3) failures.emplace_back("S family count mismatch (expected 3)");
    if (z != 3) failures.emplace_back("Z family count mismatch (expected 3)");
    return failures;
}

}  // namespace

int main() {
    auto shape_failures = validate_dataset_shape();
    if (!shape_failures.empty()) {
        std::cerr << "[twist_dataset] Invalid dataset shape:\n";
        for (const auto& failure : shape_failures) {
            std::cerr << "  - " << failure << '\n';
        }
        return 1;
    }

    std::size_t failed = 0;
    for (const auto& tc : kTwistCases) {
        bool passed = run_case(tc);
        if (passed) {
            std::cout << kColorGreen
                      << "[twist_dataset] PASS"
                      << kColorReset
                      << " " << tc.id << " family=" << tc.family << '\n';
        } else {
            ++failed;
            std::cout << kColorRed
                      << "[twist_dataset] FAIL"
                      << kColorReset
                      << " " << tc.id << " family=" << tc.family << '\n';
        }
    }

    if (failed > 0) {
        std::cerr << "\n[twist_dataset] Failed cases: " << failed << " / " << kTwistCases.size() << '\n';
        return 1;
    }
    return 0;
}
