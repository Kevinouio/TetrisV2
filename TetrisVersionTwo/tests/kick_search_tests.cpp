#include <array>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "tetris_v2/env.hpp"
#include "tetris_v2/piece_defs.hpp"

namespace {

using namespace tetris_v2;

constexpr const char* kColorGreen = "\x1b[32m";
constexpr const char* kColorRed = "\x1b[31m";
constexpr const char* kColorReset = "\x1b[0m";

const char* action_name(Action action) {
    switch (action) {
        case Action::RotateCW: return "CW";
        case Action::RotateCCW: return "CCW";
        default: return "?";
    }
}

void print_board_with_piece(const Board& board, const ActivePiece& piece, const std::string& label) {
    std::array<std::string, Board::kVisibleRows> rows{};
    for (int row = 0; row < Board::kVisibleRows; ++row) {
        int y = (Board::kVisibleRows - 1) - row;
        std::string line;
        line.reserve(Board::kWidth);
        auto mask = board.row_mask(y);
        for (int x = 0; x < Board::kWidth; ++x) {
            line.push_back((mask & (1u << x)) ? '#' : '.');
        }
        rows[static_cast<std::size_t>(row)] = std::move(line);
    }

    auto cells = piece_cells(piece.piece, piece.rotation);
    for (const auto& c : cells) {
        int x = piece.x + c.x;
        int y = piece.y + c.y;
        if (x < 0 || x >= Board::kWidth || y < 0 || y >= Board::kVisibleRows) {
            continue;
        }
        int row = (Board::kVisibleRows - 1) - y;
        rows[static_cast<std::size_t>(row)][static_cast<std::size_t>(x)] =
            piece_to_char(piece.piece);
    }

    std::cout << label << '\n';
    for (const auto& row : rows) {
        std::cout << row << '\n';
    }
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

ActivePiece raw_rotated(const ActivePiece& piece, Action action) {
    ActivePiece out = piece;
    if (action == Action::RotateCW) {
        out.rotation = rotate_cw(out.rotation);
    } else {
        out.rotation = rotate_ccw(out.rotation);
    }
    return out;
}

bool kicked_rotation_found_for_piece(Piece piece, Action action) {
    EnvConfig config;
    config.seed = 7;
    config.gravity_per_step = 0.0f;
    config.lock_delay_steps = 999999;
    config.allow_rotate_180 = true;

    ModernTetrisEnv env(config);
    EnvState state = env.snapshot();
    state.game_over = false;
    state.top_out = false;
    state.hold.reset();
    state.hold_available = true;
    state.queue.clear();
    state.combo = -1;
    state.back_to_back = false;
    state.lock_timer = 0;
    state.lock_resets_used = 0;
    state.gravity_accumulator = 0.0f;

    const auto rotations = std::array<Rotation, 4>{
        Rotation::North, Rotation::East, Rotation::South, Rotation::West};

    auto has_kick_on_state = [&](const Board& board, const ActivePiece& active) {
        if (collides_local(board, active)) {
            return false;
        }
        auto raw = raw_rotated(active, action);
        if (!collides_local(board, raw)) {
            return false;  // not a kick-required state
        }

        EnvState s = state;
        s.board = board;
        s.active = active;
        env.restore(s);

        auto result = env.step(action);
        auto after = env.state().active;

        if (!result.action_succeeded || result.piece_locked || result.game_over) {
            return false;
        }
        if (after.rotation != raw.rotation) {
            return false;
        }
        // Kick occurred iff rotation succeeded despite raw collision and pose changed beyond raw.
        bool kicked = (after.x != raw.x) || (after.y != raw.y);
        if (kicked) {
            std::cout << "\n[kick_search] Found kick-required state "
                      << "piece=" << piece_name(piece)
                      << " action=" << action_name(action)
                      << " from=(" << active.x << "," << active.y << ","
                      << static_cast<int>(active.rotation) << ")"
                      << " raw=(" << raw.x << "," << raw.y << ","
                      << static_cast<int>(raw.rotation) << ")"
                      << " after=(" << after.x << "," << after.y << ","
                      << static_cast<int>(after.rotation) << ")\n";
            print_board_with_piece(board, active, "Board + active before rotate:");
            print_board_with_piece(board, raw, "Board + raw rotated pose (blocked):");
            print_board_with_piece(board, after, "Board + final kicked pose:");
        }
        return kicked;
    };

    for (auto rotation : rotations) {
        for (int x = -2; x <= 11; ++x) {
            for (int y = 0; y <= 26; ++y) {
                ActivePiece active{piece, rotation, x, y};

                Board empty{};
                empty.clear();
                if (has_kick_on_state(empty, active)) {
                    return true;
                }

                // Search additional obstructed states: place one blocker around the piece.
                for (int dx = -3; dx <= 3; ++dx) {
                    for (int dy = -3; dy <= 3; ++dy) {
                        Board board{};
                        board.clear();
                        board.set_cell(x + dx, y + dy, true);
                        if (has_kick_on_state(board, active)) {
                            return true;
                        }
                    }
                }
            }
        }
    }

    return false;
}

bool test_kicks_found_for_all_pieces(Action action) {
    int failures = 0;
    for (auto piece : kPlayablePieces) {
        bool found = kicked_rotation_found_for_piece(piece, action);
        if (found) {
            std::cout << kColorGreen
                      << "[kick_search] FOUND"
                      << kColorReset
                      << " piece=" << piece_name(piece)
                      << " action=" << action_name(action) << '\n';
        } else {
            ++failures;
            std::cout << kColorRed
                      << "[kick_search] NOT FOUND"
                      << kColorReset
                      << " piece=" << piece_name(piece)
                      << " action=" << action_name(action) << '\n';
        }
    }
    if (failures > 0) {
        std::cerr << "[kick_search] Missing kick-required states for action="
                  << action_name(action) << " failures=" << failures << '\n';
        return false;
    }
    return true;
}

}  // namespace

int main() {
    bool ok_cw = test_kicks_found_for_all_pieces(Action::RotateCW);
    bool ok_ccw = test_kicks_found_for_all_pieces(Action::RotateCCW);
    return (ok_cw && ok_ccw) ? 0 : 1;
}
