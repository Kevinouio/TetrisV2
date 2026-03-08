#include <array>
#include <cassert>
#include <vector>

#include "tetris_v2/env.hpp"
#include "tetris_v2/piece_defs.hpp"

namespace {

using namespace tetris_v2;

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
        return (after.x != raw.x) || (after.y != raw.y);
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

void test_kicks_found_for_all_pieces(Action action) {
    for (auto piece : kPlayablePieces) {
        bool found = kicked_rotation_found_for_piece(piece, action);
        assert(found && "No kick-required state found for piece");
    }
}

}  // namespace

int main() {
    test_kicks_found_for_all_pieces(Action::RotateCW);
    test_kicks_found_for_all_pieces(Action::RotateCCW);
    return 0;
}
