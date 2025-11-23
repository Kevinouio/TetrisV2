#include <cassert>

#include "tetris_v2/tetris_env.hpp"

int main() {
    tetris_v2::TetrisEnv env(42);
    auto added = env.ensure_queue(5);
    assert(env.queue().size() >= 5);
    (void)added;

    auto moves = cold_clear_cpp::find_moves(env.state().board, env.queue().front());
    assert(!moves.empty());
    auto info = env.apply(moves.front().first);
    assert(info.lines_cleared <= 4);
    return 0;
}
