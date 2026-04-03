#include <cassert>

#include "tetris_v2/env.hpp"
#include "tetris_v2/observation.hpp"

int main() {
    tetris_v2::EnvConfig config;
    config.seed = 42;
    tetris_v2::ModernTetrisEnv env(config);

    auto obs = tetris_v2::encode_observation(env.state(), config.queue_size, true);
    assert(!obs.empty());

    auto options = env.enumerate_active_piece_placements();
    assert(!options.empty());

    auto step = env.apply_placement(options.front().placement);
    assert(step.piece_locked);
    return 0;
}
