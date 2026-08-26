#include <cassert>
#include <numeric>

#include "tetris_v2/env.hpp"
#include "tetris_v2/observation.hpp"

int main() {
    tetris_v2::EnvConfig config;
    config.seed = 42;
    tetris_v2::ModernTetrisEnv env(config);

    auto obs = tetris_v2::encode_observation(env.state(), config.queue_size, true);
    assert(!obs.empty());

    const auto layout = tetris_v2::build_observation_layout(config.queue_size, false);
    auto visible_obs = tetris_v2::encode_observation(env.state(), config.queue_size, false);
    const auto active_begin = visible_obs.begin() + static_cast<std::ptrdiff_t>(layout.active_piece_offset);
    const auto active_end = active_begin + 8;
    assert(std::accumulate(active_begin, active_end, 0.0f) == 1.0f);
    assert(visible_obs[layout.active_piece_offset + tetris_v2::piece_index(env.state().active.piece)] == 1.0f);

    auto other_state = env.state();
    other_state.active.piece = other_state.active.piece == tetris_v2::Piece::I
                                   ? tetris_v2::Piece::O
                                   : tetris_v2::Piece::I;
    auto other_obs = tetris_v2::encode_observation(other_state, config.queue_size, false);
    assert(visible_obs != other_obs);

    auto options = env.enumerate_active_piece_placements();
    assert(!options.empty());

    auto step = env.apply_placement(options.front().placement);
    assert(step.piece_locked);
    return 0;
}
