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

    tetris_v2::EnvConfig timing_config;
    timing_config.seed = 91;
    timing_config.spawn_y = 19;
    timing_config.gravity_per_step = 1.0 / 60.0;
    timing_config.lock_delay_steps = 999;
    tetris_v2::ModernTetrisEnv timing_env(timing_config);
    const int spawn_y = timing_env.state().active.y;
    for (int tick = 0; tick < 59; ++tick) {
        timing_env.tick();
        assert(timing_env.state().active.y == spawn_y);
    }
    timing_env.tick();
    assert(timing_env.state().active.y == spawn_y - 1);

    auto airborne = timing_env.snapshot();
    airborne.state.active.y = 10;
    airborne.state.lock_timer = 12;
    airborne.state.lock_resets_used = timing_config.max_lock_resets;
    airborne.state.gravity_accumulator = 0.0;
    timing_env.restore(airborne);
    timing_env.input(tetris_v2::Action::Left);
    timing_env.tick();
    assert(timing_env.state().lock_timer == 12);
    assert(timing_env.state().lock_resets_used == timing_config.max_lock_resets);

    airborne.state.lock_resets_used = timing_config.max_lock_resets - 1;
    timing_env.restore(airborne);
    timing_env.tick();
    assert(timing_env.state().lock_timer == 0);
    assert(timing_env.state().lock_resets_used == timing_config.max_lock_resets - 1);

    tetris_v2::EnvConfig ledge_config;
    ledge_config.seed = 92;
    ledge_config.gravity_per_step = 1.0;
    ledge_config.lock_delay_steps = 13;
    ledge_config.max_lock_resets = 2;
    tetris_v2::ModernTetrisEnv ledge_env(ledge_config);
    auto ledge = ledge_env.snapshot();
    ledge.state.board.clear();
    ledge.state.board.set_cell(4, 0);
    ledge.state.active =
        tetris_v2::ActivePiece{tetris_v2::Piece::O, tetris_v2::Rotation::North, 4, 1};
    ledge.state.lock_timer = 12;
    ledge.state.lock_resets_used = ledge_config.max_lock_resets;
    ledge.state.gravity_accumulator = 0.0;
    ledge_env.restore(ledge);

    const auto left_ledge = ledge_env.input(tetris_v2::Action::Right);
    assert(left_ledge.action_succeeded);
    assert(!left_ledge.piece_locked);
    assert(ledge_env.state().active.x == 5);
    assert(ledge_env.state().lock_timer == 12);
    assert(ledge_env.state().lock_resets_used == ledge_config.max_lock_resets);

    const auto recontact = ledge_env.tick();
    assert(recontact.piece_locked);
    return 0;
}
