#include <cassert>
#include <memory>

#include "tetris_v2/cc_agent.hpp"

int main() {
    tetris_v2::TetrisEnv env(7);
    env.ensure_queue(8);

    cold_clear_cpp::BotOptions options;
    options.speculate = true;
    options.config = std::make_shared<cold_clear_cpp::BotConfig>();

    tetris_v2::CCAgent agent(options, env);

    auto suggested = agent.choose_move(64);
    assert(suggested.has_value());
    env.apply(*suggested);
    agent.on_advance(*suggested);

    return 0;
}
