#pragma once

#include <vector>
#include <optional>

#include "cold_clear_cpp/bot.hpp"
#include "tetris_v2/tetris_env.hpp"

namespace tetris_v2 {

// Thin wrapper around cold_clear_cpp::Bot to keep it in sync with the environment.
class CCAgent {
public:
    CCAgent(const cold_clear_cpp::BotOptions& options, const TetrisEnv& env);

    // Run search and return a suggested move. Returns std::nullopt if no move is available.
    std::optional<cold_clear_cpp::Placement> choose_move(std::size_t iterations = 64);

    // Inform the agent of new pieces appended to the queue (same order as env).
    void on_new_pieces(const std::vector<cold_clear_cpp::Piece>& pieces);

    // Inform the agent that a move has been applied.
    void on_advance(const cold_clear_cpp::Placement& mv);

    const cold_clear_cpp::Statistics& last_stats() const { return last_stats_; }

private:
    cold_clear_cpp::Bot bot_;
    cold_clear_cpp::BotOptions options_;
    cold_clear_cpp::Statistics last_stats_{};
};

}  // namespace tetris_v2
