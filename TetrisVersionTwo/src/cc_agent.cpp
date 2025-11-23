#include "tetris_v2/cc_agent.hpp"

#include <vector>

using namespace cold_clear_cpp;

namespace tetris_v2 {

CCAgent::CCAgent(const BotOptions& options, const TetrisEnv& env)
    : bot_(options, env.state(), env.queue()), options_(options) {}

std::optional<Placement> CCAgent::choose_move(std::size_t iterations) {
    last_stats_ = {};
    for (std::size_t i = 0; i < iterations; ++i) {
        auto stats = bot_.do_work();
        last_stats_.accumulate(stats);
    }
    auto suggestions = bot_.suggest();
    if (suggestions.empty()) {
        return std::nullopt;
    }
    return suggestions.front();
}

void CCAgent::on_new_pieces(const std::vector<Piece>& pieces) {
    for (auto p : pieces) {
        bot_.new_piece(p);
    }
}

void CCAgent::on_advance(const Placement& mv) { bot_.advance(mv); }

}  // namespace tetris_v2
