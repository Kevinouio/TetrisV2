#include "cold_clear_cpp/bot.hpp"

#include <array>

#include "cold_clear_cpp/profiling.hpp"

namespace cold_clear_cpp {

Bot::Bot(const BotOptions& options, GameState root, const std::vector<Piece>& queue)
    : options_(options), current_(root), queue_(queue.begin(), queue.end()), dag_(root, queue) {}

void Bot::advance(const Placement& mv) {
    PROFILE_FUNCTION();
    if (!queue_.empty()) {
        current_.advance(queue_.front(), mv);
        queue_.pop_front();
    }
    dag_.advance(mv);
}

void Bot::new_piece(Piece piece) {
    PROFILE_FUNCTION();
    queue_.push_back(piece);
    dag_.add_piece(piece);
}

std::vector<Placement> Bot::suggest() const {
    PROFILE_FUNCTION();
    return dag_.suggest();
}

Statistics Bot::do_work() const {
    PROFILE_FUNCTION();
    Statistics stats{};
    stats.selections += 1;

    auto selected = dag_.select(options_.speculate, options_.config->freestyle_exploitation);
    if (!selected) {
        return stats;
    }

    auto state = selected->state;
    PieceSet next_possibilities;
    if (selected->next_piece) {
        next_possibilities.reset();
        next_possibilities.set(piece_index(*selected->next_piece));
    } else {
        next_possibilities = state.bag;
    }

    std::array<std::vector<std::pair<Placement, std::uint32_t>>, 7> moves;
    {
        PROFILE_SCOPE("movegen");
        for (auto p : kAllPieces) {
            if (next_possibilities.test(piece_index(p)) || p == state.reserve) {
                moves[piece_index(p)] = find_moves(state.board, p);
            }
        }
    }

    ChildrenByPiece children{};
    PROFILE_SCOPE("eval");
    for (auto next : kAllPieces) {
        if (!next_possibilities.test(piece_index(next))) {
            continue;
        }
        const auto& move_list = moves[piece_index(next)];
        const auto& hold_moves =
            (next == state.reserve) ? std::vector<std::pair<Placement, std::uint32_t>>{}
                                    : moves[piece_index(state.reserve)];

        for (const auto& entry : move_list) {
            auto [mv, sd_distance] = entry;
            auto simulated = state;
            auto info = simulated.advance(next, mv);
            auto eval_pair =
                evaluate(options_.config->freestyle_weights, simulated, info, sd_distance);
            children[piece_index(next)].push_back(
                ChildData{simulated, mv, eval_pair.first, eval_pair.second});
        }

        if (next != state.reserve) {
            for (const auto& entry : hold_moves) {
                auto [mv, sd_distance] = entry;
                auto simulated = state;
                auto info = simulated.advance(next, mv);
                auto eval_pair =
                    evaluate(options_.config->freestyle_weights, simulated, info, sd_distance);
                children[piece_index(next)].push_back(
                    ChildData{simulated, mv, eval_pair.first, eval_pair.second});
            }
        }

        stats.nodes += static_cast<std::uint64_t>(children[piece_index(next)].size());
    }

    stats.expansions += 1;
    selected->expand(children);
    return stats;
}

}  // namespace cold_clear_cpp
