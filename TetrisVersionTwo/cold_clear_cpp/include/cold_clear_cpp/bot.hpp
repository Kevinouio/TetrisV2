#pragma once

#include <deque>
#include <memory>
#include <vector>

#include "cold_clear_cpp/dag.hpp"

namespace cold_clear_cpp {

struct BotConfig {
    Weights freestyle_weights{default_weights()};
    double freestyle_exploitation{0.6931471805599453};
};

struct BotOptions {
    bool speculate{false};
    std::shared_ptr<BotConfig> config{std::make_shared<BotConfig>()};
};

struct Statistics {
    std::uint64_t nodes{0};
    std::uint64_t selections{0};
    std::uint64_t expansions{0};

    void accumulate(const Statistics& other) {
        nodes += other.nodes;
        selections += other.selections;
        expansions += other.expansions;
    }
};

class Bot {
public:
    Bot(const BotOptions& options, GameState root, const std::vector<Piece>& queue);

    void advance(const Placement& mv);
    void new_piece(Piece piece);
    std::vector<Placement> suggest() const;
    Statistics do_work() const;

private:
    BotOptions options_;
    GameState current_;
    std::deque<Piece> queue_;
    Dag dag_;
};

}  // namespace cold_clear_cpp
