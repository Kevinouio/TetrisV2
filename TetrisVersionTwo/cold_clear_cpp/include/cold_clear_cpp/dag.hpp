#pragma once

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "cold_clear_cpp/eval.hpp"
#include "cold_clear_cpp/movegen.hpp"
#include "cold_clear_cpp/state_map.hpp"

namespace cold_clear_cpp {

struct ChildData {
    GameState resulting_state;
    Placement mv;
    Eval eval;
    Eval::Reward reward;
};

using ChildList = std::vector<ChildData>;
using ChildrenByPiece = std::array<ChildList, 7>;

class Dag {
public:
    Dag(const GameState& root, const std::vector<Piece>& queue);
    ~Dag();
    Dag(Dag&&) noexcept;
    Dag& operator=(Dag&&) noexcept;
    Dag(const Dag&) = delete;
    Dag& operator=(const Dag&) = delete;

    void advance(const Placement& mv);
    void add_piece(Piece piece);
    std::vector<Placement> suggest() const;

    struct Selection {
        GameState state;
        std::optional<Piece> next_piece;
        std::function<void(const ChildrenByPiece&)> expand;
    };

    std::optional<Selection> select(bool speculate, double exploration) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cold_clear_cpp
