#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <random>
#include <unordered_map>
#include <vector>

#include "tetris_v2/cc2_data.hpp"
#include "tetris_v2/cc2_eval.hpp"

namespace tetris_v2::cc2 {

struct DagStatistics {
    std::uint64_t nodes{0};
    std::uint64_t selections{0};
    std::uint64_t expansions{0};

    void accumulate(const DagStatistics& rhs) {
        nodes += rhs.nodes;
        selections += rhs.selections;
        expansions += rhs.expansions;
    }
};

struct DagSuggestion {
    bool valid{false};
    Placement placement{};
    bool use_hold{false};
    float score{0.0f};
};

class Dag {
public:
    Dag();

    void reset(GameState root, const std::deque<Piece>& queue, bool speculate);
    void advance(Piece next_piece, const Placement& placement);
    void add_piece(Piece piece);
    DagSuggestion suggest() const;

    DagStatistics do_work(const FreestyleWeights& weights, double exploitation);

private:
    struct NodeKey {
        GameState state{};
        std::uint16_t depth{0};

        bool operator==(const NodeKey& rhs) const {
            return state.board.cols == rhs.state.board.cols &&
                state.bag_mask == rhs.state.bag_mask &&
                state.reserve == rhs.state.reserve &&
                state.back_to_back == rhs.state.back_to_back &&
                state.combo == rhs.state.combo &&
                depth == rhs.depth;
        }
    };

    struct NodeKeyHash {
        std::size_t operator()(const NodeKey& key) const noexcept;
    };

    struct Edge {
        Placement placement{};
        bool use_hold{false};
        std::uint32_t softdrop{0};
        float immediate{0.0f};
        float cached{0.0f};
        NodeKey child{};
    };

    struct Node {
        bool expanded{false};
        float value{-1000.0f};
        std::array<std::vector<Edge>, 7> children{};
    };

    struct PathEntry {
        NodeKey node{};
        Piece chosen_piece{Piece::None};
        std::size_t edge_index{0};
    };

    static std::size_t piece_slot(Piece piece);
    static std::vector<Piece> pieces_from_mask(PieceMask mask);

    std::vector<Piece> next_possibilities(const NodeKey& key) const;
    Piece choose_speculated_piece(const std::vector<Piece>& pieces);
    std::size_t choose_edge_index(const std::vector<Edge>& edges, double exploitation);

    void sort_edges(std::vector<Edge>* edges);
    float recompute_node_value(const NodeKey& key, Node* node);
    Node& ensure_node(const NodeKey& key);
    const Node* find_node(const NodeKey& key) const;

    GameState root_{};
    std::deque<Piece> queue_{};
    bool speculate_{true};
    mutable std::mt19937 rng_{12345};
    std::unordered_map<NodeKey, Node, NodeKeyHash> nodes_{};
};

}  // namespace tetris_v2::cc2

