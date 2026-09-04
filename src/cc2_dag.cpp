#include "tetris_v2/cc2_dag.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "tetris_v2/cc2_movegen.hpp"

namespace tetris_v2::cc2 {

namespace {

constexpr float kInvalidEval = -1000.0f;

std::size_t hash_combine(std::size_t seed, std::size_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
    return seed;
}

}  // namespace

Dag::Dag() { nodes_.reserve(4096); }

void Dag::reset(GameState root, const std::deque<Piece>& queue, bool speculate) {
    root_ = root;
    queue_ = queue;
    speculate_ = speculate;
    rng_.seed(kRngSeed);
    nodes_.clear();
    ensure_node(NodeKey{root_, 0});
}

DagSuggestion Dag::suggest() const {
    DagSuggestion out{};
    const NodeKey root_key{root_, 0};
    const Node* root = find_node(root_key);
    if (!root) {
        return out;
    }

    const auto possible = next_possibilities(root_key);
    float best = -std::numeric_limits<float>::infinity();
    std::optional<Edge> best_edge{};
    for (const auto piece : possible) {
        const auto& edges = root->children[piece_slot(piece)];
        if (edges.empty()) {
            continue;
        }
        if (edges.front().cached > best) {
            best = edges.front().cached;
            best_edge = edges.front();
        }
    }

    if (!best_edge.has_value() || !std::isfinite(best)) {
        return out;
    }

    out.valid = true;
    out.placement = best_edge->placement;
    out.use_hold = best_edge->use_hold;
    out.score = best;
    return out;
}

DagStatistics Dag::do_work(const FreestyleWeights& weights, double exploitation) {
    DagStatistics out{};
    out.selections = 1;

    std::vector<PathEntry> path;
    path.reserve(64);
    NodeKey key{root_, 0};

    for (int depth_guard = 0; depth_guard < 256; ++depth_guard) {
        Node& node = ensure_node(key);
        const auto possible = next_possibilities(key);
        if (possible.empty()) {
            node.value = kInvalidEval;
            break;
        }

        const Piece chosen =
            (possible.size() == 1) ? possible.front() : choose_speculated_piece(possible);
        auto& edges = node.children[piece_slot(chosen)];
        if (edges.empty()) {
            const auto next_moves = find_moves(key.state.board, chosen);
            std::vector<std::pair<Placement, std::uint32_t>> hold_moves{};
            std::uint16_t hold_depth_advance = 1;
            if (key.state.hold_available) {
                if (key.state.reserve != Piece::None && key.state.reserve != chosen) {
                    hold_moves = find_moves(key.state.board, key.state.reserve);
                } else if (key.state.reserve == Piece::None) {
                    const std::size_t preview_index = static_cast<std::size_t>(key.depth) + 1u;
                    if (preview_index < queue_.size() && queue_[preview_index] != Piece::None) {
                        hold_moves = find_moves(key.state.board, queue_[preview_index]);
                        hold_depth_advance = 2;
                    }
                }
            }
            edges.reserve(next_moves.size() + hold_moves.size());

            auto append_child =
                [&](const std::pair<Placement, std::uint32_t>& move,
                    bool use_hold,
                    std::uint16_t depth_advance) {
                    GameState child_state = key.state;
                    const auto info = child_state.advance(chosen, move.first, use_hold);
                    const auto scored = evaluate_freestyle(weights, child_state, info, move.second);
                    const std::uint16_t next_depth = static_cast<std::uint16_t>(
                        std::min<unsigned int>(65535u, key.depth + depth_advance));
                    const NodeKey child_key{child_state, next_depth};
                    Node& child = ensure_node(child_key);
                    if (!child.expanded) {
                        child.value = scored.eval;
                    }

                    Edge edge{};
                    edge.placement = move.first;
                    edge.use_hold = use_hold;
                    edge.immediate = scored.reward;
                    edge.cached = scored.reward + child.value;
                    edge.child = child_key;
                    edges.push_back(edge);
                    ++out.nodes;
                };

            for (const auto& move : next_moves) {
                append_child(move, false, 1);
            }
            for (const auto& move : hold_moves) {
                append_child(move, true, hold_depth_advance);
            }
            sort_edges(&edges);
            node.expanded = true;
            recompute_node_value(key, &node);
            ++out.expansions;
            break;
        }

        const std::size_t index = choose_edge_index(edges, exploitation);
        path.push_back(PathEntry{key, chosen, index});
        key = edges[index].child;
    }

    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        Node& parent = ensure_node(it->node);
        auto& edges = parent.children[piece_slot(it->chosen_piece)];
        if (it->edge_index >= edges.size()) {
            continue;
        }
        Edge& edge = edges[it->edge_index];
        const Node* child = find_node(edge.child);
        edge.cached = edge.immediate + (child ? child->value : kInvalidEval);
        sort_edges(&edges);
        recompute_node_value(it->node, &parent);
    }

    return out;
}

std::size_t Dag::NodeKeyHash::operator()(const NodeKey& key) const noexcept {
    std::size_t h = 1469598103934665603ull;
    for (auto col : key.state.board.cols) {
        h = hash_combine(h, std::hash<std::uint64_t>{}(col));
    }
    h = hash_combine(h, std::hash<std::uint8_t>{}(key.state.bag_mask));
    h = hash_combine(h, std::hash<int>{}(static_cast<int>(key.state.reserve)));
    h = hash_combine(h, std::hash<bool>{}(key.state.hold_available));
    h = hash_combine(h, std::hash<bool>{}(key.state.back_to_back));
    h = hash_combine(h, std::hash<std::uint8_t>{}(key.state.combo));
    h = hash_combine(h, std::hash<std::uint16_t>{}(key.depth));
    return h;
}

std::size_t Dag::piece_slot(Piece piece) {
    if (piece == Piece::None) {
        return 0;
    }
    return std::min<std::size_t>(6, static_cast<std::size_t>(piece));
}

std::vector<Piece> Dag::pieces_from_mask(PieceMask mask) {
    std::vector<Piece> out;
    out.reserve(7);
    for (auto piece : kPlayablePieces) {
        if (piece_mask_contains(mask, piece)) {
            out.push_back(piece);
        }
    }
    return out;
}

std::vector<Piece> Dag::next_possibilities(const NodeKey& key) const {
    if (key.depth < queue_.size()) {
        const auto next = queue_[key.depth];
        if (next != Piece::None) {
            return {next};
        }
    }
    if (!speculate_) {
        return {};
    }
    return pieces_from_mask(key.state.bag_mask);
}

Piece Dag::choose_speculated_piece(const std::vector<Piece>& pieces) {
    if (pieces.empty()) {
        return Piece::None;
    }
    std::uniform_int_distribution<std::size_t> dist(0, pieces.size() - 1);
    return pieces[dist(rng_)];
}

std::size_t Dag::choose_edge_index(const std::vector<Edge>& edges, double exploitation) {
    if (edges.empty()) {
        return 0;
    }
    if (exploitation <= 0.0) {
        return 0;
    }

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double sample = dist(rng_);
    if (sample < 1e-9) {
        sample = 1e-9;
    }
    const double raw = std::fmod((-std::log(sample) / exploitation), static_cast<double>(edges.size()));
    const std::size_t idx = static_cast<std::size_t>(raw);
    return std::min<std::size_t>(edges.size() - 1, idx);
}

void Dag::sort_edges(std::vector<Edge>* edges) {
    if (!edges) {
        return;
    }
    std::sort(edges->begin(), edges->end(), [](const Edge& a, const Edge& b) {
        return a.cached > b.cached;
    });
}

float Dag::recompute_node_value(const NodeKey& key, Node* node) {
    if (!node) {
        return kInvalidEval;
    }

    const auto possible = next_possibilities(key);
    if (possible.empty()) {
        node->value = kInvalidEval;
        return node->value;
    }

    float sum = 0.0f;
    std::size_t count = 0;
    for (const auto piece : possible) {
        const auto& edges = node->children[piece_slot(piece)];
        if (edges.empty()) {
            sum += kInvalidEval;
        } else {
            sum += edges.front().cached;
        }
        ++count;
    }
    if (count == 0) {
        node->value = kInvalidEval;
    } else {
        node->value = sum / static_cast<float>(count);
    }
    return node->value;
}

Dag::Node& Dag::ensure_node(const NodeKey& key) {
    auto [it, inserted] = nodes_.emplace(key, Node{});
    if (inserted) {
        it->second.value = kInvalidEval;
        it->second.expanded = false;
    }
    return it->second;
}

const Dag::Node* Dag::find_node(const NodeKey& key) const {
    auto it = nodes_.find(key);
    if (it == nodes_.end()) {
        return nullptr;
    }
    return &it->second;
}

}  // namespace tetris_v2::cc2
