#include "cold_clear_cpp/dag.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <vector>

namespace cold_clear_cpp {

namespace {

struct Child {
    Placement mv{};
    Eval::Reward reward{};
    Eval cached_eval{};
};

struct BackpropUpdate {
    std::size_t parent{};
    Placement mv{};
    Piece speculation_piece{};
    std::size_t child{};
};

struct SelectResult {
    enum class Kind { Failed, Done, Advance };
    Kind kind{Kind::Failed};
    Piece next_piece{Piece::I};
    Placement mv{};
};

enum class LayerType { Speculated, Known };

struct LayerCommon;
struct SpeculatedLayer;

struct KnownNode {
    std::vector<std::tuple<std::size_t, Placement, Piece>> parents;
    Eval eval{};
    std::vector<Child> children;
    std::atomic_bool expanding{false};

    KnownNode() = default;
    KnownNode(const KnownNode& other)
        : parents(other.parents), eval(other.eval), children(other.children),
          expanding(other.expanding.load(std::memory_order_relaxed)) {}
    KnownNode& operator=(const KnownNode& other) {
        parents = other.parents;
        eval = other.eval;
        children = other.children;
        expanding.store(other.expanding.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }
};

struct SpeculatedNode {
    std::vector<std::tuple<std::size_t, Placement, Piece>> parents;
    Eval eval{};
    std::array<std::vector<Child>, 7> children;
    std::atomic_bool expanding{false};
    PieceSet bag{all_pieces()};

    SpeculatedNode() = default;
    SpeculatedNode(const SpeculatedNode& other)
        : parents(other.parents),
          eval(other.eval),
          children(other.children),
          expanding(other.expanding.load(std::memory_order_relaxed)),
          bag(other.bag) {}
    SpeculatedNode& operator=(const SpeculatedNode& other) {
        parents = other.parents;
        eval = other.eval;
        children = other.children;
        expanding.store(other.expanding.load(std::memory_order_relaxed), std::memory_order_relaxed);
        bag = other.bag;
        return *this;
    }
};

bool update_child(std::vector<Child>& list, const Placement& placement, const Eval& child_eval) {
    auto it = std::find_if(list.begin(), list.end(), [&](const Child& c) { return c.mv == placement; });
    if (it == list.end()) {
        return false;
    }

    it->cached_eval = child_eval + it->reward;
    auto index = static_cast<std::size_t>(std::distance(list.begin(), it));

    while (index > 0 && list[index - 1].cached_eval < list[index].cached_eval) {
        std::swap(list[index - 1], list[index]);
        --index;
    }
    while (index + 1 < list.size() && list[index + 1].cached_eval > list[index].cached_eval) {
        std::swap(list[index + 1], list[index]);
        ++index;
    }

    return index == 0;
}

struct KnownLayer {
    KnownLayer() = default;
    explicit KnownLayer(Piece p) : piece(p) {}

    void initialize_root(const GameState& root);
    std::vector<Placement> suggest(const GameState& state) const;
    SelectResult select(const GameState& state, double exploration) const;
    Eval get_eval(std::size_t raw) const;
    Eval create_node(const ChildData& child, std::size_t parent, Piece speculation_piece);
    std::vector<BackpropUpdate> expand(
        LayerCommon* next_layer, const GameState& parent_state, const ChildrenByPiece& children);
    std::vector<BackpropUpdate> backprop(
        const std::vector<BackpropUpdate>& updates, LayerCommon* next_layer);
    void absorb_from_speculated(const SpeculatedLayer& layer, Piece p);

    mutable StateMap<KnownNode> states;
    Piece piece{Piece::I};
};

struct SpeculatedLayer {
    void initialize_root(const GameState& root);
    std::vector<Placement> suggest(const GameState& state) const;
    SelectResult select(const GameState& state, double exploration) const;
    Eval get_eval(std::size_t raw) const;
    Eval create_node(const ChildData& child, std::size_t parent, Piece speculation_piece);
    std::vector<BackpropUpdate> expand(
        LayerCommon* next_layer, const GameState& parent_state, const ChildrenByPiece& children);
    std::vector<BackpropUpdate> backprop(
        const std::vector<BackpropUpdate>& updates, LayerCommon* next_layer);

    mutable StateMap<SpeculatedNode> states;
};

struct LayerCommon {
    LayerType type{LayerType::Speculated};
    Piece piece{Piece::I};
    SpeculatedLayer speculated{};
    KnownLayer known{};
    std::unique_ptr<LayerCommon> next{};

    LayerCommon();

    LayerCommon* ensure_next();
    void initialize_root(const GameState& root);
    std::optional<Piece> piece_if_known() const;
    bool despeculate(Piece target_piece);
    std::vector<Placement> suggest(const GameState& state) const;
    SelectResult select(const GameState& state, bool speculate, double exploration) const;
    std::vector<BackpropUpdate> expand(
        LayerCommon* next_layer, const GameState& parent_state, const ChildrenByPiece& children);
    std::vector<BackpropUpdate> backprop(
        const std::vector<BackpropUpdate>& updates, LayerCommon* next_layer);
    Eval get_eval(std::size_t raw) const;
    std::vector<Eval> create_nodes(
        const std::vector<ChildData>& children, std::size_t parent_index, Piece speculation_piece);
};

LayerCommon::LayerCommon() = default;

LayerCommon* LayerCommon::ensure_next() {
    if (!next) {
        next = std::make_unique<LayerCommon>();
    }
    return next.get();
}

void LayerCommon::initialize_root(const GameState& root) {
    if (type == LayerType::Known) {
        known.initialize_root(root);
    } else {
        speculated.initialize_root(root);
    }
}

std::optional<Piece> LayerCommon::piece_if_known() const {
    return type == LayerType::Known ? std::optional<Piece>{piece} : std::nullopt;
}

bool LayerCommon::despeculate(Piece target_piece) {
    if (type == LayerType::Known) {
        return false;
    }
    known.absorb_from_speculated(speculated, target_piece);
    piece = target_piece;
    type = LayerType::Known;
    return true;
}

std::vector<Placement> LayerCommon::suggest(const GameState& state) const {
    if (type == LayerType::Known) {
        return known.suggest(state);
    }
    return speculated.suggest(state);
}

SelectResult LayerCommon::select(const GameState& state, bool speculate, double exploration) const {
    if (type == LayerType::Known) {
        return known.select(state, exploration);
    }
    if (!speculate) {
        return {SelectResult::Kind::Failed};
    }
    return speculated.select(state, exploration);
}

std::vector<BackpropUpdate> LayerCommon::expand(
    LayerCommon* next_layer, const GameState& parent_state, const ChildrenByPiece& children) {
    if (type == LayerType::Known) {
        return known.expand(next_layer, parent_state, children);
    }
    return speculated.expand(next_layer, parent_state, children);
}

std::vector<BackpropUpdate> LayerCommon::backprop(
    const std::vector<BackpropUpdate>& updates, LayerCommon* next_layer) {
    if (type == LayerType::Known) {
        return known.backprop(updates, next_layer);
    }
    return speculated.backprop(updates, next_layer);
}

Eval LayerCommon::get_eval(std::size_t raw) const {
    if (type == LayerType::Known) {
        return known.get_eval(raw);
    }
    return speculated.get_eval(raw);
}

std::vector<Eval> LayerCommon::create_nodes(
    const std::vector<ChildData>& children, std::size_t parent_index, Piece speculation_piece) {
    std::vector<Eval> evals;
    evals.reserve(children.size());
    if (type == LayerType::Known) {
        for (const auto& child : children) {
            evals.push_back(known.create_node(child, parent_index, speculation_piece));
        }
    } else {
        for (const auto& child : children) {
            evals.push_back(speculated.create_node(child, parent_index, speculation_piece));
        }
    }
    return evals;
}

void KnownLayer::initialize_root(const GameState& root) {
    states.get_or_insert_with(root, [] { return KnownNode{}; });
}

std::vector<Placement> KnownLayer::suggest(const GameState& state) const {
    std::vector<Placement> result;
    auto* node = states.get(state);
    if (!node || node->children.empty()) {
        return result;
    }
    result.push_back(node->children.front().mv);
    return result;
}

SelectResult KnownLayer::select(const GameState& state, double exploration) const {
    auto* node = states.get(state);
    if (!node) {
        return {SelectResult::Kind::Failed};
    }
    if (node->children.empty()) {
        bool already = node->expanding.exchange(true, std::memory_order_relaxed);
        return {already ? SelectResult::Kind::Failed : SelectResult::Kind::Done};
    }

    std::random_device rd;
    thread_local std::mt19937 rng(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double s = dist(rng);
    auto idx =
        static_cast<std::size_t>(std::fmod(-std::log(s) / exploration, static_cast<double>(node->children.size())));
    return {SelectResult::Kind::Advance, piece, node->children[idx].mv};
}

Eval KnownLayer::get_eval(std::size_t raw) const {
    auto* node = states.get_raw(raw);
    return node ? node->eval : Eval{};
}

Eval KnownLayer::create_node(const ChildData& child, std::size_t parent, Piece speculation_piece) {
    auto& node = states.get_or_insert_with(child.resulting_state, [&] {
        KnownNode n;
        n.eval = child.eval;
        return n;
    });
    node.parents.emplace_back(parent, child.mv, speculation_piece);
    return node.eval;
}

std::vector<BackpropUpdate> KnownLayer::expand(
    LayerCommon* next_layer, const GameState& parent_state, const ChildrenByPiece& children) {
    std::vector<BackpropUpdate> next;
    const auto parent_index = states.index(parent_state);
    auto* parent = states.get_raw(parent_index);
    if (!parent) {
        parent = &states.get_or_insert_with(parent_state, [] { return KnownNode{}; });
    }

    const auto& list = children[piece_index(piece)];
    auto evals = next_layer->create_nodes(list, parent_index, piece);

    std::vector<Child> childs;
    childs.reserve(list.size());
    for (std::size_t i = 0; i < list.size(); ++i) {
        childs.push_back(Child{
            list[i].mv,
            list[i].reward,
            evals[i] + list[i].reward,
        });
    }

    std::sort(childs.begin(), childs.end(), [](const Child& a, const Child& b) {
        return a.cached_eval > b.cached_eval;
    });

    std::vector<std::optional<Eval>> opts;
    opts.push_back(childs.empty() ? std::optional<Eval>{} : std::optional<Eval>{childs.front().cached_eval});
    parent->eval = Eval::average(opts);
    parent->children = std::move(childs);

    for (const auto& [grand, mv, speculation_piece] : parent->parents) {
        next.push_back(BackpropUpdate{grand, mv, speculation_piece, parent_index});
    }
    parent->expanding.store(false, std::memory_order_relaxed);
    return next;
}

std::vector<BackpropUpdate> KnownLayer::backprop(
    const std::vector<BackpropUpdate>& updates, LayerCommon* next_layer) {
    std::vector<BackpropUpdate> new_updates;
    for (const auto& update : updates) {
        if (update.speculation_piece != piece) {
            continue;
        }
        auto* parent = states.get_raw(update.parent);
        if (!parent) {
            continue;
        }
        auto child_eval = next_layer->get_eval(update.child);
        bool is_best = update_child(parent->children, update.mv, child_eval);
        if (is_best) {
            auto new_eval = parent->children.front().cached_eval;
            if (parent->eval != new_eval) {
                parent->eval = new_eval;
                for (const auto& [grand, mv, speculation_piece] : parent->parents) {
                    new_updates.push_back(BackpropUpdate{grand, mv, speculation_piece, update.parent});
                }
            }
        }
    }
    return new_updates;
}

void KnownLayer::absorb_from_speculated(const SpeculatedLayer& layer, Piece p) {
    piece = p;
    StateMap<KnownNode> new_states;
    layer.states.for_each([&](const GameState& state, const SpeculatedNode& node) {
        KnownNode known_node;
        known_node.parents = node.parents;
        known_node.eval = node.eval;
        known_node.children = node.children[piece_index(p)];
        known_node.expanding.store(node.expanding.load(std::memory_order_relaxed), std::memory_order_relaxed);
        new_states.get_or_insert_with(state, [&] { return known_node; });
    });
    states = std::move(new_states);
}

void SpeculatedLayer::initialize_root(const GameState& root) {
    states.get_or_insert_with(root, [&] {
        SpeculatedNode node;
        node.bag = root.bag;
        return node;
    });
}

std::vector<Placement> SpeculatedLayer::suggest(const GameState& state) const {
    std::vector<Placement> result;
    auto* node = states.get(state);
    if (!node) {
        return result;
    }
    for (auto p : kAllPieces) {
        if (!state.bag.test(piece_index(p))) {
            continue;
        }
        const auto& list = node->children[piece_index(p)];
        if (!list.empty()) {
            result.push_back(list.front().mv);
        }
    }
    return result;
}

SelectResult SpeculatedLayer::select(const GameState& state, double exploration) const {
    auto* node = states.get(state);
    if (!node) {
        return {SelectResult::Kind::Failed};
    }

    bool has_children = false;
    for (auto p : kAllPieces) {
        if (!node->children[piece_index(p)].empty()) {
            has_children = true;
            break;
        }
    }
    if (!has_children) {
        bool already = node->expanding.exchange(true, std::memory_order_relaxed);
        return {already ? SelectResult::Kind::Failed : SelectResult::Kind::Done};
    }

    std::random_device rd;
    thread_local std::mt19937 rng(rd());
    std::uniform_int_distribution<int> pick_piece(0, static_cast<int>(state.bag.count() - 1));
    int piece_choice = pick_piece(rng);
    Piece chosen{Piece::I};
    int seen = 0;
    for (auto p : kAllPieces) {
        if (!state.bag.test(piece_index(p))) {
            continue;
        }
        if (seen == piece_choice) {
            chosen = p;
            break;
        }
        ++seen;
    }
    auto& list = node->children[piece_index(chosen)];
    if (list.empty()) {
        return {SelectResult::Kind::Failed};
    }

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double s = dist(rng);
    auto idx =
        static_cast<std::size_t>(std::fmod(-std::log(s) / exploration, static_cast<double>(list.size())));
    return {SelectResult::Kind::Advance, chosen, list[idx].mv};
}

Eval SpeculatedLayer::get_eval(std::size_t raw) const {
    auto* node = states.get_raw(raw);
    return node ? node->eval : Eval{};
}

Eval SpeculatedLayer::create_node(const ChildData& child, std::size_t parent, Piece speculation_piece) {
    auto& node = states.get_or_insert_with(child.resulting_state, [&] {
        SpeculatedNode n;
        n.eval = child.eval;
        n.bag = child.resulting_state.bag;
        return n;
    });
    node.parents.emplace_back(parent, child.mv, speculation_piece);
    return node.eval;
}

std::vector<BackpropUpdate> SpeculatedLayer::expand(
    LayerCommon* next_layer, const GameState& parent_state, const ChildrenByPiece& children) {
    std::vector<BackpropUpdate> next;
    const auto parent_index = states.index(parent_state);
    auto* parent = states.get_raw(parent_index);
    if (!parent) {
        parent = &states.get_or_insert_with(parent_state, [&] {
            SpeculatedNode node;
            node.bag = parent_state.bag;
            return node;
        });
    }

    for (auto p : kAllPieces) {
        const auto& list = children[piece_index(p)];
        auto evals = next_layer->create_nodes(list, parent_index, p);
        auto& target_list = parent->children[piece_index(p)];
        target_list.clear();
        target_list.reserve(list.size());
        for (std::size_t i = 0; i < list.size(); ++i) {
            target_list.push_back(Child{
                list[i].mv,
                list[i].reward,
                evals[i] + list[i].reward,
            });
        }
        std::sort(
            target_list.begin(), target_list.end(),
            [](const Child& a, const Child& b) { return a.cached_eval > b.cached_eval; });
    }

    std::vector<std::optional<Eval>> best;
    for (auto p : kAllPieces) {
        if (!parent->bag.test(piece_index(p))) {
            continue;
        }
        const auto& child_list = parent->children[piece_index(p)];
        best.push_back(
            child_list.empty() ? std::optional<Eval>{} : std::optional<Eval>{child_list.front().cached_eval});
    }
    parent->eval = Eval::average(best);

    for (const auto& [grand, mv, speculation_piece] : parent->parents) {
        next.push_back(BackpropUpdate{grand, mv, speculation_piece, parent_index});
    }
    parent->expanding.store(false, std::memory_order_relaxed);
    return next;
}

std::vector<BackpropUpdate> SpeculatedLayer::backprop(
    const std::vector<BackpropUpdate>& updates, LayerCommon* next_layer) {
    std::vector<BackpropUpdate> new_updates;
    for (const auto& update : updates) {
        auto* parent = states.get_raw(update.parent);
        if (!parent) {
            continue;
        }
        auto child_eval = next_layer->get_eval(update.child);
        auto& list = parent->children[piece_index(update.speculation_piece)];
        bool is_best = update_child(list, update.mv, child_eval);
        if (is_best) {
            std::vector<std::optional<Eval>> best;
            for (auto p : kAllPieces) {
                if (!parent->bag.test(piece_index(p))) {
                    continue;
                }
                const auto& child_list = parent->children[piece_index(p)];
                best.push_back(
                    child_list.empty() ? std::optional<Eval>{}
                                       : std::optional<Eval>{child_list.front().cached_eval});
            }
            auto new_eval = Eval::average(best);
            if (parent->eval != new_eval) {
                parent->eval = new_eval;
                for (const auto& [grand, mv, speculation_piece] : parent->parents) {
                    new_updates.push_back(BackpropUpdate{grand, mv, speculation_piece, update.parent});
                }
            }
        }
    }
    return new_updates;
}

}  // namespace

struct Dag::Impl {
    GameState root{};
    std::unique_ptr<LayerCommon> top_layer;

    Impl(const GameState& root_state, const std::vector<Piece>& queue) : root(root_state) {
        top_layer = std::make_unique<LayerCommon>();
        top_layer->initialize_root(root);
        auto* layer = top_layer.get();
        for (auto piece : queue) {
            layer->despeculate(piece);
            layer = layer->ensure_next();
        }
    }

    std::optional<Selection> select(bool speculate, double exploration) const {
        std::vector<LayerCommon*> layers;
        layers.push_back(top_layer.get());
        GameState game_state = root;
        auto* self = const_cast<Impl*>(this);

        while (true) {
            auto* layer = layers.back();
            auto res = layer->select(game_state, speculate, exploration);
            if (res.kind == SelectResult::Kind::Failed) {
                return std::nullopt;
            } else if (res.kind == SelectResult::Kind::Done) {
                auto next_piece = layer->piece_if_known();
                return Selection{
                    game_state,
                    next_piece,
                    [self, layers, game_state](const ChildrenByPiece& children) {
                        self->expand(layers, game_state, children);
                    }};
            } else if (res.kind == SelectResult::Kind::Advance) {
                game_state.advance(res.next_piece, res.mv);
                layer = layer->ensure_next();
                layers.push_back(layer);
            }
        }
    }

    void expand(std::vector<LayerCommon*> layers, const GameState& state, const ChildrenByPiece& children) {
        auto* start_layer = layers.back();
        layers.pop_back();
        auto* child_layer = start_layer->ensure_next();
        auto updates = start_layer->expand(child_layer, state, children);
        while (!layers.empty() && !updates.empty()) {
            auto* layer = layers.back();
            layers.pop_back();
            auto* next_layer = child_layer;
            child_layer = layer;
            updates = layer->backprop(updates, next_layer);
        }
    }

    void advance(const Placement& mv) {
        auto current = std::move(top_layer);
        auto piece = current->piece_if_known().value_or(Piece::I);
        root.advance(piece, mv);
        auto next = std::move(current->next);
        if (!next) {
            next = std::make_unique<LayerCommon>();
        }
        top_layer = std::move(next);
        top_layer->initialize_root(root);
    }

    void add_piece(Piece piece) {
        auto* layer = top_layer.get();
        while (true) {
            if (layer->despeculate(piece)) {
                return;
            }
            layer = layer->ensure_next();
        }
    }

    std::vector<Placement> suggest() const { return top_layer->suggest(root); }
};

Dag::Dag(const GameState& root, const std::vector<Piece>& queue) : impl_(std::make_unique<Impl>(root, queue)) {}
Dag::~Dag() = default;
Dag::Dag(Dag&&) noexcept = default;
Dag& Dag::operator=(Dag&&) noexcept = default;

void Dag::advance(const Placement& mv) { impl_->advance(mv); }

void Dag::add_piece(Piece piece) { impl_->add_piece(piece); }

std::vector<Placement> Dag::suggest() const { return impl_->suggest(); }

std::optional<Dag::Selection> Dag::select(bool speculate, double exploration) const {
    return impl_->select(speculate, exploration);
}

}  // namespace cold_clear_cpp
