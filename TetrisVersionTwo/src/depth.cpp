#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tetris_v2/env.hpp"
#include "tetris_v2/piece_defs.hpp"

namespace tetris_v2::depth {

namespace {

constexpr double kTieEps = 1e-9;

std::size_t hash_combine(std::size_t seed, std::size_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
    return seed;
}

int to_int(bool v) { return v ? 1 : 0; }

std::uint64_t to_u64_hash(std::size_t h) {
    if constexpr (sizeof(std::size_t) >= sizeof(std::uint64_t)) {
        return static_cast<std::uint64_t>(h);
    }
    return static_cast<std::uint64_t>(h) ^ (static_cast<std::uint64_t>(h) << 32u);
}

}  // namespace

// exhaustive search section
struct SearchConfig {
    int depth{1};
    double gamma{1.0};
    bool deduplicate_successors{true};
    bool use_transposition_table{false};
    bool collect_debug_info{true};
    std::uint64_t max_nodes{0};
};

// evaluation section
struct HeuristicWeights {
    double holes{-10.0};
    double aggregate_height{-0.4};
    double max_height{-0.9};
    double bumpiness{-0.35};
    double rows_cleared{1.5};
    double landing_height{-0.2};
    double row_transitions{-0.55};
    double column_transitions{-0.45};
    double cumulative_wells{-0.35};
    double max_well_depth{-0.2};
    double hole_depth{-0.45};
    double rows_with_holes{-0.8};
    double covered_holes{-0.9};
    double eroded_cells{0.8};

    double immediate_score_delta{1.0};
    double immediate_lines_cleared{10.0};
    double terminal_penalty{-100000.0};
    double leaf_terminal_penalty{-100000.0};
};

// feature extraction section
struct BoardFeatures {
    double holes{0.0};
    double aggregate_height{0.0};
    double max_height{0.0};
    double bumpiness{0.0};
    double rows_cleared{0.0};
    double landing_height{0.0};
    double row_transitions{0.0};
    double column_transitions{0.0};
    double cumulative_wells{0.0};
    double max_well_depth{0.0};
    double hole_depth{0.0};
    double rows_with_holes{0.0};
    double covered_holes{0.0};
    double eroded_cells{0.0};
    bool terminal{false};
};

struct CandidateInfo {
    std::size_t action_index{0};
    bool use_hold{false};
    std::size_t placement_index{0};
    double immediate_reward{0.0};
    double child_value{0.0};
    double total_value{0.0};
};

struct SearchResult {
    double value{-std::numeric_limits<double>::infinity()};
    std::size_t best_action_index{0};
    bool has_action{false};
    std::uint64_t nodes_expanded{0};
    bool hit_node_limit{false};
    std::vector<CandidateInfo> candidates{};
};

struct CandidateAction {
    bool use_hold{false};
    std::size_t placement_index{0};
    std::size_t action_index{0};
    PlacementOption option{};
};

struct NodeState {
    EnvSnapshot snapshot{};
    int rows_cleared{0};
    int landing_height{0};
    int eroded_cells{0};
};

struct Transition {
    NodeState child{};
    StepResult step_result{};
};

struct TTKey {
    std::uint64_t hash{0};
    int depth{0};

    bool operator==(const TTKey& rhs) const { return hash == rhs.hash && depth == rhs.depth; }
};

struct TTKeyHash {
    std::size_t operator()(const TTKey& key) const noexcept {
        std::size_t h = static_cast<std::size_t>(key.hash);
        return hash_combine(h, static_cast<std::size_t>(key.depth));
    }
};

struct Choice {
    bool use_hold{false};
    std::size_t placement_index{0};
    float score{0.0f};
};

struct ThinkStats {
    std::uint64_t nodes{0};
    std::uint64_t selections{0};
    std::uint64_t expansions{0};
    double think_ms{0.0};
    double nps{0.0};
    int budget_miss{0};
};

class Bot {
public:
    bool sync_from_env(const ModernTetrisEnv& env) {
        std::lock_guard<std::mutex> lock(mutex_);
        synced_snapshot_ = env.snapshot();
        synced_valid_ = true;
        last_choice_.reset();
        last_root_candidates_.clear();
        return true;
    }

    bool set_config(
        int depth,
        double gamma,
        bool deduplicate_successors,
        bool use_transposition_table,
        bool collect_debug_info,
        std::uint64_t max_nodes) {
        if (depth < 1 || !std::isfinite(gamma) || gamma < 0.0) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        config_.depth = depth;
        config_.gamma = gamma;
        config_.deduplicate_successors = deduplicate_successors;
        config_.use_transposition_table = use_transposition_table;
        config_.collect_debug_info = collect_debug_info;
        config_.max_nodes = max_nodes;
        return true;
    }

    bool choose(int think_ms, Choice* choice_out, ThinkStats* stats_out) {
        (void)think_ms;

        EnvSnapshot snapshot{};
        SearchConfig config{};
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!synced_valid_) {
                return false;
            }
            snapshot = synced_snapshot_;
            config = config_;
        }

        const auto t0 = std::chrono::steady_clock::now();
        NodeState root{};
        root.snapshot = snapshot;
        root.rows_cleared = 0;
        root.landing_height = 0;
        root.eroded_cells = 0;

        search_cache_.transposition.clear();
        search_cache_.node_limit_hit = false;
        const SearchResult result = choose_action_exhaustive(root, config, weights_, &search_cache_);
        const auto t1 = std::chrono::steady_clock::now();

        if (!result.has_action || !std::isfinite(result.value)) {
            return false;
        }

        Choice choice{};
        choice.use_hold = result.candidates[result.best_action_index].use_hold;
        choice.placement_index = result.candidates[result.best_action_index].placement_index;
        choice.score = static_cast<float>(result.value);

        const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ThinkStats stats{};
        stats.nodes = result.nodes_expanded;
        stats.selections = result.nodes_expanded;
        stats.expansions = result.nodes_expanded;
        stats.think_ms = elapsed_ms;
        if (elapsed_ms > 0.0) {
            stats.nps = static_cast<double>(stats.nodes) / (elapsed_ms / 1000.0);
        }
        stats.budget_miss = 0;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            last_choice_ = choice;
            if (config.collect_debug_info) {
                last_root_candidates_ = result.candidates;
            } else {
                last_root_candidates_.clear();
            }
        }

        if (choice_out) {
            *choice_out = choice;
        }
        if (stats_out) {
            *stats_out = stats;
        }
        return true;
    }

    bool apply_last_choice(
        ModernTetrisEnv& env,
        StepResult* result_out,
        int* used_hold_out,
        std::size_t* placement_index_out) {
        std::optional<Choice> choice{};
        {
            std::lock_guard<std::mutex> lock(mutex_);
            choice = last_choice_;
        }
        if (!choice.has_value()) {
            return false;
        }

        if (used_hold_out) {
            *used_hold_out = 0;
        }
        if (placement_index_out) {
            *placement_index_out = choice->placement_index;
        }

        if (choice->use_hold) {
            auto hold_result = env.step(Action::Hold);
            if (!hold_result.hold_used || hold_result.game_over) {
                if (result_out) {
                    *result_out = hold_result;
                }
                return false;
            }
            if (used_hold_out) {
                *used_hold_out = 1;
            }
        }

        auto apply_result = env.apply_placement_index(choice->placement_index);
        if (result_out) {
            *result_out = apply_result;
        }
        if (!apply_result.action_succeeded) {
            return false;
        }

        sync_from_env(env);
        return true;
    }

    bool choose_and_apply(
        ModernTetrisEnv& env,
        int think_ms,
        StepResult* result_out,
        Choice* choice_out,
        ThinkStats* stats_out,
        int* used_hold_out,
        std::size_t* placement_index_out) {
        if (!sync_from_env(env)) {
            return false;
        }

        Choice choice{};
        ThinkStats stats{};
        if (!choose(think_ms, &choice, &stats)) {
            return false;
        }
        if (!apply_last_choice(env, result_out, used_hold_out, placement_index_out)) {
            return false;
        }

        if (choice_out) {
            *choice_out = choice;
        }
        if (stats_out) {
            *stats_out = stats;
        }
        return true;
    }

private:
    // environment integration boundary
    static std::vector<CandidateAction> enumerate_candidates(const EnvSnapshot& snapshot) {
        std::vector<CandidateAction> out{};

        ModernTetrisEnv env{};
        env.restore(snapshot);
        if (env.state().game_over || env.state().active.piece == Piece::None) {
            return out;
        }

        auto append_from_env = [&](const ModernTetrisEnv& local_env, bool use_hold) {
            const auto options = local_env.enumerate_active_piece_placements();
            const std::size_t base = out.size();
            out.reserve(base + options.size());
            for (std::size_t i = 0; i < options.size(); ++i) {
                CandidateAction action{};
                action.use_hold = use_hold;
                action.placement_index = i;
                action.action_index = out.size();
                action.option = options[i];
                out.push_back(action);
            }
        };

        append_from_env(env, false);

        if (env.state().hold_available) {
            ModernTetrisEnv hold_env{};
            hold_env.restore(snapshot);
            const auto hold_result = hold_env.step(Action::Hold);
            if (hold_result.hold_used && !hold_env.state().game_over) {
                append_from_env(hold_env, true);
            }
        }

        return out;
    }

    static int compute_landing_height(const PlacementOption& option) {
        const auto cells = piece_cells(option.placement.piece, option.placement.rotation);
        int sum_y = 0;
        for (const auto& c : cells) {
            sum_y += (option.placement.y + c.y);
        }
        return sum_y / 4;
    }

    static int compute_eroded_cells(const Board& board_before, const PlacementOption& option) {
        if (option.lines_cleared <= 0) {
            return 0;
        }

        const auto cells = piece_cells(option.placement.piece, option.placement.rotation);
        int piece_cells_in_cleared_rows = 0;
        for (const auto& c : cells) {
            const int x = option.placement.x + c.x;
            const int y = option.placement.y + c.y;
            if (x < 0 || x >= Board::kWidth || y < 0 || y >= Board::kRows) {
                continue;
            }
            if (!option.cleared_rows[static_cast<std::size_t>(y)]) {
                continue;
            }
            if (!board_before.occupied(x, y)) {
                ++piece_cells_in_cleared_rows;
            }
        }

        return option.lines_cleared * piece_cells_in_cleared_rows;
    }

    static bool apply_candidate(
        const EnvSnapshot& snapshot,
        const CandidateAction& action,
        Transition* transition_out) {
        if (!transition_out) {
            return false;
        }

        ModernTetrisEnv env{};
        env.restore(snapshot);
        const auto board_before = env.state().board;

        if (action.use_hold) {
            const auto hold_result = env.step(Action::Hold);
            if (!hold_result.hold_used || hold_result.game_over) {
                return false;
            }
        }

        const auto result = env.apply_placement_index(action.placement_index);
        if (!result.action_succeeded) {
            return false;
        }

        NodeState child{};
        child.snapshot = env.snapshot();
        child.rows_cleared = result.lines_cleared;
        child.landing_height = compute_landing_height(action.option);
        child.eroded_cells = compute_eroded_cells(board_before, action.option);

        transition_out->child = child;
        transition_out->step_result = result;
        return true;
    }

    // feature extraction section
    static BoardFeatures extract_board_features(const NodeState& state) {
        BoardFeatures out{};
        out.rows_cleared = static_cast<double>(state.rows_cleared);
        out.landing_height = static_cast<double>(state.landing_height);
        out.eroded_cells = static_cast<double>(state.eroded_cells);
        out.terminal = state.snapshot.state.game_over || state.snapshot.state.top_out;

        constexpr int kRows = Board::kVisibleRows;
        constexpr int kCols = Board::kWidth;

        std::array<int, kCols> heights{};
        std::array<bool, kRows> rows_with_hole{};
        rows_with_hole.fill(false);

        double holes = 0.0;
        double covered_holes = 0.0;
        double hole_depth = 0.0;

        for (int x = 0; x < kCols; ++x) {
            int highest = 0;
            int filled_above = 0;
            for (int y = kRows - 1; y >= 0; --y) {
                const bool filled = state.snapshot.state.board.occupied(x, y);
                if (filled) {
                    if (highest == 0) {
                        highest = y + 1;
                    }
                    ++filled_above;
                } else if (filled_above > 0) {
                    holes += 1.0;
                    covered_holes += 1.0;
                    hole_depth += static_cast<double>(filled_above);
                    rows_with_hole[static_cast<std::size_t>(y)] = true;
                }
            }
            heights[static_cast<std::size_t>(x)] = highest;
        }

        out.holes = holes;
        out.covered_holes = covered_holes;
        out.hole_depth = hole_depth;

        double aggregate_height = 0.0;
        double max_height = 0.0;
        double bumpiness = 0.0;
        for (int x = 0; x < kCols; ++x) {
            const double h = static_cast<double>(heights[static_cast<std::size_t>(x)]);
            aggregate_height += h;
            max_height = std::max(max_height, h);
            if (x > 0) {
                const double prev = static_cast<double>(heights[static_cast<std::size_t>(x - 1)]);
                bumpiness += std::abs(h - prev);
            }
        }
        out.aggregate_height = aggregate_height;
        out.max_height = max_height;
        out.bumpiness = bumpiness;

        double rows_with_holes = 0.0;
        for (bool has_hole : rows_with_hole) {
            rows_with_holes += has_hole ? 1.0 : 0.0;
        }
        out.rows_with_holes = rows_with_holes;

        double row_transitions = 0.0;
        for (int y = 0; y < kRows; ++y) {
            bool prev_filled = true;
            for (int x = 0; x < kCols; ++x) {
                const bool filled = state.snapshot.state.board.occupied(x, y);
                if (filled != prev_filled) {
                    row_transitions += 1.0;
                }
                prev_filled = filled;
            }
            if (!prev_filled) {
                row_transitions += 1.0;
            }
        }
        out.row_transitions = row_transitions;

        double column_transitions = 0.0;
        for (int x = 0; x < kCols; ++x) {
            bool prev_filled = true;
            for (int y = kRows - 1; y >= 0; --y) {
                const bool filled = state.snapshot.state.board.occupied(x, y);
                if (filled != prev_filled) {
                    column_transitions += 1.0;
                }
                prev_filled = filled;
            }
            if (!prev_filled) {
                column_transitions += 1.0;
            }
        }
        out.column_transitions = column_transitions;

        double cumulative_wells = 0.0;
        double max_well_depth = 0.0;
        for (int x = 0; x < kCols; ++x) {
            int left_h = (x == 0) ? kRows : heights[static_cast<std::size_t>(x - 1)];
            int right_h = (x + 1 == kCols) ? kRows : heights[static_cast<std::size_t>(x + 1)];
            int depth = std::max(0, std::min(left_h, right_h) - heights[static_cast<std::size_t>(x)]);
            if (depth > 0) {
                cumulative_wells += static_cast<double>(depth * (depth + 1) / 2);
                max_well_depth = std::max(max_well_depth, static_cast<double>(depth));
            }
        }
        out.cumulative_wells = cumulative_wells;
        out.max_well_depth = max_well_depth;

        return out;
    }

    // evaluation section
    static double evaluate_board_features(const BoardFeatures& f, const HeuristicWeights& w) {
        if (f.terminal) {
            return w.leaf_terminal_penalty;
        }

        double value = 0.0;
        value += w.holes * f.holes;
        value += w.aggregate_height * f.aggregate_height;
        value += w.max_height * f.max_height;
        value += w.bumpiness * f.bumpiness;
        value += w.rows_cleared * f.rows_cleared;
        value += w.landing_height * f.landing_height;
        value += w.row_transitions * f.row_transitions;
        value += w.column_transitions * f.column_transitions;
        value += w.cumulative_wells * f.cumulative_wells;
        value += w.max_well_depth * f.max_well_depth;
        value += w.hole_depth * f.hole_depth;
        value += w.rows_with_holes * f.rows_with_holes;
        value += w.covered_holes * f.covered_holes;
        value += w.eroded_cells * f.eroded_cells;
        return value;
    }

    static double compute_immediate_reward(const StepResult& result, const HeuristicWeights& w) {
        double reward = 0.0;
        reward += w.immediate_score_delta * static_cast<double>(result.reward);
        reward += w.immediate_lines_cleared * static_cast<double>(result.lines_cleared);
        if (result.game_over || result.top_out) {
            reward += w.terminal_penalty;
        }
        return reward;
    }

    static std::uint64_t hash_snapshot(const EnvSnapshot& snapshot) {
        std::size_t h = 1469598103934665603ull;

        for (const auto row : snapshot.state.board.rows()) {
            h = hash_combine(h, static_cast<std::size_t>(row));
        }

        const auto& state = snapshot.state;
        h = hash_combine(h, static_cast<std::size_t>(state.active.piece));
        h = hash_combine(h, static_cast<std::size_t>(state.active.rotation));
        h = hash_combine(h, static_cast<std::size_t>(state.active.x + 64));
        h = hash_combine(h, static_cast<std::size_t>(state.active.y + 64));
        h = hash_combine(h, static_cast<std::size_t>(state.hold.has_value() ? *state.hold : Piece::None));
        h = hash_combine(h, static_cast<std::size_t>(to_int(state.hold_available)));
        h = hash_combine(h, static_cast<std::size_t>(to_int(state.game_over)));
        h = hash_combine(h, static_cast<std::size_t>(to_int(state.top_out)));
        h = hash_combine(h, static_cast<std::size_t>(state.combo + 2));
        h = hash_combine(h, static_cast<std::size_t>(to_int(state.back_to_back)));
        h = hash_combine(h, static_cast<std::size_t>(state.total_lines_cleared));
        h = hash_combine(h, state.queue.size());
        for (const auto piece : state.queue) {
            h = hash_combine(h, static_cast<std::size_t>(piece));
        }

        const auto& bag = snapshot.randomizer.bag_order();
        h = hash_combine(h, bag.size());
        h = hash_combine(h, snapshot.randomizer.bag_index());
        for (const auto piece : bag) {
            h = hash_combine(h, static_cast<std::size_t>(piece));
        }

        return to_u64_hash(h);
    }

    struct SearchCache {
        std::unordered_map<TTKey, double, TTKeyHash> transposition{};
        bool node_limit_hit{false};
    };

    static double exhaustive_value(
        const NodeState& state,
        int depth,
        const SearchConfig& config,
        const HeuristicWeights& weights,
        std::uint64_t* nodes_expanded,
        SearchCache* cache) {
        if (!nodes_expanded || !cache) {
            return -std::numeric_limits<double>::infinity();
        }

        if (config.max_nodes > 0 && *nodes_expanded >= config.max_nodes) {
            cache->node_limit_hit = true;
            return evaluate_board_features(extract_board_features(state), weights);
        }

        ++(*nodes_expanded);

        if (depth <= 0 || state.snapshot.state.game_over || state.snapshot.state.top_out) {
            return evaluate_board_features(extract_board_features(state), weights);
        }

        if (config.use_transposition_table) {
            const TTKey key{hash_snapshot(state.snapshot), depth};
            auto it = cache->transposition.find(key);
            if (it != cache->transposition.end()) {
                return it->second;
            }
        }

        const auto actions = enumerate_candidates(state.snapshot);
        if (actions.empty()) {
            return evaluate_board_features(extract_board_features(state), weights);
        }

        double best = -std::numeric_limits<double>::infinity();
        std::unordered_set<std::uint64_t> dedup{};
        if (config.deduplicate_successors) {
            dedup.reserve(actions.size() * 2);
        }

        for (const auto& action : actions) {
            Transition transition{};
            if (!apply_candidate(state.snapshot, action, &transition)) {
                continue;
            }

            if (config.deduplicate_successors) {
                const auto h = hash_snapshot(transition.child.snapshot);
                if (!dedup.emplace(h).second) {
                    continue;
                }
            }

            const double immediate = compute_immediate_reward(transition.step_result, weights);
            const double child = exhaustive_value(
                transition.child, depth - 1, config, weights, nodes_expanded, cache);
            const double total = immediate + config.gamma * child;
            if (total > best) {
                best = total;
            }

            if (cache->node_limit_hit) {
                break;
            }
        }

        if (!std::isfinite(best)) {
            best = evaluate_board_features(extract_board_features(state), weights);
        }

        if (config.use_transposition_table) {
            cache->transposition[TTKey{hash_snapshot(state.snapshot), depth}] = best;
        }
        return best;
    }

    static SearchResult choose_action_exhaustive(
        const NodeState& root,
        const SearchConfig& config,
        const HeuristicWeights& weights,
        SearchCache* cache) {
        SearchResult result{};
        if (!cache) {
            return result;
        }

        const auto actions = enumerate_candidates(root.snapshot);
        if (actions.empty()) {
            result.value = evaluate_board_features(extract_board_features(root), weights);
            result.has_action = false;
            return result;
        }

        result.candidates.reserve(actions.size());
        std::unordered_set<std::uint64_t> dedup{};
        if (config.deduplicate_successors) {
            dedup.reserve(actions.size() * 2);
        }

        for (const auto& action : actions) {
            Transition transition{};
            if (!apply_candidate(root.snapshot, action, &transition)) {
                continue;
            }

            if (config.deduplicate_successors) {
                const auto h = hash_snapshot(transition.child.snapshot);
                if (!dedup.emplace(h).second) {
                    continue;
                }
            }

            const double immediate = compute_immediate_reward(transition.step_result, weights);
            double child_value = 0.0;
            if (config.depth > 1) {
                child_value = exhaustive_value(
                    transition.child,
                    config.depth - 1,
                    config,
                    weights,
                    &result.nodes_expanded,
                    cache);
            } else {
                child_value = evaluate_board_features(extract_board_features(transition.child), weights);
            }
            const double total = immediate + config.gamma * child_value;

            CandidateInfo candidate{};
            candidate.action_index = action.action_index;
            candidate.use_hold = action.use_hold;
            candidate.placement_index = action.placement_index;
            candidate.immediate_reward = immediate;
            candidate.child_value = child_value;
            candidate.total_value = total;

            const std::size_t idx = result.candidates.size();
            result.candidates.push_back(candidate);

            if (!result.has_action || total > result.value + kTieEps) {
                result.value = total;
                result.best_action_index = idx;
                result.has_action = true;
            } else if (std::abs(total - result.value) <= kTieEps) {
                const auto& best = result.candidates[result.best_action_index];
                if (candidate.action_index < best.action_index ||
                    (candidate.action_index == best.action_index &&
                     candidate.placement_index < best.placement_index)) {
                    result.best_action_index = idx;
                }
            }

            if (cache->node_limit_hit) {
                break;
            }
        }

        result.hit_node_limit = cache->node_limit_hit;
        return result;
    }

private:
    mutable std::mutex mutex_{};
    SearchConfig config_{};
    HeuristicWeights weights_{};
    EnvSnapshot synced_snapshot_{};
    bool synced_valid_{false};
    std::optional<Choice> last_choice_{};
    std::vector<CandidateInfo> last_root_candidates_{};
    SearchCache search_cache_{};
};

// bot entrypoint section
Bot* create_default() {
    try {
        return new Bot{};
    } catch (...) {
        return nullptr;
    }
}

void destroy(Bot* bot) { delete bot; }

bool sync_from_env(Bot* bot, const ModernTetrisEnv& env) {
    if (!bot) {
        return false;
    }
    return bot->sync_from_env(env);
}

bool set_config(
    Bot* bot,
    int depth,
    double gamma,
    bool deduplicate_successors,
    bool use_transposition_table,
    bool collect_debug_info,
    std::uint64_t max_nodes) {
    if (!bot) {
        return false;
    }
    return bot->set_config(
        depth,
        gamma,
        deduplicate_successors,
        use_transposition_table,
        collect_debug_info,
        max_nodes);
}

bool choose(
    Bot* bot,
    int think_ms,
    bool* use_hold_out,
    std::size_t* placement_index_out,
    float* score_out,
    std::uint64_t* nodes_out,
    double* think_ms_out,
    double* nps_out,
    int* budget_miss_out) {
    if (!bot) {
        return false;
    }

    Choice choice{};
    ThinkStats stats{};
    if (!bot->choose(think_ms, &choice, &stats)) {
        return false;
    }

    if (use_hold_out) {
        *use_hold_out = choice.use_hold;
    }
    if (placement_index_out) {
        *placement_index_out = choice.placement_index;
    }
    if (score_out) {
        *score_out = choice.score;
    }
    if (nodes_out) {
        *nodes_out = stats.nodes;
    }
    if (think_ms_out) {
        *think_ms_out = stats.think_ms;
    }
    if (nps_out) {
        *nps_out = stats.nps;
    }
    if (budget_miss_out) {
        *budget_miss_out = stats.budget_miss;
    }
    return true;
}

bool apply_last_choice(
    Bot* bot,
    ModernTetrisEnv& env,
    StepResult* result_out,
    int* used_hold_out,
    std::size_t* placement_index_out) {
    if (!bot) {
        return false;
    }
    return bot->apply_last_choice(env, result_out, used_hold_out, placement_index_out);
}

bool choose_and_apply(
    Bot* bot,
    ModernTetrisEnv& env,
    int think_ms,
    StepResult* result_out,
    bool* use_hold_out,
    std::size_t* placement_index_out,
    float* score_out,
    std::uint64_t* nodes_out,
    double* think_ms_out,
    double* nps_out,
    int* budget_miss_out,
    int* used_hold_out) {
    if (!bot) {
        return false;
    }

    Choice choice{};
    ThinkStats stats{};
    std::size_t placement_index = 0;
    int used_hold = 0;
    if (!bot->choose_and_apply(
            env,
            think_ms,
            result_out,
            &choice,
            &stats,
            &used_hold,
            &placement_index)) {
        return false;
    }

    if (use_hold_out) {
        *use_hold_out = choice.use_hold;
    }
    if (placement_index_out) {
        *placement_index_out = placement_index;
    }
    if (score_out) {
        *score_out = choice.score;
    }
    if (nodes_out) {
        *nodes_out = stats.nodes;
    }
    if (think_ms_out) {
        *think_ms_out = stats.think_ms;
    }
    if (nps_out) {
        *nps_out = stats.nps;
    }
    if (budget_miss_out) {
        *budget_miss_out = stats.budget_miss;
    }
    if (used_hold_out) {
        *used_hold_out = used_hold;
    }
    return true;
}

}  // namespace tetris_v2::depth
