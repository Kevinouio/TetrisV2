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
#include <utility>
#include <vector>

#include "tetris_v2/env.hpp"
#include "tetris_v2/piece_defs.hpp"

namespace tetris_v2::beam {

namespace {

constexpr double kTieEps = 1e-9;

#ifndef TETRIS_V2_BEAM_PROFILE
#define TETRIS_V2_BEAM_PROFILE 0
#endif

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

std::uint64_t cleared_rows_to_mask(const std::array<bool, Board::kRows>& rows) {
    std::uint64_t mask = 0;
    constexpr int kMaxBits = 64;
    static_assert(Board::kRows <= kMaxBits, "Board rows exceed cleared-row mask width");
    for (int y = 0; y < Board::kRows; ++y) {
        if (rows[static_cast<std::size_t>(y)]) {
            mask |= (std::uint64_t{1} << static_cast<unsigned>(y));
        }
    }
    return mask;
}

bool row_is_cleared(std::uint64_t mask, int y) {
    if (y < 0 || y >= 64) {
        return false;
    }
    return ((mask >> static_cast<unsigned>(y)) & std::uint64_t{1}) != 0;
}

}  // namespace

// beam search section
struct SearchConfig {
    int depth{2};
    int beam_width{8};
    double gamma{1.0};
    bool deduplicate_successors{true};
    bool use_transposition_table{false};
    bool collect_debug_info{true};
    std::uint64_t max_nodes{0};
    bool enable_cheap_pruning{false};
};

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

    double top_out_penalty{-100000.0};
    double survival_bonus{0.0};
    double combo{0.0};
    double b2b{0.0};
    double perfect_clear{0.0};

    double immediate_score_delta{1.0};
    double immediate_lines_cleared{10.0};
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

struct BeamNode {
    NodeState state{};
    std::uint64_t state_hash{0};
    std::size_t root_action_index{0};
    double cumulative_reward{0.0};
    double heuristic_value{0.0};
    double priority_score{0.0};
    double final_value{0.0};
    int depth_remaining{0};
    std::size_t action_index{0};
    std::size_t placement_index{0};
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

// performance instrumentation
struct PerfMetrics {
    bool enabled{false};

    double total_ms{0.0};
    double legal_generation_ms{0.0};
    double apply_ms{0.0};
    double feature_ms{0.0};
    double eval_ms{0.0};
    double dedup_ms{0.0};
    double pruning_ms{0.0};

    std::uint64_t nodes_expanded{0};
    std::uint64_t total_children_generated{0};
    std::uint64_t total_children_after_dedup{0};
    std::uint64_t total_children_after_prune{0};

    std::vector<std::uint32_t> children_generated_per_layer{};
    std::vector<std::uint32_t> children_after_dedup_per_layer{};
    std::vector<std::uint32_t> children_after_prune_per_layer{};

    void reset(int depth, bool enable_runtime) {
        enabled = enable_runtime;
        total_ms = 0.0;
        legal_generation_ms = 0.0;
        apply_ms = 0.0;
        feature_ms = 0.0;
        eval_ms = 0.0;
        dedup_ms = 0.0;
        pruning_ms = 0.0;
        nodes_expanded = 0;
        total_children_generated = 0;
        total_children_after_dedup = 0;
        total_children_after_prune = 0;

        const std::size_t layers = static_cast<std::size_t>(std::max(1, depth));
        children_generated_per_layer.assign(layers, 0);
        children_after_dedup_per_layer.assign(layers, 0);
        children_after_prune_per_layer.assign(layers, 0);
    }

    void add_children_generated(int layer, std::uint32_t count) {
        if (layer < 0 || static_cast<std::size_t>(layer) >= children_generated_per_layer.size()) {
            return;
        }
        children_generated_per_layer[static_cast<std::size_t>(layer)] += count;
        total_children_generated += count;
    }

    void set_children_after_dedup(int layer, std::uint32_t count) {
        if (layer < 0 || static_cast<std::size_t>(layer) >= children_after_dedup_per_layer.size()) {
            return;
        }
        children_after_dedup_per_layer[static_cast<std::size_t>(layer)] = count;
    }

    void set_children_after_prune(int layer, std::uint32_t count) {
        if (layer < 0 || static_cast<std::size_t>(layer) >= children_after_prune_per_layer.size()) {
            return;
        }
        children_after_prune_per_layer[static_cast<std::size_t>(layer)] = count;
    }

    void finalize_totals() {
        total_children_after_dedup = 0;
        total_children_after_prune = 0;
        for (auto v : children_after_dedup_per_layer) {
            total_children_after_dedup += v;
        }
        for (auto v : children_after_prune_per_layer) {
            total_children_after_prune += v;
        }
    }
};

#if TETRIS_V2_BEAM_PROFILE
class ScopedTimer {
public:
    using Clock = std::chrono::steady_clock;

    ScopedTimer(bool enabled, double* dst) : enabled_(enabled && dst != nullptr), dst_(dst) {
        if (enabled_) {
            start_ = Clock::now();
        }
    }

    ~ScopedTimer() {
        if (!enabled_) {
            return;
        }
        const auto end = Clock::now();
        *dst_ += std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    bool enabled_{false};
    double* dst_{nullptr};
    Clock::time_point start_{};
};
#else
class ScopedTimer {
public:
    ScopedTimer(bool, double*) {}
};
#endif

struct SearchCache {
    std::unordered_map<TTKey, double, TTKeyHash> transposition{};
    bool node_limit_hit{false};
};

struct SearchScratch {
    ModernTetrisEnv env{};
    std::vector<BeamNode> beam_current{};
    std::vector<BeamNode> beam_next{};
    std::unordered_map<std::uint64_t, std::size_t> dedup_index{};
    std::unordered_map<std::uint64_t, std::size_t> root_dedup_index{};

    void clear_for_search(const SearchConfig& config) {
        beam_current.clear();
        beam_next.clear();
        dedup_index.clear();
        root_dedup_index.clear();

        const std::size_t width = static_cast<std::size_t>(std::max(1, config.beam_width));
        beam_current.reserve(width * 4);
        beam_next.reserve(width * width);
        dedup_index.reserve(width * width * 2);
        root_dedup_index.reserve(width * 8);
    }
};

struct Choice {
    bool use_hold{false};
    std::size_t placement_index{0};
    float score{0.0f};
};

// benchmark / debug helpers
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
        int beam_width,
        double gamma,
        bool deduplicate_successors,
        bool use_transposition_table,
        bool collect_debug_info,
        std::uint64_t max_nodes) {
        if (depth < 1 || beam_width < 1 || !std::isfinite(gamma) || gamma < 0.0) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        config_.depth = depth;
        config_.beam_width = beam_width;
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
        config.beam_width = std::max(1, config.beam_width);

        PerfMetrics perf{};
        perf.reset(config.depth, config.collect_debug_info);
        search_scratch_.clear_for_search(config);

        const SearchResult result = choose_action_beam(
            root, config, weights_, &search_cache_, &search_scratch_, &perf);

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
                last_perf_ = perf;
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
    static int compute_landing_height(const ActivePiece& placement) {
        const auto cells = piece_cells(placement.piece, placement.rotation);
        int sum_y = 0;
        for (const auto& c : cells) {
            sum_y += (placement.y + c.y);
        }
        return sum_y / 4;
    }

    static int compute_eroded_cells(const Board& board_before, const PlacementOption& option) {
        if (option.lines_cleared <= 0) {
            return 0;
        }

        const auto cells = piece_cells(option.placement.piece, option.placement.rotation);
        const std::uint64_t cleared_rows_mask = cleared_rows_to_mask(option.cleared_rows);
        int piece_cells_in_cleared_rows = 0;
        for (const auto& c : cells) {
            const int x = option.placement.x + c.x;
            const int y = option.placement.y + c.y;
            if (x < 0 || x >= Board::kWidth || y < 0 || y >= Board::kRows) {
                continue;
            }
            if (!row_is_cleared(cleared_rows_mask, y)) {
                continue;
            }
            if (!board_before.occupied(x, y)) {
                ++piece_cells_in_cleared_rows;
            }
        }

        return option.lines_cleared * piece_cells_in_cleared_rows;
    }

    static bool apply_placement_option(
        const EnvSnapshot& snapshot_before,
        const PlacementOption& option,
        Transition* transition_out,
        SearchScratch* scratch,
        PerfMetrics* perf) {
        if (!transition_out || !scratch) {
            return false;
        }

        ScopedTimer timer(perf && perf->enabled, perf ? &perf->apply_ms : nullptr);

        auto& env = scratch->env;
        env.restore(snapshot_before);
        const auto& board_before = snapshot_before.state.board;

        const auto result = env.apply_placement_option_fast(option);
        if (!result.action_succeeded) {
            return false;
        }

        NodeState child{};
        child.snapshot = env.snapshot();
        child.rows_cleared = result.lines_cleared;
        child.landing_height = compute_landing_height(option.placement);
        child.eroded_cells = compute_eroded_cells(board_before, option);

        transition_out->child = std::move(child);
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
            return w.top_out_penalty;
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
        value += w.survival_bonus;
        return value;
    }

    static double evaluate_board_features_cheap(const BoardFeatures& f, const HeuristicWeights& w) {
        if (f.terminal) {
            return w.top_out_penalty;
        }

        double value = 0.0;
        value += w.holes * f.holes;
        value += w.aggregate_height * f.aggregate_height;
        value += w.max_height * f.max_height;
        value += w.bumpiness * f.bumpiness;
        value += w.rows_cleared * f.rows_cleared;
        value += w.landing_height * f.landing_height;
        value += w.survival_bonus;
        return value;
    }

    static double compute_immediate_reward(const StepResult& result, const HeuristicWeights& w) {
        double reward = 0.0;
        reward += w.immediate_score_delta * static_cast<double>(result.reward);
        reward += w.immediate_lines_cleared * static_cast<double>(result.lines_cleared);
        if (result.game_over || result.top_out) {
            reward += w.top_out_penalty;
        }
        return reward;
    }

    // deduplication / caching
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

    static double compute_priority(
        double cumulative_reward,
        double heuristic_for_priority,
        const SearchConfig& config,
        const HeuristicWeights& weights) {
        (void)config;
        (void)weights;
        return cumulative_reward + heuristic_for_priority;
    }

    static double compute_final_value(
        double cumulative_reward,
        double heuristic_value,
        const SearchConfig& config,
        const HeuristicWeights& weights) {
        (void)weights;
        return cumulative_reward + config.gamma * heuristic_value;
    }

    static bool tie_break_before(const BeamNode& lhs, const BeamNode& rhs) {
        if (lhs.root_action_index != rhs.root_action_index) {
            return lhs.root_action_index < rhs.root_action_index;
        }
        if (lhs.action_index != rhs.action_index) {
            return lhs.action_index < rhs.action_index;
        }
        return lhs.placement_index < rhs.placement_index;
    }

    static bool better_by_priority(const BeamNode& lhs, const BeamNode& rhs) {
        if (lhs.priority_score > rhs.priority_score + kTieEps) {
            return true;
        }
        if (rhs.priority_score > lhs.priority_score + kTieEps) {
            return false;
        }
        if (lhs.final_value > rhs.final_value + kTieEps) {
            return true;
        }
        if (rhs.final_value > lhs.final_value + kTieEps) {
            return false;
        }
        return tie_break_before(lhs, rhs);
    }

    static bool better_by_final_value(const BeamNode& lhs, const BeamNode& rhs) {
        if (lhs.final_value > rhs.final_value + kTieEps) {
            return true;
        }
        if (rhs.final_value > lhs.final_value + kTieEps) {
            return false;
        }
        if (lhs.priority_score > rhs.priority_score + kTieEps) {
            return true;
        }
        if (rhs.priority_score > lhs.priority_score + kTieEps) {
            return false;
        }
        return tie_break_before(lhs, rhs);
    }

    static void prune_to_beam_width(
        std::vector<BeamNode>* nodes,
        int beam_width,
        PerfMetrics* perf) {
        if (!nodes) {
            return;
        }

        ScopedTimer timer(perf && perf->enabled, perf ? &perf->pruning_ms : nullptr);

        const std::size_t width = static_cast<std::size_t>(std::max(1, beam_width));
        if (nodes->size() <= width) {
            std::sort(nodes->begin(), nodes->end(), better_by_priority);
            return;
        }

        auto nth = nodes->begin() + static_cast<std::ptrdiff_t>(width);
        std::nth_element(nodes->begin(), nth, nodes->end(), better_by_priority);
        nodes->resize(width);
        std::sort(nodes->begin(), nodes->end(), better_by_priority);
    }

    static void push_node_with_optional_dedup(
        std::vector<BeamNode>* nodes,
        std::unordered_map<std::uint64_t, std::size_t>* dedup_index,
        bool dedup_enabled,
        BeamNode&& node,
        PerfMetrics* perf) {
        if (!nodes) {
            return;
        }

        ScopedTimer timer(perf && perf->enabled, perf ? &perf->dedup_ms : nullptr);

        if (!dedup_enabled || !dedup_index) {
            nodes->push_back(std::move(node));
            return;
        }

        auto [it, inserted] = dedup_index->emplace(node.state_hash, nodes->size());
        if (inserted) {
            nodes->push_back(std::move(node));
            return;
        }

        auto& incumbent = (*nodes)[it->second];
        if (better_by_priority(node, incumbent)) {
            incumbent = std::move(node);
        }
    }

    static bool transposition_allows(
        const BeamNode& node,
        const SearchConfig& config,
        SearchCache* cache) {
        if (!cache || !config.use_transposition_table) {
            return true;
        }
        const TTKey key{node.state_hash, node.depth_remaining};
        auto it = cache->transposition.find(key);
        if (it != cache->transposition.end() && it->second > node.priority_score + kTieEps) {
            return false;
        }
        if (it == cache->transposition.end() || node.priority_score > it->second) {
            cache->transposition[key] = node.priority_score;
        }
        return true;
    }

    // beam search
    static SearchResult choose_action_beam(
        const NodeState& root,
        const SearchConfig& config,
        const HeuristicWeights& weights,
        SearchCache* cache,
        SearchScratch* scratch,
        PerfMetrics* perf) {
        SearchResult result{};
        if (!cache || !scratch) {
            return result;
        }

        ScopedTimer total_timer(perf && perf->enabled, perf ? &perf->total_ms : nullptr);

        auto& env = scratch->env;
        bool has_root_actions = false;
        {
            ScopedTimer legal_timer(perf && perf->enabled, perf ? &perf->legal_generation_ms : nullptr);
            env.restore(root.snapshot);
            if (!env.state().game_over && env.state().active.piece != Piece::None &&
                !env.placement_options_view().empty()) {
                has_root_actions = true;
            }
            if (!has_root_actions && root.snapshot.state.hold_available) {
                env.restore(root.snapshot);
                const auto hold_result = env.step(Action::Hold);
                if (hold_result.hold_used && !env.state().game_over &&
                    !env.placement_options_view().empty()) {
                    has_root_actions = true;
                }
            }
        }
        if (!has_root_actions) {
            {
                ScopedTimer feature_timer(perf && perf->enabled, perf ? &perf->feature_ms : nullptr);
                const auto f = extract_board_features(root);
                ScopedTimer eval_timer(perf && perf->enabled, perf ? &perf->eval_ms : nullptr);
                result.value = evaluate_board_features(f, weights);
            }
            result.has_action = false;
            return result;
        }

        result.candidates.clear();

        auto& beam = scratch->beam_current;
        beam.clear();

        auto& root_dedup_index = scratch->root_dedup_index;
        root_dedup_index.clear();
        if (config.deduplicate_successors) {
            root_dedup_index.reserve(static_cast<std::size_t>(std::max(1, config.beam_width)) * 8);
        }

        std::uint32_t root_generated = 0;
        std::size_t root_action_index_counter = 0;

        auto evaluate_and_push_root = [&](const EnvSnapshot& context_snapshot,
                                          bool use_hold,
                                          const std::vector<PlacementOption>& options) {
            for (std::size_t placement_index = 0; placement_index < options.size(); ++placement_index) {
                const std::size_t action_index = root_action_index_counter++;
                CandidateInfo candidate{};
                candidate.action_index = action_index;
                candidate.use_hold = use_hold;
                candidate.placement_index = placement_index;
                candidate.immediate_reward = 0.0;
                candidate.child_value = -std::numeric_limits<double>::infinity();
                candidate.total_value = -std::numeric_limits<double>::infinity();
                result.candidates.push_back(candidate);
                const std::size_t root_idx = result.candidates.size() - 1;

                if (config.max_nodes > 0 && result.nodes_expanded >= config.max_nodes) {
                    cache->node_limit_hit = true;
                    break;
                }

                Transition transition{};
                if (!apply_placement_option(
                        context_snapshot, options[placement_index], &transition, scratch, perf)) {
                    continue;
                }
                ++root_generated;
                ++result.nodes_expanded;

                BoardFeatures features{};
                {
                    ScopedTimer feature_timer(perf && perf->enabled, perf ? &perf->feature_ms : nullptr);
                    features = extract_board_features(transition.child);
                }

                double heuristic_full = 0.0;
                double heuristic_priority = 0.0;
                {
                    ScopedTimer eval_timer(perf && perf->enabled, perf ? &perf->eval_ms : nullptr);
                    heuristic_full = evaluate_board_features(features, weights);
                    heuristic_priority = config.enable_cheap_pruning
                        ? evaluate_board_features_cheap(features, weights)
                        : heuristic_full;
                }

                const double immediate = compute_immediate_reward(transition.step_result, weights);

                BeamNode node{};
                node.state = std::move(transition.child);
                node.state_hash = hash_snapshot(node.state.snapshot);
                node.root_action_index = root_idx;
                node.cumulative_reward = immediate;
                node.heuristic_value = heuristic_full;
                node.depth_remaining = std::max(0, config.depth - 1);
                node.action_index = action_index;
                node.placement_index = placement_index;
                node.priority_score =
                    compute_priority(node.cumulative_reward, heuristic_priority, config, weights);
                node.final_value =
                    compute_final_value(node.cumulative_reward, node.heuristic_value, config, weights);

                auto& candidate_row = result.candidates[root_idx];
                candidate_row.immediate_reward = immediate;
                candidate_row.child_value = heuristic_full;
                candidate_row.total_value = node.final_value;

                if (!transposition_allows(node, config, cache)) {
                    continue;
                }

                push_node_with_optional_dedup(
                    &beam,
                    &root_dedup_index,
                    config.deduplicate_successors,
                    std::move(node),
                    perf);
            }
        };

        {
            ScopedTimer legal_timer(perf && perf->enabled, perf ? &perf->legal_generation_ms : nullptr);
            env.restore(root.snapshot);
            if (!env.state().game_over && env.state().active.piece != Piece::None) {
                const auto& options = env.placement_options_view();
                if (!options.empty()) {
                    evaluate_and_push_root(root.snapshot, false, options);
                }
            }
        }

        if (!cache->node_limit_hit && root.snapshot.state.hold_available) {
            EnvSnapshot hold_snapshot{};
            const std::vector<PlacementOption>* hold_options = nullptr;
            {
                ScopedTimer legal_timer(perf && perf->enabled, perf ? &perf->legal_generation_ms : nullptr);
                env.restore(root.snapshot);
                const auto hold_result = env.step(Action::Hold);
                if (hold_result.hold_used && !env.state().game_over) {
                    hold_snapshot = env.snapshot();
                    hold_options = &env.placement_options_view();
                }
            }
            if (hold_options && !hold_options->empty()) {
                evaluate_and_push_root(hold_snapshot, true, *hold_options);
            }
        }

        if (perf) {
            perf->add_children_generated(0, root_generated);
            perf->set_children_after_dedup(0, static_cast<std::uint32_t>(beam.size()));
        }

        if (!beam.empty()) {
            prune_to_beam_width(&beam, config.beam_width, perf);
        }
        if (perf) {
            perf->set_children_after_prune(0, static_cast<std::uint32_t>(beam.size()));
        }

        for (int layer = 1; layer < config.depth && !beam.empty() && !cache->node_limit_hit; ++layer) {
            auto& next_beam = scratch->beam_next;
            next_beam.clear();
            next_beam.reserve(
                beam.size() * static_cast<std::size_t>(std::max(1, config.beam_width)));

            auto& dedup_index = scratch->dedup_index;
            dedup_index.clear();
            if (config.deduplicate_successors) {
                dedup_index.reserve(next_beam.capacity() * 2 + 1);
            }

            std::uint32_t generated = 0;
            for (const auto& node : beam) {
                if (cache->node_limit_hit) {
                    break;
                }

                if (node.depth_remaining <= 0 || node.state.snapshot.state.game_over ||
                    node.state.snapshot.state.top_out) {
                    push_node_with_optional_dedup(
                        &next_beam,
                        &dedup_index,
                        config.deduplicate_successors,
                        BeamNode{node},
                        perf);
                    continue;
                }

                bool any_child_generated = false;
                std::size_t action_index_offset = 0;

                auto expand_context = [&](const EnvSnapshot& context_snapshot,
                                          const std::vector<PlacementOption>& options) {
                    for (std::size_t placement_index = 0; placement_index < options.size(); ++placement_index) {
                        if (config.max_nodes > 0 && result.nodes_expanded >= config.max_nodes) {
                            cache->node_limit_hit = true;
                            break;
                        }

                        const std::size_t action_index = action_index_offset + placement_index;
                        Transition transition{};
                        if (!apply_placement_option(
                                context_snapshot, options[placement_index], &transition, scratch, perf)) {
                            continue;
                        }
                        any_child_generated = true;
                        ++generated;
                        ++result.nodes_expanded;

                        BoardFeatures features{};
                        {
                            ScopedTimer feature_timer(
                                perf && perf->enabled, perf ? &perf->feature_ms : nullptr);
                            features = extract_board_features(transition.child);
                        }

                        double heuristic_full = 0.0;
                        double heuristic_priority = 0.0;
                        {
                            ScopedTimer eval_timer(perf && perf->enabled, perf ? &perf->eval_ms : nullptr);
                            heuristic_full = evaluate_board_features(features, weights);
                            heuristic_priority = config.enable_cheap_pruning
                                ? evaluate_board_features_cheap(features, weights)
                                : heuristic_full;
                        }

                        const double immediate = compute_immediate_reward(transition.step_result, weights);

                        BeamNode child{};
                        child.state = std::move(transition.child);
                        child.state_hash = hash_snapshot(child.state.snapshot);
                        child.root_action_index = node.root_action_index;
                        child.cumulative_reward = node.cumulative_reward + immediate;
                        child.heuristic_value = heuristic_full;
                        child.depth_remaining = std::max(0, node.depth_remaining - 1);
                        child.action_index = action_index;
                        child.placement_index = placement_index;
                        child.priority_score =
                            compute_priority(child.cumulative_reward, heuristic_priority, config, weights);
                        child.final_value =
                            compute_final_value(child.cumulative_reward, child.heuristic_value, config, weights);

                        if (!transposition_allows(child, config, cache)) {
                            continue;
                        }

                        push_node_with_optional_dedup(
                            &next_beam,
                            &dedup_index,
                            config.deduplicate_successors,
                            std::move(child),
                            perf);
                    }
                };

                {
                    ScopedTimer legal_timer(
                        perf && perf->enabled, perf ? &perf->legal_generation_ms : nullptr);
                    env.restore(node.state.snapshot);
                    if (!env.state().game_over && env.state().active.piece != Piece::None) {
                        const auto& options = env.placement_options_view();
                        expand_context(node.state.snapshot, options);
                        action_index_offset += options.size();
                    }
                }

                if (!cache->node_limit_hit && node.state.snapshot.state.hold_available) {
                    EnvSnapshot hold_snapshot{};
                    const std::vector<PlacementOption>* hold_options = nullptr;
                    {
                        ScopedTimer legal_timer(
                            perf && perf->enabled, perf ? &perf->legal_generation_ms : nullptr);
                        env.restore(node.state.snapshot);
                        const auto hold_result = env.step(Action::Hold);
                        if (hold_result.hold_used && !env.state().game_over) {
                            hold_snapshot = env.snapshot();
                            hold_options = &env.placement_options_view();
                        }
                    }
                    if (hold_options && !hold_options->empty()) {
                        expand_context(hold_snapshot, *hold_options);
                        action_index_offset += hold_options->size();
                    }
                }

                if (!any_child_generated) {
                    push_node_with_optional_dedup(
                        &next_beam,
                        &dedup_index,
                        config.deduplicate_successors,
                        BeamNode{node},
                        perf);
                }
            }

            if (perf) {
                perf->add_children_generated(layer, generated);
                perf->set_children_after_dedup(layer, static_cast<std::uint32_t>(next_beam.size()));
            }

            if (next_beam.empty()) {
                break;
            }

            prune_to_beam_width(&next_beam, config.beam_width, perf);
            if (perf) {
                perf->set_children_after_prune(layer, static_cast<std::uint32_t>(next_beam.size()));
            }
            beam.swap(next_beam);
        }

        for (const auto& node : beam) {
            if (node.root_action_index >= result.candidates.size()) {
                continue;
            }
            auto& candidate = result.candidates[node.root_action_index];
            if (!std::isfinite(candidate.total_value) || node.final_value > candidate.total_value + kTieEps) {
                candidate.total_value = node.final_value;
            } else if (std::abs(node.final_value - candidate.total_value) <= kTieEps) {
                if (node.action_index < candidate.action_index ||
                    (node.action_index == candidate.action_index &&
                     node.placement_index < candidate.placement_index)) {
                    candidate.total_value = node.final_value;
                }
            }
        }

        for (std::size_t i = 0; i < result.candidates.size(); ++i) {
            const auto& candidate = result.candidates[i];
            if (!std::isfinite(candidate.total_value)) {
                continue;
            }
            if (!result.has_action || candidate.total_value > result.value + kTieEps) {
                result.value = candidate.total_value;
                result.best_action_index = i;
                result.has_action = true;
            } else if (std::abs(candidate.total_value - result.value) <= kTieEps) {
                const auto& best = result.candidates[result.best_action_index];
                if (candidate.action_index < best.action_index ||
                    (candidate.action_index == best.action_index &&
                     candidate.placement_index < best.placement_index)) {
                    result.best_action_index = i;
                }
            }
        }

        if (!result.has_action) {
            BoardFeatures f{};
            {
                ScopedTimer feature_timer(perf && perf->enabled, perf ? &perf->feature_ms : nullptr);
                f = extract_board_features(root);
            }
            {
                ScopedTimer eval_timer(perf && perf->enabled, perf ? &perf->eval_ms : nullptr);
                result.value = evaluate_board_features(f, weights);
            }
        }

        if (perf) {
            perf->nodes_expanded = result.nodes_expanded;
            perf->finalize_totals();
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
    SearchScratch search_scratch_{};
    PerfMetrics last_perf_{};
};

// bot entrypoint
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
    int beam_width,
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
        beam_width,
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

}  // namespace tetris_v2::beam
