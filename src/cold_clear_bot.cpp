#include "tetris_v2/cold_clear_bot.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>

namespace tetris_v2 {

namespace {

void maybe_set_int(int* dst, int value) {
    if (dst) {
        *dst = value;
    }
}

cc2::PieceLocation canonical_location(const cc2::PieceLocation& location) {
    return location.canonical_form();
}

}  // namespace

ColdClearBot::ColdClearBot() {
    synchronizer_.set_weights(weights_);
    synchronizer_.set_exploitation(exploitation_);
}

ColdClearBot::~ColdClearBot() = default;

bool ColdClearBot::sync_from_env(const ModernTetrisEnv& env) {
    std::lock_guard<std::mutex> lock(mutex_);
    synced_snapshot_.config = env.config();
    synced_snapshot_.env_snapshot = env.snapshot();
    synced_snapshot_.valid = true;
    last_choice_.reset();
    return true;
}

bool ColdClearBot::start_search_locked(const Snapshot& snapshot, bool background) {
    if (!snapshot.valid) {
        return false;
    }
    const auto& state = snapshot.env_snapshot.state;
    if (state.game_over || state.active.piece == Piece::None) {
        return false;
    }

    const cc2::GameState root = build_root_state_for_dag(snapshot);
    const auto queue = build_known_queue_for_dag(state);
    synchronizer_.set_weights(weights_);
    synchronizer_.set_exploitation(exploitation_);
    synchronizer_.start(root, queue, true, background);
    return true;
}

bool ColdClearBot::map_suggestion_to_choice(
    const Snapshot& snapshot,
    const cc2::DagSuggestion& suggestion,
    ColdClearBotChoice* choice_out) const {
    if (!choice_out || !snapshot.valid || !suggestion.valid) {
        return false;
    }

    const auto candidates = enumerate_candidates(snapshot);
    if (candidates.empty()) {
        return false;
    }

    auto find_match = [&]() -> const Candidate* {
        for (const auto& c : candidates) {
            if (c.use_hold != suggestion.use_hold) {
                continue;
            }
            if (!(canonical_location(c.placement.location) ==
                  canonical_location(suggestion.placement.location))) {
                continue;
            }
            if (c.placement.spin != suggestion.placement.spin) {
                continue;
            }
            return &c;
        }
        return nullptr;
    };

    const Candidate* selected = find_match();
    if (!selected) {
        return false;
    }

    choice_out->use_hold = selected->use_hold;
    choice_out->placement_index = selected->placement_index;
    choice_out->score = std::isfinite(suggestion.score) ? suggestion.score : -1000.0f;
    return true;
}

bool ColdClearBot::choose(
    int think_ms,
    ColdClearBotChoice* choice_out,
    ColdClearBotThinkStats* stats_out) {
    Snapshot snapshot{};
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!synced_snapshot_.valid) {
            return false;
        }
        snapshot = synced_snapshot_;
    }

    const bool deterministic_effort = think_ms == 0;
    const int normalized_think_ms = std::max(1, think_ms);
    if (!start_search_locked(snapshot, !deterministic_effort)) {
        return false;
    }

    const auto t0 = std::chrono::steady_clock::now();
    cc2::SyncSnapshot sync{};
    if (deterministic_effort) {
        sync = synchronizer_.run_work_units(kDeterministicWorkUnits);
    } else {
        const auto deadline = t0 + std::chrono::milliseconds(normalized_think_ms);
        synchronizer_.wait_until(deadline);
        sync = synchronizer_.snapshot();
    }
    synchronizer_.stop();
    const auto t1 = std::chrono::steady_clock::now();

    ColdClearBotChoice choice{};
    if (!sync.suggestion.valid ||
        !map_suggestion_to_choice(snapshot, sync.suggestion, &choice)) {
        const auto fallback = enumerate_candidates(snapshot);
        if (fallback.empty()) {
            return false;
        }
        choice.use_hold = fallback.front().use_hold;
        choice.placement_index = fallback.front().placement_index;
        choice.score = -1000.0f;
    }
    if (!std::isfinite(choice.score)) {
        return false;
    }

    const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    ColdClearBotThinkStats stats{};
    stats.nodes = sync.stats.nodes;
    stats.selections = sync.stats.selections;
    stats.expansions = sync.stats.expansions;
    stats.think_ms = elapsed_ms;
    stats.nps = sync.nps;
    stats.budget_miss =
        deterministic_effort
            ? 0
            : (elapsed_ms > static_cast<double>(normalized_think_ms) + 1.0 ? 1 : 0);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        last_choice_ = choice;
    }

    if (choice_out) {
        *choice_out = choice;
    }
    if (stats_out) {
        *stats_out = stats;
    }
    return true;
}

bool ColdClearBot::apply_last_choice(
    ModernTetrisEnv& env,
    StepResult* result_out,
    int* used_hold_out,
    std::size_t* placement_index_out) {
    std::optional<ColdClearBotChoice> choice{};
    {
        std::lock_guard<std::mutex> lock(mutex_);
        choice = last_choice_;
    }

    if (!choice.has_value()) {
        return false;
    }

    maybe_set_int(used_hold_out, 0);
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
        maybe_set_int(used_hold_out, 1);
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

bool ColdClearBot::choose_and_apply(
    ModernTetrisEnv& env,
    int think_ms,
    StepResult* result_out,
    ColdClearBotChoice* choice_out,
    ColdClearBotThinkStats* stats_out,
    int* used_hold_out,
    std::size_t* placement_index_out) {
    if (!sync_from_env(env)) {
        return false;
    }

    ColdClearBotChoice choice{};
    ColdClearBotThinkStats stats{};
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

cc2::GameState ColdClearBot::build_root_state_for_dag(const Snapshot& snapshot) {
    cc2::GameState root{};
    const auto& env_state = snapshot.env_snapshot.state;
    root.board = cc2::Board::from_env_board(env_state.board);
    root.bag_mask = bag_mask_from_snapshot(snapshot.env_snapshot);
    root.reserve = env_state.hold.value_or(Piece::None);
    root.hold_available = env_state.hold_available;
    root.back_to_back = env_state.back_to_back;
    root.combo = combo_to_cc2(env_state.combo);
    return root;
}

std::deque<Piece> ColdClearBot::build_known_queue_for_dag(const EnvState& state) {
    std::deque<Piece> queue{};
    if (state.active.piece != Piece::None) {
        queue.push_back(state.active.piece);
    }
    for (const auto piece : state.queue) {
        queue.push_back(piece);
    }
    return queue;
}

cc2::PieceMask ColdClearBot::bag_mask_from_snapshot(const EnvSnapshot& snapshot) {
    cc2::PieceMask bag_mask = 0;
    const auto& bag = snapshot.randomizer.bag_order();
    std::size_t idx = snapshot.randomizer.bag_index();
    if (idx > bag.size()) {
        idx = bag.size();
    }

    for (std::size_t i = idx; i < bag.size(); ++i) {
        if (bag[i] != Piece::None) {
            bag_mask = static_cast<cc2::PieceMask>(bag_mask | cc2::piece_bit(bag[i]));
        }
    }
    if (bag_mask == 0) {
        bag_mask = cc2::kAllPiecesMask;
    }

    for (auto it = snapshot.state.queue.rbegin(); it != snapshot.state.queue.rend(); ++it) {
        if (bag_mask == cc2::kAllPiecesMask) {
            bag_mask = 0;
        }
        if (*it != Piece::None) {
            bag_mask = static_cast<cc2::PieceMask>(bag_mask | cc2::piece_bit(*it));
        }
    }

    return bag_mask;
}

cc2::PieceLocation ColdClearBot::to_cc2_location(const ActivePiece& piece) {
    return cc2::PieceLocation{
        piece.piece,
        cc2::rotation_from_env(piece.rotation),
        static_cast<std::int8_t>(piece.x),
        static_cast<std::int8_t>(piece.y)};
}

std::uint8_t ColdClearBot::combo_to_cc2(int combo) {
    if (combo < 0) {
        return 0;
    }
    return static_cast<std::uint8_t>(std::min(combo + 1, 255));
}

std::vector<ColdClearBot::Candidate> ColdClearBot::enumerate_candidates(const Snapshot& snapshot) const {
    std::vector<Candidate> out;
    if (!snapshot.valid) {
        return out;
    }

    ModernTetrisEnv root(snapshot.config);
    root.restore(snapshot.env_snapshot);
    if (root.state().game_over || root.state().active.piece == Piece::None) {
        return out;
    }

    auto append_candidates = [&](const ModernTetrisEnv& env, bool use_hold) {
        const Piece piece = env.state().active.piece;
        if (piece == Piece::None) {
            return;
        }

        const auto options = env.enumerate_active_piece_placements();
        if (options.empty()) {
            return;
        }

        for (std::size_t i = 0; i < options.size(); ++i) {
            cc2::Placement placement{
                to_cc2_location(options[i].placement),
                cc2::spin_from_env(options[i].spin_type)};
            out.push_back(Candidate{use_hold, i, placement});
        }
    };

    append_candidates(root, false);

    if (root.state().hold_available) {
        ModernTetrisEnv hold_env(root.config());
        hold_env.restore(snapshot.env_snapshot);
        auto hold_result = hold_env.step(Action::Hold);
        if (hold_result.hold_used && !hold_env.state().game_over) {
            append_candidates(hold_env, true);
        }
    }

    return out;
}

}  // namespace tetris_v2
