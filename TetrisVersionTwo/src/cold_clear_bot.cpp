#include "tetris_v2/cold_clear_bot.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <unordered_map>

#include "tetris_v2/cc2_movegen.hpp"

namespace tetris_v2 {

namespace {

void maybe_set_int(int* dst, int value) {
    if (dst) {
        *dst = value;
    }
}

struct LocationKey {
    Piece piece{Piece::None};
    cc2::Rotation rotation{cc2::Rotation::North};
    std::int8_t x{0};
    std::int8_t y{0};

    bool operator==(const LocationKey& rhs) const {
        return piece == rhs.piece && rotation == rhs.rotation && x == rhs.x && y == rhs.y;
    }
};

struct LocationKeyHash {
    std::size_t operator()(const LocationKey& k) const noexcept {
        std::size_t h = 1469598103934665603ull;
        auto mix = [&h](std::size_t v) {
            h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        };
        mix(static_cast<std::size_t>(k.piece));
        mix(static_cast<std::size_t>(k.rotation));
        mix(static_cast<std::size_t>(static_cast<int>(k.x) + 128));
        mix(static_cast<std::size_t>(static_cast<int>(k.y) + 128));
        return h;
    }
};

LocationKey canonical_key(const cc2::PieceLocation& location) {
    const auto c = location.canonical_form();
    return LocationKey{c.piece, c.rotation, c.x, c.y};
}

}  // namespace

ColdClearBot::ColdClearBot() {
    synchronizer_.set_weights(weights_);
    synchronizer_.set_exploitation(exploitation_);
}

ColdClearBot::~ColdClearBot() = default;

bool ColdClearBot::sync_from_env(const ModernTetrisEnv& env) {
    std::lock_guard<std::mutex> lock(mutex_);
    synced_snapshot_.env_snapshot = env.snapshot();
    synced_snapshot_.valid = true;
    last_choice_.reset();
    return true;
}

bool ColdClearBot::start_search_locked(const Snapshot& snapshot) {
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
    synchronizer_.start(root, queue, true);
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

    auto find_match = [&](bool require_hold, bool require_spin) -> const Candidate* {
        for (const auto& c : candidates) {
            if (require_hold && c.use_hold != suggestion.use_hold) {
                continue;
            }
            if (!(canonical_key(c.placement.location) == canonical_key(suggestion.placement.location))) {
                continue;
            }
            if (require_spin && c.placement.spin != suggestion.placement.spin) {
                continue;
            }
            return &c;
        }
        return nullptr;
    };

    const Candidate* selected = nullptr;
    selected = find_match(true, true);
    if (!selected) {
        selected = find_match(true, false);
    }
    if (!selected) {
        selected = find_match(false, true);
    }
    if (!selected) {
        selected = find_match(false, false);
    }

    if (!selected) {
        selected = &candidates.front();
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

    const int normalized_think_ms = std::max(1, think_ms);
    if (!start_search_locked(snapshot)) {
        return false;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const auto deadline = t0 + std::chrono::milliseconds(normalized_think_ms);
    synchronizer_.wait_until(deadline);
    auto sync = synchronizer_.snapshot();
    synchronizer_.stop();
    const auto t1 = std::chrono::steady_clock::now();

    ColdClearBotChoice choice{};
    if (sync.suggestion.valid) {
        if (!map_suggestion_to_choice(snapshot, sync.suggestion, &choice)) {
            return false;
        }
    } else {
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
    stats.budget_miss = elapsed_ms > static_cast<double>(normalized_think_ms) + 1.0 ? 1 : 0;

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
    root.reserve = env_state.hold.value_or(deque_front_or_none(env_state.queue));
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

Piece ColdClearBot::deque_front_or_none(const std::deque<Piece>& queue) {
    if (queue.empty()) {
        return Piece::None;
    }
    return queue.front();
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

    ModernTetrisEnv root;
    root.restore(snapshot.env_snapshot);
    if (root.state().game_over || root.state().active.piece == Piece::None) {
        return out;
    }

    const cc2::Board board = cc2::Board::from_env_board(root.state().board);

    auto append_candidates = [&](const ModernTetrisEnv& env, bool use_hold) {
        const Piece piece = env.state().active.piece;
        if (piece == Piece::None) {
            return;
        }

        const auto options = env.enumerate_active_piece_placements();
        if (options.empty()) {
            return;
        }

        std::unordered_map<LocationKey, std::size_t, LocationKeyHash> option_index_by_canonical;
        option_index_by_canonical.reserve(options.size() * 2);
        for (std::size_t i = 0; i < options.size(); ++i) {
            option_index_by_canonical.emplace(canonical_key(to_cc2_location(options[i].placement)), i);
        }

        const auto moves = cc2::find_moves(board, piece);
        std::vector<bool> used_option(options.size(), false);
        std::size_t mapped = 0;
        for (const auto& move : moves) {
            auto it = option_index_by_canonical.find(canonical_key(move.first.location));
            if (it == option_index_by_canonical.end()) {
                continue;
            }
            if (used_option[it->second]) {
                continue;
            }
            used_option[it->second] = true;
            ++mapped;
            out.push_back(Candidate{use_hold, it->second, move.first, move.second, piece});
        }

        if (mapped > 0) {
            return;
        }

        for (std::size_t i = 0; i < options.size(); ++i) {
            cc2::Spin spin = cc2::Spin::None;
            if (options[i].spin_clear_candidate) {
                spin = options[i].last_rotate_kick_index_path == 4 ? cc2::Spin::Full : cc2::Spin::Mini;
            }
            cc2::Placement placement{to_cc2_location(options[i].placement), spin};
            std::uint32_t softdrop = 0;
            if (env.state().active.y > options[i].placement.y) {
                softdrop = static_cast<std::uint32_t>(env.state().active.y - options[i].placement.y);
            }
            out.push_back(Candidate{use_hold, i, placement, softdrop, piece});
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
