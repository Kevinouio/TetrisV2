#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <vector>

#include "tetris_v2/cc2_dag.hpp"
#include "tetris_v2/cc2_data.hpp"
#include "tetris_v2/cc2_eval.hpp"
#include "tetris_v2/cc2_sync.hpp"
#include "tetris_v2/env.hpp"

namespace tetris_v2 {

struct ColdClearBotChoice {
    bool use_hold{false};
    std::size_t placement_index{0};
    float score{0.0f};
};

struct ColdClearBotThinkStats {
    std::uint64_t nodes{0};
    std::uint64_t selections{0};
    std::uint64_t expansions{0};
    double think_ms{0.0};
    double nps{0.0};
    int budget_miss{0};
};

class ColdClearBot {
public:
    ColdClearBot();
    ~ColdClearBot();

    bool sync_from_env(const ModernTetrisEnv& env);
    bool choose(
        int think_ms,
        ColdClearBotChoice* choice_out = nullptr,
        ColdClearBotThinkStats* stats_out = nullptr);
    bool apply_last_choice(
        ModernTetrisEnv& env,
        StepResult* result_out = nullptr,
        int* used_hold_out = nullptr,
        std::size_t* placement_index_out = nullptr);
    bool choose_and_apply(
        ModernTetrisEnv& env,
        int think_ms,
        StepResult* result_out = nullptr,
        ColdClearBotChoice* choice_out = nullptr,
        ColdClearBotThinkStats* stats_out = nullptr,
        int* used_hold_out = nullptr,
        std::size_t* placement_index_out = nullptr);

private:
    struct Snapshot {
        EnvSnapshot env_snapshot{};
        bool valid{false};
    };

    struct Candidate {
        bool use_hold{false};
        std::size_t placement_index{0};
        cc2::Placement placement{};
        std::uint32_t softdrop{0};
        Piece placed_piece{Piece::None};
    };

    bool start_search_locked(const Snapshot& snapshot);
    bool map_suggestion_to_choice(
        const Snapshot& snapshot,
        const cc2::DagSuggestion& suggestion,
        ColdClearBotChoice* choice_out) const;

    static std::deque<Piece> build_known_queue_for_dag(const EnvState& state);
    static cc2::GameState build_root_state_for_dag(const Snapshot& snapshot);

    static cc2::PieceMask bag_mask_from_snapshot(const EnvSnapshot& snapshot);
    static cc2::PieceLocation to_cc2_location(const ActivePiece& piece);
    static Piece deque_front_or_none(const std::deque<Piece>& queue);
    static std::uint8_t combo_to_cc2(int combo);

    std::vector<Candidate> enumerate_candidates(const Snapshot& snapshot) const;

    cc2::FreestyleWeights weights_{};
    double exploitation_{0.6931471805599453};  // ln(2)

    mutable std::mutex mutex_;
    Snapshot synced_snapshot_{};
    std::optional<ColdClearBotChoice> last_choice_{};
    cc2::Synchronizer synchronizer_{};
};

}  // namespace tetris_v2
