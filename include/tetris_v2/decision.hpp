#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "tetris_v2/cc_env.hpp"

namespace tetris_v2 {

inline constexpr std::size_t kDecisionXCount = Board::kWidth;
inline constexpr std::size_t kDecisionYCount = Board::kRows;
inline constexpr std::size_t kDecisionRotationCount = 4;
inline constexpr std::size_t kDecisionPoseCount =
    kDecisionXCount * kDecisionYCount * kDecisionRotationCount;
inline constexpr std::size_t kDecisionActionDim = 2 * kDecisionPoseCount;

struct StableDecision {
    bool use_hold{false};
    std::size_t placement_index{0};
    std::size_t action{0};
    ActivePiece placement{};
};

std::optional<std::size_t> stable_decision_action(
    bool use_hold, const ActivePiece& placement);
std::vector<StableDecision> enumerate_stable_decisions(const cc::Env& env);
std::optional<StableDecision> stable_decision_at_action(
    const cc::Env& env, std::size_t action);
std::optional<std::size_t> stable_decision_for_choice(
    const cc::Env& env, bool use_hold, std::size_t placement_index);
bool apply_stable_decision(
    cc::Env& env,
    std::size_t action,
    StepResult* result_out = nullptr,
    int* used_hold_out = nullptr,
    std::size_t* placement_index_out = nullptr);

}  // namespace tetris_v2
