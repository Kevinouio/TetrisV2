#include "tetris_v2/decision.hpp"

#include <algorithm>

namespace tetris_v2 {

std::optional<std::size_t> stable_decision_action(
    bool use_hold, const ActivePiece& placement) {
    const int rotation = static_cast<int>(placement.rotation);
    if (placement.x < 0 || placement.x >= static_cast<int>(kDecisionXCount) ||
        placement.y < 0 || placement.y >= static_cast<int>(kDecisionYCount) ||
        rotation < 0 || rotation >= static_cast<int>(kDecisionRotationCount)) {
        return std::nullopt;
    }
    const auto pose =
        (static_cast<std::size_t>(rotation) * kDecisionYCount +
         static_cast<std::size_t>(placement.y)) *
            kDecisionXCount +
        static_cast<std::size_t>(placement.x);
    return (use_hold ? kDecisionPoseCount : 0) + pose;
}

std::vector<StableDecision> enumerate_stable_decisions(const cc::Env& env) {
    std::vector<StableDecision> out;
    auto append = [&](const cc::Env& source, bool use_hold) {
        const auto options = source.enumerate_active_piece_placements();
        for (std::size_t i = 0; i < options.size(); ++i) {
            auto action = stable_decision_action(use_hold, options[i].placement);
            if (action.has_value()) {
                out.push_back(StableDecision{use_hold, i, *action, options[i].placement});
            }
        }
    };

    append(env, false);
    if (env.state().hold_available) {
        cc::Env held(env.config());
        held.restore(env.snapshot());
        const auto result = held.step(Action::Hold);
        if (result.hold_used && !result.game_over) {
            append(held, true);
        }
    }
    return out;
}

std::optional<StableDecision> stable_decision_at_action(
    const cc::Env& env, std::size_t action) {
    if (action >= kDecisionActionDim) {
        return std::nullopt;
    }
    const auto options = enumerate_stable_decisions(env);
    const auto it = std::find_if(
        options.begin(), options.end(),
        [&](const StableDecision& option) { return option.action == action; });
    return it == options.end() ? std::nullopt : std::optional<StableDecision>(*it);
}

std::optional<std::size_t> stable_decision_for_choice(
    const cc::Env& env, bool use_hold, std::size_t placement_index) {
    const auto options = enumerate_stable_decisions(env);
    const auto it = std::find_if(
        options.begin(), options.end(), [&](const StableDecision& option) {
            return option.use_hold == use_hold && option.placement_index == placement_index;
        });
    return it == options.end() ? std::nullopt : std::optional<std::size_t>(it->action);
}

bool apply_stable_decision(
    cc::Env& env,
    std::size_t action,
    StepResult* result_out,
    int* used_hold_out,
    std::size_t* placement_index_out) {
    const auto decision = stable_decision_at_action(env, action);
    if (!decision.has_value()) {
        return false;
    }
    if (decision->use_hold) {
        const auto held = env.step(Action::Hold);
        if (!held.hold_used || held.game_over) {
            return false;
        }
    }
    const auto result = env.apply_placement_index(decision->placement_index);
    if (!result.action_succeeded) {
        return false;
    }
    if (result_out) {
        *result_out = result;
    }
    if (used_hold_out) {
        *used_hold_out = decision->use_hold ? 1 : 0;
    }
    if (placement_index_out) {
        *placement_index_out = decision->placement_index;
    }
    return true;
}

}  // namespace tetris_v2
