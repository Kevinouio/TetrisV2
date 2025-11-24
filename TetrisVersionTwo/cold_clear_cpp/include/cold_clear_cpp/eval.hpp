#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "cold_clear_cpp/state.hpp"

namespace cold_clear_cpp {

struct Weights {
    // Survival-first tuning: larger penalties for mess/height, modest line-clear rewards,
    // and toned-down spin/B2B incentives.
    float cell_coveredness{-0.7f};
    std::uint32_t max_cell_covered_height{6};
    float holes{-3.5f};
    float row_transitions{-0.6f};
    float height{-1.0f};
    float height_upper_half{-3.0f};
    float height_upper_quarter{-7.0f};
    float tetris_well_depth{0.15f};
    std::array<float, 4> tslot{{0.0f, 0.2f, 0.4f, 0.6f}};

    float has_back_to_back{0.2f};
    float wasted_t{-0.8f};
    float softdrop{-0.1f};

    std::array<float, 5> normal_clears{{0.2f, 0.6f, 0.9f, 1.2f, 1.8f}};
    std::array<float, 3> mini_spin_clears{{0.0f, -1.5f, -1.0f}};
    std::array<float, 4> spin_clears{{0.0f, 0.2f, 0.8f, 1.5f}};
    float back_to_back_clear{0.2f};
    float combo_attack{0.3f};
    float perfect_clear{8.0f};
    bool perfect_clear_override{false};
};

struct Eval {
    double value{0.0};

    struct Reward {
        double value{0.0};
    };

    using RewardType = Reward;

    Eval operator+(Reward rhs) const { return Eval{value + rhs.value}; }
    bool operator<(const Eval& rhs) const { return value < rhs.value; }
    bool operator>(const Eval& rhs) const { return value > rhs.value; }
    bool operator==(const Eval& rhs) const { return value == rhs.value; }
    bool operator!=(const Eval& rhs) const { return value != rhs.value; }

    static Eval average(const std::vector<std::optional<Eval>>& values);
};

std::pair<Eval, Eval::Reward> evaluate(
    const Weights& weights, GameState state, const PlacementInfo& info, std::uint32_t softdrop);

Weights default_weights();

}  // namespace cold_clear_cpp
