#pragma once

#include <array>
#include <cstdint>

#include "tetris_v2/cc2_data.hpp"

namespace tetris_v2::cc2 {

struct FreestyleWeights {
    float cell_coveredness{-0.2f};
    std::uint32_t max_cell_covered_height{6};
    float holes{-1.5f};
    float row_transitions{-0.2f};
    float height{-0.4f};
    float height_upper_half{-1.5f};
    float height_upper_quarter{-5.0f};
    float tetris_well_depth{0.3f};
    std::array<float, 4> tslot{0.1f, 1.5f, 2.0f, 4.0f};

    float has_back_to_back{0.5f};
    float wasted_t{-1.5f};
    float softdrop{-0.2f};

    std::array<float, 5> normal_clears{0.0f, -2.0f, -1.5f, -1.0f, 3.5f};
    std::array<float, 3> mini_spin_clears{0.0f, -1.5f, -1.0f};
    std::array<float, 4> spin_clears{0.0f, 1.0f, 4.0f, 6.0f};
    float back_to_back_clear{1.0f};
    float combo_attack{1.5f};
    float perfect_clear{15.0f};
    bool perfect_clear_override{true};
};

struct FreestyleEvalResult {
    float eval{0.0f};
    float reward{0.0f};
    float total{0.0f};
};

FreestyleEvalResult evaluate_freestyle(
    const FreestyleWeights& weights,
    GameState state,
    const PlacementInfo& info,
    std::uint32_t softdrop);

}  // namespace tetris_v2::cc2

