#pragma once

#include <cstddef>
#include <vector>

#include "tetris_v2/env.hpp"

namespace tetris_v2 {

struct ObservationLayout {
    std::size_t board_offset{0};
    std::size_t active_offset{0};
    std::size_t hold_offset{0};
    std::size_t queue_offset{0};
    std::size_t meta_offset{0};
    std::size_t total_size{0};
    int rows_encoded{0};
};

ObservationLayout build_observation_layout(int queue_size, bool include_hidden_rows);
std::size_t observation_size(int queue_size, bool include_hidden_rows);
std::vector<float> encode_observation(
    const EnvState& state, int queue_size, bool include_hidden_rows);

}  // namespace tetris_v2
