#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "tetris_v2/types.hpp"

namespace tetris_v2 {

class SevenBagRandomizer {
public:
    explicit SevenBagRandomizer(std::uint32_t seed = 0);

    void reseed(std::uint32_t seed);
    Piece next_piece();
    const std::vector<Piece>& bag_order() const { return bag_; }
    std::size_t bag_index() const { return index_; }

private:
    void refill_bag();

    std::mt19937 rng_;
    std::vector<Piece> bag_;
    std::size_t index_{0};
};

}  // namespace tetris_v2
