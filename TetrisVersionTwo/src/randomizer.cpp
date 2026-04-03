#include "tetris_v2/randomizer.hpp"

#include <algorithm>
#include <random>

namespace tetris_v2 {

SevenBagRandomizer::SevenBagRandomizer(std::uint32_t seed) : rng_(seed) { refill_bag(); }

void SevenBagRandomizer::reseed(std::uint32_t seed) {
    rng_.seed(seed);
    refill_bag();
}

Piece SevenBagRandomizer::next_piece() {
    if (index_ >= bag_.size()) {
        refill_bag();
    }
    return bag_[index_++];
}

void SevenBagRandomizer::refill_bag() {
    bag_.assign(kPlayablePieces.begin(), kPlayablePieces.end());
    std::shuffle(bag_.begin(), bag_.end(), rng_);
    index_ = 0;
}

}  // namespace tetris_v2
