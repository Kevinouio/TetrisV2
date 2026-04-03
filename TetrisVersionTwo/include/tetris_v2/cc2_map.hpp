#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <unordered_map>

#include "tetris_v2/cc2_data.hpp"

namespace tetris_v2::cc2 {

inline std::size_t hash_combine(std::size_t seed, std::size_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
    return seed;
}

struct GameStateHash {
    std::size_t operator()(const GameState& state) const noexcept {
        std::size_t h = 1469598103934665603ull;
        for (auto col : state.board.cols) {
            h = hash_combine(h, std::hash<std::uint64_t>{}(col));
        }
        h = hash_combine(h, std::hash<std::uint8_t>{}(state.bag_mask));
        h = hash_combine(h, std::hash<int>{}(static_cast<int>(state.reserve)));
        h = hash_combine(h, std::hash<bool>{}(state.back_to_back));
        h = hash_combine(h, std::hash<std::uint8_t>{}(state.combo));
        return h;
    }
};

template <typename Key, typename Value, typename Hash = std::hash<Key>, std::size_t Shards = 256>
class ShardedMap {
public:
    ShardedMap() = default;

    bool get_copy(const Key& key, Value* out) const {
        if (!out) {
            return false;
        }
        auto& shard = shard_for(key);
        std::lock_guard<std::mutex> lock(shard.mutex);
        auto it = shard.values.find(key);
        if (it == shard.values.end()) {
            return false;
        }
        *out = it->second;
        return true;
    }

    template <typename InitFn, typename UpdateFn>
    void with_mut(const Key& key, InitFn init_fn, UpdateFn update_fn) {
        auto& shard = shard_for(key);
        std::lock_guard<std::mutex> lock(shard.mutex);
        auto [it, inserted] = shard.values.emplace(key, init_fn());
        (void)inserted;
        update_fn(it->second);
    }

    void clear() {
        for (auto& shard : shards_) {
            std::lock_guard<std::mutex> lock(shard.mutex);
            shard.values.clear();
        }
    }

    std::size_t size() const {
        std::size_t total = 0;
        for (const auto& shard : shards_) {
            std::lock_guard<std::mutex> lock(shard.mutex);
            total += shard.values.size();
        }
        return total;
    }

private:
    struct Bucket {
        mutable std::mutex mutex{};
        std::unordered_map<Key, Value, Hash> values{};
    };

    Bucket& shard_for(const Key& key) {
        const std::size_t idx = hasher_(key) % Shards;
        return shards_[idx];
    }

    const Bucket& shard_for(const Key& key) const {
        const std::size_t idx = hasher_(key) % Shards;
        return shards_[idx];
    }

    Hash hasher_{};
    std::array<Bucket, Shards> shards_{};
};

}  // namespace tetris_v2::cc2

