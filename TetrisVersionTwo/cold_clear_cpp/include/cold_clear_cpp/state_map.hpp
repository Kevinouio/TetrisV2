#pragma once

#include <array>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cold_clear_cpp/state.hpp"

namespace cold_clear_cpp {

template <typename V>
class StateMapReadGuard {
public:
    StateMapReadGuard() = default;
    StateMapReadGuard(V* value, std::shared_lock<std::shared_mutex>&& lock)
        : lock_(std::move(lock)), value_(value) {}
    StateMapReadGuard(StateMapReadGuard&&) noexcept = default;
    StateMapReadGuard& operator=(StateMapReadGuard&&) noexcept = default;

    V* operator->() const { return value_; }
    V& operator*() const { return *value_; }
    explicit operator bool() const { return value_ != nullptr; }

private:
    std::shared_lock<std::shared_mutex> lock_;
    V* value_{nullptr};
};

template <typename V>
class StateMapWriteGuard {
public:
    StateMapWriteGuard() = default;
    StateMapWriteGuard(V* value, std::unique_lock<std::shared_mutex>&& lock)
        : lock_(std::move(lock)), value_(value) {}
    StateMapWriteGuard(StateMapWriteGuard&&) noexcept = default;
    StateMapWriteGuard& operator=(StateMapWriteGuard&&) noexcept = default;

    V* operator->() const { return value_; }
    V& operator*() const { return *value_; }
    explicit operator bool() const { return value_ != nullptr; }

private:
    std::unique_lock<std::shared_mutex> lock_;
    V* value_{nullptr};
};

template <typename V>
class StateMap {
public:
    using ReadGuard = StateMapReadGuard<V>;
    using ConstReadGuard = StateMapReadGuard<const V>;
    using WriteGuard = StateMapWriteGuard<V>;

    StateMap() = default;
    StateMap(const StateMap&) = delete;
    StateMap& operator=(const StateMap&) = delete;

    StateMap(StateMap&& other) noexcept {
        hasher_ = other.hasher_;
        for (std::size_t i = 0; i < kShards; ++i) {
            std::unique_lock<std::shared_mutex> lock_other(other.shards_[i].mutex);
            shards_[i].entries = std::move(other.shards_[i].entries);
        }
    }

    StateMap& operator=(StateMap&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        hasher_ = other.hasher_;
        for (std::size_t i = 0; i < kShards; ++i) {
            std::unique_lock<std::shared_mutex> lock_self(shards_[i].mutex);
            std::unique_lock<std::shared_mutex> lock_other(other.shards_[i].mutex);
            shards_[i].entries = std::move(other.shards_[i].entries);
        }
        return *this;
    }

    std::uint64_t index(const GameState& k) const { return static_cast<std::uint64_t>(hasher_(k)); }

    std::optional<ReadGuard> get(const GameState& k) {
        auto idx = index(k);
        auto& shard = shard_for(idx);
        std::shared_lock<std::shared_mutex> lock(shard.mutex);
        auto it = shard.entries.find(idx);
        if (it == shard.entries.end() || !(it->second.state == k)) {
            return std::nullopt;
        }
        return ReadGuard(&it->second.value, std::move(lock));
    }

    std::optional<ConstReadGuard> get(const GameState& k) const {
        auto idx = index(k);
        const auto& shard = shard_for(idx);
        std::shared_lock<std::shared_mutex> lock(shard.mutex);
        auto it = shard.entries.find(idx);
        if (it == shard.entries.end() || !(it->second.state == k)) {
            return std::nullopt;
        }
        return ConstReadGuard(&it->second.value, std::move(lock));
    }

    std::optional<ReadGuard> get_raw(std::uint64_t idx) {
        auto& shard = shard_for(idx);
        std::shared_lock<std::shared_mutex> lock(shard.mutex);
        auto it = shard.entries.find(idx);
        if (it == shard.entries.end()) {
            return std::nullopt;
        }
        return ReadGuard(&it->second.value, std::move(lock));
    }

    std::optional<ConstReadGuard> get_raw(std::uint64_t idx) const {
        const auto& shard = shard_for(idx);
        std::shared_lock<std::shared_mutex> lock(shard.mutex);
        auto it = shard.entries.find(idx);
        if (it == shard.entries.end()) {
            return std::nullopt;
        }
        return ConstReadGuard(&it->second.value, std::move(lock));
    }

    std::optional<WriteGuard> get_raw_mut(std::uint64_t idx) {
        auto& shard = shard_for(idx);
        std::unique_lock<std::shared_mutex> lock(shard.mutex);
        auto it = shard.entries.find(idx);
        if (it == shard.entries.end()) {
            return std::nullopt;
        }
        return WriteGuard(&it->second.value, std::move(lock));
    }

    template <typename F>
    WriteGuard get_raw_or_insert_with(std::uint64_t idx, const GameState& state, F&& f) {
        auto& shard = shard_for(idx);
        std::unique_lock<std::shared_mutex> lock(shard.mutex);
        auto [it, inserted] = shard.entries.try_emplace(idx, Entry{state, V{}});
        if (inserted) {
            it->second.value = f();
        }
        return WriteGuard(&it->second.value, std::move(lock));
    }

    template <typename F>
    WriteGuard get_or_insert_with(const GameState& k, F&& f) {
        auto idx = index(k);
        return get_raw_or_insert_with(idx, k, std::forward<F>(f));
    }

    template <typename T, typename F>
    StateMap<T> map_values(F&& f) const {
        StateMap<T> out;
        out.hasher_ = hasher_;
        for (std::size_t shard_idx = 0; shard_idx < kShards; ++shard_idx) {
            const auto& shard = shards_[shard_idx];
            std::shared_lock<std::shared_mutex> lock(shard.mutex);
            for (const auto& kv : shard.entries) {
                out.shards_[shard_idx].entries.emplace(
                    kv.first, typename StateMap<T>::Entry{kv.second.state, f(kv.second.value)});
            }
        }
        return out;
    }

    template <typename Func>
    void for_each(Func&& f) const {
        for (const auto& shard : shards_) {
            std::shared_lock<std::shared_mutex> lock(shard.mutex);
            for (const auto& kv : shard.entries) {
                f(kv.second.state, kv.second.value);
            }
        }
    }

private:
    template <typename>
    friend class StateMap;

    struct Entry {
        GameState state;
        V value;
    };

    struct Shard {
        mutable std::shared_mutex mutex;
        std::unordered_map<std::uint64_t, Entry> entries;
    };

    static constexpr std::size_t kShardIndexBits = 12;
    static constexpr std::size_t kShardIndexShift = 32;
    static constexpr std::size_t kShards = 1u << kShardIndexBits;

    std::size_t shard_index(std::uint64_t idx) const {
        return static_cast<std::size_t>((idx >> kShardIndexShift) & (kShards - 1));
    }

    Shard& shard_for(std::uint64_t idx) { return shards_[shard_index(idx)]; }
    const Shard& shard_for(std::uint64_t idx) const { return shards_[shard_index(idx)]; }

    std::array<Shard, kShards> shards_{};
    GameStateHash hasher_{};
};

}  // namespace cold_clear_cpp
