#pragma once

#include <unordered_map>
#include <utility>

#include "cold_clear_cpp/state.hpp"

namespace cold_clear_cpp {

template <typename V>
class StateMap {
public:
    StateMap() = default;

    std::size_t index(const GameState& k) const { return hasher_(k); }

    V* get(const GameState& k) {
        auto it = entries_.find(index(k));
        if (it == entries_.end() || !(it->second.state == k)) {
            return nullptr;
        }
        return &it->second.value;
    }

    const V* get(const GameState& k) const {
        auto it = entries_.find(index(k));
        if (it == entries_.end() || !(it->second.state == k)) {
            return nullptr;
        }
        return &it->second.value;
    }

    V* get_raw(std::size_t idx) {
        auto it = entries_.find(idx);
        return it == entries_.end() ? nullptr : &it->second.value;
    }

    const V* get_raw(std::size_t idx) const {
        auto it = entries_.find(idx);
        return it == entries_.end() ? nullptr : &it->second.value;
    }

    template <typename F>
    V& get_raw_or_insert_with(std::size_t idx, const GameState& state, F&& f) {
        auto [it, inserted] = entries_.try_emplace(idx, Entry{state, V{}});
        if (inserted) {
            it->second.value = f();
        }
        return it->second.value;
    }

    template <typename F>
    V& get_or_insert_with(const GameState& k, F&& f) {
        auto idx = index(k);
        return get_raw_or_insert_with(idx, k, std::forward<F>(f));
    }

    template <typename T, typename F>
    StateMap<T> map_values(F&& f) const {
        StateMap<T> out;
        for (const auto& [hash, entry] : entries_) {
            out.entries_.emplace(hash, typename StateMap<T>::Entry{entry.state, f(entry.value)});
        }
        return out;
    }

    const auto& entries() const { return entries_; }

    template <typename Func>
    void for_each(Func&& f) const {
        for (const auto& kv : entries_) {
            f(kv.second.state, kv.second.value);
        }
    }

private:
    template <typename>
    friend class StateMap;

    struct Entry {
        GameState state;
        V value;
    };

    std::unordered_map<std::size_t, Entry> entries_;
    GameStateHash hasher_{};
};

}  // namespace cold_clear_cpp
