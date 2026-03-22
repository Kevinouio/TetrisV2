#include "tetris_v2/cc_env.hpp"

namespace tetris_v2::cc {

namespace {

EnvConfig parity_config(EnvConfig config) {
    config.allow_rotate_180 = false;
    return config;
}

}  // namespace

Env::Env(const EnvConfig& config) : env_(parity_config(config)) {}

void Env::reset(std::optional<std::uint32_t> seed) { env_.reset(seed); }

StepResult Env::step(Action action) { return env_.step(action); }

StepResult Env::apply_placement_index(std::size_t index) {
    return env_.apply_placement_index(index);
}

std::vector<PlacementOption> Env::enumerate_active_piece_placements() const {
    return env_.enumerate_active_piece_placements();
}

std::optional<PlacementOption> Env::placement_option_at(std::size_t index) const {
    return env_.placement_option_at(index);
}

RotationTrace Env::rotation_trace(Action rotate_action) const {
    return env_.rotation_trace(rotate_action);
}

std::vector<std::uint8_t> Env::visible_board_piece_ids(bool include_active) const {
    return env_.visible_board_piece_ids(include_active);
}

std::vector<std::uint8_t> Env::visible_placement_piece_ids(std::size_t index) const {
    return env_.visible_placement_piece_ids(index);
}

EnvSnapshot Env::snapshot() const { return env_.snapshot(); }

void Env::restore(const EnvSnapshot& snapshot) { env_.restore(snapshot); }

void Env::restore(const EnvState& state) { env_.restore(state); }

BotAdapterState Env::bot_adapter_state() const {
    BotAdapterState out{};
    const auto& s = env_.state();
    out.next = s.active.piece;
    if (s.hold.has_value()) {
        out.reserve = *s.hold;
    } else if (!s.queue.empty()) {
        out.reserve = s.queue.front();
    }
    out.hold_available = s.hold_available;
    return out;
}

}  // namespace tetris_v2::cc
