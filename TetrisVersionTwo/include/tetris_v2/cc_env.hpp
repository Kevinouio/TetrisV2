#pragma once

#include "tetris_v2/env.hpp"

namespace tetris_v2::cc {

struct BotAdapterState {
    Piece next{Piece::None};
    Piece reserve{Piece::None};
    bool hold_available{false};
};

class Env {
public:
    explicit Env(const EnvConfig& config = {});

    void reset(std::optional<std::uint32_t> seed = std::nullopt);
    StepResult step(Action action);
    StepResult apply_placement_index(std::size_t index);

    std::vector<PlacementOption> enumerate_active_piece_placements() const;
    std::optional<PlacementOption> placement_option_at(std::size_t index) const;
    RotationTrace rotation_trace(Action rotate_action) const;
    std::vector<std::uint8_t> visible_board_piece_ids(bool include_active) const;
    std::vector<std::uint8_t> visible_placement_piece_ids(std::size_t index) const;

    EnvSnapshot snapshot() const;
    void restore(const EnvSnapshot& snapshot);
    void restore(const EnvState& state);

    const EnvState& state() const { return env_.state(); }
    const EnvConfig& config() const { return env_.config(); }

    BotAdapterState bot_adapter_state() const;

    ModernTetrisEnv& raw() { return env_; }
    const ModernTetrisEnv& raw() const { return env_; }

private:
    ModernTetrisEnv env_;
};

}  // namespace tetris_v2::cc
