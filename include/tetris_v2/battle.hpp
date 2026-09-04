#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <random>
#include <vector>

#include "tetris_v2/cc_env.hpp"

namespace tetris_v2 {

inline constexpr std::size_t kBattlePlayerCount = 2;
inline constexpr std::size_t kBattleOwnObservationSize = 254;
inline constexpr std::size_t kBattleOpponentBoardSize =
    Board::kVisibleRows * Board::kWidth;
inline constexpr std::size_t kBattlePublicFeatureCount = 16;
inline constexpr std::size_t kBattleObservationSize =
    kBattleOwnObservationSize + kBattleOpponentBoardSize + kBattlePublicFeatureCount;

struct BattleConfig {
    std::uint32_t seed{1};
    std::array<int, 5> attack_table{0, 0, 1, 2, 4};
    int garbage_delay{1};
    int max_joint_steps{500};
    bool same_piece_sequence{true};
};

struct GarbagePacket {
    int due_step{0};
    std::deque<int> hole_columns{};
};

struct BattlePlayerStats {
    int placements{0};
    float score{0.0f};
    int lines_cleared{0};
    int attack_generated{0};
    int garbage_cancelled{0};
    int garbage_sent{0};
    int garbage_received{0};
    int garbage_applied{0};
    int top_outs{0};
};

struct BattlePlayerStepResult {
    bool action_succeeded{false};
    bool used_hold{false};
    std::size_t placement_index{0};
    float reward{0.0f};
    int lines_cleared{0};
    int attack_generated{0};
    int garbage_cancelled{0};
    int garbage_sent{0};
    int garbage_received{0};
    int garbage_applied{0};
    int incoming_garbage{0};
    int next_garbage_delay{-1};
    bool top_out{false};
};

struct BattleStepResult {
    bool success{false};
    bool terminated{false};
    int winner{-1};
    int joint_step{0};
    std::array<BattlePlayerStepResult, kBattlePlayerCount> players{};
};

struct BattleSnapshot {
    std::array<EnvSnapshot, kBattlePlayerCount> players{};
    std::array<std::deque<GarbagePacket>, kBattlePlayerCount> incoming{};
    std::array<std::mt19937, kBattlePlayerCount> garbage_rng{};
    std::array<BattlePlayerStats, kBattlePlayerCount> stats{};
    int joint_steps{0};
    bool terminated{false};
    int winner{-1};
};

class BattleEnv {
public:
    explicit BattleEnv(const BattleConfig& config = {});

    void reset(std::optional<std::uint32_t> seed = std::nullopt);
    BattleStepResult step(std::size_t player0_action, std::size_t player1_action);
    bool enqueue_garbage(
        std::size_t player,
        const std::vector<int>& hole_columns,
        int delay);

    std::vector<float> observation(std::size_t perspective_player) const;
    int pending_garbage(std::size_t player) const;
    int next_garbage_delay(std::size_t player) const;

    BattleSnapshot snapshot() const;
    void restore(const BattleSnapshot& snapshot);

    const BattleConfig& config() const { return config_; }
    int joint_steps() const { return joint_steps_; }
    bool terminated() const { return terminated_; }
    int winner() const { return winner_; }
    const BattlePlayerStats& stats(std::size_t player) const;
    cc::Env& player_env(std::size_t player);
    const cc::Env& player_env(std::size_t player) const;

private:
    static BattleConfig validate_config(BattleConfig config);
    static std::uint32_t mixed_seed(std::uint32_t seed, std::uint32_t stream);
    static EnvConfig player_config(std::uint32_t seed);
    int attack_for_lines(int lines) const;
    int random_hole(std::size_t recipient);
    int cancel_pending(std::size_t player, int attack);
    void enqueue_generated(std::size_t recipient, int lines, int delay);
    int apply_due_garbage(std::size_t player);
    void finish_if_terminal(BattleStepResult& result);

    BattleConfig config_{};
    std::uint32_t seed_{1};
    std::array<cc::Env, kBattlePlayerCount> players_;
    std::array<std::deque<GarbagePacket>, kBattlePlayerCount> incoming_{};
    std::array<std::mt19937, kBattlePlayerCount> garbage_rng_{};
    std::array<BattlePlayerStats, kBattlePlayerCount> stats_{};
    int joint_steps_{0};
    bool terminated_{false};
    int winner_{-1};
};

}  // namespace tetris_v2
