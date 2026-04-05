#pragma once

#include <cstddef>
#include <array>
#include <chrono>
#include <deque>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <cstdint>

#include "tetris_v2/board.hpp"
#include "tetris_v2/randomizer.hpp"
#include "tetris_v2/types.hpp"

namespace tetris_v2 {

struct ScoringConfig {
    std::array<float, 5> normal_clear_reward{0.0f, 100.0f, 300.0f, 500.0f, 800.0f};
    std::array<float, 5> spin_clear_reward{0.0f, 200.0f, 500.0f, 800.0f, 1200.0f};
    float b2b_bonus{100.0f};
    float combo_unit_bonus{50.0f};
};

struct AttackConfig {
    AttackRoundingMode rounding_mode{AttackRoundingMode::Down};
    int all_clear_bonus{5};
    int b2b_charging_surge_start_streak{4};
    int b2b_charging_non_quickplay_base{4};
    bool all_mini_plus{true};
};

struct EnvConfig {
    std::uint32_t seed{1};
    GameMode mode{GameMode::Legacy};
    int queue_size{5};
    float gravity_per_step{1.0f / 60.0f};
    int lock_delay_steps{30};
    int max_lock_resets{15};
    bool allow_rotate_180{true};
    ScoringConfig scoring{};
    AttackConfig attack{};
};

struct EnvState {
    Board board{};
    std::array<std::array<std::int8_t, Board::kWidth>, Board::kRows> piece_ids{};
    ActivePiece active{};
    std::optional<Piece> hold{};
    bool hold_available{true};
    std::deque<Piece> queue{};
    bool game_over{false};
    bool top_out{false};
    int combo{-1};
    bool back_to_back{false};
    int b2b_streak{0};
    int b2b_surge_charge{0};
    int total_lines_cleared{0};
    int lock_timer{0};
    int lock_resets_used{0};
    float gravity_accumulator{0.0f};
    bool spin_eligible{false};
    bool rotated_this_piece{false};
    bool last_rotate_used_kick{false};
    int last_rotate_kick_index{-1};
    bool last_clear_spin{false};
    SpinType last_clear_spin_type{SpinType::None};
    bool last_clear_difficult{false};
    bool last_clear_b2b_bonus{false};
    bool last_clear_all_clear{false};
    int last_attack_base{0};
    float last_attack_combo_scaled{0.0f};
    int last_attack_rounded{0};
    int last_attack_b2b_bonus{0};
    int last_attack_all_clear_bonus{0};
    int last_attack_surge_charge{0};
    int last_attack_surge_release{0};
    int last_attack_total{0};
    int blitz_score_total{0};
    int blitz_level{1};
    int blitz_lines_to_next{0};
    int blitz_time_remaining_ms{0};
    bool blitz_timed_out{false};
};

struct EnvSnapshot {
    EnvState state{};
    SevenBagRandomizer randomizer{};
    std::uint32_t garbage_rng_state{1u};
    GameMode mode{GameMode::Legacy};
};

struct PlacementOption {
    ActivePiece placement{};
    Board board_after_lock{};
    std::array<bool, Board::kRows> cleared_rows{};
    int lines_cleared{0};
    bool spin_eligible_path{false};
    bool last_rotate_used_kick_path{false};
    int last_rotate_kick_index_path{-1};
    bool spin_clear_candidate{false};
    bool difficult_clear_candidate{false};
};

struct KickTest {
    int test_index{0};
    int phase{0};
    int kick_index{0};
    Cell offset{};
    ActivePiece candidate{};
    bool collides{true};
    bool passed{false};
};

struct RotationTrace {
    Action action{Action::None};
    Rotation from_rotation{Rotation::North};
    Rotation target_rotation{Rotation::North};
    std::vector<KickTest> tests{};
    bool success{false};
    std::optional<ActivePiece> final_pose{};
};

class ModernTetrisEnv {
public:
    explicit ModernTetrisEnv(const EnvConfig& config = {});

    void reset(std::optional<std::uint32_t> seed = std::nullopt);
    void set_mode(GameMode mode);
    GameMode mode() const { return config_.mode; }
    void set_blitz_time_limit_ms(int time_limit_ms);
    int blitz_time_limit_ms() const { return blitz_time_limit_ms_; }
    void refresh_runtime_state();
    bool apply_incoming_garbage(int lines, int* lines_applied = nullptr);
    StepResult step(Action action);
    std::vector<PlacementOption> enumerate_active_piece_placements() const;
    const std::vector<PlacementOption>& placement_options_view() const;
    std::optional<PlacementOption> placement_option_at(std::size_t index) const;
    std::vector<std::uint8_t> visible_board_piece_ids(bool include_active) const;
    std::vector<std::uint8_t> visible_placement_piece_ids(std::size_t index) const;
    StepResult apply_placement(const ActivePiece& placement);
    StepResult apply_placement_index(std::size_t index);
    StepResult apply_placement_option_fast(const PlacementOption& option);
    RotationTrace rotation_trace(Action rotate_action) const;

    EnvSnapshot snapshot() const;
    void restore(const EnvSnapshot& snapshot);
    void restore(const EnvState& state);

    const EnvState& state() const { return state_; }
    const EnvConfig& config() const { return config_; }
    const StepResult& last_step_result() const { return last_step_result_; }

    std::vector<std::string> render_rows(int visible_rows = Board::kVisibleRows) const;
    std::string render_ascii(int visible_rows = Board::kVisibleRows) const;

private:
    void invalidate_placement_cache() const;
    const std::vector<PlacementOption>& placement_options_cached() const;
    std::vector<PlacementOption> build_placement_options_uncached() const;
    void ensure_queue(std::size_t minimum);
    void spawn_next_piece(bool reset_hold_availability);
    bool collides(const ActivePiece& piece) const;
    bool try_move(int dx, int dy);
    bool try_rotate(Rotation target_rotation, bool* used_kick = nullptr, int* kick_index = nullptr);
    std::pair<std::optional<ActivePiece>, std::vector<KickTest>> kicked_rotation_with_tests(
        const ActivePiece& from, Rotation target_rotation, int phase, int start_test_index) const;
    std::optional<ActivePiece> kicked_rotation(const ActivePiece& from, Rotation target_rotation) const;
    std::optional<std::pair<ActivePiece, int>> kicked_rotation_quick(
        const ActivePiece& from, Rotation target_rotation) const;
    std::optional<std::pair<ActivePiece, bool>> kicked_rotate_180_with_kick(const ActivePiece& from) const;
    std::optional<ActivePiece> kicked_rotate_180(const ActivePiece& from) const;
    bool touching_ground() const;
    bool apply_hold();
    void lock_active_piece(StepResult& result);
    StepResult apply_option_impl(const PlacementOption& option);
    float line_clear_reward(int lines, bool spin_clear) const;
    float combo_bonus(int combo) const;
    bool is_mode_legacy() const { return config_.mode == GameMode::Legacy; }
    bool is_mode_blitz() const { return config_.mode == GameMode::Scoring; }
    bool is_mode_charging() const {
        return config_.mode == GameMode::Zen || config_.mode == GameMode::Versus;
    }
    bool is_mode_chaining() const { return false; }
    bool is_mode_tetrio_like() const {
        return config_.mode == GameMode::Zen || config_.mode == GameMode::Versus;
    }
    bool piece_immobile(const ActivePiece& piece) const;
    int apply_attack_rounding(float attack) const;
    int b2b_chaining_extra(int streak) const;
    int versus_guideline_base_attack(int lines, SpinType spin_type) const;
    int versus_guideline_combo_bonus(int combo) const;
    int blitz_clear_points(int lines, SpinType spin_type) const;
    int blitz_level_from_total_lines(int total_lines) const;
    int blitz_lines_to_next(int level, int total_lines) const;
    void reset_blitz_state();
    void refresh_blitz_timer();
    void update_blitz_level_and_gravity();
    int attack_base_for_clear(int lines, SpinType spin_type, bool b2b_active) const;
    bool is_difficult_clear(int lines, SpinType spin_type) const;
    bool is_all_clear_after_line_clear() const;
    StepResult make_result_defaults() const;
    void sync_state_to_result(StepResult& result) const;

    EnvConfig config_;
    SevenBagRandomizer randomizer_;
    std::uint32_t garbage_rng_state_{1u};
    float base_gravity_per_step_{1.0f / 60.0f};
    int blitz_time_limit_ms_{120000};
    std::chrono::steady_clock::time_point blitz_last_wall_time_{};
    bool blitz_clock_started_{false};
    EnvState state_{};
    StepResult last_step_result_{};
    mutable std::vector<PlacementOption> placement_cache_{};
    mutable bool placement_cache_valid_{false};
    mutable std::uint64_t placement_cache_epoch_{0};
    mutable std::uint64_t placement_cache_built_epoch_{0};
};

}  // namespace tetris_v2
