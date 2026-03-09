#pragma once

#include <cstddef>
#include <array>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tetris_v2/board.hpp"
#include "tetris_v2/randomizer.hpp"
#include "tetris_v2/types.hpp"

namespace tetris_v2 {

struct EnvConfig {
    std::uint32_t seed{1};
    int queue_size{5};
    float gravity_per_step{1.0f / 60.0f};
    int lock_delay_steps{30};
    int max_lock_resets{15};
    bool allow_rotate_180{true};
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
    int total_lines_cleared{0};
    int lock_timer{0};
    int lock_resets_used{0};
    float gravity_accumulator{0.0f};
};

struct PlacementOption {
    ActivePiece placement{};
    Board board_after_lock{};
    std::array<bool, Board::kRows> cleared_rows{};
    int lines_cleared{0};
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
    StepResult step(Action action);
    std::vector<PlacementOption> enumerate_active_piece_placements() const;
    std::optional<PlacementOption> placement_option_at(std::size_t index) const;
    std::vector<std::uint8_t> visible_board_piece_ids(bool include_active) const;
    std::vector<std::uint8_t> visible_placement_piece_ids(std::size_t index) const;
    StepResult apply_placement(const ActivePiece& placement);
    StepResult apply_placement_index(std::size_t index);
    RotationTrace rotation_trace(Action rotate_action) const;

    EnvState snapshot() const;
    void restore(const EnvState& state);

    const EnvState& state() const { return state_; }
    const EnvConfig& config() const { return config_; }

    std::vector<std::string> render_rows(int visible_rows = Board::kVisibleRows) const;
    std::string render_ascii(int visible_rows = Board::kVisibleRows) const;

private:
    void ensure_queue(std::size_t minimum);
    void spawn_next_piece(bool reset_hold_availability);
    bool collides(const ActivePiece& piece) const;
    bool try_move(int dx, int dy);
    bool try_rotate(Rotation target_rotation);
    std::pair<std::optional<ActivePiece>, std::vector<KickTest>> kicked_rotation_with_tests(
        const ActivePiece& from, Rotation target_rotation, int phase, int start_test_index) const;
    std::optional<ActivePiece> kicked_rotation(const ActivePiece& from, Rotation target_rotation) const;
    std::optional<ActivePiece> kicked_rotate_180(const ActivePiece& from) const;
    bool touching_ground() const;
    bool apply_hold();
    void lock_active_piece(StepResult& result);
    float line_clear_reward(int lines) const;

    EnvConfig config_;
    SevenBagRandomizer randomizer_;
    EnvState state_{};
};

}  // namespace tetris_v2
