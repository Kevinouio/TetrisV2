#pragma once

#include <random>
#include <string>
#include <vector>

#include "cold_clear_cpp/bot.hpp"
#include "cold_clear_cpp/board.hpp"

namespace tetris_v2 {

// Lightweight single-player environment that uses the cold_clear_cpp core types.
class TetrisEnv {
public:
    explicit TetrisEnv(unsigned int seed = std::random_device{}());

    const cold_clear_cpp::GameState& state() const { return state_; }
    const std::vector<cold_clear_cpp::Piece>& queue() const { return queue_; }

    // Ensure the visible queue has at least min_count pieces; returns the pieces that were appended.
    std::vector<cold_clear_cpp::Piece> ensure_queue(std::size_t min_count);

    // Apply a placement for the current queue front. Returns the placement info.
    cold_clear_cpp::PlacementInfo apply(const cold_clear_cpp::Placement& mv);

    bool game_over() const { return game_over_; }

    // Render the visible 20 rows of the board as ASCII.
    std::string render() const;

    // Return visible rows (top to bottom) for custom rendering.
    std::vector<std::string> display_rows(int visible_rows = 20) const;

    cold_clear_cpp::Piece hold_piece() const { return state_.reserve; }

    static char piece_label(cold_clear_cpp::Piece p);

private:
    void place_on_display(const cold_clear_cpp::Placement& mv);
    void clear_display_lines(std::uint64_t mask);
    cold_clear_cpp::Board preview_board_after(const cold_clear_cpp::Placement& mv) const;
    cold_clear_cpp::Piece draw_from_bag();

    cold_clear_cpp::GameState state_{};
    std::vector<cold_clear_cpp::Piece> queue_;
    std::vector<cold_clear_cpp::Piece> bag_buffer_;
    std::mt19937 rng_;
    std::array<std::array<char, 40>, 10> display_{};
    bool game_over_{false};
};

}  // namespace tetris_v2
