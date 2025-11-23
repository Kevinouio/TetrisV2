#include "tetris_v2/tetris_env.hpp"

#include <algorithm>
#include <sstream>

using namespace cold_clear_cpp;

namespace tetris_v2 {

TetrisEnv::TetrisEnv(unsigned int seed) : rng_(seed) {
    state_.bag = all_pieces();
    state_.back_to_back = false;
    state_.combo = 0;
    state_.reserve = draw_from_bag();  // start with a hold piece
    queue_.push_back(draw_from_bag());
    ensure_queue(6);
    for (auto& col : display_) {
        col.fill('.');
    }
}

std::vector<Piece> TetrisEnv::ensure_queue(std::size_t min_count) {
    std::vector<Piece> added;
    while (queue_.size() < min_count) {
        auto p = draw_from_bag();
        queue_.push_back(p);
        added.push_back(p);
    }
    return added;
}

PlacementInfo TetrisEnv::apply(const Placement& mv) {
    if (queue_.empty()) {
        game_over_ = true;
        return {};
    }
    // Compute line clears for display before mutating state.
    auto preview = preview_board_after(mv);
    auto cleared_mask = preview.line_clears();

    place_on_display(mv);
    if (cleared_mask) {
        clear_display_lines(cleared_mask);
    }

    auto next = queue_.front();
    queue_.erase(queue_.begin());
    auto info = state_.advance(next, mv);
    ensure_queue(6);
    return info;
}

std::string TetrisEnv::render() const {
    std::ostringstream oss;
    for (const auto& row : display_rows()) {
        oss << row << '\n';
    }
    return oss.str();
}

std::vector<std::string> TetrisEnv::display_rows(int visible_rows) const {
    std::vector<std::string> rows;
    rows.reserve(visible_rows);
    for (int y = visible_rows - 1; y >= 0; --y) {
        std::string line;
        line.reserve(10);
        for (int x = 0; x < 10; ++x) {
            char c = display_[static_cast<std::size_t>(x)][static_cast<std::size_t>(y)];
            line.push_back(c == '.' ? '.' : c);
        }
        rows.push_back(std::move(line));
    }
    return rows;
}

Piece TetrisEnv::draw_from_bag() {
    if (bag_buffer_.empty()) {
        bag_buffer_.assign(kAllPieces.begin(), kAllPieces.end());
        std::shuffle(bag_buffer_.begin(), bag_buffer_.end(), rng_);
    }
    auto p = bag_buffer_.back();
    bag_buffer_.pop_back();
    return p;
}

char TetrisEnv::piece_label(Piece p) {
    switch (p) {
        case Piece::I: return 'I';
        case Piece::O: return 'O';
        case Piece::T: return 'T';
        case Piece::L: return 'L';
        case Piece::J: return 'J';
        case Piece::S: return 'S';
        case Piece::Z: return 'Z';
    }
    return '?';
}

void TetrisEnv::place_on_display(const Placement& mv) {
    char c = piece_label(mv.location.piece);
    for (auto cell : mv.location.cells()) {
        auto [x, y] = cell;
        if (x < 0 || x >= 10 || y < 0 || y >= 40) {
            continue;
        }
        display_[static_cast<std::size_t>(x)][static_cast<std::size_t>(y)] = c;
    }
}

void TetrisEnv::clear_display_lines(std::uint64_t mask) {
    for (auto& col : display_) {
        std::array<char, 40> new_col{};
        new_col.fill('.');
        int dst = 0;
        for (int src = 0; src < 40; ++src) {
            if (mask & (1ull << src)) {
                continue;
            }
            new_col[dst++] = col[static_cast<std::size_t>(src)];
        }
        col = new_col;
    }
}

Board TetrisEnv::preview_board_after(const Placement& mv) const {
    Board b = state_.board;
    b.place(mv.location);
    return b;
}

}  // namespace tetris_v2
