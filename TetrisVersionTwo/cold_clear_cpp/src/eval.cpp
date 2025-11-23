#include "cold_clear_cpp/eval.hpp"

#include <algorithm>
#include <cmath>
#include <optional>

namespace cold_clear_cpp {

namespace {

inline int popcount64(std::uint64_t v) {
#if defined(__GNUG__)
    return __builtin_popcountll(v);
#else
    int count = 0;
    while (v) {
        v &= v - 1;
        ++count;
    }
    return count;
#endif
}

inline int column_height(std::uint64_t col) {
    if (col == 0) {
        return 0;
    }
#if defined(__GNUG__)
    return 64 - __builtin_clzll(col);
#else
    int height = 0;
    while (col) {
        col >>= 1;
        ++height;
    }
    return height;
#endif
}

inline int trailing_ones(std::uint64_t v) {
    int c = 0;
    while (v & 1ull) {
        ++c;
        v >>= 1;
    }
    return c;
}

inline int ctz64(std::uint64_t v) {
#if defined(__GNUG__)
    return __builtin_ctzll(v);
#else
    int count = 0;
    if (v == 0) {
        return 64;
    }
    while ((v & 1ull) == 0) {
        v >>= 1;
        ++count;
    }
    return count;
#endif
}

std::optional<PieceLocation> well_known_tslot_left(const Board& board) {
    for (std::size_t x = 0; x + 2 < board.cols.size(); ++x) {
        auto y = column_height(board.cols[x]);
        auto h1 = column_height(board.cols[x + 1]);
        if (h1 >= y) {
            continue;
        }
        if (!board.occupied({static_cast<std::int8_t>(x + 2), static_cast<std::int8_t>(y - 1)})) {
            continue;
        }
        if (board.occupied({static_cast<std::int8_t>(x + 2), static_cast<std::int8_t>(y)})) {
            continue;
        }
        if (!board.occupied({static_cast<std::int8_t>(x + 2), static_cast<std::int8_t>(y + 1)})) {
            continue;
        }
        return PieceLocation{
            Piece::T, Rotation::South, static_cast<std::int8_t>(x + 1), static_cast<std::int8_t>(y)};
    }
    return std::nullopt;
}

std::optional<PieceLocation> well_known_tslot_right(const Board& board) {
    for (std::size_t x = 0; x + 2 < board.cols.size(); ++x) {
        auto y = column_height(board.cols[x + 2]);
        auto h1 = column_height(board.cols[x + 1]);
        if (h1 >= y) {
            continue;
        }
        if (!board.occupied({static_cast<std::int8_t>(x), static_cast<std::int8_t>(y - 1)})) {
            continue;
        }
        if (board.occupied({static_cast<std::int8_t>(x), static_cast<std::int8_t>(y)})) {
            continue;
        }
        if (!board.occupied({static_cast<std::int8_t>(x), static_cast<std::int8_t>(y + 1)})) {
            continue;
        }
        return PieceLocation{
            Piece::T, Rotation::South, static_cast<std::int8_t>(x + 1), static_cast<std::int8_t>(y)};
    }
    return std::nullopt;
}

}  // namespace

Eval Eval::average(const std::vector<std::optional<Eval>>& values) {
    double sum = 0.0;
    std::size_t count = 0;
    for (const auto& v : values) {
        sum += v.has_value() ? v->value : -1000.0;
        ++count;
    }
    return Eval{count == 0 ? 0.0 : sum / static_cast<double>(count)};
}

std::pair<Eval, Eval::Reward> evaluate(
    const Weights& weights, GameState state, const PlacementInfo& info, std::uint32_t softdrop) {
    double eval = 0.0;
    double reward = 0.0;

    if (info.perfect_clear) {
        reward += weights.perfect_clear;
    }
    if (!info.perfect_clear || !weights.perfect_clear_override) {
        if (info.back_to_back) {
            reward += weights.back_to_back_clear;
        }
        switch (info.placement.spin) {
            case Spin::None:
                reward += weights.normal_clears[info.lines_cleared];
                break;
            case Spin::Mini:
                reward += weights.mini_spin_clears[info.lines_cleared];
                break;
            case Spin::Full:
                reward += weights.spin_clears[info.lines_cleared];
                break;
        }
        reward += weights.combo_attack * static_cast<float>((info.combo > 1) ? ((info.combo - 1) / 2) : 0);
    }

    if (info.placement.location.piece == Piece::T &&
        (info.lines_cleared < 2 || info.placement.spin != Spin::Full)) {
        reward += weights.wasted_t;
    }
    if (state.back_to_back) {
        eval += weights.has_back_to_back;
    }
    reward += weights.softdrop * static_cast<float>(softdrop);

    int cutout_count = static_cast<int>(state.bag.test(piece_index(Piece::T))) +
                       static_cast<int>(state.reserve == Piece::T) +
                       static_cast<int>(state.bag.count() <= 3);
    for (int i = 0; i < cutout_count; ++i) {
        auto location = well_known_tslot_left(state.board);
        if (!location) {
            location = well_known_tslot_right(state.board);
        }
        if (!location) {
            break;
        }
        Board board_copy = state.board;
        board_copy.place(*location);
        auto lines = board_copy.line_clears();
        auto cleared = popcount64(lines);
        eval += weights.tslot[static_cast<std::size_t>(cleared)];
        if (cleared > 1) {
            board_copy.remove_lines(lines);
            state.board = board_copy;
        }
    }

    eval += weights.holes *
            static_cast<float>([&]() {
                std::uint32_t total = 0;
                for (auto c : state.board.cols) {
                    auto height = column_height(c);
                    std::uint64_t underneath = (height >= 64) ? ~0ull : ((1ull << height) - 1);
                    auto holes = (~c) & underneath;
                    total += static_cast<std::uint32_t>(popcount64(holes));
                }
                return total;
            }());

    int coveredness = 0;
    for (auto c : state.board.cols) {
        auto height = column_height(c);
        std::uint64_t underneath = (height >= 64) ? ~0ull : ((1ull << height) - 1);
        auto holes = (~c) & underneath;
        while (holes != 0) {
            auto y = ctz64(holes);
            coveredness += std::min<int>(height - y, static_cast<int>(weights.max_cell_covered_height));
            holes &= holes - 1;
        }
    }
    eval += weights.cell_coveredness * static_cast<float>(coveredness);

    std::size_t tetris_well_column = 0;
    int tetris_well_height = column_height(state.board.cols[0]);
    for (std::size_t i = 1; i < state.board.cols.size(); ++i) {
        int h = column_height(state.board.cols[i]);
        if (h < tetris_well_height) {
            tetris_well_height = h;
            tetris_well_column = i;
        }
    }
    std::uint64_t full_lines_except_well = ~0ull;
    for (std::size_t i = 0; i < state.board.cols.size(); ++i) {
        if (i == tetris_well_column) {
            continue;
        }
        full_lines_except_well &= state.board.cols[i];
    }
    auto depth = trailing_ones(full_lines_except_well >> tetris_well_height);
    eval += static_cast<float>(depth) * weights.tetris_well_depth;

    int highest_point = 0;
    for (auto c : state.board.cols) {
        highest_point = std::max(highest_point, column_height(c));
    }
    eval += weights.height * static_cast<float>(highest_point);
    if (highest_point > 10) {
        eval += weights.height_upper_half * static_cast<float>(highest_point - 10);
    }
    if (highest_point > 15) {
        eval += weights.height_upper_quarter * static_cast<float>(highest_point - 15);
    }

    std::uint32_t row_transitions = popcount64(~state.board.cols[0]) + popcount64(~state.board.cols[9]);
    for (std::size_t i = 0; i + 1 < state.board.cols.size(); ++i) {
        row_transitions += popcount64(state.board.cols[i] ^ state.board.cols[i + 1]);
    }
    eval += static_cast<float>(row_transitions) * weights.row_transitions;

    return {Eval{eval}, Eval::Reward{reward}};
}

Weights default_weights() { return Weights{}; }

}  // namespace cold_clear_cpp
