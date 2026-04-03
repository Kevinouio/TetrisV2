#include "tetris_v2/cc2_eval.hpp"

#include <algorithm>
#include <optional>

namespace tetris_v2::cc2 {

namespace {

std::uint32_t popcount_u64(std::uint64_t v) {
    return static_cast<std::uint32_t>(__builtin_popcountll(v));
}

std::uint32_t height_of_col(std::uint64_t col) {
    if (col == 0) {
        return 0;
    }
    return static_cast<std::uint32_t>(64 - __builtin_clzll(col));
}

std::uint32_t trailing_ones(std::uint64_t v) {
    const std::uint64_t inv = ~v;
    if (inv == 0) {
        return 64;
    }
    return static_cast<std::uint32_t>(__builtin_ctzll(inv));
}

std::uint64_t lower_bits_mask(std::uint32_t bits) {
    if (bits == 0) {
        return 0;
    }
    if (bits >= 64) {
        return ~0ull;
    }
    return (1ull << bits) - 1ull;
}

std::optional<PieceLocation> well_known_tslot_left(const Board& board) {
    for (int x = 0; x <= 7; ++x) {
        const auto y = static_cast<std::int8_t>(height_of_col(board.cols[static_cast<std::size_t>(x)]));
        if (height_of_col(board.cols[static_cast<std::size_t>(x + 1)]) >=
            static_cast<std::uint32_t>(y)) {
            continue;
        }
        if (!board.occupied(static_cast<std::int8_t>(x + 2), static_cast<std::int8_t>(y - 1))) {
            continue;
        }
        if (board.occupied(static_cast<std::int8_t>(x + 2), y)) {
            continue;
        }
        if (!board.occupied(static_cast<std::int8_t>(x + 2), static_cast<std::int8_t>(y + 1))) {
            continue;
        }
        return PieceLocation{Piece::T, Rotation::South, static_cast<std::int8_t>(x + 1), y};
    }
    return std::nullopt;
}

std::optional<PieceLocation> well_known_tslot_right(const Board& board) {
    for (int x = 0; x <= 7; ++x) {
        const auto y = static_cast<std::int8_t>(height_of_col(board.cols[static_cast<std::size_t>(x + 2)]));
        if (height_of_col(board.cols[static_cast<std::size_t>(x + 1)]) >=
            static_cast<std::uint32_t>(y)) {
            continue;
        }
        if (!board.occupied(static_cast<std::int8_t>(x), static_cast<std::int8_t>(y - 1))) {
            continue;
        }
        if (board.occupied(static_cast<std::int8_t>(x), y)) {
            continue;
        }
        if (!board.occupied(static_cast<std::int8_t>(x), static_cast<std::int8_t>(y + 1))) {
            continue;
        }
        return PieceLocation{Piece::T, Rotation::South, static_cast<std::int8_t>(x + 1), y};
    }
    return std::nullopt;
}

}  // namespace

FreestyleEvalResult evaluate_freestyle(
    const FreestyleWeights& weights,
    GameState state,
    const PlacementInfo& info,
    std::uint32_t softdrop) {
    float eval = 0.0f;
    float reward = 0.0f;

    if (info.perfect_clear) {
        reward += weights.perfect_clear;
    }
    if (!info.perfect_clear || !weights.perfect_clear_override) {
        if (info.back_to_back) {
            reward += weights.back_to_back_clear;
        }
        const auto lines = static_cast<std::size_t>(info.lines_cleared);
        switch (info.placement.spin) {
            case Spin::None:
                reward += weights.normal_clears[std::min<std::size_t>(lines, weights.normal_clears.size() - 1)];
                break;
            case Spin::Mini:
                reward += weights.mini_spin_clears
                    [std::min<std::size_t>(lines, weights.mini_spin_clears.size() - 1)];
                break;
            case Spin::Full:
                reward += weights.spin_clears[std::min<std::size_t>(lines, weights.spin_clears.size() - 1)];
                break;
        }
        reward += weights.combo_attack * static_cast<float>((info.combo > 0 ? (info.combo - 1) : 0) / 2);
    }

    if (info.placement.location.piece == Piece::T &&
        (info.lines_cleared < 2 || info.placement.spin != Spin::Full)) {
        reward += weights.wasted_t;
    }
    if (state.back_to_back) {
        eval += weights.has_back_to_back;
    }
    reward += weights.softdrop * static_cast<float>(softdrop);

    const std::size_t cutout_count = (piece_mask_contains(state.bag_mask, Piece::T) ? 1u : 0u) +
        (state.reserve == Piece::T ? 1u : 0u) + (piece_mask_count(state.bag_mask) <= 3u ? 1u : 0u);
    for (std::size_t i = 0; i < cutout_count; ++i) {
        auto location = well_known_tslot_left(state.board);
        if (!location.has_value()) {
            location = well_known_tslot_right(state.board);
        }
        if (!location.has_value()) {
            break;
        }

        Board board = state.board;
        board.place(*location);
        const auto lines = popcount_u64(board.line_clears());
        eval += weights.tslot[std::min<std::size_t>(lines, weights.tslot.size() - 1)];
        if (lines > 1) {
            const auto clears = board.line_clears();
            board.remove_lines(clears);
            state.board = board;
        }
    }

    std::uint32_t holes = 0;
    for (const auto c : state.board.cols) {
        const auto height = height_of_col(c);
        const auto underneath = lower_bits_mask(height);
        const auto hole_bits = (~c) & underneath;
        holes += popcount_u64(hole_bits);
    }
    eval += weights.holes * static_cast<float>(holes);

    std::uint32_t coveredness = 0;
    for (const auto c : state.board.cols) {
        const auto height = height_of_col(c);
        const auto underneath = lower_bits_mask(height);
        std::uint64_t hole_bits = (~c) & underneath;
        while (hole_bits != 0) {
            const auto y = static_cast<std::uint32_t>(__builtin_ctzll(hole_bits));
            coveredness += std::min(height - y, weights.max_cell_covered_height);
            hole_bits &= ~(1ull << y);
        }
    }
    eval += weights.cell_coveredness * static_cast<float>(coveredness);

    std::size_t well_column = 0;
    std::uint32_t well_height = height_of_col(state.board.cols[0]);
    for (std::size_t x = 1; x < state.board.cols.size(); ++x) {
        const auto h = height_of_col(state.board.cols[x]);
        if (h < well_height) {
            well_height = h;
            well_column = x;
        }
    }

    std::uint64_t full_lines_except_well = ~0ull;
    for (std::size_t x = 0; x < state.board.cols.size(); ++x) {
        if (x == well_column) {
            continue;
        }
        full_lines_except_well &= state.board.cols[x];
    }
    const std::uint64_t shifted =
        (well_height >= 64) ? 0ull : (full_lines_except_well >> static_cast<unsigned int>(well_height));
    const auto well_depth = trailing_ones(shifted);
    eval += static_cast<float>(well_depth) * weights.tetris_well_depth;

    std::uint32_t highest = 0;
    for (const auto c : state.board.cols) {
        highest = std::max(highest, height_of_col(c));
    }
    eval += weights.height * static_cast<float>(highest);
    if (highest > 10) {
        eval += weights.height_upper_half * static_cast<float>(highest - 10);
    }
    if (highest > 15) {
        eval += weights.height_upper_quarter * static_cast<float>(highest - 15);
    }

    std::uint32_t row_transitions = popcount_u64(~state.board.cols[0]) + popcount_u64(~state.board.cols[9]);
    for (std::size_t x = 0; x + 1 < state.board.cols.size(); ++x) {
        row_transitions += popcount_u64(state.board.cols[x] ^ state.board.cols[x + 1]);
    }
    eval += static_cast<float>(row_transitions) * weights.row_transitions;

    return FreestyleEvalResult{eval, reward, eval + reward};
}

}  // namespace tetris_v2::cc2
