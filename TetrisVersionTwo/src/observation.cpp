#include "tetris_v2/observation.hpp"

#include <algorithm>

#include "tetris_v2/piece_defs.hpp"

namespace tetris_v2 {

namespace {

constexpr std::size_t kHoldClasses = 8;
constexpr std::size_t kMetaFeatures = 8;

}  // namespace

ObservationLayout build_observation_layout(int queue_size, bool include_hidden_rows) {
    ObservationLayout layout;
    layout.rows_encoded = include_hidden_rows ? Board::kRows : Board::kVisibleRows;
    std::size_t plane_size =
        static_cast<std::size_t>(layout.rows_encoded * Board::kWidth);

    layout.board_offset = 0;
    layout.active_offset = layout.board_offset + plane_size;
    layout.hold_offset = layout.active_offset + plane_size;
    layout.queue_offset = layout.hold_offset + kHoldClasses;
    layout.meta_offset = layout.queue_offset + static_cast<std::size_t>(std::max(0, queue_size) * 7);
    layout.total_size = layout.meta_offset + kMetaFeatures;
    return layout;
}

std::size_t observation_size(int queue_size, bool include_hidden_rows) {
    return build_observation_layout(queue_size, include_hidden_rows).total_size;
}

std::vector<float> encode_observation(
    const EnvState& state, int queue_size, bool include_hidden_rows) {
    auto layout = build_observation_layout(queue_size, include_hidden_rows);
    std::vector<float> obs(layout.total_size, 0.0f);

    const int rows = layout.rows_encoded;
    const int width = Board::kWidth;

    for (int y = 0; y < rows; ++y) {
        auto mask = state.board.row_mask(y);
        for (int x = 0; x < width; ++x) {
            bool occupied = (mask & (1u << x)) != 0;
            if (!occupied) {
                continue;
            }
            std::size_t idx = static_cast<std::size_t>(y * width + x);
            obs[layout.board_offset + idx] = 1.0f;
        }
    }

    auto active_cells = piece_cells(state.active.piece, state.active.rotation);
    for (const auto& cell : active_cells) {
        int x = state.active.x + cell.x;
        int y = state.active.y + cell.y;
        if (x < 0 || x >= width || y < 0 || y >= rows) {
            continue;
        }
        std::size_t idx = static_cast<std::size_t>(y * width + x);
        obs[layout.active_offset + idx] = 1.0f;
    }

    std::size_t hold_class = state.hold.has_value()
                                 ? piece_index(*state.hold)
                                 : static_cast<std::size_t>(Piece::None);
    obs[layout.hold_offset + hold_class] = 1.0f;

    int limit = std::max(0, queue_size);
    for (int i = 0; i < limit && i < static_cast<int>(state.queue.size()); ++i) {
        std::size_t class_index = piece_index(state.queue[static_cast<std::size_t>(i)]);
        std::size_t idx = layout.queue_offset + static_cast<std::size_t>(i * 7) + class_index;
        obs[idx] = 1.0f;
    }

    obs[layout.meta_offset + 0] = state.hold_available ? 1.0f : 0.0f;
    obs[layout.meta_offset + 1] = static_cast<float>(state.combo);
    obs[layout.meta_offset + 2] = state.back_to_back ? 1.0f : 0.0f;
    obs[layout.meta_offset + 3] = static_cast<float>(state.lock_timer);
    obs[layout.meta_offset + 4] = static_cast<float>(state.lock_resets_used);
    obs[layout.meta_offset + 5] = static_cast<float>(state.total_lines_cleared);
    obs[layout.meta_offset + 6] = state.game_over ? 1.0f : 0.0f;
    obs[layout.meta_offset + 7] = state.top_out ? 1.0f : 0.0f;

    return obs;
}

}  // namespace tetris_v2
