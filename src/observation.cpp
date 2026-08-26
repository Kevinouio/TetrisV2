#include "tetris_v2/observation.hpp"

#include <algorithm>

#include "tetris_v2/piece_defs.hpp"

namespace tetris_v2 {

namespace {

constexpr std::size_t kHoldClasses = 8;
constexpr std::size_t kActivePieceClasses = 8;
constexpr std::size_t kMetaFeatures = 3;

}  // namespace

ObservationLayout build_observation_layout(int queue_size, bool include_hidden_rows) {
    ObservationLayout layout;
    layout.rows_encoded = include_hidden_rows ? Board::kRows : Board::kVisibleRows;
    std::size_t plane_size =
        static_cast<std::size_t>(layout.rows_encoded * Board::kWidth);

    layout.board_offset = 0;
    layout.active_piece_offset = layout.board_offset + plane_size;
    layout.hold_offset = layout.active_piece_offset + kActivePieceClasses;
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

    obs[layout.active_piece_offset + piece_index(state.active.piece)] = 1.0f;

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
    const int bounded_combo = std::clamp(state.combo, -1, 9);
    obs[layout.meta_offset + 1] = static_cast<float>(bounded_combo + 1) / 10.0f;
    obs[layout.meta_offset + 2] = state.back_to_back ? 1.0f : 0.0f;

    return obs;
}

}  // namespace tetris_v2
