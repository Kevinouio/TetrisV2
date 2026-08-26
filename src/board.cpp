#include "tetris_v2/board.hpp"

#include <algorithm>

namespace tetris_v2 {

void Board::clear() { rows_.fill(0); }

bool Board::occupied(int x, int y) const {
    if (x < 0 || x >= kWidth || y < 0 || y >= kRows) {
        return true;
    }
    return (rows_[static_cast<std::size_t>(y)] & (RowMask{1u} << x)) != 0;
}

void Board::set_cell(int x, int y, bool value) {
    if (x < 0 || x >= kWidth || y < 0 || y >= kRows) {
        return;
    }
    auto bit = static_cast<RowMask>(RowMask{1u} << x);
    auto& row = rows_[static_cast<std::size_t>(y)];
    if (value) {
        row = static_cast<RowMask>(row | bit);
    } else {
        row = static_cast<RowMask>(row & static_cast<RowMask>(~bit));
    }
}

int Board::clear_filled_lines() {
    std::array<RowMask, kRows> compacted{};
    int dst = 0;
    int cleared = 0;
    for (int src = 0; src < kRows; ++src) {
        if ((rows_[static_cast<std::size_t>(src)] & kFullRowMask) == kFullRowMask) {
            ++cleared;
            continue;
        }
        compacted[static_cast<std::size_t>(dst++)] = rows_[static_cast<std::size_t>(src)];
    }
    while (dst < kRows) {
        compacted[static_cast<std::size_t>(dst++)] = 0;
    }
    rows_ = compacted;
    return cleared;
}

bool Board::is_empty() const {
    return std::all_of(rows_.begin(), rows_.end(), [](RowMask row) { return row == 0; });
}

Board::RowMask Board::row_mask(int y) const {
    if (y < 0 || y >= kRows) {
        return 0;
    }
    return rows_[static_cast<std::size_t>(y)];
}

std::vector<std::string> Board::render_rows(int visible_rows) const {
    visible_rows = std::max(1, std::min(visible_rows, kRows));
    std::vector<std::string> out;
    out.reserve(static_cast<std::size_t>(visible_rows));
    for (int y = visible_rows - 1; y >= 0; --y) {
        std::string row;
        row.reserve(kWidth);
        auto mask = rows_[static_cast<std::size_t>(y)];
        for (int x = 0; x < kWidth; ++x) {
            bool filled = (mask & (RowMask{1u} << x)) != 0;
            row.push_back(filled ? '#' : '.');
        }
        out.push_back(std::move(row));
    }
    return out;
}

}  // namespace tetris_v2
