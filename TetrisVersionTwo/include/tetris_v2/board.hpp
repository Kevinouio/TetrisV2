#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace tetris_v2 {

class Board {
public:
    static constexpr int kWidth = 10;
    static constexpr int kRows = 40;
    static constexpr int kVisibleRows = 20;
    static constexpr std::uint16_t kFullRowMask = (1u << kWidth) - 1u;

    using RowMask = std::uint16_t;

    void clear();
    bool occupied(int x, int y) const;
    void set_cell(int x, int y, bool value = true);
    int clear_filled_lines();
    bool is_empty() const;

    RowMask row_mask(int y) const;
    const std::array<RowMask, kRows>& rows() const { return rows_; }

    std::vector<std::string> render_rows(int visible_rows = kVisibleRows) const;

private:
    std::array<RowMask, kRows> rows_{};
};

}  // namespace tetris_v2
