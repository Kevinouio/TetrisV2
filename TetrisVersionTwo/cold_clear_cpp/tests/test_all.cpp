#include <cassert>
#include <iostream>
#include <vector>

#include "cold_clear_cpp/bot.hpp"
#include "cold_clear_cpp/movegen.hpp"

using namespace cold_clear_cpp;

void test_line_clear() {
    Board board;
    for (auto& col : board.cols) {
        col |= 1ull;  // fill row 0
    }
    auto mask = board.line_clears();
    assert((mask & 1ull) != 0);
    board.remove_lines(mask);
    for (auto col : board.cols) {
        assert(col == 0);
    }
}

void test_move_generation() {
    Board board;
    auto moves = find_moves(board, Piece::T);
    assert(!moves.empty());
    bool grounded = false;
    for (const auto& entry : moves) {
        for (auto cell : entry.first.location.cells()) {
            assert(!board.occupied(cell));
            if (cell.second == 0) {
                grounded = true;
            }
        }
    }
    assert(grounded);
}

void test_eval_penalizes_holes() {
    PlacementInfo info{};
    info.placement.location.piece = Piece::T;
    info.perfect_clear = false;

    GameState clean;
    auto clean_eval = evaluate(default_weights(), clean, info, 0).first.value;

    GameState holes = clean;
    holes.board.cols[0] |= (1ull << 0) | (1ull << 3);  // holes at y=1 and y=2
    auto hole_eval = evaluate(default_weights(), holes, info, 0).first.value;

    assert(hole_eval < clean_eval);
}

void test_bot_suggests_move() {
    GameState root;
    root.reserve = Piece::I;
    std::vector<Piece> queue{Piece::I, Piece::O};
    BotOptions options;
    options.speculate = true;

    Bot bot(options, root, queue);
    auto stats = bot.do_work();
    assert(stats.selections >= 1);
    auto suggestions = bot.suggest();
    assert(!suggestions.empty());
}

int main() {
    test_line_clear();
    test_move_generation();
    test_eval_penalizes_holes();
    test_bot_suggests_move();
    std::cout << "All tests passed\n";
    return 0;
}
