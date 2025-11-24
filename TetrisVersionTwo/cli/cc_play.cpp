#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "tetris_v2/cc_agent.hpp"
#include "tetris_v2/tetris_env.hpp"

using cold_clear_cpp::Placement;
using cold_clear_cpp::Piece;
using cold_clear_cpp::Spin;

namespace {

struct Options {
    bool json{false};
    int steps{200};
    int delay_ms{120};
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--json") {
            o.json = true;
        } else if (arg == "--steps" && i + 1 < argc) {
            o.steps = std::atoi(argv[++i]);
        } else if (arg == "--delay-ms" && i + 1 < argc) {
            o.delay_ms = std::atoi(argv[++i]);
        }
    }
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace std::chrono_literals;

    auto opts = parse(argc, argv);

    tetris_v2::TetrisEnv env;
    env.ensure_queue(8);

    cold_clear_cpp::BotOptions options;
    options.speculate = true;
    options.config = std::make_shared<cold_clear_cpp::BotConfig>();

    tetris_v2::CCAgent agent(options, env);

    for (int step = 0; step < opts.steps && !env.game_over(); ++step) {
        // Keep the bot's queue in sync with the environment.
        auto added = env.ensure_queue(8);
        agent.on_new_pieces(added);

        auto mv = agent.choose_move(1024);
        if (!mv) {
            std::cout << "No available moves; game over.\n";
            break;
        }

        auto used_piece = mv->location.piece;
        auto info = env.apply(*mv);
        agent.on_advance(*mv);

        if (opts.json) {
            auto rows = env.display_rows();
            std::cout << "{\"step\":" << step << ",\"piece\":\"" << tetris_v2::TetrisEnv::piece_label(used_piece)
                      << "\",\"lines\":" << info.lines_cleared << ",\"spin\":" << static_cast<int>(info.placement.spin)
                      << ",\"combo\":" << info.combo << ",\"b2b\":" << (info.back_to_back ? "true" : "false")
                      << ",\"hold\":\"" << tetris_v2::TetrisEnv::piece_label(env.hold_piece()) << "\",\"queue\":[";
            const auto& q = env.queue();
            for (std::size_t i = 0; i < q.size(); ++i) {
                if (i) std::cout << ',';
                std::cout << "\"" << tetris_v2::TetrisEnv::piece_label(q[i]) << "\"";
            }
            std::cout << "],\"board\":[";
            for (std::size_t i = 0; i < rows.size(); ++i) {
                if (i) std::cout << ',';
                std::cout << "\"" << rows[i] << "\"";
            }
            std::cout << "]}\n" << std::flush;
        } else {
            std::cout << "Step " << step << " | piece " << static_cast<int>(used_piece)
                      << " | lines=" << info.lines_cleared << " spin=" << static_cast<int>(info.placement.spin)
                      << " combo=" << info.combo << " b2b=" << info.back_to_back << " hold="
                      << tetris_v2::TetrisEnv::piece_label(env.hold_piece()) << "\n";
            std::cout << env.render() << std::flush;
        }

        // After consuming the current piece, append more and inform the agent.
        auto appended_after = env.ensure_queue(8);
        agent.on_new_pieces(appended_after);

        std::this_thread::sleep_for(std::chrono::milliseconds(opts.delay_ms));
    }

    return 0;
}
