#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <thread>

#include "tetris_v2/cc_bot.hpp"
#include "tetris_v2/cc_env.hpp"
#include "tetris_v2/types.hpp"

namespace {

struct Options {
    std::uint32_t seed{1234};
    int steps{500};
    int think_ms{20};
    int delay_ms{40};
    bool auto_reset{false};
};

int parse_int(const char* s, int fallback) {
    if (!s) {
        return fallback;
    }
    char* end = nullptr;
    long value = std::strtol(s, &end, 10);
    if (end == s || *end != '\0') {
        return fallback;
    }
    return static_cast<int>(value);
}

std::uint32_t parse_u32(const char* s, std::uint32_t fallback) {
    if (!s) {
        return fallback;
    }
    char* end = nullptr;
    unsigned long value = std::strtoul(s, &end, 10);
    if (end == s || *end != '\0') {
        return fallback;
    }
    return static_cast<std::uint32_t>(value);
}

Options parse_args(int argc, char** argv) {
    Options options{};
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--seed" && i + 1 < argc) {
            options.seed = parse_u32(argv[++i], options.seed);
        } else if (arg == "--steps" && i + 1 < argc) {
            options.steps = parse_int(argv[++i], options.steps);
        } else if (arg == "--think-ms" && i + 1 < argc) {
            options.think_ms = parse_int(argv[++i], options.think_ms);
        } else if (arg == "--delay-ms" && i + 1 < argc) {
            options.delay_ms = parse_int(argv[++i], options.delay_ms);
        } else if (arg == "--auto-reset") {
            options.auto_reset = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "tetris_bot_cli options:\n"
                << "  --seed <u32>       Initial seed (default 1234)\n"
                << "  --steps <int>      Max moves to run (default 500)\n"
                << "  --think-ms <int>   Bot think budget per move (default 20)\n"
                << "  --delay-ms <int>   Frame delay in ms (default 40)\n"
                << "  --auto-reset       Continue after topout with next seed\n";
            std::exit(0);
        }
    }
    options.steps = std::max(1, options.steps);
    options.think_ms = std::max(1, options.think_ms);
    options.delay_ms = std::max(0, options.delay_ms);
    return options;
}

void clear_terminal() {
    std::cout << "\x1b[2J\x1b[H";
}

}  // namespace

int main(int argc, char** argv) {
    const auto options = parse_args(argc, argv);

    tetris_v2::EnvConfig config;
    config.seed = options.seed;
    tetris_v2::cc::Env env(config);
    tetris_v2::cc::Bot bot;
    bot.sync_from_env(env);

    std::uint32_t seed = options.seed;
    int pieces = 0;
    int total_lines = 0;
    int topouts = 0;
    auto started = std::chrono::steady_clock::now();

    for (int step = 0; step < options.steps; ++step) {
        if (env.state().game_over) {
            ++topouts;
            if (!options.auto_reset) {
                break;
            }
            ++seed;
            env.reset(seed);
            bot.sync_from_env(env);
        }

        tetris_v2::StepResult move_result{};
        tetris_v2::cc::BotChoice choice{};
        tetris_v2::cc::BotThinkStats stats{};
        int used_hold = 0;
        std::size_t placement_index = 0;

        const bool ok = bot.choose_and_apply(
            env,
            options.think_ms,
            &move_result,
            &choice,
            &stats,
            &used_hold,
            &placement_index);

        if (!ok) {
            std::cerr << "Bot choose_and_apply failed at step " << step << '\n';
            return 1;
        }

        ++pieces;
        total_lines += move_result.lines_cleared;
        const auto now = std::chrono::steady_clock::now();
        const double elapsed_s =
            std::max(1e-9, std::chrono::duration<double>(now - started).count());
        const double pps = static_cast<double>(pieces) / elapsed_s;

        clear_terminal();
        std::cout << "C++ Cold Clear Bot Viewer\n";
        std::cout << "seed=" << seed << " step=" << (step + 1) << "/" << options.steps
                  << " pieces=" << pieces << " lines=" << total_lines
                  << " topouts=" << topouts << " pps=" << pps << '\n';
        std::cout << "choice: hold=" << used_hold << " idx=" << placement_index
                  << " score=" << choice.score << '\n';
        std::cout << "think: ms=" << stats.think_ms << " nodes=" << stats.nodes
                  << " nps=" << stats.nps
                  << " budget_miss=" << stats.budget_miss << '\n';
        std::cout << env.raw().render_ascii();
        std::cout << std::flush;

        if (options.delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(options.delay_ms));
        }
    }

    std::cout << "\nDone.\n";
    return 0;
}
