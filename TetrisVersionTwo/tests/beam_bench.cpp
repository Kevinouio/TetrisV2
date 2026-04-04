#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "tetris_v2/c_api.h"

namespace {

struct BenchConfig {
    std::vector<std::uint32_t> seeds{1234u};
    int steps{300};
    int think_ms{20};
    int depth{2};
    int beam_width{8};
    double gamma{1.0};
    std::string trace_out{};
};

void fail_usage(const std::string& msg) {
    if (!msg.empty()) {
        std::cerr << "error: " << msg << "\n";
    }
    std::cerr << "usage: tetris_beam_bench [--seed N] [--steps N] [--think-ms N] "
                 "[--depth N] [--beam-width N] [--gamma X] [--trace-out PATH]\n";
    std::exit(2);
}

BenchConfig parse_args(int argc, char** argv) {
    BenchConfig cfg{};
    cfg.seeds.clear();
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                fail_usage(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--seed") {
            cfg.seeds.push_back(static_cast<std::uint32_t>(std::strtoul(need_value("--seed"), nullptr, 10)));
        } else if (arg == "--steps") {
            cfg.steps = std::atoi(need_value("--steps"));
        } else if (arg == "--think-ms") {
            cfg.think_ms = std::atoi(need_value("--think-ms"));
        } else if (arg == "--depth") {
            cfg.depth = std::atoi(need_value("--depth"));
        } else if (arg == "--beam-width") {
            cfg.beam_width = std::atoi(need_value("--beam-width"));
        } else if (arg == "--gamma") {
            cfg.gamma = std::atof(need_value("--gamma"));
        } else if (arg == "--trace-out") {
            cfg.trace_out = need_value("--trace-out");
        } else {
            fail_usage(std::string("unknown argument: ") + arg);
        }
    }

    if (cfg.seeds.empty()) {
        cfg.seeds.push_back(1234u);
    }
    if (cfg.steps <= 0) {
        fail_usage("--steps must be > 0");
    }
    if (cfg.think_ms <= 0) {
        fail_usage("--think-ms must be > 0");
    }
    if (cfg.depth <= 0) {
        fail_usage("--depth must be > 0");
    }
    if (cfg.beam_width <= 0) {
        fail_usage("--beam-width must be > 0");
    }
    if (!std::isfinite(cfg.gamma) || cfg.gamma < 0.0) {
        fail_usage("--gamma must be finite and >= 0");
    }
    return cfg;
}

struct RunSummary {
    int steps{0};
    std::uint64_t nodes{0};
    double think_ms_sum{0.0};
    double wall_seconds{0.0};
};

RunSummary run_seed(const BenchConfig& cfg, std::uint32_t seed, std::ofstream* trace_out) {
    RunSummary summary{};

    tetris_cc_env_handle* env = tetris_cc_env_create(seed);
    tetris_cc_bot_handle* bot = tetris_cc_bot_create_default();
    if (!env || !bot) {
        std::cerr << "failed to create C API handles\n";
        std::exit(3);
    }

    tetris_cc_env_reset(env, seed);
    if (!tetris_cc_bot_set_backend(bot, TETRIS_CC_BOT_BACKEND_BEAM) ||
        !tetris_cc_bot_set_beam_config(
            bot, cfg.depth, cfg.beam_width, cfg.gamma, 1, 0, 1, 0) ||
        !tetris_cc_bot_sync_from_env(bot, env)) {
        std::cerr << "failed to configure beam backend\n";
        std::exit(3);
    }

    const auto wall_t0 = std::chrono::steady_clock::now();
    for (int step_idx = 0; step_idx < cfg.steps; ++step_idx) {
        float reward = 0.0f;
        int lines_cleared = 0;
        int game_over = 0;
        int used_hold = 0;
        std::size_t placement_index = 0;
        float score = 0.0f;
        std::uint64_t nodes = 0;
        double think_ms = 0.0;
        double nps = 0.0;
        int budget_miss = 0;

        const int ok = tetris_cc_bot_choose_and_apply_ex(
            bot,
            env,
            cfg.think_ms,
            &reward,
            &lines_cleared,
            &game_over,
            &used_hold,
            &placement_index,
            &score,
            &nodes,
            &think_ms,
            &nps,
            &budget_miss);
        if (!ok) {
            break;
        }

        ++summary.steps;
        summary.nodes += nodes;
        summary.think_ms_sum += think_ms;

        if (trace_out && trace_out->is_open()) {
            (*trace_out) << seed << " " << step_idx << " " << used_hold << " " << placement_index
                         << " " << std::setprecision(9) << score << "\n";
        }

        if (game_over) {
            break;
        }
    }
    const auto wall_t1 = std::chrono::steady_clock::now();
    summary.wall_seconds = std::chrono::duration<double>(wall_t1 - wall_t0).count();

    tetris_cc_bot_destroy(bot);
    tetris_cc_env_destroy(env);
    return summary;
}

}  // namespace

int main(int argc, char** argv) {
    const BenchConfig cfg = parse_args(argc, argv);

    std::ofstream trace_file{};
    if (!cfg.trace_out.empty()) {
        trace_file.open(cfg.trace_out, std::ios::out | std::ios::trunc);
        if (!trace_file.is_open()) {
            std::cerr << "failed to open trace file: " << cfg.trace_out << "\n";
            return 3;
        }
    }

    double avg_think_ms_sum = 0.0;
    double nodes_per_second_sum = 0.0;
    int run_count = 0;

    for (const auto seed : cfg.seeds) {
        const RunSummary run = run_seed(cfg, seed, trace_file.is_open() ? &trace_file : nullptr);
        const double avg_think_ms = (run.steps > 0) ? (run.think_ms_sum / run.steps) : 0.0;
        const double nodes_per_second =
            (run.wall_seconds > 0.0) ? (static_cast<double>(run.nodes) / run.wall_seconds) : 0.0;
        std::cout << "seed=" << seed << " steps=" << run.steps
                  << " avg_think_ms=" << std::fixed << std::setprecision(6) << avg_think_ms
                  << " nodes=" << run.nodes << " wall_s=" << run.wall_seconds
                  << " nodes_per_s=" << nodes_per_second << "\n";
        avg_think_ms_sum += avg_think_ms;
        nodes_per_second_sum += nodes_per_second;
        ++run_count;
    }

    if (run_count > 0) {
        std::cout << "mean_avg_think_ms=" << std::fixed << std::setprecision(6)
                  << (avg_think_ms_sum / run_count)
                  << " mean_nodes_per_s=" << (nodes_per_second_sum / run_count) << "\n";
    }

    return 0;
}
