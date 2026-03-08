#include <iostream>
#include <string>
#include <vector>

#include "tetris_v2/env.hpp"
#include "tetris_v2/observation.hpp"
#include "tetris_v2/types.hpp"

namespace {

const char* rotation_name(tetris_v2::Rotation rotation) {
    switch (rotation) {
        case tetris_v2::Rotation::North: return "N";
        case tetris_v2::Rotation::East: return "E";
        case tetris_v2::Rotation::South: return "S";
        case tetris_v2::Rotation::West: return "W";
    }
    return "?";
}

bool try_parse_index(const std::string& s, std::size_t* out) {
    try {
        std::size_t consumed = 0;
        auto value = std::stoul(s, &consumed);
        if (consumed != s.size()) {
            return false;
        }
        *out = value;
        return true;
    } catch (...) {
        return false;
    }
}

void print_help() {
    std::cout << "Placement Search Mode\n";
    std::cout << "  Enter a number to place that lock-state index\n";
    std::cout << "  p = print all resulting boards\n";
    std::cout << "  h = use hold\n";
    std::cout << "  q = quit\n";
}

}  // namespace

int main() {
    tetris_v2::EnvConfig config;
    config.seed = 1234;
    config.gravity_per_step = 1.0f / 8.0f;
    config.lock_delay_steps = 24;
    tetris_v2::ModernTetrisEnv env(config);

    print_help();
    while (!env.state().game_over) {
        auto options = env.enumerate_active_piece_placements();
        std::cout << "\n----------------------\n";
        std::cout << env.render_ascii();
        std::cout << "Active: " << tetris_v2::piece_name(env.state().active.piece)
                  << " @ (" << env.state().active.x << "," << env.state().active.y << ") "
                  << rotation_name(env.state().active.rotation) << '\n';
        std::cout << "Hold: "
                  << (env.state().hold.has_value()
                          ? tetris_v2::piece_name(*env.state().hold)
                          : "None")
                  << " | Hold Available: " << (env.state().hold_available ? "yes" : "no") << '\n';

        std::cout << "Queue: ";
        int shown = 0;
        for (auto p : env.state().queue) {
            std::cout << tetris_v2::piece_name(p) << ' ';
            if (++shown >= config.queue_size) {
                break;
            }
        }
        std::cout << '\n';

        auto obs_size = tetris_v2::observation_size(config.queue_size, true);
        std::cout << "Observation size (hidden rows): " << obs_size << '\n';
        std::cout << "Lock states: " << options.size() << '\n';
        for (std::size_t i = 0; i < options.size(); ++i) {
            const auto& opt = options[i];
            std::cout << "  [" << i << "] x=" << opt.placement.x << " y=" << opt.placement.y
                      << " rot=" << rotation_name(opt.placement.rotation)
                      << " lines=" << opt.lines_cleared << '\n';
        }
        if (options.empty()) {
            std::cout << "No lock placements available. Game over.\n";
            break;
        }

        std::cout << "Command> ";
        std::string line;
        if (!std::getline(std::cin, line)) {
            break;
        }
        if (line.empty()) {
            continue;
        }
        if (line == "q") {
            break;
        }
        if (line == "h") {
            auto result = env.step(tetris_v2::Action::Hold);
            std::cout << "hold=" << result.hold_used << " game_over=" << result.game_over << '\n';
            continue;
        }
        if (line == "p") {
            for (std::size_t i = 0; i < options.size(); ++i) {
                const auto& opt = options[i];
                std::cout << "\n== Placement " << i << " == x=" << opt.placement.x
                          << " y=" << opt.placement.y
                          << " rot=" << rotation_name(opt.placement.rotation)
                          << " lines=" << opt.lines_cleared << '\n';
                auto rows = opt.board_after_lock.render_rows();
                for (const auto& row : rows) {
                    std::cout << row << '\n';
                }
            }
            continue;
        }

        std::size_t choice = 0;
        if (!try_parse_index(line, &choice) || choice >= options.size()) {
            std::cout << "Invalid command/index.\n";
            continue;
        }

        auto result = env.apply_placement(options[choice].placement);
        std::cout << "reward=" << result.reward << " lines=" << result.lines_cleared
                  << " combo=" << result.combo << " b2b=" << result.back_to_back
                  << " locked=" << result.piece_locked << '\n';
    }

    std::cout << "\nGame over\n";
    return 0;
}
