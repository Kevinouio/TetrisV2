#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "cold_clear_cpp/bot.hpp"
#include "cold_clear_cpp/tbp.hpp"
#include "nlohmann/json.hpp"

namespace {

struct CliOptions {
    bool profile{false};
    std::optional<std::string> config_path;
};

CliOptions parse_cli(int argc, char** argv) {
    CliOptions opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--profile") {
            opts.profile = true;
        } else if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
            opts.config_path = argv[++i];
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    return opts;
}

cold_clear_cpp::Weights parse_weights(const nlohmann::json& j) {
    cold_clear_cpp::Weights w;
    w.cell_coveredness = j.at("cell_coveredness").get<float>();
    w.max_cell_covered_height = j.at("max_cell_covered_height").get<std::uint32_t>();
    w.holes = j.at("holes").get<float>();
    w.row_transitions = j.at("row_transitions").get<float>();
    w.height = j.at("height").get<float>();
    w.height_upper_half = j.at("height_upper_half").get<float>();
    w.height_upper_quarter = j.at("height_upper_quarter").get<float>();
    w.tetris_well_depth = j.at("tetris_well_depth").get<float>();
    w.tslot = j.at("tslot").get<std::array<float, 4>>();

    w.has_back_to_back = j.at("has_back_to_back").get<float>();
    w.wasted_t = j.at("wasted_t").get<float>();
    w.softdrop = j.at("softdrop").get<float>();

    w.normal_clears = j.at("normal_clears").get<std::array<float, 5>>();
    w.mini_spin_clears = j.at("mini_spin_clears").get<std::array<float, 3>>();
    w.spin_clears = j.at("spin_clears").get<std::array<float, 4>>();
    w.back_to_back_clear = j.at("back_to_back_clear").get<float>();
    w.combo_attack = j.at("combo_attack").get<float>();
    w.perfect_clear = j.at("perfect_clear").get<float>();
    w.perfect_clear_override = j.at("perfect_clear_override").get<bool>();
    return w;
}

cold_clear_cpp::BotConfig load_config(const std::optional<std::string>& path) {
    if (!path.has_value()) {
        return cold_clear_cpp::BotConfig{};
    }
    std::ifstream in(*path);
    if (!in) {
        throw std::runtime_error("Failed to open config file: " + *path);
    }
    nlohmann::json j;
    in >> j;
    cold_clear_cpp::BotConfig cfg;
    if (j.contains("freestyle_weights")) {
        cfg.freestyle_weights = parse_weights(j.at("freestyle_weights"));
    }
    if (j.contains("freestyle_exploitation")) {
        cfg.freestyle_exploitation = j.at("freestyle_exploitation").get<double>();
    }
    return cfg;
}

}  // namespace

int main(int argc, char** argv) {
    auto options = parse_cli(argc, argv);
    auto config = std::make_shared<cold_clear_cpp::BotConfig>(load_config(options.config_path));

#ifdef PROFILE_COLD_CLEAR
    if (options.profile) {
        // Placeholder: hook in a profiling backend here if desired.
    }
#else
    (void)options.profile;
#endif

    cold_clear_cpp::run_tbp(std::cin, std::cout, config);
    return 0;
}
