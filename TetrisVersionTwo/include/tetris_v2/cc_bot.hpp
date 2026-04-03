#pragma once

#include "tetris_v2/cc_env.hpp"
#include "tetris_v2/cold_clear_bot.hpp"

namespace tetris_v2::cc {

using BotChoice = ColdClearBotChoice;
using BotThinkStats = ColdClearBotThinkStats;

class Bot {
public:
    Bot() = default;

    bool sync_from_env(const Env& env) { return bot_.sync_from_env(env.raw()); }
    bool choose(int think_ms, BotChoice* choice_out = nullptr, BotThinkStats* stats_out = nullptr) {
        return bot_.choose(think_ms, choice_out, stats_out);
    }
    bool apply_last_choice(
        Env& env,
        StepResult* result_out = nullptr,
        int* used_hold_out = nullptr,
        std::size_t* placement_index_out = nullptr) {
        return bot_.apply_last_choice(env.raw(), result_out, used_hold_out, placement_index_out);
    }
    bool choose_and_apply(
        Env& env,
        int think_ms,
        StepResult* result_out = nullptr,
        BotChoice* choice_out = nullptr,
        BotThinkStats* stats_out = nullptr,
        int* used_hold_out = nullptr,
        std::size_t* placement_index_out = nullptr) {
        return bot_.choose_and_apply(
            env.raw(),
            think_ms,
            result_out,
            choice_out,
            stats_out,
            used_hold_out,
            placement_index_out);
    }

private:
    ColdClearBot bot_{};
};

}  // namespace tetris_v2::cc

