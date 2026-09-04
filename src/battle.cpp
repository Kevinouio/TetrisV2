#include "tetris_v2/battle.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <utility>

#include "tetris_v2/decision.hpp"
#include "tetris_v2/observation.hpp"

namespace tetris_v2 {

namespace {

std::size_t other_player(std::size_t player) { return 1u - player; }

float unit_value(float value) { return std::clamp(value, 0.0f, 1.0f); }

struct PublicBoardStats {
    int aggregate_height{0};
    int max_height{0};
    int holes{0};
    int bumpiness{0};
    int wells{0};
};

PublicBoardStats public_board_stats(const Board& board) {
    std::array<int, Board::kWidth> heights{};
    PublicBoardStats stats{};
    for (int x = 0; x < Board::kWidth; ++x) {
        for (int y = Board::kVisibleRows - 1; y >= 0; --y) {
            if (board.occupied(x, y)) {
                heights[static_cast<std::size_t>(x)] = y + 1;
                break;
            }
        }
        const int height = heights[static_cast<std::size_t>(x)];
        stats.aggregate_height += height;
        stats.max_height = std::max(stats.max_height, height);
        for (int y = 0; y < height; ++y) {
            if (!board.occupied(x, y)) {
                ++stats.holes;
            }
        }
    }
    for (int x = 0; x < Board::kWidth; ++x) {
        const int height = heights[static_cast<std::size_t>(x)];
        if (x + 1 < Board::kWidth) {
            stats.bumpiness += std::abs(
                height - heights[static_cast<std::size_t>(x + 1)]);
        }
        const int left =
            x == 0 ? Board::kVisibleRows : heights[static_cast<std::size_t>(x - 1)];
        const int right = x + 1 == Board::kWidth
                              ? Board::kVisibleRows
                              : heights[static_cast<std::size_t>(x + 1)];
        const int depth = std::max(0, std::min(left, right) - height);
        stats.wells += depth * (depth + 1) / 2;
    }
    return stats;
}

}  // namespace

BattleConfig BattleEnv::validate_config(BattleConfig config) {
    if (config.garbage_delay < 0) {
        throw std::invalid_argument("garbage_delay must be nonnegative.");
    }
    if (config.max_joint_steps < 0) {
        throw std::invalid_argument("max_joint_steps must be nonnegative.");
    }
    for (const int attack : config.attack_table) {
        if (attack < 0) {
            throw std::invalid_argument("attack_table entries must be nonnegative.");
        }
    }
    return config;
}

std::uint32_t BattleEnv::mixed_seed(std::uint32_t seed, std::uint32_t stream) {
    std::uint32_t value = seed + 0x9e3779b9u * (stream + 1u);
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

EnvConfig BattleEnv::player_config(std::uint32_t seed) {
    EnvConfig config{};
    config.seed = seed;
    return config;
}

BattleEnv::BattleEnv(const BattleConfig& config)
    : config_(validate_config(config)),
      seed_(config_.seed),
      players_{
          cc::Env(player_config(seed_)),
          cc::Env(player_config(
              config_.same_piece_sequence ? seed_ : mixed_seed(seed_, 0u)))},
      garbage_rng_{
          std::mt19937(mixed_seed(seed_, 1u)),
          std::mt19937(mixed_seed(seed_, 2u))} {
    reset(seed_);
}

void BattleEnv::reset(std::optional<std::uint32_t> seed) {
    seed_ = seed.value_or(config_.seed);
    players_[0].reset(seed_);
    players_[1].reset(
        config_.same_piece_sequence ? seed_ : mixed_seed(seed_, 0u));
    incoming_ = {};
    stats_ = {};
    garbage_rng_[0].seed(mixed_seed(seed_, 1u));
    garbage_rng_[1].seed(mixed_seed(seed_, 2u));
    joint_steps_ = 0;
    terminated_ = false;
    winner_ = -1;
}

int BattleEnv::attack_for_lines(int lines) const {
    const auto index = static_cast<std::size_t>(std::clamp(lines, 0, 4));
    return config_.attack_table[index];
}

int BattleEnv::random_hole(std::size_t recipient) {
    std::uniform_int_distribution<int> distribution(0, Board::kWidth - 1);
    return distribution(garbage_rng_[recipient]);
}

int BattleEnv::pending_garbage(std::size_t player) const {
    if (player >= kBattlePlayerCount) {
        throw std::out_of_range("battle player index");
    }
    int total = 0;
    for (const auto& packet : incoming_[player]) {
        total += static_cast<int>(packet.hole_columns.size());
    }
    return total;
}

int BattleEnv::next_garbage_delay(std::size_t player) const {
    if (player >= kBattlePlayerCount) {
        throw std::out_of_range("battle player index");
    }
    if (incoming_[player].empty()) {
        return -1;
    }
    const auto next = std::min_element(
        incoming_[player].begin(), incoming_[player].end(),
        [](const GarbagePacket& lhs, const GarbagePacket& rhs) {
            return lhs.due_step < rhs.due_step;
        });
    return std::max(0, next->due_step - joint_steps_);
}

int BattleEnv::cancel_pending(std::size_t player, int attack) {
    int cancelled = 0;
    while (attack > 0 && !incoming_[player].empty()) {
        auto& packet = incoming_[player].front();
        while (attack > 0 && !packet.hole_columns.empty()) {
            packet.hole_columns.pop_front();
            --attack;
            ++cancelled;
        }
        if (packet.hole_columns.empty()) {
            incoming_[player].pop_front();
        }
    }
    return cancelled;
}

void BattleEnv::enqueue_generated(std::size_t recipient, int lines, int delay) {
    if (lines <= 0) {
        return;
    }
    GarbagePacket packet{};
    packet.due_step = joint_steps_ + delay;
    for (int line = 0; line < lines; ++line) {
        packet.hole_columns.push_back(random_hole(recipient));
    }
    incoming_[recipient].push_back(std::move(packet));
}

bool BattleEnv::enqueue_garbage(
    std::size_t player,
    const std::vector<int>& hole_columns,
    int delay) {
    if (player >= kBattlePlayerCount || delay < 0 || terminated_) {
        return false;
    }
    for (const int hole : hole_columns) {
        if (hole < 0 || hole >= Board::kWidth) {
            return false;
        }
    }
    if (hole_columns.empty()) {
        return true;
    }
    GarbagePacket packet{};
    packet.due_step = joint_steps_ + 1 + delay;
    packet.hole_columns.assign(hole_columns.begin(), hole_columns.end());
    incoming_[player].push_back(std::move(packet));
    stats_[player].garbage_received += static_cast<int>(hole_columns.size());
    return true;
}

int BattleEnv::apply_due_garbage(std::size_t player) {
    if (players_[player].state().game_over) {
        return 0;
    }
    std::vector<int> holes;
    auto packet = incoming_[player].begin();
    while (packet != incoming_[player].end()) {
        if (packet->due_step > joint_steps_) {
            ++packet;
            continue;
        }
        holes.insert(
            holes.end(), packet->hole_columns.begin(), packet->hole_columns.end());
        packet = incoming_[player].erase(packet);
    }
    if (holes.empty()) {
        return 0;
    }
    const auto result = players_[player].insert_garbage_rows(holes);
    return result.rows_applied;
}

void BattleEnv::finish_if_terminal(BattleStepResult& result) {
    const bool player0_topout = players_[0].state().top_out;
    const bool player1_topout = players_[1].state().top_out;
    if (player0_topout || player1_topout) {
        terminated_ = true;
        winner_ = player0_topout == player1_topout ? -1 : (player0_topout ? 1 : 0);
    } else if (config_.max_joint_steps > 0 &&
               joint_steps_ >= config_.max_joint_steps) {
        terminated_ = true;
        winner_ = -1;
    }
    result.terminated = terminated_;
    result.winner = winner_;
}

BattleStepResult BattleEnv::step(
    std::size_t player0_action, std::size_t player1_action) {
    BattleStepResult result{};
    result.joint_step = joint_steps_;
    result.terminated = terminated_;
    result.winner = winner_;
    if (terminated_ ||
        !stable_decision_at_action(players_[0], player0_action).has_value() ||
        !stable_decision_at_action(players_[1], player1_action).has_value()) {
        return result;
    }

    const auto before = snapshot();
    try {
        std::array<StepResult, kBattlePlayerCount> placement_results{};
        std::array<int, kBattlePlayerCount> used_hold{};
        std::array<std::size_t, kBattlePlayerCount> placement_indices{};
        if (!apply_stable_decision(
                players_[0], player0_action, &placement_results[0],
                &used_hold[0], &placement_indices[0]) ||
            !apply_stable_decision(
                players_[1], player1_action, &placement_results[1],
                &used_hold[1], &placement_indices[1])) {
            restore(before);
            return result;
        }

        ++joint_steps_;
        result.joint_step = joint_steps_;
        for (std::size_t player = 0; player < kBattlePlayerCount; ++player) {
            auto& step_player = result.players[player];
            auto& cumulative = stats_[player];
            step_player.action_succeeded = true;
            step_player.used_hold = used_hold[player] != 0;
            step_player.placement_index = placement_indices[player];
            step_player.reward = placement_results[player].reward;
            step_player.lines_cleared = placement_results[player].lines_cleared;
            step_player.attack_generated = attack_for_lines(step_player.lines_cleared);
            ++cumulative.placements;
            cumulative.score += step_player.reward;
            cumulative.lines_cleared += step_player.lines_cleared;
            cumulative.attack_generated += step_player.attack_generated;
        }

        for (std::size_t player = 0; player < kBattlePlayerCount; ++player) {
            auto& step_player = result.players[player];
            step_player.garbage_cancelled =
                cancel_pending(player, step_player.attack_generated);
            step_player.garbage_sent =
                step_player.attack_generated - step_player.garbage_cancelled;
            stats_[player].garbage_cancelled += step_player.garbage_cancelled;
            stats_[player].garbage_sent += step_player.garbage_sent;
        }

        for (std::size_t sender = 0; sender < kBattlePlayerCount; ++sender) {
            const auto recipient = other_player(sender);
            const int sent = result.players[sender].garbage_sent;
            enqueue_generated(recipient, sent, config_.garbage_delay);
            result.players[recipient].garbage_received += sent;
            stats_[recipient].garbage_received += sent;
        }

        for (std::size_t player = 0; player < kBattlePlayerCount; ++player) {
            auto& step_player = result.players[player];
            step_player.garbage_applied = apply_due_garbage(player);
            stats_[player].garbage_applied += step_player.garbage_applied;
            step_player.incoming_garbage = pending_garbage(player);
            step_player.next_garbage_delay = next_garbage_delay(player);
            step_player.top_out = players_[player].state().top_out;
            if (step_player.top_out) {
                stats_[player].top_outs = 1;
            }
        }

        result.success = true;
        finish_if_terminal(result);
        return result;
    } catch (...) {
        restore(before);
        throw;
    }
}

std::vector<float> BattleEnv::observation(std::size_t perspective_player) const {
    if (perspective_player >= kBattlePlayerCount) {
        throw std::out_of_range("battle player index");
    }
    const auto opponent = other_player(perspective_player);
    auto out = encode_observation(
        players_[perspective_player].state(), 5, false);
    if (out.size() != kBattleOwnObservationSize) {
        throw std::runtime_error("Unexpected single-player observation layout.");
    }
    out.reserve(kBattleObservationSize);
    const auto& opponent_board = players_[opponent].state().board;
    for (int y = 0; y < Board::kVisibleRows; ++y) {
        const auto row = opponent_board.row_mask(y);
        for (int x = 0; x < Board::kWidth; ++x) {
            out.push_back((row & (Board::RowMask{1u} << x)) != 0 ? 1.0f : 0.0f);
        }
    }

    const int own_pending = pending_garbage(perspective_player);
    const int opponent_pending = pending_garbage(opponent);
    const int own_delay = next_garbage_delay(perspective_player);
    const int opponent_delay = next_garbage_delay(opponent);
    const auto own_stats =
        public_board_stats(players_[perspective_player].state().board);
    const auto opponent_stats = public_board_stats(opponent_board);
    const float delay_scale = static_cast<float>(std::max(1, config_.garbage_delay));
    out.push_back(unit_value(static_cast<float>(own_pending) / Board::kRows));
    out.push_back(own_delay < 0 ? 0.0f : unit_value(static_cast<float>(own_delay) / delay_scale));
    out.push_back(unit_value(static_cast<float>(opponent_pending) / Board::kRows));
    out.push_back(
        opponent_delay < 0 ? 0.0f
                           : unit_value(static_cast<float>(opponent_delay) / delay_scale));
    out.push_back(unit_value(static_cast<float>(own_stats.aggregate_height) / 200.0f));
    out.push_back(unit_value(static_cast<float>(own_stats.max_height) / 20.0f));
    out.push_back(unit_value(static_cast<float>(own_stats.holes) / 200.0f));
    out.push_back(unit_value(static_cast<float>(own_stats.bumpiness) / 180.0f));
    out.push_back(unit_value(static_cast<float>(own_stats.wells) / 420.0f));
    out.push_back(unit_value(
        static_cast<float>(opponent_stats.aggregate_height) / 200.0f));
    out.push_back(unit_value(static_cast<float>(opponent_stats.max_height) / 20.0f));
    out.push_back(unit_value(static_cast<float>(opponent_stats.holes) / 200.0f));
    out.push_back(unit_value(static_cast<float>(opponent_stats.bumpiness) / 180.0f));
    out.push_back(unit_value(static_cast<float>(opponent_stats.wells) / 420.0f));
    out.push_back(unit_value(
        static_cast<float>(opponent_stats.max_height - own_stats.max_height + 20) /
        40.0f));
    out.push_back(unit_value(
        static_cast<float>(opponent_stats.holes - own_stats.holes + 200) /
        400.0f));
    return out;
}

BattleSnapshot BattleEnv::snapshot() const {
    BattleSnapshot out{};
    out.players = {players_[0].snapshot(), players_[1].snapshot()};
    out.incoming = incoming_;
    out.garbage_rng = garbage_rng_;
    out.stats = stats_;
    out.joint_steps = joint_steps_;
    out.terminated = terminated_;
    out.winner = winner_;
    return out;
}

void BattleEnv::restore(const BattleSnapshot& snapshot) {
    players_[0].restore(snapshot.players[0]);
    players_[1].restore(snapshot.players[1]);
    incoming_ = snapshot.incoming;
    garbage_rng_ = snapshot.garbage_rng;
    stats_ = snapshot.stats;
    joint_steps_ = snapshot.joint_steps;
    terminated_ = snapshot.terminated;
    winner_ = snapshot.winner;
}

const BattlePlayerStats& BattleEnv::stats(std::size_t player) const {
    if (player >= kBattlePlayerCount) {
        throw std::out_of_range("battle player index");
    }
    return stats_[player];
}

cc::Env& BattleEnv::player_env(std::size_t player) {
    if (player >= kBattlePlayerCount) {
        throw std::out_of_range("battle player index");
    }
    return players_[player];
}

const cc::Env& BattleEnv::player_env(std::size_t player) const {
    if (player >= kBattlePlayerCount) {
        throw std::out_of_range("battle player index");
    }
    return players_[player];
}

}  // namespace tetris_v2
