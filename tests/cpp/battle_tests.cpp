#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <vector>

#include "tetris_v2/battle.hpp"
#include "tetris_v2/c_api.h"
#include "tetris_v2/cc_bot.hpp"
#include "tetris_v2/cold_clear_bot.hpp"
#include "tetris_v2/decision.hpp"
#include "tetris_v2/env.hpp"
#include "tetris_v2/piece_defs.hpp"

namespace {

using namespace tetris_v2;

EnvState clean_state(const cc::Env& env, Piece active_piece) {
    auto state = env.snapshot().state;
    state.board.clear();
    for (auto& row : state.piece_ids) {
        row.fill(kEmptyPieceId);
    }
    state.active = spawn_piece(active_piece);
    state.hold.reset();
    state.hold_available = true;
    state.queue = {Piece::I, Piece::O, Piece::T, Piece::L, Piece::J, Piece::S};
    state.game_over = false;
    state.top_out = false;
    state.combo = -1;
    state.back_to_back = false;
    state.total_lines_cleared = 0;
    state.lock_timer = 0;
    state.lock_resets_used = 0;
    state.gravity_accumulator = 0.0f;
    state.spin_eligible = false;
    state.last_rotate_used_kick = false;
    state.last_rotate_kick_index = -1;
    state.last_clear_spin = false;
    state.last_clear_spin_type = SpinType::None;
    state.last_clear_difficult = false;
    state.last_clear_b2b_bonus = false;
    return state;
}

void fill_row_with_gap(EnvState& state, int y, int gap_x, std::int8_t id = 0) {
    for (int x = 0; x < Board::kWidth; ++x) {
        if (x == gap_x) {
            continue;
        }
        state.board.set_cell(x, y);
        state.piece_ids[static_cast<std::size_t>(y)][static_cast<std::size_t>(x)] = id;
    }
}

std::size_t action_clearing(const cc::Env& env, int lines) {
    const auto decisions = enumerate_stable_decisions(env);
    const auto it = std::find_if(
        decisions.begin(), decisions.end(), [&](const StableDecision& decision) {
            if (decision.use_hold) {
                return false;
            }
            const auto option = env.placement_option_at(decision.placement_index);
            return option.has_value() && option->lines_cleared == lines;
        });
    assert(it != decisions.end());
    return it->action;
}

std::size_t first_legal_action(const cc::Env& env) {
    const auto decisions = enumerate_stable_decisions(env);
    assert(!decisions.empty());
    return decisions.front().action;
}

void prepare_tetris(BattleEnv& battle, std::size_t player) {
    auto state = clean_state(battle.player_env(player), Piece::I);
    for (int y = 0; y < 4; ++y) {
        fill_row_with_gap(state, y, 4);
    }
    battle.player_env(player).restore(state);
    assert(action_clearing(battle.player_env(player), 4) < kDecisionActionDim);
}

void prepare_empty(BattleEnv& battle, std::size_t player, Piece piece = Piece::O) {
    battle.player_env(player).restore(clean_state(battle.player_env(player), piece));
}

void assert_same_board(const cc::Env& lhs, const cc::Env& rhs) {
    assert(lhs.state().board.rows() == rhs.state().board.rows());
    assert(lhs.state().piece_ids == rhs.state().piece_ids);
    assert(lhs.state().active == rhs.state().active);
    assert(lhs.state().queue == rhs.state().queue);
    assert(lhs.state().hold == rhs.state().hold);
    assert(lhs.state().game_over == rhs.state().game_over);
    assert(lhs.state().top_out == rhs.state().top_out);
}

void test_engine_garbage_insertion_and_piece_ids() {
    cc::Env env(EnvConfig{11});
    auto state = clean_state(env, Piece::O);
    state.board.set_cell(6, 0);
    state.piece_ids[0][6] = static_cast<std::int8_t>(piece_index(Piece::T));
    env.restore(state);

    const auto result = env.insert_garbage_rows({1, 2});
    assert(result.rows_requested == 2);
    assert(result.rows_applied == 2);
    assert(!result.overflow);
    assert(!result.active_collision);
    assert(!result.top_out);
    assert(env.state().board.occupied(6, 2));
    assert(env.state().piece_ids[2][6] == static_cast<std::int8_t>(piece_index(Piece::T)));
    for (int x = 0; x < Board::kWidth; ++x) {
        assert(env.state().board.occupied(x, 0) == (x != 2));
        assert(env.state().board.occupied(x, 1) == (x != 1));
        if (x != 2) {
            assert(env.state().piece_ids[0][static_cast<std::size_t>(x)] == kGarbagePieceId);
        }
        if (x != 1) {
            assert(env.state().piece_ids[1][static_cast<std::size_t>(x)] == kGarbagePieceId);
        }
    }

    const auto visible_ids = env.visible_board_piece_ids(false);
    assert(visible_ids[19u * 10u] == kGarbagePieceId);
    assert(visible_ids[19u * 10u + 2u] == 255u);
}

void test_engine_garbage_topout_modes() {
    cc::Env invalid_env(EnvConfig{10});
    const auto invalid_before = invalid_env.snapshot();
    bool rejected_hole = false;
    try {
        (void)invalid_env.insert_garbage_rows({Board::kWidth});
    } catch (const std::invalid_argument&) {
        rejected_hole = true;
    }
    assert(rejected_hole);
    assert(invalid_env.state().board.rows() == invalid_before.state.board.rows());

    cc::Env overflow_env(EnvConfig{12});
    auto overflow_state = clean_state(overflow_env, Piece::O);
    overflow_state.active.piece = Piece::None;
    overflow_state.board.set_cell(3, Board::kRows - 1);
    overflow_state.piece_ids[Board::kRows - 1][3] = 2;
    overflow_env.restore(overflow_state);
    const auto overflow = overflow_env.insert_garbage_rows({0});
    assert(overflow.overflow);
    assert(!overflow.active_collision);
    assert(overflow.top_out);
    assert(overflow_env.state().game_over);

    cc::Env collision_env(EnvConfig{13});
    collision_env.restore(clean_state(collision_env, Piece::O));
    const std::vector<int> holes(22, 0);
    const auto collision = collision_env.insert_garbage_rows(holes);
    assert(!collision.overflow);
    assert(collision.active_collision);
    assert(collision.top_out);
}

void test_deterministic_joint_steps_and_prevalidation() {
    BattleConfig config{};
    config.seed = 98765;
    config.garbage_delay = 2;
    config.max_joint_steps = 50;
    BattleEnv lhs(config);
    BattleEnv rhs(config);

    assert(lhs.player_env(0).state().active.piece == lhs.player_env(1).state().active.piece);
    assert(lhs.player_env(0).state().queue == lhs.player_env(1).state().queue);
    std::array<float, kBattlePlayerCount> expected_scores{};
    for (int step = 0; step < 12 && !lhs.terminated(); ++step) {
        assert(lhs.observation(0) == rhs.observation(0));
        assert(lhs.observation(1) == rhs.observation(1));
        const auto action0 = first_legal_action(lhs.player_env(0));
        const auto action1 = first_legal_action(lhs.player_env(1));
        const auto left_result = lhs.step(action0, action1);
        const auto right_result = rhs.step(action0, action1);
        assert(left_result.success == right_result.success);
        assert(left_result.terminated == right_result.terminated);
        assert(left_result.winner == right_result.winner);
        for (std::size_t player = 0; player < kBattlePlayerCount; ++player) {
            expected_scores[player] += left_result.players[player].reward;
            assert_same_board(lhs.player_env(player), rhs.player_env(player));
            assert(lhs.pending_garbage(player) == rhs.pending_garbage(player));
            assert(lhs.stats(player).score == expected_scores[player]);
            assert(rhs.stats(player).score == expected_scores[player]);
        }
    }

    BattleEnv atomic(config);
    const auto before0 = atomic.observation(0);
    const auto before1 = atomic.observation(1);
    const auto legal1 = first_legal_action(atomic.player_env(1));
    std::array<bool, kDecisionActionDim> legal0{};
    for (const auto& decision : enumerate_stable_decisions(atomic.player_env(0))) {
        legal0[decision.action] = true;
    }
    const auto illegal = std::find(legal0.begin(), legal0.end(), false);
    assert(illegal != legal0.end());
    const auto invalid_action = static_cast<std::size_t>(
        std::distance(legal0.begin(), illegal));
    const auto rejected = atomic.step(invalid_action, legal1);
    assert(!rejected.success);
    assert(atomic.joint_steps() == 0);
    assert(atomic.observation(0) == before0);
    assert(atomic.observation(1) == before1);
    assert(atomic.stats(0).placements == 0);
    assert(atomic.stats(1).placements == 0);
}

void test_cold_clear_zero_budget_is_fixed_work() {
    cc::Env env(EnvConfig{654321});
    cc::Bot bot;
    cc::BotChoice expected_choice{};
    cc::BotThinkStats expected_stats{};
    for (int repeat = 0; repeat < 6; ++repeat) {
        assert(bot.sync_from_env(env));
        cc::BotChoice choice{};
        cc::BotThinkStats stats{};
        assert(bot.choose(0, &choice, &stats));
        assert(stats.selections == ColdClearBot::kDeterministicWorkUnits);
        assert(stats.budget_miss == 0);
        if (repeat == 0) {
            expected_choice = choice;
            expected_stats = stats;
            continue;
        }
        assert(choice.use_hold == expected_choice.use_hold);
        assert(choice.placement_index == expected_choice.placement_index);
        assert(choice.score == expected_choice.score);
        assert(stats.nodes == expected_stats.nodes);
        assert(stats.selections == expected_stats.selections);
        assert(stats.expansions == expected_stats.expansions);
    }
}

void test_delay_cancellation_and_partial_send() {
    BattleConfig config{};
    config.seed = 77;
    config.garbage_delay = 1;
    config.max_joint_steps = 20;
    BattleEnv battle(config);
    prepare_tetris(battle, 0);
    prepare_empty(battle, 1);
    assert(battle.enqueue_garbage(0, {1, 2}, 5));

    const auto first = battle.step(
        action_clearing(battle.player_env(0), 4),
        action_clearing(battle.player_env(1), 0));
    assert(first.success);
    assert(first.players[0].attack_generated == 4);
    assert(first.players[0].garbage_cancelled == 2);
    assert(first.players[0].garbage_sent == 2);
    assert(first.players[1].garbage_received == 2);
    assert(first.players[1].garbage_applied == 0);
    assert(first.players[1].incoming_garbage == 2);
    assert(first.players[1].next_garbage_delay == 1);

    const auto second = battle.step(
        action_clearing(battle.player_env(0), 0),
        action_clearing(battle.player_env(1), 0));
    assert(second.success);
    assert(second.players[1].garbage_applied == 2);
    assert(second.players[1].incoming_garbage == 0);
    assert(second.players[1].next_garbage_delay == -1);
    assert(battle.stats(0).score == first.players[0].reward + second.players[0].reward);
    assert(battle.stats(1).score == first.players[1].reward + second.players[1].reward);

    BattleEnv repeat(config);
    prepare_tetris(repeat, 0);
    prepare_empty(repeat, 1);
    assert(repeat.enqueue_garbage(0, {1, 2}, 5));
    const auto repeat_first = repeat.step(
        action_clearing(repeat.player_env(0), 4),
        action_clearing(repeat.player_env(1), 0));
    const auto repeat_second = repeat.step(
        action_clearing(repeat.player_env(0), 0),
        action_clearing(repeat.player_env(1), 0));
    assert(repeat_first.success && repeat_second.success);
    for (int y = 0; y < 2; ++y) {
        const auto row = battle.player_env(1).state().board.row_mask(y);
        assert(row == repeat.player_env(1).state().board.row_mask(y));
        int occupied = 0;
        for (int x = 0; x < Board::kWidth; ++x) {
            occupied += (row & (Board::RowMask{1u} << x)) != 0 ? 1 : 0;
        }
        assert(occupied == Board::kWidth - 1);
    }
}

void test_oldest_first_cancellation_and_due_packet_scan() {
    BattleConfig config{};
    config.seed = 91;
    config.garbage_delay = 2;
    config.attack_table[4] = 2;
    BattleEnv battle(config);
    prepare_tetris(battle, 0);
    prepare_empty(battle, 1);
    assert(battle.enqueue_garbage(0, {1, 1}, 5));
    assert(battle.enqueue_garbage(0, {7}, 0));
    assert(battle.next_garbage_delay(0) == 1);

    const auto result = battle.step(
        action_clearing(battle.player_env(0), 4),
        action_clearing(battle.player_env(1), 0));
    assert(result.success);
    assert(result.players[0].attack_generated == 2);
    assert(result.players[0].garbage_cancelled == 2);
    assert(result.players[0].garbage_sent == 0);
    assert(result.players[0].garbage_applied == 1);
    assert(result.players[0].incoming_garbage == 0);
    assert(!battle.player_env(0).state().board.occupied(7, 0));
    for (int x = 0; x < Board::kWidth; ++x) {
        if (x != 7) {
            assert(battle.player_env(0).state().board.occupied(x, 0));
        }
    }
}

void test_simultaneous_attacks_topouts_and_max_draw() {
    BattleConfig delayed{};
    delayed.seed = 123;
    delayed.garbage_delay = 3;
    BattleEnv simultaneous_attack(delayed);
    prepare_tetris(simultaneous_attack, 0);
    prepare_tetris(simultaneous_attack, 1);
    const auto attacks = simultaneous_attack.step(
        action_clearing(simultaneous_attack.player_env(0), 4),
        action_clearing(simultaneous_attack.player_env(1), 4));
    assert(attacks.success);
    assert(attacks.players[0].garbage_sent == 4);
    assert(attacks.players[1].garbage_sent == 4);
    assert(attacks.players[0].garbage_received == 4);
    assert(attacks.players[1].garbage_received == 4);
    assert(attacks.players[0].incoming_garbage == 4);
    assert(attacks.players[1].incoming_garbage == 4);

    BattleConfig immediate{};
    immediate.seed = 124;
    immediate.garbage_delay = 0;
    BattleEnv simultaneous_topout(immediate);
    const std::vector<int> holes(40, 0);
    assert(simultaneous_topout.enqueue_garbage(0, holes, 0));
    assert(simultaneous_topout.enqueue_garbage(1, holes, 0));
    const auto topout = simultaneous_topout.step(
        first_legal_action(simultaneous_topout.player_env(0)),
        first_legal_action(simultaneous_topout.player_env(1)));
    assert(topout.success);
    assert(topout.terminated);
    assert(topout.winner == -1);
    assert(topout.players[0].top_out);
    assert(topout.players[1].top_out);

    BattleEnv one_sided_topout(immediate);
    assert(one_sided_topout.enqueue_garbage(0, holes, 0));
    const auto loss = one_sided_topout.step(
        first_legal_action(one_sided_topout.player_env(0)),
        first_legal_action(one_sided_topout.player_env(1)));
    assert(loss.success);
    assert(loss.terminated);
    assert(loss.winner == 1);
    assert(loss.players[0].top_out);
    assert(!loss.players[1].top_out);

    BattleEnv oversized_garbage(immediate);
    const std::vector<int> oversized_holes(45, 0);
    assert(oversized_garbage.enqueue_garbage(0, oversized_holes, 0));
    const auto oversized = oversized_garbage.step(
        first_legal_action(oversized_garbage.player_env(0)),
        first_legal_action(oversized_garbage.player_env(1)));
    assert(oversized.success);
    assert(oversized.players[0].garbage_received == 0);
    assert(oversized.players[0].garbage_applied == Board::kRows);
    assert(oversized_garbage.stats(0).garbage_received == 45);
    assert(oversized_garbage.stats(0).garbage_applied == Board::kRows);
    assert(oversized.players[0].top_out);

    BattleConfig limited{};
    limited.seed = 125;
    limited.max_joint_steps = 1;
    BattleEnv max_draw(limited);
    const auto draw = max_draw.step(
        first_legal_action(max_draw.player_env(0)),
        first_legal_action(max_draw.player_env(1)));
    assert(draw.success);
    assert(draw.terminated);
    assert(draw.winner == -1);
    assert(!draw.players[0].top_out);
    assert(!draw.players[1].top_out);
}

void test_canonical_observation_contract() {
    BattleEnv battle(BattleConfig{});
    auto state = clean_state(battle.player_env(0), Piece::O);
    for (int x = 1; x < Board::kWidth; x += 2) {
        for (int y = 0; y < Board::kVisibleRows; ++y) {
            state.board.set_cell(x, y);
        }
    }
    battle.player_env(0).restore(state);

    const auto player0 = battle.observation(0);
    const auto player1 = battle.observation(1);
    assert(player0.size() == kBattleObservationSize);
    assert(player1.size() == kBattleObservationSize);
    assert(kBattleObservationSize == TETRIS_CC_BATTLE_OBSERVATION_SIZE);
    assert(std::all_of(player0.begin(), player0.end(), [](float value) {
        return value >= 0.0f && value <= 1.0f;
    }));
    assert(std::all_of(player1.begin(), player1.end(), [](float value) {
        return value >= 0.0f && value <= 1.0f;
    }));

    constexpr std::size_t public_offset =
        kBattleOwnObservationSize + kBattleOpponentBoardSize;
    assert(player0[public_offset + 4] == 0.5f);
    assert(player0[public_offset + 5] == 1.0f);
    assert(player0[public_offset + 7] == 1.0f);
    assert(player0[public_offset + 8] == 1.0f);
    assert(player0[public_offset + 14] == 0.0f);
    assert(player0[public_offset + 15] == 0.5f);
    assert(player1[public_offset + 9] == 0.5f);
    assert(player1[public_offset + 10] == 1.0f);
    assert(player1[public_offset + 12] == 1.0f);
    assert(player1[public_offset + 13] == 1.0f);
    assert(player1[public_offset + 14] == 1.0f);
    assert(player1[public_offset + 15] == 0.5f);
}

std::size_t first_mask_action(const std::vector<std::uint8_t>& mask) {
    const auto it = std::find(mask.begin(), mask.end(), static_cast<std::uint8_t>(1));
    assert(it != mask.end());
    return static_cast<std::size_t>(std::distance(mask.begin(), it));
}

void test_battle_c_api() {
    tetris_cc_battle_config config{};
    tetris_cc_battle_config_default(&config);
    assert(config.max_joint_steps == 500);
    config.seed = 20260827u;
    config.garbage_delay = 1;
    auto* battle = tetris_cc_battle_create(&config);
    assert(battle != nullptr);
    assert(tetris_cc_battle_action_dim() == kDecisionActionDim);
    assert(tetris_cc_battle_observation_size(battle) == TETRIS_CC_BATTLE_OBSERVATION_SIZE);

    std::vector<float> observation(TETRIS_CC_BATTLE_OBSERVATION_SIZE, 0.0f);
    assert(tetris_cc_battle_observation_write(
               battle, 0, observation.data(), observation.size()) == observation.size());
    std::vector<std::uint8_t> mask0(kDecisionActionDim, 0);
    std::vector<std::uint8_t> mask1(kDecisionActionDim, 0);
    assert(tetris_cc_battle_decision_mask_write(
               battle, 0, mask0.data(), mask0.size()) == mask0.size());
    assert(tetris_cc_battle_decision_mask_write(
               battle, 1, mask1.data(), mask1.size()) == mask1.size());

    std::size_t bot_action = 0;
    float bot_score = 0.0f;
    std::uint64_t nodes = 0;
    double think_ms = 0.0;
    double nps = 0.0;
    int budget_miss = 0;
    assert(tetris_cc_battle_bot_choose(
               battle,
               0,
               0,
               &bot_action,
               &bot_score,
               &nodes,
               &think_ms,
               &nps,
               &budget_miss) == 1);
    assert(mask0[bot_action] == 1);
    assert(std::isfinite(bot_score));
    assert(nodes > 0);
    assert(think_ms >= 0.0);
    assert(nps >= 0.0);
    assert(budget_miss == 0);
    const auto deterministic_action = bot_action;
    const auto deterministic_nodes = nodes;
    const auto deterministic_score = bot_score;
    for (int repeat = 0; repeat < 5; ++repeat) {
        bot_action = 0;
        nodes = 0;
        budget_miss = -1;
        assert(tetris_cc_battle_bot_choose(
                   battle,
                   0,
                   0,
                   &bot_action,
                   &bot_score,
                   &nodes,
                   &think_ms,
                   &nps,
                   &budget_miss) == 1);
        assert(bot_action == deterministic_action);
        assert(nodes == deterministic_nodes);
        assert(bot_score == deterministic_score);
        assert(budget_miss == 0);
    }

    const int hole = 3;
    assert(tetris_cc_battle_enqueue_garbage(battle, 0, &hole, 1, 0) == 1);
    tetris_cc_battle_step_result step{};
    assert(tetris_cc_battle_step(
               battle, bot_action, first_mask_action(mask1), &step) == 1);
    assert(step.success == 1);
    assert(step.players[0].garbage_applied == 1);
    assert(step.players[0].next_garbage_delay == -1);

    tetris_cc_battle_meta meta{};
    assert(tetris_cc_battle_meta_get(battle, &meta) == 1);
    assert(meta.joint_steps == 1);
    assert(meta.players[0].placements == 1);
    assert(meta.players[0].score == step.players[0].reward);
    assert(meta.players[0].garbage_applied == 1);

    std::vector<std::uint8_t> board(200, 0);
    std::vector<std::uint8_t> ids(200, 255);
    assert(tetris_cc_battle_board_write(
               battle, 0, 0, board.data(), board.size()) == board.size());
    assert(tetris_cc_battle_board_piece_ids_write(
               battle, 0, 0, ids.data(), ids.size()) == ids.size());
    for (int x = 0; x < Board::kWidth; ++x) {
        assert(board[190u + static_cast<std::size_t>(x)] == (x == hole ? 0 : 1));
        assert(ids[190u + static_cast<std::size_t>(x)] ==
               (x == hole ? 255 : TETRIS_CC_GARBAGE_PIECE_ID));
    }

    assert(tetris_cc_battle_observation_write(
               battle, 1, observation.data(), observation.size()) == observation.size());
    for (int x = 0; x < Board::kWidth; ++x) {
        assert(observation[254u + static_cast<std::size_t>(x)] ==
               (x == hole ? 0.0f : 1.0f));
    }
    tetris_cc_battle_destroy(battle);

    tetris_cc_battle_config_default(&config);
    config.max_joint_steps = 1;
    battle = tetris_cc_battle_create(&config);
    assert(battle != nullptr);
    assert(tetris_cc_battle_decision_mask_write(
               battle, 0, mask0.data(), mask0.size()) == mask0.size());
    assert(tetris_cc_battle_decision_mask_write(
               battle, 1, mask1.data(), mask1.size()) == mask1.size());
    assert(tetris_cc_battle_step(
               battle, first_mask_action(mask0), first_mask_action(mask1), &step) == 1);
    assert(step.terminated == 1);
    assert(step.winner == -1);
    std::fill(mask0.begin(), mask0.end(), 0);
    assert(tetris_cc_battle_decision_mask_write(
               battle, 0, mask0.data(), mask0.size()) == mask0.size());
    assert(std::any_of(mask0.begin(), mask0.end(), [](std::uint8_t value) {
        return value != 0;
    }));
    tetris_cc_battle_destroy(battle);
}

}  // namespace

int main() {
    test_engine_garbage_insertion_and_piece_ids();
    test_engine_garbage_topout_modes();
    test_deterministic_joint_steps_and_prevalidation();
    test_cold_clear_zero_budget_is_fixed_work();
    test_delay_cancellation_and_partial_send();
    test_oldest_first_cancellation_and_due_packet_scan();
    test_simultaneous_attacks_topouts_and_max_draw();
    test_canonical_observation_contract();
    test_battle_c_api();
    return 0;
}
