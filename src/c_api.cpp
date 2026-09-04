#include "tetris_v2/c_api.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <new>
#include <optional>
#include <vector>

#include "tetris_v2/battle.hpp"
#include "tetris_v2/cc2_data.hpp"
#include "tetris_v2/cc2_eval.hpp"
#include "tetris_v2/cc_bot.hpp"
#include "tetris_v2/cc_env.hpp"
#include "tetris_v2/decision.hpp"
#include "tetris_v2/observation.hpp"
#include "tetris_v2/piece_defs.hpp"

static_assert(
    TETRIS_CC_BATTLE_PLAYER_COUNT == tetris_v2::kBattlePlayerCount,
    "battle C ABI player count drifted");
static_assert(
    TETRIS_CC_BATTLE_OBSERVATION_SIZE == tetris_v2::kBattleObservationSize,
    "battle C ABI observation size drifted");

struct tetris_cc_env_handle {
    explicit tetris_cc_env_handle(std::uint32_t seed, bool play_mode = false)
        : env(config_for(seed, play_mode), !play_mode) {}

    static tetris_v2::EnvConfig config_for(std::uint32_t seed, bool play_mode) {
        tetris_v2::EnvConfig config{};
        config.seed = seed;
        if (play_mode) {
            config.spawn_y = 19;
        }
        return config;
    }

    tetris_v2::cc::Env env;
};

struct tetris_cc_bot_handle {
    tetris_v2::cc::Bot bot;
};

struct tetris_cc_snapshot_handle {
    tetris_v2::EnvSnapshot snapshot{};
};

struct tetris_cc_battle_handle {
    explicit tetris_cc_battle_handle(const tetris_v2::BattleConfig& config)
        : battle(config) {}

    tetris_v2::BattleEnv battle;
    std::array<tetris_v2::cc::Bot, tetris_v2::kBattlePlayerCount> bots{};
};

namespace {

tetris_v2::Action parse_action(int action) {
    if (action < static_cast<int>(tetris_v2::Action::None) ||
        action > static_cast<int>(tetris_v2::Action::Hold)) {
        return tetris_v2::Action::None;
    }
    return static_cast<tetris_v2::Action>(action);
}

std::optional<tetris_v2::Action> parse_rotate_action(int action) {
    auto parsed = parse_action(action);
    if (parsed == tetris_v2::Action::RotateCW || parsed == tetris_v2::Action::RotateCCW) {
        return parsed;
    }
    return std::nullopt;
}

size_t write_visible_board(
    const tetris_v2::Board& board,
    const std::optional<tetris_v2::ActivePiece>& active,
    uint8_t* out,
    size_t out_len) {
    if (!out) {
        return 0;
    }
    constexpr int kRows = tetris_v2::Board::kVisibleRows;
    constexpr int kCols = tetris_v2::Board::kWidth;
    constexpr size_t kTotal = static_cast<size_t>(kRows * kCols);

    std::array<uint8_t, kTotal> cells{};
    cells.fill(0);

    for (int row = 0; row < kRows; ++row) {
        int y = (kRows - 1) - row;
        auto mask = board.row_mask(y);
        for (int x = 0; x < kCols; ++x) {
            if (mask & (1u << x)) {
                cells[static_cast<size_t>(row * kCols + x)] = 1;
            }
        }
    }

    if (active.has_value()) {
        auto piece_cells = tetris_v2::piece_cells(active->piece, active->rotation);
        for (const auto& c : piece_cells) {
            int x = active->x + c.x;
            int y = active->y + c.y;
            if (x < 0 || x >= kCols || y < 0 || y >= kRows) {
                continue;
            }
            int row = (kRows - 1) - y;
            cells[static_cast<size_t>(row * kCols + x)] = 1;
        }
    }

    auto n = std::min(out_len, kTotal);
    std::copy(cells.begin(), cells.begin() + static_cast<std::ptrdiff_t>(n), out);
    return n;
}

void maybe_set_int(int* dst, int value) {
    if (dst) {
        *dst = value;
    }
}

void maybe_set_u64(std::uint64_t* dst, std::uint64_t value) {
    if (dst) {
        *dst = value;
    }
}

void maybe_set_double(double* dst, double value) {
    if (dst) {
        *dst = value;
    }
}

void write_step_result(
    const tetris_v2::StepResult& result, tetris_cc_env_step_result* out) {
    out->action_succeeded = result.action_succeeded ? 1 : 0;
    out->piece_locked = result.piece_locked ? 1 : 0;
    out->hold_used = result.hold_used ? 1 : 0;
    out->lines_cleared = result.lines_cleared;
    out->spin_clear = result.spin_clear ? 1 : 0;
    out->spin_type = static_cast<int>(result.spin_type);
    out->difficult_clear = result.difficult_clear ? 1 : 0;
    out->b2b_bonus_applied = result.b2b_bonus_applied ? 1 : 0;
    out->combo = result.combo;
    out->back_to_back = result.back_to_back ? 1 : 0;
    out->reward = result.reward;
    out->game_over = result.game_over ? 1 : 0;
    out->top_out = result.top_out ? 1 : 0;
}

constexpr std::size_t kRlPlacementSlots = 96;
constexpr std::size_t kRlActionDim = 97;
constexpr float kIllegalScore = -1.0e6f;
constexpr float kFallbackFloor = -1000.0f;

std::uint8_t combo_to_cc2(int combo) {
    if (combo < 0) {
        return 0;
    }
    return static_cast<std::uint8_t>(std::min(combo + 1, 255));
}

tetris_v2::cc2::PieceMask bag_mask_from_snapshot(const tetris_v2::EnvSnapshot& snapshot) {
    using tetris_v2::Piece;
    using tetris_v2::cc2::PieceMask;
    PieceMask bag_mask = 0;

    const auto& bag = snapshot.randomizer.bag_order();
    std::size_t idx = snapshot.randomizer.bag_index();
    if (idx > bag.size()) {
        idx = bag.size();
    }
    for (std::size_t i = idx; i < bag.size(); ++i) {
        if (bag[i] != Piece::None) {
            bag_mask = static_cast<PieceMask>(bag_mask | tetris_v2::cc2::piece_bit(bag[i]));
        }
    }
    if (bag_mask == 0) {
        bag_mask = tetris_v2::cc2::kAllPiecesMask;
    }

    for (auto it = snapshot.state.queue.rbegin(); it != snapshot.state.queue.rend(); ++it) {
        if (bag_mask == tetris_v2::cc2::kAllPiecesMask) {
            bag_mask = 0;
        }
        if (*it != Piece::None) {
            bag_mask = static_cast<PieceMask>(bag_mask | tetris_v2::cc2::piece_bit(*it));
        }
    }
    return bag_mask;
}

tetris_v2::cc2::GameState root_state_from_snapshot(const tetris_v2::EnvSnapshot& snapshot) {
    tetris_v2::cc2::GameState root{};
    const auto& env_state = snapshot.state;
    root.board = tetris_v2::cc2::Board::from_env_board(env_state.board);
    root.bag_mask = bag_mask_from_snapshot(snapshot);
    root.reserve = env_state.hold.value_or(tetris_v2::Piece::None);
    root.hold_available = env_state.hold_available;
    root.back_to_back = env_state.back_to_back;
    root.combo = combo_to_cc2(env_state.combo);
    return root;
}

tetris_v2::cc2::Placement option_to_cc2_placement(const tetris_v2::PlacementOption& option) {
    return tetris_v2::cc2::Placement{
        tetris_v2::cc2::PieceLocation{
            option.placement.piece,
            tetris_v2::cc2::rotation_from_env(option.placement.rotation),
            static_cast<std::int8_t>(option.placement.x),
            static_cast<std::int8_t>(option.placement.y),
        },
        tetris_v2::cc2::spin_from_env(option.spin_type),
    };
}

std::uint32_t softdrop_for_option(
    const tetris_v2::ActivePiece& active,
    const tetris_v2::PlacementOption& option) {
    if (active.y <= option.placement.y) {
        return 0;
    }
    return static_cast<std::uint32_t>(active.y - option.placement.y);
}

float fallback_one_step_eval_score(const tetris_v2::cc::Env& env) {
    const auto& state = env.state();
    if (state.game_over || state.active.piece == tetris_v2::Piece::None) {
        return kFallbackFloor;
    }

    const auto options = env.enumerate_active_piece_placements();
    if (options.empty()) {
        return kFallbackFloor;
    }

    const auto snapshot = env.snapshot();
    const auto root = root_state_from_snapshot(snapshot);
    const auto next_piece = state.active.piece;
    tetris_v2::cc2::FreestyleWeights weights{};

    float best = -std::numeric_limits<float>::infinity();
    for (const auto& option : options) {
        auto candidate = root;
        const auto placement = option_to_cc2_placement(option);
        const auto info = candidate.advance(next_piece, placement);
        const auto eval = tetris_v2::cc2::evaluate_freestyle(
            weights,
            candidate,
            info,
            softdrop_for_option(state.active, option));
        if (std::isfinite(eval.total) && eval.total > best) {
            best = eval.total;
        }
    }

    if (!std::isfinite(best)) {
        return kFallbackFloor;
    }
    return best;
}

}  // namespace

extern "C" {

tetris_cc_env_handle* tetris_cc_env_create(uint32_t seed) {
    try {
        return new tetris_cc_env_handle(seed);
    } catch (...) {
        return nullptr;
    }
}

tetris_cc_env_handle* tetris_cc_env_create_play(uint32_t seed) {
    try {
        return new tetris_cc_env_handle(seed, true);
    } catch (...) {
        return nullptr;
    }
}

void tetris_cc_env_destroy(tetris_cc_env_handle* handle) { delete handle; }

void tetris_cc_env_reset(tetris_cc_env_handle* handle, uint32_t seed) {
    if (!handle) {
        return;
    }
    handle->env.reset(seed);
}

int tetris_cc_env_step(tetris_cc_env_handle* handle, int action, float* reward_out) {
    if (!handle) {
        return 0;
    }
    auto result = handle->env.step(parse_action(action));
    if (reward_out) {
        *reward_out = result.reward;
    }
    return result.game_over ? 1 : 0;
}

int tetris_cc_env_step_ex(
    tetris_cc_env_handle* handle, int action, tetris_cc_env_step_result* result_out) {
    if (!handle || !result_out ||
        action < static_cast<int>(tetris_v2::Action::None) ||
        action > static_cast<int>(tetris_v2::Action::Hold)) {
        return 0;
    }
    const auto result = handle->env.step(static_cast<tetris_v2::Action>(action));
    write_step_result(result, result_out);
    return 1;
}

int tetris_cc_env_input_ex(
    tetris_cc_env_handle* handle, int action, tetris_cc_env_step_result* result_out) {
    if (!handle || !result_out ||
        action < static_cast<int>(tetris_v2::Action::None) ||
        action > static_cast<int>(tetris_v2::Action::Hold)) {
        return 0;
    }
    const auto result = handle->env.input(static_cast<tetris_v2::Action>(action));
    write_step_result(result, result_out);
    return 1;
}

int tetris_cc_env_tick_ex(
    tetris_cc_env_handle* handle, tetris_cc_env_step_result* result_out) {
    if (!handle || !result_out) {
        return 0;
    }
    const auto result = handle->env.tick();
    write_step_result(result, result_out);
    return 1;
}

int tetris_cc_env_hold(tetris_cc_env_handle* handle, float* reward_out) {
    if (!handle) {
        return 0;
    }
    auto result = handle->env.step(tetris_v2::Action::Hold);
    if (reward_out) {
        *reward_out = result.reward;
    }
    return result.action_succeeded ? 1 : 0;
}

tetris_cc_snapshot_handle* tetris_cc_env_snapshot_create(const tetris_cc_env_handle* handle) {
    if (!handle) {
        return nullptr;
    }
    try {
        auto* out = new tetris_cc_snapshot_handle{};
        out->snapshot = handle->env.snapshot();
        return out;
    } catch (...) {
        return nullptr;
    }
}

void tetris_cc_snapshot_destroy(tetris_cc_snapshot_handle* snapshot) { delete snapshot; }

int tetris_cc_env_restore_snapshot(tetris_cc_env_handle* handle, const tetris_cc_snapshot_handle* snapshot) {
    if (!handle || !snapshot) {
        return 0;
    }
    handle->env.restore(snapshot->snapshot);
    return 1;
}

size_t tetris_cc_env_observation_size(const tetris_cc_env_handle* handle, int include_hidden_rows) {
    if (!handle) {
        return 0;
    }
    return tetris_v2::observation_size(
        handle->env.config().queue_size, include_hidden_rows != 0);
}

size_t tetris_cc_env_observation_write(
    const tetris_cc_env_handle* handle, int include_hidden_rows, float* out, size_t out_len) {
    if (!handle || !out) {
        return 0;
    }
    auto obs = tetris_v2::encode_observation(
        handle->env.state(), handle->env.config().queue_size, include_hidden_rows != 0);
    auto n = std::min(out_len, obs.size());
    std::copy(obs.begin(), obs.begin() + static_cast<std::ptrdiff_t>(n), out);
    return n;
}

size_t tetris_cc_env_board_write(
    const tetris_cc_env_handle* handle, int include_active, uint8_t* out, size_t out_len) {
    if (!handle || !out) {
        return 0;
    }
    std::optional<tetris_v2::ActivePiece> active{};
    if (include_active != 0) {
        active = handle->env.state().active;
    }
    return write_visible_board(handle->env.state().board, active, out, out_len);
}

size_t tetris_cc_env_board_piece_ids_write(
    const tetris_cc_env_handle* handle, int include_active, uint8_t* out, size_t out_len) {
    if (!handle || !out) {
        return 0;
    }
    auto ids = handle->env.visible_board_piece_ids(include_active != 0);
    auto n = std::min(out_len, ids.size());
    std::copy(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(n), out);
    return n;
}

int tetris_cc_env_active_piece(
    const tetris_cc_env_handle* handle, int* piece, int* rotation, int* x, int* y) {
    if (!handle) {
        return 0;
    }
    const auto& a = handle->env.state().active;
    maybe_set_int(piece, static_cast<int>(a.piece));
    maybe_set_int(rotation, static_cast<int>(a.rotation));
    maybe_set_int(x, a.x);
    maybe_set_int(y, a.y);
    return 1;
}

int tetris_cc_env_ghost_piece(
    const tetris_cc_env_handle* handle, int* piece, int* rotation, int* x, int* landing_y) {
    if (!handle) {
        return 0;
    }
    const auto ghost = handle->env.ghost_piece();
    if (!ghost.has_value()) {
        return 0;
    }
    maybe_set_int(piece, static_cast<int>(ghost->piece));
    maybe_set_int(rotation, static_cast<int>(ghost->rotation));
    maybe_set_int(x, ghost->x);
    maybe_set_int(landing_y, ghost->y);
    return 1;
}

int tetris_cc_env_hold_piece(
    const tetris_cc_env_handle* handle, int* has_hold, int* hold_piece, int* hold_available) {
    if (!handle) {
        return 0;
    }
    const auto& state = handle->env.state();
    maybe_set_int(has_hold, state.hold.has_value() ? 1 : 0);
    maybe_set_int(
        hold_piece, static_cast<int>(state.hold.has_value() ? *state.hold : tetris_v2::Piece::None));
    maybe_set_int(hold_available, state.hold_available ? 1 : 0);
    return 1;
}

size_t tetris_cc_env_queue_count(const tetris_cc_env_handle* handle) {
    if (!handle) {
        return 0;
    }
    return handle->env.state().queue.size();
}

int tetris_cc_env_queue_get(const tetris_cc_env_handle* handle, size_t index, int* piece) {
    if (!handle || !piece) {
        return 0;
    }
    const auto& queue = handle->env.state().queue;
    if (index >= queue.size()) {
        return 0;
    }
    *piece = static_cast<int>(queue[index]);
    return 1;
}

int tetris_cc_env_meta(
    const tetris_cc_env_handle* handle,
    int* game_over,
    int* top_out,
    int* combo,
    int* back_to_back,
    int* total_lines_cleared,
    int* lock_timer,
    int* lock_resets_used) {
    if (!handle) {
        return 0;
    }
    const auto& state = handle->env.state();
    maybe_set_int(game_over, state.game_over ? 1 : 0);
    maybe_set_int(top_out, state.top_out ? 1 : 0);
    maybe_set_int(combo, state.combo);
    maybe_set_int(back_to_back, state.back_to_back ? 1 : 0);
    maybe_set_int(total_lines_cleared, state.total_lines_cleared);
    maybe_set_int(lock_timer, state.lock_timer);
    maybe_set_int(lock_resets_used, state.lock_resets_used);
    return 1;
}

size_t tetris_cc_env_placement_count(const tetris_cc_env_handle* handle) {
    if (!handle) {
        return 0;
    }
    return handle->env.enumerate_active_piece_placements().size();
}

int tetris_cc_env_placement_get(
    const tetris_cc_env_handle* handle, size_t index, int* x, int* y, int* rotation, int* lines_cleared) {
    if (!handle) {
        return 0;
    }
    auto option = handle->env.placement_option_at(index);
    if (!option.has_value()) {
        return 0;
    }
    maybe_set_int(x, option->placement.x);
    maybe_set_int(y, option->placement.y);
    maybe_set_int(rotation, static_cast<int>(option->placement.rotation));
    maybe_set_int(lines_cleared, option->lines_cleared);
    return 1;
}

size_t tetris_cc_env_placement_board_write(
    const tetris_cc_env_handle* handle, size_t index, uint8_t* out, size_t out_len) {
    if (!handle || !out) {
        return 0;
    }
    auto option = handle->env.placement_option_at(index);
    if (!option.has_value()) {
        return 0;
    }
    return write_visible_board(option->board_after_lock, std::nullopt, out, out_len);
}

size_t tetris_cc_env_placement_board_piece_ids_write(
    const tetris_cc_env_handle* handle, size_t index, uint8_t* out, size_t out_len) {
    if (!handle || !out) {
        return 0;
    }
    auto option = handle->env.placement_option_at(index);
    if (!option.has_value()) {
        return 0;
    }
    auto ids = handle->env.visible_placement_piece_ids(index);
    auto n = std::min(out_len, ids.size());
    std::copy(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(n), out);
    return n;
}

int tetris_cc_env_apply_placement_index(
    tetris_cc_env_handle* handle, size_t index, float* reward_out, int* lines_cleared_out, int* game_over_out) {
    if (!handle) {
        return 0;
    }
    auto result = handle->env.apply_placement_index(index);
    if (!result.action_succeeded) {
        return 0;
    }
    if (reward_out) {
        *reward_out = result.reward;
    }
    maybe_set_int(lines_cleared_out, result.lines_cleared);
    maybe_set_int(game_over_out, result.game_over ? 1 : 0);
    return 1;
}

size_t tetris_cc_env_decision_action_dim(void) { return tetris_v2::kDecisionActionDim; }

size_t tetris_cc_env_decision_mask_write(
    const tetris_cc_env_handle* handle, uint8_t* out, size_t out_len) {
    if (!handle || !out || out_len < tetris_v2::kDecisionActionDim) {
        return 0;
    }
    std::fill_n(
        out,
        static_cast<std::ptrdiff_t>(tetris_v2::kDecisionActionDim),
        static_cast<uint8_t>(0));
    for (const auto& option : tetris_v2::enumerate_stable_decisions(handle->env)) {
        out[option.action] = static_cast<uint8_t>(1);
    }
    return tetris_v2::kDecisionActionDim;
}

int tetris_cc_env_decision_get(
    const tetris_cc_env_handle* handle,
    size_t action,
    int* use_hold,
    size_t* placement_index,
    int* x,
    int* y,
    int* rotation) {
    if (!handle || action >= tetris_v2::kDecisionActionDim) {
        return 0;
    }
    const auto option = tetris_v2::stable_decision_at_action(handle->env, action);
    if (!option.has_value()) {
        return 0;
    }
    maybe_set_int(use_hold, option->use_hold ? 1 : 0);
    if (placement_index) {
        *placement_index = option->placement_index;
    }
    maybe_set_int(x, option->placement.x);
    maybe_set_int(y, option->placement.y);
    maybe_set_int(rotation, static_cast<int>(option->placement.rotation));
    return 1;
}

int tetris_cc_env_decision_action_for_choice(
    const tetris_cc_env_handle* handle,
    int use_hold,
    size_t placement_index,
    size_t* action_out) {
    if (!handle || !action_out) {
        return 0;
    }
    const auto action = tetris_v2::stable_decision_for_choice(
        handle->env, use_hold != 0, placement_index);
    if (!action.has_value()) {
        return 0;
    }
    *action_out = *action;
    return 1;
}

int tetris_cc_env_apply_decision(
    tetris_cc_env_handle* handle,
    size_t action,
    float* reward_out,
    int* lines_cleared_out,
    int* game_over_out,
    int* used_hold_out,
    size_t* placement_index_out) {
    if (!handle || action >= tetris_v2::kDecisionActionDim) {
        return 0;
    }
    tetris_v2::StepResult result{};
    int used_hold = 0;
    std::size_t placement_index = 0;
    if (!tetris_v2::apply_stable_decision(
            handle->env, action, &result, &used_hold, &placement_index)) {
        return 0;
    }
    if (reward_out) {
        *reward_out = result.reward;
    }
    maybe_set_int(lines_cleared_out, result.lines_cleared);
    maybe_set_int(game_over_out, result.game_over ? 1 : 0);
    maybe_set_int(used_hold_out, used_hold);
    if (placement_index_out) {
        *placement_index_out = placement_index;
    }
    return 1;
}

int tetris_cc_env_last_clear_meta(
    const tetris_cc_env_handle* handle,
    int* spin_clear,
    int* difficult_clear,
    int* b2b_bonus_applied) {
    if (!handle) {
        return 0;
    }
    const auto& state = handle->env.state();
    maybe_set_int(spin_clear, state.last_clear_spin ? 1 : 0);
    maybe_set_int(difficult_clear, state.last_clear_difficult ? 1 : 0);
    maybe_set_int(b2b_bonus_applied, state.last_clear_b2b_bonus ? 1 : 0);
    return 1;
}

int tetris_cc_env_last_clear_spin_type(const tetris_cc_env_handle* handle, int* spin_type) {
    if (!handle) {
        return 0;
    }
    maybe_set_int(
        spin_type,
        static_cast<int>(handle->env.state().last_clear_spin_type));
    return 1;
}

size_t tetris_cc_env_rotation_trace_count(const tetris_cc_env_handle* handle, int rotate_action) {
    if (!handle) {
        return 0;
    }
    auto action = parse_rotate_action(rotate_action);
    if (!action.has_value()) {
        return 0;
    }
    auto trace = handle->env.rotation_trace(*action);
    return trace.tests.size();
}

int tetris_cc_env_rotation_trace_get(
    const tetris_cc_env_handle* handle,
    int rotate_action,
    size_t index,
    int* test_index,
    int* phase,
    int* kick_index,
    int* dx,
    int* dy,
    int* passed,
    int* candidate_x,
    int* candidate_y,
    int* candidate_rotation,
    int* candidate_collides) {
    if (!handle) {
        return 0;
    }
    auto action = parse_rotate_action(rotate_action);
    if (!action.has_value()) {
        return 0;
    }
    auto trace = handle->env.rotation_trace(*action);
    if (index >= trace.tests.size()) {
        return 0;
    }
    const auto& test = trace.tests[index];
    maybe_set_int(test_index, test.test_index);
    maybe_set_int(phase, test.phase);
    maybe_set_int(kick_index, test.kick_index);
    maybe_set_int(dx, test.offset.x);
    maybe_set_int(dy, test.offset.y);
    maybe_set_int(passed, test.passed ? 1 : 0);
    maybe_set_int(candidate_x, test.candidate.x);
    maybe_set_int(candidate_y, test.candidate.y);
    maybe_set_int(candidate_rotation, static_cast<int>(test.candidate.rotation));
    maybe_set_int(candidate_collides, test.collides ? 1 : 0);
    return 1;
}

int tetris_cc_env_rotation_trace_meta(
    const tetris_cc_env_handle* handle,
    int rotate_action,
    int* success,
    int* final_x,
    int* final_y,
    int* final_rotation) {
    if (!handle) {
        return 0;
    }
    auto action = parse_rotate_action(rotate_action);
    if (!action.has_value()) {
        return 0;
    }
    auto trace = handle->env.rotation_trace(*action);
    maybe_set_int(success, trace.success ? 1 : 0);
    if (trace.final_pose.has_value()) {
        maybe_set_int(final_x, trace.final_pose->x);
        maybe_set_int(final_y, trace.final_pose->y);
        maybe_set_int(final_rotation, static_cast<int>(trace.final_pose->rotation));
    } else {
        maybe_set_int(final_x, -1);
        maybe_set_int(final_y, -1);
        maybe_set_int(final_rotation, -1);
    }
    return 1;
}

tetris_cc_bot_handle* tetris_cc_bot_create_default(void) {
    try {
        return new tetris_cc_bot_handle{};
    } catch (...) {
        return nullptr;
    }
}

void tetris_cc_bot_destroy(tetris_cc_bot_handle* handle) { delete handle; }

int tetris_cc_bot_sync_from_env(tetris_cc_bot_handle* bot, const tetris_cc_env_handle* env) {
    if (!bot || !env) {
        return 0;
    }
    return bot->bot.sync_from_env(env->env) ? 1 : 0;
}

int tetris_cc_bot_choose(
    tetris_cc_bot_handle* bot,
    int think_ms,
    int* use_hold_out,
    size_t* placement_index_out,
    float* score_out,
    std::uint64_t* nodes_out,
    double* think_ms_out,
    double* nps_out,
    int* budget_miss_out) {
    if (!bot) {
        return 0;
    }

    tetris_v2::cc::BotChoice choice{};
    tetris_v2::cc::BotThinkStats stats{};
    if (!bot->bot.choose(think_ms, &choice, &stats)) {
        return 0;
    }

    maybe_set_int(use_hold_out, choice.use_hold ? 1 : 0);
    if (placement_index_out) {
        *placement_index_out = choice.placement_index;
    }
    if (score_out) {
        *score_out = choice.score;
    }
    maybe_set_u64(nodes_out, stats.nodes);
    maybe_set_double(think_ms_out, stats.think_ms);
    maybe_set_double(nps_out, stats.nps);
    maybe_set_int(budget_miss_out, stats.budget_miss);
    return 1;
}

int tetris_cc_bot_rank_actions(
    tetris_cc_bot_handle* bot,
    const tetris_cc_env_handle* env,
    int think_ms,
    float* scores_out,
    size_t scores_len,
    uint8_t* legal_mask_out,
    size_t legal_mask_len,
    std::uint64_t* nodes_out,
    double* think_ms_out,
    double* nps_out,
    int* budget_miss_out,
    int* placement_count_raw_out,
    int* placement_overflow_out,
    int* unexpanded_count_out) {
    if (!bot || !env || !scores_out || !legal_mask_out) {
        return 0;
    }
    if (scores_len < kRlActionDim || legal_mask_len < kRlActionDim) {
        return 0;
    }

    std::fill_n(scores_out, static_cast<std::ptrdiff_t>(kRlActionDim), kIllegalScore);
    std::fill_n(legal_mask_out, static_cast<std::ptrdiff_t>(kRlActionDim), static_cast<std::uint8_t>(0));

    const auto snapshot = env->env.snapshot();
    const auto root_options = env->env.enumerate_active_piece_placements();
    const std::size_t placement_count_raw = root_options.size();
    const std::size_t placement_slots = std::min<std::size_t>(kRlPlacementSlots, placement_count_raw);
    const bool placement_overflow = placement_count_raw > kRlPlacementSlots;
    const bool hold_legal = env->env.state().hold_available;
    const int normalized_think_ms = std::max(1, think_ms);
    std::size_t remaining_slots = placement_slots + (hold_legal ? 1u : 0u);
    const auto t0 = std::chrono::steady_clock::now();
    const auto deadline = t0 + std::chrono::milliseconds(normalized_think_ms);

    std::uint64_t nodes_total = 0;
    int budget_miss_total = 0;
    int unexpanded_count = 0;

    auto evaluate_slot = [&](bool use_hold, std::size_t placement_index) -> float {
        const auto slots_including_current = remaining_slots;
        --remaining_slots;

        tetris_v2::cc::Env sim(env->env.config());
        sim.restore(snapshot);

        tetris_v2::StepResult step{};
        if (use_hold) {
            step = sim.step(tetris_v2::Action::Hold);
        } else {
            step = sim.apply_placement_index(placement_index);
        }

        float immediate = step.reward;
        if (!std::isfinite(immediate)) {
            immediate = 0.0f;
        }
        if (!step.action_succeeded) {
            return kFallbackFloor;
        }
        if (step.game_over) {
            return immediate + kFallbackFloor;
        }

        tetris_v2::cc::BotChoice choice{};
        tetris_v2::cc::BotThinkStats stats{};
        bool used_fallback = true;
        float future = kFallbackFloor;
        const auto remaining_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count();
        if (remaining_ms > 0) {
            const auto share_ms = remaining_ms / static_cast<std::int64_t>(slots_including_current);
            const int slot_think_ms = static_cast<int>(std::max<std::int64_t>(1, share_ms));
            if (bot->bot.sync_from_env(sim) && bot->bot.choose(slot_think_ms, &choice, &stats) &&
                stats.nodes > 0 && std::isfinite(choice.score)) {
                used_fallback = false;
                future = choice.score;
                nodes_total += stats.nodes;
                budget_miss_total += std::max(0, stats.budget_miss);
            }
        }

        if (used_fallback) {
            ++unexpanded_count;
            future = fallback_one_step_eval_score(sim);
        }

        float out = immediate + future;
        if (!std::isfinite(out)) {
            out = kFallbackFloor;
        }
        return out;
    };

    for (std::size_t i = 0; i < placement_slots; ++i) {
        legal_mask_out[i] = static_cast<std::uint8_t>(1);
        scores_out[i] = evaluate_slot(false, i);
    }

    if (hold_legal) {
        legal_mask_out[kRlPlacementSlots] = static_cast<std::uint8_t>(1);
        scores_out[kRlPlacementSlots] = evaluate_slot(true, 0);
    }

    for (std::size_t i = 0; i < kRlActionDim; ++i) {
        if (legal_mask_out[i] == 0) {
            continue;
        }
        if (!std::isfinite(scores_out[i])) {
            scores_out[i] = kFallbackFloor;
            ++unexpanded_count;
        }
    }

    // Leave bot synchronized with the caller env after ranking.
    (void)bot->bot.sync_from_env(env->env);

    auto t1 = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double elapsed_s = std::max(1.0e-9, elapsed_ms / 1000.0);
    const double nps = static_cast<double>(nodes_total) / elapsed_s;

    maybe_set_u64(nodes_out, nodes_total);
    maybe_set_double(think_ms_out, elapsed_ms);
    maybe_set_double(nps_out, nps);
    maybe_set_int(budget_miss_out, budget_miss_total);
    maybe_set_int(placement_count_raw_out, static_cast<int>(placement_count_raw));
    maybe_set_int(placement_overflow_out, placement_overflow ? 1 : 0);
    maybe_set_int(unexpanded_count_out, unexpanded_count);
    return 1;
}

int tetris_cc_bot_apply_choice(
    tetris_cc_bot_handle* bot,
    tetris_cc_env_handle* env,
    float* reward_out,
    int* lines_cleared_out,
    int* game_over_out,
    int* used_hold_out,
    size_t* placement_index_out) {
    if (!bot || !env) {
        return 0;
    }

    tetris_v2::StepResult result{};
    int used_hold = 0;
    std::size_t placement_index = 0;
    if (!bot->bot.apply_last_choice(env->env, &result, &used_hold, &placement_index)) {
        return 0;
    }

    if (reward_out) {
        *reward_out = result.reward;
    }
    maybe_set_int(lines_cleared_out, result.lines_cleared);
    maybe_set_int(game_over_out, result.game_over ? 1 : 0);
    maybe_set_int(used_hold_out, used_hold);
    if (placement_index_out) {
        *placement_index_out = placement_index;
    }
    return 1;
}

int tetris_cc_bot_choose_and_apply(
    tetris_cc_bot_handle* bot,
    tetris_cc_env_handle* env,
    int think_ms,
    float* reward_out,
    int* lines_cleared_out,
    int* game_over_out,
    int* used_hold_out,
    size_t* placement_index_out,
    float* score_out,
    std::uint64_t* nodes_out,
    double* think_ms_out,
    double* nps_out,
    int* budget_miss_out) {
    if (!bot || !env) {
        return 0;
    }

    tetris_v2::StepResult result{};
    tetris_v2::cc::BotChoice choice{};
    tetris_v2::cc::BotThinkStats stats{};
    int used_hold = 0;
    std::size_t placement_index = 0;
    if (!bot->bot.choose_and_apply(
            env->env,
            think_ms,
            &result,
            &choice,
            &stats,
            &used_hold,
            &placement_index)) {
        return 0;
    }

    if (reward_out) {
        *reward_out = result.reward;
    }
    maybe_set_int(lines_cleared_out, result.lines_cleared);
    maybe_set_int(game_over_out, result.game_over ? 1 : 0);
    maybe_set_int(used_hold_out, used_hold);
    if (placement_index_out) {
        *placement_index_out = placement_index;
    }
    if (score_out) {
        *score_out = choice.score;
    }
    maybe_set_u64(nodes_out, stats.nodes);
    maybe_set_double(think_ms_out, stats.think_ms);
    maybe_set_double(nps_out, stats.nps);
    maybe_set_int(budget_miss_out, stats.budget_miss);
    return 1;
}

void tetris_cc_battle_config_default(tetris_cc_battle_config* config_out) {
    if (!config_out) {
        return;
    }
    const tetris_v2::BattleConfig config{};
    config_out->seed = config.seed;
    for (std::size_t i = 0; i < config.attack_table.size(); ++i) {
        config_out->attack_table[i] = config.attack_table[i];
    }
    config_out->garbage_delay = config.garbage_delay;
    config_out->max_joint_steps = config.max_joint_steps;
    config_out->same_piece_sequence = config.same_piece_sequence ? 1 : 0;
}

tetris_cc_battle_handle* tetris_cc_battle_create(
    const tetris_cc_battle_config* config) {
    try {
        tetris_v2::BattleConfig native{};
        if (config) {
            native.seed = config->seed;
            for (std::size_t i = 0; i < native.attack_table.size(); ++i) {
                native.attack_table[i] = config->attack_table[i];
            }
            native.garbage_delay = config->garbage_delay;
            native.max_joint_steps = config->max_joint_steps;
            native.same_piece_sequence = config->same_piece_sequence != 0;
        }
        return new tetris_cc_battle_handle(native);
    } catch (...) {
        return nullptr;
    }
}

void tetris_cc_battle_destroy(tetris_cc_battle_handle* handle) { delete handle; }

int tetris_cc_battle_reset(tetris_cc_battle_handle* handle, uint32_t seed) {
    if (!handle) {
        return 0;
    }
    try {
        handle->battle.reset(seed);
        return 1;
    } catch (...) {
        return 0;
    }
}

size_t tetris_cc_battle_action_dim(void) { return tetris_v2::kDecisionActionDim; }

size_t tetris_cc_battle_observation_size(const tetris_cc_battle_handle* handle) {
    return handle ? tetris_v2::kBattleObservationSize : 0;
}

size_t tetris_cc_battle_observation_write(
    const tetris_cc_battle_handle* handle,
    size_t perspective_player,
    float* out,
    size_t out_len) {
    if (!handle || !out || perspective_player >= tetris_v2::kBattlePlayerCount) {
        return 0;
    }
    try {
        const auto observation = handle->battle.observation(perspective_player);
        const auto count = std::min(out_len, observation.size());
        std::copy(
            observation.begin(),
            observation.begin() + static_cast<std::ptrdiff_t>(count),
            out);
        return count;
    } catch (...) {
        return 0;
    }
}

size_t tetris_cc_battle_decision_mask_write(
    const tetris_cc_battle_handle* handle,
    size_t player,
    uint8_t* out,
    size_t out_len) {
    if (!handle || !out || player >= tetris_v2::kBattlePlayerCount ||
        out_len < tetris_v2::kDecisionActionDim) {
        return 0;
    }
    std::fill_n(
        out,
        static_cast<std::ptrdiff_t>(tetris_v2::kDecisionActionDim),
        static_cast<std::uint8_t>(0));
    const bool top_out_terminal =
        handle->battle.player_env(0).state().top_out ||
        handle->battle.player_env(1).state().top_out;
    // A max-step draw is a time-limit truncation in the Python API. Preserve
    // the legal successor mask so bootstrapping remains possible.
    if (!handle->battle.terminated() || !top_out_terminal) {
        for (const auto& decision : tetris_v2::enumerate_stable_decisions(
                 handle->battle.player_env(player))) {
            out[decision.action] = static_cast<std::uint8_t>(1);
        }
    }
    return tetris_v2::kDecisionActionDim;
}

int tetris_cc_battle_step(
    tetris_cc_battle_handle* handle,
    size_t player0_action,
    size_t player1_action,
    tetris_cc_battle_step_result* result_out) {
    if (!handle || !result_out) {
        return 0;
    }
    try {
        const auto result = handle->battle.step(player0_action, player1_action);
        *result_out = {};
        result_out->success = result.success ? 1 : 0;
        result_out->terminated = result.terminated ? 1 : 0;
        result_out->winner = result.winner;
        result_out->joint_step = result.joint_step;
        for (std::size_t player = 0; player < tetris_v2::kBattlePlayerCount; ++player) {
            const auto& native = result.players[player];
            auto& output = result_out->players[player];
            output.action_succeeded = native.action_succeeded ? 1 : 0;
            output.used_hold = native.used_hold ? 1 : 0;
            output.placement_index = native.placement_index;
            output.reward = native.reward;
            output.lines_cleared = native.lines_cleared;
            output.attack_generated = native.attack_generated;
            output.garbage_cancelled = native.garbage_cancelled;
            output.garbage_sent = native.garbage_sent;
            output.garbage_received = native.garbage_received;
            output.garbage_applied = native.garbage_applied;
            output.incoming_garbage = native.incoming_garbage;
            output.next_garbage_delay = native.next_garbage_delay;
            output.top_out = native.top_out ? 1 : 0;
        }
        return result.success ? 1 : 0;
    } catch (...) {
        *result_out = {};
        result_out->winner = -1;
        return 0;
    }
}

int tetris_cc_battle_meta_get(
    const tetris_cc_battle_handle* handle,
    tetris_cc_battle_meta* meta_out) {
    if (!handle || !meta_out) {
        return 0;
    }
    try {
        *meta_out = {};
        meta_out->joint_steps = handle->battle.joint_steps();
        meta_out->terminated = handle->battle.terminated() ? 1 : 0;
        meta_out->winner = handle->battle.winner();
        for (std::size_t player = 0; player < tetris_v2::kBattlePlayerCount; ++player) {
            meta_out->pending_garbage[player] = handle->battle.pending_garbage(player);
            meta_out->next_garbage_delay[player] =
                handle->battle.next_garbage_delay(player);
            const auto& native = handle->battle.stats(player);
            auto& output = meta_out->players[player];
            output.placements = native.placements;
            output.score = native.score;
            output.lines_cleared = native.lines_cleared;
            output.attack_generated = native.attack_generated;
            output.garbage_cancelled = native.garbage_cancelled;
            output.garbage_sent = native.garbage_sent;
            output.garbage_received = native.garbage_received;
            output.garbage_applied = native.garbage_applied;
            output.top_outs = native.top_outs;
        }
        return 1;
    } catch (...) {
        return 0;
    }
}

size_t tetris_cc_battle_board_write(
    const tetris_cc_battle_handle* handle,
    size_t player,
    int include_active,
    uint8_t* out,
    size_t out_len) {
    if (!handle || !out || player >= tetris_v2::kBattlePlayerCount) {
        return 0;
    }
    std::optional<tetris_v2::ActivePiece> active{};
    if (include_active != 0) {
        active = handle->battle.player_env(player).state().active;
    }
    return write_visible_board(
        handle->battle.player_env(player).state().board, active, out, out_len);
}

size_t tetris_cc_battle_board_piece_ids_write(
    const tetris_cc_battle_handle* handle,
    size_t player,
    int include_active,
    uint8_t* out,
    size_t out_len) {
    if (!handle || !out || player >= tetris_v2::kBattlePlayerCount) {
        return 0;
    }
    const auto ids =
        handle->battle.player_env(player).visible_board_piece_ids(include_active != 0);
    const auto count = std::min(out_len, ids.size());
    std::copy(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(count), out);
    return count;
}

int tetris_cc_battle_enqueue_garbage(
    tetris_cc_battle_handle* handle,
    size_t player,
    const int* hole_columns,
    size_t row_count,
    int delay) {
    if (!handle || player >= tetris_v2::kBattlePlayerCount ||
        (!hole_columns && row_count != 0)) {
        return 0;
    }
    try {
        std::vector<int> holes;
        if (row_count != 0) {
            holes.assign(hole_columns, hole_columns + row_count);
        }
        return handle->battle.enqueue_garbage(player, holes, delay) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

int tetris_cc_battle_bot_choose(
    tetris_cc_battle_handle* handle,
    size_t player,
    int think_ms,
    size_t* action_out,
    float* score_out,
    uint64_t* nodes_out,
    double* think_ms_out,
    double* nps_out,
    int* budget_miss_out) {
    if (!handle || !action_out || player >= tetris_v2::kBattlePlayerCount ||
        handle->battle.terminated()) {
        return 0;
    }
    try {
        auto& bot = handle->bots[player];
        const auto& env = handle->battle.player_env(player);
        if (!bot.sync_from_env(env)) {
            return 0;
        }
        tetris_v2::cc::BotChoice choice{};
        tetris_v2::cc::BotThinkStats stats{};
        if (!bot.choose(think_ms, &choice, &stats)) {
            return 0;
        }
        const auto action = tetris_v2::stable_decision_for_choice(
            env, choice.use_hold, choice.placement_index);
        if (!action.has_value()) {
            return 0;
        }
        *action_out = *action;
        if (score_out) {
            *score_out = choice.score;
        }
        maybe_set_u64(nodes_out, stats.nodes);
        maybe_set_double(think_ms_out, stats.think_ms);
        maybe_set_double(nps_out, stats.nps);
        maybe_set_int(budget_miss_out, stats.budget_miss);
        return 1;
    } catch (...) {
        return 0;
    }
}

}  // extern "C"
