#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tetris_cc_env_handle tetris_cc_env_handle;
typedef struct tetris_cc_bot_handle tetris_cc_bot_handle;
typedef struct tetris_cc_snapshot_handle tetris_cc_snapshot_handle;
typedef struct tetris_cc_battle_handle tetris_cc_battle_handle;

#define TETRIS_CC_GARBAGE_PIECE_ID 7
#define TETRIS_CC_BATTLE_PLAYER_COUNT 2
#define TETRIS_CC_BATTLE_ATTACK_TABLE_SIZE 5
/* Own 254 fields + opponent visible board 200 + normalized public battle 16. */
#define TETRIS_CC_BATTLE_OBSERVATION_SIZE 470

typedef struct tetris_cc_env_step_result {
    int action_succeeded;
    int piece_locked;
    int hold_used;
    int lines_cleared;
    int spin_clear;
    int spin_type;
    int difficult_clear;
    int b2b_bonus_applied;
    int combo;
    int back_to_back;
    float reward;
    int game_over;
    int top_out;
} tetris_cc_env_step_result;

/*
 * Clean-rewrite CC API surface (action-step primary).
 * Spin type encoding: 0=None, 1=Mini, 2=Full.
 */
tetris_cc_env_handle* tetris_cc_env_create(uint32_t seed);
/* Human-play environment: preserves modern 180-degree rotation support. */
tetris_cc_env_handle* tetris_cc_env_create_play(uint32_t seed);
void tetris_cc_env_destroy(tetris_cc_env_handle* handle);
void tetris_cc_env_reset(tetris_cc_env_handle* handle, uint32_t seed);
int tetris_cc_env_step(tetris_cc_env_handle* handle, int action, float* reward_out);
/* Returns 1 when the call is valid and writes the full result; otherwise 0. */
int tetris_cc_env_step_ex(
    tetris_cc_env_handle* handle, int action, tetris_cc_env_step_result* result_out);
/* Applies one action without advancing gravity or the lock timer. */
int tetris_cc_env_input_ex(
    tetris_cc_env_handle* handle, int action, tetris_cc_env_step_result* result_out);
/* Advances gravity and the lock timer exactly once with no action. */
int tetris_cc_env_tick_ex(
    tetris_cc_env_handle* handle, tetris_cc_env_step_result* result_out);
int tetris_cc_env_hold(tetris_cc_env_handle* handle, float* reward_out);

tetris_cc_snapshot_handle* tetris_cc_env_snapshot_create(const tetris_cc_env_handle* handle);
void tetris_cc_snapshot_destroy(tetris_cc_snapshot_handle* snapshot);
int tetris_cc_env_restore_snapshot(tetris_cc_env_handle* handle, const tetris_cc_snapshot_handle* snapshot);

size_t tetris_cc_env_observation_size(const tetris_cc_env_handle* handle, int include_hidden_rows);
size_t tetris_cc_env_observation_write(
    const tetris_cc_env_handle* handle, int include_hidden_rows, float* out, size_t out_len);
size_t tetris_cc_env_board_write(
    const tetris_cc_env_handle* handle, int include_active, uint8_t* out, size_t out_len);
size_t tetris_cc_env_board_piece_ids_write(
    const tetris_cc_env_handle* handle, int include_active, uint8_t* out, size_t out_len);

int tetris_cc_env_active_piece(
    const tetris_cc_env_handle* handle, int* piece, int* rotation, int* x, int* y);
int tetris_cc_env_ghost_piece(
    const tetris_cc_env_handle* handle, int* piece, int* rotation, int* x, int* landing_y);
int tetris_cc_env_hold_piece(
    const tetris_cc_env_handle* handle, int* has_hold, int* hold_piece, int* hold_available);
size_t tetris_cc_env_queue_count(const tetris_cc_env_handle* handle);
int tetris_cc_env_queue_get(const tetris_cc_env_handle* handle, size_t index, int* piece);
int tetris_cc_env_meta(
    const tetris_cc_env_handle* handle,
    int* game_over,
    int* top_out,
    int* combo,
    int* back_to_back,
    int* total_lines_cleared,
    int* lock_timer,
    int* lock_resets_used);

size_t tetris_cc_env_placement_count(const tetris_cc_env_handle* handle);
int tetris_cc_env_placement_get(
    const tetris_cc_env_handle* handle, size_t index, int* x, int* y, int* rotation, int* lines_cleared);
size_t tetris_cc_env_placement_board_write(
    const tetris_cc_env_handle* handle, size_t index, uint8_t* out, size_t out_len);
size_t tetris_cc_env_placement_board_piece_ids_write(
    const tetris_cc_env_handle* handle, size_t index, uint8_t* out, size_t out_len);
int tetris_cc_env_apply_placement_index(
    tetris_cc_env_handle* handle, size_t index, float* reward_out, int* lines_cleared_out, int* game_over_out);

/*
 * Stable placement decisions used by RL. Each action encodes
 * (use_hold, rotation, landing_y, x) and locks exactly one piece.
 */
size_t tetris_cc_env_decision_action_dim(void);
size_t tetris_cc_env_decision_mask_write(
    const tetris_cc_env_handle* handle, uint8_t* out, size_t out_len);
int tetris_cc_env_decision_get(
    const tetris_cc_env_handle* handle,
    size_t action,
    int* use_hold,
    size_t* placement_index,
    int* x,
    int* y,
    int* rotation);
int tetris_cc_env_decision_action_for_choice(
    const tetris_cc_env_handle* handle,
    int use_hold,
    size_t placement_index,
    size_t* action_out);
int tetris_cc_env_apply_decision(
    tetris_cc_env_handle* handle,
    size_t action,
    float* reward_out,
    int* lines_cleared_out,
    int* game_over_out,
    int* used_hold_out,
    size_t* placement_index_out);
int tetris_cc_env_last_clear_meta(
    const tetris_cc_env_handle* handle,
    int* spin_clear,
    int* difficult_clear,
    int* b2b_bonus_applied);
int tetris_cc_env_last_clear_spin_type(const tetris_cc_env_handle* handle, int* spin_type);

size_t tetris_cc_env_rotation_trace_count(const tetris_cc_env_handle* handle, int rotate_action);
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
    int* candidate_collides);
int tetris_cc_env_rotation_trace_meta(
    const tetris_cc_env_handle* handle,
    int rotate_action,
    int* success,
    int* final_x,
    int* final_y,
    int* final_rotation);

tetris_cc_bot_handle* tetris_cc_bot_create_default(void);
void tetris_cc_bot_destroy(tetris_cc_bot_handle* handle);
int tetris_cc_bot_sync_from_env(tetris_cc_bot_handle* bot, const tetris_cc_env_handle* env);
/* think_ms==0 runs 8 deterministic DAG work units; think_ms>0 is wall-clock budgeted. */
int tetris_cc_bot_choose(
    tetris_cc_bot_handle* bot,
    int think_ms,
    int* use_hold_out,
    size_t* placement_index_out,
    float* score_out,
    uint64_t* nodes_out,
    double* think_ms_out,
    double* nps_out,
    int* budget_miss_out);
// think_ms is the total search budget shared across all legal action scores.
int tetris_cc_bot_rank_actions(
    tetris_cc_bot_handle* bot,
    const tetris_cc_env_handle* env,
    int think_ms,
    float* scores_out,
    size_t scores_len,
    uint8_t* legal_mask_out,
    size_t legal_mask_len,
    uint64_t* nodes_out,
    double* think_ms_out,
    double* nps_out,
    int* budget_miss_out,
    int* placement_count_raw_out,
    int* placement_overflow_out,
    int* unexpanded_count_out);
int tetris_cc_bot_apply_choice(
    tetris_cc_bot_handle* bot,
    tetris_cc_env_handle* env,
    float* reward_out,
    int* lines_cleared_out,
    int* game_over_out,
    int* used_hold_out,
    size_t* placement_index_out);
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
    uint64_t* nodes_out,
    double* think_ms_out,
    double* nps_out,
    int* budget_miss_out);

/*
 * Deterministic two-player battle API. Joint actions use the same stable
 * 3,200-action (hold, rotation, y, x) encoding as the single-player API.
 * winner is 0 or 1, or -1 for a draw/ongoing match (inspect terminated).
 */
typedef struct tetris_cc_battle_config {
    uint32_t seed;
    int attack_table[TETRIS_CC_BATTLE_ATTACK_TABLE_SIZE];
    int garbage_delay;
    int max_joint_steps;
    int same_piece_sequence;
} tetris_cc_battle_config;

typedef struct tetris_cc_battle_player_step_result {
    int action_succeeded;
    int used_hold;
    size_t placement_index;
    float reward;
    int lines_cleared;
    int attack_generated;
    int garbage_cancelled;
    int garbage_sent;
    int garbage_received;
    int garbage_applied;
    int incoming_garbage;
    int next_garbage_delay;
    int top_out;
} tetris_cc_battle_player_step_result;

typedef struct tetris_cc_battle_step_result {
    int success;
    int terminated;
    int winner;
    int joint_step;
    tetris_cc_battle_player_step_result players[TETRIS_CC_BATTLE_PLAYER_COUNT];
} tetris_cc_battle_step_result;

typedef struct tetris_cc_battle_player_stats {
    int placements;
    float score;
    int lines_cleared;
    int attack_generated;
    int garbage_cancelled;
    int garbage_sent;
    int garbage_received;
    int garbage_applied;
    int top_outs;
} tetris_cc_battle_player_stats;

typedef struct tetris_cc_battle_meta {
    int joint_steps;
    int terminated;
    int winner;
    int pending_garbage[TETRIS_CC_BATTLE_PLAYER_COUNT];
    int next_garbage_delay[TETRIS_CC_BATTLE_PLAYER_COUNT];
    tetris_cc_battle_player_stats players[TETRIS_CC_BATTLE_PLAYER_COUNT];
} tetris_cc_battle_meta;

void tetris_cc_battle_config_default(tetris_cc_battle_config* config_out);
tetris_cc_battle_handle* tetris_cc_battle_create(const tetris_cc_battle_config* config);
void tetris_cc_battle_destroy(tetris_cc_battle_handle* handle);
int tetris_cc_battle_reset(tetris_cc_battle_handle* handle, uint32_t seed);
size_t tetris_cc_battle_action_dim(void);
size_t tetris_cc_battle_observation_size(const tetris_cc_battle_handle* handle);
size_t tetris_cc_battle_observation_write(
    const tetris_cc_battle_handle* handle,
    size_t perspective_player,
    float* out,
    size_t out_len);
size_t tetris_cc_battle_decision_mask_write(
    const tetris_cc_battle_handle* handle,
    size_t player,
    uint8_t* out,
    size_t out_len);
int tetris_cc_battle_step(
    tetris_cc_battle_handle* handle,
    size_t player0_action,
    size_t player1_action,
    tetris_cc_battle_step_result* result_out);
int tetris_cc_battle_meta_get(
    const tetris_cc_battle_handle* handle,
    tetris_cc_battle_meta* meta_out);
size_t tetris_cc_battle_board_write(
    const tetris_cc_battle_handle* handle,
    size_t player,
    int include_active,
    uint8_t* out,
    size_t out_len);
size_t tetris_cc_battle_board_piece_ids_write(
    const tetris_cc_battle_handle* handle,
    size_t player,
    int include_active,
    uint8_t* out,
    size_t out_len);

/* Test/evaluation pressure injection; normal play sends garbage via joint step. */
int tetris_cc_battle_enqueue_garbage(
    tetris_cc_battle_handle* handle,
    size_t player,
    const int* hole_columns,
    size_t row_count,
    int delay);

/*
 * Chooses a legal current stable action without mutating the match.
 * think_ms==0 runs 8 deterministic DAG work units; positive values retain the
 * existing wall-clock millisecond budget.
 */
int tetris_cc_battle_bot_choose(
    tetris_cc_battle_handle* handle,
    size_t player,
    int think_ms,
    size_t* action_out,
    float* score_out,
    uint64_t* nodes_out,
    double* think_ms_out,
    double* nps_out,
    int* budget_miss_out);
#ifdef __cplusplus
}
#endif
