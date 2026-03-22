#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tetris_cc_env_handle tetris_cc_env_handle;
typedef struct tetris_cc_bot_handle tetris_cc_bot_handle;
typedef struct tetris_cc_snapshot_handle tetris_cc_snapshot_handle;

/*
 * Clean-rewrite CC API surface (action-step primary).
 * Spin type encoding: 0=None, 1=Mini, 2=Full.
 */
tetris_cc_env_handle* tetris_cc_env_create(uint32_t seed);
void tetris_cc_env_destroy(tetris_cc_env_handle* handle);
void tetris_cc_env_reset(tetris_cc_env_handle* handle, uint32_t seed);
int tetris_cc_env_step(tetris_cc_env_handle* handle, int action, float* reward_out);
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
int tetris_cc_env_placement_get_ex(
    const tetris_cc_env_handle* handle,
    size_t index,
    int* x,
    int* y,
    int* rotation,
    int* lines_cleared,
    int* spin_candidate,
    int* difficult_candidate,
    int* last_rotate_used_kick);
size_t tetris_cc_env_placement_board_write(
    const tetris_cc_env_handle* handle, size_t index, uint8_t* out, size_t out_len);
size_t tetris_cc_env_placement_board_piece_ids_write(
    const tetris_cc_env_handle* handle, size_t index, uint8_t* out, size_t out_len);
int tetris_cc_env_apply_placement_index(
    tetris_cc_env_handle* handle, size_t index, float* reward_out, int* lines_cleared_out, int* game_over_out);
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
int tetris_cc_bot_choose_ex(
    tetris_cc_bot_handle* bot,
    int think_ms,
    int* use_hold_out,
    size_t* placement_index_out,
    float* score_out,
    uint64_t* nodes_out,
    double* think_ms_out,
    double* nps_out,
    int* budget_miss_out);
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
int tetris_cc_bot_choose_and_apply_ex(
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

#ifdef __cplusplus
}
#endif
