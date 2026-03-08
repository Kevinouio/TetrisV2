#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tetris_env_handle tetris_env_handle;

tetris_env_handle* tetris_env_create(uint32_t seed);
void tetris_env_destroy(tetris_env_handle* handle);

void tetris_env_reset(tetris_env_handle* handle, uint32_t seed);
int tetris_env_step(tetris_env_handle* handle, int action, float* reward_out);
int tetris_env_hold(tetris_env_handle* handle, float* reward_out);

size_t tetris_env_observation_size(const tetris_env_handle* handle, int include_hidden_rows);
size_t tetris_env_observation_write(
    const tetris_env_handle* handle, int include_hidden_rows, float* out, size_t out_len);

// Writes visible board occupancy (20x10 = 200 bytes), row-major top-to-bottom.
// include_active != 0 overlays the active piece cells as occupied.
size_t tetris_env_board_write(
    const tetris_env_handle* handle, int include_active, uint8_t* out, size_t out_len);

int tetris_env_active_piece(
    const tetris_env_handle* handle, int* piece, int* rotation, int* x, int* y);
int tetris_env_hold_piece(
    const tetris_env_handle* handle, int* has_hold, int* hold_piece, int* hold_available);

size_t tetris_env_queue_count(const tetris_env_handle* handle);
int tetris_env_queue_get(const tetris_env_handle* handle, size_t index, int* piece);

int tetris_env_meta(
    const tetris_env_handle* handle,
    int* game_over,
    int* top_out,
    int* combo,
    int* back_to_back,
    int* total_lines_cleared,
    int* lock_timer,
    int* lock_resets_used);

size_t tetris_env_placement_count(const tetris_env_handle* handle);
int tetris_env_placement_get(
    const tetris_env_handle* handle, size_t index, int* x, int* y, int* rotation, int* lines_cleared);
size_t tetris_env_placement_board_write(
    const tetris_env_handle* handle, size_t index, uint8_t* out, size_t out_len);
int tetris_env_apply_placement_index(
    tetris_env_handle* handle, size_t index, float* reward_out, int* lines_cleared_out, int* game_over_out);

size_t tetris_env_rotation_trace_count(const tetris_env_handle* handle, int rotate_action);
int tetris_env_rotation_trace_get(
    const tetris_env_handle* handle,
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
int tetris_env_rotation_trace_meta(
    const tetris_env_handle* handle,
    int rotate_action,
    int* success,
    int* final_x,
    int* final_y,
    int* final_rotation);

#ifdef __cplusplus
}
#endif
