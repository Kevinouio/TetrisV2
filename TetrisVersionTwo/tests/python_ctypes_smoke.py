import ctypes
from pathlib import Path


BOARD_ROWS = 20
BOARD_COLS = 10

ACTION_NONE = 0
ACTION_CW = 5


def find_library() -> Path:
    candidates = [
        Path("build/TetrisVersionTwo/tetris_v2_c_api.dll"),
        Path("build/TetrisVersionTwo/Debug/tetris_v2_c_api.dll"),
        Path("build/TetrisVersionTwo/Release/tetris_v2_c_api.dll"),
        Path("build/TetrisVersionTwo/libtetris_v2_c_api.so"),
        Path("build/TetrisVersionTwo/libtetris_v2_c_api.dylib"),
        Path("TetrisVersionTwo/build/tetris_v2_c_api.dll"),
        Path("TetrisVersionTwo/build/Debug/tetris_v2_c_api.dll"),
        Path("TetrisVersionTwo/build/Release/tetris_v2_c_api.dll"),
        Path("TetrisVersionTwo/build/libtetris_v2_c_api.so"),
        Path("TetrisVersionTwo/build/libtetris_v2_c_api.dylib"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find tetris_v2_c_api shared library")


def main():
    lib = ctypes.CDLL(str(find_library()))
    void_p = ctypes.c_void_p
    int_p = ctypes.POINTER(ctypes.c_int)

    lib.tetris_env_create.argtypes = [ctypes.c_uint32]
    lib.tetris_env_create.restype = void_p
    lib.tetris_env_destroy.argtypes = [void_p]
    lib.tetris_env_destroy.restype = None
    lib.tetris_env_reset.argtypes = [void_p, ctypes.c_uint32]
    lib.tetris_env_reset.restype = None
    lib.tetris_env_step.argtypes = [void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    lib.tetris_env_step.restype = ctypes.c_int
    lib.tetris_env_board_write.argtypes = [void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    lib.tetris_env_board_write.restype = ctypes.c_size_t
    lib.tetris_env_queue_count.argtypes = [void_p]
    lib.tetris_env_queue_count.restype = ctypes.c_size_t
    lib.tetris_env_placement_count.argtypes = [void_p]
    lib.tetris_env_placement_count.restype = ctypes.c_size_t
    lib.tetris_env_apply_placement_index.argtypes = [void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_float), int_p, int_p]
    lib.tetris_env_apply_placement_index.restype = ctypes.c_int
    lib.tetris_env_rotation_trace_count.argtypes = [void_p, ctypes.c_int]
    lib.tetris_env_rotation_trace_count.restype = ctypes.c_size_t

    handle = lib.tetris_env_create(ctypes.c_uint32(123))
    if not handle:
        raise RuntimeError("Failed to create env")

    try:
        for _ in range(120):
            board = (ctypes.c_uint8 * (BOARD_ROWS * BOARD_COLS))()
            written = lib.tetris_env_board_write(handle, 0, board, len(board))
            assert written == len(board)

            queue_count = int(lib.tetris_env_queue_count(handle))
            assert queue_count >= 0

            placement_count = int(lib.tetris_env_placement_count(handle))
            assert placement_count >= 0

            trace_count = int(lib.tetris_env_rotation_trace_count(handle, ACTION_CW))
            assert trace_count >= 0

            if placement_count > 0:
                reward = ctypes.c_float(0.0)
                lines = ctypes.c_int(0)
                game_over = ctypes.c_int(0)
                ok = lib.tetris_env_apply_placement_index(
                    handle, 0, ctypes.byref(reward), ctypes.byref(lines), ctypes.byref(game_over)
                )
                assert ok in (0, 1)
            else:
                reward = ctypes.c_float(0.0)
                lib.tetris_env_step(handle, ACTION_NONE, ctypes.byref(reward))
    finally:
        lib.tetris_env_destroy(handle)

    print("python_ctypes_smoke: PASS")


if __name__ == "__main__":
    main()
