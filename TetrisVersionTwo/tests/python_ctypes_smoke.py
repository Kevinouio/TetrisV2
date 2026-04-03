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

    lib.tetris_cc_env_create.argtypes = [ctypes.c_uint32]
    lib.tetris_cc_env_create.restype = void_p
    lib.tetris_cc_env_destroy.argtypes = [void_p]
    lib.tetris_cc_env_destroy.restype = None
    lib.tetris_cc_env_reset.argtypes = [void_p, ctypes.c_uint32]
    lib.tetris_cc_env_reset.restype = None
    lib.tetris_cc_env_step.argtypes = [void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    lib.tetris_cc_env_step.restype = ctypes.c_int
    lib.tetris_cc_env_board_write.argtypes = [void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    lib.tetris_cc_env_board_write.restype = ctypes.c_size_t
    lib.tetris_cc_env_queue_count.argtypes = [void_p]
    lib.tetris_cc_env_queue_count.restype = ctypes.c_size_t
    lib.tetris_cc_env_placement_count.argtypes = [void_p]
    lib.tetris_cc_env_placement_count.restype = ctypes.c_size_t
    lib.tetris_cc_env_apply_placement_index.argtypes = [void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_float), int_p, int_p]
    lib.tetris_cc_env_apply_placement_index.restype = ctypes.c_int
    lib.tetris_cc_env_rotation_trace_count.argtypes = [void_p, ctypes.c_int]
    lib.tetris_cc_env_rotation_trace_count.restype = ctypes.c_size_t

    has_bot_api = all(
        hasattr(lib, name)
        for name in (
            "tetris_cc_bot_create_default",
            "tetris_cc_bot_destroy",
            "tetris_cc_bot_sync_from_env",
            "tetris_cc_bot_choose_and_apply",
        )
    )
    has_bot_api_ex = has_bot_api and all(
        hasattr(lib, name)
        for name in (
            "tetris_cc_bot_choose_ex",
            "tetris_cc_bot_choose_and_apply_ex",
        )
    )

    if has_bot_api:
        size_p = ctypes.POINTER(ctypes.c_size_t)
        u64_p = ctypes.POINTER(ctypes.c_uint64)
        dbl_p = ctypes.POINTER(ctypes.c_double)

        lib.tetris_cc_bot_create_default.argtypes = []
        lib.tetris_cc_bot_create_default.restype = void_p
        lib.tetris_cc_bot_destroy.argtypes = [void_p]
        lib.tetris_cc_bot_destroy.restype = None
        lib.tetris_cc_bot_sync_from_env.argtypes = [void_p, void_p]
        lib.tetris_cc_bot_sync_from_env.restype = ctypes.c_int
        lib.tetris_cc_bot_choose_and_apply.argtypes = [
            void_p,
            void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            int_p,
            int_p,
            int_p,
            size_p,
            ctypes.POINTER(ctypes.c_float),
            u64_p,
            dbl_p,
            dbl_p,
            int_p,
        ]
        lib.tetris_cc_bot_choose_and_apply.restype = ctypes.c_int
        if has_bot_api_ex:
            lib.tetris_cc_bot_choose_and_apply_ex.argtypes = [
                void_p,
                void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                int_p,
                int_p,
                int_p,
                size_p,
                ctypes.POINTER(ctypes.c_float),
                u64_p,
                dbl_p,
                dbl_p,
                int_p,
            ]
            lib.tetris_cc_bot_choose_and_apply_ex.restype = ctypes.c_int

    handle = lib.tetris_cc_env_create(ctypes.c_uint32(123))
    if not handle:
        raise RuntimeError("Failed to create env")

    bot = None
    if has_bot_api:
        bot = lib.tetris_cc_bot_create_default()
        if not bot:
            raise RuntimeError("Failed to create bot")
        assert lib.tetris_cc_bot_sync_from_env(bot, handle) == 1

    try:
        for _ in range(120):
            board = (ctypes.c_uint8 * (BOARD_ROWS * BOARD_COLS))()
            written = lib.tetris_cc_env_board_write(handle, 0, board, len(board))
            assert written == len(board)

            queue_count = int(lib.tetris_cc_env_queue_count(handle))
            assert queue_count >= 0

            placement_count = int(lib.tetris_cc_env_placement_count(handle))
            assert placement_count >= 0

            trace_count = int(lib.tetris_cc_env_rotation_trace_count(handle, ACTION_CW))
            assert trace_count >= 0

            if has_bot_api and bot:
                reward = ctypes.c_float(0.0)
                lines = ctypes.c_int(0)
                game_over = ctypes.c_int(0)
                used_hold = ctypes.c_int(0)
                placement_index = ctypes.c_size_t(0)
                score = ctypes.c_float(0.0)
                nodes = ctypes.c_uint64(0)
                think_ms = ctypes.c_double(0.0)
                nps = ctypes.c_double(0.0)
                budget_miss = ctypes.c_int(0)
                if has_bot_api_ex:
                    ok = lib.tetris_cc_bot_choose_and_apply_ex(
                        bot,
                        handle,
                        5,
                        ctypes.byref(reward),
                        ctypes.byref(lines),
                        ctypes.byref(game_over),
                        ctypes.byref(used_hold),
                        ctypes.byref(placement_index),
                        ctypes.byref(score),
                        ctypes.byref(nodes),
                        ctypes.byref(think_ms),
                        ctypes.byref(nps),
                        ctypes.byref(budget_miss),
                    )
                else:
                    ok = lib.tetris_cc_bot_choose_and_apply(
                        bot,
                        handle,
                        5,
                        ctypes.byref(reward),
                        ctypes.byref(lines),
                        ctypes.byref(game_over),
                        ctypes.byref(used_hold),
                        ctypes.byref(placement_index),
                        ctypes.byref(score),
                        ctypes.byref(nodes),
                        ctypes.byref(think_ms),
                        ctypes.byref(nps),
                        ctypes.byref(budget_miss),
                    )
                assert ok == 1
                assert int(lines.value) >= 0
                assert int(lines.value) <= 4
                assert int(nodes.value) > 0
                if has_bot_api_ex:
                    assert int(budget_miss.value) in (0, 1)
                if game_over.value:
                    lib.tetris_cc_env_reset(handle, ctypes.c_uint32(123))
                    assert lib.tetris_cc_bot_sync_from_env(bot, handle) == 1
            elif placement_count > 0:
                reward = ctypes.c_float(0.0)
                lines = ctypes.c_int(0)
                game_over = ctypes.c_int(0)
                ok = lib.tetris_cc_env_apply_placement_index(
                    handle, 0, ctypes.byref(reward), ctypes.byref(lines), ctypes.byref(game_over)
                )
                assert ok in (0, 1)
            else:
                reward = ctypes.c_float(0.0)
                lib.tetris_cc_env_step(handle, ACTION_NONE, ctypes.byref(reward))
    finally:
        if has_bot_api and bot:
            lib.tetris_cc_bot_destroy(bot)
        lib.tetris_cc_env_destroy(handle)

    print("python_ctypes_smoke: PASS")


if __name__ == "__main__":
    main()
