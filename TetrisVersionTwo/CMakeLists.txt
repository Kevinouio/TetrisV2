cmake_minimum_required(VERSION 3.16)

add_subdirectory(cold_clear_cpp)

add_library(tetris_v2 STATIC
    src/tetris_env.cpp
    src/batch_env.cpp
    src/agent.cpp
    src/cc_agent.cpp
)

target_include_directories(tetris_v2 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cold_clear_cpp/include
)

target_compile_features(tetris_v2 PUBLIC cxx_std_17)
target_link_libraries(tetris_v2 PUBLIC cold_clear_cpp)

add_executable(tetris_v2_tests
    tests/test_tetris_env.cpp
)
target_link_libraries(tetris_v2_tests PRIVATE tetris_v2)
target_compile_features(tetris_v2_tests PRIVATE cxx_std_17)
add_test(NAME tetris_v2_tests COMMAND tetris_v2_tests)

add_executable(tetris_cc_tests
    tests/test_cc_agent.cpp
)
target_link_libraries(tetris_cc_tests PRIVATE tetris_v2)
target_compile_features(tetris_cc_tests PRIVATE cxx_std_17)
add_test(NAME tetris_cc_tests COMMAND tetris_cc_tests)

add_executable(cc_play
    cli/cc_play.cpp
)
target_link_libraries(cc_play PRIVATE tetris_v2)
target_compile_features(cc_play PRIVATE cxx_std_17)
