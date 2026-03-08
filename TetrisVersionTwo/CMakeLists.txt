cmake_minimum_required(VERSION 3.16)
project(TetrisVersionTwo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(
    tetris_v2
    src/board.cpp
    src/piece_defs.cpp
    src/randomizer.cpp
    src/env.cpp
    src/observation.cpp
)

target_include_directories(tetris_v2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(tetris_v2 PUBLIC cxx_std_17)
set_target_properties(tetris_v2 PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(tetris_v2_c_api SHARED src/c_api.cpp)
target_include_directories(tetris_v2_c_api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(tetris_v2_c_api PUBLIC tetris_v2)
target_compile_features(tetris_v2_c_api PUBLIC cxx_std_17)
set_target_properties(tetris_v2_c_api PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_library(tetris_v2_c_api_static STATIC src/c_api.cpp)
target_include_directories(tetris_v2_c_api_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(tetris_v2_c_api_static PUBLIC tetris_v2)
target_compile_features(tetris_v2_c_api_static PUBLIC cxx_std_17)

add_executable(tetris_cli apps/cli_play.cpp)
target_link_libraries(tetris_cli PRIVATE tetris_v2)

add_executable(tetris_smoke tests/smoke_env.cpp)
target_link_libraries(tetris_smoke PRIVATE tetris_v2)

add_executable(tetris_kick_search_tests tests/kick_search_tests.cpp)
target_link_libraries(tetris_kick_search_tests PRIVATE tetris_v2)

add_executable(tetris_c_api_tests tests/c_api_tests.cpp)
target_link_libraries(tetris_c_api_tests PRIVATE tetris_v2 tetris_v2_c_api_static)
