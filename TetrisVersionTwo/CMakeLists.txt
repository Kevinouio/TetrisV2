cmake_minimum_required(VERSION 3.16)
project(TetrisVersionTwo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter)

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

add_executable(tetris_b2b_spin_tests tests/b2b_spin_tests.cpp)
target_link_libraries(tetris_b2b_spin_tests PRIVATE tetris_v2)

set(TWIST_CASES_JSON ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/twist_cases_srs.json)
set(TWIST_CASES_GENERATOR ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/generate_twist_cases.py)
set(TWIST_CASES_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/tests/generated/twist_cases_generated.hpp)

if(Python3_Interpreter_FOUND)
    add_custom_command(
        OUTPUT ${TWIST_CASES_HEADER}
        COMMAND ${Python3_EXECUTABLE} ${TWIST_CASES_GENERATOR} ${TWIST_CASES_JSON} ${TWIST_CASES_HEADER}
        DEPENDS ${TWIST_CASES_JSON} ${TWIST_CASES_GENERATOR}
        COMMENT "Generating twist dataset header"
        VERBATIM
    )
    add_custom_target(tetris_twist_dataset_gen DEPENDS ${TWIST_CASES_HEADER})
else()
    message(WARNING "Python3 not found; using checked-in twist dataset header without regeneration.")
    add_custom_target(tetris_twist_dataset_gen)
endif()

add_executable(tetris_twist_dataset_tests tests/twist_dataset_tests.cpp)
add_dependencies(tetris_twist_dataset_tests tetris_twist_dataset_gen)
target_link_libraries(tetris_twist_dataset_tests PRIVATE tetris_v2)

add_executable(tetris_c_api_tests tests/c_api_tests.cpp)
target_link_libraries(tetris_c_api_tests PRIVATE tetris_v2 tetris_v2_c_api_static)
