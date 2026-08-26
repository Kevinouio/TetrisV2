cmake_minimum_required(VERSION 3.16)
project(TetrisV2 LANGUAGES CXX)

include(CTest)
find_package(Threads REQUIRED)

set(CMAKE_CXX_EXTENSIONS OFF)
option(TETRIS_STRICT_WARNINGS "Enable extra compiler warnings" ON)

add_library(tetris_v2
    src/board.cpp
    src/cc2_data.cpp
    src/cc2_dag.cpp
    src/cc2_eval.cpp
    src/cc2_movegen.cpp
    src/cc2_sync.cpp
    src/cc_env.cpp
    src/cold_clear_bot.cpp
    src/env.cpp
    src/observation.cpp
    src/piece_defs.cpp
    src/randomizer.cpp
)
target_include_directories(tetris_v2 PUBLIC include)
target_compile_features(tetris_v2 PUBLIC cxx_std_17)
target_link_libraries(tetris_v2 PUBLIC Threads::Threads)
set_target_properties(tetris_v2 PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(tetris_v2_c_api SHARED src/c_api.cpp)
target_include_directories(tetris_v2_c_api PUBLIC include)
target_compile_features(tetris_v2_c_api PRIVATE cxx_std_17)
target_link_libraries(tetris_v2_c_api PUBLIC tetris_v2)
set_target_properties(tetris_v2_c_api PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(TETRIS_STRICT_WARNINGS)
    if(MSVC)
        target_compile_options(tetris_v2 PRIVATE /W4)
        target_compile_options(tetris_v2_c_api PRIVATE /W4)
    else()
        target_compile_options(tetris_v2 PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(tetris_v2_c_api PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

add_executable(tetris_bot_cli apps/cli_bot_play.cpp)
target_link_libraries(tetris_bot_cli PRIVATE tetris_v2)

if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    set(TWIST_CASES_JSON ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/twist_cases_srs.json)
    set(TWIST_CASES_GENERATOR ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/generate_twist_cases.py)
    set(TWIST_CASES_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/twist_cases_generated.hpp)

    add_custom_command(
        OUTPUT ${TWIST_CASES_HEADER}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/generated
        COMMAND ${Python3_EXECUTABLE} ${TWIST_CASES_GENERATOR} ${TWIST_CASES_JSON} ${TWIST_CASES_HEADER}
        DEPENDS ${TWIST_CASES_JSON} ${TWIST_CASES_GENERATOR}
        COMMENT "Generating twist dataset header"
        VERBATIM
    )
    add_custom_target(tetris_twist_dataset_gen DEPENDS ${TWIST_CASES_HEADER})

    function(tetris_add_test target source)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE tetris_v2 ${ARGN})
        if(MSVC)
            target_compile_options(${target} PRIVATE /UNDEBUG)
        else()
            target_compile_options(${target} PRIVATE -UNDEBUG)
        endif()
        if(TETRIS_STRICT_WARNINGS)
            if(MSVC)
                target_compile_options(${target} PRIVATE /W4)
            else()
                target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
            endif()
        endif()
        add_test(NAME ${target} COMMAND ${target})
    endfunction()

    tetris_add_test(tetris_smoke tests/cpp/smoke_env.cpp)
    tetris_add_test(tetris_kick_search_tests tests/cpp/kick_search_tests.cpp)
    tetris_add_test(tetris_b2b_spin_tests tests/cpp/b2b_spin_tests.cpp)
    tetris_add_test(tetris_c_api_tests tests/cpp/c_api_tests.cpp tetris_v2_c_api)

    tetris_add_test(tetris_twist_dataset_tests tests/cpp/twist_dataset_tests.cpp)
    add_dependencies(tetris_twist_dataset_tests tetris_twist_dataset_gen)
    target_include_directories(tetris_twist_dataset_tests PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
endif()
