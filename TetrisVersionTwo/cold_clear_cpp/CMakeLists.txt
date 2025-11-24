cmake_minimum_required(VERSION 3.16)
project(cold_clear_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

add_library(cold_clear_cpp
    src/board.cpp
    src/state.cpp
    src/movegen.cpp
    src/eval.cpp
    src/dag.cpp
    src/bot.cpp
    src/tbp.cpp
)

target_include_directories(cold_clear_cpp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(cold_clear_cpp PUBLIC Threads::Threads)

execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE COLD_CLEAR_CPP_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if (COLD_CLEAR_CPP_GIT_HASH STREQUAL "")
    set(COLD_CLEAR_CPP_GIT_HASH "unknown")
endif()

target_compile_definitions(cold_clear_cpp
    PUBLIC
        COLD_CLEAR_CPP_VERSION="${PROJECT_VERSION}"
        COLD_CLEAR_CPP_GIT_HASH="${COLD_CLEAR_CPP_GIT_HASH}"
)

enable_testing()

add_executable(cold_clear_cpp_tests
    tests/test_all.cpp
)

target_link_libraries(cold_clear_cpp_tests PRIVATE cold_clear_cpp)

add_test(NAME cold_clear_cpp_tests COMMAND cold_clear_cpp_tests)

add_executable(cold_clear_cpp_tbp
    src/main.cpp
)

target_link_libraries(cold_clear_cpp_tbp PRIVATE cold_clear_cpp)
