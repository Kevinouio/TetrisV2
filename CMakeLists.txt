cmake_minimum_required(VERSION 3.16)
project(TetrisVersionOneCCEnv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/TetrisVersionOne/cc_like_env/src/cc_like_env.cpp)
    add_subdirectory(TetrisVersionOne/cc_like_env)
endif()

add_subdirectory(TetrisVersionTwo)
