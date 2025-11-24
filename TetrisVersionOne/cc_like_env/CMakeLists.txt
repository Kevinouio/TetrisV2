add_library(cc_like_env STATIC
    src/cc_like_env.cpp
)

target_include_directories(cc_like_env PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(cc_like_env PUBLIC cxx_std_17)
