add_executable(cc_like_env_tests
    cc_like_env_tests.cpp
)

target_link_libraries(cc_like_env_tests PRIVATE cc_like_env)
target_compile_features(cc_like_env_tests PRIVATE cxx_std_17)

add_test(NAME cc_like_env_tests COMMAND cc_like_env_tests)
