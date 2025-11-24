#pragma once

#include <string_view>

#ifdef PROFILE_COLD_CLEAR
#include <chrono>

namespace cold_clear_cpp {

class ProfileScope {
public:
    explicit ProfileScope(std::string_view name)
        : name_(name), start_(std::chrono::steady_clock::now()) {}
    ~ProfileScope() = default;

private:
    std::string_view name_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace cold_clear_cpp

#define PROFILE_SCOPE(name) ::cold_clear_cpp::ProfileScope profile_scope_##__LINE__(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__func__)
#else
#define PROFILE_SCOPE(name) ((void)0)
#define PROFILE_FUNCTION() ((void)0)
#endif
