#pragma once

#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace tetris_v2::bit_ops {

inline unsigned int countl_zero_u64(std::uint64_t value) noexcept {
    if (value == 0) {
        return 64u;
    }

#if defined(_MSC_VER)
    unsigned long index = 0;
#if defined(_M_X64) || defined(_M_ARM64) || defined(_M_ARM64EC)
    _BitScanReverse64(&index, static_cast<unsigned __int64>(value));
    return 63u - static_cast<unsigned int>(index);
#else
    const auto high = static_cast<unsigned long>(value >> 32u);
    if (high != 0) {
        _BitScanReverse(&index, high);
        return 31u - static_cast<unsigned int>(index);
    }
    _BitScanReverse(&index, static_cast<unsigned long>(value));
    return 63u - static_cast<unsigned int>(index);
#endif
#elif defined(__GNUC__) || defined(__clang__)
    return static_cast<unsigned int>(__builtin_clzll(value));
#else
    unsigned int count = 0;
    for (std::uint64_t mask = std::uint64_t{1} << 63u; (value & mask) == 0; mask >>= 1u) {
        ++count;
    }
    return count;
#endif
}

inline unsigned int countr_zero_u64(std::uint64_t value) noexcept {
    if (value == 0) {
        return 64u;
    }

#if defined(_MSC_VER)
    unsigned long index = 0;
#if defined(_M_X64) || defined(_M_ARM64) || defined(_M_ARM64EC)
    _BitScanForward64(&index, static_cast<unsigned __int64>(value));
    return static_cast<unsigned int>(index);
#else
    const auto low = static_cast<unsigned long>(value);
    if (low != 0) {
        _BitScanForward(&index, low);
        return static_cast<unsigned int>(index);
    }
    _BitScanForward(&index, static_cast<unsigned long>(value >> 32u));
    return 32u + static_cast<unsigned int>(index);
#endif
#elif defined(__GNUC__) || defined(__clang__)
    return static_cast<unsigned int>(__builtin_ctzll(value));
#else
    unsigned int count = 0;
    while ((value & 1u) == 0) {
        value >>= 1u;
        ++count;
    }
    return count;
#endif
}

inline unsigned int popcount_u64(std::uint64_t value) noexcept {
#if defined(_MSC_VER) && defined(_M_X64)
    return static_cast<unsigned int>(__popcnt64(static_cast<unsigned __int64>(value)));
#elif defined(_MSC_VER) && defined(_M_IX86)
    return static_cast<unsigned int>(__popcnt(static_cast<unsigned int>(value))) +
        static_cast<unsigned int>(__popcnt(static_cast<unsigned int>(value >> 32u)));
#elif defined(__GNUC__) || defined(__clang__)
    return static_cast<unsigned int>(__builtin_popcountll(value));
#else
    value -= (value >> 1u) & 0x5555555555555555ull;
    value = (value & 0x3333333333333333ull) + ((value >> 2u) & 0x3333333333333333ull);
    value = (value + (value >> 4u)) & 0x0F0F0F0F0F0F0F0Full;
    return static_cast<unsigned int>((value * 0x0101010101010101ull) >> 56u);
#endif
}

}  // namespace tetris_v2::bit_ops
