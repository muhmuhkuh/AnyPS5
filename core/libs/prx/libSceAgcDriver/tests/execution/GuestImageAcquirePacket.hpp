#pragma once
#include <array>
#include <cstdint>

constexpr std::array<std::uint32_t, 8> ImageBdaAcquireFullCache() {
    return {0xc0065800u, 0u, 0xffffffffu, 0xffu, 0u, 0u, 10u, 0xc200u};
}
