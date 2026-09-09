#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "../math/Vector.hpp"

namespace graphics {
    // Normalized straight RGBA becomes an ARGB integer, independently of SIMD
    // lane order and the current floating-point rounding mode.
    [[nodiscard]] inline std::uint32_t PackColor(const math::Vector& rgba) noexcept {
        const auto byte = [](float channel) -> std::uint32_t {
            if(std::isnan(channel)) return 0;
            return static_cast<std::uint32_t>(std::floor(std::clamp(channel, 0.f, 1.f) * 255.f + 0.5f));
        };
        return (byte(rgba.W) << 24) | (byte(rgba.X) << 16) | (byte(rgba.Y) << 8) | byte(rgba.Z);
    }

    // Straight-alpha source-over, including destinations with partial alpha.
    // With an opaque destination RGB uses /255 with rounding, not /256.
    [[nodiscard]] inline std::uint32_t AlphaBlend(std::uint32_t source, std::uint32_t destination) noexcept {
        const std::uint32_t sourceAlpha = source >> 24;
        if(sourceAlpha == 0) return destination;
        if(sourceAlpha == 255) return source;
        const std::uint32_t destinationAlpha = destination >> 24;
        const std::uint32_t inverseAlpha = 255 - sourceAlpha;
        const std::uint32_t alphaNumerator = sourceAlpha * 255 + destinationAlpha * inverseAlpha;
        const auto channel = [&](unsigned shift) {
            const std::uint32_t sourceChannel = (source >> shift) & 255;
            const std::uint32_t destinationChannel = (destination >> shift) & 255;
            const std::uint32_t numerator =
                sourceChannel * sourceAlpha * 255 + destinationChannel * destinationAlpha * inverseAlpha;
            return (numerator + alphaNumerator / 2) / alphaNumerator;
        };
        return (((alphaNumerator + 127) / 255) << 24) | (channel(16) << 16) | (channel(8) << 8) | channel(0);
    }
}
