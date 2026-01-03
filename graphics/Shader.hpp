#pragma once

#include <algorithm>

#include "../math/Math.hpp"

namespace shader {
    struct Vertex {
        math::Vector Pos;
        math::Vector Color;
    };

    struct Default {
        math::Matrix MVP;
        math::Matrix Viewport;

        inline math::Vector Vertex(const math::Vector& pos) const {
            const math::Vector clipPos = MVP * pos;

            const float invW = (std::abs(clipPos.W) > 1e-6f) ? (1.f / clipPos.W) : 1.f;
            const math::Vector ndcPos(clipPos.X * invW, clipPos.Y * invW, clipPos.Z * invW, 1.f);

            return Viewport * ndcPos;
        }

        inline std::uint32_t Color(const math::Vector& color) const {
            auto Byte = [](float v) -> std::uint32_t {
                if(v <= 0.0f) return 0;
                if(v >= 1.0f) return 255;
                return static_cast<std::uint32_t>(v * 255.0f + 0.5f);
            };

            return (Byte(color.W) << 24) | (Byte(color.Z) << 16) | (Byte(color.Y) << 8) | Byte(color.X);
        }
    };
}