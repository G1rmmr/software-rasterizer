#pragma once

#include <algorithm>

#include "../math/Math.hpp"

namespace shader {
    struct Vertex {
        math::Vector Pos;
        math::Vector Normal;
        math::Vector Color;
    };

    struct Default {
        math::Matrix Model;
        math::Matrix MVP;
        math::Matrix Viewport;
        math::Vector LightDir;

        inline math::Vector Vertex(const math::Vector& pos) const {
            const math::Vector clipPos = MVP * pos;

            if(clipPos.W < 0.1f) {
                return math::Vector(-10000.f, -10000.f, 0.f, 1.f);
            }

            const float invW = (std::abs(clipPos.W) > 1e-6f) ? (1.f / clipPos.W) : 1.f;
            const math::Vector ndcPos(clipPos.X * invW, clipPos.Y * invW, clipPos.Z * invW, 1.f);

            return Viewport * ndcPos;
        }

        inline math::Vector Normal(const math::Vector& normal) const {
            math::Vector n = Model * math::Vector(normal.X, normal.Y, normal.Z, 0.f);
            return n.Norm();
        }

        inline std::uint32_t Color(const math::Vector& color, const math::Vector& normal) const {
            const float intensity = std::max(normal.Dot(LightDir), 0.1f);

            simd::Floats lDir = simd::Set(intensity, intensity, intensity, 1.f);
            simd::Floats cFloats = simd::Mul(color.V, lDir);

            cFloats = simd::Mul(cFloats, simd::Set(255.f));
            return simd::PackRGBA(simd::Clamp(cFloats, simd::Set(0.f), simd::Set(255.f)));
        }
    };
}