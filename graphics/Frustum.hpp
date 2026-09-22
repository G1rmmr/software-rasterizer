#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

#include "../math/Math.hpp"

namespace graphics {
    class Frustum {
    public:
        Frustum() { Update(math::Matrix()); }
        explicit Frustum(const math::Matrix& viewProjection) { Update(viewProjection); }

        void Update(const math::Matrix& viewProjection) {
            const auto row = [&](int index) {
                return math::Vector(viewProjection[0][index], viewProjection[1][index], viewProjection[2][index],
                                    viewProjection[3][index]);
            };
            const auto x = row(0), y = row(1), z = row(2), w = row(3);
            const std::array<math::Vector, 6> equations = {w + x, w - x, w + y, w - y, z, w - z};
            std::array<Plane, 6> next;
            for(std::size_t i = 0; i < equations.size(); ++i) {
                const auto& equation = equations[i];
                const float length = equation.Length();
                if(!std::isfinite(length) || length <= 1e-8f || !std::isfinite(equation.W))
                    throw std::invalid_argument("Frustum requires six finite nondegenerate planes");
                next[i] = {math::Vector(equation.X / length, equation.Y / length, equation.Z / length),
                           equation.W / length};
            }
            planes = next;
        }

        [[nodiscard]] bool IsSphereInside(const math::Vector& center, float radius) const noexcept {
            if(!std::isfinite(center.X) || !std::isfinite(center.Y) || !std::isfinite(center.Z) ||
               !std::isfinite(radius) || radius < 0.f)
                return false;
            for(const auto& plane : planes)
                if(plane.normal.Dot(center) + plane.distance < -radius) return false;
            return true;
        }

    private:
        struct Plane {
            math::Vector normal;
            float distance;
        };
        std::array<Plane, 6> planes;
    };
}
