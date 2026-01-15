#pragma once

#include <array>

#include "../math/Math.hpp"

namespace graphics {
    struct Plane {
        math::Vector Normal;
        float Dist;

        void Normalize() {
            float len = std::sqrt(Normal.X * Normal.X + Normal.Y * Normal.Y + Normal.Z * Normal.Z);
            Normal = Normal * (1.f / len);
            Dist /= len;
        }

        float Distance(const math::Vector& pt) const { return Normal.Dot(pt) + Dist; }
    };

    class Frustum {
    public:
        std::array<Plane, 6> planes;

        void Update(const math::Matrix& vp) {
            const auto& m = vp;

            // Left
            planes[0].Normal.X = m[3][0] + m[0][0];
            planes[0].Normal.Y = m[3][1] + m[0][1];
            planes[0].Normal.Z = m[3][2] + m[0][2];
            planes[0].Normal = m[3][3] + m[0][3];

            // Right
            planes[1].Normal.X = m[3][0] - m[0][0];
            planes[1].Normal.Y = m[3][1] - m[0][1];
            planes[1].Normal.Z = m[3][2] - m[0][2];
            planes[1].Dist = m[3][3] - m[0][3];

            // Bottom
            planes[2].Normal.X = m[3][0] + m[1][0];
            planes[2].Normal.Y = m[3][1] + m[1][1];
            planes[2].Normal.Z = m[3][2] + m[1][2];
            planes[2].Dist = m[3][3] + m[1][3];

            // Top
            planes[3].Normal.X = m[3][0] - m[1][0];
            planes[3].Normal.Y = m[3][1] - m[1][1];
            planes[3].Normal.Z = m[3][2] - m[1][2];
            planes[3].Dist = m[3][3] - m[1][3];

            // Near
            planes[4].Normal.X = m[3][0] + m[2][0];
            planes[4].Normal.Y = m[3][1] + m[2][1];
            planes[4].Normal.Z = m[3][2] + m[2][2];
            planes[4].Dist = m[3][3] + m[2][3];

            // Far
            planes[5].Normal.X = m[3][0] - m[2][0];
            planes[5].Normal.Y = m[3][1] - m[2][1];
            planes[5].Normal.Z = m[3][2] - m[2][2];
            planes[5].Dist = m[3][3] - m[2][3];

            for(Plane& plane : planes) plane.Normalize();
        }

        bool IsSphereInside(const math::Vector& center, float radius) const {
            for(const Plane& plane : planes)
                if(plane.Distance(center) < -radius) return false;

            return true;
        }
    };
}