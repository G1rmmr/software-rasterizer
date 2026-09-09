#pragma once
#include "../math/Math.hpp"
#include "Frustum.hpp"
#include <cmath>
#include <stdexcept>

namespace graphics {
    class Camera final {
    public:
        Camera(float fov, float aspect, float nearPlane, float farPlane) {
            SetPerspective(fov, aspect, nearPlane, farPlane);
        }
        void SetPerspective(float fov, float aspect, float nearPlane, float farPlane) {
            const auto projection = math::CreatePerspective(fov, aspect, nearPlane, farPlane);
            // Invert this projection's known sparse form without a generic
            // determinant threshold silently producing a zero inverse.
            math::Matrix inverse(0.f);
            inverse[0][0] = 1.f / projection[0][0];
            inverse[1][1] = 1.f / projection[1][1];
            inverse[3][2] = -1.f;
            inverse[2][3] = 1.f / projection[3][2];
            inverse[3][3] = projection[2][2] / projection[3][2];
            for(int column = 0; column < 4; ++column)
                for(int row = 0; row < 4; ++row)
                    if(!std::isfinite(projection[column][row]) || !std::isfinite(inverse[column][row]))
                        throw std::invalid_argument("Camera projection exceeds finite float precision");
            Frustum frustum(projection * view_);
            projection_ = projection;
            inverseProjection_ = inverse;
            frustum_ = frustum;
        }
        void LookAt(const math::Vector& eye, const math::Vector& target, const math::Vector& up = {0.f, 1.f, 0.f}) {
            const auto finite = [](const math::Vector& v) {
                return std::isfinite(v.X) && std::isfinite(v.Y) && std::isfinite(v.Z);
            };
            const float distance = (eye - target).Length();
            const auto forward = (eye - target).NormalizedDirection();
            const auto normalizedUp = up.NormalizedDirection();
            const float crossLength = normalizedUp.Cross(forward).Length();
            if(!finite(eye) || !finite(target) || !finite(up) || !std::isfinite(distance) ||
               !std::isfinite(crossLength) || distance < 1e-6f || crossLength < 1e-6f)
                throw std::invalid_argument("Camera needs finite, distinct eye/target and a nonparallel up direction");
            const auto view = math::CreateLookAt(eye, target, normalizedUp);
            Frustum frustum(projection_ * view);
            position_ = {eye.X, eye.Y, eye.Z, 1.f};
            view_ = view;
            frustum_ = frustum;
        }
        [[nodiscard]] const math::Matrix& GetView() const noexcept { return view_; }
        [[nodiscard]] const math::Matrix& GetProjection() const noexcept { return projection_; }
        [[nodiscard]] const math::Matrix& GetInverseProjection() const noexcept { return inverseProjection_; }
        [[nodiscard]] const math::Vector& GetPosition() const noexcept { return position_; }
        [[nodiscard]] const Frustum& GetFrustum() const noexcept { return frustum_; }

    private:
        math::Vector position_{0.f, 0.f, 0.f, 1.f};
        math::Matrix view_;
        math::Matrix projection_;
        math::Matrix inverseProjection_;
        Frustum frustum_;
    };
}
