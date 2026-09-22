#pragma once
#include "../math/Math.hpp"
#include <cmath>
#include <stdexcept>

namespace graphics {
    class DirectionalLight final {
    public:
        explicit DirectionalLight(const math::Vector& direction = {-.5f, 1.f, 1.f, 0.f}) { SetDirection(direction); }
        void SetDirection(const math::Vector& direction) {
            const float length = direction.Length();
            if(!std::isfinite(length) || !std::isfinite(direction.X) || !std::isfinite(direction.Y) ||
               !std::isfinite(direction.Z) || length <= 1e-6f)
                throw std::invalid_argument("Light direction must be finite and nonzero");
            direction_ = math::Vector(direction.X, direction.Y, direction.Z, 0.f).Norm();
        }
        [[nodiscard]] const math::Vector& GetDirection() const noexcept { return direction_; }

    private:
        math::Vector direction_;
    };
}
