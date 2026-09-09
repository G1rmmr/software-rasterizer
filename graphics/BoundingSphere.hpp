#pragma once

#include "../math/Vector.hpp"

namespace graphics {
    struct BoundingSphere {
        math::Vector Center{0.f, 0.f, 0.f, 1.f};
        float Radius = 0.f;
    };
}
