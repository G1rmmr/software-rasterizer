#pragma once
#include "../math/Vector.hpp"

namespace graphics {
    // Model-space data. Tangent.W is handedness, not a fourth spatial coordinate.
    struct Vertex {
        math::Vector Pos{0.f, 0.f, 0.f, 1.f};
        math::Vector Normal{0.f, 0.f, 1.f, 0.f};
        math::Vector Color{1.f, 1.f, 1.f, 1.f};
        math::Vector UV{};
        math::Vector Tangent{1.f, 0.f, 0.f, 1.f};
    };
}
