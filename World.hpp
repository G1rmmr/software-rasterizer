#pragma once

#include <cstdint>

#include "graphics/FrameBuffer.hpp"
#include "graphics/Shader.hpp"
#include "math/Math.hpp"

namespace world {
    constexpr inline std::uint32_t WIDTH = 800;
    constexpr inline std::uint32_t HEIGHT = 450;
    constexpr inline std::uint32_t COLOR = 0xFF333333;

    inline std::vector<shader::Vertex> ModelVertices = {
        {{-1.f, -1.f, 1.f, 1.f}, {1.f, 0.f, 0.f, 1.f}},  // 0
        {{1.f, -1.f, 1.f, 1.f}, {0.f, 1.f, 0.f, 1.f}},   // 1
        {{1.f, 1.f, 1.f, 1.f}, {0.f, 0.f, 1.f, 1.f}},    // 2
        {{-1.f, 1.f, 1.f, 1.f}, {1.f, 1.f, 0.f, 1.f}},   // 3
        {{-1.f, -1.f, -1.f, 1.f}, {1.f, 0.f, 1.f, 1.f}}, // 4
        {{1.f, -1.f, -1.f, 1.f}, {0.f, 1.f, 1.f, 1.f}},  // 5
        {{1.f, 1.f, -1.f, 1.f}, {1.f, 1.f, 1.f, 1.f}},   // 6
        {{-1.f, 1.f, -1.f, 1.f}, {0.f, 0.f, 0.f, 1.f}}   // 7
    };

    inline std::vector<std::uint32_t> ModelIndices = {
        0, 1, 2, 0, 2, 3, // front
        1, 5, 6, 1, 6, 2, // right
        5, 4, 7, 5, 7, 6, // rear
        4, 0, 3, 4, 3, 7, // left
        3, 2, 6, 3, 6, 7, // top
        4, 5, 1, 4, 1, 0  // bottom
    };

    inline math::Matrix GetMVP(const float angle) {
        math::Matrix model = math::CreateRotation({0.f, 1.f, 0.f}, angle);

        math::Vector target(0.f, 0.f, 0.f);
        math::Matrix view = math::CreateLookAt({0.f, 0.f, 5.f}, target, {0.f, 1.f, 0.f});

        float aspect = static_cast<float>(WIDTH) / HEIGHT;
        math::Matrix proj = math::CreatePerspective(math::ToRadian(45.f), aspect, 0.1f, 100.f);

        return proj * view * model;
    }
}