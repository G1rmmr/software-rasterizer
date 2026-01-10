#pragma once

#include <cstdint>
#include <map>

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

        constexpr float aspect = static_cast<float>(WIDTH) / HEIGHT;
        math::Matrix proj = math::CreatePerspective(math::ToRadian(45.f), aspect, 0.1f, 100.f);

        return proj * view * model;
    }

    inline std::uint32_t GetMidpoint(
        const float radius,
        const std::uint32_t p1, 
        const std::uint32_t p2,
        std::map<std::uint64_t, std::uint32_t>& cache) {
        std::uint64_t key = (static_cast<std::uint64_t>(std::min(p1, p2)) << 32) | std::max(p1, p2);
        if (cache.contains(key)) return cache[key];

        math::Vector v1 = {ModelVertices[p1].Pos.X, ModelVertices[p1].Pos.Y, ModelVertices[p1].Pos.Z};
        math::Vector v2 = {ModelVertices[p2].Pos.X, ModelVertices[p2].Pos.Y, ModelVertices[p2].Pos.Z};
        
        math::Vector middle = (v1 + v2) * 0.5f;
        const float length = std::sqrt(middle.Dot(middle));
        middle = (middle / length) * radius;

        ModelVertices.push_back({{middle.X, middle.Y, middle.Z, 1.f}, {1.f, 1.f, 1.f, 1.f}});
        return cache[key] = static_cast<std::uint32_t>(ModelVertices.size() - 1);
    }

    inline void CreateIcoSphere(const float radius, const std::uint32_t subdivisions) {
        ModelVertices.clear();
        ModelIndices.clear();

        const float t = (1.0f + std::sqrt(5.0f)) / 2.0f;
        std::vector<math::Vector> basePos = {
            {-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0},
            {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t},
            {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}
        };

        for (auto& p : basePos) {
            float len = std::sqrt(p.Dot(p));
            math::Vector n = (p / len) * radius;
            ModelVertices.push_back({{n.X, n.Y, n.Z, 1.f}, {1.f, 1.f, 1.f, 1.f}});
        }

        std::vector<std::uint32_t> faces = {
            0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
            1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
            3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
            4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1
        };

        std::map<std::uint64_t, std::uint32_t> midpointCache;

        for (std::uint32_t i = 0; i < subdivisions; ++i) {
            std::vector<std::uint32_t> nextFaces;
            for (std::size_t j = 0; j < faces.size(); j += 3) {
                std::uint32_t v1 = faces[j];
                std::uint32_t v2 = faces[j + 1];
                std::uint32_t v3 = faces[j + 2];

                std::uint32_t a = GetMidpoint(radius, v1, v2, midpointCache);
                std::uint32_t b = GetMidpoint(radius, v2, v3, midpointCache);
                std::uint32_t c = GetMidpoint(radius, v3, v1, midpointCache);

                nextFaces.insert(nextFaces.end(), {v1, a, c, v2, b, a, v3, c, b, a, b, c});
            }
            faces = std::move(nextFaces);
        }
        ModelIndices = std::move(faces);
    }
}