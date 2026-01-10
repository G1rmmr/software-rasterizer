#pragma once

#include <cstdint>
#include <map>

#include "graphics/FrameBuffer.hpp"
#include "graphics/Shader.hpp"
#include "math/Math.hpp"

namespace world {
    constexpr inline std::uint32_t WIDTH = 800;
    constexpr inline std::uint32_t HEIGHT = 450;
    constexpr inline std::uint32_t COLOR = 0xFF000000;

    const inline math::Vector UP(0.f, 1.f, 0.f);
    const inline math::Vector EYE(0.f, 0.f, 5.f);

    constexpr inline float FOV_ANGLE = 45.f;
    constexpr inline float NEAR = 0.1f;
    constexpr inline float FAR = 100.f;

    inline shader::Uniforms Uniform;
    inline math::Vector Target(0.f, 0.f, 0.f);

    inline std::vector<shader::Vertex> ModelVertices = {
        // Front face (+Z)
        {{-1.f, -1.f, 1.f, 1.f}, {0.f, 0.f, 1.f, 0.f}, {1.f, 0.f, 0.f, 1.f}}, // 0
        {{1.f, -1.f, 1.f, 1.f}, {0.f, 0.f, 1.f, 0.f}, {0.f, 1.f, 0.f, 1.f}},  // 1
        {{1.f, 1.f, 1.f, 1.f}, {0.f, 0.f, 1.f, 0.f}, {0.f, 0.f, 1.f, 1.f}},   // 2
        {{-1.f, 1.f, 1.f, 1.f}, {0.f, 0.f, 1.f, 0.f}, {1.f, 1.f, 0.f, 1.f}},  // 3

        // Back face (-Z)
        {{1.f, -1.f, -1.f, 1.f}, {0.f, 0.f, -1.f, 0.f}, {0.f, 1.f, 1.f, 1.f}},  // 4
        {{-1.f, -1.f, -1.f, 1.f}, {0.f, 0.f, -1.f, 0.f}, {1.f, 0.f, 1.f, 1.f}}, // 5
        {{-1.f, 1.f, -1.f, 1.f}, {0.f, 0.f, -1.f, 0.f}, {0.f, 0.f, 0.f, 1.f}},  // 6
        {{1.f, 1.f, -1.f, 1.f}, {0.f, 0.f, -1.f, 0.f}, {1.f, 1.f, 1.f, 1.f}},   // 7

        // Top face (+Y)
        {{-1.f, 1.f, 1.f, 1.f}, {0.f, 1.f, 0.f, 0.f}, {1.f, 1.f, 0.f, 1.f}},  // 8
        {{1.f, 1.f, 1.f, 1.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 1.f}},   // 9
        {{1.f, 1.f, -1.f, 1.f}, {0.f, 1.f, 0.f, 0.f}, {1.f, 1.f, 1.f, 1.f}},  // 10
        {{-1.f, 1.f, -1.f, 1.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 1.f}}, // 11

        // Bottom face (-Y)
        {{-1.f, -1.f, -1.f, 1.f}, {0.f, -1.f, 0.f, 0.f}, {1.f, 0.f, 1.f, 1.f}}, // 12
        {{1.f, -1.f, -1.f, 1.f}, {0.f, -1.f, 0.f, 0.f}, {0.f, 1.f, 1.f, 1.f}},  // 13
        {{1.f, -1.f, 1.f, 1.f}, {0.f, -1.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 1.f}},   // 14
        {{-1.f, -1.f, 1.f, 1.f}, {0.f, -1.f, 0.f, 0.f}, {1.f, 0.f, 0.f, 1.f}},  // 15

        // Right face (+X)
        {{1.f, -1.f, 1.f, 1.f}, {1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 1.f}},  // 16
        {{1.f, -1.f, -1.f, 1.f}, {1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 1.f, 1.f}}, // 17
        {{1.f, 1.f, -1.f, 1.f}, {1.f, 0.f, 0.f, 0.f}, {1.f, 1.f, 1.f, 1.f}},  // 18
        {{1.f, 1.f, 1.f, 1.f}, {1.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 1.f}},   // 19

        // Left face (-X)
        {{-1.f, -1.f, -1.f, 1.f}, {-1.f, 0.f, 0.f, 0.f}, {1.f, 0.f, 1.f, 1.f}}, // 20
        {{-1.f, -1.f, 1.f, 1.f}, {-1.f, 0.f, 0.f, 0.f}, {1.f, 0.f, 0.f, 1.f}},  // 21
        {{-1.f, 1.f, 1.f, 1.f}, {-1.f, 0.f, 0.f, 0.f}, {1.f, 1.f, 0.f, 1.f}},   // 22
        {{-1.f, 1.f, -1.f, 1.f}, {-1.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 1.f}}   // 23
    };

    inline std::vector<std::uint32_t> ModelIndices = {
        0,  1,  2,  0,  2,  3,  // Front
        4,  5,  6,  4,  6,  7,  // Back
        8,  9,  10, 8,  10, 11, // Top
        12, 13, 14, 12, 14, 15, // Bottom
        16, 17, 18, 16, 18, 19, // Right
        20, 21, 22, 20, 22, 23  // Left
    };

    inline std::uint32_t GetMidpoint(const float radius, const std::uint32_t p1, const std::uint32_t p2,
                                     std::map<std::uint64_t, std::uint32_t>& cache) {
        std::uint64_t key = (static_cast<std::uint64_t>(std::min(p1, p2)) << 32) | std::max(p1, p2);
        if(cache.contains(key)) return cache[key];

        math::Vector middle = (ModelVertices[p1].Pos + ModelVertices[p2].Pos) * 0.5f;
        const float len = middle.Length();

        math::Vector normal = (len > 1e-6f) ? (middle / len) : ModelVertices[p1].Normal;
        math::Vector pos = normal * radius;
        pos.W = 1.f;
        normal.W = 0.f;

        math::Vector color = (ModelVertices[p1].Color + ModelVertices[p2].Color) * 0.5f;
        ModelVertices.push_back({pos, normal, color});
        return cache[key] = static_cast<std::uint32_t>(ModelVertices.size() - 1);
    }

    inline void CreateIcoSphere(const float radius, const std::uint32_t subdivisions) {
        ModelVertices.clear();
        ModelIndices.clear();

        const float t = (1.f + std::sqrt(5.f)) / 2.f;
        std::vector<math::Vector> basePos = {{-1, t, 0},  {1, t, 0},  {-1, -t, 0}, {1, -t, 0}, {0, -1, t},  {0, 1, t},
                                             {0, -1, -t}, {0, 1, -t}, {t, 0, -1},  {t, 0, 1},  {-t, 0, -1}, {-t, 0, 1}};

        for(auto& p : basePos) {
            float len = std::sqrt(p.Dot(p));
            math::Vector normal = (p / len);
            math::Vector pos = normal * radius;

            pos.W = 1.f;
            normal.W = 0.f;

            math::Vector color = math::CreateRandomVector(0.f, 1.f);
            color.W = 1.f;

            ModelVertices.push_back({pos, normal, {color.X, color.Y, color.Z, 1.f}});
        }

        std::vector<std::uint32_t> faces = {0, 11, 5,  0, 5,  1, 0, 1, 7, 0, 7,  10, 0, 10, 11, 1, 5, 9, 5, 11,
                                            4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3,  9,  4, 3,  4,  2, 3, 2, 6, 3,
                                            6, 8,  3,  8, 9,  4, 9, 5, 2, 4, 11, 6,  2, 10, 8,  6, 7, 9, 8, 1};

        std::map<std::uint64_t, std::uint32_t> midpointCache;

        for(std::uint32_t i = 0; i < subdivisions; ++i) {
            std::vector<std::uint32_t> nextFaces;
            for(std::size_t j = 0; j < faces.size(); j += 3) {
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