#pragma once

#include "Object.hpp"

class Diablo : public Object {
public:
    inline void Create() override { createIcoSphere(2.f, 3); }

private:
    inline void createIcoSphere(const float radius, const std::uint32_t subdivisions) {
        std::vector<shader::Vertex> vertices;
        std::vector<std::uint32_t> indices;

        vertices.reserve(10000);
        indices.reserve(10000);

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

            vertices.push_back({pos, normal, {color.X, color.Y, color.Z, 1.f}, {}, {}});
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

                std::uint32_t a = getMidpoint(radius, v1, v2, midpointCache, vertices);
                std::uint32_t b = getMidpoint(radius, v2, v3, midpointCache, vertices);
                std::uint32_t c = getMidpoint(radius, v3, v1, midpointCache, vertices);

                nextFaces.insert(nextFaces.end(), {v1, a, c, v2, b, a, v3, c, b, a, b, c});
            }
            faces = std::move(nextFaces);
        }
        indices = std::move(faces);

        this->Meshes.push_back(graphics::Mesh{vertices, indices, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr});
    }

    inline std::uint32_t getMidpoint(const float radius, const std::uint32_t p1, const std::uint32_t p2,
                                     std::map<std::uint64_t, std::uint32_t>& cache,
                                     std::vector<shader::Vertex>& vertices) {
        std::uint64_t key = (static_cast<std::uint64_t>(std::min(p1, p2)) << 32) | std::max(p1, p2);
        if(cache.contains(key)) return cache[key];

        math::Vector middle = (vertices[p1].Pos + vertices[p2].Pos) * 0.5f;
        const float len = middle.Length();

        math::Vector normal = (len > 1e-6f) ? (middle / len) : vertices[p1].Normal;
        math::Vector pos = normal * radius;
        pos.W = 1.f;
        normal.W = 0.f;

        math::Vector color = (vertices[p1].Color + vertices[p2].Color) * 0.5f;
        vertices.push_back({pos, normal, color});
        return cache[key] = static_cast<std::uint32_t>(vertices.size() - 1);
    }
};