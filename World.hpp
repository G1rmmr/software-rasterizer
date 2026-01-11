#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "graphics/FrameBuffer.hpp"
#include "graphics/Mesh.hpp"
#include "graphics/Shader.hpp"
#include "graphics/Texture.hpp"

#include "math/Math.hpp"

namespace world {
    constexpr inline std::uint32_t WIDTH = 1024;
    constexpr inline std::uint32_t HEIGHT = 1024;
    constexpr inline std::uint32_t COLOR = 0xFF222222;

    const inline math::Vector UP(0.f, 1.f, 0.f);
    const inline math::Vector EYE(0.f, 0.f, 5.f);

    constexpr inline float FOV_ANGLE = 45.f;
    constexpr inline float NEAR = 0.1f;
    constexpr inline float FAR = 100.f;

    inline shader::Uniforms Uniform;
    inline math::Vector Target(0.f, 0.f, 0.f);
    inline std::map<std::string, std::vector<graphics::Mesh>> SubMeshes = {};

    namespace {
        inline std::uint32_t GetMidpoint(const float radius, const std::uint32_t p1, const std::uint32_t p2,
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

        inline void LoadObj(const std::string& filePath, std::vector<shader::Vertex>& vertices,
                            std::vector<std::uint32_t>& indices) {
            std::ifstream file(filePath);
            if(!file.is_open()) return;

            std::vector<math::Vector> tempVertices;
            std::vector<math::Vector> tempNormals;
            std::vector<math::Vector> tempUVs;

            tempVertices.reserve(10000);
            tempNormals.reserve(10000);
            tempUVs.reserve(10000);

            std::map<std::string, std::uint32_t> vertexCache;

            std::string line;
            while(std::getline(file, line)) {
                if(line.empty()) continue;
                std::stringstream ss(line);
                std::string type;
                ss >> type;

                if(type == "v") {
                    float x = 0.f;
                    float y = 0.f;
                    float z = 0.f;

                    ss >> x >> y >> z;
                    tempVertices.push_back({x, y, z, 1.0f});
                }
                else if(type == "vt") {
                    float u = 0.f;
                    float v = 0.f;

                    ss >> u >> v;
                    tempUVs.push_back({u, v, 0.f, 0.f});
                }
                else if(type == "vn") {
                    float x = 0.f;
                    float y = 0.f;
                    float z = 0.f;

                    ss >> x >> y >> z;
                    tempNormals.push_back({x, y, z, 0.0f});
                }
                else if(type == "f") {
                    std::string vertexData;
                    for(int i = 0; i < 3; ++i) {
                        ss >> vertexData;

                        if(vertexCache.find(vertexData) != vertexCache.end()) {
                            indices.push_back(vertexCache[vertexData]);
                            continue;
                        }

                        std::stringstream vss(vertexData);
                        std::string index;

                        std::int32_t vIdx = -1;
                        std::int32_t vtIdx = -1;
                        std::int32_t vnIdx = -1;

                        if(std::getline(vss, index, '/')) {
                            if(!index.empty()) vIdx = std::stoi(index) - 1;
                        }
                        if(std::getline(vss, index, '/')) {
                            if(!index.empty()) vtIdx = std::stoi(index) - 1;
                        }
                        if(std::getline(vss, index, '/')) {
                            if(!index.empty()) vnIdx = std::stoi(index) - 1;
                        }

                        math::Vector pos =
                            (vIdx >= 0 && vIdx < tempVertices.size()) ? tempVertices[vIdx] : math::Vector{0, 0, 0, 1};
                        math::Vector uv =
                            (vtIdx >= 0 && vtIdx < tempUVs.size()) ? tempUVs[vtIdx] : math::Vector{0, 0, 0, 0};
                        math::Vector normal =
                            (vnIdx >= 0 && vnIdx < tempNormals.size()) ? tempNormals[vnIdx] : math::Vector{0, 0, 1, 0};

                        vertices.push_back({pos, normal, {1.f, 1.f, 1.f, 1.f}, uv, {0, 0, 0, 0}});

                        std::uint32_t newIdx = static_cast<std::uint32_t>(vertices.size() - 1);
                        indices.push_back(newIdx);
                        vertexCache[vertexData] = newIdx;
                    }
                }
            }

            std::vector<math::Vector> tan1(vertices.size(), math::Vector(0, 0, 0, 0));
            std::vector<math::Vector> tan2(vertices.size(), math::Vector(0, 0, 0, 0));

            for(size_t i = 0; i < indices.size(); i += 3) {
                long i1 = indices[i];
                long i2 = indices[i + 1];
                long i3 = indices[i + 2];

                const math::Vector& v1 = vertices[i1].Pos;
                const math::Vector& v2 = vertices[i2].Pos;
                const math::Vector& v3 = vertices[i3].Pos;

                const math::Vector& w1 = vertices[i1].UV;
                const math::Vector& w2 = vertices[i2].UV;
                const math::Vector& w3 = vertices[i3].UV;

                float x1 = v2.X - v1.X;
                float x2 = v3.X - v1.X;
                float y1 = v2.Y - v1.Y;
                float y2 = v3.Y - v1.Y;
                float z1 = v2.Z - v1.Z;
                float z2 = v3.Z - v1.Z;

                float s1 = w2.X - w1.X;
                float s2 = w3.X - w1.X;
                float t1 = w2.Y - w1.Y;
                float t2 = w3.Y - w1.Y;

                float r = 1.f / (s1 * t2 - s2 * t1);
                math::Vector sdir((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r, 0.f);
                math::Vector tdir((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r, 0.f);

                tan1[i1] = tan1[i1] + sdir;
                tan1[i2] = tan1[i2] + sdir;
                tan1[i3] = tan1[i3] + sdir;

                tan2[i1] = tan2[i1] + tdir;
                tan2[i2] = tan2[i2] + tdir;
                tan2[i3] = tan2[i3] + tdir;
            }

            for(long a = 0; a < vertices.size(); a++) {
                const math::Vector& n = vertices[a].Normal;
                const math::Vector& t = tan1[a];

                math::Vector xyz = (t - n * n.Dot(t)).Norm();

                float w = (n.Cross(t).Dot(tan2[a]) < 0.f) ? -1.f : 1.f;
                vertices[a].Tangent = {xyz.X, xyz.Y, xyz.Z, w};
            }
        }
    }

    inline void CreateCube() {
        std::vector<shader::Vertex> vertices = {
            // Front face (+Z)
            {{-1.f, -1.f, 1.f, 1.f}, {0.f, 0.f, 1.f, 0.f}, {1.f, 0.f, 0.f, 1.f}, {}, {}}, // 0
            {{1.f, -1.f, 1.f, 1.f}, {0.f, 0.f, 1.f, 0.f}, {0.f, 1.f, 0.f, 1.f}, {}, {}},  // 1
            {{1.f, 1.f, 1.f, 1.f}, {0.f, 0.f, 1.f, 0.f}, {0.f, 0.f, 1.f, 1.f}, {}, {}},   // 2
            {{-1.f, 1.f, 1.f, 1.f}, {0.f, 0.f, 1.f, 0.f}, {1.f, 1.f, 0.f, 1.f}, {}, {}},  // 3

            // Back face (-Z)
            {{1.f, -1.f, -1.f, 1.f}, {0.f, 0.f, -1.f, 0.f}, {0.f, 1.f, 1.f, 1.f}, {}, {}},  // 4
            {{-1.f, -1.f, -1.f, 1.f}, {0.f, 0.f, -1.f, 0.f}, {1.f, 0.f, 1.f, 1.f}, {}, {}}, // 5
            {{-1.f, 1.f, -1.f, 1.f}, {0.f, 0.f, -1.f, 0.f}, {0.f, 0.f, 0.f, 1.f}, {}, {}},  // 6
            {{1.f, 1.f, -1.f, 1.f}, {0.f, 0.f, -1.f, 0.f}, {1.f, 1.f, 1.f, 1.f}, {}, {}},   // 7

            // Top face (+Y)
            {{-1.f, 1.f, 1.f, 1.f}, {0.f, 1.f, 0.f, 0.f}, {1.f, 1.f, 0.f, 1.f}, {}, {}},  // 8
            {{1.f, 1.f, 1.f, 1.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 1.f}, {}, {}},   // 9
            {{1.f, 1.f, -1.f, 1.f}, {0.f, 1.f, 0.f, 0.f}, {1.f, 1.f, 1.f, 1.f}, {}, {}},  // 10
            {{-1.f, 1.f, -1.f, 1.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 1.f}, {}, {}}, // 11

            // Bottom face (-Y)
            {{-1.f, -1.f, -1.f, 1.f}, {0.f, -1.f, 0.f, 0.f}, {1.f, 0.f, 1.f, 1.f}, {}, {}}, // 12
            {{1.f, -1.f, -1.f, 1.f}, {0.f, -1.f, 0.f, 0.f}, {0.f, 1.f, 1.f, 1.f}, {}, {}},  // 13
            {{1.f, -1.f, 1.f, 1.f}, {0.f, -1.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 1.f}, {}, {}},   // 14
            {{-1.f, -1.f, 1.f, 1.f}, {0.f, -1.f, 0.f, 0.f}, {1.f, 0.f, 0.f, 1.f}, {}, {}},  // 15

            // Right face (+X)
            {{1.f, -1.f, 1.f, 1.f}, {1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 1.f}, {}, {}},  // 16
            {{1.f, -1.f, -1.f, 1.f}, {1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 1.f, 1.f}, {}, {}}, // 17
            {{1.f, 1.f, -1.f, 1.f}, {1.f, 0.f, 0.f, 0.f}, {1.f, 1.f, 1.f, 1.f}, {}, {}},  // 18
            {{1.f, 1.f, 1.f, 1.f}, {1.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 1.f}, {}, {}},   // 19

            // Left face (-X)
            {{-1.f, -1.f, -1.f, 1.f}, {-1.f, 0.f, 0.f, 0.f}, {1.f, 0.f, 1.f, 1.f}, {}, {}}, // 20
            {{-1.f, -1.f, 1.f, 1.f}, {-1.f, 0.f, 0.f, 0.f}, {1.f, 0.f, 0.f, 1.f}, {}, {}},  // 21
            {{-1.f, 1.f, 1.f, 1.f}, {-1.f, 0.f, 0.f, 0.f}, {1.f, 1.f, 0.f, 1.f}, {}, {}},   // 22
            {{-1.f, 1.f, -1.f, 1.f}, {-1.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 1.f}, {}, {}}   // 23
        };

        std::vector<std::uint32_t> indices = {
            0,  1,  2,  0,  2,  3,  // Front
            4,  5,  6,  4,  6,  7,  // Back
            8,  9,  10, 8,  10, 11, // Top
            12, 13, 14, 12, 14, 15, // Bottom
            16, 17, 18, 16, 18, 19, // Right
            20, 21, 22, 20, 22, 23  // Left
        };

        SubMeshes["cube"].push_back(
            graphics::Mesh{vertices, indices, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr});
    }

    inline void CreateIcoSphere(const float radius, const std::uint32_t subdivisions) {
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

                std::uint32_t a = GetMidpoint(radius, v1, v2, midpointCache, vertices);
                std::uint32_t b = GetMidpoint(radius, v2, v3, midpointCache, vertices);
                std::uint32_t c = GetMidpoint(radius, v3, v1, midpointCache, vertices);

                nextFaces.insert(nextFaces.end(), {v1, a, c, v2, b, a, v3, c, b, a, b, c});
            }
            faces = std::move(nextFaces);
        }
        indices = std::move(faces);

        SubMeshes["sphere"].push_back(
            graphics::Mesh{vertices, indices, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr});
    }

    inline void CreateDiablo() {
        std::vector<shader::Vertex> vertices;
        std::vector<std::uint32_t> indices;

        vertices.reserve(10000);
        indices.reserve(10000);

        LoadObj("test\\diablo\\diablo3_pose.obj", vertices, indices);

        SubMeshes["diablo"].push_back(
            graphics::Mesh{vertices, indices, new graphics::Texture("test\\diablo\\diablo3_pose_diffuse.tga"),
                           new graphics::Texture("test\\diablo\\diablo3_pose_nm_tangent.tga", true),
                           new graphics::Texture("test\\diablo\\diablo3_pose_spec.tga", true), nullptr,
                           new graphics::Texture("test\\diablo\\diablo3_pose_glow.tga"), nullptr});
    }

    inline void CreateAfrican() {
        std::vector<shader::Vertex> headVertices;
        std::vector<std::uint32_t> headIndices;

        headVertices.reserve(10000);
        headIndices.reserve(10000);

        LoadObj("test\\african\\african_head.obj", headVertices, headIndices);
        SubMeshes["african"].push_back(
            graphics::Mesh{headVertices, headIndices, new graphics::Texture("test\\african\\african_head_diffuse.tga"),
                           new graphics::Texture("test\\african\\african_head_nm_tangent.tga", true),
                           new graphics::Texture("test\\african\\african_head_spec.tga", true), nullptr, nullptr,
                           new graphics::Texture("test\\african\\african_head_SSS.jpg")});

        std::vector<shader::Vertex> innerVertices;
        std::vector<std::uint32_t> innerIndices;

        innerVertices.reserve(10000);
        innerIndices.reserve(10000);

        LoadObj("test\\african\\african_head_eye_inner.obj", innerVertices, innerIndices);
        SubMeshes["african"].push_back(graphics::Mesh{
            innerVertices, innerIndices, new graphics::Texture("test\\african\\african_head_eye_inner_diffuse.tga"),
            new graphics::Texture("test\\african\\african_head_eye_inner_nm_tangent.tga", true),
            new graphics::Texture("test\\african\\african_head_eye_inner_spec.tga", true), nullptr, nullptr, nullptr});

        std::vector<shader::Vertex> outerVertices;
        std::vector<std::uint32_t> outerIndices;

        outerVertices.reserve(10000);
        outerIndices.reserve(10000);

        LoadObj("test\\african\\african_head_eye_outer.obj", outerVertices, outerIndices);

        SubMeshes["african"].push_back(graphics::Mesh{
            outerVertices, outerIndices, new graphics::Texture("test\\african\\african_head_eye_outer_diffuse.tga"),
            new graphics::Texture("test\\african\\african_head_eye_outer_nm_tangent.tga", true),
            new graphics::Texture("test\\african\\african_head_eye_outer_spec.tga", true),
            new graphics::Texture("test\\african\\african_head_eye_outer_gloss.tga", true), nullptr, nullptr});
    }
}