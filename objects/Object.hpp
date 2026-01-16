#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../graphics/Mesh.hpp"
#include "../graphics/Texture.hpp"

#include "../shaders/Elements.hpp"

class Object {
public:
    std::vector<graphics::Mesh> Meshes;
    math::Matrix Model;

    math::Vector LocalCenter = {0.f, 0.f, 0.f};
    float LocalRadius = 0.f;

    bool ShouldRender = true;
    bool IsStatic = true;

    inline Object() = default;
    inline virtual ~Object() = default;
    inline virtual void Create() {}

    void CalculateBounds() {
        if(Meshes.empty()) return;

        math::Vector minPos(1e9f), maxPos(-1e9f);

        for(const graphics::Mesh& mesh : Meshes) {
            for(const shader::Vertex& v : mesh.Vertices) {
                math::Vector pos = v.Pos;

                if(pos.X < minPos.X) minPos.X = pos.X;
                if(pos.Y < minPos.Y) minPos.Y = pos.Y;
                if(pos.Z < minPos.Z) minPos.Z = pos.Z;

                if(pos.X > maxPos.X) maxPos.X = pos.X;
                if(pos.Y > maxPos.Y) maxPos.Y = pos.Y;
                if(pos.Z > maxPos.Z) maxPos.Z = pos.Z;
            }
        }

        LocalCenter = (minPos + maxPos) * 0.5f;
        LocalRadius = (maxPos - minPos).Length() * 0.5f;
    }

    std::pair<math::Vector, float> GetWorldBounds() const {
        math::Vector worldCenter = Model * math::Vector(LocalCenter.X, LocalCenter.Y, LocalCenter.Z, 1.f);

        float maxScale = 1.f;

        math::Vector right = Model * math::Vector(1.f, 0.f, 0.f, 0.f);
        math::Vector up = Model * math::Vector(0.f, 1.f, 0.f, 0.f);
        math::Vector fwd = Model * math::Vector(0.f, 0.f, 1.f, 0.f);
        maxScale = std::max({right.Length(), up.Length(), fwd.Length()});

        return {worldCenter, LocalRadius * maxScale};
    }

protected:
    static inline void loadObj(const std::string& filePath, std::vector<shader::Vertex>& vertices,
                               std::vector<std::uint32_t>& indices) {
        std::ifstream file(filePath);
        if(!file.is_open()) {
            std::fprintf(stderr, "did not exist : %s\n", filePath.c_str());
            return;
        }

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
};