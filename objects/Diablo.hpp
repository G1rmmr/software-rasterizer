#pragma once

#include "Object.hpp"

class Diablo : public Object {
public:
    inline void Create() override {
        std::vector<shader::Vertex> vertices;
        std::vector<std::uint32_t> indices;

        vertices.reserve(10000);
        indices.reserve(10000);

        loadObj("assets/diablo/diablo3_pose.obj", vertices, indices);

        this->Meshes.push_back(graphics::Mesh{
            vertices, indices, std::make_shared<graphics::Texture>("assets/diablo/diablo3_pose_diffuse.tga"),
            std::make_shared<graphics::Texture>("assets/diablo/diablo3_pose_nm_tangent.tga"),
            std::make_shared<graphics::Texture>("assets/diablo/diablo3_pose_spec.tga"), nullptr,
            std::make_shared<graphics::Texture>("assets/diablo/diablo3_pose_glow.tga"), nullptr});
    }
};