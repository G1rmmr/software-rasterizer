#pragma once

#include "Object.hpp"

class African : public Object {
public:
    inline void Create() override {
        std::vector<shader::Vertex> headVertices;
        std::vector<std::uint32_t> headIndices;

        headVertices.reserve(10000);
        headIndices.reserve(10000);

        loadObj("assets/african/african_head.obj", headVertices, headIndices);
        this->Meshes.push_back(graphics::Mesh{
            headVertices, headIndices, std::make_shared<graphics::Texture>("assets/african/african_head_diffuse.tga"),
            std::make_shared<graphics::Texture>("assets/african/african_head_nm_tangent.tga"),
            std::make_shared<graphics::Texture>("assets/african/african_head_spec.tga"), nullptr, nullptr,
            std::make_shared<graphics::Texture>("assets/african/african_head_SSS.jpg")});

        std::vector<shader::Vertex> innerVertices;
        std::vector<std::uint32_t> innerIndices;

        innerVertices.reserve(10000);
        innerIndices.reserve(10000);

        loadObj("assets/african/african_head_eye_inner.obj", innerVertices, innerIndices);
        this->Meshes.push_back(
            graphics::Mesh{innerVertices, innerIndices,
                           std::make_shared<graphics::Texture>("assets/african/african_head_eye_inner_diffuse.tga"),
                           std::make_shared<graphics::Texture>("assets/african/african_head_eye_inner_nm_tangent.tga"),
                           std::make_shared<graphics::Texture>("assets/african/african_head_eye_inner_spec.tga"),
                           nullptr, nullptr, nullptr});

        std::vector<shader::Vertex> outerVertices;
        std::vector<std::uint32_t> outerIndices;

        outerVertices.reserve(10000);
        outerIndices.reserve(10000);

        loadObj("assets/african/african_head_eye_outer.obj", outerVertices, outerIndices);

        this->Meshes.push_back(graphics::Mesh{
            outerVertices, outerIndices,
            std::make_shared<graphics::Texture>("assets/african/african_head_eye_outer_diffuse.tga"),
            std::make_shared<graphics::Texture>("assets/african/african_head_eye_outer_nm_tangent.tga"),
            std::make_shared<graphics::Texture>("assets/african/african_head_eye_outer_spec.tga"),
            std::make_shared<graphics::Texture>("assets/african/african_head_eye_outer_gloss.tga"), nullptr, nullptr});
    }
};