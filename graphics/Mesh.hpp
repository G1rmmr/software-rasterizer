#pragma once

#include <memory>
#include <string>

#include "Shader.hpp"
#include "Texture.hpp"

namespace graphics {
    struct Mesh {
        std::vector<shader::Vertex> Vertices;
        std::vector<std::uint32_t> Indices;

        std::shared_ptr<graphics::Texture> DiffuseMap = nullptr;
        std::shared_ptr<graphics::Texture> NormalMap = nullptr;
        std::shared_ptr<graphics::Texture> SpecularMap = nullptr;
        std::shared_ptr<graphics::Texture> GlossMap = nullptr;
        std::shared_ptr<graphics::Texture> GlowMap = nullptr;
        std::shared_ptr<graphics::Texture> SSSMap = nullptr;
    };
}