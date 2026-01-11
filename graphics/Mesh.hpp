#pragma once

#include <string>

#include "Shader.hpp"
#include "Texture.hpp"

namespace graphics {
    struct Mesh {
        std::vector<shader::Vertex> Vertices;
        std::vector<std::uint32_t> Indices;

        graphics::Texture* DiffuseMap = nullptr;
        graphics::Texture* NormalMap = nullptr;
        graphics::Texture* SpecularMap = nullptr;
        graphics::Texture* GlossMap = nullptr;
        graphics::Texture* GlowMap = nullptr;
        graphics::Texture* SSSMap = nullptr;
    };
}