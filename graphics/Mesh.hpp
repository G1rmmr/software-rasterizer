#pragma once

#include "Texture.hpp"
#include <string>

namespace graphics {
    struct Mesh {
        std::vector<std::uint32_t> Indices;
        graphics::Texture* DiffuseMap = nullptr;
        graphics::Texture* NormalMap = nullptr;
        graphics::Texture* SpecularMap = nullptr;
        graphics::Texture* GlossMap = nullptr;
        graphics::Texture* GlowMap = nullptr;
        graphics::Texture* SSSMap = nullptr;
    };
}