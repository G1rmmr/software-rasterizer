#pragma once

#include <memory>

#include "Texture.hpp"

namespace graphics {
    enum class AlphaMode { Opaque, Mask, Blend };

    struct Material {
        std::shared_ptr<const Texture> DiffuseMap;
        std::shared_ptr<const Texture> NormalMap;
        std::shared_ptr<const Texture> SpecularMap;
        std::shared_ptr<const Texture> GlossMap;
        std::shared_ptr<const Texture> GlowMap;
        std::shared_ptr<const Texture> SSSMap;
        AlphaMode Alpha = AlphaMode::Opaque;
        float AlphaCutoff = 0.05f;
        bool DoubleSided = false;
    };
}
