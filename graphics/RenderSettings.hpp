#pragma once
#include "RenderTypes.hpp"
#include "post/AntiAliasingPass.hpp"
#include "post/SsaoPass.hpp"
#include <cstdint>

namespace graphics {
    struct RenderSettings {
        PrimitiveType Primitive = PrimitiveType::Triangles;
        bool Shadows = true;
        bool AmbientOcclusion = true;
        bool AntiAliasing = true;
        bool Toon = false;
        std::uint32_t ClearColor = 0xff000000u;
        SsaoSettings Ssao;
        AntiAliasingSettings AA;
    };
}
