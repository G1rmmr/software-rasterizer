#pragma once
#include "../../scene/Scene.hpp"
#include "../../shaders/ShadowMap.hpp"
#include "../DirectionalLight.hpp"
#include "../FrameBuffer.hpp"
#include "../Rasterizer.hpp"

namespace graphics {
    class ShadowPass final {
    public:
        explicit ShadowPass(Rasterizer& rasterizer, std::uint32_t resolution = 512)
            : rasterizer_(rasterizer), target_(resolution, resolution) {}
        shader::ShadowMap Execute(const scene::Scene& scene, const DirectionalLight& light);

    private:
        Rasterizer& rasterizer_;
        FrameBuffer target_;
    };
}
