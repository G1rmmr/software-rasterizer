#pragma once
#include "../../scene/Scene.hpp"
#include "../../shaders/ShadowMap.hpp"
#include "../Camera.hpp"
#include "../DirectionalLight.hpp"
#include "../Rasterizer.hpp"
#include "../RenderSettings.hpp"

namespace graphics {
    enum class GeometryLayer { Opaque, Transparent };
    class GeometryPass final {
    public:
        explicit GeometryPass(Rasterizer& rasterizer) : rasterizer_(rasterizer) {}
        void Execute(FrameBuffer& target, const scene::Scene& scene, const Camera& camera,
                     const DirectionalLight& light, const RenderSettings& settings, GeometryLayer layer,
                     const shader::ShadowMap* shadows);

    private:
        Rasterizer& rasterizer_;
    };
}
