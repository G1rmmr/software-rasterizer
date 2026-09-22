#include "Renderer.hpp"
#include <optional>

namespace graphics {
    Renderer::Renderer(std::uint32_t width, std::uint32_t height, std::size_t workers)
        : executor_(workers),
          rasterizer_(executor_),
          shadow_(rasterizer_),
          geometry_(rasterizer_),
          ssao_(executor_),
          aa_(executor_),
          frame_(width, height) {}

    const FrameBuffer& Renderer::Render(const scene::Scene& scene, const Camera& camera, const DirectionalLight& light,
                                        const RenderSettings& settings) {
        timings_ = {};
        debug::ScopedTimer total(timings_.TotalFrameTime);
        frame_.Clear(executor_, settings.ClearColor);
        std::optional<shader::ShadowMap> shadows;
        if(settings.Shadows) {
            debug::ScopedTimer timer(timings_.ShadowPassTime);
            shadows.emplace(shadow_.Execute(scene, light));
        }
        {
            debug::ScopedTimer timer(timings_.MainPassTime);
            geometry_.Execute(frame_, scene, camera, light, settings, GeometryLayer::Opaque,
                              shadows ? &*shadows : nullptr);
        }
        // Diagnostic point/wireframe modes have no surface G-buffer.
        if(settings.AmbientOcclusion && settings.Primitive == PrimitiveType::Triangles) {
            debug::ScopedTimer timer(timings_.PostPassTime);
            ssao_.Execute(frame_, camera.GetProjection(), camera.GetInverseProjection(), settings.Ssao);
        }
        float transparentTime = 0.f;
        {
            debug::ScopedTimer timer(transparentTime);
            geometry_.Execute(frame_, scene, camera, light, settings, GeometryLayer::Transparent,
                              shadows ? &*shadows : nullptr);
        }
        timings_.MainPassTime += transparentTime;
        if(settings.AntiAliasing) {
            debug::ScopedTimer timer(timings_.AAPassTime);
            aa_.Execute(frame_, settings.AA);
        }
        return frame_;
    }
}
