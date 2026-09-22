#include "ShadowPass.hpp"
#include "../../shaders/Shadow.hpp"
#include "../Frustum.hpp"
#include "DrawUniforms.hpp"
#include <cmath>

namespace graphics {
    shader::ShadowMap ShadowPass::Execute(const scene::Scene& scene, const DirectionalLight& light) {
        constexpr float nearPlane = -200.f, farPlane = 200.f, extent = 15.f;
        const auto direction = light.GetDirection();
        const auto position = direction * 100.f;
        const math::Vector up = std::abs(direction.Y) > .99f ? math::Vector(0, 0, 1) : math::Vector(0, 1, 0);
        const auto view = math::CreateLookAt(position, {}, up);
        const auto projection = math::CreateOrtho(-extent, extent, -extent, extent, nearPlane, farPlane);
        const auto viewProjection = projection * view;
        const Frustum frustum(viewProjection);
        target_.ClearDepth();
        for(const auto& object : scene.GetObjects()) {
            if(!object.IsVisible() || !object.CastsShadow()) continue;
            const auto bounds = object.GetWorldBounds();
            if(!frustum.IsSphereInside(bounds.Center, bounds.Radius)) continue;
            const auto uniforms = MakeDrawUniforms(object, view, viewProjection, position, direction);
            for(const auto& mesh : object.GetModel().GetMeshes()) {
                const auto& material = mesh.GetMaterial();
                RasterizerOptions options;
                // Both sides cast shadows: a plane still casts when the light moves behind it.
                options.Cull = CullMode::None;
                if(material.Alpha == AlphaMode::Opaque) {
                    const shader::OpaqueShadow shader(uniforms.ModelViewProjection);
                    rasterizer_.Render(target_, shader, mesh.GetVertices(), mesh.GetIndices(), options);
                }
                else {
                    const shader::Shadow shader(uniforms, material);
                    rasterizer_.Render(target_, shader, mesh.GetVertices(), mesh.GetIndices(), options);
                }
            }
        }
        return shader::ShadowMap(target_, viewProjection, nearPlane, farPlane);
    }
}
