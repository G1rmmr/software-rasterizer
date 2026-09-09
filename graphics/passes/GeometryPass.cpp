#include "GeometryPass.hpp"
#include "../../shaders/Model.hpp"
#include "../../shaders/Toon.hpp"
#include "DrawUniforms.hpp"
#include <algorithm>
#include <vector>

namespace graphics {
    void GeometryPass::Execute(FrameBuffer& target, const scene::Scene& scene, const Camera& camera,
                               const DirectionalLight& light, const RenderSettings& settings, GeometryLayer layer,
                               const shader::ShadowMap* shadows) {
        struct Draw {
            const scene::SceneObject* Object;
            const Mesh* MeshData;
            float Distance;
        };
        std::vector<Draw> draws;
        const bool blend = layer == GeometryLayer::Transparent;
        for(const auto& object : scene.GetObjects()) {
            if(!object.IsVisible()) continue;
            const auto bounds = object.GetWorldBounds();
            if(!camera.GetFrustum().IsSphereInside(bounds.Center, bounds.Radius)) continue;
            for(const auto& mesh : object.GetModel().GetMeshes()) {
                if((mesh.GetMaterial().Alpha == AlphaMode::Blend) != blend) continue;
                const auto& center = mesh.GetBounds().Center;
                const auto viewCenter =
                    camera.GetView() * (object.GetTransform() * math::Vector(center.X, center.Y, center.Z, 1.f));
                draws.push_back({&object, &mesh, -viewCenter.Z});
            }
        }
        if(blend)
            std::stable_sort(draws.begin(), draws.end(),
                             [](const Draw& a, const Draw& b) { return a.Distance > b.Distance; });
        const auto viewProjection = camera.GetProjection() * camera.GetView();
        const auto draw = [&](const Draw& command) {
            const auto& material = command.MeshData->GetMaterial();
            const auto uniforms = MakeDrawUniforms(*command.Object, camera.GetView(), viewProjection,
                                                   camera.GetPosition(), light.GetDirection());
            RasterizerOptions options;
            options.Primitive = settings.Primitive;
            options.Cull = material.DoubleSided ? CullMode::None : CullMode::Back;
            options.FlipWinding = uniforms.Orientation < 0.f;
            options.Blend = blend && settings.Primitive == PrimitiveType::Triangles;
            options.DepthWrite = !options.Blend;
            if(settings.Toon) {
                const shader::Toon shader(uniforms, material, shadows);
                rasterizer_.Render(target, shader, command.MeshData->GetVertices(), command.MeshData->GetIndices(),
                                   options);
            }
            else {
                const shader::Model shader(uniforms, material, shadows);
                rasterizer_.Render(target, shader, command.MeshData->GetVertices(), command.MeshData->GetIndices(),
                                   options);
            }
        };
        for(const auto& command : draws) draw(command);
    }
}
