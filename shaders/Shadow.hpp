#pragma once
#include "Elements.hpp"
#include "Surface.hpp"
#include <utility>

namespace shader {
    // Opaque coverage needs only clip position and depth. The rasterizer can
    // omit attribute interpolation and surface evaluation for this contract.
    class OpaqueShadow final {
    public:
        static constexpr bool DepthOnly = true;
        explicit OpaqueShadow(math::Matrix modelViewProjection) : transform_(std::move(modelViewProjection)) {}
        [[nodiscard]] Varyings Process(const Vertex& vertex) const noexcept {
            Varyings result;
            result.Pos = transform_ * vertex.Pos;
            return result;
        }
        [[nodiscard]] FragmentOutput Shade(const Fragment&) const noexcept { return {0xffffffffu, {}}; }

    private:
        math::Matrix transform_;
    };

    class Shadow final {
    public:
        Shadow(DrawUniforms uniforms, const graphics::Material& material)
            : uniforms_(std::move(uniforms)), material_(material) {}
        [[nodiscard]] Varyings Process(const Vertex& vertex) const noexcept {
            return TransformVertex(vertex, uniforms_);
        }
        [[nodiscard]] FragmentOutput Shade(const Fragment& fragment) const {
            // A depth shadow map supports alpha coverage, not colored transmission.
            const auto albedo = SampleAlbedo(material_, fragment);
            if(albedo.W <= 0.f || (material_.Alpha == graphics::AlphaMode::Blend && albedo.W < material_.AlphaCutoff))
                return {};
            return {0xffffffffu, {}};
        }

    private:
        DrawUniforms uniforms_;
        const graphics::Material& material_;
    };
}
