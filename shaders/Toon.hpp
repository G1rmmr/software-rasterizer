#pragma once
#include "../graphics/Color.hpp"
#include "Elements.hpp"
#include "ShadowMap.hpp"
#include "Surface.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace shader {
    class Toon final {
    public:
        Toon(DrawUniforms uniforms, const graphics::Material& material, const ShadowMap* shadows = nullptr)
            : uniforms_(std::move(uniforms)), material_(material), shadows_(shadows) {}
        [[nodiscard]] Varyings Process(const Vertex& vertex) const noexcept {
            return TransformVertex(vertex, uniforms_);
        }
        [[nodiscard]] FragmentOutput Shade(const Fragment& fragment) const {
            auto albedo = SampleAlbedo(material_, fragment);
            if(albedo.W <= 0.f) return {};
            const auto normal = SampleNormal(material_, fragment);
            const auto view = (uniforms_.CameraPos - fragment.WorldPos).Norm();
            const float edge = view.Dot(normal);
            if(edge >= 0.f && edge < .3f)
                return {graphics::PackColor({0.f, 0.f, 0.f, albedo.W}), ToViewNormal(normal, uniforms_.View)};
            albedo.X = QuantizeAlbedo(albedo.X);
            albedo.Y = QuantizeAlbedo(albedo.Y);
            albedo.Z = QuantizeAlbedo(albedo.Z);
            float intensity = std::clamp(normal.Dot(uniforms_.LightDir) * .5f + .5f, 0.f, 1.f);
            intensity *= intensity;
            if(shadows_ && shadows_->Visibility(fragment.WorldPos, normal, uniforms_.LightDir) < .5f)
                intensity = std::min(intensity, .3f);
            const float tone = intensity > .9f ? 1.f : intensity > .6f ? .7f : intensity > .4f ? .4f : .15f;
            const auto halfDirection = (uniforms_.LightDir + view).Norm();
            const float specular = intensity > 0.f && normal.Dot(halfDirection) > .98f ? 1.f : 0.f;
            math::Vector result(albedo.X * tone + specular, albedo.Y * tone + specular, albedo.Z * tone + specular,
                                albedo.W);
            return {graphics::PackColor(result), ToViewNormal(normal, uniforms_.View)};
        }

    private:
        static float QuantizeAlbedo(float channel) noexcept {
            const float scaled = std::clamp(channel, 0.f, 1.f) * 10.f;
            // Perspective interpolation can place a constant color a few ULPs below
            // an exact bin boundary. Absorb roundoff without rounding to another bin.
            constexpr float roundoff = 8.f * std::numeric_limits<float>::epsilon();
            const float tolerance = roundoff * std::max(1.f, scaled);
            return std::floor(scaled + tolerance) / 10.f;
        }

        DrawUniforms uniforms_;
        const graphics::Material& material_;
        const ShadowMap* shadows_;
    };
}
