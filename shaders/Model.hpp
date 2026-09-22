#pragma once
#include "../graphics/Color.hpp"
#include "Elements.hpp"
#include "ShadowMap.hpp"
#include "Surface.hpp"
#include <algorithm>
#include <utility>

namespace shader {
    class Model final {
    public:
        Model(DrawUniforms uniforms, const graphics::Material& material, const ShadowMap* shadows = nullptr)
            : uniforms_(std::move(uniforms)), material_(material), shadows_(shadows) {}
        [[nodiscard]] Varyings Process(const Vertex& vertex) const noexcept {
            return TransformVertex(vertex, uniforms_);
        }
        [[nodiscard]] FragmentOutput Shade(const Fragment& fragment) const {
            const auto albedo = SampleAlbedo(material_, fragment);
            if(albedo.W <= 0.f) return {};
            const auto normal = SampleNormal(material_, fragment);
            const float nDotL = std::max(normal.Dot(uniforms_.LightDir), 0.f);
            const float visibility =
                shadows_ && nDotL > 0.f ? shadows_->Visibility(fragment.WorldPos, normal, uniforms_.LightDir) : 1.f;
            const float diffuse = nDotL * visibility;
            float specular = 0.f;
            if(diffuse > 0.f && material_.SpecularMap) {
                const auto view = (uniforms_.CameraPos - fragment.WorldPos).Norm();
                const auto halfDirection = (view + uniforms_.LightDir).Norm();
                const float h = std::max(normal.Dot(halfDirection), 0.f);
                const float p2 = h * h, p4 = p2 * p2, p8 = p4 * p4, p16 = p8 * p8, p32 = p16 * p16;
                const float gloss =
                    material_.GlossMap ? material_.GlossMap->Sample(fragment.UV.X, fragment.UV.Y).X : 0.f;
                float power;
                if(gloss < .33f)
                    power = std::lerp(p2, p8, gloss * 3.f);
                else if(gloss < .66f)
                    power = std::lerp(p8, p16, (gloss - .33f) * 3.f);
                else
                    power = std::lerp(p16, p32, (gloss - .66f) * 3.f);
                specular = power * material_.SpecularMap->Sample(fragment.UV.X, fragment.UV.Y).X * visibility;
            }
            auto result = albedo * (.1f + diffuse) + math::Vector(specular, specular, specular, 0.f);
            if(material_.SSSMap) {
                const float transmission = std::pow(std::max(-normal.Dot(uniforms_.LightDir), 0.f), 2.f) * .5f;
                result += material_.SSSMap->Sample(fragment.UV.X, fragment.UV.Y) * transmission;
            }
            if(material_.GlowMap) result += material_.GlowMap->Sample(fragment.UV.X, fragment.UV.Y);
            result.W = albedo.W;
            return {graphics::PackColor(result), ToViewNormal(normal, uniforms_.View)};
        }

    private:
        DrawUniforms uniforms_;
        const graphics::Material& material_;
        const ShadowMap* shadows_;
    };
}
