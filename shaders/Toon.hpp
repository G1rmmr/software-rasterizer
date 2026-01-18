#pragma once

#include <algorithm>

#include "../graphics/Mesh.hpp"
#include "../graphics/Texture.hpp"
#include "../math/Math.hpp"

#include "Elements.hpp"

namespace shader {
    struct Toon {
        Uniforms Uniform;

        std::shared_ptr<graphics::Texture> DiffuseMap = nullptr;
        std::vector<float>* ShadowMap = nullptr;

        float ShadowMapWidth = 0.f;
        float ShadowMapHeight = 0.f;

        ENGINE_INLINE math::Vector Vertex(const math::Vector& pos) const {
            return Uniform.Proj * Uniform.View * Uniform.Model * pos;
        }

        ENGINE_INLINE math::Vector Normal(const math::Vector& normal) const {
            math::Vector n = Uniform.Model * math::Vector(normal.X, normal.Y, normal.Z, 0.f);
            return n.Norm();
        }

        ENGINE_INLINE std::uint32_t Color(const math::Vector& color, const math::Vector& normal,
                                          const math::Vector& worldPos, const math::Vector& uv,
                                          const math::Vector& inTangent) const {
            math::Vector albedo = color;
            if(DiffuseMap) albedo = DiffuseMap->Sample(uv.X, uv.Y);
            if(albedo.W < 0.05f) return 0x00000000;

            const float colorLevels = 10.f;

            albedo.X = std::floor(albedo.X * colorLevels) / colorLevels;
            albedo.Y = std::floor(albedo.Y * colorLevels) / colorLevels;
            albedo.Z = std::floor(albedo.Z * colorLevels) / colorLevels;

            math::Vector normDir = (Uniform.Model * math::Vector(normal.X, normal.Y, normal.Z, 0.f)).Norm();
            math::Vector lightDir = Uniform.LightDir;
            math::Vector viewDir = (Uniform.CameraPos - worldPos).Norm();

            float edge = viewDir.Dot(normDir);
            if(edge >= 0.f && edge < 0.3f) {
                return 0xFF000000;
            }

            float ndotl = normDir.Dot(lightDir);
            float intensity = ndotl * 0.5f + 0.5f;

            intensity *= intensity;

            if(ShadowMap) {
                float shadow = calculateShadow(worldPos, normDir, lightDir);
                if(shadow < 0.5f) intensity = std::min(intensity, 0.3f);
            }

            float tone = 0.f;
            if(intensity > 0.9f)
                tone = 1.f;
            else if(intensity > 0.6f)
                tone = 0.7f;
            else if(intensity > 0.4f)
                tone = 0.4f;
            else
                tone = 0.35f;

            float spec = 0.f;
            if(intensity > 0.f) {
                math::Vector halfDir = (lightDir + viewDir).Norm();
                float NdotH = std::max(normDir.Dot(halfDir), 0.f);
                if(NdotH > 0.98f) spec = 1.f;
            }

            math::Vector finalColor;
            finalColor.X = albedo.X * tone + spec;
            finalColor.Y = albedo.Y * tone + spec;
            finalColor.Z = albedo.Z * tone + spec;
            finalColor.W = 1.f;

            std::swap(finalColor.X, finalColor.Z);

            return simd::PackRGBA(
                simd::Clamp(simd::Mul(finalColor.V, simd::Set(255.f)), simd::Set(0.f), simd::Set(255.f)));
        }

        ENGINE_INLINE Varyings Process(const shader::Vertex& in) const {
            Varyings out;

            math::Vector clipPos = Vertex(in.Pos);
            clipPos.Z += Uniform.DepthBias * clipPos.W;

            out.Pos = clipPos;

            out.WorldPos = Uniform.Model * in.Pos;
            out.WorldPos.W = 1.f;

            math::Vector tDir = math::Vector(in.Tangent.X, in.Tangent.Y, in.Tangent.Z, 0.f);
            tDir = (Uniform.Model * tDir).Norm();

            math::Vector norm = Normal(in.Normal);

            out.Normal = norm;
            out.Color = in.Color;
            out.Tangent = math::Vector(tDir.X, tDir.Y, tDir.Z, in.Tangent.W);
            out.UV = in.UV;
            return out;
        }

    private:
        ENGINE_INLINE float calculateShadow(const math::Vector& worldPos, const math::Vector& normal,
                                            const math::Vector& lightDir) const {
            if(!ShadowMap) return 1.f;

            math::Vector lightSpacePos = Uniform.LightSpace * worldPos;
            math::Vector projCoords = lightSpacePos * (1.f / lightSpacePos.W);
            projCoords.X = projCoords.X * 0.5f + 0.5f;
            projCoords.Y = 1.f - (projCoords.Y * 0.5f + 0.5f);
            if(projCoords.Z > 1.f || projCoords.X < 0.f || projCoords.X > 1.f || projCoords.Y < 0.f ||
               projCoords.Y > 1.f)
                return 1.f;

            float currentDepth = projCoords.Z;
            float bias = 0.005f;
            float closestDepth = ShadowMap->at(getShadowIndex(projCoords.X, projCoords.Y));
            return (currentDepth - bias > closestDepth) ? 0.f : 1.f;
        }

        ENGINE_INLINE std::int32_t getShadowIndex(const float u, const float v) const {
            std::int32_t x = std::clamp(static_cast<std::int32_t>(u * ShadowMapWidth), 0,
                                        static_cast<std::int32_t>(ShadowMapWidth - 1));
            std::int32_t y = std::clamp(static_cast<std::int32_t>(v * ShadowMapHeight), 0,
                                        static_cast<std::int32_t>(ShadowMapHeight - 1));
            return y * ShadowMapWidth + x;
        }
    };
}