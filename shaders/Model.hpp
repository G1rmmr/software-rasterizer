#pragma once

#include <algorithm>

#include "../graphics/Mesh.hpp"
#include "../graphics/Texture.hpp"
#include "../math/Math.hpp"

#include "Elements.hpp"

namespace shader {
    struct Model {
        Uniforms Uniform;

        std::shared_ptr<graphics::Texture> DiffuseMap = nullptr;
        std::shared_ptr<graphics::Texture> NormalMap = nullptr;
        std::shared_ptr<graphics::Texture> SpecularMap = nullptr;
        std::shared_ptr<graphics::Texture> GlossMap = nullptr;
        std::shared_ptr<graphics::Texture> GlowMap = nullptr;
        std::shared_ptr<graphics::Texture> SSSMap = nullptr;
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

            float specIntensity = 0.f;
            if(SpecularMap) specIntensity = SpecularMap->Sample(uv.X, uv.Y).X;

            math::Vector normDir = normal;
            if(NormalMap) {
                math::Vector tangent = math::Vector(inTangent.X, inTangent.Y, inTangent.Z, 0.f);

                tangent = (tangent - normDir * normDir.Dot(tangent)).Norm();

                math::Vector bitangent = normDir.Cross(tangent).Norm() * inTangent.W;
                math::Vector mapNormal = NormalMap->Sample(uv.X, uv.Y);
                mapNormal = mapNormal * 2.f - 1.f;

                mapNormal.Y = -mapNormal.Y;

                math::Vector transformedNormal =
                    (tangent * mapNormal.X) + (bitangent * mapNormal.Y) + (normDir * mapNormal.Z);

                normDir = transformedNormal.Norm();
            }

            math::Vector lightDir = Uniform.LightDir;
            float nDotL = std::max(normDir.Dot(lightDir), 0.f);

            float shadow = 1.f;

            if(nDotL > 0.f) shadow = calculateShadow(worldPos, normDir, lightDir);

            float diff = nDotL * shadow;
            float shininess = 64.f;
            if(GlossMap) {
                float glossSample = GlossMap->Sample(uv.X, uv.Y).X;
                shininess = std::pow(2.f, 1.f + glossSample * 10.f);
            }

            float spec = 0.f;
            if(diff > 0.f) {
                math::Vector viewDir = (Uniform.CameraPos - worldPos).Norm();
                math::Vector halfDir = (lightDir + viewDir).Norm();

                spec = std::pow(std::max(normDir.Dot(halfDir), 0.f), shininess);
                spec *= specIntensity;
            }
            spec *= shadow;

            math::Vector sssColor(0.f);
            if(SSSMap) sssColor = SSSMap->Sample(uv.X, uv.Y);

            math::Vector ambientColor = albedo * 0.1f;
            math::Vector diffuseColor = (albedo + sssColor) * diff;
            math::Vector specColor = math::Vector(1.f, 1.f, 1.f, 1.f) * spec;

            math::Vector glowColor = {0.f, 0.f, 0.f, 0.f};
            if(GlowMap) glowColor = GlowMap->Sample(uv.X, uv.Y);

            math::Vector result = ambientColor + diffuseColor + specColor + glowColor;
            std::swap(result.X, result.Z);

            simd::Floats out = result.V;
            out = simd::Mul(out, simd::Set(255.f));

            float finalAlphaByte = std::min(albedo.W, 1.f) * 255.f;

            math::Vector finalColor = result;
            finalColor.W = std::min(albedo.W, 1.f);

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

            if(projCoords.X < 0.f || projCoords.X > 1.f || projCoords.Y < 0.f || projCoords.Y > 1.f ||
               projCoords.Z > 1.f)
                return 1.f;

            float currentDepth = projCoords.Z;

            float cosTheta = std::clamp(normal.Dot(lightDir), 0.f, 1.f);
            float bias = std::max(0.005f * (1.0f - cosTheta), 0.0005f);

            float avgBlockerDepth = 0.f;
            int blockers = 0;
            float searchWidth = 10.f / ShadowMapWidth;

            for(int i = 0; i < 4; ++i) {
                float z = ShadowMap->at(getShadowIndex(projCoords.X + poissonDisk[i].X * searchWidth,
                                                       projCoords.Y + poissonDisk[i].Y * searchWidth));
                if(z < currentDepth - bias) {
                    avgBlockerDepth += z;
                    blockers++;
                }
            }

            if(blockers == 0) return 1.f;

            avgBlockerDepth /= blockers;
            float penumbraRatio = (currentDepth - avgBlockerDepth) / avgBlockerDepth;
            float filterRadius = penumbraRatio * 10.f * (1.f / ShadowMapWidth);

            float shadow = 0.f;
            int samples = 0;

            for(int i = 0; i < 4; ++i) {
                float pcfDepth = ShadowMap->at(getShadowIndex(projCoords.X + poissonDisk[i].X * filterRadius,
                                                              projCoords.Y + poissonDisk[i].Y * filterRadius));
                shadow += (currentDepth - bias > pcfDepth) ? 0.f : 1.f;
                samples++;
            }

            return shadow / samples;
        }

        const math::Vector poissonDisk[4] = {
            math::Vector(-0.94201624, -0.39906216, 0.f, 0.f), math::Vector(0.94558609, -0.76890725, 0.f, 0.f),
            math::Vector(-0.094184101, -0.92938870, 0.f, 0.f), math::Vector(0.34495938, 0.29387760, 0.f, 0.f)};

        ENGINE_INLINE std::int32_t getShadowIndex(const float u, const float v) const {
            std::int32_t x = static_cast<std::int32_t>(u * ShadowMapWidth);
            std::int32_t y = static_cast<std::int32_t>(v * ShadowMapHeight);

            x = std::clamp(x, 0, static_cast<std::int32_t>(ShadowMapWidth - 1));
            y = std::clamp(y, 0, static_cast<std::int32_t>(ShadowMapHeight - 1));
            return y * ShadowMapWidth + x;
        }
    };
}