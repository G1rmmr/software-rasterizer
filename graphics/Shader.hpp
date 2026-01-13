#pragma once

#include <algorithm>

#include "../math/Math.hpp"
#include "Mesh.hpp"
#include "Texture.hpp"

namespace shader {
    struct Vertex {
        math::Vector Pos;
        math::Vector Normal;
        math::Vector Color;
        math::Vector UV;
        math::Vector Tangent;
    };

    struct Varyings {
        math::Vector Pos;
        math::Vector WorldPos;
        math::Vector Normal;
        math::Vector Color;
        math::Vector UV;
        math::Vector Tangent;
        math::Vector Bitangent;
        float RecipW;

        ENGINE_INLINE static Varyings Lerp(const Varyings& v1, const Varyings& v2, float t) {
            Varyings out;
            out.Pos = v1.Pos * (1.f - t) + v2.Pos * t;
            out.WorldPos = v1.WorldPos * (1.f - t) + v2.WorldPos * t;
            out.Normal = (v1.Normal * (1.f - t) + v2.Normal * t).Norm();
            out.Color = v1.Color * (1.f - t) + v2.Color * t;
            out.UV = v1.UV * (1.f - t) + v2.UV * t;
            out.Tangent = (v1.Tangent * (1.f - t) + v2.Tangent * t).Norm();
            return out;
        }
    };

    struct Uniforms {
        math::Vector CameraPos;
        math::Vector LightDir;
        math::Matrix Model;
        math::Matrix View;
        math::Matrix Proj;
        math::Matrix LightSpace;

        float DepthBias = 0.0f;
    };

    struct Model {
        Uniforms Uniform;

        std::shared_ptr<graphics::Texture> DiffuseMap = nullptr;
        std::shared_ptr<graphics::Texture> NormalMap = nullptr;
        std::shared_ptr<graphics::Texture> SpecularMap = nullptr;
        std::shared_ptr<graphics::Texture> GlossMap = nullptr;
        std::shared_ptr<graphics::Texture> GlowMap = nullptr;
        std::shared_ptr<graphics::Texture> SSSMap = nullptr;
        const std::vector<float>* ShadowMap = nullptr;

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
            float shadow = CalculateShadow(worldPos, normDir, lightDir);

            float diff = std::max(normDir.Dot(lightDir), 0.f) * shadow;
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

        ENGINE_INLINE float CalculateShadow(const math::Vector& worldPos, const math::Vector& normal,
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

            float cosTheta = std::abs(normal.Dot(lightDir));
            float bias = std::max(0.05f * (1.f - cosTheta), 0.005f);

            float shadow = 0.f;

            float texX = 1.f / ShadowMapWidth;
            float texY = 1.f / ShadowMapHeight;

            std::uint16_t iWidth = static_cast<std::uint16_t>(ShadowMapWidth);
            std::uint16_t iHeight = static_cast<std::uint16_t>(ShadowMapHeight);

            std::int32_t baseX = static_cast<std::int32_t>(projCoords.X * ShadowMapWidth);
            std::int32_t baseY = static_cast<std::int32_t>(projCoords.Y * ShadowMapHeight);

            auto checkDepth = [&](std::int32_t ox, std::int32_t oy) {
                std::int32_t px = baseX + ox;
                std::int32_t py = baseY + oy;

                if(px >= 0 && px < iWidth && py >= 0 && py < iHeight) {
                    float pcfDepth = (*ShadowMap)[py * iWidth + px];
                    return (currentDepth - bias > pcfDepth) ? 0.f : 1.f;
                }
                return 1.f;
            };

            shadow += checkDepth(-1, -1);
            shadow += checkDepth(0, -1);
            shadow += checkDepth(1, -1);
            shadow += checkDepth(-1, 0);
            shadow += checkDepth(0, 0);
            shadow += checkDepth(1, 0);
            shadow += checkDepth(-1, 1);
            shadow += checkDepth(0, 1);
            shadow += checkDepth(1, 1);

            return shadow * 0.111111f;
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
    };

    struct Shadow {
        Uniforms Uniform;

        ENGINE_INLINE math::Vector Vertex(const math::Vector& pos) const {
            return Uniform.LightSpace * Uniform.Model * pos;
        }

        ENGINE_INLINE math::Vector Normal(const math::Vector& normal) const {
            math::Vector n = Uniform.Model * math::Vector(normal.X, normal.Y, normal.Z, 0.f);
            return n.Norm();
        }

        ENGINE_INLINE std::uint32_t Color(const math::Vector& color, const math::Vector& normal,
                                          const math::Vector& worldPos, const math::Vector& uv,
                                          const math::Vector& inTangent) const {
            return 0xFFFFFFFF;
        }

        ENGINE_INLINE Varyings Process(const shader::Vertex& in) const {
            Varyings out;
            out.Pos = Vertex(in.Pos);
            return out;
        }
    };
}