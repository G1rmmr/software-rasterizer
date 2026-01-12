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
    };

    struct Uniforms {
        math::Vector CameraPos;
        math::Vector LightDir;
        math::Matrix Model;
        math::Matrix View;
        math::Matrix Proj;

        math::Matrix LightSpace;
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

        inline math::Vector Vertex(const math::Vector& pos) const {
            return Uniform.Proj * Uniform.View * Uniform.Model * pos;
        }

        inline math::Vector Normal(const math::Vector& normal) const {
            math::Vector n = Uniform.Model * math::Vector(normal.X, normal.Y, normal.Z, 0.f);
            return n.Norm();
        }

        inline std::uint32_t Color(const math::Vector& color, const math::Vector& normal, const math::Vector& worldPos,
                                   const math::Vector& uv, const math::Vector& inTangent) const {
            math::Vector albedo = color;
            if(DiffuseMap) albedo = DiffuseMap->Sample(uv.X, uv.Y);
            if(albedo.W < 0.05f) return 0x00000000;

            float specIntensity = 0.f;
            if(SpecularMap) specIntensity = SpecularMap->Sample(uv.X, uv.Y).X;

            math::Vector normDir = normal.Norm();
            if(NormalMap) {
                math::Vector norm = normal.Norm();
                math::Vector tangent = math::Vector(inTangent.X, inTangent.Y, inTangent.Z, 0.f).Norm();

                tangent = (tangent - norm * norm.Dot(tangent)).Norm();

                math::Vector bitangent = norm.Cross(tangent).Norm() * inTangent.W;
                math::Vector mapNormal = NormalMap->Sample(uv.X, uv.Y);
                mapNormal = mapNormal * 2.f - 1.f;

                mapNormal.Y = -mapNormal.Y;

                math::Vector transformedNormal =
                    (tangent * mapNormal.X) + (bitangent * mapNormal.Y) + (norm * mapNormal.Z);

                normDir = transformedNormal.Norm();
            }

            math::Vector lightDir = Uniform.LightDir.Norm();
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

        inline float CalculateShadow(const math::Vector& worldPos, const math::Vector& normal,
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
            float bias = 0.001f;

            float shadow = 0.f;

            math::Vector texelSize = {1.f / ShadowMapWidth, 1.f / ShadowMapHeight, 0.f, 0.f};

            for(int x = -1; x <= 1; ++x) {
                for(int y = -1; y <= 1; ++y) {
                    float pcfDepth = 1.f;

                    int pcfX = static_cast<int>((projCoords.X + x * texelSize.X) * ShadowMapWidth);
                    int pcfY = static_cast<int>((projCoords.Y + y * texelSize.Y) * ShadowMapHeight);

                    if(pcfX >= 0 && pcfX < (int)ShadowMapWidth && pcfY >= 0 && pcfY < (int)ShadowMapHeight) {
                        pcfDepth = (*ShadowMap)[pcfY * (int)ShadowMapWidth + pcfX];
                    }

                    shadow += (currentDepth - bias > pcfDepth) ? 0.f : 1.f;
                }
            }

            return shadow / 9.f;
        }

        inline Varyings Process(const shader::Vertex& in) const {
            Varyings out;

            out.Pos = Vertex(in.Pos);

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

        inline math::Vector Vertex(const math::Vector& pos) const { return Uniform.LightSpace * Uniform.Model * pos; }

        inline math::Vector Normal(const math::Vector& normal) const {
            math::Vector n = Uniform.Model * math::Vector(normal.X, normal.Y, normal.Z, 0.f);
            return n.Norm();
        }

        inline std::uint32_t Color(const math::Vector& color, const math::Vector& normal, const math::Vector& worldPos,
                                   const math::Vector& uv, const math::Vector& inTangent) const {
            return 0xFFFFFFFF;
        }

        inline Varyings Process(const shader::Vertex& in) const {
            Varyings out;
            out.Pos = Vertex(in.Pos);
            return out;
        }
    };
}