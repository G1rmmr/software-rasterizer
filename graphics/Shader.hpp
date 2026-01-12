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
    };

    struct Default {
        Uniforms Uniform;

        inline math::Vector Vertex(const math::Vector& pos) const {
            return Uniform.Proj * Uniform.View * Uniform.Model * pos;
        }

        inline math::Vector Normal(const math::Vector& normal) const {
            math::Vector n = Uniform.Model * math::Vector(normal.X, normal.Y, normal.Z, 0.f);
            return n.Norm();
        }

        inline std::uint32_t Color(const math::Vector& color, const math::Vector& normal,
                                   const math::Vector& worldPos) const {
            math::Vector normDir = normal.Norm();
            math::Vector lightDir = Uniform.LightDir.Norm();
            math::Vector view = (Uniform.CameraPos - worldPos).Norm();
            math::Vector half = (lightDir + view).Norm();

            const float ambient = 0.1f;
            const float diffuse = std::max(normDir.Dot(lightDir), 0.0f);

            float spec = 0.f;
            if(diffuse > 0.f) spec = std::pow(std::max(normDir.Dot(half), 0.f), 64.f);

            const float intensity = ambient + diffuse + spec;

            simd::Floats lightVec = simd::Set(intensity, intensity, intensity, 1.f);
            simd::Floats out = simd::Mul(color.V, lightVec);

            out = simd::Mul(out, simd::Set(255.f));
            return simd::PackRGBA(simd::Clamp(out, simd::Set(0.0f), simd::Set(255.0f)));
        }

        inline Varyings Process(const shader::Vertex& in) const {
            Varyings out;

            out.Pos = Vertex(in.Pos);

            out.WorldPos = Uniform.Model * in.Pos;
            out.WorldPos.W = 1.f;

            out.Normal = Normal(in.Normal);
            out.Color = in.Color;
            return out;
        }
    };

    struct Model {
        Uniforms Uniform;

        std::shared_ptr<graphics::Texture> DiffuseMap = nullptr;
        std::shared_ptr<graphics::Texture> NormalMap = nullptr;
        std::shared_ptr<graphics::Texture> SpecularMap = nullptr;
        std::shared_ptr<graphics::Texture> GlossMap = nullptr;
        std::shared_ptr<graphics::Texture> GlowMap = nullptr;
        std::shared_ptr<graphics::Texture> SSSMap = nullptr;

        inline math::Vector Vertex(const math::Vector& pos) const {
            return Uniform.Proj * Uniform.View * Uniform.Model * pos;
        }

        inline math::Vector Normal(const math::Vector& normal) const {
            math::Vector n = Uniform.Model * math::Vector(normal.X, normal.Y, normal.Z, 0.f);
            return n.Norm();
        }

        inline std::uint32_t Color(const math::Vector& color, const math::Vector& normal, const math::Vector& worldPos,
                                   const math::Vector& uv, const math::Vector& inTangent) const {
            math::Vector albedo = {1.f, 1.f, 1.f, 1.f};
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
            float diff = std::max(normDir.Dot(lightDir), 0.f);

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

            math::Vector glowColor = {0.f, 0.f, 0.f, 0.f};
            if(GlowMap) glowColor = GlowMap->Sample(uv.X, uv.Y);

            float sssStrength = 0.f;
            if(SSSMap) sssStrength = SSSMap->Sample(uv.X, uv.Y).X;

            math::Vector skinTint = {1.f, 0.2f, 0.1f, 1.f};
            math::Vector sssTerm = skinTint * sssStrength * 0.5f;

            math::Vector ambient = {0.1f, 0.1f, 0.1f, 1.f};
            math::Vector result = (albedo * (diff + 0.1f)) + (math::Vector(1, 1, 1, 1) * spec) + glowColor + sssTerm;

            simd::Floats out = result.V;
            out = simd::Mul(out, simd::Set(255.f));

            float finalAlphaByte = std::min(albedo.W, 1.f) * 255.f;

            math::Vector finalColor = result;
            finalColor.W = std::min(albedo.W, 1.f);

            return simd::PackRGBA(
                simd::Clamp(simd::Mul(finalColor.V, simd::Set(255.f)), simd::Set(0.f), simd::Set(255.f)));
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
}