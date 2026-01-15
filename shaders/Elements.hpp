#pragma once

#include "../math/Math.hpp"

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
        math::Matrix Model;
        math::Matrix View;
        math::Matrix Proj;
        math::Matrix LightSpace;
        math::Matrix InvProj;

        math::Vector CameraPos;
        math::Vector LightDir;

        float DepthBias = 0.f;
        float ScreenWidth = 0.f;
        float ScreenHeight = 0.f;
        float RadiusAO = 150.f;
        float BiasAO = 0.05f;
        float StrengthAO = 1.5f;

        std::int32_t KernelSizeAO = 16;
    };
}