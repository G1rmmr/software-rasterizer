#pragma once
#include "../graphics/Vertex.hpp"
#include "../math/Math.hpp"
#include <cstdint>

namespace shader {
    using Vertex = graphics::Vertex;

    struct Varyings {
        math::Vector Pos{}; // Homogeneous clip coordinates, Z in [0, W].
        math::Vector WorldPos{};
        math::Vector Normal{}; // World-space direction, normalized after interpolation.
        math::Vector Color{};  // Logical RGBA, before framebuffer packing.
        math::Vector UV{};
        math::Vector Tangent{}; // World-space XYZ, handedness in W.
        float RecipW = 0.f;

        static Varyings Lerp(const Varyings& a, const Varyings& b, float t) noexcept {
            Varyings result;
            result.Pos = a.Pos * (1.f - t) + b.Pos * t;
            result.WorldPos = a.WorldPos * (1.f - t) + b.WorldPos * t;
            result.Normal = a.Normal * (1.f - t) + b.Normal * t;
            result.Color = a.Color * (1.f - t) + b.Color * t;
            result.UV = a.UV * (1.f - t) + b.UV * t;
            result.Tangent = a.Tangent * (1.f - t) + b.Tangent * t;
            return result;
        }
    };

    struct Fragment {
        math::Vector WorldPos{};
        math::Vector Normal{};
        math::Vector Color{};
        math::Vector UV{};
        math::Vector Tangent{};
    };

    struct FragmentOutput {
        std::uint32_t Color = 0; // 0xAARRGGBB; alpha zero discards the fragment.
        math::Vector ViewNormal{};
    };

    // Immutable inputs for one draw. No application or post-processing state.
    struct DrawUniforms {
        math::Matrix Model;
        math::Matrix NormalMatrix;
        math::Matrix View;
        math::Matrix ModelViewProjection;
        math::Vector CameraPos;
        math::Vector LightDir;
        float Orientation = 1.f;
    };

    inline Varyings TransformVertex(const Vertex& vertex, const DrawUniforms& uniforms) noexcept {
        Varyings result;
        result.Pos = uniforms.ModelViewProjection * vertex.Pos;
        result.WorldPos = uniforms.Model * vertex.Pos;
        result.Normal =
            (uniforms.NormalMatrix * math::Vector(vertex.Normal.X, vertex.Normal.Y, vertex.Normal.Z, 0.f)).Norm();
        const auto tangent =
            (uniforms.Model * math::Vector(vertex.Tangent.X, vertex.Tangent.Y, vertex.Tangent.Z, 0.f)).Norm();
        result.Tangent = {tangent.X, tangent.Y, tangent.Z, vertex.Tangent.W * uniforms.Orientation};
        result.Color = vertex.Color;
        result.UV = vertex.UV;
        return result;
    }
}
