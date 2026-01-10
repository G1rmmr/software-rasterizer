#pragma once

#include <algorithm>

#include "../math/Math.hpp"

namespace shader {
    struct Vertex {
        math::Vector Pos;
        math::Vector Normal;
        math::Vector Color;
    };

    struct Varyings {
        math::Vector Pos;
        math::Vector WorldPos;
        math::Vector Normal;
        math::Vector Color;
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
            math::Vector V = (Uniform.CameraPos - worldPos).Norm();
            math::Vector half = (lightDir + V).Norm();

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
            out.WorldPos.W = 1.0f;

            out.Normal = Normal(in.Normal);
            out.Color = in.Color;
            return out;
        }
    };
}