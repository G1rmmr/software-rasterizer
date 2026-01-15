#pragma once

#include "Elements.hpp"

namespace shader {
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