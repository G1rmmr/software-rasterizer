#pragma once
#include "../graphics/Material.hpp"
#include "Elements.hpp"
#include <algorithm>

namespace shader {
    inline math::Vector SampleAlbedo(const graphics::Material& material, const Fragment& fragment) {
        auto color = material.DiffuseMap ? material.DiffuseMap->Sample(fragment.UV.X, fragment.UV.Y) : fragment.Color;
        switch(material.Alpha) {
        case graphics::AlphaMode::Opaque: color.W = 1.f; break;
        case graphics::AlphaMode::Mask: color.W = color.W >= material.AlphaCutoff ? 1.f : 0.f; break;
        case graphics::AlphaMode::Blend: color.W = std::clamp(color.W, 0.f, 1.f); break;
        }
        return color;
    }

    inline math::Vector SampleNormal(const graphics::Material& material, const Fragment& fragment) {
        auto normal = math::Vector(fragment.Normal.X, fragment.Normal.Y, fragment.Normal.Z, 0.f).Norm();
        if(!material.NormalMap) return normal;
        auto tangent = math::Vector(fragment.Tangent.X, fragment.Tangent.Y, fragment.Tangent.Z, 0.f);
        tangent = (tangent - normal * normal.Dot(tangent)).Norm();
        if(tangent.Length() < 1e-6f) return normal;
        const float handedness = fragment.Tangent.W < 0.f ? -1.f : 1.f;
        const auto bitangent = normal.Cross(tangent).Norm() * handedness;
        auto sampled = material.NormalMap->Sample(fragment.UV.X, fragment.UV.Y) * 2.f - 1.f;
        // The bundled assets use the opposite tangent-space green convention.
        sampled.Y = -sampled.Y;
        return (tangent * sampled.X + bitangent * sampled.Y + normal * sampled.Z).Norm();
    }

    inline math::Vector ToViewNormal(const math::Vector& worldNormal, const math::Matrix& view) noexcept {
        return (view * math::Vector(worldNormal.X, worldNormal.Y, worldNormal.Z, 0.f)).Norm();
    }
}
