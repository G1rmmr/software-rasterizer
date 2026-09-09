#pragma once
#include "../../scene/SceneObject.hpp"
#include "../../shaders/Elements.hpp"

namespace graphics {
    inline shader::DrawUniforms MakeDrawUniforms(const scene::SceneObject& object, const math::Matrix& view,
                                                 const math::Matrix& viewProjection, const math::Vector& cameraPosition,
                                                 const math::Vector& lightDirection) {
        shader::DrawUniforms uniforms;
        uniforms.Model = object.GetTransform();
        uniforms.NormalMatrix = object.GetNormalTransform();
        uniforms.View = view;
        uniforms.ModelViewProjection = viewProjection * uniforms.Model;
        uniforms.CameraPos = cameraPosition;
        uniforms.LightDir = lightDirection;
        const auto& m = uniforms.Model;
        const math::Vector x(m[0][0], m[0][1], m[0][2]), y(m[1][0], m[1][1], m[1][2]), z(m[2][0], m[2][1], m[2][2]);
        uniforms.Orientation = x.Cross(y).Dot(z) < 0.f ? -1.f : 1.f;
        return uniforms;
    }
}
