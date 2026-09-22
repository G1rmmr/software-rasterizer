#include "../graphics/DirectionalLight.hpp"
#include "../graphics/Rasterizer.hpp"
#include "../shaders/Model.hpp"
#include "../shaders/Shadow.hpp"
#include "../shaders/Toon.hpp"
#include "TestSupport.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

void RunShaderTests() {
    shader::DrawUniforms uniforms;
    uniforms.CameraPos = {0.f, 0.f, 5.f, 1.f};
    uniforms.LightDir = {0.f, 0.f, 1.f, 0.f};
    shader::Fragment fragment;
    fragment.Normal = {0.f, 0.f, 1.f};
    fragment.Tangent = {1.f, 0.f, 0.f, 1.f};
    fragment.Color = {.8f, .4f, .2f, .4f};
    fragment.WorldPos = {0.f, 0.f, 0.f, 1.f};
    graphics::Material material;

    material.Alpha = graphics::AlphaMode::Opaque;
    CHECK((shader::Model(uniforms, material).Shade(fragment).Color >> 24) == 255u);
    CHECK((shader::Toon(uniforms, material).Shade(fragment).Color >> 24) == 255u);
    material.Alpha = graphics::AlphaMode::Mask;
    material.AlphaCutoff = .5f;
    CHECK(shader::Model(uniforms, material).Shade(fragment).Color == 0u);
    CHECK(shader::Toon(uniforms, material).Shade(fragment).Color == 0u);
    CHECK(shader::Shadow(uniforms, material).Shade(fragment).Color == 0u);
    material.AlphaCutoff = .4f;
    CHECK((shader::Model(uniforms, material).Shade(fragment).Color >> 24) == 255u);
    CHECK((shader::Shadow(uniforms, material).Shade(fragment).Color >> 24) == 255u);
    material.Alpha = graphics::AlphaMode::Blend;
    CHECK((shader::Model(uniforms, material).Shade(fragment).Color >> 24) == 102u);
    CHECK((shader::Toon(uniforms, material).Shade(fragment).Color >> 24) == 102u);
    material.AlphaCutoff = .5f;
    CHECK(shader::Shadow(uniforms, material).Shade(fragment).Color == 0u);

    // Anisotropic model transforms use inverse-transpose normals; the G-buffer
    // gets exactly one subsequent view transform in both lighting styles.
    uniforms.Model = math::CreateScale({2.f, 1.f, .5f});
    uniforms.NormalMatrix = uniforms.Model.Inv().Transpose();
    uniforms.ModelViewProjection = uniforms.Model;
    uniforms.View = math::CreateRotation({0.f, 1.f, 0.f}, math::ToRadian(90.f));
    shader::Vertex vertex;
    vertex.Pos = {0.f, 0.f, 0.f, 1.f};
    vertex.Normal = {1.f, 1.f, 0.f, 0.f};
    vertex.Tangent = {1.f, -1.f, 0.f, -1.f};
    vertex.Color = {1.f, 1.f, 1.f, 1.f};
    material.Alpha = graphics::AlphaMode::Opaque;
    const shader::Model modelShader(uniforms, material);
    const shader::Toon toonShader(uniforms, material);
    const auto varying = modelShader.Process(vertex);
    CHECK_NEAR(varying.Normal.X, 1.f / std::sqrt(5.f), 1e-6);
    CHECK_NEAR(varying.Normal.Y, 2.f / std::sqrt(5.f), 1e-6);
    CHECK_NEAR(varying.Tangent.W, -1.f, 1e-6);
    fragment.Normal = varying.Normal;
    fragment.Tangent = varying.Tangent;
    const auto lit = modelShader.Shade(fragment), toon = toonShader.Shade(fragment);
    for(const auto normal : {lit.ViewNormal, toon.ViewNormal}) {
        CHECK_NEAR(normal.X, 0.f, 1e-6);
        CHECK_NEAR(normal.Y, 2.f / std::sqrt(5.f), 1e-6);
        CHECK_NEAR(normal.Z, -1.f / std::sqrt(5.f), 1e-6);
        CHECK_NEAR(normal.Length(), 1.f, 1e-6);
    }

    const auto texture =
        std::make_shared<const graphics::Texture>(1, 1, std::vector<std::uint8_t>{255u, 128u, 64u, 128u});
    fragment.UV.X = std::numeric_limits<float>::quiet_NaN();
    material.DiffuseMap = texture;
    // Invalid texture coordinates propagate as recoverable draw errors rather than terminating.
    CHECK_THROWS(shader::SampleAlbedo(material, fragment));
    CHECK_THROWS(shader::Model(uniforms, material).Shade(fragment));
    CHECK_THROWS(shader::Toon(uniforms, material).Shade(fragment));
    CHECK_THROWS(shader::Shadow(uniforms, material).Shade(fragment));
    material.DiffuseMap.reset();
    material.NormalMap = texture;
    CHECK_THROWS(shader::SampleNormal(material, fragment));
    CHECK_THROWS(shader::Model(uniforms, material).Shade(fragment));
    CHECK_THROWS(shader::Toon(uniforms, material).Shade(fragment));

    graphics::DirectionalLight light({1.f, 2.f, 3.f});
    const auto saved = light.GetDirection();
    CHECK_NEAR(saved.Length(), 1.f, 1e-6);
    CHECK_THROWS(light.SetDirection({1e30f, 1e30f, 0.f}));
    CHECK(light.GetDirection() == saved);
    CHECK_THROWS(light.SetDirection({0.f, 0.f, 0.f}));
    CHECK_THROWS(light.SetDirection({1e-6f, 0.f, 0.f}));
    CHECK_THROWS(light.SetDirection({std::numeric_limits<float>::infinity(), 0.f, 0.f}));
    CHECK(light.GetDirection() == saved);

    // Orthographic shadow sampling uses the same zero-to-one depth range as drawing.
    graphics::FrameBuffer shadowFrame(8, 8);
    const auto projection = math::CreateOrtho(-1.f, 1.f, -1.f, 1.f, 1.f, 11.f);
    const shader::ShadowMap shadows(shadowFrame, projection, 1.f, 11.f);
    CHECK_NEAR(shadows.Visibility({0.f, 0.f, -6.f}, {0.f, 0.f, 1.f}, {0.f, 0.f, 1.f}), 1.f, 1e-6);
    for(unsigned y = 0; y < 8; ++y)
        for(unsigned x = 0; x < 8; ++x) shadowFrame.SetDepth(x, y, .25f);
    CHECK_NEAR(shadows.Visibility({0.f, 0.f, -6.f}, {0.f, 0.f, 1.f}, {0.f, 0.f, 1.f}), 0.f, 1e-6);
    CHECK_NEAR(shadows.Visibility({0.f, 0.f, -2.f}, {0.f, 0.f, 1.f}, {0.f, 0.f, 1.f}), 1.f, 1e-6);

    // Quantization boundaries must not amplify a one-ULP interpolation difference
    // into a visible step. Keep a value genuinely below the boundary in its old bin.
    graphics::Material flatMaterial;
    shader::DrawUniforms flatUniforms;
    flatUniforms.CameraPos = {0.f, 0.f, 5.f, 1.f};
    flatUniforms.LightDir = {1.f, 0.f, 0.f};
    shader::Fragment flatFragment;
    flatFragment.Normal = {0.f, 0.f, 1.f};
    flatFragment.WorldPos = {0.f, 0.f, 0.f, 1.f};
    const shader::Toon flatShader(flatUniforms, flatMaterial);
    for(const float channel : {std::nextafter(.7f, 0.f), .7f, std::nextafter(.7f, 1.f)}) {
        flatFragment.Color = {channel, channel, channel, 1.f};
        CHECK_NEAR(flatShader.Shade(flatFragment).Color & 255u, 27u, 0);
    }
    flatFragment.Color = {.699f, .699f, .699f, 1.f};
    CHECK_NEAR(flatShader.Shade(flatFragment).Color & 255u, 23u, 0);

    // Render a real tilted, untextured plane. Its perspective-correct color is
    // constant even though the vertices have different reciprocal W values.
    // Only the rasterizer and Toon shader run: no shadow, AO, or AA can hide noise.
    ParallelExecutor executor(0);
    graphics::Rasterizer rasterizer(executor);
    graphics::FrameBuffer planeFrame(128, 96);
    planeFrame.Clear(0xff000000u);
    flatUniforms.CameraPos = {0.f, 0.f, 0.f, 1.f};
    flatUniforms.LightDir = math::Vector(2.f, 0.f, -1.f).NormalizedDirection();
    flatUniforms.ModelViewProjection = math::CreatePerspective(math::ToRadian(60.f), 128.f / 96.f, .1f, 100.f);
    const math::Vector planeNormal = math::Vector(1.f, 0.f, 2.f).NormalizedDirection();
    std::array<shader::Vertex, 4> planeVertices;
    const std::array<math::Vector, 4> planeCorners = {
        math::Vector(-.8f, -.7f, -2.1f, 1.f), math::Vector(.8f, -.7f, -2.9f, 1.f), math::Vector(.8f, .7f, -2.9f, 1.f),
        math::Vector(-.8f, .7f, -2.1f, 1.f)};
    for(std::size_t i = 0; i < planeVertices.size(); ++i) {
        planeVertices[i].Pos = planeCorners[i];
        planeVertices[i].Normal = planeNormal;
        planeVertices[i].Color = {.7f, .7f, .7f, 1.f};
    }
    const std::array<std::uint32_t, 6> planeIndices = {0, 1, 2, 0, 2, 3};
    graphics::RasterizerOptions rasterOptions;
    rasterOptions.Cull = graphics::CullMode::None;
    const shader::Toon planeShader(flatUniforms, flatMaterial);
    rasterizer.Render(planeFrame, planeShader, planeVertices, planeIndices, rasterOptions);
    std::size_t covered = 0;
    for(const auto color : planeFrame.GetColors()) {
        if(color == 0xff000000u) continue;
        ++covered;
        CHECK(color == 0xff1b1b1bu);
    }
    CHECK(covered > 500);

    // The opaque shadow specialization must preserve the full coverage shader's
    // depth while leaving unrelated attachments untouched.
    graphics::FrameBuffer fullShadow(128, 96), depthShadow(128, 96);
    depthShadow.Clear(0xff123456u);
    rasterizer.Render(fullShadow, shader::Shadow(flatUniforms, flatMaterial), planeVertices, planeIndices,
                      rasterOptions);
    rasterizer.Render(depthShadow, shader::OpaqueShadow(flatUniforms.ModelViewProjection), planeVertices, planeIndices,
                      rasterOptions);
    for(std::size_t i = 0; i < fullShadow.GetDepths().size(); ++i) {
        CHECK(fullShadow.GetDepths()[i] == depthShadow.GetDepths()[i]);
        CHECK(depthShadow.GetColors()[i] == 0xff123456u);
        CHECK(depthShadow.GetNormals()[i] == math::Vector{});
    }
}
