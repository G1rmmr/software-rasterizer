#include "TestSupport.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "../graphics/Camera.hpp"
#include "../graphics/Renderer.hpp"
#include "../resources/ModelFactory.hpp"
#include "../scene/Scene.hpp"

namespace {
    graphics::Camera CameraFor(std::uint32_t width, std::uint32_t height) {
        graphics::Camera camera(math::ToRadian(55.f), static_cast<float>(width) / height, .1f, 30.f);
        camera.LookAt({0.f, 0.f, 5.f}, {0.f, 0.f, 0.f});
        return camera;
    }

    graphics::RenderSettings BasicSettings() {
        graphics::RenderSettings settings;
        settings.Shadows = false;
        settings.AmbientOcclusion = false;
        settings.AntiAliasing = false;
        settings.ClearColor = 0xff102030u;
        return settings;
    }

    void CheckFiniteAttachments(const graphics::FrameBuffer& frame) {
        const auto size = static_cast<std::size_t>(frame.GetWidth()) * frame.GetHeight();
        CHECK(frame.GetColors().size() == size);
        CHECK(frame.GetDepths().size() == size);
        CHECK(frame.GetNormals().size() == size);
        for(const auto depth : frame.GetDepths()) CHECK(std::isfinite(depth) && depth >= 0.f && depth <= 1.f);
        for(const auto& normal : frame.GetNormals()) {
            CHECK(std::isfinite(normal.X));
            CHECK(std::isfinite(normal.Y));
            CHECK(std::isfinite(normal.Z));
            CHECK(std::isfinite(normal.W));
        }
    }

    void CheckSameFrame(const graphics::FrameBuffer& a, const graphics::FrameBuffer& b) {
        CHECK(a.GetWidth() == b.GetWidth());
        CHECK(a.GetHeight() == b.GetHeight());
        CHECK(std::equal(a.GetColors().begin(), a.GetColors().end(), b.GetColors().begin()));
        for(std::size_t i = 0; i < a.GetDepths().size(); ++i) CHECK_NEAR(a.GetDepths()[i], b.GetDepths()[i], 1e-6f);
    }

    void FrameBufferAndCameraContracts() {
        graphics::FrameBuffer frame(64, 48);
        frame.SetPixel(0, 0, 0xffff0000u);
        frame.SetDepth(0, 0, .25f);
        frame.SetNormal(0, 0, {0.f, 0.f, 1.f, 0.f});
        CHECK_THROWS(frame.Resize(0, 49));
        CHECK(frame.GetWidth() == 64 && frame.GetHeight() == 48);
        CHECK(frame.GetPixel(0, 0) == 0xffff0000u);
        CHECK_NEAR(frame.GetDepth(0, 0), .25f, 1e-6f);
        frame.ClearDepth();
        CHECK(frame.GetDepth(0, 0) == 1.f);
        CHECK(frame.GetPixel(0, 0) == 0xffff0000u);
        CHECK(frame.GetNormal(0, 0) == math::Vector(0.f, 0.f, 1.f, 0.f));
        frame.Resize(65, 49);
        CheckFiniteAttachments(frame);
        CHECK(frame.GetWidth() == 65 && frame.GetHeight() == 49);
        frame.Clear(0xff345678u);
        for(const auto pixel : frame.GetColors()) CHECK(pixel == 0xff345678u);
        for(const auto depth : frame.GetDepths()) CHECK(depth == 1.f);
        for(const auto& normal : frame.GetNormals()) CHECK_NEAR(normal.Length(), 0.f, 1e-6f);

        auto camera = CameraFor(64, 48);
        const auto view = camera.GetView();
        const auto projection = camera.GetProjection();
        CHECK_THROWS(camera.LookAt({1.f, 2.f, 3.f}, {1.f, 2.f, 3.f}));
        CHECK_THROWS(camera.LookAt({0.f, 0.f, 5.f}, {}, {0.f, 0.f, 1.f}));
        CHECK_THROWS(camera.LookAt({0.f, 0.f, 5.f}, {}, {1e30f, 1e30f, 0.f}));
        CHECK_THROWS(camera.LookAt({0.f, 0.f, 100.f}, {}, {1e-7f, 0.f, 0.f}));
        CHECK(camera.GetView() == view);
        CHECK_THROWS(camera.SetPerspective(0.f, 1.f, .1f, 30.f));
        CHECK(camera.GetProjection() == projection);
        CHECK(camera.GetFrustum().IsSphereInside({0.f, 0.f, 0.f}, .5f));
        CHECK(!camera.GetFrustum().IsSphereInside({0.f, 0.f, 10.f}, .5f));
        const auto nearClip = projection * math::Vector(0.f, 0.f, -.1f, 1.f);
        const auto farClip = projection * math::Vector(0.f, 0.f, -30.f, 1.f);
        CHECK_NEAR(nearClip.Z / nearClip.W, 0.f, 1e-6f);
        CHECK_NEAR(farClip.Z / farClip.W, 1.f, 1e-6f);
        const auto restored = camera.GetInverseProjection() * (projection * math::Vector(.2f, -.3f, -2.f, 1.f));
        CHECK_NEAR(restored.X / restored.W, .2f, 1e-5f);
        CHECK_NEAR(restored.Y / restored.W, -.3f, 1e-5f);
        CHECK_NEAR(restored.Z / restored.W, -2.f, 1e-5f);
    }

    scene::Scene ProceduralScene(resources::ModelFactory& factory) {
        scene::Scene scene;
        const auto cube = scene.Add(factory.CreateCube());
        scene.Get(cube).SetTransform(math::CreateTranslation({-.95f, -.05f, 0.f}) * math::CreateScale({.6f, .6f, .6f}));
        const auto sphere = scene.Add(factory.CreateSphere(.8f, 1));
        scene.Get(sphere).SetTransform(math::CreateTranslation({.85f, .1f, 0.f}));
        const auto background = scene.Add(factory.CreatePlane(2.5f));
        scene.Get(background).SetTransform(math::CreateTranslation({0.f, 0.f, -1.3f}));
        return scene;
    }

    void RenderPassesAndResize(resources::ModelFactory& factory) {
        auto scene = ProceduralScene(factory);
        scene::Scene empty;
        auto camera = CameraFor(64, 48);
        const graphics::DirectionalLight light({-.3f, .6f, 1.f});
        graphics::Renderer serial(64, 48, 0), parallel(64, 48, 2);
        graphics::RenderSettings settings;
        settings.ClearColor = 0xff102030u;

        for(const bool toon : {false, true}) {
            settings.Toon = toon;
            serial.Render(scene, camera, light, settings);
            parallel.Render(scene, camera, light, settings);
            CheckFiniteAttachments(serial.GetFrame());
            CheckFiniteAttachments(parallel.GetFrame());
            CheckSameFrame(serial.GetFrame(), parallel.GetFrame());
            CHECK(std::count_if(serial.GetFrame().GetDepths().begin(), serial.GetFrame().GetDepths().end(),
                                [](float depth) { return depth < 1.f; }) > 30);
            CHECK(std::any_of(serial.GetFrame().GetColors().begin(), serial.GetFrame().GetColors().end(),
                              [&](std::uint32_t color) { return color != settings.ClearColor; }));
        }

        serial.Resize(65, 49);
        parallel.Resize(65, 49);
        camera.SetPerspective(math::ToRadian(55.f), 65.f / 49.f, .1f, 30.f);
        serial.Render(scene, camera, light, settings);
        parallel.Render(scene, camera, light, settings);
        CHECK(serial.GetFrame().GetWidth() == 65 && serial.GetFrame().GetHeight() == 49);
        CheckFiniteAttachments(serial.GetFrame());
        CheckSameFrame(serial.GetFrame(), parallel.GetFrame());

        // Clearing an empty scene must remove color, depth and normals from the
        // preceding scene even while shadows, SSAO and AA remain enabled.
        settings.ClearColor = 0xff426384u;
        const auto& cleared = serial.Render(empty, camera, light, settings);
        for(const auto pixel : cleared.GetColors()) CHECK(pixel == settings.ClearColor);
        for(const auto depth : cleared.GetDepths()) CHECK(depth == 1.f);
        for(const auto& normal : cleared.GetNormals()) CHECK_NEAR(normal.Length(), 0.f, 1e-6f);
        CheckFiniteAttachments(cleared);

        // Diagnostic modes take their own path through the real geometry pass.
        settings = BasicSettings();
        settings.AmbientOcclusion = true;
        settings.AntiAliasing = true;
        for(const auto primitive : {graphics::PrimitiveType::Points, graphics::PrimitiveType::Lines}) {
            settings.Primitive = primitive;
            const auto& frame = serial.Render(scene, camera, light, settings);
            CheckFiniteAttachments(frame);
            CHECK(std::any_of(frame.GetDepths().begin(), frame.GetDepths().end(),
                              [](float depth) { return depth < 1.f; }));
        }
    }

    void ReflectionAndIndependentOwners(resources::ModelFactory& factory) {
        auto camera = CameraFor(65, 49);
        const graphics::DirectionalLight light({0.f, 0.f, 1.f});
        const auto settings = BasicSettings();
        scene::Scene planeScene;
        const auto plane = planeScene.Add(factory.CreatePlane(1.f));
        graphics::Renderer first(65, 49, 0), second(65, 49, 2);
        first.Render(planeScene, camera, light, settings);
        const std::vector<std::uint32_t> originalColors(first.GetFrame().GetColors().begin(),
                                                        first.GetFrame().GetColors().end());
        const std::vector<float> originalDepths(first.GetFrame().GetDepths().begin(),
                                                first.GetFrame().GetDepths().end());
        CHECK(first.GetFrame().GetDepth(32, 24) < 1.f);

        planeScene.Get(plane).SetTransform(math::CreateScale({-1.f, 1.f, 1.f}));
        second.Render(planeScene, camera, light, settings);
        CHECK(second.GetFrame().GetDepth(32, 24) < 1.f);
        CHECK(std::equal(originalColors.begin(), originalColors.end(), second.GetFrame().GetColors().begin()));
        for(std::size_t i = 0; i < originalDepths.size(); ++i)
            CHECK_NEAR(originalDepths[i], second.GetFrame().GetDepths()[i], 1e-6f);

        // Another renderer, target size, camera and scene must not mutate an
        // already completed frame or leave hidden process-wide pass state.
        scene::Scene empty;
        second.Resize(64, 48);
        auto otherCamera = CameraFor(64, 48);
        otherCamera.LookAt({0.f, 0.f, -5.f}, {});
        auto otherSettings = settings;
        otherSettings.ClearColor = 0xffabcdefu;
        second.Render(empty, otherCamera, light, otherSettings);
        CHECK(first.GetFrame().GetWidth() == 65 && first.GetFrame().GetHeight() == 49);
        CHECK(std::equal(originalColors.begin(), originalColors.end(), first.GetFrame().GetColors().begin()));
        for(const auto pixel : second.GetFrame().GetColors()) CHECK(pixel == 0xffabcdefu);
        CHECK(camera.GetPosition().Z == 5.f);

        planeScene.Get(plane).SetTransform(math::Matrix{});
        first.Render(planeScene, camera, light, settings);
        CHECK(std::equal(originalColors.begin(), originalColors.end(), first.GetFrame().GetColors().begin()));
        planeScene.Get(plane).SetVisible(false);
        first.Render(planeScene, camera, light, settings);
        for(const auto pixel : first.GetFrame().GetColors()) CHECK(pixel == settings.ClearColor);
    }
}

void RunRendererTests() {
    // Only procedural factory methods are used; this path is never opened.
    resources::ModelFactory factory("unused-procedural-assets");
    FrameBufferAndCameraContracts();
    RenderPassesAndResize(factory);
    ReflectionAndIndependentOwners(factory);
}
