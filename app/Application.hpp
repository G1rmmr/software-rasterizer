#pragma once
#include "../AppState.hpp"
#include "../Input.hpp"
#include "../Preferences.hpp"
#include "../graphics/Renderer.hpp"
#include "../resources/ModelFactory.hpp"
#include "../scene/MirScene.hpp"
#include "../scene/Scene.hpp"
#include "ProfilerView.hpp"
#include "Window.hpp"
#include <filesystem>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace app {
    enum class SceneBackend { Oop, Mir };

    struct Options {
        bool Headless = false;
        bool Benchmark = false;
        std::uint32_t WarmupFrames = 30;
        float CameraDistance = 45.f;
        bool Profiler = true;
        std::uint32_t Width = config::Width, Height = config::Height;
        std::uint32_t Frames = 0; // Interactive: unlimited. Headless: one frame by default.
        std::uint32_t Instances = 1;
        SceneBackend Backend = SceneBackend::Oop;
        std::size_t Workers = ParallelExecutor::DefaultWorkerCount();
        std::filesystem::path Assets = "assets";
        std::filesystem::path Output;
        std::string Model = "diablo";
        graphics::RenderSettings Rendering{.ClearColor = config::ClearColor};
    };
    class Application final {
    public:
        explicit Application(Options options);
        int Run();

    private:
        const graphics::FrameBuffer& RenderFrame(float angle);
        void BuildScene();
        Options options_;
        AppState state_;
        InputController input_;
        resources::ModelFactory assets_;
        scene::Scene scene_;
        std::unique_ptr<scene::MirScene> mirScene_;
        struct AnimatedInstance {
            scene::Scene::Handle Handle = 0;
            scene::MirScene::Handle MirHandle{};
            math::Vector Position;
            float Phase = 0.f;
        };
        std::vector<AnimatedInstance> animatedModels_;
        float modelScale_ = 8.f;
        graphics::Camera camera_;
        graphics::DirectionalLight light_;
        graphics::Renderer renderer_;
        ProfilerView profiler_;
        // Windows and C callbacks are released before their referenced state.
        std::unique_ptr<Window> window_;
        std::unique_ptr<Window> profilerWindow_;
    };
}
