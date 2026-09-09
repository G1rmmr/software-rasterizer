#include "Application.hpp"
#include "ImageWriter.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace app {
    Application::Application(Options options)
        : options_(std::move(options)),
          state_(options_.Width, options_.Height, options_.Rendering),
          input_(state_),
          assets_(options_.Assets),
          camera_(math::ToRadian(config::FieldOfViewDegrees), static_cast<float>(options_.Width) / options_.Height,
                  config::NearPlane, config::FarPlane),
          light_(state_.GetLightDirection()),
          renderer_(options_.Width, options_.Height, options_.Workers) {
        state_.SetCameraDistance(options_.CameraDistance);
        BuildScene();
    }
    void Application::BuildScene() {
        const auto plane = assets_.CreatePlane();
        const auto wall = scene_.Add(plane);
        scene_.Get(wall).SetTransform(math::CreateTranslation({0.f, 0.f, -15.f}) * math::CreateScale({.5f, .5f, .5f}));
        const auto floor = scene_.Add(plane);
        scene_.Get(floor).SetTransform(math::CreateTranslation({0.f, -3.f, -5.f}) *
                                       math::CreateRotation({1.f, 0.f, 0.f}, math::ToRadian(-90.f)) *
                                       math::CreateScale({.5f, .5f, .5f}));
        std::shared_ptr<const scene::Model> model;
        if(options_.Model == "diablo")
            model = assets_.LoadDiablo();
        else if(options_.Model == "african")
            model = assets_.LoadAfrican();
        else if(options_.Model == "cube") {
            model = assets_.CreateCube();
            modelScale_ = 4.f;
        }
        else if(options_.Model == "sphere") {
            model = assets_.CreateSphere();
            modelScale_ = 3.f;
        }
        else
            throw std::invalid_argument("Model must be diablo, african, cube, or sphere");
        animatedModel_ = scene_.Add(std::move(model));
    }
    const graphics::FrameBuffer& Application::RenderFrame(float angle) {
        const auto width = state_.GetWidth(), height = state_.GetHeight();
        if(width != renderer_.GetFrame().GetWidth() || height != renderer_.GetFrame().GetHeight()) {
            renderer_.Resize(width, height);
            camera_.SetPerspective(math::ToRadian(config::FieldOfViewDegrees), static_cast<float>(width) / height,
                                   config::NearPlane, config::FarPlane);
        }
        camera_.LookAt({0.f, 4.f, state_.GetCameraDistance(), 1.f}, {0.f, 4.f, -5.f, 1.f});
        light_.SetDirection(state_.GetLightDirection());
        scene_.Get(animatedModel_)
            .SetTransform(math::CreateTranslation({0.f, 4.75f, -5.f}) * math::CreateRotation({0.f, 1.f, 0.f}, angle) *
                          math::CreateScale({modelScale_, modelScale_, modelScale_}));
        return renderer_.Render(scene_, camera_, light_, state_.GetRenderSettings());
    }
    int Application::Run() {
        if(options_.Headless) {
            const auto frames = options_.Frames ? options_.Frames : options_.Benchmark ? 300u : 1u;
            std::vector<float> frameTimes;
            debug::TimeData sums;
            if(options_.Benchmark) {
                frameTimes.reserve(frames);
                for(std::uint32_t i = 0; i < options_.WarmupFrames; ++i) RenderFrame(static_cast<float>(i) * .01f);
            }
            for(std::uint32_t i = 0; i < frames; ++i) {
                const auto start = std::chrono::steady_clock::now();
                RenderFrame(static_cast<float>(i) * .01f);
                if(options_.Benchmark) {
                    frameTimes.push_back(
                        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count());
                    const auto& times = renderer_.GetTimings();
                    sums.ShadowPassTime += times.ShadowPassTime;
                    sums.MainPassTime += times.MainPassTime;
                    sums.PostPassTime += times.PostPassTime;
                    sums.AAPassTime += times.AAPassTime;
                    sums.TotalFrameTime += times.TotalFrameTime;
                }
            }
            if(options_.Benchmark) {
                std::sort(frameTimes.begin(), frameTimes.end());
                const auto percentile = [&](double fraction) {
                    return frameTimes[static_cast<std::size_t>(std::ceil(fraction * frames)) - 1];
                };
                const float mean = std::accumulate(frameTimes.begin(), frameTimes.end(), 0.f) / frames;
                const auto overBudget =
                    std::count_if(frameTimes.begin(), frameTimes.end(), [](float ms) { return ms > 1000.f / 60.f; });
                std::cout << std::fixed << std::setprecision(3) << "Benchmark: model=" << options_.Model
                          << " size=" << options_.Width << "x" << options_.Height << " workers=" << options_.Workers
                          << " distance=" << state_.GetCameraDistance() << " warmup=" << options_.WarmupFrames
                          << " frames=" << frames << '\n'
                          << "Effects: shadow=" << options_.Rendering.Shadows
                          << " ssao=" << options_.Rendering.AmbientOcclusion
                          << " aa=" << options_.Rendering.AntiAliasing << " toon=" << options_.Rendering.Toon << '\n'
                          << "CPU frame ms: mean=" << mean << " p50=" << percentile(.5) << " p95=" << percentile(.95)
                          << " p99=" << percentile(.99) << " max=" << frameTimes.back() << " fps=" << 1000.f / mean
                          << " over16.667=" << overBudget << '/' << frames << '\n'
                          << "Mean pass ms: shadow=" << sums.ShadowPassTime / frames
                          << " geometry=" << sums.MainPassTime / frames << " ssao=" << sums.PostPassTime / frames
                          << " aa=" << sums.AAPassTime / frames << " renderer=" << sums.TotalFrameTime / frames << '\n'
                          << "Includes scene update and CPU rendering; excludes loading, image output, window "
                             "presentation and vsync.\n";
            }
            if(!options_.Output.empty()) WriteBmp(options_.Output, renderer_.GetFrame());
            std::cout << "Rendered " << frames << " frame(s), " << options_.Width << "x" << options_.Height
                      << ", final render " << renderer_.GetTimings().TotalFrameTime << " ms\n";
            return 0;
        }
        window_ = std::make_unique<Window>(config::WindowTitle, state_.GetWidth(), state_.GetHeight(), true);
        input_.Attach(window_->NativeHandle());
        if(options_.Profiler)
            profilerWindow_ = std::make_unique<Window>("Performance | FPS / frame time / render passes",
                                                       ProfilerView::Width, ProfilerView::Height);
        const auto start = std::chrono::steady_clock::now();
        std::uint32_t frames = 0;
        std::vector<float> displayWorkTimes;
        if(options_.Frames > options_.WarmupFrames) displayWorkTimes.reserve(options_.Frames - options_.WarmupFrames);
        auto firstPresented = start, lastPresented = start;
        auto previousFrameStart = start;
        ProfilerMetrics profile;
        profile.ViewportWidth = state_.GetWidth();
        profile.ViewportHeight = state_.GetHeight();
        profile.CameraDistance = state_.GetCameraDistance();
        profile.Workers = options_.Workers;
        debug::TimeData previousTimings;
        while(state_.IsRunning() && window_->IsOpen()) {
            const auto frameStart = std::chrono::steady_clock::now();
            // Measure complete application frames, including both window updates
            // and frame pacing. Render-only timing cannot report actual loop FPS.
            if(frames > 0)
                profile.FrameIntervalMs =
                    std::chrono::duration<float, std::milli>(frameStart - previousFrameStart).count();
            previousFrameStart = frameStart;
            const float seconds = std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
            const auto& frame = RenderFrame(seconds * .6f);
            const auto renderEnd = std::chrono::steady_clock::now();
            if(!window_->Present(frame)) break;
            if(profilerWindow_ && profilerWindow_->IsOpen())
                profilerWindow_->Present(profiler_.Draw(previousTimings, state_.GetRenderSettings(), profile));
            const auto presentEnd = std::chrono::steady_clock::now();
            previousTimings = renderer_.GetTimings();
            profile.CpuWorkMs = std::chrono::duration<float, std::milli>(renderEnd - frameStart).count();
            profile.PresentMs = std::chrono::duration<float, std::milli>(presentEnd - renderEnd).count();
            profile.ViewportWidth = frame.GetWidth();
            profile.ViewportHeight = frame.GetHeight();
            profile.CameraDistance = state_.GetCameraDistance();
            ++frames;
            if(options_.Frames && frames > options_.WarmupFrames) {
                const auto presented = std::chrono::steady_clock::now();
                if(displayWorkTimes.empty()) firstPresented = presented;
                lastPresented = presented;
                displayWorkTimes.push_back(std::chrono::duration<float, std::milli>(presented - frameStart).count());
            }
            if(options_.Frames && frames >= options_.Frames) break;
            const auto waitStart = std::chrono::steady_clock::now();
            if(!window_->Wait()) break;
            profile.WaitMs =
                std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - waitStart).count();
        }
        if(displayWorkTimes.size() > 1) {
            const auto count = displayWorkTimes.size();
            const float seconds = std::chrono::duration<float>(lastPresented - firstPresented).count();
            const float mean = std::accumulate(displayWorkTimes.begin(), displayWorkTimes.end(), 0.f) / count;
            std::sort(displayWorkTimes.begin(), displayWorkTimes.end());
            std::cout
                << std::fixed << std::setprecision(3) << "Window: measured=" << count
                << " display_fps=" << (count - 1) / seconds << " work_mean_ms=" << mean
                << " work_p95_ms=" << displayWorkTimes[static_cast<std::size_t>(std::ceil(.95 * count)) - 1] << '\n'
                << "Work includes scene, renderer, profiler and window updates; display FPS includes frame pacing.\n";
        }
        if(!options_.Output.empty()) WriteBmp(options_.Output, renderer_.GetFrame());
        return 0;
    }
}
