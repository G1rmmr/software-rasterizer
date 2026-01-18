#include <chrono>
#include <cstdio>
#include <memory>
#include <vector>

#include "Preferences.hpp"

#include "graphics/FrameBuffer.hpp"
#include "graphics/Rasterizer.hpp"

#include "objects/African.hpp"
#include "objects/Diablo.hpp"
#include "objects/Plane.hpp"

int main() {
    struct mfb_window* window = preferences::Init();
    if(!window) return -1;

    struct mfb_window* debugger = preferences::debug::Init();
    if(!debugger) return -1;

    std::shared_ptr<Plane> backWall = std::make_shared<Plane>();
    backWall->Create();
    backWall->Model = math::CreateTranslation({0.f, 0.f, -15.f}) * math::CreateScale({0.5f, 0.5f, 0.5f});
    backWall->CalculateBounds();
    preferences::Objects.push_back(backWall);

    std::shared_ptr<Plane> plane = std::make_shared<Plane>();
    plane->Create();
    plane->Model = math::CreateTranslation({0.f, -3.f, -5.f}) *
                   math::CreateRotation({1.f, 0.f, 0.f}, math::ToRadian(-90.f)) * math::CreateScale({0.5f, 0.5f, 0.5f});

    plane->CalculateBounds();
    preferences::Objects.push_back(plane);

    std::shared_ptr<Object> model = std::make_shared<African>();
    model->Create();
    backWall->IsStatic = false;
    preferences::Objects.push_back(model);

    float angle = 0.f;
    do {
        preferences::UpdateUniform(window);

        model->Model = math::CreateTranslation({0.f, 4.75f, -5.f}) * math::CreateRotation({0.f, 1.f, 0.f}, angle) *
                       math::CreateScale({8.f, 8.f, 8.f});

        model->CalculateBounds();

        std::chrono::steady_clock::time_point frameStart = std::chrono::high_resolution_clock::now();

        std::chrono::steady_clock::time_point t0 = std::chrono::high_resolution_clock::now();
        preferences::pre::Render(preferences::State.IsShowingShadowMap);
        debug::Profiler.ShadowPassTime =
            std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

        std::chrono::steady_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        preferences::main::Render();
        debug::Profiler.MainPassTime =
            std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t1).count();

        std::chrono::steady_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        preferences::post::Render(preferences::State.IsShowingSSAO);
        debug::Profiler.PostPassTime =
            std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t2).count();

        std::chrono::steady_clock::time_point t3 = std::chrono::high_resolution_clock::now();
        preferences::CurrFrame->AntiAlias(preferences::State.IsShowingAA);
        debug::Profiler.AAPassTime =
            std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t3).count();

        debug::Profiler.TotalFrameTime =
            std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - frameStart).count();

        preferences::debug::Draw(debugger);

        int state = mfb_update(window, preferences::CurrFrame->GetColor());

        if(mfb_update(debugger, preferences::debug::Buffer.data()) < 0) debugger = nullptr;

        if(state < 0) {
            window = nullptr;
            break;
        }
        angle += 0.01f;
    } while(mfb_wait_sync(window));
    return 0;
}
