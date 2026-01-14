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

    std::shared_ptr<Plane> backWall = std::make_shared<Plane>();
    backWall->Create();
    backWall->Model = math::CreateTranslation({0.f, 0.f, -15.f}) * math::CreateScale({0.5f, 0.5f, 0.5f});
    preferences::Objects.push_back(backWall);

    std::shared_ptr<Plane> plane = std::make_shared<Plane>();
    plane->Create();
    plane->Model = math::CreateTranslation({0.f, -3.f, -5.f}) *
                   math::CreateRotation({1.f, 0.f, 0.f}, math::ToRadian(-90.f)) * math::CreateScale({0.5f, 0.5f, 0.5f});

    preferences::Objects.push_back(plane);

    std::shared_ptr<Object> model = std::make_shared<Diablo>();
    model->Create();
    preferences::Objects.push_back(model);

    float angle = 0.f;
    do {
        preferences::Frame.Clear(preferences::COLOR);
        preferences::UpdateUniform(window);

        model->Model = math::CreateTranslation({0.f, 2.f, -5.f}) * math::CreateRotation({0.f, 1.f, 0.f}, angle) *
                       math::CreateScale({5.f, 5.f, 5.f});

        preferences::MapShadow();
        preferences::Render();
        preferences::Rasterizer.ApplySSAO(preferences::Uniform, preferences::NEAR, preferences::FAR);
        preferences::Rasterizer.ApplyPostAA();

        int state = mfb_update(window, preferences::Frame.GetColor());
        if(state < 0) {
            window = nullptr;
            break;
        }
        angle += 0.01f;
    } while(mfb_wait_sync(window));
    return 0;
}
