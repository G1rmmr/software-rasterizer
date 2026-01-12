#include <cstdio>
#include <memory>
#include <vector>

#include "Preferences.hpp"

#include "graphics/FrameBuffer.hpp"
#include "graphics/Rasterizer.hpp"

#include "objects/African.hpp"
#include "objects/Diablo.hpp"
#include "objects/Wall.hpp"

int main() {
    struct mfb_window* window = preferences::Init();
    if(!window) return -1;

    std::shared_ptr<Wall> backWall = std::make_shared<Wall>();
    backWall->Create();
    backWall->Model = math::CreateTranslation({0.f, 0.f, -15.f});
    preferences::Objects.push_back(backWall);

    /*
    std::shared_ptr<African> african = std::make_shared<African>();
    african->Create();
    preferences::Objects.push_back(african);
    */

    std::shared_ptr<Diablo> model = std::make_shared<Diablo>();
    model->Create();
    preferences::Objects.push_back(model);

    float angle = 0.f;
    do {
        preferences::Frame.Clear(preferences::COLOR);
        preferences::UpdateUniform(window);

        model->Model = math::CreateTranslation({-2.f, 0.f, 0.f}) * math::CreateRotation({0.f, 1.f, 0.f}, angle) *
                       math::CreateScale({2.f, 2.f, 2.f});

        preferences::MapShadow();
        preferences::Render();
        // preferences::Rasterizer.ApplyPostAA();

        int state = mfb_update(window, preferences::Frame.GetColor());
        if(state < 0) {
            window = nullptr;
            break;
        }
        angle += 0.01f;
    } while(mfb_wait_sync(window));
    return 0;
}
