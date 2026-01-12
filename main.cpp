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

    std::shared_ptr<Wall> wall = std::make_shared<Wall>();
    wall->Create();
    preferences::Objects.push_back(wall);

    std::shared_ptr<Diablo> diablo = std::make_shared<Diablo>();
    diablo->Create();
    diablo->ShouldRender = false;

    preferences::Objects.push_back(diablo);

    std::shared_ptr<African> african = std::make_shared<African>();
    african->Create();
    preferences::Objects.push_back(african);

    float angle = 0.f;

    do {
        preferences::Frame.Clear(preferences::COLOR);
        preferences::UpdateUniform(window);

        diablo->Model = math::CreateRotation({0.f, 1.f, 0.f}, angle);
        african->Model = math::CreateRotation({0.f, 1.f, 0.f}, angle);

        preferences::Render();
        // preferences::Rasterizer.ApplyPostAA();

        int state = mfb_update(window, preferences::Frame.GetColor());
        if(state < 0) {
            window = nullptr;
            break;
        }
        angle += 0.005f;

    } while(mfb_wait_sync(window));
    return 0;
}
