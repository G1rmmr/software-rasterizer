#include <cstdio>
#include <vector>

#include <MiniFB.h>

#include "World.hpp"
#include "graphics/FrameBuffer.hpp"
#include "graphics/Rasterizer.hpp"

int main() {
    const char* WINDOW_TITLE = "Software Rasterizer";

    struct mfb_window* window = mfb_open_ex(WINDOW_TITLE, world::WIDTH, world::HEIGHT, WF_RESIZABLE);
    if(!window) {
        std::fprintf(stderr, "ERROR: Could not open minifb window\n");
        return -1;
    }

    graphics::FrameBuffer frame(world::WIDTH, world::HEIGHT);

    world::CreateIcoSphere(2.1f, 3);

    float angle = 0.f;
    float trans = 0.f;
    float step = -0.02f;

    do {
        frame.Clear(world::COLOR);
        world::Model = math::CreateTranslation({0.f, 0.f, trans}) * math::CreateRotation({0.f, 1.f, 0.f}, angle);

        shader::Default shader{world::Model, world::GetMVP(), math::CreateViewport(world::WIDTH, world::HEIGHT),
                               world::LightDir};

        graphics::Render(frame, shader, world::ModelVertices, world::ModelIndices, graphics::PrimitiveType::Triangles);

        int state = mfb_update(window, frame.GetColor());
        if(state < 0) {
            window = nullptr;
            break;
        }

        angle += 0.01f;
        trans += step;

        if(trans < -10.f) {
            step = 0.02f;
        }

        if(trans >= 2.f) {
            step = -0.02f;
        }

    } while(mfb_wait_sync(window));

    return 0;
}
