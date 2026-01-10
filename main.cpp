#include <cstdio>
#include <vector>

#include <MiniFB.h>

#include "World.hpp"
#include "graphics/FrameBuffer.hpp"
#include "graphics/Rasterizer.hpp"

int main() {
    const char* WINDOW_TITLE = "Software Rasterizer (NEON/Minifb)";

    struct mfb_window *window = mfb_open_ex(WINDOW_TITLE, world::WIDTH, world::HEIGHT, WF_RESIZABLE);
    if (!window) {
        std::fprintf(stderr, "ERROR: Could not open minifb window\n");
        return -1;
    }

    graphics::FrameBuffer frame(world::WIDTH, world::HEIGHT);

    world::CreateIcoSphere(1.f, 3);

    float angle = 0.f;
    do {
        frame.Clear(world::COLOR);

        angle += 0.02f;
        shader::Default shader{
            world::GetMVP(angle),
            math::CreateViewport(static_cast<float>(world::WIDTH), static_cast<float>(world::HEIGHT))
        };

        graphics::Render(frame, shader, world::ModelVertices, world::ModelIndices, graphics::PrimitiveType::Lines);

        int state = mfb_update(window, frame.GetColor());
        if (state < 0) {
            window = nullptr;
            break;
        }

    } while (mfb_wait_sync(window));

    return 0;
}
