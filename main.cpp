#include <cstdio>
#include <vector>

#include <MiniFB.h>

#include "World.hpp"
#include "graphics/FrameBuffer.hpp"
#include "graphics/Rasterizer.hpp"

struct AppState {
    float Width = static_cast<float>(world::WIDTH);
    float Height = static_cast<float>(world::HEIGHT);

    float X = 0.f;
    float Y = 0.f;

    graphics::PrimitiveType NowType = graphics::PrimitiveType::Triangles;
    bool IsLeftDown = false;
} State;

void MouseButtonCallback(struct mfb_window* window, mfb_mouse_button button, mfb_key_mod mod, bool isPressed) {
    State.IsLeftDown = button == MOUSE_BTN_1 && isPressed;
}

void MouseMoveCallback(struct mfb_window* window, int x, int y) {
    State.X = static_cast<float>(x);
    State.Y = static_cast<float>(y);
}

void ResizeCallback(struct mfb_window* window, int width, int height) {
    State.Width = static_cast<float>(width);
    State.Height = static_cast<float>(height);
}

void KeyboardCallback(struct mfb_window* window, mfb_key key, mfb_key_mod mod, bool isPressed) {
    switch(key) {
    case KB_KEY_SPACE:
        if(isPressed) {
            if(State.NowType == graphics::PrimitiveType::Triangles) {
                State.NowType = graphics::PrimitiveType::Lines;
            }
            else if(State.NowType == graphics::PrimitiveType::Lines) {
                State.NowType = graphics::PrimitiveType::Points;
            }
            else {
                State.NowType = graphics::PrimitiveType::Triangles;
            }
        }
        break;
    default: break;
    }
}

void SetWorldUniform(mfb_window* window) {
    world::Uniform.CameraPos = world::EYE;
    world::Uniform.View = math::CreateLookAt(world::Uniform.CameraPos, world::Target, world::UP);

    const std::uint8_t* mouseBtn = mfb_get_mouse_button_buffer(window);
    if(State.IsLeftDown) {
        float ndcX = (State.X / State.Width) * 2.f - 1.f;
        float ndcY = -((State.Y / State.Height) * 2.f - 1.f);

        float distSq = ndcX * ndcX + ndcY * ndcY;
        float z = std::sqrt(1.f - distSq);

        math::Vector viewLightDir = math::Vector(ndcX, ndcY, z, 0.f);
        math::Vector worldLightDir;

        worldLightDir.X = world::Uniform.View[0][0] * viewLightDir.X + world::Uniform.View[1][0] * viewLightDir.Y +
                          world::Uniform.View[2][0] * viewLightDir.Z;
        worldLightDir.Y = world::Uniform.View[0][1] * viewLightDir.X + world::Uniform.View[1][1] * viewLightDir.Y +
                          world::Uniform.View[2][1] * viewLightDir.Z;
        worldLightDir.Z = world::Uniform.View[0][2] * viewLightDir.X + world::Uniform.View[1][2] * viewLightDir.Y +
                          world::Uniform.View[2][2] * viewLightDir.Z;
        worldLightDir.W = 0.f;

        world::Uniform.LightDir = worldLightDir.Norm();
    }

    constexpr float aspect = static_cast<float>(world::WIDTH) / world::HEIGHT;
    world::Uniform.Proj = math::CreatePerspective(math::ToRadian(world::FOV_ANGLE), aspect, world::NEAR, world::FAR);
}

int main() {
    const char* WINDOW_TITLE = "Software Rasterizer";

    struct mfb_window* window = mfb_open_ex(WINDOW_TITLE, world::WIDTH, world::HEIGHT, WF_RESIZABLE);
    if(!window) {
        return -1;
    }

    mfb_set_mouse_button_callback(window, MouseButtonCallback);
    mfb_set_mouse_move_callback(window, MouseMoveCallback);
    mfb_set_resize_callback(window, ResizeCallback);
    mfb_set_keyboard_callback(window, KeyboardCallback);

    graphics::FrameBuffer frame(world::WIDTH, world::HEIGHT);
    graphics::Rasterizer rasterizer(frame);

    world::CreateIcoSphere(2.f, 3);
    // world::CreateDiablo();

    world::Uniform.LightDir = math::Vector(0.f, 0.f, 1.f, 0.f).Norm();

    float angle = 0.f;
    float trans = 0.f;
    float step = -0.02f;

    float currentVisibleIndices = 0.0f;
    float speed = 30.0f;

    do {
        frame.Clear(world::COLOR);

        world::Uniform.Model = math::CreateRotation({0.f, 1.f, 0.f}, angle);
        world::Target = world::Uniform.Model * math::Vector(0.f, 0.f, 0.f, 1.f);

        SetWorldUniform(window);

        shader::Default shader;
        shader.Uniform = world::Uniform;

        rasterizer.Render(shader, world::ModelVertices, world::ModelIndices, State.NowType, static_cast<std::size_t>(currentVisibleIndices));
        // rasterizer.ApplyPostAA();

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

        if(trans >= 1.f) {
            step = -0.02f;
        }

        currentVisibleIndices = currentVisibleIndices < world::ModelIndices.size() ? currentVisibleIndices + speed : 0.f;

    } while(mfb_wait_sync(window));

    return 0;
}
