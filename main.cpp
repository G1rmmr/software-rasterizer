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

    float PanX = 0.f;
    float PanY = 0.f;
    float ZoomRadius = 5.f;

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

void MouseScrollCallback(struct mfb_window* window, mfb_key_mod mod, float deltaX, float deltaY) {
    float ndcX = (State.X / State.Width) * 2.f - 1.f;
    float ndcY = -((State.Y / State.Height) * 2.f - 1.f);

    float zoomSpeed = State.ZoomRadius * 0.1f;

    if(deltaY > 0) {
        State.ZoomRadius -= zoomSpeed;

        State.PanX += ndcX * zoomSpeed * 0.2f;
        State.PanY += ndcY * zoomSpeed * 0.2f;
    }
    else {
        State.ZoomRadius += zoomSpeed;
    }

    if(State.ZoomRadius < 0.1f) State.ZoomRadius = 0.1f;
    if(State.ZoomRadius > 50.f) State.ZoomRadius = 50.f;
}

void SetWorldUniform(mfb_window* window) {
    math::Vector camPos = math::Vector(State.PanX, State.PanY, State.ZoomRadius);
    math::Vector targetPos = math::Vector(State.PanX, State.PanY, 0.f);

    world::Uniform.CameraPos = camPos;
    world::Uniform.View = math::CreateLookAt(camPos, targetPos, world::UP);

    const std::uint8_t* mouseBtn = mfb_get_mouse_button_buffer(window);
    if(State.IsLeftDown) {
        float ndcX = ((State.X / State.Width) * 2.f - 1.f);
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
    mfb_set_mouse_scroll_callback(window, MouseScrollCallback);

    graphics::FrameBuffer frame(world::WIDTH, world::HEIGHT);
    graphics::Rasterizer rasterizer(frame);

    world::Uniform.LightDir = math::Vector(0.f, 0.f, 2.f, 0.f).Norm();

    // world::CreateIcoSphere(2.f, 3);
    // world::CreateDiablo();
    world::CreateAfrican();

    float angle = 0.f;
    do {
        frame.Clear(world::COLOR);
        world::Uniform.Model = math::CreateRotation({0.f, 1.f, 0.f, 0.f}, angle);
        SetWorldUniform(window);

        shader::Model shader;
        shader.Uniform = world::Uniform;

        for(const auto& [_, mesh] : world::SubMeshes) {
            shader.DiffuseMap = mesh.DiffuseMap;
            shader.NormalMap = mesh.NormalMap;
            shader.SpecularMap = mesh.SpecularMap;
            shader.GlossMap = mesh.GlossMap;
            shader.GlowMap = mesh.GlowMap;
            shader.SSSMap = mesh.SSSMap;
            rasterizer.Render(shader, world::ModelVertices, mesh.Indices, State.NowType);
        }

        rasterizer.ApplyPostAA();

        int state = mfb_update(window, frame.GetColor());
        if(state < 0) {
            window = nullptr;
            break;
        }

        angle += 0.001f;

    } while(mfb_wait_sync(window));

    for(const auto& [_, mesh] : world::SubMeshes) {
        delete mesh.DiffuseMap;
        delete mesh.NormalMap;
        delete mesh.SpecularMap;
    }

    return 0;
}
