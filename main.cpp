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
    math::Vector targetPos = math::Vector(State.PanX, State.PanY, State.ZoomRadius);
    math::Vector camPos = targetPos + math::Vector(0.f, 0.f, State.ZoomRadius);

    world::Uniform.CameraPos = camPos;
    world::Uniform.View = math::CreateLookAt(camPos, targetPos, world::UP);

    const std::uint8_t* mouseBtn = mfb_get_mouse_button_buffer(window);
    if(State.IsLeftDown) {
        float ndcX = ((State.X / State.Width) * 2.f - 1.f);
        float ndcY = -((State.Y / State.Height) * 2.f - 1.f);

        float yaw = ndcX * std::numbers::pi_v<float>;
        float pitch = ndcY * 1.5f;

        math::Vector worldLightDir;
        worldLightDir.X = std::sin(yaw) * std::cos(pitch);
        worldLightDir.Y = std::sin(pitch);
        worldLightDir.Z = std::cos(yaw) * std::cos(pitch);
        worldLightDir.W = 0.f;

        world::Uniform.LightDir = worldLightDir.Norm();
    }

    constexpr float aspect = static_cast<float>(world::WIDTH) / world::HEIGHT;
    world::Uniform.Proj = math::CreatePerspective(math::ToRadian(world::FOV_ANGLE), aspect, world::NEAR, world::FAR);
}

int main() {
    const char* WINDOW_TITLE = "Software Rasterizer";

    struct mfb_window* window = mfb_open_ex(WINDOW_TITLE, world::WIDTH, world::HEIGHT, WF_RESIZABLE);
    if(!window) return -1;

    mfb_set_mouse_button_callback(window, MouseButtonCallback);
    mfb_set_mouse_move_callback(window, MouseMoveCallback);
    mfb_set_resize_callback(window, ResizeCallback);
    mfb_set_keyboard_callback(window, KeyboardCallback);
    mfb_set_mouse_scroll_callback(window, MouseScrollCallback);

    graphics::FrameBuffer frame(world::WIDTH, world::HEIGHT);
    graphics::Rasterizer rasterizer(frame);

    world::Uniform.LightDir = math::Vector(0.f, 0.f, 2.f, 0.f).Norm();

    // world::CreateDiablo();
    world::CreateAfrican();

    math::Matrix diabloTrans = math::CreateTranslation({2.f, 0.f, 0.f, 1.f});
    math::Matrix diabloScale = math::CreateScale({2.f, 2.f, 2.f, 1.f});

    math::Matrix africanTrans = math::CreateTranslation({-2.f, 0.f, 0.f, 1.f});

    shader::Model shader;

    float angle = 0.f;
    do {
        frame.Clear(world::COLOR);
        SetWorldUniform(window);

        world::Uniform.Model = diabloTrans * math::CreateRotation({0.f, 1.f, 0.f, 0.f}, angle) * diabloScale;
        shader.Uniform = world::Uniform;

        for(const graphics::Mesh& mesh : world::SubMeshes["diablo"]) {
            shader.DiffuseMap = mesh.DiffuseMap;
            shader.NormalMap = mesh.NormalMap;
            shader.SpecularMap = mesh.SpecularMap;
            shader.GlossMap = mesh.GlossMap;
            shader.GlowMap = mesh.GlowMap;
            shader.SSSMap = mesh.SSSMap;

            rasterizer.Render(shader, mesh.Vertices, mesh.Indices, State.NowType);
        }

        world::Uniform.Model = math::CreateRotation({0.f, 1.f, 0.f, 0.f}, angle);
        shader.Uniform = world::Uniform;

        for(const graphics::Mesh& mesh : world::SubMeshes["african"]) {
            shader.DiffuseMap = mesh.DiffuseMap;
            shader.NormalMap = mesh.NormalMap;
            shader.SpecularMap = mesh.SpecularMap;
            shader.GlossMap = mesh.GlossMap;
            shader.GlowMap = mesh.GlowMap;
            shader.SSSMap = mesh.SSSMap;

            rasterizer.Render(shader, mesh.Vertices, mesh.Indices, State.NowType);
        }

        // rasterizer.ApplyPostAA();

        int state = mfb_update(window, frame.GetColor());
        if(state < 0) {
            window = nullptr;
            break;
        }

        angle += 0.005f;

    } while(mfb_wait_sync(window));

    for(const graphics::Mesh& mesh : world::SubMeshes["diablo"]) {
        delete mesh.DiffuseMap;
        delete mesh.NormalMap;
        delete mesh.SpecularMap;
        delete mesh.GlossMap;
        delete mesh.GlowMap;
        delete mesh.SSSMap;
    }

    for(const graphics::Mesh& mesh : world::SubMeshes["african"]) {
        delete mesh.DiffuseMap;
        delete mesh.NormalMap;
        delete mesh.SpecularMap;
        delete mesh.GlossMap;
        delete mesh.GlowMap;
        delete mesh.SSSMap;
    }
    return 0;
}
