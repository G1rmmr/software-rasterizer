#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <MiniFB.h>
#include <MiniFB_enums.h>

#include "graphics/FrameBuffer.hpp"
#include "graphics/Mesh.hpp"
#include "graphics/Rasterizer.hpp"
#include "graphics/Shader.hpp"
#include "graphics/Texture.hpp"

#include "math/Math.hpp"

#include "objects/Object.hpp"

namespace preferences {
    constexpr inline std::uint32_t WIDTH = 800;
    constexpr inline std::uint32_t HEIGHT = 450;
    constexpr inline std::uint32_t COLOR = 0xFF000000;

    constexpr inline float FOV_ANGLE = 45.f;
    constexpr inline float NEAR = 0.1f;
    constexpr inline float FAR = 100.f;

    const inline math::Vector UP(0.f, 1.f, 0.f);
    const inline math::Vector EYE(0.f, 0.f, 5.f);
    const char* WINDOW_TITLE = "software-rasterizer";

    struct AppState {
        float Width = static_cast<float>(WIDTH);
        float Height = static_cast<float>(HEIGHT);

        float X = 0.f;
        float Y = 0.f;

        float PanX = 0.f;
        float PanY = 0.f;
        float ZoomRadius = 5.f;

        graphics::PrimitiveType NowType = graphics::PrimitiveType::Triangles;
        bool IsLeftDown = false;
    } State;

    inline std::vector<std::shared_ptr<Object>> Objects;

    inline graphics::FrameBuffer Frame(WIDTH, HEIGHT);
    inline graphics::Rasterizer Rasterizer(Frame);
    inline shader::Model Shader;

    inline shader::Uniforms Uniform;
    inline math::Vector Target(0.f, 0.f, 0.f);

    namespace {
        inline void MouseButtonCallback(struct mfb_window* window, mfb_mouse_button button, mfb_key_mod mod,
                                        bool isPressed) {
            State.IsLeftDown = button == MOUSE_BTN_1 && isPressed;
        }

        inline void MouseMoveCallback(struct mfb_window* window, int x, int y) {
            State.X = static_cast<float>(x);
            State.Y = static_cast<float>(y);
        }

        inline void ResizeCallback(struct mfb_window* window, int width, int height) {
            State.Width = static_cast<float>(width);
            State.Height = static_cast<float>(height);
        }

        inline void KeyboardCallback(struct mfb_window* window, mfb_key key, mfb_key_mod mod, bool isPressed) {
            float zoomSpeed = State.ZoomRadius * 0.1f;
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

            case KB_KEY_UP:
                if(isPressed) {
                    State.ZoomRadius -= zoomSpeed;
                    if(State.ZoomRadius < 0.1f) State.ZoomRadius = 0.1f;
                }
                break;

            case KB_KEY_DOWN:
                if(isPressed) {
                    State.ZoomRadius += zoomSpeed;
                    if(State.ZoomRadius > 50.f) State.ZoomRadius = 50.f;
                }
                break;

            case KB_KEY_ESCAPE:
                if(isPressed) mfb_close(window);
                break;

            default: break;
            }
        }

        inline void MouseScrollCallback(struct mfb_window* window, mfb_key_mod mod, float deltaX, float deltaY) {
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

    }

    inline mfb_window* Init() {
        struct mfb_window* window = mfb_open_ex(WINDOW_TITLE, WIDTH, HEIGHT, WF_RESIZABLE);
        if(!window) return nullptr;

        mfb_set_mouse_button_callback(window, MouseButtonCallback);
        mfb_set_mouse_move_callback(window, MouseMoveCallback);
        mfb_set_resize_callback(window, ResizeCallback);
        mfb_set_keyboard_callback(window, KeyboardCallback);
        mfb_set_mouse_scroll_callback(window, MouseScrollCallback);

        Uniform.LightDir = math::Vector(0.f, 0.f, 2.f, 0.f).Norm();
        return window;
    }

    inline void UpdateUniform(mfb_window* window) {
        math::Vector targetPos = math::Vector(State.PanX, State.PanY, State.ZoomRadius);
        math::Vector camPos = targetPos + math::Vector(0.f, 0.f, State.ZoomRadius);

        Uniform.CameraPos = camPos;
        Uniform.View = math::CreateLookAt(camPos, targetPos, UP);

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

            Uniform.LightDir = worldLightDir.Norm();
        }

        constexpr float aspect = static_cast<float>(WIDTH) / HEIGHT;
        Uniform.Proj = math::CreatePerspective(math::ToRadian(FOV_ANGLE), aspect, NEAR, FAR);
    }

    inline void Render() {
        for(const std::shared_ptr<Object> object : Objects) {
            if(!object->ShouldRender) continue;
            Uniform.Model = object->Model;
            Shader.Uniform = Uniform;

            for(const graphics::Mesh& mesh : object->Meshes) {
                Shader.DiffuseMap = mesh.DiffuseMap;
                Shader.NormalMap = mesh.NormalMap;
                Shader.SpecularMap = mesh.SpecularMap;
                Shader.GlossMap = mesh.GlossMap;
                Shader.GlowMap = mesh.GlowMap;
                Shader.SSSMap = mesh.SSSMap;

                Rasterizer.Render(Shader, mesh.Vertices, mesh.Indices, preferences::State.NowType);
            }
        }
    }
}