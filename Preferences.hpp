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
    const char* WINDOW_TITLE = "software-rasterizer";
    constexpr inline std::uint32_t COLOR = 0xFF000000;

    constexpr inline std::uint32_t WIDTH = 1280;
    constexpr inline std::uint32_t HEIGHT = 720;

    constexpr inline std::uint32_t SHADOW_WIDTH = 1024;
    constexpr inline std::uint32_t SHADOW_HEIGHT = 1024;

    constexpr inline float FOV_ANGLE = 45.f;
    constexpr inline float NEAR = 0.1f;
    constexpr inline float FAR = 500.f;

    const inline math::Vector UP(0.f, 1.f, 0.f);
    const inline math::Vector EYE(0.f, 0.f, 5.f);

    struct AppState {
        float Width = static_cast<float>(WIDTH);
        float Height = static_cast<float>(HEIGHT);

        float X = 0.f;
        float Y = 0.f;
        float CamDistance = 10.f;

        graphics::PrimitiveType NowType = graphics::PrimitiveType::Triangles;

        bool IsLeftDown = false;
        bool IsShowingShadowMap = false;
    } State;

    inline std::vector<std::shared_ptr<Object>> Objects;

    inline graphics::FrameBuffer Frame(WIDTH, HEIGHT);
    inline graphics::Rasterizer Rasterizer(Frame);
    inline shader::Model Shader;

    inline graphics::FrameBuffer ShadowFrame(SHADOW_WIDTH, SHADOW_HEIGHT);
    inline graphics::Rasterizer ShadowRasterizer(ShadowFrame);
    inline shader::Shadow ShadowShader;

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

            case KB_KEY_BACKSPACE:
                if(isPressed) State.IsShowingShadowMap = !State.IsShowingShadowMap;
                break;

            case KB_KEY_ESCAPE:
                if(isPressed) mfb_close(window);
                break;

            default: break;
            }
        }

        inline void MouseScrollCallback(struct mfb_window* window, mfb_key_mod mod, float deltaX, float deltaY) {
            float zoomSpeed = 2.f;

            if(deltaY > 0)
                State.CamDistance -= zoomSpeed;
            else
                State.CamDistance += zoomSpeed;

            if(State.CamDistance < 1.f) State.CamDistance = 1.f;
            if(State.CamDistance > 200.f) State.CamDistance = 200.f;
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

        Uniform.LightDir = math::Vector(-0.5f, 1.f, 1.f, 0.f).Norm();

        ShadowFrame.Clear(0xFFFFFFFF);
        Shader.ShadowMap = &ShadowFrame.GetDepthes();
        Shader.ShadowMapWidth = (float)SHADOW_WIDTH;
        Shader.ShadowMapHeight = (float)SHADOW_HEIGHT;

        return window;
    }

    inline void UpdateUniform(mfb_window* window) {
        static const math::Vector viewDir = math::Vector(0.f, 0.5f, 1.f).Norm();

        math::Vector camPos = viewDir * State.CamDistance;
        math::Vector targetPos = math::Vector(0.f, 0.f, -5.f);

        Uniform.CameraPos = camPos;
        Uniform.View = math::CreateLookAt(camPos, targetPos, UP);

        const std::uint8_t* mouseBtn = mfb_get_mouse_button_buffer(window);
        if(State.IsLeftDown) {
            float ndcX = ((State.X / State.Width) * 2.f - 1.f);
            float yaw = ndcX * std::numbers::pi_v<float>;

            math::Vector worldLightDir;
            worldLightDir.X = std::sin(yaw);
            worldLightDir.Y = 1.f;
            worldLightDir.Z = std::cos(yaw);
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

                Rasterizer.Render(Shader, mesh.Vertices, mesh.Indices, NEAR, preferences::State.NowType);
            }
        }
    }

    inline void MapShadow() {
        math::Vector lightDir = Uniform.LightDir.Norm();
        math::Vector lightPos = lightDir * 100.f;
        math::Vector lightTarget = {0.f, 0.f, 0.f};

        math::Vector lightUp = {0.f, 1.f, 0.f};
        if(std::abs(lightDir.Y) > 0.99f) lightUp = {0.f, 0.f, 1.f};

        math::Matrix lightView = math::CreateLookAt(lightPos, lightTarget, lightUp);

        const float orthoSize = 15.0f;
        math::Matrix lightProj = math::CreateOrtho(-orthoSize, orthoSize, -orthoSize, orthoSize, -200.f, 200.f);
        math::Matrix lightSpaceMatrix = lightProj * lightView;

        ShadowShader.Uniform.LightSpace = lightSpaceMatrix;
        ShadowShader.Uniform.View = lightView;
        ShadowShader.Uniform.Proj = lightProj;

        ShadowFrame.Clear(0xFFFFFFFF);
        for(const std::shared_ptr<Object> object : Objects) {
            if(!object->ShouldRender) continue;

            ShadowShader.Uniform.Model = object->Model;
            for(const graphics::Mesh& mesh : object->Meshes) {
                ShadowRasterizer.Render(ShadowShader, mesh.Vertices, mesh.Indices, NEAR,
                                        graphics::PrimitiveType::Triangles);
            }
        }
        Uniform.LightSpace = lightSpaceMatrix;
    }
}