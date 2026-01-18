#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <MiniFB.h>
#include <MiniFB_enums.h>

#include "graphics/FrameBuffer.hpp"
#include "graphics/Frustrum.hpp"
#include "graphics/Mesh.hpp"
#include "graphics/Rasterizer.hpp"
#include "graphics/Texture.hpp"

#include "math/Math.hpp"

#include "shaders/Elements.hpp"
#include "shaders/Model.hpp"
#include "shaders/Post.hpp"
#include "shaders/Shadow.hpp"

#include "Profiler.hpp"

#include "objects/Object.hpp"

namespace preferences {
    const char* WINDOW_TITLE = "software-rasterizer";

    constexpr inline std::uint32_t COLOR = 0xFF000000;

    constexpr inline std::uint32_t WIDTH = 1280;
    constexpr inline std::uint32_t HEIGHT = 720;

    constexpr inline float FOV_ANGLE = 45.f;
    constexpr inline float NEAR = 0.1f;
    constexpr inline float FAR = 100.f;

    const inline math::Vector UP(0.f, 1.f, 0.f);
    const inline math::Vector EYE(0.f, 0.f, 5.f);

    struct AppState {
        float Width = static_cast<float>(WIDTH);
        float Height = static_cast<float>(HEIGHT);

        float X = 0.f;
        float Y = 0.f;
        float CamDistance = 45.f;

        graphics::PrimitiveType NowType = graphics::PrimitiveType::Triangles;

        bool IsLeftDown = false;
        bool IsShowingShadowMap = true;
        bool IsShowingSSAO = true;
        bool IsShowingAA = true;
    } State;

    inline std::vector<std::shared_ptr<Object>> Objects;
    inline shader::Uniforms Uniform;
    inline math::Vector Target(0.f, 0.f, 0.f);
    inline graphics::FrameBuffer* CurrFrame = nullptr;
    inline graphics::Frustum CameraFrustum;

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

            case KB_KEY_Q:
                if(isPressed) State.IsShowingShadowMap = !State.IsShowingShadowMap;
                break;

            case KB_KEY_W:
                if(isPressed) State.IsShowingSSAO = !State.IsShowingSSAO;
                break;

            case KB_KEY_E:
                if(isPressed) State.IsShowingAA = !State.IsShowingAA;
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
        return window;
    }

    inline void UpdateUniform(mfb_window* window) {
        static const math::Vector viewDir = math::Vector(0.f, 0.5f, 1.f).Norm();

        math::Vector camPos = viewDir * State.CamDistance;
        math::Vector targetPos = math::Vector(0.f, 2.f, -5.f);

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
        Uniform.InvProj = Uniform.Proj.Inv();

        math::Matrix vp = Uniform.Proj * Uniform.View;
        CameraFrustum.Update(vp);
    }

    namespace debug {
        constexpr inline std::uint32_t WIDTH = 400;
        constexpr inline std::uint32_t HEIGHT = 300;
        constexpr inline std::int32_t HISTORY_COUNT = 100;

        inline std::vector<std::uint32_t> Buffer;

        inline std::vector<float> FrameHistory(HISTORY_COUNT, 0.f);
        inline std::int32_t HistoryIdx = 0;

        inline void PushHistory(const float ms) {
            FrameHistory[HistoryIdx] = ms;
            HistoryIdx = (HistoryIdx + 1) % HISTORY_COUNT;
        }

        inline mfb_window* Init() {
            struct mfb_window* window = mfb_open_ex("profiler", WIDTH, HEIGHT, WF_RESIZABLE);
            Buffer.resize(WIDTH * HEIGHT, 0xFF000000);
            return window;
        }

        inline void Graph(std::int32_t x0, std::int32_t y0, std::int32_t x1, std::int32_t y1, std::uint32_t color) {
            std::int32_t dx = std::abs(x1 - x0);
            std::int32_t sx = x0 < x1 ? 1 : -1;

            std::int32_t dy = -std::abs(y1 - y0);
            std::int32_t sy = y0 < y1 ? 1 : -1;

            std::int32_t err = dx + dy, e2;

            while(true) {
                if(x0 >= 0 && x0 < WIDTH && y0 >= 0 && y0 < HEIGHT) Buffer[y0 * WIDTH + x0] = color;
                if(x0 == x1 && y0 == y1) break;
                e2 = 2 * err;
                if(e2 >= dy) {
                    err += dy;
                    x0 += sx;
                }
                if(e2 <= dx) {
                    err += dx;
                    y0 += sy;
                }
            }
        }

        inline void Draw(mfb_window* window, bool shouldRender = true) {
            if(!shouldRender) return;
            std::fill(Buffer.begin(), Buffer.end(), 0xFF111111);
            PushHistory(::debug::Profiler.TotalFrameTime);

            auto DrawIndicator = [&](std::int32_t x, std::int32_t y, bool isOn, std::uint32_t color) {
                std::uint32_t finalColor = isOn ? color : (color & 0xFF333333);
                for(std::int32_t i = 0; i < 10; ++i)
                    for(std::int32_t j = 0; j < 10; ++j) Buffer[(y + i) * WIDTH + (x + j)] = finalColor;
            };

            DrawIndicator(10, 10, State.IsShowingShadowMap, 0xFFFFCC00);
            DrawIndicator(30, 10, State.IsShowingSSAO, 0xFFFF00FF);
            DrawIndicator(50, 10, State.IsShowingAA, 0xFF00FF00);

            float maxMs = 40.f;
            float xStep = static_cast<float>(WIDTH) / HISTORY_COUNT;

            auto DrawHorizontalLine = [&](float ms, std::int32_t color) {
                std::int32_t y = HEIGHT - static_cast<std::int32_t>((ms / maxMs) * HEIGHT);
                if(y < 0 || y >= HEIGHT) return;
                for(std::int32_t x = 0; x < WIDTH; ++x) Buffer[y * WIDTH + x] = color;
            };

            DrawHorizontalLine(16.6f, 0xFF555555);
            DrawHorizontalLine(33.3f, 0xFF882222);

            for(std::int32_t i = 0; i < HISTORY_COUNT; i += 10) {
                std::int32_t x = static_cast<int>(i * xStep);
                for(std::int32_t y = 0; y < HEIGHT; ++y)
                    if(y % 4 == 0) Buffer[y * WIDTH + x] = 0xFF333333;
            }

            for(std::int32_t i = 0; i < HISTORY_COUNT - 1; ++i) {
                std::int32_t idx1 = (HistoryIdx + i) % HISTORY_COUNT;
                std::int32_t idx2 = (HistoryIdx + i + 1) % HISTORY_COUNT;

                std::int32_t y0 =
                    HEIGHT - static_cast<std::int32_t>((std::min(FrameHistory[idx1], maxMs) / maxMs) * HEIGHT);

                std::int32_t y1 =
                    HEIGHT - static_cast<std::int32_t>((std::min(FrameHistory[idx2], maxMs) / maxMs) * HEIGHT);

                uint32_t color = (FrameHistory[idx2] > 16.6f) ? 0xFFFF0000 : 0xFF00FF00;
                Graph(static_cast<std::int32_t>(i * xStep), y0, static_cast<std::int32_t>((i + 1) * xStep), y1, color);
            }

            mfb_update(window, Buffer.data());
        }
    }

    namespace pre {
        constexpr inline std::uint32_t SHADOW_WIDTH = 512;
        constexpr inline std::uint32_t SHADOW_HEIGHT = 512;

        inline std::future<void> StaticShadowTask;

        inline graphics::FrameBuffer Frame(SHADOW_WIDTH, SHADOW_HEIGHT);
        inline graphics::FrameBuffer StaticFrame(SHADOW_WIDTH, SHADOW_HEIGHT);
        inline graphics::Rasterizer Rasterizer(Frame);
        inline shader::Shadow Shader;
        inline graphics::Frustum LightFrustum;

        inline void UpdateLightFrustum() {
            LightFrustum.Update(Uniform.LightSpace);
        }

        inline bool IsStaticShadowDirty = true;

        inline std::shared_future<void> Render() {
            if(!State.IsShowingShadowMap) return {};

            math::Vector lightDir = Uniform.LightDir.Norm();
            math::Vector lightPos = lightDir * 100.f;
            math::Vector lightTarget = {0.f, 0.f, 0.f};

            math::Vector lightUp = {0.f, 1.f, 0.f};
            if(std::abs(lightDir.Y) > 0.99f) lightUp = {0.f, 0.f, 1.f};

            math::Matrix lightView = math::CreateLookAt(lightPos, lightTarget, lightUp);

            const float orthoSize = 15.f;
            math::Matrix lightProj = math::CreateOrtho(-orthoSize, orthoSize, -orthoSize, orthoSize, -200.f, 200.f);

            Uniform.LightSpace = lightProj * lightView;

            Shader.Uniform.LightSpace = Uniform.LightSpace;
            Shader.Uniform.View = lightView;
            Shader.Uniform.Proj = lightProj;

            LightFrustum.Update(Uniform.LightSpace);

            if(IsStaticShadowDirty) {
                StaticFrame.Clear(0xFFFFFFFF);
                for(const std::shared_ptr<Object>& object : Objects) {
                    if(!object->IsStatic) continue;

                    auto [worldCenter, worldRadius] = object->GetWorldBounds();
                    if(!LightFrustum.IsSphereInside(worldCenter, worldRadius)) continue;

                    Shader.Uniform.Model = object->Model;
                    for(const graphics::Mesh& mesh : object->Meshes)
                        Rasterizer.Render(Shader, mesh.Vertices, mesh.Indices, NEAR).get();
                }
                IsStaticShadowDirty = false;
            }

            Frame.Clear(0xFFFFFFFF);

            std::shared_future<void> lastTask;
            for(const std::shared_ptr<Object> object : Objects) {
                if(!object->ShouldRender) continue;

                Shader.Uniform.Model = object->Model;
                for(const graphics::Mesh& mesh : object->Meshes)
                    lastTask = Rasterizer.Render(Shader, mesh.Vertices, mesh.Indices, NEAR,
                                                 graphics::PrimitiveType::Triangles);
            }

            CurrFrame = &Frame;
            return lastTask;
        }
    }

    namespace main {
        inline graphics::FrameBuffer Frame(WIDTH, HEIGHT);
        inline graphics::Rasterizer Rasterizer(Frame);
        inline shader::Model Shader;

        inline std::shared_future<void> Render(std::shared_future<void> shadowDependency) {
            Shader.ShadowMap = nullptr;

            if(State.IsShowingShadowMap) {
                Shader.ShadowMap = &pre::Frame.GetDepthes();
                Shader.ShadowMapWidth = static_cast<float>(pre::SHADOW_WIDTH);
                Shader.ShadowMapHeight = static_cast<float>(pre::SHADOW_HEIGHT);
            }

            Frame.Clear(preferences::COLOR);

            std::shared_future<void> lastTask;
            for(const std::shared_ptr<Object> object : Objects) {
                if(!object->ShouldRender) continue;

                auto [worldCenter, worldRadius] = object->GetWorldBounds();

                if(!CameraFrustum.IsSphereInside(worldCenter, worldRadius)) continue;

                Uniform.Model = object->Model;
                Uniform.DepthBias = 0.f;
                Shader.Uniform = Uniform;

                for(const graphics::Mesh& mesh : object->Meshes) {
                    Shader.DiffuseMap = mesh.DiffuseMap;
                    Shader.NormalMap = mesh.NormalMap;
                    Shader.SpecularMap = mesh.SpecularMap;
                    Shader.GlossMap = mesh.GlossMap;
                    Shader.GlowMap = mesh.GlowMap;
                    Shader.SSSMap = mesh.SSSMap;

                    lastTask = Rasterizer.Render(Shader, mesh.Vertices, mesh.Indices, NEAR, preferences::State.NowType,
                                                 shadowDependency);
                }
            }

            CurrFrame = &Frame;
            return lastTask;
        }
    }

    namespace post {
        inline graphics::FrameBuffer Frame(WIDTH, HEIGHT);
        inline graphics::Rasterizer Rasterizer(Frame);
        inline shader::Post Shader;

        inline void Render(std::shared_future<void> mainDependency) {
            if(mainDependency.valid()) mainDependency.get();
            if(!State.IsShowingSSAO) return;

            Shader.Uniform = Uniform;

            Shader.Uniform.ScreenWidth = static_cast<float>(WIDTH);
            Shader.Uniform.ScreenHeight = static_cast<float>(HEIGHT);

            Shader.DepthMap = &main::Frame.GetDepthes();
            Shader.NormalMap = &main::Frame.GetNormals();
            Shader.ColorMap = &main::Frame.GetColors();

            const std::int32_t w = static_cast<std::int32_t>(WIDTH * 0.5f);
            const std::int32_t h = static_cast<std::int32_t>(HEIGHT * 0.5f);
            const std::size_t bufferSize = w * h;

            if(Shader.AOBuffer.size() != bufferSize) {
                Shader.AOBuffer.resize(bufferSize);
                Shader.TempBuffer.resize(bufferSize);
            }

            if(Shader.Uniform.KernelSamples.empty() ||
               Shader.Uniform.KernelSamples.size() != Shader.Uniform.KernelSizeAO)
                Shader.GenerateKernel(Shader.Uniform);

            ParallelExecutor::GetInstance().ParallelFor(
                0, h,
                [&](std::int32_t y) {
                    for(std::int32_t x = 0; x < w; ++x) Shader.AOBuffer[y * w + x] = Shader.ComputeRawAO(x, y);
                },
                64);

            ParallelExecutor::GetInstance().ParallelFor(0, h, [&](std::int32_t y) { Shader.ProcessBlur(y); }, 64);

            ParallelExecutor::GetInstance().ParallelFor(
                0, static_cast<std::int32_t>(HEIGHT), [&](std::int32_t y) { Shader.Composite(y); }, 64);

            CurrFrame = &main::Frame;
        }
    }
}