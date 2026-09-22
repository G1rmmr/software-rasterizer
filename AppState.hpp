#pragma once
#include "graphics/RenderSettings.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace app {
    enum class Action { CyclePrimitive, ToggleShadows, ToggleSsao, ToggleAA, ToggleToon, Quit };

    // UI state belongs to one application. Callbacks express actions, never edit render resources.
    class AppState final {
    public:
        AppState(std::uint32_t width, std::uint32_t height, graphics::RenderSettings settings = {})
            : width_(width), height_(height), settings_(settings) {
            if(!width || !height) throw std::invalid_argument("Viewport dimensions must be positive");
        }
        void Apply(Action action) noexcept {
            switch(action) {
            case Action::CyclePrimitive:
                settings_.Primitive =
                    settings_.Primitive == graphics::PrimitiveType::Triangles ? graphics::PrimitiveType::Lines
                    : settings_.Primitive == graphics::PrimitiveType::Lines   ? graphics::PrimitiveType::Points
                                                                              : graphics::PrimitiveType::Triangles;
                break;
            case Action::ToggleShadows: settings_.Shadows = !settings_.Shadows; break;
            case Action::ToggleSsao: settings_.AmbientOcclusion = !settings_.AmbientOcclusion; break;
            case Action::ToggleAA: settings_.AntiAliasing = !settings_.AntiAliasing; break;
            case Action::ToggleToon: settings_.Toon = !settings_.Toon; break;
            case Action::Quit: running_ = false; break;
            }
        }
        void Resize(int width, int height) noexcept {
            if(width > 0 && height > 0) {
                width_ = static_cast<std::uint32_t>(width);
                height_ = static_cast<std::uint32_t>(height);
            }
        }
        void SetDragging(bool dragging) noexcept { dragging_ = dragging; }
        void MovePointer(int x, int y) noexcept {
            if(!dragging_) return;
            lightDirection_ = math::Vector(2.f * static_cast<float>(x) / static_cast<float>(width_) - 1.f,
                                           1.f - 2.f * static_cast<float>(y) / static_cast<float>(height_), 1.f, 0.f)
                                  .NormalizedDirection();
        }
        void Scroll(float delta) noexcept {
            if(std::isfinite(delta)) cameraDistance_ = std::clamp(cameraDistance_ - 2.f * delta, 1.f, 200.f);
        }
        void SetCameraDistance(float distance) {
            if(!std::isfinite(distance) || distance < 1.f || distance > 200.f)
                throw std::invalid_argument("Camera distance must be finite and within 1..200");
            cameraDistance_ = distance;
        }
        [[nodiscard]] std::uint32_t GetWidth() const noexcept { return width_; }
        [[nodiscard]] std::uint32_t GetHeight() const noexcept { return height_; }
        [[nodiscard]] float GetCameraDistance() const noexcept { return cameraDistance_; }
        [[nodiscard]] bool IsRunning() const noexcept { return running_; }
        [[nodiscard]] const math::Vector& GetLightDirection() const noexcept { return lightDirection_; }
        [[nodiscard]] const graphics::RenderSettings& GetRenderSettings() const noexcept { return settings_; }

    private:
        std::uint32_t width_, height_;
        graphics::RenderSettings settings_;
        float cameraDistance_ = 45.f;
        math::Vector lightDirection_ = math::Vector(1.f, 1.f, 1.f).NormalizedDirection();
        bool dragging_ = false;
        bool running_ = true;
    };
}
