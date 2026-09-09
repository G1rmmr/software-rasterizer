#pragma once

#include "../Profiler.hpp"
#include "../graphics/FrameBuffer.hpp"
#include "../graphics/RenderSettings.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace app {
    // Supply measurements from the completed application frame. FrameIntervalMs
    // includes presentation and waiting; it is the only source of the displayed FPS.
    struct ProfilerMetrics {
        float FrameIntervalMs = 0.f;
        float CpuWorkMs = 0.f;
        float PresentMs = 0.f;
        float WaitMs = 0.f;
        std::uint32_t ViewportWidth = 0;
        std::uint32_t ViewportHeight = 0;
        float CameraDistance = 0.f;
        std::size_t Workers = 0;
#ifdef NDEBUG
        bool DebugBuild = false;
#else
        bool DebugBuild = true;
#endif
    };

    class ProfilerView final {
    public:
        static constexpr std::uint32_t Width = 560, Height = 384;
        ProfilerView() : frame_(Width, Height) {}
        const graphics::FrameBuffer& Draw(const debug::TimeData& times, const graphics::RenderSettings& settings,
                                          const ProfilerMetrics& metrics = {});

    private:
        static constexpr int GraphX = 16, GraphY = 216, GraphWidth = 528, GraphHeight = 64;
        void Rectangle(int x, int y, int width, int height, std::uint32_t color) noexcept;
        void Text(int x, int y, std::string_view text, std::uint32_t color, int scale = 1) noexcept;
        void Line(int x0, int y0, int x1, int y1, std::uint32_t color) noexcept;
        void DrawHistory(float frameIntervalMs);
        graphics::FrameBuffer frame_;
        std::array<float, GraphWidth> history_{};
        std::size_t cursor_ = 0;
        std::size_t historyCount_ = 0;
    };
}
