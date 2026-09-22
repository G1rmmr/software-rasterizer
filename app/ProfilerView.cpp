#include "ProfilerView.hpp"
#include "BitmapFont.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

namespace app {
    namespace {
        constexpr std::uint32_t Background = 0xff101722u;
        constexpr std::uint32_t Panel = 0xff182332u;
        constexpr std::uint32_t TextColor = 0xffedf3fau;
        constexpr std::uint32_t Muted = 0xff9caec3u;
        constexpr std::uint32_t Grid = 0xff344254u;
        constexpr std::uint32_t Good = 0xff68d9a4u;
        constexpr std::uint32_t Warning = 0xffffc56bu;
        constexpr float FrameBudgetMs = 1000.f / 60.f;
        constexpr float GraphMaximumMs = 50.f;

        bool NonnegativeFinite(float value) noexcept {
            return std::isfinite(value) && value >= 0.f;
        }

        void Milliseconds(char* buffer, std::size_t size, float value, bool measured = true) noexcept {
            if(!measured || !NonnegativeFinite(value))
                std::snprintf(buffer, size, "--.-- MS");
            else if(value >= 1000.f)
                std::snprintf(buffer, size, "999+ MS");
            else
                std::snprintf(buffer, size, "%.2f MS", static_cast<double>(value));
        }

        void Dimension(char* buffer, std::size_t size, std::uint32_t value) noexcept {
            if(value == 0)
                std::snprintf(buffer, size, "--");
            else if(value > 99999u)
                std::snprintf(buffer, size, "99999+");
            else
                std::snprintf(buffer, size, "%u", value);
        }
    }

    void ProfilerView::Rectangle(int x, int y, int width, int height, std::uint32_t color) noexcept {
        auto pixels = frame_.Colors();
        const int left = std::max(0, x), right = std::min(static_cast<int>(Width), x + width);
        const int top = std::max(0, y), bottom = std::min(static_cast<int>(Height), y + height);
        if(left >= right || top >= bottom) return;
        for(int row = top; row < bottom; ++row)
            std::fill_n(pixels.data() + static_cast<std::size_t>(row) * Width + left, right - left, color);
    }

    void ProfilerView::Text(int x, int y, std::string_view text, std::uint32_t color, int scale) noexcept {
        BitmapFont::Draw(frame_, x, y, text, color, scale);
    }

    void ProfilerView::Line(int x0, int y0, int x1, int y1, std::uint32_t color) noexcept {
        const int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        const int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int error = dx + dy;
        for(;;) {
            if(x0 >= 0 && x0 < static_cast<int>(Width) && y0 >= 0 && y0 < static_cast<int>(Height))
                frame_.SetPixel(static_cast<std::uint32_t>(x0), static_cast<std::uint32_t>(y0), color);
            if(x0 == x1 && y0 == y1) break;
            const int twice = 2 * error;
            if(twice >= dy) {
                error += dy;
                x0 += sx;
            }
            if(twice <= dx) {
                error += dx;
                y0 += sy;
            }
        }
    }

    void ProfilerView::DrawHistory(float frameIntervalMs) {
        if(NonnegativeFinite(frameIntervalMs) && frameIntervalMs > 0.f) {
            history_[cursor_] = frameIntervalMs;
            cursor_ = (cursor_ + 1u) % history_.size();
            historyCount_ = std::min(historyCount_ + 1u, history_.size());
        }
        Text(GraphX, 201, "FRAME / 0-50 MS", Muted, 2);
        constexpr std::string_view legend = "TARGET 16.67 MS";
        Text(static_cast<int>(Width) - 16 - BitmapFont::TextWidth(legend, 2), 201, legend, Good, 2);
        Rectangle(GraphX, GraphY, GraphWidth, GraphHeight, Panel);
        const auto graphY = [](float time) {
            const float fraction = std::clamp(time / GraphMaximumMs, 0.f, 1.f);
            return GraphY + GraphHeight - 1 - static_cast<int>(fraction * (GraphHeight - 1));
        };
        const int targetY = graphY(FrameBudgetMs);
        for(int x = GraphX; x < GraphX + GraphWidth; x += 4) Rectangle(x, targetY, 2, 1, Grid);
        const auto oldest = (cursor_ + history_.size() - historyCount_) % history_.size();
        const int firstX = GraphX + GraphWidth - static_cast<int>(historyCount_);
        int previousX = firstX, previousY = GraphY + GraphHeight - 1;
        for(std::size_t i = 0; i < historyCount_; ++i) {
            const float value = history_[(oldest + i) % history_.size()];
            const int x = firstX + static_cast<int>(i), y = graphY(value);
            const auto color = value <= FrameBudgetMs ? Good : Warning;
            if(i == 0)
                Rectangle(x, y, 1, 1, color);
            else
                Line(previousX, previousY, x, y, color);
            previousX = x;
            previousY = y;
        }
    }

    const graphics::FrameBuffer& ProfilerView::Draw(const debug::TimeData& times,
                                                    const graphics::RenderSettings& settings,
                                                    const ProfilerMetrics& metrics) {
        // This window only displays pixels. Keep its untouched depth/normal attachments out of the hot path.
        std::fill(frame_.Colors().begin(), frame_.Colors().end(), Background);
        Text(16, 14, "PERFORMANCE", TextColor, 2);
        const std::string_view build = metrics.DebugBuild ? "DEBUG" : "RELEASE";
        Text(static_cast<int>(Width) - 16 - BitmapFont::TextWidth(build, 2), 14, build,
             metrics.DebugBuild ? Warning : Good, 2);

        Rectangle(10, 38, 540, 48, Panel);
        Text(16, 42, "ACTUAL FPS", Muted, 2);
        Text(196, 42, "FRAME", Muted, 2);
        Text(376, 42, "CPU TOTAL", Muted, 2);
        char text[96];
        const bool measuredFrame = std::isfinite(metrics.FrameIntervalMs) && metrics.FrameIntervalMs > 0.f;
        if(!measuredFrame)
            std::snprintf(text, sizeof(text), "--.-");
        else {
            const float fps = 1000.f / metrics.FrameIntervalMs;
            if(fps > 999.9f)
                std::snprintf(text, sizeof(text), "999+");
            else
                std::snprintf(text, sizeof(text), "%.1f", static_cast<double>(fps));
        }
        const auto frameColor = !measuredFrame ? Muted : metrics.FrameIntervalMs <= FrameBudgetMs ? Good : Warning;
        Text(16, 58, text, frameColor, 3);
        Milliseconds(text, sizeof(text), metrics.FrameIntervalMs, measuredFrame);
        Text(196, 60, text, TextColor, 2);
        Milliseconds(text, sizeof(text), metrics.CpuWorkMs, metrics.CpuWorkMs > 0.f);
        Text(376, 60, text, metrics.CpuWorkMs > FrameBudgetMs ? Warning : TextColor, 2);

        Text(16, 91, "PASS / CPU MS", Muted);
        constexpr std::string_view limit = "BAR LIMIT 16.67 MS";
        Text(static_cast<int>(Width) - 16 - BitmapFont::TextWidth(limit), 91, limit, Muted);
        const std::array<float, 4> passes = {times.ShadowPassTime, times.MainPassTime, times.PostPassTime,
                                             times.AAPassTime};
        const std::array<std::string_view, 4> labels = {"SHADOW", "GEOMETRY", "SSAO", "AA"};
        const std::array<std::uint32_t, 4> colors = {0xffdcb56cu, 0xff7ab3ffu, 0xffbea0ffu, 0xff71d5c3u};
        for(std::size_t i = 0; i < passes.size(); ++i) {
            const int y = 103 + static_cast<int>(i) * 20;
            Text(16, y, labels[i], colors[i], 2);
            Milliseconds(text, sizeof(text), passes[i]);
            Text(130, y, text, TextColor, 2);
            Rectangle(262, y + 3, 282, 8, Grid);
            const float share = NonnegativeFinite(passes[i]) ? std::clamp(passes[i] / FrameBudgetMs, 0.f, 1.f) : 0.f;
            Rectangle(262, y + 3, static_cast<int>(282.f * share), 8, colors[i]);
            if(NonnegativeFinite(passes[i]) && passes[i] > FrameBudgetMs) Rectangle(542, y + 1, 2, 12, Warning);
        }

        Text(16, 183, "PRESENT", Muted, 2);
        Milliseconds(text, sizeof(text), metrics.PresentMs);
        Text(112, 183, text, TextColor, 2);
        Text(302, 183, "WAIT", Muted, 2);
        Milliseconds(text, sizeof(text), metrics.WaitMs);
        Text(362, 183, text, TextColor, 2);
        DrawHistory(metrics.FrameIntervalMs);

        char width[16], height[16], camera[16], workers[16];
        Dimension(width, sizeof(width), metrics.ViewportWidth);
        Dimension(height, sizeof(height), metrics.ViewportHeight);
        if(!std::isfinite(metrics.CameraDistance) || metrics.CameraDistance <= 0.f)
            std::snprintf(camera, sizeof(camera), "--.-");
        else if(metrics.CameraDistance >= 10000.f)
            std::snprintf(camera, sizeof(camera), "9999+");
        else
            std::snprintf(camera, sizeof(camera), "%.1f", static_cast<double>(metrics.CameraDistance));
        if(metrics.Workers > 999u)
            std::snprintf(workers, sizeof(workers), "999+");
        else
            std::snprintf(workers, sizeof(workers), "%zu", metrics.Workers);
        std::snprintf(text, sizeof(text), "%sX%s  CAMERA %s  WORKERS %s", width, height, camera, workers);
        Text(16, 292, text, TextColor, 2);

        const std::array<bool, 4> enabled = {settings.Shadows, settings.AmbientOcclusion, settings.AntiAliasing,
                                             settings.Toon};
        const std::array<std::string_view, 4> keys = {"Q", "W", "E", "R"};
        const std::array<std::string_view, 4> effects = {"SHADOW", "SSAO", "AA", "TOON"};
        for(std::size_t i = 0; i < enabled.size(); ++i) {
            const int x = 16 + static_cast<int>(i) * 134;
            Rectangle(x, 322, 126, 40, Panel);
            Text(x + 6, 326, keys[i], TextColor, 2);
            Text(x + 24, 326, effects[i], Muted, 2);
            const std::string_view state = enabled[i] ? "ON" : "OFF";
            Text(x + 118 - BitmapFont::TextWidth(state, 2), 346, state, enabled[i] ? Good : Muted, 2);
        }
        Text(16, 373, "SPACE: PRIMITIVE  DRAG: LIGHT WHEEL: ZOOM  ESC: QUIT", Muted);
        return frame_;
    }
}
