#include "../app/BitmapFont.hpp"
#include "../app/ImageWriter.hpp"
#include "../app/ProfilerView.hpp"
#include "TestSupport.hpp"
#include <algorithm>
#include <limits>
#include <vector>

namespace {
    app::ProfilerMetrics SampleMetrics() {
        app::ProfilerMetrics metrics;
        metrics.FrameIntervalMs = 16.89f;
        metrics.CpuWorkMs = 11.32f;
        metrics.PresentMs = .87f;
        metrics.WaitMs = 4.70f;
        metrics.ViewportWidth = 1280;
        metrics.ViewportHeight = 720;
        metrics.CameraDistance = 12.f;
        metrics.Workers = 23;
        metrics.DebugBuild = false;
        return metrics;
    }

    debug::TimeData SampleTimes() {
        debug::TimeData times;
        times.ShadowPassTime = 2.40f;
        times.MainPassTime = 4.15f;
        times.PostPassTime = 2.95f;
        times.AAPassTime = 1.03f;
        times.TotalFrameTime = 10.53f;
        return times;
    }

    std::size_t Differences(const graphics::FrameBuffer& first, const graphics::FrameBuffer& second, unsigned left,
                            unsigned top, unsigned right, unsigned bottom) {
        std::size_t differences = 0;
        for(unsigned y = top; y < bottom; ++y)
            for(unsigned x = left; x < right; ++x) differences += first.GetPixel(x, y) != second.GetPixel(x, y);
        return differences;
    }
}

// Snapshot support: create a ProfilerView, call this function and pass the returned
// framebuffer to app::WriteBmp. The history and measurements are deterministic.
const graphics::FrameBuffer& DrawProfilerTestSnapshot(app::ProfilerView& view) {
    const auto times = SampleTimes();
    graphics::RenderSettings settings;
    auto metrics = SampleMetrics();
    for(unsigned i = 0; i < app::ProfilerView::Width - 32u; ++i) {
        metrics.FrameIntervalMs = 14.f + static_cast<float>(i % 20u) * .2f + (i % 79u == 0 ? 8.f : 0.f);
        view.Draw(times, settings, metrics);
    }
    metrics = SampleMetrics();
    return view.Draw(times, settings, metrics);
}

void WriteProfilerSnapshot(const char* path) {
    app::ProfilerView view;
    app::WriteBmp(path, DrawProfilerTestSnapshot(view));
}

void RunProfilerTests() {
    graphics::FrameBuffer glyphFrame(32, 24);
    glyphFrame.Clear(0xff010203u);
    app::BitmapFont::Draw(glyphFrame, 2, 3, "A", 0xffffffffu, 2);
    CHECK(glyphFrame.GetPixel(4, 3) == 0xffffffffu);
    CHECK(glyphFrame.GetPixel(2, 3) == 0xff010203u);
    CHECK(app::BitmapFont::GetGlyph('a') == app::BitmapFont::GetGlyph('A'));
    CHECK(app::BitmapFont::TextWidth("FPS", 2) == 34);
    CHECK(app::BitmapFont::TextWidth("", 2) == 0);
    app::BitmapFont::Draw(glyphFrame, -4, -6, "W0", 0xff223344u, 2);
    app::BitmapFont::Draw(glyphFrame, std::numeric_limits<int>::max(), 0, "FPS", 0xffffffffu, 2);
    app::BitmapFont::Draw(glyphFrame, 0, std::numeric_limits<int>::max(), "FPS", 0xffffffffu, 2);
    CHECK(glyphFrame.GetColors().size() == 32u * 24u);

    const auto times = SampleTimes();
    const auto metrics = SampleMetrics();
    graphics::RenderSettings settings;
    app::ProfilerView snapshot;
    const auto& rendered = DrawProfilerTestSnapshot(snapshot);
    CHECK(rendered.GetWidth() == 560 && rendered.GetHeight() == 384);
    std::size_t headingPixels = 0;
    for(unsigned y = 0; y < 36; ++y)
        for(unsigned x = 0; x < rendered.GetWidth(); ++x) headingPixels += rendered.GetPixel(x, y) == 0xffedf3fau;
    CHECK(headingPixels > 100);
    for(const float depth : rendered.GetDepths()) CHECK(depth == 1.f);

    // FPS/FRAME follow actual frame intervals, independently of identical render times.
    app::ProfilerView fastView, slowView;
    auto fast = metrics, slow = metrics;
    fast.FrameIntervalMs = 16.f;
    slow.FrameIntervalMs = 32.f;
    const auto& fastFrame = fastView.Draw(times, settings, fast);
    const auto& slowFrame = slowView.Draw(times, settings, slow);
    CHECK(Differences(fastFrame, slowFrame, 16, 54, 356, 82) > 0);
    CHECK(Differences(fastFrame, slowFrame, 376, 54, 544, 82) == 0);

    // ON/OFF labels are visible, and the reported build mode has its own label.
    app::ProfilerView onView, offView;
    auto disabled = settings;
    disabled.AmbientOcclusion = false;
    auto debugMetrics = metrics;
    debugMetrics.DebugBuild = true;
    const auto& onFrame = onView.Draw(times, settings, metrics);
    const auto& offFrame = offView.Draw(times, disabled, debugMetrics);
    CHECK(Differences(onFrame, offFrame, 150, 322, 276, 362) > 0);
    CHECK(Differences(onFrame, offFrame, 440, 10, 544, 32) > 0);

    // Existing two-argument callers remain valid until application measurements are wired in.
    app::ProfilerView compatible;
    const auto& legacy = compatible.Draw(times, settings);
    CHECK(legacy.GetColors().size() == static_cast<std::size_t>(app::ProfilerView::Width) * app::ProfilerView::Height);

    // Missing/invalid measurements render placeholders without float-to-int overflow.
    auto invalidTimes = times;
    invalidTimes.ShadowPassTime = std::numeric_limits<float>::quiet_NaN();
    invalidTimes.MainPassTime = std::numeric_limits<float>::infinity();
    invalidTimes.PostPassTime = -1.f;
    invalidTimes.TotalFrameTime = std::numeric_limits<float>::quiet_NaN();
    auto invalid = metrics;
    invalid.FrameIntervalMs = std::numeric_limits<float>::quiet_NaN();
    invalid.CpuWorkMs = std::numeric_limits<float>::infinity();
    invalid.PresentMs = -1.f;
    invalid.WaitMs = std::numeric_limits<float>::infinity();
    invalid.CameraDistance = std::numeric_limits<float>::quiet_NaN();
    invalid.ViewportWidth = invalid.ViewportHeight = 0;
    invalid.Workers = std::numeric_limits<std::size_t>::max();
    const auto& invalidFrame = compatible.Draw(invalidTimes, settings, invalid);
    CHECK(invalidFrame.GetColors().size() == legacy.GetColors().size());
}
