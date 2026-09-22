#pragma once
#include <chrono>

namespace debug {
    struct TimeData {
        float ShadowPassTime = 0.f;
        float MainPassTime = 0.f;
        float PostPassTime = 0.f;
        float AAPassTime = 0.f;
        float TotalFrameTime = 0.f;
    };

    class ScopedTimer final {
    public:
        explicit ScopedTimer(float& output) noexcept : output_(output), start_(Clock::now()) {}
        ~ScopedTimer() { output_ = std::chrono::duration<float, std::milli>(Clock::now() - start_).count(); }
        ScopedTimer(const ScopedTimer&) = delete;
        ScopedTimer& operator=(const ScopedTimer&) = delete;

    private:
        using Clock = std::chrono::steady_clock;
        float& output_;
        Clock::time_point start_;
    };
}
