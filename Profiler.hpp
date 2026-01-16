#pragma once

namespace debug {
    struct TimeData {
        float ShadowPassTime = 0.f;
        float MainPassTime = 0.f;
        float PostPassTime = 0.f;
        float AAPassTime = 0.f;
        float TotalFrameTime = 0.f;
    };

    inline TimeData Profiler;
}