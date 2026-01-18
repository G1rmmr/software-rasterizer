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

    template <typename Func> decltype(auto) Measure(float& outTime, Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();

        if constexpr(std::is_void_v<std::invoke_result_t<Func>>) {
            std::forward<Func>(func)(); // 실행
            auto end = std::chrono::high_resolution_clock::now();
            outTime = std::chrono::duration<float, std::milli>(end - start).count();
        }
        else {
            decltype(auto) result = std::forward<Func>(func)();
            auto end = std::chrono::high_resolution_clock::now();
            outTime = std::chrono::duration<float, std::milli>(end - start).count();
            return result;
        }
    }
}