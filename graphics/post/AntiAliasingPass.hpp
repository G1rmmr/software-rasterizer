#pragma once

#include "../FrameBuffer.hpp"
#include "../ParallelExecutor.hpp"
#include <cstdint>
#include <vector>

namespace graphics {
    struct AntiAliasingSettings {
        float DepthThreshold = 0.01f;      // Normalized [0,1] depth.
        float NormalThreshold = 0.95f;     // Cosine of the maximum smooth angle.
        std::uint32_t ColorThreshold = 30; // Largest byte-channel difference.
    };

    class AntiAliasingPass final {
    public:
        explicit AntiAliasingPass(ParallelExecutor& executor) noexcept : executor_(executor) {}
        void Execute(FrameBuffer& frame, const AntiAliasingSettings& settings = {});

    private:
        ParallelExecutor& executor_;
        std::vector<std::uint32_t> sourceColors_;
        // XYZ retain the exact normalization result; W caches whether it is nonzero.
        std::vector<math::Vector> normalizedNormals_;
    };
}
