#pragma once

#include "../../math/Math.hpp"
#include "../FrameBuffer.hpp"
#include "../ParallelExecutor.hpp"
#include <cstddef>
#include <vector>

namespace graphics {
    struct SsaoSettings {
        float Radius = 1.f; // View-space units, independent of resolution and projection.
        float Bias = 0.05f;
        float Strength = 1.5f;
        std::size_t KernelSize = 8;
    };

    // Owns reusable half-resolution working attachments. The executor outlives the pass.
    class SsaoPass final {
    public:
        explicit SsaoPass(ParallelExecutor& executor) noexcept : executor_(executor) {}
        void Execute(FrameBuffer& frame, const math::Matrix& projection, const math::Matrix& inverseProjection,
                     const SsaoSettings& settings = {});

    private:
        void PrepareKernel(std::size_t count);
        void PrepareReconstruction(std::uint32_t width, std::uint32_t height, const math::Matrix& inverseProjection);
        void PrepareRotations(std::uint32_t width, std::uint32_t height);
        struct Rotation {
            float Cosine, Sine;
        };
        ParallelExecutor& executor_;
        math::Matrix cachedInverseProjection_;
        bool reconstructionReady_ = false;
        std::vector<math::Vector> reconstructionX_;
        std::vector<math::Vector> reconstructionY_;
        std::uint32_t rotationWidth_ = 0, rotationHeight_ = 0;
        std::vector<Rotation> rotations_;
        std::vector<math::Vector> kernel_;
        std::vector<math::Vector> positions_;
        std::vector<math::Vector> normals_;
        std::vector<float> visibility_;
        std::vector<float> blurred_;
        // One bit per raw sample that differs from full visibility; rows own their words.
        std::vector<std::uint64_t> occlusionWords_;
        std::vector<std::uint64_t> blurWords_;
    };
}
