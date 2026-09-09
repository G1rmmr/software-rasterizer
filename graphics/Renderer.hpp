#pragma once
#include "../Profiler.hpp"
#include "Camera.hpp"
#include "DirectionalLight.hpp"
#include "FrameBuffer.hpp"
#include "ParallelExecutor.hpp"
#include "Rasterizer.hpp"
#include "RenderSettings.hpp"
#include "passes/GeometryPass.hpp"
#include "passes/ShadowPass.hpp"
#include "post/AntiAliasingPass.hpp"
#include "post/SsaoPass.hpp"

namespace graphics {
    // One renderer owns one executor and all of its targets/scratch storage.
    // Render returns only after every pass has completed; no work outlives its owner.
    class Renderer final {
    public:
        Renderer(std::uint32_t width, std::uint32_t height,
                 std::size_t workers = ParallelExecutor::DefaultWorkerCount());
        void Resize(std::uint32_t width, std::uint32_t height) { frame_.Resize(width, height); }
        const FrameBuffer& Render(const scene::Scene& scene, const Camera& camera, const DirectionalLight& light,
                                  const RenderSettings& settings = {});
        [[nodiscard]] const debug::TimeData& GetTimings() const noexcept { return timings_; }
        [[nodiscard]] const FrameBuffer& GetFrame() const noexcept { return frame_; }

    private:
        ParallelExecutor executor_;
        Rasterizer rasterizer_;
        ShadowPass shadow_;
        GeometryPass geometry_;
        SsaoPass ssao_;
        AntiAliasingPass aa_;
        FrameBuffer frame_;
        debug::TimeData timings_;
    };
}
