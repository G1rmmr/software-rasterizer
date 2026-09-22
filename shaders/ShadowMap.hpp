#pragma once
#include "../graphics/FrameBuffer.hpp"
#include "../math/Math.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <span>

namespace shader {
    // A read-only view of a completed shadow pass. The framebuffer must outlive
    // this view; resizing its attachments invalidates the cached depth span.
    // ShadowPass constructs a new view after rendering its fixed-size target.
    class ShadowMap final {
    public:
        ShadowMap(const graphics::FrameBuffer& frame, const math::Matrix& lightSpace, float nearPlane, float farPlane)
            : depths_(frame.GetDepths()),
              lightSpace_(lightSpace),
              width_(frame.GetWidth()),
              widthFloat_(static_cast<float>(frame.GetWidth())),
              heightFloat_(static_cast<float>(frame.GetHeight())),
              maxX_(static_cast<float>(frame.GetWidth() - 1)),
              maxY_(static_cast<float>(frame.GetHeight() - 1)),
              near_(nearPlane),
              depthRange_(farPlane - nearPlane),
              powerOfTwoExtent_(std::has_single_bit(frame.GetWidth()) && std::has_single_bit(frame.GetHeight())) {
            for(std::size_t i = 0; i < samples_.size(); ++i) {
                // Preserve the original multiply-then-divide rounding for blocker search.
                blockerOffsets_[i] = {samples_[i].X * 3.f / widthFloat_, samples_[i].Y * 3.f / heightFloat_};
                filterOffsets_[i] = {samples_[i].X / widthFloat_, samples_[i].Y / heightFloat_};
            }
        }

        [[nodiscard]] float Visibility(const math::Vector& position, const math::Vector& normal,
                                       const math::Vector& lightDirection) const noexcept {
            const auto clip = lightSpace_ * math::Vector(position.X, position.Y, position.Z, 1.f);
            if(clip.W <= 0.f) return 1.f;
            const auto ndc = clip / clip.W;
            if(ndc.X < -1.f || ndc.X > 1.f || ndc.Y < -1.f || ndc.Y > 1.f || ndc.Z < 0.f || ndc.Z > 1.f) return 1.f;
            const float u = ndc.X * .5f + .5f, v = .5f - ndc.Y * .5f;
            const float bias = std::max(.0005f, .0025f * (1.f - std::clamp(normal.Dot(lightDirection), 0.f, 1.f)));
            const float comparisonDepth = ndc.Z - bias;
            float blockers = 0.f, blockerDepth = 0.f;
            for(const auto& offset : blockerOffsets_) {
                const float depth = Depth(u + offset.X, v + offset.Y);
                if(depth < comparisonDepth) {
                    blockerDepth += depth;
                    blockers += 1.f;
                }
            }
            if(blockers == 0.f) return 1.f;
            const float receiverDistance = near_ + ndc.Z * depthRange_;
            const float blockerDistance = near_ + (blockerDepth / blockers) * depthRange_;
            const float ratio = std::max(0.f, receiverDistance - blockerDistance) / std::max(.01f, blockerDistance);
            const float radius = std::clamp(1.f + ratio * 10.f, 1.f, 8.f);
            float visible = 0.f;
            if(powerOfTwoExtent_) {
                // Power-of-two scaling is exact for these normal-valued offsets
                // and radius [1,8], so moving division before multiplication
                // preserves the sampled texel. Other extents keep the original order.
                for(const auto& offset : filterOffsets_) {
                    const float depth = Depth(u + offset.X * radius, v + offset.Y * radius);
                    visible += comparisonDepth <= depth ? 1.f : 0.f;
                }
            }
            else {
                for(const auto& sample : samples_) {
                    const float depth =
                        Depth(u + sample.X * radius / widthFloat_, v + sample.Y * radius / heightFloat_);
                    visible += comparisonDepth <= depth ? 1.f : 0.f;
                }
            }
            return visible / static_cast<float>(samples_.size());
        }

    private:
        float Depth(float u, float v) const noexcept {
            const auto x = static_cast<std::uint32_t>(std::clamp(u * widthFloat_, 0.f, maxX_));
            const auto y = static_cast<std::uint32_t>(std::clamp(v * heightFloat_, 0.f, maxY_));
            return depths_[static_cast<std::size_t>(y) * width_ + x];
        }
        struct Offset {
            float X, Y;
        };
        std::span<const float> depths_;
        math::Matrix lightSpace_;
        std::uint32_t width_;
        float widthFloat_, heightFloat_, maxX_, maxY_, near_, depthRange_;
        bool powerOfTwoExtent_;
        std::array<Offset, 8> blockerOffsets_, filterOffsets_;
        static constexpr std::array<Offset, 8> samples_{{{-.942f, -.399f},
                                                         {.946f, -.769f},
                                                         {-.094f, -.929f},
                                                         {.345f, .294f},
                                                         {-.916f, .458f},
                                                         {-.815f, -.879f},
                                                         {-.382f, .277f},
                                                         {.975f, .756f}}};
    };
}
