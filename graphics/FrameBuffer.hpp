#pragma once

#include "../math/Vector.hpp"
#include "ParallelExecutor.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace graphics {
    // Owns equally sized color, depth and view-space normal attachments.
    // Views permit editing pixels, never changing an attachment's extent.
    class FrameBuffer final {
    public:
        FrameBuffer(std::uint32_t width, std::uint32_t height) { Resize(width, height); }
        FrameBuffer(const FrameBuffer&) = delete;
        FrameBuffer& operator=(const FrameBuffer&) = delete;
        FrameBuffer(FrameBuffer&&) = delete;
        FrameBuffer& operator=(FrameBuffer&&) = delete;

        void Resize(std::uint32_t width, std::uint32_t height) {
            if(width == 0 || height == 0 || width > INT32_MAX || height > INT32_MAX)
                throw std::invalid_argument("FrameBuffer dimensions must be positive signed 32-bit values");
            const auto count = static_cast<std::size_t>(width) * height;
            if(count > std::numeric_limits<std::size_t>::max() / sizeof(math::Vector))
                throw std::length_error("FrameBuffer dimensions overflow storage size");
            if(width == width_ && height == height_) return;
            std::vector<std::uint32_t> colors(count, 0xff000000u);
            std::vector<float> depths(count, 1.f);
            std::vector<math::Vector> normals(count);
            colors_.swap(colors);
            depths_.swap(depths);
            normals_.swap(normals);
            width_ = width;
            height_ = height;
        }

        void Clear(std::uint32_t color = 0xff000000u) noexcept {
            std::fill(colors_.begin(), colors_.end(), color);
            std::fill(depths_.begin(), depths_.end(), 1.f);
            std::fill(normals_.begin(), normals_.end(), math::Vector{});
        }

        // A depth-only pass does not need to touch the other attachments.
        void ClearDepth() noexcept { std::fill(depths_.begin(), depths_.end(), 1.f); }

        void Clear(ParallelExecutor& executor, std::uint32_t color = 0xff000000u) {
            executor.ParallelFor(
                0, height_,
                [&](std::size_t y) {
                    const auto offset = y * width_;
                    std::fill_n(colors_.data() + offset, width_, color);
                    std::fill_n(depths_.data() + offset, width_, 1.f);
                    std::fill_n(normals_.data() + offset, width_, math::Vector{});
                },
                16);
        }

        [[nodiscard]] std::uint32_t GetWidth() const noexcept { return width_; }
        [[nodiscard]] std::uint32_t GetHeight() const noexcept { return height_; }
        [[nodiscard]] std::span<const std::uint32_t> GetColors() const noexcept { return colors_; }
        [[nodiscard]] std::span<const float> GetDepths() const noexcept { return depths_; }
        [[nodiscard]] std::span<const math::Vector> GetNormals() const noexcept { return normals_; }
        [[nodiscard]] std::span<std::uint32_t> Colors() noexcept { return colors_; }

        [[nodiscard]] std::uint32_t GetPixel(std::uint32_t x, std::uint32_t y) const noexcept {
            return colors_[Index(x, y)];
        }
        [[nodiscard]] float GetDepth(std::uint32_t x, std::uint32_t y) const noexcept { return depths_[Index(x, y)]; }
        [[nodiscard]] math::Vector GetNormal(std::uint32_t x, std::uint32_t y) const noexcept {
            return normals_[Index(x, y)];
        }
        [[nodiscard]] bool TestDepth(std::uint32_t x, std::uint32_t y, float z) const noexcept {
            return z >= 0.f && z <= 1.f && z < depths_[Index(x, y)];
        }
        void SetPixel(std::uint32_t x, std::uint32_t y, std::uint32_t color) noexcept { colors_[Index(x, y)] = color; }
        void SetDepth(std::uint32_t x, std::uint32_t y, float depth) noexcept {
            assert(std::isfinite(depth) && depth >= 0.f && depth <= 1.f);
            depths_[Index(x, y)] = depth;
        }
        void SetNormal(std::uint32_t x, std::uint32_t y, const math::Vector& normal) noexcept {
            normals_[Index(x, y)] = normal;
        }

    private:
        [[nodiscard]] std::size_t Index(std::uint32_t x, std::uint32_t y) const noexcept {
            assert(x < width_ && y < height_);
            return static_cast<std::size_t>(y) * width_ + x;
        }
        std::uint32_t width_ = 0;
        std::uint32_t height_ = 0;
        std::vector<std::uint32_t> colors_;
        std::vector<float> depths_;
        std::vector<math::Vector> normals_;
    };
}
