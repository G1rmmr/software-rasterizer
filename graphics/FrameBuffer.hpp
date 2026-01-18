#pragma once

#include <algorithm>
#include <vector>

#include "../math/Math.hpp"
#include "ParallelExecutor.hpp"

namespace graphics {
    struct BoundingBox {
        std::int32_t MinX;
        std::int32_t MaxX;
        std::int32_t MinY;
        std::int32_t MaxY;
        bool ShouldRender;
    };

    class FrameBuffer {
    public:
        FrameBuffer(const std::uint32_t width, const std::uint32_t height) noexcept
            : colors(width * height, 0),
              depthes(width * height, 1.f),
              normals(width * height, math::Vector(0.f, 0.f, 0.f)),
              width(width),
              height(height) {}

        ~FrameBuffer() noexcept = default;

        FrameBuffer(const FrameBuffer&) = default;
        FrameBuffer& operator=(const FrameBuffer&) = default;

        FrameBuffer(FrameBuffer&& other) noexcept
            : colors(std::move(other.colors)),
              depthes(std::move(other.depthes)),
              normals(std::move(other.normals)),
              width(other.width),
              height(other.height) {}

        FrameBuffer& operator=(FrameBuffer&& other) noexcept {
            if(this != &other) {
                colors = std::move(other.colors);
                depthes = std::move(other.depthes);
                normals = std::move(other.normals);
                width = other.width;
                height = other.height;
            }
            return *this;
        }

        ENGINE_INLINE void SetWidth(const std::uint32_t width) noexcept { this->width = width; }
        [[nodiscard]] ENGINE_INLINE std::uint32_t GetWidth() const noexcept { return width; }

        ENGINE_INLINE void SetHeight(const std::uint32_t height) noexcept { this->height = height; }
        [[nodiscard]] ENGINE_INLINE std::uint32_t GetHeight() const noexcept { return height; }

        ENGINE_INLINE void Clear(const std::uint32_t clearColor = 0) noexcept {
            std::fill(colors.begin(), colors.end(), clearColor);
            std::fill(depthes.begin(), depthes.end(), 1.f);
            std::fill(normals.begin(), normals.end(), math::Vector(0.f, 0.f, 0.f));
        }

        ENGINE_INLINE void SetPixel(const std::uint32_t x, const std::uint32_t y, const std::uint32_t color) noexcept {
            colors[y * width + x] = color;
        }

        ENGINE_INLINE std::uint32_t GetPixel(const std::uint32_t x, const std::uint32_t y) const noexcept {
            if(x < 0 || x >= width || y < 0 || y >= height) return 0x00000000;
            return colors[y * width + x];
        }

        ENGINE_INLINE bool TestDepth(const std::uint32_t x, const std::uint32_t y, const float z) const noexcept {
            return z < depthes[y * width + x];
        }

        ENGINE_INLINE void SetDepth(const std::uint32_t x, const std::uint32_t y, const float z) noexcept {
            depthes[y * width + x] = z;
        }

        ENGINE_INLINE void SetNormal(const std::uint32_t x, const std::uint32_t y,
                                     const math::Vector& normal) noexcept {
            if(x >= width || y >= height) return;
            normals[y * width + x] = normal;
        }

        [[nodiscard]] ENGINE_INLINE math::Vector GetNormal(const std::uint32_t x,
                                                           const std::uint32_t y) const noexcept {
            if(x >= width || y >= height) return math::Vector(0.f, 0.f, 0.f);
            return normals[y * width + x];
        }

        ENGINE_INLINE BoundingBox ENGINE_VECTORCALL GetBound(const math::Vector& v0, const math::Vector& v1,
                                                             const math::Vector& v2) const noexcept {
            float minX = std::min({v0.X, v1.X, v2.X});
            float maxX = std::max({v0.X, v1.X, v2.X});
            float minY = std::min({v0.Y, v1.Y, v2.Y});
            float maxY = std::max({v0.Y, v1.Y, v2.Y});

            std::int32_t left = std::max(0, static_cast<std::int32_t>(std::floor(minX)));
            std::int32_t right =
                std::min(static_cast<std::int32_t>(width) - 1, static_cast<std::int32_t>(std::ceil(maxX)));

            std::int32_t bottom = std::max(0, static_cast<std::int32_t>(std::floor(minY)));
            std::int32_t top =
                std::min(static_cast<std::int32_t>(height) - 1, static_cast<std::int32_t>(std::ceil(maxY)));

            return {static_cast<std::int32_t>(left), static_cast<std::int32_t>(right),
                    static_cast<std::int32_t>(bottom), static_cast<std::int32_t>(top), left <= right && bottom <= top};
        }

        [[nodiscard]] ENGINE_INLINE std::uint32_t* __restrict GetColor() noexcept { return colors.data(); }
        [[nodiscard]] ENGINE_INLINE float* __restrict GetDepth() noexcept { return depthes.data(); }

        ENGINE_INLINE std::vector<std::uint32_t>& GetColors() noexcept { return colors; }
        ENGINE_INLINE const std::vector<std::uint32_t>& GetColors() const noexcept { return colors; }

        ENGINE_INLINE std::vector<float>& GetDepthes() noexcept { return depthes; }
        ENGINE_INLINE const std::vector<float>& GetDepthes() const noexcept { return depthes; }

        ENGINE_INLINE std::vector<math::Vector>& GetNormals() noexcept { return normals; }
        ENGINE_INLINE const std::vector<math::Vector>& GetNormals() const noexcept { return normals; }

        ENGINE_INLINE void UpdateBuffer(const std::vector<std::uint32_t>& newColors) noexcept { colors = newColors; }

        ENGINE_INLINE void AntiAlias(const bool shouldAlias = true, const std::uint32_t threshold = 30) {
            if(!shouldAlias) return;

            std::vector<std::uint32_t> dst(width * height);

            ParallelExecutor::GetInstance().ParallelFor(
                1, height - 1,
                [&](std::size_t y) {
                    for(std::uint32_t x = 1; x < width - 1; ++x) {
                        std::uint32_t idx = y * width + x;

                        std::uint32_t current = colors[idx];

                        std::uint32_t up = colors[idx - width];
                        std::uint32_t down = colors[idx + width];
                        std::uint32_t left = colors[idx - 1];
                        std::uint32_t right = colors[idx + 1];

                        std::uint32_t diff = colorDiff(current, up) + colorDiff(current, down) +
                                             colorDiff(current, left) + colorDiff(current, right);

                        dst[idx] = diff > threshold ? mixColors(current, up, down, left, right) : current;
                    }
                },
                16);
            UpdateBuffer(dst);
        }

    private:
        std::vector<math::Vector> normals;
        std::vector<std::uint32_t> colors;
        std::vector<float> depthes;
        std::uint32_t width;
        std::uint32_t height;

        ENGINE_INLINE std::int32_t ENGINE_VECTORCALL colorDiff(const std::uint32_t c1, const std::uint32_t c2) {
            int r1 = (c1 >> 16) & 0xFF;
            int g1 = (c1 >> 8) & 0xFF;
            int b1 = c1 & 0xFF;
            int r2 = (c2 >> 16) & 0xFF;
            int g2 = (c2 >> 8) & 0xFF;
            int b2 = c2 & 0xFF;
            return std::abs(r1 - r2) + std::abs(g1 - g2) + std::abs(b1 - b2);
        }

        ENGINE_INLINE std::uint32_t ENGINE_VECTORCALL mixColors(const std::uint32_t c1, const std::uint32_t c2,
                                                                const std::uint32_t c3, const std::uint32_t c4,
                                                                const std::uint32_t c5) {
            int r = (((c1 >> 16) & 0xFF) + ((c2 >> 16) & 0xFF) + ((c3 >> 16) & 0xFF) + ((c4 >> 16) & 0xFF) +
                     ((c5 >> 16) & 0xFF)) /
                    5;

            int g = (((c1 >> 8) & 0xFF) + ((c2 >> 8) & 0xFF) + ((c3 >> 8) & 0xFF) + ((c4 >> 8) & 0xFF) +
                     ((c5 >> 8) & 0xFF)) /
                    5;

            int b = ((c1 & 0xFF) + (c2 & 0xFF) + (c3 & 0xFF) + (c4 & 0xFF) + (c5 & 0xFF)) / 5;

            return (0xFF << 24) | (r << 16) | (g << 8) | b;
        }
    };
}
