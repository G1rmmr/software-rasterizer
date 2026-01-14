#pragma once

#include <algorithm>
#include <vector>

#include "../math/Math.hpp"

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
            if(x < 0 || x >= width || y < 0 || y >= height) [[unlikely]]
                return;

            colors[y * width + x] = color;
        }

        ENGINE_INLINE std::uint32_t GetPixel(const std::uint32_t x, const std::uint32_t y) const noexcept {
            if(x < 0 || x >= width || y < 0 || y >= height) return 0x00000000;
            return colors[y * width + x];
        }

        ENGINE_INLINE bool TestDepth(const std::uint32_t x, const std::uint32_t y, const float z) const noexcept {
            if(x >= width || y >= height) return false;
            return z < depthes[y * width + x];
        }

        ENGINE_INLINE void SetDepth(const std::uint32_t x, const std::uint32_t y, const float z) noexcept {
            if(x >= width || y >= height) return;
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

        ENGINE_INLINE void UpdateBuffer(const std::vector<std::uint32_t>& newColors) noexcept { colors = newColors; }

    private:
        std::vector<math::Vector> normals;
        std::vector<std::uint32_t> colors;
        std::vector<float> depthes;
        std::uint32_t width;
        std::uint32_t height;
    };
}
