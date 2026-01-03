#pragma once

#include <algorithm>
#include <vector>

#include "../math/Math.hpp"

namespace graphics {
    struct BoundingBox {
        int MinX;
        int MaxX;
        int MinY;
        int MaxY;
        bool ShouldRender;
    };

    class FrameBuffer {
    public:
        FrameBuffer(const std::uint32_t width, const std::uint32_t height)
            : colors(width * height, 0), depthes(width * height, 1.0f), width(width), height(height) {}

        ~FrameBuffer() = default;

        FrameBuffer(const FrameBuffer& other) noexcept
            : colors(other.colors), depthes(other.depthes), width(other.width), height(other.height) {}

        FrameBuffer(FrameBuffer&& other) noexcept
            : colors(other.colors), depthes(other.depthes), width(other.width), height(other.height) {}

        FrameBuffer& operator=(const FrameBuffer& other) noexcept {
            if(this != &other) {
                colors = other.colors;
                depthes = other.depthes;
                width = other.width;
                height = other.height;
            }
            return *this;
        }

        FrameBuffer& operator=(FrameBuffer&& other) noexcept {
            if(this != &other) {
                colors = other.colors;
                depthes = other.depthes;
                width = other.width;
                height = other.height;
            }
            return *this;
        }

        inline void Clear(const std::uint32_t clearColor = 0) noexcept {
            std::fill(colors.begin(), colors.end(), clearColor);
            std::fill(depthes.begin(), depthes.end(), 1.f);
        }

        inline void SetPixel(const std::uint32_t x, const std::uint32_t y, const std::uint32_t color) noexcept {
            colors[y * width + x] = color;
        }

        inline bool IsVisible(const std::uint32_t x, const std::uint32_t y, const float z) {
            const std::uint32_t index = y * width + x;

            if(z < depthes[index]) {
                depthes[index] = z;
                return true;
            }

            return false;
        }

        inline BoundingBox GetBound(const math::Vector& v0, const math::Vector& v1, const math::Vector& v2) {
            if(v0.Z < 0.f || v1.Z < 0.f || v2.Z < 0.f) return {0, 0, 0, 0, false};

            int minX = std::max({0, static_cast<int>(std::floor(std::min({v0.X, v1.X, v2.X})))});
            int maxX =
                std::min({static_cast<int>(width - 1), static_cast<int>(std::ceil(std::max({v0.X, v1.X, v2.X})))});
            int minY = std::max({0, static_cast<int>(std::floor(std::min({v0.Y, v1.Y, v2.Y})))});
            int maxY =
                std::min({static_cast<int>(height - 1), static_cast<int>(std::ceil(std::max({v0.Y, v1.Y, v2.Y})))});

            return {minX, maxX, minY, maxY, minX <= maxX && minY <= maxY};
        }

        inline std::uint32_t* GetColor() { return colors.data(); }

    private:
        std::vector<std::uint32_t> colors;
        std::vector<float> depthes;
        std::uint32_t width;
        std::uint32_t height;
    };
}
