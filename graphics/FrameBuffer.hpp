#pragma once

#include <algorithm>
#include <vector>

#include "../math/Math.hpp"

namespace graphics {
    struct BoundingBox {
        std::uint32_t MinX;
        std::uint32_t MaxX;
        std::uint32_t MinY;
        std::uint32_t MaxY;
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

        inline void SetWidth(const std::uint32_t width) { this->width = width; }
        inline std::uint32_t GetWidth() const { return width; }

        inline void SetHeight(const std::uint32_t height) { this->height = height; }
        inline std::uint32_t GetHeight() const { return height; }

        inline void Clear(const std::uint32_t clearColor = 0) noexcept {
            std::fill(colors.begin(), colors.end(), clearColor);
            std::fill(depthes.begin(), depthes.end(), 1.f);
        }

        inline void SetPixel(const std::uint32_t x, const std::uint32_t y, const std::uint32_t color) noexcept {
            if(x < 0 || x >= width || y < 0 || y >= height) return;
            colors[y * width + x] = color;
        }

        inline bool IsVisible(const std::uint32_t x, const std::uint32_t y, const float z) {
            if(x < 0 || x >= width || y < 0 || y >= height) return false;

            const std::uint32_t index = y * width + x;
            if(z < depthes[index]) {
                depthes[index] = z;
                return true;
            }
            return false;
        }

        inline BoundingBox GetBound(const math::Vector& v0, const math::Vector& v1, const math::Vector& v2) {
            float minX = std::min({v0.X, v1.X, v2.X});
            float maxX = std::max({v0.X, v1.X, v2.X});
            float minY = std::min({v0.Y, v1.Y, v2.Y});
            float maxY = std::max({v0.Y, v1.Y, v2.Y});

            int left = std::max(0, static_cast<int>(std::floor(minX)));
            int right = std::min(static_cast<int>(width) - 1, static_cast<int>(std::ceil(maxX)));
            int bottom = std::max(0, static_cast<int>(std::floor(minY)));
            int top = std::min(static_cast<int>(height) - 1, static_cast<int>(std::ceil(maxY)));

            return {static_cast<std::uint32_t>(left), static_cast<std::uint32_t>(right),
                    static_cast<std::uint32_t>(bottom), static_cast<std::uint32_t>(top),
                    left <= right && bottom <= top};
        }

        inline std::uint32_t* GetColor() { return colors.data(); }

        const std::vector<std::uint32_t>& GetColorBuffer() const { return colors; }

        void UpdateBuffer(const std::vector<std::uint32_t>& newColors) { colors = newColors; }

    private:
        std::vector<std::uint32_t> colors;
        std::vector<float> depthes;
        std::uint32_t width;
        std::uint32_t height;
    };
}
