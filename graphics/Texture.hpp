#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <vector>

#include "../math/Vector.hpp"

namespace graphics {
    // Immutable RGBA8 image. Row zero is the bottom row, matching OBJ UVs.
    class Texture {
    public:
        explicit Texture(const std::filesystem::path& path);
        Texture(std::uint32_t width, std::uint32_t height, std::vector<std::uint8_t> rgba);
        Texture(const Texture&) = default;
        Texture& operator=(const Texture&) = delete;

        [[nodiscard]] std::uint32_t GetWidth() const noexcept { return width; }
        [[nodiscard]] std::uint32_t GetHeight() const noexcept { return height; }

        [[nodiscard]] math::Vector Sample(float u, float v) const {
            if(!std::isfinite(u) || !std::isfinite(v))
                throw std::invalid_argument("Texture coordinates must be finite");
            u -= std::floor(u);
            v -= std::floor(v);
            const auto x = std::min(static_cast<std::uint32_t>(static_cast<double>(u) * width), width - 1);
            const auto y = std::min(static_cast<std::uint32_t>(static_cast<double>(v) * height), height - 1);
            const std::size_t offset = (static_cast<std::size_t>(y) * width + x) * 4;
            constexpr float scale = 1.f / 255.f;
            return {data[offset] * scale, data[offset + 1] * scale, data[offset + 2] * scale, data[offset + 3] * scale};
        }

    private:
        std::uint32_t width;
        std::uint32_t height;
        std::vector<std::uint8_t> data;
    };
}
