#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_SIMD
#include "../libs/stb_image.h"

#include "../math/Math.hpp"

namespace graphics {
    class Texture {
    public:
        Texture() = default;

        explicit Texture(const char* filePath) {
            stbi_set_flip_vertically_on_load(true);

            std::int32_t w;
            std::int32_t h;
            std::int32_t c;

            std::uint8_t* rawData = stbi_load(filePath, &w, &h, &c, 4);
            if(!rawData) {
                std::fprintf(stderr, "Failed to load texture (stb): %s\n", filePath);
                return;
            }

            width = w;
            height = h;
            channels = 4;

            std::size_t size = width * height * 4;
            data.resize(size);

            std::memcpy(data.data(), rawData, size);
            stbi_image_free(rawData);
        }

        [[nodiscard]] ENGINE_INLINE math::Vector __vectorcall Sample(float u, float v) const noexcept {
            if(data.empty()) [[unlikely]]
                return {1.f, 0.f, 1.f, 1.f};

            u -= std::floor(u);
            v -= std::floor(v);

            std::uint32_t x = static_cast<std::uint32_t>(u * width);
            std::uint32_t y = static_cast<std::uint32_t>(v * height);

            if(x >= width) x = width - 1;
            if(y >= height) y = height - 1;

            std::size_t index = (y * width + x) * channels;

            const std::uint8_t* __restrict ptr = data.data();

            if(channels == 4) {
                float r = ptr[index] * (1.0f / 255.0f);
                float g = ptr[index + 1] * (1.0f / 255.0f);
                float b = ptr[index + 2] * (1.0f / 255.0f);
                float a = ptr[index + 3] * (1.0f / 255.0f);
                return math::Vector(r, g, b, a);
            }

            float r = ptr[index] * (1.0f / 255.0f);
            float g = (channels > 1) ? ptr[index + 1] * (1.0f / 255.0f) : r;
            float b = (channels > 2) ? ptr[index + 2] * (1.0f / 255.0f) : r;
            float a = (channels > 3) ? ptr[index + 3] * (1.0f / 255.0f) : 1.f;

            return math::Vector(r, g, b, a);
        }

    private:
        std::vector<std::uint8_t> data;

        std::uint32_t width;
        std::uint32_t height;
        std::uint8_t channels;
    };
}