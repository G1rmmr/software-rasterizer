#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "../math/Math.hpp"

namespace graphics {
    class Texture {
    public:
        Texture() = default;

        explicit Texture(const char* filePath, bool isData = false) {
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

            if(!isData) {
                for(std::size_t i = 0; i < data.size(); i += 4) {
                    std::swap(data[i], data[i + 2]);
                }
            }

            stbi_image_free(rawData);
        }

        math::Vector Sample(float u, float v) const {
            if(data.empty()) return {1.f, 0.f, 1.f, 1.f};

            u -= std::floor(u);
            v -= std::floor(v);

            std::uint32_t x = static_cast<std::uint32_t>(u * width);
            std::uint32_t y = static_cast<std::uint32_t>(v * height);

            x = std::clamp(x, 0u, width - 1);
            y = std::clamp(y, 0u, height - 1);

            std::int32_t index = (y * width + x) * channels;

            float r = data[index] / 255.f;
            float g = (channels > 1) ? data[index + 1] / 255.f : r;
            float b = (channels > 2) ? data[index + 2] / 255.f : r;
            float a = (channels > 3) ? data[index + 3] / 255.f : 1.f;

            return {r, g, b, a};
        }

    private:
        std::vector<std::uint8_t> data;

        std::uint32_t width;
        std::uint32_t height;
        std::uint8_t channels;
    };
}