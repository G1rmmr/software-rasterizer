#include "Texture.hpp"

#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_SIMD
#include "../libs/stb_image.h"

namespace {
    std::size_t ImageBytes(std::uint32_t width, std::uint32_t height) {
        if(width == 0 || height == 0 ||
           static_cast<std::size_t>(width) > std::numeric_limits<std::size_t>::max() / 4 / height)
            throw std::invalid_argument("Texture dimensions must describe a nonempty RGBA8 image");
        return static_cast<std::size_t>(width) * height * 4;
    }

    std::string PathLabel(const std::filesystem::path& path) {
        const auto utf8 = path.u8string();
        return {utf8.begin(), utf8.end()};
    }
}

namespace graphics {
    Texture::Texture(std::uint32_t width, std::uint32_t height, std::vector<std::uint8_t> rgba)
        : width(width), height(height), data(std::move(rgba)) {
        if(data.size() != ImageBytes(width, height))
            throw std::invalid_argument("Texture byte count does not match width * height * 4");
    }

    Texture::Texture(const std::filesystem::path& path) : width(0), height(0) {
        const std::string label = PathLabel(path);
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if(!file) throw std::runtime_error("Cannot open texture: " + label);
        const auto length = file.tellg();
        if(length <= 0 || length > std::numeric_limits<int>::max())
            throw std::runtime_error("Empty or oversized texture: " + label);
        std::vector<stbi_uc> encoded(static_cast<std::size_t>(length));
        file.seekg(0);
        if(!file.read(reinterpret_cast<char*>(encoded.data()), static_cast<std::streamsize>(encoded.size())))
            throw std::runtime_error("Cannot read texture: " + label);

        int w = 0, h = 0, channels = 0;
        std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> decoded(
            stbi_load_from_memory(encoded.data(), static_cast<int>(encoded.size()), &w, &h, &channels, 4),
            &stbi_image_free);
        if(!decoded) {
            const char* reason = stbi_failure_reason();
            throw std::runtime_error("Cannot decode texture " + label + ": " + (reason ? reason : "unknown format"));
        }
        width = static_cast<std::uint32_t>(w);
        height = static_cast<std::uint32_t>(h);
        data.resize(ImageBytes(width, height));
        const std::size_t rowBytes = static_cast<std::size_t>(width) * 4;
        // Flip our copy instead of changing stb's process-wide loader settings.
        for(std::uint32_t row = 0; row < height; ++row)
            std::copy_n(decoded.get() + static_cast<std::size_t>(height - row - 1) * rowBytes, rowBytes,
                        data.data() + static_cast<std::size_t>(row) * rowBytes);
    }
}
