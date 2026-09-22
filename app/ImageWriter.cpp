#include "ImageWriter.hpp"
#include <array>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace app {
    void WriteBmp(const std::filesystem::path& path, const graphics::FrameBuffer& frame) {
        const std::uint64_t bytes = static_cast<std::uint64_t>(frame.GetWidth()) * frame.GetHeight() * 4;
        if(bytes > std::numeric_limits<std::uint32_t>::max() - 54) throw std::length_error("BMP exceeds 32-bit size");
        std::ofstream file(path, std::ios::binary);
        if(!file) throw std::runtime_error("Cannot create image: " + path.string());
        std::array<unsigned char, 54> header{};
        const auto put = [&](std::size_t offset, std::uint32_t value) {
            for(int i = 0; i < 4; ++i) header[offset + i] = static_cast<unsigned char>(value >> (8 * i));
        };
        header[0] = 'B';
        header[1] = 'M';
        put(2, static_cast<std::uint32_t>(bytes) + 54);
        put(10, 54);
        put(14, 40);
        put(18, frame.GetWidth());
        put(22, frame.GetHeight());
        header[26] = 1;
        header[28] = 32;
        put(34, static_cast<std::uint32_t>(bytes));
        file.write(reinterpret_cast<const char*>(header.data()), header.size());
        for(std::uint32_t y = frame.GetHeight(); y-- > 0;)
            for(std::uint32_t x = 0; x < frame.GetWidth(); ++x) {
                const auto pixel = frame.GetPixel(x, y);
                const std::array<unsigned char, 4> bgra{static_cast<unsigned char>(pixel),
                                                        static_cast<unsigned char>(pixel >> 8),
                                                        static_cast<unsigned char>(pixel >> 16), 255};
                file.write(reinterpret_cast<const char*>(bgra.data()), bgra.size());
            }
        file.flush();
        if(!file) throw std::runtime_error("Could not write image: " + path.string());
    }
}
