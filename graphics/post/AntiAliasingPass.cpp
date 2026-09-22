#include "AntiAliasingPass.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace graphics {
    void AntiAliasingPass::Execute(FrameBuffer& frame, const AntiAliasingSettings& settings) {
        if(!std::isfinite(settings.DepthThreshold) || settings.DepthThreshold < 0.f || settings.DepthThreshold > 1.f ||
           !std::isfinite(settings.NormalThreshold) || settings.NormalThreshold < 0.f ||
           settings.NormalThreshold > 1.f || settings.ColorThreshold > 255)
            throw std::invalid_argument("Invalid antialiasing thresholds");
        const auto width = frame.GetWidth(), height = frame.GetHeight();
        if(width < 3 || height < 3) return;
        constexpr std::size_t rowChunkSize = 4;
        const auto depths = frame.GetDepths();
        const auto normals = frame.GetNormals();
        const auto inputColors = frame.GetColors();
        sourceColors_.resize(inputColors.size());
        normalizedNormals_.resize(normals.size());
        executor_.ParallelFor(
            0, height,
            [&](std::size_t y) {
                for(std::uint32_t x = 0; x < width; ++x) {
                    const std::size_t index = y * width + x;
                    sourceColors_[index] = inputColors[index];
                    auto normal = normals[index];
                    const float lengthSquared = normal.Dot(normal);
                    if(lengthSquared == 0.f)
                        normal = math::Vector{};
                    else if(lengthSquared == 1.f)
                        normal.W = 1.f;
                    else {
                        normal = normal.NormalizedDirection();
                        normal.W = normal.Dot(normal) > .5f ? 1.f : 0.f;
                    }
                    normalizedNormals_[index] = normal;
                }
            },
            rowChunkSize);
        const auto& colors = sourceColors_;
        auto output = frame.Colors();

        const auto isEdge = [&](std::size_t first, std::size_t second) {
            if(std::abs(depths[first] - depths[second]) > settings.DepthThreshold) return true;
            const auto& n0 = normalizedNormals_[first];
            const auto& n1 = normalizedNormals_[second];
            const bool valid0 = n0.W != 0.f, valid1 = n1.W != 0.f;
            if(valid0 != valid1 || (valid0 && n0.Dot(n1) < settings.NormalThreshold)) return true;
            for(unsigned shift : {0u, 8u, 16u}) {
                const int difference = static_cast<int>((colors[first] >> shift) & 255u) -
                                       static_cast<int>((colors[second] >> shift) & 255u);
                if(std::abs(difference) > static_cast<int>(settings.ColorThreshold)) return true;
            }
            return false;
        };
        executor_.ParallelFor(
            0, height,
            [&](std::size_t y) {
                for(std::uint32_t x = 0; x < width; ++x) {
                    const std::size_t index = y * width + x;
                    if(x == 0 || x + 1 == width || y == 0 || y + 1 == height) {
                        output[index] = colors[index];
                        continue;
                    }
                    const std::array<std::size_t, 4> neighbors = {index - width, index + width, index - 1, index + 1};
                    // Equal RGB values are unchanged by the five-tap average, even
                    // when depth or normals mark an edge. Alpha always stays central.
                    const std::uint32_t rgb = colors[index] & 0x00ffffffu;
                    if((colors[neighbors[0]] & 0x00ffffffu) == rgb && (colors[neighbors[1]] & 0x00ffffffu) == rgb &&
                       (colors[neighbors[2]] & 0x00ffffffu) == rgb && (colors[neighbors[3]] & 0x00ffffffu) == rgb) {
                        output[index] = colors[index];
                        continue;
                    }
                    bool edge = false;
                    for(const auto neighbor : neighbors) edge = edge || isEdge(index, neighbor);
                    if(!edge) {
                        output[index] = colors[index];
                        continue;
                    }
                    std::uint32_t result = colors[index] & 0xff000000u;
                    for(unsigned shift : {0u, 8u, 16u}) {
                        unsigned sum = (colors[index] >> shift) & 255u;
                        for(const auto neighbor : neighbors) sum += (colors[neighbor] >> shift) & 255u;
                        result |= (sum / 5u) << shift;
                    }
                    output[index] = result;
                }
            },
            rowChunkSize);
    }
}
