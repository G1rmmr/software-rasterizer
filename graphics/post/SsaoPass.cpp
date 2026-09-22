#include "SsaoPass.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>

namespace graphics {
    namespace {
        constexpr std::size_t RowChunkSize = 4;
        constexpr auto BlurWeights = [] {
            std::array<std::array<float, 5>, 5> weights{};
            for(int y = -2; y <= 2; ++y)
                for(int x = -2; x <= 2; ++x) weights[y + 2][x + 2] = 1.f / static_cast<float>(1 + x * x + y * y);
            return weights;
        }();

        bool HasOcclusionNear(const std::uint64_t* row, std::uint32_t x, std::uint32_t width) noexcept {
            const std::uint32_t left = x > 2u ? x - 2u : 0u;
            const std::uint32_t right = std::min(x + 2u, width - 1u);
            constexpr std::uint64_t allBits = ~std::uint64_t{0};
            const std::uint64_t leftMask = allBits << (left & 63u);
            const std::uint64_t rightMask = allBits >> (63u - (right & 63u));
            const auto firstWord = left / 64u, lastWord = right / 64u;
            if(firstWord == lastWord) return (row[firstWord] & leftMask & rightMask) != 0;
            // A five-sample window spans at most two words. Neither shift can be 64.
            return ((row[firstWord] & leftMask) | (row[lastWord] & rightMask)) != 0;
        }

        struct ReconstructionView {
            std::span<const math::Vector> X;
            std::span<const math::Vector> Y;
            const math::Matrix& InverseProjection;
        };

        bool Reconstruct(std::uint32_t x, std::uint32_t y, float depth, const ReconstructionView& reconstruction,
                         math::Vector& result) {
            if(!std::isfinite(depth) || depth < 0.f || depth >= 1.f) return false;
            // Cache only the pixel-coordinate terms. Preserve the original matrix
            // product's addition order and zero-to-one depth contract.
            auto homogeneous = simd::Add(reconstruction.X[x].V, reconstruction.Y[y].V);
            homogeneous = simd::Add(homogeneous, simd::Mul(reconstruction.InverseProjection.Cols[2], simd::Set(depth)));
            homogeneous = simd::Add(homogeneous, reconstruction.InverseProjection.Cols[3]);
            const math::Vector position(homogeneous);
            if(!std::isfinite(position.W) || std::abs(position.W) < 1e-8f) return false;
            result = position / position.W;
            return std::isfinite(result.X) && std::isfinite(result.Y) && std::isfinite(result.Z);
        }

        bool IsValid(const math::Vector& position) {
            return std::isfinite(position.Z);
        }

        std::uint32_t ApplyVisibility(std::uint32_t color, float visibility) {
            std::uint32_t result = color & 0xff000000u;
            for(unsigned shift : {0u, 8u, 16u})
                result |= static_cast<std::uint32_t>(static_cast<float>((color >> shift) & 255u) * visibility) << shift;
            return result;
        }
    }

    void SsaoPass::PrepareKernel(std::size_t count) {
        if(kernel_.size() == count) return;
        kernel_.resize(count);
        for(std::size_t i = 0; i < count; ++i) {
            const float fraction = (static_cast<float>(i) + .5f) / static_cast<float>(count);
            const float angle = static_cast<float>(i) * 2.39996323f;
            const float radial = std::sqrt(fraction);
            const float scale = .1f + .9f * fraction * fraction;
            kernel_[i] =
                math::Vector(std::cos(angle) * radial, std::sin(angle) * radial, std::sqrt(1.f - fraction)) * scale;
        }
    }

    void SsaoPass::PrepareReconstruction(std::uint32_t width, std::uint32_t height,
                                         const math::Matrix& inverseProjection) {
        bool changed = !reconstructionReady_ || reconstructionX_.size() != width || reconstructionY_.size() != height;
        for(int column = 0; column < 4; ++column)
            for(int row = 0; row < 4; ++row)
                changed = changed || cachedInverseProjection_[column][row] != inverseProjection[column][row];
        if(!changed) return;
        reconstructionX_.resize(width);
        reconstructionY_.resize(height);
        for(std::uint32_t x = 0; x < width; ++x) {
            const float u = (static_cast<float>(x) + .5f) / static_cast<float>(width);
            reconstructionX_[x] = math::Vector(simd::Mul(inverseProjection.Cols[0], simd::Set(u * 2.f - 1.f)));
        }
        for(std::uint32_t y = 0; y < height; ++y) {
            const float v = (static_cast<float>(y) + .5f) / static_cast<float>(height);
            reconstructionY_[y] = math::Vector(simd::Mul(inverseProjection.Cols[1], simd::Set(1.f - v * 2.f)));
        }
        cachedInverseProjection_ = inverseProjection;
        reconstructionReady_ = true;
    }

    void SsaoPass::PrepareRotations(std::uint32_t width, std::uint32_t height) {
        if(rotationWidth_ == width && rotationHeight_ == height) return;
        rotations_.resize(static_cast<std::size_t>(width) * height);
        executor_.ParallelFor(
            0, height,
            [&](std::size_t y) {
                for(std::uint32_t x = 0; x < width; ++x) {
                    const std::uint32_t hash = (x * 73856093u) ^ (static_cast<std::uint32_t>(y) * 19349663u);
                    const float angle =
                        static_cast<float>(hash & 0xffffu) * (2.f * std::numbers::pi_v<float> / 65536.f);
                    rotations_[y * width + x] = {std::cos(angle), std::sin(angle)};
                }
            },
            RowChunkSize);
        rotationWidth_ = width;
        rotationHeight_ = height;
    }

    void SsaoPass::Execute(FrameBuffer& frame, const math::Matrix& projection, const math::Matrix& inverseProjection,
                           const SsaoSettings& settings) {
        if(!std::isfinite(settings.Radius) || settings.Radius <= 0.f || !std::isfinite(settings.Bias) ||
           settings.Bias < 0.f || settings.Bias >= settings.Radius || !std::isfinite(settings.Strength) ||
           settings.Strength < 0.f || settings.KernelSize == 0 || settings.KernelSize > 256)
            throw std::invalid_argument("SSAO requires radius > bias >= 0, strength >= 0 and 1..256 samples");
        for(int column = 0; column < 4; ++column)
            for(int row = 0; row < 4; ++row)
                if(!std::isfinite(projection[column][row]) || !std::isfinite(inverseProjection[column][row]))
                    throw std::invalid_argument("SSAO projection matrices must be finite");
        if(settings.Strength == 0.f) return;

        const auto width = frame.GetWidth(), height = frame.GetHeight();
        const std::uint32_t lowWidth = (width + 1u) / 2u, lowHeight = (height + 1u) / 2u;
        const std::size_t count = static_cast<std::size_t>(lowWidth) * lowHeight;
        positions_.resize(count);
        normals_.resize(count);
        visibility_.resize(count);
        blurred_.resize(count);
        const std::size_t wordsPerRow = (static_cast<std::size_t>(lowWidth) + 63u) / 64u;
        occlusionWords_.resize(wordsPerRow * lowHeight);
        blurWords_.resize(wordsPerRow * lowHeight);
        PrepareKernel(settings.KernelSize);
        PrepareReconstruction(width, height, inverseProjection);
        PrepareRotations(lowWidth, lowHeight);
        const ReconstructionView reconstruction{reconstructionX_, reconstructionY_, inverseProjection};
        const auto depths = frame.GetDepths();
        const auto frameNormals = frame.GetNormals();
        const float invalid = std::numeric_limits<float>::infinity();

        executor_.ParallelFor(
            0, lowHeight,
            [&](std::size_t y) {
                auto* rowWords = occlusionWords_.data() + y * wordsPerRow;
                std::fill_n(rowWords, wordsPerRow, std::uint64_t{0});
                for(std::uint32_t x = 0; x < lowWidth; ++x) {
                    const std::size_t index = y * lowWidth + x;
                    const std::uint32_t sourceX = x * 2u, sourceY = static_cast<std::uint32_t>(y) * 2u;
                    const std::size_t source = static_cast<std::size_t>(sourceY) * width + sourceX;
                    auto& position = positions_[index];
                    auto& normal = normals_[index];
                    visibility_[index] = 1.f;
                    const float sourceDepth = depths[source];
                    if(!std::isfinite(sourceDepth) || sourceDepth < 0.f || sourceDepth >= 1.f) {
                        position = math::Vector(0.f, 0.f, invalid);
                        normal = math::Vector{};
                        continue;
                    }
                    normal = frameNormals[source].NormalizedDirection();
                    if(normal.Dot(normal) < .5f ||
                       !Reconstruct(sourceX, sourceY, sourceDepth, reconstruction, position)) {
                        position = math::Vector(0.f, 0.f, invalid);
                        continue;
                    }
                    // Center attachments and raw AO are independent per sample; only
                    // the subsequent blur needs all centers to have been published.
                    const auto up =
                        std::abs(normal.Z) < .9f ? math::Vector(0.f, 0.f, 1.f) : math::Vector(0.f, 1.f, 0.f);
                    const auto tangent = up.Cross(normal).NormalizedDirection();
                    const auto bitangent = normal.Cross(tangent);
                    const float cosine = rotations_[index].Cosine, sine = rotations_[index].Sine;
                    float occlusion = 0.f;
                    for(const auto& sample : kernel_) {
                        const float sx = sample.X * cosine - sample.Y * sine;
                        const float sy = sample.X * sine + sample.Y * cosine;
                        const auto samplePosition =
                            position + (tangent * sx + bitangent * sy + normal * sample.Z) * settings.Radius;
                        const auto clip =
                            projection * math::Vector(samplePosition.X, samplePosition.Y, samplePosition.Z, 1.f);
                        if(!std::isfinite(clip.W) || clip.W <= 1e-8f) continue;
                        const float u = (clip.X / clip.W) * .5f + .5f;
                        const float v = .5f - (clip.Y / clip.W) * .5f;
                        if(!(u >= 0.f && u < 1.f && v >= 0.f && v < 1.f)) continue;
                        const auto pixelX =
                            std::min(static_cast<std::uint32_t>(u * static_cast<float>(width)), width - 1u);
                        const auto pixelY =
                            std::min(static_cast<std::uint32_t>(v * static_cast<float>(height)), height - 1u);
                        const std::size_t neighborIndex = static_cast<std::size_t>(pixelY) * width + pixelX;
                        if(neighborIndex == source) continue;
                        math::Vector neighbor;
                        if(!Reconstruct(pixelX, pixelY, depths[neighborIndex], reconstruction, neighbor)) continue;
                        const auto delta = neighbor - position;
                        const float heightAboveSurface = normal.Dot(delta);
                        // A neighbor outside the biased hemisphere contributes exactly
                        // zero, so avoid its square root and divisions.
                        if(heightAboveSurface <= settings.Bias) continue;
                        const float distance = delta.Length();
                        if(!std::isfinite(distance) || distance <= settings.Bias || distance >= settings.Radius)
                            continue;
                        const float facing = std::max(heightAboveSurface - settings.Bias, 0.f) / distance;
                        occlusion += facing * (1.f - distance / settings.Radius);
                    }
                    if(occlusion != 0.f)
                        visibility_[index] =
                            std::pow(std::clamp(1.f - occlusion / static_cast<float>(kernel_.size()), 0.f, 1.f),
                                     settings.Strength);
                    if(visibility_[index] != 1.f) rowWords[x / 64u] |= std::uint64_t{1} << (x & 63u);
                }
            },
            RowChunkSize);

        const float depthTolerance = std::max(settings.Bias, settings.Radius * .1f);
        const auto sameSurface = [&](const math::Vector& position, const math::Vector& normal, std::size_t candidate) {
            return IsValid(positions_[candidate]) && std::abs(position.Z - positions_[candidate].Z) <= depthTolerance &&
                   normal.Dot(normals_[candidate]) >= .8f;
        };
        executor_.ParallelFor(
            0, lowHeight,
            [&](std::size_t y) {
                auto* rowWords = blurWords_.data() + y * wordsPerRow;
                std::fill_n(rowWords, wordsPerRow, std::uint64_t{0});
                const std::size_t firstRow = y > 2u ? y - 2u : 0u;
                const std::size_t lastRow = std::min(y + 2u, static_cast<std::size_t>(lowHeight) - 1u);
                for(std::size_t sourceRow = firstRow; sourceRow <= lastRow; ++sourceRow)
                    for(std::size_t word = 0; word < wordsPerRow; ++word)
                        rowWords[word] |= occlusionWords_[sourceRow * wordsPerRow + word];
                std::uint64_t anyOcclusion = 0;
                for(std::size_t word = 0; word < wordsPerRow; ++word) anyOcclusion |= rowWords[word];
                if(anyOcclusion == 0) {
                    std::fill_n(blurred_.data() + y * lowWidth, lowWidth, 1.f);
                    return;
                }
                for(std::uint32_t x = 0; x < lowWidth; ++x) {
                    const std::size_t index = y * lowWidth + x;
                    // With only visibility==1 samples, every accepted weight is added
                    // identically to sum and weightSum, so the original result is exactly 1.
                    if(!HasOcclusionNear(rowWords, x, lowWidth)) {
                        blurred_[index] = 1.f;
                        continue;
                    }
                    float sum = 0.f, weights = 0.f;
                    if(IsValid(positions_[index])) {
                        for(int oy = -2; oy <= 2; ++oy) {
                            const int sy = static_cast<int>(y) + oy;
                            if(sy < 0 || sy >= static_cast<int>(lowHeight)) continue;
                            for(int ox = -2; ox <= 2; ++ox) {
                                const int sx = static_cast<int>(x) + ox;
                                if(sx < 0 || sx >= static_cast<int>(lowWidth)) continue;
                                const std::size_t neighbor = static_cast<std::size_t>(sy) * lowWidth + sx;
                                if(!sameSurface(positions_[index], normals_[index], neighbor)) continue;
                                const float weight = BlurWeights[oy + 2][ox + 2];
                                sum += visibility_[neighbor] * weight;
                                weights += weight;
                            }
                        }
                    }
                    blurred_[index] = weights > 0.f ? sum / weights : visibility_[index];
                }
            },
            RowChunkSize);

        auto colors = frame.Colors();
        executor_.ParallelFor(
            0, height,
            [&](std::size_t y) {
                for(std::uint32_t x = 0; x < width; ++x) {
                    const std::size_t index = y * width + x;
                    // Every even/even output pixel is exactly its low-resolution sample.
                    if((x & 1u) == 0 && (y & 1u) == 0) {
                        colors[index] = ApplyVisibility(colors[index],
                                                        std::clamp(blurred_[(y / 2u) * lowWidth + x / 2u], 0.f, 1.f));
                        continue;
                    }
                    math::Vector position;
                    if(!Reconstruct(x, static_cast<std::uint32_t>(y), depths[index], reconstruction, position))
                        continue;
                    const auto normal = frameNormals[index].NormalizedDirection();
                    const std::uint32_t x0 = x / 2u, y0 = static_cast<std::uint32_t>(y) / 2u;
                    const std::uint32_t x1 = std::min(x0 + 1u, lowWidth - 1u), y1 = std::min(y0 + 1u, lowHeight - 1u);
                    const float fx = (x & 1u) ? .5f : 0.f, fy = (y & 1u) ? .5f : 0.f;
                    const std::array<std::size_t, 4> candidates = {
                        static_cast<std::size_t>(y0) * lowWidth + x0, static_cast<std::size_t>(y0) * lowWidth + x1,
                        static_cast<std::size_t>(y1) * lowWidth + x0, static_cast<std::size_t>(y1) * lowWidth + x1};
                    const std::array<float, 4> weights = {(1.f - fx) * (1.f - fy), fx * (1.f - fy), (1.f - fx) * fy,
                                                          fx * fy};
                    float sum = 0.f, weightSum = 0.f;
                    for(std::size_t i = 0; i < candidates.size(); ++i) {
                        if(weights[i] == 0.f || !sameSurface(position, normal, candidates[i])) continue;
                        sum += blurred_[candidates[i]] * weights[i];
                        weightSum += weights[i];
                    }
                    const float visibility = weightSum > 0.f ? std::clamp(sum / weightSum, 0.f, 1.f) : 1.f;
                    colors[index] = ApplyVisibility(colors[index], visibility);
                }
            },
            RowChunkSize);
    }
}
