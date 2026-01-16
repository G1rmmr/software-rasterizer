#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "../math/SIMD.hpp"

#include "../graphics/FrameBuffer.hpp"
#include "../graphics/ParallelExecutor.hpp"

#include "Elements.hpp"

namespace shader {
    struct Post {
        Uniforms Uniform;

        std::vector<float>* DepthMap;
        std::vector<math::Vector>* NormalMap;
        std::vector<std::uint32_t>* ColorMap;

        std::vector<float> AOBuffer;
        std::vector<float> TempBuffer;

        ENGINE_INLINE math::Vector Vertex(const math::Vector& pos) const { return Uniform.Model * pos; }

        ENGINE_INLINE math::Vector Normal(const math::Vector& normal) const {
            math::Vector n = Uniform.Model * math::Vector(normal.X, normal.Y, normal.Z, 0.f);
            return n.Norm();
        }

        ENGINE_INLINE std::uint32_t Color(const math::Vector& color, const math::Vector& normal,
                                          const math::Vector& worldPos, const math::Vector& uv,
                                          const math::Vector& inTangent) const {
            std::int32_t x = static_cast<std::int32_t>(uv.X * Uniform.ScreenWidth);
            std::int32_t y = static_cast<std::int32_t>(uv.Y * Uniform.ScreenHeight);
            float ao = ComputeRawAO(x, y);

            std::int32_t idx = y * Uniform.ScreenWidth + x;
            std::uint32_t c = ColorMap->at(idx);
            std::uint8_t a = (c >> 24) & 0xFF;
            std::uint8_t r = static_cast<std::uint8_t>(((c >> 16) & 0xFF) * ao);
            std::uint8_t g = static_cast<std::uint8_t>(((c >> 8) & 0xFF) * ao);
            std::uint8_t b = static_cast<std::uint8_t>((c & 0xFF) * ao);

            return (a << 24) | (r << 16) | (g << 8) | b;
        }

        ENGINE_INLINE Varyings Process(const shader::Vertex& in) const {
            Varyings out;
            out.Pos = in.Pos;
            out.UV.X = in.Pos.X * 0.5f + 0.5f;
            out.UV.Y = 1.f - (in.Pos.Y * 0.5f + 0.5f);
            return out;
        }

        ENGINE_INLINE float ComputeRawAO(std::int32_t x, std::int32_t y) const {
            x *= 2;
            y *= 2;

            std::int32_t idx = y * static_cast<std::int32_t>(Uniform.ScreenWidth) + x;

            float rawDepth = DepthMap->at(idx);
            if(rawDepth >= 1.f - 1e-5f) return 1.f;

            float u = (x + 0.5f) / Uniform.ScreenWidth;
            float v = (y + 0.5f) / Uniform.ScreenHeight;

            math::Vector clipPos(u * 2.f - 1.f, 1.f - v * 2.f, rawDepth * 2.f - 1.f, 1.f);
            math::Vector viewPosRaw = Uniform.InvProj * clipPos;
            math::Vector viewPos = viewPosRaw / viewPosRaw.W;
            math::Vector viewNormal = NormalMap->at(idx);

            float depthScaleBias = Uniform.BiasAO * (1.f + std::abs(viewPos.Z) * 0.05f);

            std::uint32_t hash = (x * 73856093) ^ (y * 19349663) ^ (x * y * 83492791);
            float randomVal = (hash & 0xFFFF) / 65536.0f;
            float randomRot = randomVal * 6.28318f;
            float occlusion = 0.f;
            float pixelRadius = std::clamp(Uniform.RadiusAO / -viewPos.Z, 2.f, Uniform.RadiusAO);

            for(std::size_t i = 0; i < Uniform.KernelSizeAO; ++i) {
                float angle = 2.4f * i + randomRot;
                float r = std::sqrt((float)i / Uniform.KernelSizeAO) * pixelRadius;

                std::int32_t sX = x + static_cast<std::int32_t>(std::cos(angle) * r);
                std::int32_t sY = y + static_cast<std::int32_t>(std::sin(angle) * r);

                if(sX < 0 || sX >= static_cast<std::int32_t>(Uniform.ScreenWidth) || sY < 0 ||
                   sY >= static_cast<std::int32_t>(Uniform.ScreenHeight))
                    continue;

                std::int32_t sIdx = sY * static_cast<std::int32_t>(Uniform.ScreenWidth) + sX;
                float neighborRawDepth = DepthMap->at(sIdx);

                float nU = (sX + 0.5f) / Uniform.ScreenWidth;
                float nV = (sY + 0.5f) / Uniform.ScreenHeight;

                math::Vector nClip(nU * 2.f - 1.f, 1.f - nV * 2.f, neighborRawDepth * 2.f - 1.f, 1.f);
                math::Vector nPosRaw = Uniform.InvProj * nClip;
                math::Vector neighborPos = nPosRaw / nPosRaw.W;

                math::Vector vec = neighborPos - viewPos;
                float dist = vec.Length();

                float dotVal = viewNormal.Dot(vec / (dist + 1e-6f)) - Uniform.BiasAO;
                if(dist > depthScaleBias && dist < 1.f && dotVal > 0.f) occlusion += dotVal;
            }

            return std::pow(1.f - (occlusion / Uniform.KernelSizeAO), Uniform.StrengthAO);
        }

        ENGINE_INLINE void ProcessBlur(const std::int32_t y) {
            std::int32_t w = static_cast<std::int32_t>(Uniform.ScreenWidth * 0.5f);
            std::int32_t h = static_cast<std::int32_t>(Uniform.ScreenHeight * 0.5f);

            if(y < 2 || y >= h - 2) return;

            for(std::int32_t x = 2; x < w - 2; ++x) {
                std::int32_t idx = y * w + x;
                float centerDepth = DepthMap->at(idx);

                float sum = 0.f;
                float weightSum = 0.f;

                for(std::int32_t ky = -2; ky <= 2; ++ky) {
                    for(std::int32_t kx = -2; kx <= 2; ++kx) {
                        std::int32_t neighborIdx = (y + ky) * w + (x + kx);
                        float neighborDepth = DepthMap->at(neighborIdx);

                        if(std::abs(centerDepth - neighborDepth) < 0.05f) {
                            sum += AOBuffer[neighborIdx];
                            weightSum += 1.f;
                        }
                    }
                }
                TempBuffer[idx] = weightSum > 0.f ? sum / weightSum : AOBuffer[idx];
            }
        }

        ENGINE_INLINE void Composite(std::int32_t y) {
            std::int32_t w = static_cast<std::int32_t>(Uniform.ScreenWidth);
            std::int32_t lw = w * 0.5f;

            std::int32_t lh = static_cast<std::int32_t>(Uniform.ScreenHeight * 0.5f);
            float ly = y * 0.5f;

            std::int32_t y0 = static_cast<std::int32_t>(ly);
            std::int32_t y1 = std::min(y0 + 1, lh - 1);
            float fracY = ly - y0;

            for(std::int32_t x = 0; x < w; ++x) {
                float lx = x * 0.5f;
                std::int32_t x0 = static_cast<std::int32_t>(lx);
                std::int32_t x1 = std::min(x0 + 1, lw - 1);
                float fracX = lx - x0;

                float s00 = TempBuffer[y0 * lw + x0];
                float s10 = TempBuffer[y0 * lw + x1];
                float s01 = TempBuffer[y1 * lw + x0];
                float s11 = TempBuffer[y1 * lw + x1];

                float tx0 = s00 + (s10 - s00) * fracX;
                float tx1 = s01 + (s11 - s01) * fracX;

                float ao = tx0 + (tx1 - tx0) * fracY;

                if(ao == 0.f && (x < 4 || x >= w - 4)) ao = 1.f;

                std::int32_t idx = y * w + x;
                std::uint32_t c = ColorMap->at(idx);

                std::uint8_t a = (c >> 24) & 0xFF;
                std::uint8_t r = static_cast<std::uint8_t>(((c >> 16) & 0xFF) * ao);
                std::uint8_t g = static_cast<std::uint8_t>(((c >> 8) & 0xFF) * ao);
                std::uint8_t b = static_cast<std::uint8_t>((c & 0xFF) * ao);
                ColorMap->at(idx) = (a << 24) | (r << 16) | (g << 8) | b;
            }
        }
    };
}