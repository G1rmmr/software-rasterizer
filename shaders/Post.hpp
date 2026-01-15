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
            return ProcessSSAO(x, y);
        }

        ENGINE_INLINE Varyings Process(const shader::Vertex& in) const {
            Varyings out;
            out.Pos = in.Pos;
            out.UV.X = in.Pos.X * 0.5f + 0.5f;
            out.UV.Y = 1.f - (in.Pos.Y * 0.5f + 0.5f);
            return out;
        }

        ENGINE_INLINE std::uint32_t ProcessSSAO(const std::int32_t x, const std::int32_t y,
                                                const bool shouldDebug = false) const {
            std::int32_t idx = y * static_cast<std::int32_t>(Uniform.ScreenWidth) + x;

            float rawDepth = DepthMap->at(idx);
            if(rawDepth >= 1.f - 1e-5f) return ColorMap->at(idx);

            float u = (x + 0.5f) / Uniform.ScreenWidth;
            float v = (y + 0.5f) / Uniform.ScreenHeight;

            math::Vector clipPos(u * 2.f - 1.f, 1.f - v * 2.f, rawDepth * 2.f - 1.f, 1.f);
            math::Vector viewPosRaw = Uniform.InvProj * clipPos;
            math::Vector viewPos = viewPosRaw / viewPosRaw.W;

            math::Vector viewNormal = NormalMap->at(idx);

            float depthScaleBias = Uniform.BiasAO * (1.0f + std::abs(viewPos.Z) * 0.05f);

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

            float ao = std::pow(1.f - (occlusion / Uniform.KernelSizeAO), Uniform.StrengthAO);

            if(shouldDebug) {
                std::uint8_t aoByte = static_cast<std::uint8_t>(ao * 255.f);
                return (0xFF << 24) | (aoByte << 16) | (aoByte << 8) | aoByte;
            }

            std::uint32_t c = (*ColorMap)[idx];
            std::uint8_t a = (c >> 24) & 0xFF;
            std::uint8_t r = static_cast<std::uint8_t>(((c >> 16) & 0xFF) * ao);
            std::uint8_t g = static_cast<std::uint8_t>(((c >> 8) & 0xFF) * ao);
            std::uint8_t b = static_cast<std::uint8_t>((c & 0xFF) * ao);

            return (a << 24) | (r << 16) | (g << 8) | b;
        }
    };
}