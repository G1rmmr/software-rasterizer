#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "../math/Math.hpp"
#include "FrameBuffer.hpp"
#include "ParallelExecutor.hpp"
#include "Shader.hpp"

namespace graphics {
    enum class PrimitiveType { Points, Lines, Triangles };

    constexpr std::int32_t AO_KERNEL_SIZE = 16;
    constexpr float AO_RADIUS = 15.f;
    constexpr float AO_BIAS = 0.02f;
    constexpr float AO_STRENGTH = 1.2f;

    class Rasterizer {
    public:
        explicit Rasterizer(FrameBuffer& frame)
            : frame(frame),
              width(static_cast<float>(frame.GetWidth())),
              height(static_cast<float>(frame.GetHeight())) {}

        template <typename Shader>
        ENGINE_INLINE void ENGINE_VECTORCALL Render(const Shader& shader, const std::vector<shader::Vertex>& vertices,
                                                    const PrimitiveType type = PrimitiveType::Triangles) {
            std::vector<std::uint32_t> indices(vertices.size());
            for(size_t i = 0; i < vertices.size(); ++i) indices[i] = i;

            Render(shader, vertices, indices, type);
        }

        template <typename Shader>
        ENGINE_INLINE void ENGINE_VECTORCALL Render(const Shader& shader, const std::vector<shader::Vertex>& vertices,
                                                    const std::vector<std::uint32_t>& indices, const float near,
                                                    const PrimitiveType type = PrimitiveType::Triangles,
                                                    std::size_t maxIndices = 0) {
            if(tileGrid.empty()) initTiles();
            for(auto& tile : tileGrid) tile.indices.clear();

            std::size_t actualLimit = (maxIndices == 0 || maxIndices > indices.size()) ? indices.size() : maxIndices;
            std::vector<std::uint32_t> limitedIndices(indices.begin(), indices.begin() + actualLimit);

            std::vector<shader::Varyings> screenVertices =
                processVertices<Shader>(shader, vertices, limitedIndices, near);

            dispatchPrimitives<Shader>(shader, screenVertices, limitedIndices, type);
        }

        // AA
        ENGINE_INLINE void ApplyPostAA() {
            std::uint32_t* __restrict src = frame.GetColors().data();
            std::vector<std::uint32_t> temp = frame.GetColors();
            std::uint32_t* __restrict dst = temp.data();

            const std::uint32_t w = frame.GetWidth();
            const std::uint32_t h = frame.GetHeight();
            const int THRESHOLD = 30;

            ParallelExecutor::GetInstance().ParallelFor(
                1, h - 1,
                [=](std::size_t y) {
                    for(std::uint32_t x = 1; x < w - 1; ++x) {
                        std::uint32_t idx = y * w + x;
                        std::uint32_t current = src[idx];

                        std::uint32_t up = src[idx - w];
                        std::uint32_t down = src[idx + w];
                        std::uint32_t left = src[idx - 1];
                        std::uint32_t right = src[idx + 1];

                        int diff = colorDiff(current, up) + colorDiff(current, down) + colorDiff(current, left) +
                                   colorDiff(current, right);

                        if(diff > THRESHOLD) dst[idx] = mixColors(current, up, down, left, right);
                    }
                },
                16);
            frame.UpdateBuffer(temp);
        }

        // SSAO
        ENGINE_INLINE void ApplySSAO(const shader::Uniforms& uniform, const float near, const float far) {
            std::uint32_t w = frame.GetWidth();
            std::uint32_t h = frame.GetHeight();

            const float* __restrict depthBuffer = frame.GetDepth();
            std::uint32_t* __restrict colorBuffer = frame.GetColor();
            std::vector<std::uint32_t> temp = frame.GetColors();
            std::uint32_t* __restrict dst = temp.data();

            math::Matrix invProj = uniform.Proj.Inv();

            static std::vector<std::pair<float, float>> spiralKernel;
            if(spiralKernel.empty()) {
                spiralKernel.resize(AO_KERNEL_SIZE);
                for(int i = 0; i < AO_KERNEL_SIZE; ++i) {
                    float angle = 2.4f * i;
                    float radius = std::sqrt((float)i / AO_KERNEL_SIZE);
                    spiralKernel[i] = {std::cos(angle) * radius, std::sin(angle) * radius};
                }
            }

            static std::vector<float> randomAngles;
            if(randomAngles.empty()) {
                randomAngles.resize(16);
                for(float& ang : randomAngles) ang = math::RandomFloat(0.f, 6.28f);
            }

            ParallelExecutor::GetInstance().ParallelFor(
                0, h,
                [&](std::int32_t y) {
                    for(std::int32_t x = 0; x < w; ++x) {
                        std::int32_t idx = y * w + x;
                        float currentDepthRaw = depthBuffer[idx];

                        if(currentDepthRaw >= 1.f - 1e-5f) {
                            dst[idx] = colorBuffer[idx];
                            continue;
                        }

                        float uvX = (x + 0.5f) / static_cast<float>(w);
                        float uvY = (y + 0.5f) / static_cast<float>(h);

                        math::Vector clipPos(uvX * 2.f - 1.f, 1.f - uvY * 2.f, currentDepthRaw * 2.f - 1.f, 1.f);
                        math::Vector viewPosRaw = invProj * clipPos;
                        math::Vector viewPos = viewPosRaw / viewPosRaw.W;

                        float depth = -viewPos.Z;

                        float pixelRadius = (AO_RADIUS / depth);
                        pixelRadius = std::clamp(pixelRadius, 2.f, AO_RADIUS);

                        int noiseIdx = ((y % 4) * 4) + (x % 4);
                        float rotAngle = randomAngles[noiseIdx];
                        float cosA = std::cos(rotAngle);
                        float sinA = std::sin(rotAngle);

                        float occlusion = 0.f;

                        for(const auto& k : spiralKernel) {
                            float rotatedX = k.first * cosA - k.second * sinA;
                            float rotatedY = k.first * sinA + k.second * cosA;

                            int sX = x + static_cast<int>(rotatedX * pixelRadius);
                            int sY = y + static_cast<int>(rotatedY * pixelRadius);

                            if(sX < 0 || sX >= (int)w || sY < 0 || sY >= (int)h) continue;

                            float neighborDepthRaw = depthBuffer[sY * w + sX];

                            float nUvX = (sX + 0.5f) / static_cast<float>(w);
                            float nUvY = (sY + 0.5f) / static_cast<float>(h);

                            math::Vector nClipPos(nUvX * 2.f - 1.f, 1.f - nUvY * 2.f, neighborDepthRaw * 2.f - 1.f,
                                                  1.f);
                            math::Vector nViewPosRaw = invProj * nClipPos;
                            float neighborDepth = -(nViewPosRaw.Z / nViewPosRaw.W);

                            float rangeCheck = std::abs(depth - neighborDepth) < 1.f ? 1.f : 0.f;

                            if(neighborDepth < depth - AO_BIAS) occlusion += 1.f * rangeCheck;
                        }

                        float ao = 1.f - (occlusion / AO_KERNEL_SIZE);
                        ao = std::pow(ao, AO_STRENGTH);

                        std::uint32_t c = colorBuffer[idx];
                        std::uint8_t alpha = (c >> 24) & 0xFF;
                        std::uint8_t r = static_cast<std::uint8_t>(((c >> 16) & 0xFF) * ao);
                        std::uint8_t g = static_cast<std::uint8_t>(((c >> 8) & 0xFF) * ao);
                        std::uint8_t b = static_cast<std::uint8_t>((c & 0xFF) * ao);

                        dst[idx] = (alpha << 24) | (r << 16) | (g << 8) | b;

                        std::uint8_t aoByte = static_cast<std::uint8_t>(occlusion * 255.f);
                        // dst[idx] = (0xFF << 24) | (aoByte << 16) | (aoByte << 8) | aoByte;
                    }
                },
                4);
            frame.UpdateBuffer(temp);
        }

    private:
        static constexpr std::uint8_t TILE_SIZE = 32;

        struct Tile {
            std::vector<std::uint32_t> indices;
            SpinLock mutex;
        };

        std::vector<Tile> tileGrid;
        FrameBuffer& frame;

        std::int32_t tilesX;
        std::int32_t tilesY;

        float width;
        float height;

        ENGINE_INLINE void initTiles() {
            tilesX = (static_cast<std::int32_t>(width) + TILE_SIZE - 1) / TILE_SIZE;
            tilesY = (static_cast<std::int32_t>(height) + TILE_SIZE - 1) / TILE_SIZE;

            tileGrid.clear();
            tileGrid.reserve(tilesX * tilesY);
            for(int i = 0; i < tilesX * tilesY; ++i) tileGrid.emplace_back();
        }

        ENGINE_INLINE void perspectiveDivide(shader::Varyings& v) {
            float w = v.Pos.W;
            if(std::abs(w) < 1e-6f) w = 1e-6f;

            float rhw = 1.f / w;
            v.Pos.X = (v.Pos.X * rhw + 1.f) * width * 0.5f;
            v.Pos.Y = (1.f - v.Pos.Y * rhw) * height * 0.5f;
            v.Pos.Z = v.Pos.Z * rhw;
            v.Pos.W = rhw;
            v.RecipW = rhw;
        }

        ENGINE_INLINE void clipAndPerspDivide(const shader::Varyings& v0, const shader::Varyings& v1,
                                              const shader::Varyings& v2, const float near,
                                              std::vector<shader::Varyings>& outList,
                                              std::vector<std::uint32_t>& outIndices) {
            const float W_CLIPPING_PLANE = near;

            auto isInside = [&](const math::Vector& p) {
                return p.W >= W_CLIPPING_PLANE;
            };

            auto getIntersectRatio = [&](const math::Vector& p1, const math::Vector& p2) {
                float dist1 = p1.W - W_CLIPPING_PLANE;
                float dist2 = p2.W - W_CLIPPING_PLANE;
                return dist1 / (dist1 - dist2);
            };

            const shader::Varyings* inV[3] = {&v0, &v1, &v2};

            shader::Varyings tempVerts[4];
            int tempCount = 0;

            // Sutherland-Hodgman
            for(int i = 0; i < 3; ++i) {
                const shader::Varyings& curr = *inV[i];
                const shader::Varyings& prev = *inV[(i + 3 - 1) % 3];

                bool currInside = isInside(curr.Pos);
                bool prevInside = isInside(prev.Pos);

                if(currInside && prevInside) {
                    tempVerts[tempCount++] = curr;
                }
                else if(currInside && !prevInside) {
                    float t = getIntersectRatio(prev.Pos, curr.Pos);
                    tempVerts[tempCount++] = shader::Varyings::Lerp(prev, curr, t);
                    tempVerts[tempCount++] = curr;
                }
                else if(!currInside && prevInside) {
                    float t = getIntersectRatio(prev.Pos, curr.Pos);
                    tempVerts[tempCount++] = shader::Varyings::Lerp(prev, curr, t);
                }
            }

            if(tempCount < 3) return;

            for(int i = 0; i < tempCount; ++i) perspectiveDivide(tempVerts[i]);

            std::uint32_t baseIdx = static_cast<std::uint32_t>(outList.size());

            outList.push_back(tempVerts[0]);
            outList.push_back(tempVerts[1]);
            outList.push_back(tempVerts[2]);
            outIndices.push_back(baseIdx);
            outIndices.push_back(baseIdx + 1);
            outIndices.push_back(baseIdx + 2);

            if(tempCount == 4) {
                outList.push_back(tempVerts[3]);
                outList.push_back(tempVerts[0]);
                outList.push_back(tempVerts[2]);

                outIndices.push_back(baseIdx + 3);
                outIndices.push_back(baseIdx + 4);
                outIndices.push_back(baseIdx + 5);
            }
        }

        // vertex Shader -> rasterization -> pixel Shader
        template <typename Shader>
        ENGINE_INLINE std::vector<shader::Varyings>
            ENGINE_VECTORCALL processVertices(const Shader& shader, const std::vector<shader::Vertex>& in,
                                              std::vector<std::uint32_t>& indices, const float near) {
            const std::size_t inSize = in.size();
            std::vector<shader::Varyings> out(inSize);

            std::vector<shader::Varyings> finalVaryings;
            std::vector<std::uint32_t> finalIndices;
            finalVaryings.reserve(indices.size());
            finalIndices.reserve(indices.size());

            ParallelExecutor::GetInstance().ParallelFor(
                0, inSize, [&](std::size_t i) { out[i] = shader.Process(in[i]); }, 1);

            std::size_t numTriangles = indices.size() / 3;
            for(size_t i = 0; i < numTriangles; ++i) {
                std::uint32_t i0 = indices[i * 3];
                std::uint32_t i1 = indices[i * 3 + 1];
                std::uint32_t i2 = indices[i * 3 + 2];

                clipAndPerspDivide(out[i0], out[i1], out[i2], near, finalVaryings, finalIndices);
            }

            indices = finalIndices;
            return finalVaryings;
        }

        template <typename Shader>
        ENGINE_INLINE void ENGINE_VECTORCALL dispatchPrimitives(const Shader& shader,
                                                                const std::vector<shader::Varyings>& varyings,
                                                                const std::vector<std::uint32_t>& indices,
                                                                const PrimitiveType type = PrimitiveType::Triangles) {
            switch(type) {
            case PrimitiveType::Points: {
                const std::size_t pointCount = indices.size();

                ParallelExecutor::GetInstance().ParallelFor(
                    0, pointCount,
                    [&](std::size_t i) {
                        if(indices[i] >= varyings.size()) return;
                        drawPoint(shader, varyings[indices[i]]);
                    },
                    256);
                return;
            }
            case PrimitiveType::Lines: {
                const std::size_t lineCount = indices.size() / 2;

                ParallelExecutor::GetInstance().ParallelFor(
                    0, lineCount,
                    [&](std::size_t i) {
                        const std::size_t idx = i * 3;
                        if(idx + 2 >= indices.size()) return;

                        const shader::Varyings& v0 = varyings[indices[idx]];
                        const shader::Varyings& v1 = varyings[indices[idx + 1]];
                        const shader::Varyings& v2 = varyings[indices[idx + 2]];

                        drawLine(shader, v0, v1);
                        drawLine(shader, v1, v2);
                        drawLine(shader, v2, v0);
                    },
                    256);
                return;
            }
            default: {
                const std::size_t triangleCount = indices.size() / 3;

                // Binning
                ParallelExecutor::GetInstance().ParallelFor(
                    0, triangleCount,
                    [&](std::size_t i) {
                        std::size_t idx = i * 3;
                        if(idx + 2 >= indices.size()) return;

                        const shader::Varyings& v0 = varyings[indices[idx]];
                        const shader::Varyings& v1 = varyings[indices[idx + 1]];
                        const shader::Varyings& v2 = varyings[indices[idx + 2]];

                        const std::int32_t minX = static_cast<std::int32_t>(std::min({v0.Pos.X, v1.Pos.X, v2.Pos.X}));
                        const std::int32_t minY = static_cast<std::int32_t>(std::min({v0.Pos.Y, v1.Pos.Y, v2.Pos.Y}));
                        const std::int32_t maxX = static_cast<std::int32_t>(std::max({v0.Pos.X, v1.Pos.X, v2.Pos.X}));
                        const std::int32_t maxY = static_cast<std::int32_t>(std::max({v0.Pos.Y, v1.Pos.Y, v2.Pos.Y}));

                        if(minX > width || maxX < 0 || minY > height || maxY < 0) return;

                        const std::int32_t startTX = std::max(0, static_cast<std::int32_t>(minX) / TILE_SIZE);
                        const std::int32_t endTX = std::min(tilesX - 1, static_cast<std::int32_t>(maxX) / TILE_SIZE);
                        const std::int32_t startTY = std::max(0, static_cast<std::int32_t>(minY) / TILE_SIZE);
                        const std::int32_t endTY = std::min(tilesY - 1, static_cast<std::int32_t>(maxY) / TILE_SIZE);

                        for(std::int32_t ty = startTY; ty <= endTY; ++ty) {
                            for(std::int32_t tx = startTX; tx <= endTX; ++tx) {
                                Tile& tile = tileGrid[ty * tilesX + tx];

                                std::lock_guard<SpinLock> lock(tile.mutex);
                                tile.indices.push_back(static_cast<uint32_t>(i));
                            }
                        }
                    },
                    128);

                // Tiling
                ParallelExecutor::GetInstance().ParallelFor(
                    0, tileGrid.size(),
                    [&](std::size_t tileIdx) {
                        Tile& tile = tileGrid[tileIdx];
                        if(tile.indices.empty()) return;

                        std::size_t tx = tileIdx % tilesX;
                        std::size_t ty = tileIdx / tilesX;

                        BoundingBox tileClip;
                        tileClip.MinX = tx * TILE_SIZE;
                        tileClip.MinY = ty * TILE_SIZE;
                        tileClip.MaxX = std::min((std::size_t)width - 1, (tx + 1) * TILE_SIZE - 1);
                        tileClip.MaxY = std::min((std::size_t)height - 1, (ty + 1) * TILE_SIZE - 1);

                        for(const std::uint32_t triID : tile.indices) {
                            std::size_t idx = triID * 3;
                            const shader::Varyings& v0 = varyings[indices[idx]];
                            const shader::Varyings& v1 = varyings[indices[idx + 1]];
                            const shader::Varyings& v2 = varyings[indices[idx + 2]];

                            drawTriangle(shader, v0, v1, v2, tileClip);
                        }
                    },
                    1);
                return;
            }
            }
        }

        template <typename Shader>
        ENGINE_INLINE void ENGINE_VECTORCALL drawPoint(const Shader& shader, const shader::Varyings& v) {
            std::int32_t x = static_cast<std::int32_t>(std::round(v.Pos.X));
            std::int32_t y = static_cast<std::int32_t>(std::round(v.Pos.Y));

            if(frame.TestDepth(x, y, v.Pos.Z)) {
                simd::Floats colorV = v.Color.V;

                colorV = simd::Mul(colorV, simd::Set(255.f));
                frame.SetPixel(x, y, simd::PackRGBA(colorV));
            }
        }

        // Bresenham's Line Algorithm
        template <typename Shader>
        ENGINE_INLINE void ENGINE_VECTORCALL drawLine(const Shader& shader, const shader::Varyings& v0,
                                                      const shader::Varyings& v1) {
            int x0 = static_cast<int>(std::round(v0.Pos.X));
            int y0 = static_cast<int>(std::round(v0.Pos.Y));
            int x1 = static_cast<int>(std::round(v1.Pos.X));
            int y1 = static_cast<int>(std::round(v1.Pos.Y));

            int dx = std::abs(x1 - x0);
            int dy = std::abs(y1 - y0);
            int sx = (x0 < x1) ? 1 : -1;
            int sy = (y0 < y1) ? 1 : -1;
            int err = dx - dy;

            float totalDist = std::sqrt(static_cast<float>(dx * dx + dy * dy));
            int startX = x0, startY = y0;

            while(true) {
                float t = (totalDist < 1e-6f)
                              ? 0.f
                              : std::sqrt(std::pow(x0 - startX, 2) + std::pow(y0 - startY, 2)) / totalDist;

                float z = v0.Pos.Z * (1.f - t) + v1.Pos.Z * t;
                math::Vector color = v0.Color * (1.f - t) + v1.Color * t;

                if(frame.TestDepth(x0, y0, z)) {
                    simd::Floats colorV = simd::Mul(color.V, simd::Set(255.f));
                    frame.SetPixel(x0, y0, simd::PackRGBA(colorV));
                }

                if(x0 == x1 && y0 == y1) break;

                int e2 = 2 * err;

                if(e2 > -dy) {
                    err -= dy;
                    x0 += sx;
                }

                if(e2 < dx) {
                    err += dx;
                    y0 += sy;
                }
            }
        }

        ENGINE_INLINE float ENGINE_VECTORCALL edge(const math::Vector& a, const math::Vector& b,
                                                   const math::Vector& c) {
            return (c.X - a.X) * (b.Y - a.Y) - (c.Y - a.Y) * (b.X - a.X);
        }

        // Pineda + edge
        template <typename Shader>
        ENGINE_INLINE void ENGINE_VECTORCALL drawTriangle(const Shader& shader, const shader::Varyings& v0,
                                                          const shader::Varyings& v1, const shader::Varyings& v2,
                                                          const BoundingBox& clipRect) {
            const float area = edge(v0.Pos, v1.Pos, v2.Pos);
            if(area <= 0.f) return;

            const float invArea = 1.f / area;

            BoundingBox triBound = frame.GetBound(v0.Pos, v1.Pos, v2.Pos);

            std::int32_t minX = std::max(triBound.MinX, clipRect.MinX);
            std::int32_t maxX = std::min(triBound.MaxX, clipRect.MaxX);
            std::int32_t minY = std::max(triBound.MinY, clipRect.MinY);
            std::int32_t maxY = std::min(triBound.MaxY, clipRect.MaxY);

            if(minX > maxX || minY > maxY) return;

            float row0 = edge(v1.Pos, v2.Pos, math::Vector(minX + 0.5f, minY + 0.5f, 0.f));
            float row1 = edge(v2.Pos, v0.Pos, math::Vector(minX + 0.5f, minY + 0.5f, 0.f));
            float row2 = edge(v0.Pos, v1.Pos, math::Vector(minX + 0.5f, minY + 0.5f, 0.f));

            const float dx0 = v2.Pos.Y - v1.Pos.Y;
            const float dy0 = v1.Pos.X - v2.Pos.X;

            const float dx1 = v0.Pos.Y - v2.Pos.Y;
            const float dy1 = v2.Pos.X - v0.Pos.X;

            const float dx2 = v1.Pos.Y - v0.Pos.Y;
            const float dy2 = v0.Pos.X - v1.Pos.X;

            for(int y = minY; y <= maxY; ++y) {
                float w0 = row0;
                float w1 = row1;
                float w2 = row2;

                for(int x = minX; x <= maxX; ++x) {
                    if(w0 >= 0.f && w1 >= 0.f && w2 >= 0.f) {
                        const float b0 = w0 * invArea;
                        const float b1 = w1 * invArea;
                        const float b2 = w2 * invArea;

                        const float interpolatedRecipW = (v0.RecipW * b0) + (v1.RecipW * b1) + (v2.RecipW * b2);
                        const float w = 1.f / interpolatedRecipW;

                        const float zw0 = v0.Pos.Z * v0.RecipW;
                        const float zw1 = v1.Pos.Z * v1.RecipW;
                        const float zw2 = v2.Pos.Z * v2.RecipW;

                        const float interpolatedZW = (zw0 * b0) + (zw1 * b1) + (zw2 * b2);
                        const float ndcZ = interpolatedZW / interpolatedRecipW;

                        const float z = (v0.Pos.Z * b0) + (v1.Pos.Z * b1) + (v2.Pos.Z * b2);
                        if(frame.TestDepth(x, y, z)) {
                            auto interpolate = [&](const math::Vector& a, const math::Vector& b,
                                                   const math::Vector& c) {
                                return ((a * v0.RecipW * b0) + (b * v1.RecipW * b1) + (c * v2.RecipW * b2)) * w;
                            };

                            const math::Vector worldPos = interpolate(v0.WorldPos, v1.WorldPos, v2.WorldPos);
                            const math::Vector normal = interpolate(v0.Normal, v1.Normal, v2.Normal).Norm();
                            const math::Vector uv = interpolate(v0.UV, v1.UV, v2.UV);
                            const math::Vector tangent = interpolate(v0.Tangent, v1.Tangent, v2.Tangent).Norm();
                            const math::Vector color = interpolate(v0.Color, v1.Color, v2.Color);

                            const std::uint32_t pixelColor = shader.Color(color, normal, worldPos, uv, tangent);
                            const std::uint32_t alpha = (pixelColor >> 24) & 0xFF;

                            if(alpha == 255) {
                                frame.SetPixel(x, y, pixelColor);
                                frame.SetDepth(x, y, z);

                                const math::Matrix& viewMatrix = shader.Uniform.View;
                                math::Vector viewNormal = viewMatrix * math::Vector(normal.X, normal.Y, normal.Z, 0.f);
                                frame.SetNormal(x, y, viewNormal.Norm());
                            }
                            else if(alpha > 0) {
                                const std::uint32_t dstColor = frame.GetPixel(x, y);
                                frame.SetPixel(x, y, alphaBlend(pixelColor, dstColor));
                            }
                        }
                    }
                    w0 += dx0;
                    w1 += dx1;
                    w2 += dx2;
                }
                row0 += dy0;
                row1 += dy1;
                row2 += dy2;
            }
        }

        ENGINE_INLINE std::int32_t ENGINE_VECTORCALL colorDiff(const std::uint32_t c1, const std::uint32_t c2) {
            int r1 = (c1 >> 16) & 0xFF;
            int g1 = (c1 >> 8) & 0xFF;
            int b1 = c1 & 0xFF;
            int r2 = (c2 >> 16) & 0xFF;
            int g2 = (c2 >> 8) & 0xFF;
            int b2 = c2 & 0xFF;
            return std::abs(r1 - r2) + std::abs(g1 - g2) + std::abs(b1 - b2);
        }

        ENGINE_INLINE std::uint32_t ENGINE_VECTORCALL mixColors(const std::uint32_t c1, const std::uint32_t c2,
                                                                const std::uint32_t c3, const std::uint32_t c4,
                                                                const std::uint32_t c5) {
            int r = (((c1 >> 16) & 0xFF) + ((c2 >> 16) & 0xFF) + ((c3 >> 16) & 0xFF) + ((c4 >> 16) & 0xFF) +
                     ((c5 >> 16) & 0xFF)) /
                    5;

            int g = (((c1 >> 8) & 0xFF) + ((c2 >> 8) & 0xFF) + ((c3 >> 8) & 0xFF) + ((c4 >> 8) & 0xFF) +
                     ((c5 >> 8) & 0xFF)) /
                    5;

            int b = ((c1 & 0xFF) + (c2 & 0xFF) + (c3 & 0xFF) + (c4 & 0xFF) + (c5 & 0xFF)) / 5;

            return (0xFF << 24) | (r << 16) | (g << 8) | b;
        }

        ENGINE_INLINE std::uint32_t ENGINE_VECTORCALL alphaBlend(const std::uint32_t src, const std::uint32_t dst) {
            std::uint32_t a = (src >> 24) & 0xFF;

            if(a == 0) return dst;
            if(a == 255) return src;

            std::uint32_t invA = 255 - a;

            std::uint32_t r = (((src >> 16) & 0xFF) * a + ((dst >> 16) & 0xFF) * invA) >> 8;
            std::uint32_t g = (((src >> 8) & 0xFF) * a + ((dst >> 8) & 0xFF) * invA) >> 8;
            std::uint32_t b = ((src & 0xFF) * a + (dst & 0xFF) * invA) >> 8;

            return 0xFF000000 | (r << 16) | (g << 8) | b;
        }
    };
}
