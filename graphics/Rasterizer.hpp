#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "../math/Math.hpp"
#include "FrameBuffer.hpp"
#include "Shader.hpp"

namespace graphics {
    enum class PrimitiveType { Points, Lines, Triangles };

    class Rasterizer {
    public:
        explicit Rasterizer(FrameBuffer& frame)
            : frame(frame),
              width(static_cast<float>(frame.GetWidth())),
              height(static_cast<float>(frame.GetHeight())) {}

        template <typename Shader>
        inline void Render(const Shader& shader, const std::vector<shader::Vertex>& vertices,
                           const PrimitiveType type = PrimitiveType::Triangles) {
            std::vector<std::uint32_t> indices(vertices.size());
            for(size_t i = 0; i < vertices.size(); ++i) indices[i] = i;

            Render(shader, vertices, indices, type);
        }

        template <typename Shader>
        inline void Render(const Shader& shader, const std::vector<shader::Vertex>& vertices,
                           const std::vector<std::uint32_t>& indices,
                           const PrimitiveType type = PrimitiveType::Triangles, std::size_t maxIndices = 0) {
            std::size_t actualLimit = (maxIndices == 0 || maxIndices > indices.size()) ? indices.size() : maxIndices;

            std::vector<shader::Varyings> screenVertices = processVertices<Shader>(shader, vertices);

            std::vector<std::uint32_t> limitedIndices(indices.begin(), indices.begin() + actualLimit);
            dispatchPrimitives<Shader>(shader, screenVertices, limitedIndices, type);
        }

        void ApplyPostAA() {
            std::uint32_t w = frame.GetWidth();
            std::uint32_t h = frame.GetHeight();

            const std::vector<std::uint32_t>& src = frame.GetColorBuffer();
            std::vector<std::uint32_t> dest = src;

            const int THRESHOLD = 30;

            for(std::uint32_t y = 1; y < h - 1; ++y) {
                for(std::uint32_t x = 1; x < w - 1; ++x) {
                    std::uint32_t idx = y * w + x;
                    std::uint32_t current = src[idx];

                    std::uint32_t up = src[idx - w];
                    std::uint32_t down = src[idx + w];
                    std::uint32_t left = src[idx - 1];
                    std::uint32_t right = src[idx + 1];

                    int diff = colorDiff(current, up) + colorDiff(current, down) + colorDiff(current, left) +
                               colorDiff(current, right);

                    if(diff > THRESHOLD) {
                        dest[idx] = mixColors(current, up, down, left, right);
                    }
                }
            }

            frame.UpdateBuffer(dest);
        }

    private:
        FrameBuffer& frame;
        float width;
        float height;

        // vertex Shader -> rasterization -> pixel Shader
        template <typename Shader>
        inline std::vector<shader::Varyings> processVertices(const Shader& shader,
                                                             const std::vector<shader::Vertex>& in) {
            std::vector<shader::Varyings> out(in.size());

            for(size_t i = 0; i < in.size(); ++i) {
                out[i] = shader.Process(in[i]);

                float w = out[i].Pos.W;
                if(std::abs(w) < 1e-6f) w = 1e-6f;

                float rhw = 1.f / w;
                out[i].Pos.X = (out[i].Pos.X * rhw + 1.f) * width * 0.5f;
                out[i].Pos.Y = (1.f - out[i].Pos.Y * rhw) * height * 0.5f;
                out[i].Pos.Z = out[i].Pos.Z * rhw;
                out[i].Pos.W = rhw;
                out[i].RecipW = rhw;
            }
            return out;
        }

        template <typename Shader> inline void drawPoint(const Shader& shader, const shader::Varyings& v) {
            int x = static_cast<int>(std::round(v.Pos.X));
            int y = static_cast<int>(std::round(v.Pos.Y));

            if(frame.TestDepth(x, y, v.Pos.Z)) {
                frame.SetPixel(x, y, shader.Color(v.Color, v.Normal, v.WorldPos, v.UV, v.Tangent));
            }
        }

        // Bresenham's Line Algorithm
        template <typename Shader>
        inline void drawLine(const Shader& shader, const shader::Varyings& v0, const shader::Varyings& v1) {
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
                math::Vector normal = v0.Normal * (1.f - t) + v1.Normal * t;
                math::Vector worldPos = v0.WorldPos * (1.f - t) + v1.WorldPos * t;
                math::Vector uv = v0.UV * (1.f - t) + v1.UV * t;
                math::Vector tangent = v0.Tangent * (1.f - t) + v1.Tangent * t;

                if(frame.TestDepth(x0, y0, z)) {
                    frame.SetPixel(x0, y0, shader.Color(color, normal, worldPos, uv, tangent));
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

        template <typename Shader>
        inline void drawTriangle(const Shader& shader, const shader::Varyings& v0, const shader::Varyings& v1,
                                 const shader::Varyings& v2) {
            if(v0.RecipW < 0 || v1.RecipW < 0 || v2.RecipW < 0) return;

            const float area =
                (v1.Pos.X - v0.Pos.X) * (v2.Pos.Y - v0.Pos.Y) - (v1.Pos.Y - v0.Pos.Y) * (v2.Pos.X - v0.Pos.X);

            if(area > 0.f) return;

            BoundingBox bound = frame.GetBound(v0.Pos, v1.Pos, v2.Pos);
            if(!bound.ShouldRender) return;

            for(int y = bound.MinY; y <= bound.MaxY; ++y) {
                for(int x = bound.MinX; x <= bound.MaxX; ++x) {
                    const math::Vector currPos(static_cast<float>(x), static_cast<float>(y), 0.f);
                    math::Vector bary = math::GetBarycentric(currPos, v0.Pos, v1.Pos, v2.Pos);

                    if(bary.X < 0 || bary.Y < 0 || bary.Z < 0) continue;

                    float interpolatedRecipW = (v0.RecipW * bary.X) + (v1.RecipW * bary.Y) + (v2.RecipW * bary.Z);
                    float w = 1.f / interpolatedRecipW;

                    float z = v0.Pos.Z * bary.X + v1.Pos.Z * bary.Y + v2.Pos.Z * bary.Z;
                    if(frame.TestDepth(x, y, z)) {
                        auto perspectiveInterpolate = [&](const math::Vector& a0, const math::Vector& a1,
                                                          const math::Vector& a2) {
                            return ((a0 * v0.RecipW * bary.X) + (a1 * v1.RecipW * bary.Y) + (a2 * v2.RecipW * bary.Z)) *
                                   w;
                        };

                        const math::Vector worldPos =
                            (v0.WorldPos * bary.X) + (v1.WorldPos * bary.Y) + (v2.WorldPos * bary.Z);

                        math::Vector interpolatedNormal =
                            (v0.Normal * bary.X) + (v1.Normal * bary.Y) + (v2.Normal * bary.Z);

                        const float lenSq = interpolatedNormal.Dot(interpolatedNormal);
                        if(lenSq > 1e-8f) interpolatedNormal *= (1.f / std::sqrt(lenSq));

                        const math::Vector interpolatedColor =
                            (v0.Color * bary.X) + (v1.Color * bary.Y) + (v2.Color * bary.Z);

                        math::Vector interpolatedUV = (v0.UV * bary.X) + (v1.UV * bary.Y) + (v2.UV * bary.Z);

                        math::Vector interpTangent =
                            (v0.Tangent * bary.X) + (v1.Tangent * bary.Y) + (v2.Tangent * bary.Z);

                        std::uint32_t pixelColor = shader.Color(interpolatedColor, interpolatedNormal, worldPos,
                                                                interpolatedUV, interpTangent);

                        std::uint32_t alpha = (pixelColor >> 24) & 0xFF;
                        if(alpha == 0) continue;

                        if(alpha < 255) {
                            std::uint32_t dstColor = frame.GetPixel(x, y);
                            std::uint32_t blendedColor = alphaBlend(pixelColor, dstColor);
                            frame.SetPixel(x, y, blendedColor);
                        }
                        else {
                            frame.SetPixel(x, y, pixelColor);
                            frame.SetDepth(x, y, z);
                        }
                    }
                }
            }
        }

        template <typename Shader>
        inline void dispatchPrimitives(const Shader& shader, const std::vector<shader::Varyings>& varyings,
                                       const std::vector<std::uint32_t>& indices,
                                       const PrimitiveType type = PrimitiveType::Triangles) {
            switch(type) {
            case PrimitiveType::Points:
                for(std::size_t index : indices) {
                    if(index >= varyings.size()) continue;

                    drawPoint(shader, varyings[index]);
                }
                break;

            case PrimitiveType::Lines:
                for(std::size_t i = 0; i < indices.size(); i += 3) {
                    if(i + 2 >= indices.size()) break;

                    if(indices[i] >= varyings.size() || indices[i + 1] >= varyings.size() ||
                       indices[i + 2] >= varyings.size())
                        continue;

                    const shader::Varyings& v0 = varyings[indices[i]];
                    const shader::Varyings& v1 = varyings[indices[i + 1]];
                    const shader::Varyings& v2 = varyings[indices[i + 2]];

                    drawLine(shader, v0, v1);
                    drawLine(shader, v1, v2);
                    drawLine(shader, v2, v0);
                }
                break;

            default:
                for(std::size_t i = 0; i < indices.size(); i += 3) {
                    if(i + 2 >= indices.size()) break;

                    if(indices[i] >= varyings.size() || indices[i + 1] >= varyings.size() ||
                       indices[i + 2] >= varyings.size())
                        continue;

                    const shader::Varyings& v0 = varyings[indices[i]];
                    const shader::Varyings& v1 = varyings[indices[i + 1]];
                    const shader::Varyings& v2 = varyings[indices[i + 2]];

                    drawTriangle(shader, v0, v1, v2);
                }
                break;
            }
        }

        inline int colorDiff(uint32_t c1, uint32_t c2) {
            int r1 = (c1 >> 16) & 0xFF;
            int g1 = (c1 >> 8) & 0xFF;
            int b1 = c1 & 0xFF;
            int r2 = (c2 >> 16) & 0xFF;
            int g2 = (c2 >> 8) & 0xFF;
            int b2 = c2 & 0xFF;
            return std::abs(r1 - r2) + std::abs(g1 - g2) + std::abs(b1 - b2);
        }

        inline uint32_t mixColors(uint32_t c1, uint32_t c2, uint32_t c3, uint32_t c4, uint32_t c5) {
            int r = (((c1 >> 16) & 0xFF) + ((c2 >> 16) & 0xFF) + ((c3 >> 16) & 0xFF) + ((c4 >> 16) & 0xFF) +
                     ((c5 >> 16) & 0xFF)) /
                    5;

            int g = (((c1 >> 8) & 0xFF) + ((c2 >> 8) & 0xFF) + ((c3 >> 8) & 0xFF) + ((c4 >> 8) & 0xFF) +
                     ((c5 >> 8) & 0xFF)) /
                    5;

            int b = ((c1 & 0xFF) + (c2 & 0xFF) + (c3 & 0xFF) + (c4 & 0xFF) + (c5 & 0xFF)) / 5;

            return (0xFF << 24) | (r << 16) | (g << 8) | b;
        }

        inline std::uint32_t alphaBlend(std::uint32_t src, std::uint32_t dst) {
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
