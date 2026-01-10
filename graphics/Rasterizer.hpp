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
                           const PrimitiveType type = PrimitiveType::Triangles) {
            std::vector<shader::Varyings> screenVertices = processVertices<Shader>(shader, vertices);
            dispatchPrimitives<Shader>(shader, screenVertices, indices, type);
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

            if(frame.IsVisible(x, y, v.Pos.Z)) {
                frame.SetPixel(x, y, shader.Color(v.Color, v.Normal, v.WorldPos));
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

                if(frame.IsVisible(x0, y0, z)) {
                    frame.SetPixel(x0, y0, shader.Color(color, normal, worldPos));
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

                    float z = v0.Pos.Z * bary.X + v1.Pos.Z * bary.Y + v2.Pos.Z * bary.Z;
                    if(frame.IsVisible(x, y, z)) {
                        const math::Vector worldPos =
                            (v0.WorldPos * bary.X) + (v1.WorldPos * bary.Y) + (v2.WorldPos * bary.Z);

                        math::Vector interpolatedNormal =
                            (v0.Normal * bary.X) + (v1.Normal * bary.Y) + (v2.Normal * bary.Z);

                        const float lenSq = interpolatedNormal.Dot(interpolatedNormal);
                        if(lenSq > 1e-8f) interpolatedNormal *= (1.f / std::sqrt(lenSq));

                        const math::Vector interpolatedColor =
                            (v0.Color * bary.X) + (v1.Color * bary.Y) + (v2.Color * bary.Z);

                        frame.SetPixel(x, y, shader.Color(interpolatedColor, interpolatedNormal, worldPos));
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
    };
}