#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "../math/Math.hpp"
#include "FrameBuffer.hpp"
#include "Shader.hpp"

namespace graphics {
    enum class PrimitiveType { Points, Lines, Triangles };

    template <typename Shader>
    inline void DrawPoint(FrameBuffer& frame, const Shader& shader, const shader::Vertex& v) {
        int x = static_cast<int>(std::round(v.Pos.X));
        int y = static_cast<int>(std::round(v.Pos.Y));

        if(frame.IsVisible(x, y, v.Pos.Z)) {
            frame.SetPixel(x, y, shader.Color(v.Color));
        }
    }

    // Bresenham's Line Algorithm
    template <typename Shader>
    inline void DrawLine(FrameBuffer& frame, const Shader& shader, const shader::Vertex& v0, const shader::Vertex& v1) {
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
            float t =
                (totalDist < 1e-6f) ? 0.f : std::sqrt(std::pow(x0 - startX, 2) + std::pow(y0 - startY, 2)) / totalDist;

            float z = v0.Pos.Z * (1.f - t) + v1.Pos.Z * t;
            math::Vector color = v0.Color * (1.f - t) + v1.Color * t;

            if(frame.IsVisible(x0, y0, z)) {
                frame.SetPixel(x0, y0, shader.Color(color));
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
    inline void DrawTriangle(FrameBuffer& frame, const Shader& shader, const shader::Vertex& v0,
                             const shader::Vertex& v1, const shader::Vertex& v2) {
        if((v1.Pos.X - v0.Pos.X) * (v2.Pos.Y - v0.Pos.Y) - (v1.Pos.Y - v0.Pos.Y) * (v2.Pos.X - v0.Pos.X) > 0.f) return;

        BoundingBox bound = frame.GetBound(v0.Pos, v1.Pos, v2.Pos);

        for(int y = bound.MinY; y <= bound.MaxY; ++y) {
            for(int x = bound.MinX; x <= bound.MaxX; ++x) {
                const math::Vector currPos(static_cast<float>(x), static_cast<float>(y), 0.f);
                math::Vector bary = math::GetBarycentric(currPos, v0.Pos, v1.Pos, v2.Pos);

                if(bary.X < 0 || bary.Y < 0 || bary.Z < 0) continue;

                float z = v0.Pos.Z * bary.X + v1.Pos.Z * bary.Y + v2.Pos.Z * bary.Z;
                if(frame.IsVisible(x, y, z)) {
                    const math::Vector interpolated = (v0.Color * bary.X) + (v1.Color * bary.Y) + (v2.Color * bary.Z);
                    frame.SetPixel(x, y, shader.Color(interpolated));
                }
            }
        }
    }

    template <typename Shader>
    inline void Render(FrameBuffer& frame, const Shader& shader, const std::vector<shader::Vertex>& vertices,
                       PrimitiveType type = PrimitiveType::Triangles) {
        std::vector<shader::Vertex> screenVertices;
        for(const shader::Vertex& vertex : vertices) {
            screenVertices.push_back({shader.Vertex(vertex.Pos), vertex.Color});
        }

        switch(type) {
        case PrimitiveType::Points:
            for(const shader::Vertex& vertex : screenVertices) DrawPoint(frame, shader, vertex);
            break;

        case PrimitiveType::Lines:
            for(std::size_t i = 0; i < screenVertices.size(); i += 2) {
                if(i + 1 < screenVertices.size()) DrawLine(frame, shader, screenVertices[i], screenVertices[i + 1]);
            }
            break;

        default:
            for(std::size_t i = 0; i < screenVertices.size(); i += 3) {
                if(i + 2 < screenVertices.size())
                    DrawTriangle(frame, shader, screenVertices[i], screenVertices[i + 1], screenVertices[i + 2]);
            }
            break;
        }
    }

    template <typename Shader>
    inline void Render(FrameBuffer& frame, const Shader& shader, const std::vector<shader::Vertex>& vertices,
                       const std::vector<std::uint32_t>& indices, PrimitiveType type = PrimitiveType::Triangles) {
        std::vector<shader::Vertex> screenVertices;
        screenVertices.reserve(vertices.size());

        for(const shader::Vertex& vertex : vertices) {
            screenVertices.push_back({shader.Vertex(vertex.Pos), vertex.Color});
        }

        switch(type) {
        case PrimitiveType::Points:
            for(const std::size_t& index : indices) {
                if(index >= screenVertices.size()) continue;

                DrawPoint(frame, shader, screenVertices[index]);
            }
            break;

        case PrimitiveType::Lines:
            for(std::size_t i = 0; i < indices.size(); i += 3) {
                if(i + 2 >= indices.size()) break;

                if(indices[i] >= screenVertices.size() || indices[i + 1] >= screenVertices.size() ||
                   indices[i + 2] >= screenVertices.size())
                    continue;

                const shader::Vertex& v0 = screenVertices[indices[i]];
                const shader::Vertex& v1 = screenVertices[indices[i + 1]];
                const shader::Vertex& v2 = screenVertices[indices[i + 2]];

                DrawLine(frame, shader, v0, v1);
                DrawLine(frame, shader, v1, v2);
                DrawLine(frame, shader, v2, v0);
            }
            break;

        default:
            for(std::size_t i = 0; i < indices.size(); i += 3) {
                if(indices[i] >= screenVertices.size() || indices[i + 1] >= screenVertices.size() ||
                   indices[i + 2] >= screenVertices.size())
                    continue;

                const shader::Vertex& v0 = screenVertices[indices[i]];
                const shader::Vertex& v1 = screenVertices[indices[i + 1]];
                const shader::Vertex& v2 = screenVertices[indices[i + 2]];
                DrawTriangle(frame, shader, v0, v1, v2);
            }
            break;
        }
    }
}