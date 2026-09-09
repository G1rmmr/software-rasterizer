#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "../shaders/Elements.hpp"
#include "Color.hpp"
#include "FrameBuffer.hpp"
#include "ParallelExecutor.hpp"
#include "RenderTypes.hpp"

namespace graphics {
    // Clip coordinates use -w <= x,y <= w and 0 <= z <= w. A shader supplies
    // Process(Vertex) and Shade(Fragment); uniform/resource layout stays private
    // to the shader. Each Render completes before returning and retains no target.
    class Rasterizer {
    public:
        explicit Rasterizer(ParallelExecutor& executor) noexcept : executor(executor) {}

        template <typename Shader>
        void Render(FrameBuffer& target, const Shader& shader, std::span<const shader::Vertex> vertices,
                    std::span<const std::uint32_t> indices, const RasterizerOptions& options = {}) const {
            if(options.Primitive != PrimitiveType::Points && indices.size() % 3 != 0)
                throw std::invalid_argument("Triangle and wireframe indices must contain complete triangles");
            for(const auto index : indices)
                if(index >= vertices.size()) throw std::out_of_range("Vertex index is outside the draw's vertices");
            if(indices.empty() || target.GetWidth() == 0 || target.GetHeight() == 0) return;
            // Bound fixed-point edge products to signed 64-bit arithmetic.
            if(target.GetWidth() > MaxDimension || target.GetHeight() > MaxDimension)
                throw std::length_error("Raster target exceeds the fixed-point coordinate range");

            std::vector<ScreenVertex> transformed(vertices.size());
            std::vector<std::uint8_t> clipCodes(vertices.size());
            executor.ParallelFor(
                0, vertices.size(),
                [&](std::size_t i) {
                    auto& vertex = transformed[i];
                    vertex.Attributes = shader.Process(vertices[i]);
                    if(!FinitePosition(vertex.Attributes.Pos))
                        throw std::invalid_argument("Vertex shader produced non-finite clip coordinates");
                    clipCodes[i] = ClipCode(vertex.Attributes.Pos);
                    if(clipCodes[i] == 0 && vertex.Attributes.Pos.W > 0.f) SetScreenCoordinates(vertex, target);
                },
                256);

            if(options.Primitive == PrimitiveType::Points) {
                // Points and wireframe are intentionally serial: overlapping
                // primitives must not race their depth test and pixel writes.
                for(const auto index : indices) {
                    const auto& vertex = transformed[index];
                    if(clipCodes[index] == 0 && vertex.Attributes.Pos.W > 0.f)
                        DrawPoint(target, shader, vertex, options);
                }
                return;
            }

            std::vector<Triangle> triangles;
            triangles.reserve(indices.size() / 3);
            for(std::size_t i = 0; i < indices.size(); i += 3) {
                const auto i0 = indices[i], i1 = indices[i + 1], i2 = indices[i + 2];
                const auto& a = transformed[i0];
                const auto& b = transformed[i1];
                const auto& c = transformed[i2];
                if((clipCodes[i0] & clipCodes[i1] & clipCodes[i2]) != 0) continue;
                if((clipCodes[i0] | clipCodes[i1] | clipCodes[i2]) == 0) {
                    // Shared indexed vertices keep their screen conversion. Most
                    // visible primitives need neither polygon copies nor clipping.
                    if(a.Attributes.Pos.W <= 0.f || b.Attributes.Pos.W <= 0.f || c.Attributes.Pos.W <= 0.f) continue;
                    if(options.Primitive == PrimitiveType::Lines) {
                        if(SamePosition(a.Attributes.Pos, b.Attributes.Pos) ||
                           SamePosition(b.Attributes.Pos, c.Attributes.Pos) ||
                           SamePosition(c.Attributes.Pos, a.Attributes.Pos))
                            continue;
                        DrawLine(target, shader, a, b, options);
                        DrawLine(target, shader, b, c, options);
                        DrawLine(target, shader, c, a, options);
                    }
                    else
                        AppendTriangle(triangles, transformed, i0, i1, i2, target, options);
                    continue;
                }
                const auto polygon = ClipTriangle(a.Attributes, b.Attributes, c.Attributes);
                if(polygon.Count < 3) continue;
                std::array<ScreenVertex, MaxClipVertices> screen;
                bool valid = true;
                for(std::size_t j = 0; j < polygon.Count; ++j) {
                    if(polygon.Vertices[j].Pos.W <= 0.f) {
                        valid = false;
                        break;
                    }
                    screen[j] = ToScreen(polygon.Vertices[j], target);
                }
                if(!valid) continue;

                if(options.Primitive == PrimitiveType::Lines) {
                    // Lines means mesh wireframe. Draw the clipped polygon's
                    // perimeter, not the artificial diagonals of its triangle fan.
                    for(std::size_t j = 0; j < polygon.Count; ++j)
                        DrawLine(target, shader, screen[j], screen[(j + 1) % polygon.Count], options);
                    continue;
                }

                // Indices survive pool growth. No triangle retains references
                // into this storage until preparation is complete.
                const auto first = transformed.size();
                transformed.insert(transformed.end(), screen.begin(), screen.begin() + polygon.Count);
                for(std::size_t j = 1; j + 1 < polygon.Count; ++j)
                    AppendTriangle(triangles, transformed, first, first + j, first + j + 1, target, options);
            }

            // Centroid sorting is deterministic and correct for separated layers.
            // Intersecting transparent triangles need splitting or an OIT method;
            // ordering different draws belongs to the renderer above this class.
            DrawTiles(target, shader, triangles, transformed, options);
        }

    private:
        static constexpr std::int64_t Subpixel = 256;
        static constexpr std::uint32_t MaxDimension = 1u << 22;
        static constexpr int TileSize = 32;
        static constexpr std::size_t MaxClipVertices = 12;
        ParallelExecutor& executor;

        // A depth-only shader guarantees coverage without a fragment callback.
        // Its draw updates depth while preserving color and normal attachments.
        template <typename Shader>
        static constexpr bool IsDepthOnly = [] {
            if constexpr(requires { Shader::DepthOnly; })
                return static_cast<bool>(Shader::DepthOnly);
            else
                return false;
        }();

        struct Polygon {
            std::array<shader::Varyings, MaxClipVertices> Vertices;
            std::size_t Count = 0;
        };
        struct ScreenVertex {
            shader::Varyings Attributes;
            std::int64_t X = 0;
            std::int64_t Y = 0;
            float Depth = 0.f;
            double ReciprocalW = 0.;
        };
        struct PixelBounds {
            int MinX = 0, MaxX = -1, MinY = 0, MaxY = -1;
        };
        struct Triangle {
            std::array<std::size_t, 3> Vertices;
            std::array<std::int64_t, 3> StepX{};
            std::array<std::int64_t, 3> StepY{};
            std::array<bool, 3> IncludeEdge{};
            float InverseArea = 0.f;
            PixelBounds Bounds;
            float SortDepth = 0.f;
        };

        static bool FinitePosition(const math::Vector& p) noexcept {
            return std::isfinite(p.X) && std::isfinite(p.Y) && std::isfinite(p.Z) && std::isfinite(p.W);
        }

        static double PlaneDistance(const math::Vector& p, int plane) noexcept {
            switch(plane) {
            case 0: return static_cast<double>(p.W) + p.X;
            case 1: return static_cast<double>(p.W) - p.X;
            case 2: return static_cast<double>(p.W) + p.Y;
            case 3: return static_cast<double>(p.W) - p.Y;
            case 4: return p.Z;
            default: return static_cast<double>(p.W) - p.Z;
            }
        }

        static std::uint8_t ClipCode(const math::Vector& p) noexcept {
            std::uint8_t code = 0;
            for(int plane = 0; plane < 6; ++plane)
                if(PlaneDistance(p, plane) < 0.) code |= static_cast<std::uint8_t>(1u << plane);
            return code;
        }

        static bool SamePosition(const math::Vector& a, const math::Vector& b) noexcept {
            return a.X == b.X && a.Y == b.Y && a.Z == b.Z && a.W == b.W;
        }

        static void Append(Polygon& polygon, const shader::Varyings& vertex) {
            if(polygon.Count != 0 && SamePosition(polygon.Vertices[polygon.Count - 1].Pos, vertex.Pos)) return;
            if(polygon.Count == MaxClipVertices) throw std::logic_error("Clipped polygon exceeded its vertex bound");
            polygon.Vertices[polygon.Count++] = vertex;
        }

        static shader::Varyings Lerp(const shader::Varyings& a, const shader::Varyings& b, float t) {
            shader::Varyings result{};
            result.Pos = a.Pos * (1.f - t) + b.Pos * t;
            result.WorldPos = a.WorldPos * (1.f - t) + b.WorldPos * t;
            result.Normal = a.Normal * (1.f - t) + b.Normal * t;
            result.Color = a.Color * (1.f - t) + b.Color * t;
            result.UV = a.UV * (1.f - t) + b.UV * t;
            result.Tangent = a.Tangent * (1.f - t) + b.Tangent * t;
            return result;
        }

        static Polygon ClipTriangle(const shader::Varyings& a, const shader::Varyings& b, const shader::Varyings& c) {
            Polygon polygon;
            polygon.Vertices[0] = a;
            polygon.Vertices[1] = b;
            polygon.Vertices[2] = c;
            polygon.Count = 3;
            for(int plane = 0; plane < 6 && polygon.Count != 0; ++plane) {
                Polygon output;
                for(std::size_t i = 0; i < polygon.Count; ++i) {
                    const auto& previous = polygon.Vertices[(i + polygon.Count - 1) % polygon.Count];
                    const auto& current = polygon.Vertices[i];
                    const double previousDistance = PlaneDistance(previous.Pos, plane);
                    const double currentDistance = PlaneDistance(current.Pos, plane);
                    const bool previousInside = previousDistance >= 0.;
                    const bool currentInside = currentDistance >= 0.;
                    if(previousInside != currentInside) {
                        // Always evaluate the intersection from the inside vertex:
                        // shared edges traversed in reverse generate the same point.
                        const auto& inside = previousInside ? previous : current;
                        const auto& outside = previousInside ? current : previous;
                        const double insideDistance = previousInside ? previousDistance : currentDistance;
                        const double outsideDistance = previousInside ? currentDistance : previousDistance;
                        auto intersection = Lerp(
                            inside, outside, static_cast<float>(insideDistance / (insideDistance - outsideDistance)));
                        // Keep the computed boundary exact after float conversion.
                        switch(plane) {
                        case 0: intersection.Pos.X = -intersection.Pos.W; break;
                        case 1: intersection.Pos.X = intersection.Pos.W; break;
                        case 2: intersection.Pos.Y = -intersection.Pos.W; break;
                        case 3: intersection.Pos.Y = intersection.Pos.W; break;
                        case 4: intersection.Pos.Z = 0.f; break;
                        case 5: intersection.Pos.Z = intersection.Pos.W; break;
                        }
                        Append(output, intersection);
                    }
                    if(currentInside) Append(output, current);
                }
                if(output.Count > 1 && SamePosition(output.Vertices[0].Pos, output.Vertices[output.Count - 1].Pos))
                    --output.Count;
                polygon = std::move(output);
            }
            return polygon;
        }

        static ScreenVertex ToScreen(const shader::Varyings& vertex, const FrameBuffer& target) {
            ScreenVertex result;
            result.Attributes = vertex;
            SetScreenCoordinates(result, target);
            return result;
        }

        static void SetScreenCoordinates(ScreenVertex& vertex, const FrameBuffer& target) {
            const auto& position = vertex.Attributes.Pos;
            const double reciprocal = 1. / position.W;
            vertex.ReciprocalW = reciprocal;
            const double x = (position.X * reciprocal + 1.) * target.GetWidth() * .5;
            const double y = (1. - position.Y * reciprocal) * target.GetHeight() * .5;
            vertex.X = static_cast<std::int64_t>(std::llround(x * Subpixel));
            vertex.Y = static_cast<std::int64_t>(std::llround(y * Subpixel));
            vertex.Depth = static_cast<float>(position.Z * reciprocal);
        }

        static std::int64_t Edge(const ScreenVertex& a, const ScreenVertex& b, std::int64_t x,
                                 std::int64_t y) noexcept {
            return (x - a.X) * (b.Y - a.Y) - (y - a.Y) * (b.X - a.X);
        }
        static std::int64_t Edge(const ScreenVertex& a, const ScreenVertex& b, const ScreenVertex& c) noexcept {
            return Edge(a, b, c.X, c.Y);
        }
        static bool TopLeft(const ScreenVertex& a, const ScreenVertex& b) noexcept {
            const auto dy = b.Y - a.Y;
            const auto dx = b.X - a.X;
            return dy > 0 || (dy == 0 && dx < 0);
        }

        static PixelBounds BoundsOf(const ScreenVertex& a, const ScreenVertex& b, const ScreenVertex& c,
                                    const FrameBuffer& target) {
            const auto minX = std::min({a.X, b.X, c.X});
            const auto maxX = std::max({a.X, b.X, c.X});
            const auto minY = std::min({a.Y, b.Y, c.Y});
            const auto maxY = std::max({a.Y, b.Y, c.Y});
            return {std::max(0, static_cast<int>(minX / Subpixel)),
                    std::min(static_cast<int>(target.GetWidth()) - 1, static_cast<int>(maxX / Subpixel)),
                    std::max(0, static_cast<int>(minY / Subpixel)),
                    std::min(static_cast<int>(target.GetHeight()) - 1, static_cast<int>(maxY / Subpixel))};
        }

        static void AppendTriangle(std::vector<Triangle>& triangles, std::span<const ScreenVertex> vertices,
                                   std::size_t i0, std::size_t i1, std::size_t i2, const FrameBuffer& target,
                                   const RasterizerOptions& options) {
            const auto& a = vertices[i0];
            const auto& b = vertices[i1];
            const auto& c = vertices[i2];
            const auto area = Edge(a, b, c);
            const bool frontFacing = options.FlipWinding ? area < 0 : area > 0;
            if(area == 0 || (options.Cull == CullMode::Back && !frontFacing)) return;
            Triangle triangle{{i0, area > 0 ? i1 : i2, area > 0 ? i2 : i1}};
            triangle.Bounds = BoundsOf(a, b, c, target);
            if(triangle.Bounds.MinX > triangle.Bounds.MaxX || triangle.Bounds.MinY > triangle.Bounds.MaxY) return;
            triangle.InverseArea = 1.f / static_cast<float>(area > 0 ? area : -area);
            triangle.SortDepth = (a.Depth + b.Depth + c.Depth) / 3.f;
            for(std::size_t edge = 0; edge < 3; ++edge) {
                const auto& start = vertices[triangle.Vertices[(edge + 1) % 3]];
                const auto& end = vertices[triangle.Vertices[(edge + 2) % 3]];
                triangle.StepX[edge] = (end.Y - start.Y) * Subpixel;
                triangle.StepY[edge] = -(end.X - start.X) * Subpixel;
                triangle.IncludeEdge[edge] = TopLeft(start, end);
            }
            triangles.push_back(std::move(triangle));
        }

        static math::Vector NormalizeDirection(const math::Vector& value) {
            return math::Vector(value.X, value.Y, value.Z, 0.f).Norm();
        }
        static shader::Fragment FragmentFrom(const shader::Varyings& attributes) {
            shader::Fragment result{};
            result.WorldPos = attributes.WorldPos;
            result.Normal = NormalizeDirection(attributes.Normal);
            result.Color = attributes.Color;
            result.UV = attributes.UV;
            result.Tangent = NormalizeDirection(attributes.Tangent);
            result.Tangent.W = attributes.Tangent.W < 0.f ? -1.f : 1.f;
            return result;
        }
        static shader::Fragment Interpolate(const ScreenVertex& a, const ScreenVertex& b, const ScreenVertex& c,
                                            float b0, float b1, float b2) {
            const double p0 = b0 * a.ReciprocalW;
            const double p1 = b1 * b.ReciprocalW;
            const double p2 = b2 * c.ReciprocalW;
            const double sum = p0 + p1 + p2;
            const float weight0 = static_cast<float>(p0 / sum);
            const float weight1 = static_cast<float>(p1 / sum);
            const float weight2 = static_cast<float>(p2 / sum);
            const auto interpolate = [&](auto member) {
                return (a.Attributes.*member) * weight0 + (b.Attributes.*member) * weight1 +
                       (c.Attributes.*member) * weight2;
            };
            shader::Varyings attributes{};
            attributes.WorldPos = interpolate(&shader::Varyings::WorldPos);
            attributes.Normal = interpolate(&shader::Varyings::Normal);
            attributes.Color = interpolate(&shader::Varyings::Color);
            attributes.UV = interpolate(&shader::Varyings::UV);
            attributes.Tangent = interpolate(&shader::Varyings::Tangent);
            return FragmentFrom(attributes);
        }

        static void WriteFragment(FrameBuffer& target, int x, int y, float depth, const shader::FragmentOutput& output,
                                  const RasterizerOptions& options) {
            if((output.Color >> 24) == 0) return;
            target.SetPixel(x, y, options.Blend ? AlphaBlend(output.Color, target.GetPixel(x, y)) : output.Color);
            if(options.DepthWrite) {
                target.SetDepth(x, y, depth);
                target.SetNormal(x, y, output.ViewNormal);
            }
        }
        static bool DepthPasses(const FrameBuffer& target, int x, int y, float depth) {
            return std::isfinite(depth) && depth >= 0.f && depth <= 1.f && target.TestDepth(x, y, depth);
        }

        template <typename Shader>
        static void DrawPoint(FrameBuffer& target, const Shader& shader, const ScreenVertex& vertex,
                              const RasterizerOptions& options) {
            const int x = static_cast<int>(vertex.X / Subpixel);
            const int y = static_cast<int>(vertex.Y / Subpixel);
            if(x < 0 || y < 0 || x >= static_cast<int>(target.GetWidth()) || y >= static_cast<int>(target.GetHeight()))
                return;
            if(!DepthPasses(target, x, y, vertex.Depth)) return;
            if constexpr(IsDepthOnly<Shader>) {
                if(options.DepthWrite) target.SetDepth(x, y, vertex.Depth);
            }
            else {
                auto fragment = FragmentFrom(vertex.Attributes);
                auto output = shader.Shade(fragment);
                output.Color = PackColor(fragment.Color);
                WriteFragment(target, x, y, vertex.Depth, output, options);
            }
        }

        template <typename Shader>
        static void DrawLine(FrameBuffer& target, const Shader& shader, const ScreenVertex& a, const ScreenVertex& b,
                             const RasterizerOptions& options) {
            int x = static_cast<int>(a.X / Subpixel), y = static_cast<int>(a.Y / Subpixel);
            const int endX = static_cast<int>(b.X / Subpixel), endY = static_cast<int>(b.Y / Subpixel);
            const int dx = std::abs(endX - x), dy = std::abs(endY - y);
            const int steps = std::max(dx, dy);
            const int sx = x < endX ? 1 : -1, sy = y < endY ? 1 : -1;
            int error = dx - dy;
            for(int step = 0;; ++step) {
                const float t = steps == 0 ? 0.f : static_cast<float>(step) / steps;
                const float depth = a.Depth * (1.f - t) + b.Depth * t;
                if(x >= 0 && y >= 0 && x < static_cast<int>(target.GetWidth()) &&
                   y < static_cast<int>(target.GetHeight()) && DepthPasses(target, x, y, depth)) {
                    if constexpr(IsDepthOnly<Shader>) {
                        if(options.DepthWrite) target.SetDepth(x, y, depth);
                    }
                    else {
                        const double weightA = (1.f - t) * a.ReciprocalW;
                        const double weightB = t * b.ReciprocalW;
                        auto attributes =
                            Lerp(a.Attributes, b.Attributes, static_cast<float>(weightB / (weightA + weightB)));
                        auto fragment = FragmentFrom(attributes);
                        auto output = shader.Shade(fragment);
                        output.Color = PackColor(fragment.Color);
                        WriteFragment(target, x, y, depth, output, options);
                    }
                }
                if(x == endX && y == endY) break;
                const int twiceError = 2 * error;
                if(twiceError > -dy) {
                    error -= dy;
                    x += sx;
                }
                if(twiceError < dx) {
                    error += dx;
                    y += sy;
                }
            }
        }

        template <typename Shader>
        void DrawTiles(FrameBuffer& target, const Shader& shader, const std::vector<Triangle>& triangles,
                       std::span<const ScreenVertex> vertices, const RasterizerOptions& options) const {
            if(triangles.empty()) return;
            const int tilesX = (static_cast<int>(target.GetWidth()) + TileSize - 1) / TileSize;
            const int tilesY = (static_cast<int>(target.GetHeight()) + TileSize - 1) / TileSize;
            const auto tileCount = static_cast<std::size_t>(tilesX) * tilesY;
            // Count, prefix and fill one contiguous index buffer. All scratch is
            // draw-local, so concurrent renders never share mutable bin storage.
            std::vector<std::size_t> offsets(tileCount + 1);
            const auto visitTiles = [&](const PixelBounds& bounds, auto&& visit) {
                for(int y = bounds.MinY / TileSize; y <= bounds.MaxY / TileSize; ++y)
                    for(int x = bounds.MinX / TileSize; x <= bounds.MaxX / TileSize; ++x)
                        visit(static_cast<std::size_t>(y) * tilesX + x);
            };
            for(const auto& triangle : triangles)
                visitTiles(triangle.Bounds, [&](std::size_t tile) { ++offsets[tile + 1]; });
            for(std::size_t tile = 1; tile <= tileCount; ++tile) offsets[tile] += offsets[tile - 1];
            std::vector<std::size_t> bins(offsets.back());
            std::vector<std::size_t> cursors(offsets.begin(), offsets.end() - 1);
            const auto binTriangle = [&](std::size_t index) {
                visitTiles(triangles[index].Bounds, [&](std::size_t tile) { bins[cursors[tile]++] = index; });
            };
            // Sort small handles rather than moving aligned vertex payloads.
            // Ordered filling preserves the draw's primitive order in every tile.
            if(options.Blend) {
                std::vector<std::size_t> order(triangles.size());
                for(std::size_t i = 0; i < order.size(); ++i) order[i] = i;
                std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                    return triangles[a].SortDepth > triangles[b].SortDepth;
                });
                for(const auto index : order) binTriangle(index);
            }
            else
                for(std::size_t index = 0; index < triangles.size(); ++index) binTriangle(index);
            std::vector<std::size_t> activeTiles;
            activeTiles.reserve(tileCount);
            for(std::size_t tile = 0; tile < tileCount; ++tile)
                if(offsets[tile] != offsets[tile + 1]) activeTiles.push_back(tile);
            executor.ParallelFor(
                0, activeTiles.size(),
                [&](std::size_t activeIndex) {
                    const auto index = activeTiles[activeIndex];
                    const int x = static_cast<int>(index % tilesX) * TileSize;
                    const int y = static_cast<int>(index / tilesX) * TileSize;
                    const PixelBounds clip{x, std::min(x + TileSize - 1, static_cast<int>(target.GetWidth()) - 1), y,
                                           std::min(y + TileSize - 1, static_cast<int>(target.GetHeight()) - 1)};
                    for(std::size_t item = offsets[index]; item < offsets[index + 1]; ++item)
                        DrawTriangle(target, shader, triangles[bins[item]], vertices, clip, options);
                },
                1);
        }

        template <typename Shader>
        static void DrawTriangle(FrameBuffer& target, const Shader& shader, const Triangle& triangle,
                                 std::span<const ScreenVertex> vertices, const PixelBounds& clip,
                                 const RasterizerOptions& options) {
            const auto& a = vertices[triangle.Vertices[0]];
            const auto& b = vertices[triangle.Vertices[1]];
            const auto& c = vertices[triangle.Vertices[2]];
            const bool include0 = triangle.IncludeEdge[0], include1 = triangle.IncludeEdge[1],
                       include2 = triangle.IncludeEdge[2];
            const float inverseArea = triangle.InverseArea;
            const int minX = std::max(clip.MinX, triangle.Bounds.MinX),
                      maxX = std::min(clip.MaxX, triangle.Bounds.MaxX);
            const int minY = std::max(clip.MinY, triangle.Bounds.MinY),
                      maxY = std::min(clip.MaxY, triangle.Bounds.MaxY);
            const auto sampleX = static_cast<std::int64_t>(minX) * Subpixel + Subpixel / 2;
            const auto sampleY = static_cast<std::int64_t>(minY) * Subpixel + Subpixel / 2;
            auto row0 = Edge(b, c, sampleX, sampleY);
            auto row1 = Edge(c, a, sampleX, sampleY);
            auto row2 = Edge(a, b, sampleX, sampleY);
            // Fixed-point additions are exactly the original edge evaluations;
            // they preserve top-left ownership without per-sample products.
            for(int y = minY; y <= maxY;
                ++y, row0 += triangle.StepY[0], row1 += triangle.StepY[1], row2 += triangle.StepY[2]) {
                auto e0 = row0, e1 = row1, e2 = row2;
                for(int x = minX; x <= maxX;
                    ++x, e0 += triangle.StepX[0], e1 += triangle.StepX[1], e2 += triangle.StepX[2]) {
                    if(e0 < 0 || (e0 == 0 && !include0) || e1 < 0 || (e1 == 0 && !include1) || e2 < 0 ||
                       (e2 == 0 && !include2))
                        continue;
                    const float b0 = static_cast<float>(e0) * inverseArea;
                    const float b1 = static_cast<float>(e1) * inverseArea;
                    const float b2 = static_cast<float>(e2) * inverseArea;
                    // NDC depth is screen-linear; other attributes use 1/w.
                    const float depth = std::clamp(a.Depth * b0 + b.Depth * b1 + c.Depth * b2, 0.f, 1.f);
                    if(!DepthPasses(target, x, y, depth)) continue;
                    if constexpr(IsDepthOnly<Shader>) {
                        if(options.DepthWrite) target.SetDepth(x, y, depth);
                    }
                    else
                        WriteFragment(target, x, y, depth, shader.Shade(Interpolate(a, b, c, b0, b1, b2)), options);
                }
            }
        }
    };
}
