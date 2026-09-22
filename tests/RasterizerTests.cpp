#include "TestSupport.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <future>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../graphics/Color.hpp"
#include "../graphics/Rasterizer.hpp"

namespace {
    struct TestShader {
        enum class Output { Color, UV, Tangent } Mode = Output::Color;
        std::atomic<int>* TransformCount = nullptr;
        bool ThrowOnShade = false;

        shader::Varyings Process(const shader::Vertex& input) const {
            if(TransformCount) ++*TransformCount;
            shader::Varyings output{};
            output.Pos = input.Pos;
            output.WorldPos = input.Pos;
            output.Normal = input.Normal;
            output.Color = input.Color;
            output.UV = input.UV;
            output.Tangent = input.Tangent;
            return output;
        }
        shader::FragmentOutput Shade(const shader::Fragment& fragment) const {
            if(ThrowOnShade) throw std::runtime_error("shader failure");
            auto color = fragment.Color;
            if(Mode == Output::UV) color = {fragment.UV.X, fragment.UV.Y, 0.f, 1.f};
            if(Mode == Output::Tangent)
                color = {std::abs(fragment.Tangent.W) == 1.f ? 1.f : 0.f, fragment.Tangent.Length(),
                         fragment.Tangent.W < 0.f ? 1.f : 0.f, 1.f};
            return {graphics::PackColor(color), fragment.Normal};
        }
    };

    // Depth-only coverage is a static contract: this shader intentionally has
    // no fragment callback or interpolated surface attributes.
    struct DepthOnlyShader {
        static constexpr bool DepthOnly = true;
        shader::Varyings Process(const shader::Vertex& input) const {
            shader::Varyings output{};
            output.Pos = input.Pos;
            return output;
        }
    };

    shader::Vertex Vertex(float x, float y, float depth = .5f, float w = 1.f,
                          math::Vector color = {1.f, 1.f, 1.f, 1.f}) {
        shader::Vertex vertex{};
        vertex.Pos = {x * w, y * w, depth * w, w};
        vertex.Normal = {0.f, 0.f, 1.f, 0.f};
        vertex.Color = color;
        vertex.Tangent = {1.f, 0.f, 0.f, 1.f};
        return vertex;
    }
    std::vector<shader::Vertex> Quad(float depth = .5f, math::Vector color = {1.f, 1.f, 1.f, 1.f}) {
        return {Vertex(-1.f, 1.f, depth, 1.f, color), Vertex(-1.f, -1.f, depth, 1.f, color),
                Vertex(1.f, 1.f, depth, 1.f, color), Vertex(1.f, -1.f, depth, 1.f, color)};
    }
    constexpr std::array<std::uint32_t, 6> QuadIndices{0, 1, 2, 2, 1, 3};
    constexpr std::array<std::uint32_t, 3> TriangleIndices{0, 1, 2};

    void TestExecutor() {
        ParallelExecutor executor(2);
        std::vector<int> visited(1025);
        executor.ParallelFor(0, visited.size(), [&](std::size_t i) { ++visited[i]; }, 7);
        CHECK(std::all_of(visited.begin(), visited.end(), [](int count) { return count == 1; }));

        std::atomic<int> nested{0};
        executor.ParallelFor(
            0, 32, [&](std::size_t) { executor.ParallelFor(0, 4, [&](std::size_t) { ++nested; }, 1); }, 1);
        CHECK(nested == 128);

        std::atomic<int> concurrent{0};
        auto first = std::async(std::launch::async,
                                [&] { executor.ParallelFor(0, 500, [&](std::size_t) { ++concurrent; }, 3); });
        executor.ParallelFor(0, 700, [&](std::size_t) { ++concurrent; }, 5);
        first.get();
        CHECK(concurrent == 1200);
        CHECK_THROWS(executor.ParallelFor(
            0, 10,
            [](std::size_t i) {
                if(i == 3) throw std::runtime_error("worker failure");
            },
            1));
        CHECK_THROWS(executor.ParallelFor(0, 1, [](std::size_t) {}, 0));
        std::atomic<int> recovered{0};
        executor.ParallelFor(0, 20, [&](std::size_t) { ++recovered; }, 1);
        CHECK(recovered == 20);

        ParallelExecutor serial(0);
        int serialCount = 0;
        serial.ParallelFor(0, 3, [&](std::size_t) { ++serialCount; });
        CHECK(serialCount == 3);
        CHECK_THROWS(serial.ParallelFor(0, 3, [](std::size_t i) {
            if(i == 1) throw std::runtime_error("serial failure");
        }));

        // No arithmetic overflow when a short iteration range ends at SIZE_MAX.
        std::atomic<int> nearLimit{0};
        executor.ParallelFor(
            std::numeric_limits<std::size_t>::max() - 3, std::numeric_limits<std::size_t>::max(),
            [&](std::size_t) { ++nearLimit; }, 64);
        CHECK(nearLimit == 3);
    }

    void TestColor() {
        CHECK(graphics::PackColor({1.f, 0.f, 0.f, 1.f}) == 0xffff0000u);
        CHECK(graphics::PackColor({.5f, .5f, .5f, 1.f}) == 0xff808080u);
        CHECK(graphics::PackColor({2.f, -1.f, 0.f, 1.f}) == 0xffff0000u);
        CHECK(graphics::AlphaBlend(0x80ffffffu, 0xff000000u) == 0xff808080u);
        CHECK(graphics::AlphaBlend(0x80ffffffu, 0xffffffffu) == 0xffffffffu);
        CHECK(graphics::AlphaBlend(0x80ff0000u, 0u) == 0x80ff0000u);
        CHECK(graphics::AlphaBlend(0x00ffffffu, 0x12345678u) == 0x12345678u);
    }

    void TestCoverageAndDepth() {
        ParallelExecutor executor(2);
        graphics::Rasterizer rasterizer(executor);
        graphics::FrameBuffer frame(67, 35);
        auto vertices = Quad(.5f, {1.f, 1.f, 1.f, 128.f / 255.f});
        graphics::RasterizerOptions blend;
        blend.Blend = true;
        blend.DepthWrite = false;
        rasterizer.Render(frame, TestShader{}, vertices, QuadIndices, blend);
        // Shared diagonal and tile boundaries must have exactly one owner.
        for(const auto color : frame.GetColors()) CHECK(color == 0xff808080u);
        for(const auto depth : frame.GetDepths()) CHECK(depth == 1.f);

        frame.Clear();
        rasterizer.Render(frame, TestShader{}, Quad(.8f, {0.f, 0.f, 1.f, 1.f}), QuadIndices);
        rasterizer.Render(frame, TestShader{}, Quad(.2f, {1.f, 0.f, 0.f, 1.f}), QuadIndices);
        rasterizer.Render(frame, TestShader{}, Quad(.7f, {0.f, 1.f, 0.f, 1.f}), QuadIndices);
        CHECK(frame.GetPixel(20, 10) == 0xffff0000u);
        CHECK_NEAR(frame.GetDepth(20, 10), .2f, 1e-6f);
        CHECK_NEAR(frame.GetNormal(20, 10).Z, 1.f, 1e-6f);

        frame.Clear();
        const std::array<std::uint32_t, 6> reversed{0, 2, 1, 2, 3, 1};
        rasterizer.Render(frame, TestShader{}, Quad(), reversed);
        CHECK(frame.GetPixel(20, 10) == 0xff000000u);
        graphics::RasterizerOptions twoSided;
        twoSided.Cull = graphics::CullMode::None;
        rasterizer.Render(frame, TestShader{}, Quad(), reversed, twoSided);
        CHECK(frame.GetPixel(20, 10) == 0xffffffffu);

        auto reflected = Quad();
        for(auto& vertex : reflected) vertex.Pos.X = -vertex.Pos.X;
        frame.Clear();
        rasterizer.Render(frame, TestShader{}, reflected, QuadIndices);
        CHECK(frame.GetPixel(20, 10) == 0xff000000u);
        graphics::RasterizerOptions reflection;
        reflection.FlipWinding = true;
        rasterizer.Render(frame, TestShader{}, reflected, QuadIndices, reflection);
        for(const auto pixel : frame.GetColors()) CHECK(pixel == 0xffffffffu);

        frame.Clear();
        rasterizer.Render(frame, TestShader{}, Quad(.1f, {1.f, 0.f, 0.f, 0.f}), QuadIndices);
        CHECK(frame.GetPixel(20, 10) == 0xff000000u);
        CHECK(frame.GetDepth(20, 10) == 1.f);
    }

    void TestClippingAndDrawContract() {
        ParallelExecutor executor(2);
        graphics::Rasterizer rasterizer(executor);
        graphics::FrameBuffer frame(16, 16);
        for(int plane = 0; plane < 6; ++plane) {
            auto vertices = Quad();
            for(auto& vertex : vertices) {
                if(plane == 0) vertex.Pos.X -= 3.f;
                if(plane == 1) vertex.Pos.X += 3.f;
                if(plane == 2) vertex.Pos.Y -= 3.f;
                if(plane == 3) vertex.Pos.Y += 3.f;
                if(plane == 4) vertex.Pos.Z = -.1f;
                if(plane == 5) vertex.Pos.Z = 1.1f;
            }
            frame.Clear();
            rasterizer.Render(frame, TestShader{}, vertices, QuadIndices);
            for(const auto pixel : frame.GetColors()) CHECK(pixel == 0xff000000u);
        }

        auto crossing = std::vector{Vertex(-1.f, 1.f, -.5f), Vertex(-1.f, -1.f, .5f), Vertex(1.f, 1.f, .5f)};
        frame.Clear();
        rasterizer.Render(frame, TestShader{}, crossing, TriangleIndices);
        CHECK(frame.GetPixel(0, 0) == 0xff000000u);
        CHECK(frame.GetPixel(7, 7) == 0xffffffffu);
        for(const auto depth : frame.GetDepths()) CHECK(depth >= 0.f && depth <= 1.f);

        // Clipping must preserve the original linear attribute field. Compare
        // with an explicitly cut polygon whose intersection normals are not
        // normalized until the fragment stage.
        crossing[0].Normal = {1.f, 0.f, 0.f, 0.f};
        crossing[1].Normal = {0.f, 1.f, 0.f, 0.f};
        crossing[2].Normal = {0.f, 0.f, 1.f, 0.f};
        const auto middle = [](const shader::Vertex& a, const shader::Vertex& b) {
            auto vertex = a;
            vertex.Pos = (a.Pos + b.Pos) * .5f;
            vertex.Normal = (a.Normal + b.Normal) * .5f;
            vertex.Tangent = (a.Tangent + b.Tangent) * .5f;
            return vertex;
        };
        const std::vector cut{middle(crossing[2], crossing[0]), middle(crossing[1], crossing[0]), crossing[1],
                              crossing[2]};
        const std::array<std::uint32_t, 6> cutIndices{0, 1, 2, 0, 2, 3};
        graphics::FrameBuffer reference(16, 16);
        frame.Clear();
        rasterizer.Render(frame, TestShader{}, crossing, TriangleIndices);
        rasterizer.Render(reference, TestShader{}, cut, cutIndices);
        for(std::size_t i = 0; i < frame.GetNormals().size(); ++i) {
            CHECK_NEAR(frame.GetNormals()[i].X, reference.GetNormals()[i].X, 1e-6f);
            CHECK_NEAR(frame.GetNormals()[i].Y, reference.GetNormals()[i].Y, 1e-6f);
            CHECK_NEAR(frame.GetNormals()[i].Z, reference.GetNormals()[i].Z, 1e-6f);
        }

        // Large offscreen vertices are clipped before fixed-point conversion.
        frame.Clear();
        auto huge = Quad();
        for(auto& vertex : huge) {
            vertex.Pos.X *= 1e8f;
            vertex.Pos.Y *= 1e8f;
        }
        rasterizer.Render(frame, TestShader{}, huge, QuadIndices);
        CHECK(frame.GetPixel(8, 8) == 0xffffffffu);

        auto vertices = Quad();
        std::atomic<int> transformed{0};
        TestShader counted;
        counted.TransformCount = &transformed;
        frame.Clear();
        rasterizer.Render(frame, counted, vertices, QuadIndices);
        CHECK(transformed == 4);
        const std::array<std::uint32_t, 3> badIndex{0, 1, 99};
        CHECK_THROWS(rasterizer.Render(frame, TestShader{}, vertices, badIndex));
        const std::array<std::uint32_t, 2> incomplete{0, 1};
        CHECK_THROWS(rasterizer.Render(frame, TestShader{}, vertices, incomplete));
        vertices[0].Pos.X = std::numeric_limits<float>::quiet_NaN();
        CHECK_THROWS(rasterizer.Render(frame, TestShader{}, vertices, QuadIndices));
        TestShader failing;
        failing.ThrowOnShade = true;
        frame.Clear();
        CHECK_THROWS(rasterizer.Render(frame, failing, Quad(), QuadIndices));
        frame.Clear();
        rasterizer.Render(frame, TestShader{}, Quad(), QuadIndices);
        CHECK(frame.GetPixel(8, 8) == 0xffffffffu);

        // A different target extent on the next call cannot use stale tiles.
        graphics::FrameBuffer small(3, 2);
        rasterizer.Render(small, TestShader{}, Quad(), QuadIndices);
        for(const auto pixel : small.GetColors()) CHECK(pixel == 0xffffffffu);
    }

    void TestPerspectiveAndTangents() {
        ParallelExecutor executor(0);
        graphics::Rasterizer rasterizer(executor);
        graphics::FrameBuffer frame(4, 4);
        std::vector vertices{Vertex(-1.f, 1.f, .2f, 1.f), Vertex(-1.f, -1.f, .6f, 2.f), Vertex(1.f, 1.f, .8f, 4.f)};
        vertices[0].UV = {0.f, 0.f, 0.f, 0.f};
        vertices[1].UV = {1.f, 0.f, 0.f, 0.f};
        vertices[2].UV = {0.f, 1.f, 0.f, 0.f};
        TestShader shader;
        shader.Mode = TestShader::Output::UV;
        rasterizer.Render(frame, shader, vertices, TriangleIndices);
        // At sample (.5,.5), screen weights are (.75,.125,.125).
        const float denominator = .75f + .125f / 2.f + .125f / 4.f;
        CHECK(frame.GetPixel(0, 0) ==
              graphics::PackColor({(.125f / 2.f) / denominator, (.125f / 4.f) / denominator, 0.f, 1.f}));
        CHECK_NEAR(frame.GetDepth(0, 0), .75f * .2f + .125f * .6f + .125f * .8f, 1e-6f);

        vertices = {Vertex(-1.f, 1.f), Vertex(-1.f, -1.f), Vertex(1.f, 1.f)};
        vertices[0].Tangent = {1.f, 0.f, 0.f, -1.f};
        vertices[1].Tangent = {0.f, 1.f, 0.f, -1.f};
        vertices[2].Tangent = {1.f, 0.f, 0.f, -1.f};
        shader.Mode = TestShader::Output::Tangent;
        frame.Clear();
        rasterizer.Render(frame, shader, vertices, TriangleIndices);
        CHECK(frame.GetPixel(0, 1) == 0xffffffffu); // XYZ unit length, W exactly -1.
    }

    void TestTransparencyAndModes() {
        ParallelExecutor serial(0), parallel(2);
        graphics::Rasterizer one(serial), many(parallel);
        graphics::FrameBuffer a(67, 35), b(67, 35);
        std::vector<shader::Vertex> vertices;
        std::vector<std::uint32_t> indices;
        // Deliberately near-to-far input, across multiple primitive work chunks and
        // several tiles. Sorting and binning must not depend on worker timing.
        for(int layer = 0; layer < 140; ++layer) {
            const auto quad = Quad(.1f + layer * .005f, {layer % 2 ? 1.f : 0.f, 0.f, layer % 2 ? 0.f : 1.f, .05f});
            const auto offset = static_cast<std::uint32_t>(vertices.size());
            vertices.insert(vertices.end(), quad.begin(), quad.end());
            for(auto index : QuadIndices) indices.push_back(offset + index);
        }
        graphics::RasterizerOptions blend;
        blend.Blend = true;
        blend.DepthWrite = false;
        one.Render(a, TestShader{}, vertices, indices, blend);
        many.Render(b, TestShader{}, vertices, indices, blend);
        CHECK(std::equal(a.GetColors().begin(), a.GetColors().end(), b.GetColors().begin()));
        auto expected = 0xff000000u;
        for(int layer = 139; layer >= 0; --layer)
            expected = graphics::AlphaBlend(
                graphics::PackColor({layer % 2 ? 1.f : 0.f, 0.f, layer % 2 ? 0.f : 1.f, .05f}), expected);
        CHECK(a.GetPixel(20, 10) == expected);

        graphics::FrameBuffer pointFrame(8, 8);
        std::vector points{Vertex(0.f, 0.f, .2f, 1.f, {1.f, 0.f, 0.f, 1.f}),
                           Vertex(0.f, 0.f, .8f, 1.f, {0.f, 0.f, 1.f, 1.f})};
        std::array<std::uint32_t, 2> pointIndices{0, 1};
        graphics::RasterizerOptions pointOptions;
        pointOptions.Primitive = graphics::PrimitiveType::Points;
        many.Render(pointFrame, TestShader{}, points, pointIndices, pointOptions);
        CHECK(pointFrame.GetPixel(4, 4) == 0xffff0000u);
        CHECK_NEAR(pointFrame.GetDepth(4, 4), .2f, 1e-6f);

        graphics::FrameBuffer lineFrame(8, 8);
        graphics::RasterizerOptions lineOptions;
        lineOptions.Primitive = graphics::PrimitiveType::Lines;
        many.Render(lineFrame, TestShader{}, Quad(.2f, {1.f, 0.f, 0.f, 1.f}), QuadIndices, lineOptions);
        many.Render(lineFrame, TestShader{}, Quad(.8f, {0.f, 0.f, 1.f, 1.f}), QuadIndices, lineOptions);
        CHECK(lineFrame.GetPixel(0, 4) == 0xffff0000u);
        CHECK_NEAR(lineFrame.GetDepth(0, 4), .2f, 1e-6f);
    }

    void TestDepthOnlyAndConcurrentTargets() {
        ParallelExecutor executor(2);
        graphics::Rasterizer rasterizer(executor);
        graphics::FrameBuffer expected(67, 35), depthOnly(67, 35);
        // Exercise both cached inside vertices and a near-plane intersection.
        const std::vector vertices{Vertex(-1.3f, .9f, -.2f, 1.f), Vertex(-.7f, -.9f, .7f, 2.f),
                                   Vertex(.9f, .7f, .5f, 4.f),    Vertex(-.8f, .8f, .4f),
                                   Vertex(-.8f, -.8f, .4f),       Vertex(.8f, .8f, .4f)};
        const std::array<std::uint32_t, 6> indices{0, 1, 2, 3, 4, 5};
        graphics::RasterizerOptions options;
        options.Cull = graphics::CullMode::None;
        rasterizer.Render(expected, TestShader{}, vertices, indices, options);
        depthOnly.Clear(0xff123456u);
        for(std::uint32_t y = 0; y < depthOnly.GetHeight(); ++y)
            for(std::uint32_t x = 0; x < depthOnly.GetWidth(); ++x) depthOnly.SetNormal(x, y, {1.f, 2.f, 3.f, 0.f});
        rasterizer.Render(depthOnly, DepthOnlyShader{}, vertices, indices, options);
        CHECK(std::equal(expected.GetDepths().begin(), expected.GetDepths().end(), depthOnly.GetDepths().begin()));
        CHECK(std::any_of(depthOnly.GetDepths().begin(), depthOnly.GetDepths().end(),
                          [](float depth) { return depth < 1.f; }));
        for(std::size_t i = 0; i < depthOnly.GetColors().size(); ++i) {
            CHECK(depthOnly.GetColors()[i] == 0xff123456u);
            CHECK(depthOnly.GetNormals()[i].X == 1.f && depthOnly.GetNormals()[i].Y == 2.f &&
                  depthOnly.GetNormals()[i].Z == 3.f);
        }

        depthOnly.Clear(0xff123456u);
        options.DepthWrite = false;
        rasterizer.Render(depthOnly, DepthOnlyShader{}, vertices, indices, options);
        for(const auto depth : depthOnly.GetDepths()) CHECK(depth == 1.f);
        for(const auto color : depthOnly.GetColors()) CHECK(color == 0xff123456u);

        options.DepthWrite = true;
        options.Primitive = graphics::PrimitiveType::Lines;
        rasterizer.Render(depthOnly, DepthOnlyShader{}, Quad(.3f), QuadIndices, options);
        CHECK_NEAR(depthOnly.GetDepth(0, 16), .3f, 1e-6f);
        CHECK(depthOnly.GetPixel(0, 16) == 0xff123456u);
        options.Primitive = graphics::PrimitiveType::Points;
        const std::array point{Vertex(0.f, 0.f, .1f)};
        const std::array<std::uint32_t, 1> pointIndex{0};
        rasterizer.Render(depthOnly, DepthOnlyShader{}, point, pointIndex, options);
        CHECK_NEAR(depthOnly.GetDepth(33, 17), .1f, 1e-6f);

        // One rasterizer can prepare and bin concurrent draws of different
        // extents. Local vertex/bin caches must not leak across their lifetimes.
        graphics::FrameBuffer first(67, 35), second(35, 67), secondReference(35, 67);
        options.Primitive = graphics::PrimitiveType::Triangles;
        rasterizer.Render(secondReference, TestShader{}, vertices, indices, options);
        auto concurrent =
            std::async(std::launch::async, [&] { rasterizer.Render(first, TestShader{}, vertices, indices, options); });
        rasterizer.Render(second, TestShader{}, vertices, indices, options);
        concurrent.get();
        CHECK(std::equal(expected.GetColors().begin(), expected.GetColors().end(), first.GetColors().begin()));
        CHECK(std::equal(expected.GetDepths().begin(), expected.GetDepths().end(), first.GetDepths().begin()));
        CHECK(std::equal(secondReference.GetColors().begin(), secondReference.GetColors().end(),
                         second.GetColors().begin()));
        CHECK(std::equal(secondReference.GetDepths().begin(), secondReference.GetDepths().end(),
                         second.GetDepths().begin()));
    }

    void TestClippedVertexPoolGrowth() {
        ParallelExecutor serial(0), parallel(2);
        graphics::Rasterizer referenceRasterizer(serial), rasterizer(parallel);
        graphics::FrameBuffer expected(65, 49), actual(65, 49);
        graphics::RasterizerOptions options;
        options.Cull = graphics::CullMode::None;
        options.Blend = true;
        options.DepthWrite = false;
        std::vector<shader::Vertex> vertices;
        std::vector<std::uint32_t> indices;
        const auto append = [&](std::span<const shader::Vertex> drawVertices,
                                std::span<const std::uint32_t> drawIndices) {
            const auto first = static_cast<std::uint32_t>(vertices.size());
            vertices.insert(vertices.end(), drawVertices.begin(), drawVertices.end());
            for(const auto index : drawIndices) indices.push_back(first + index);
            referenceRasterizer.Render(expected, TestShader{}, drawVertices, drawIndices, options);
        };
        // Store cached inside vertices before many clipped polygons. Their
        // generated vertices outnumber the inputs and grow the backing pool
        // several times; earlier triangle handles must remain valid throughout.
        append(Quad(.95f, {1.f, 0.f, 0.f, .25f}), QuadIndices);
        for(int layer = 0; layer < 48; ++layer) {
            const float depth = .85f - layer * .01f;
            const math::Vector color(layer % 3 == 0 ? 1.f : 0.f, layer % 3 == 1 ? 1.f : 0.f, layer % 3 == 2 ? 1.f : 0.f,
                                     .15f);
            const std::array clipped{Vertex(-2.5f, 1.8f, depth, 1.f, color), Vertex(-2.4f, -2.2f, depth, 2.f, color),
                                     Vertex(2.6f, 1.7f, depth, 3.f, color)};
            append(clipped, TriangleIndices);
        }
        append(Quad(.1f, {0.f, 1.f, 1.f, .25f}), QuadIndices);
        rasterizer.Render(actual, TestShader{}, vertices, indices, options);
        CHECK(std::equal(expected.GetColors().begin(), expected.GetColors().end(), actual.GetColors().begin()));
        CHECK(actual.GetPixel(20, 20) != 0xff000000u);
        for(const auto depth : actual.GetDepths()) CHECK(depth == 1.f);
    }
}

void RunRasterizerTests() {
    TestExecutor();
    TestColor();
    TestCoverageAndDepth();
    TestClippingAndDrawContract();
    TestPerspectiveAndTangents();
    TestTransparencyAndModes();
    TestDepthOnlyAndConcurrentTargets();
    TestClippedVertexPoolGrowth();
}
