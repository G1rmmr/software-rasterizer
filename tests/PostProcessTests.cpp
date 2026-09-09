#include "../graphics/post/AntiAliasingPass.hpp"
#include "../graphics/post/SsaoPass.hpp"
#include "TestSupport.hpp"

#include <algorithm>
#include <array>
#include <vector>

namespace {
    void FillPlane(graphics::FrameBuffer& frame, const math::Matrix& projection, float distance,
                   std::uint32_t color = 0xffc08040u) {
        const auto clip = projection * math::Vector(0.f, 0.f, -distance, 1.f);
        frame.Clear(color);
        for(std::uint32_t y = 0; y < frame.GetHeight(); ++y)
            for(std::uint32_t x = 0; x < frame.GetWidth(); ++x) {
                frame.SetDepth(x, y, clip.Z / clip.W);
                frame.SetNormal(x, y, {0.f, 0.f, 1.f});
            }
    }
}

void RunPostProcessTests() {
    ParallelExecutor executor(0);
    graphics::SsaoPass ssao(executor);
    graphics::AntiAliasingPass antialiasing(executor);
    const auto projection = math::CreatePerspective(math::ToRadian(60.f), 1.f, .1f, 100.f);
    const auto inverse = projection.Inv();
    graphics::FrameBuffer frame(7, 5);
    // A single plane cannot occlude itself. This exercises all image borders, odd sizes,
    // near/far depth reconstruction and cached scratch buffers after a resize.
    for(const auto dimensions : {std::array<unsigned, 2>{7, 5}, {1, 1}, {8, 6}, {5, 7}}) {
        frame.Resize(dimensions[0], dimensions[1]);
        for(float distance : {.5f, 5.f, 45.f}) {
            FillPlane(frame, projection, distance);
            ssao.Execute(frame, projection, inverse);
            for(const auto color : frame.GetColors()) CHECK(color == 0xffc08040u);
        }
        frame.Clear(0xff70a0b0u);
        ssao.Execute(frame, projection, inverse);
        for(const auto color : frame.GetColors()) CHECK(color == 0xff70a0b0u);
    }
    graphics::SsaoSettings invalidAo;
    invalidAo.Radius = 0.f;
    CHECK_THROWS(ssao.Execute(frame, projection, inverse, invalidAo));
    invalidAo = {};
    invalidAo.KernelSize = 0;
    CHECK_THROWS(ssao.Execute(frame, projection, inverse, invalidAo));

    frame.Resize(65, 33);
    const auto ortho = math::CreateOrtho(-2.f, 2.f, -1.f, 1.f, .1f, 20.f);
    const auto orthoInverse = ortho.Inv();
    FillPlane(frame, ortho, 5.f, 0xffffffffu);
    // A nearer parallel slab must occlude the farther plane near their screen-space join.
    // Its own front face stays unoccluded: filtering must not bleed foreign AO across the depth step.
    const float foregroundDepth = (ortho * math::Vector(0.f, 0.f, -4.7f, 1.f)).Z;
    for(unsigned y = 0; y < frame.GetHeight(); ++y)
        for(unsigned x = 33; x < frame.GetWidth(); ++x) frame.SetDepth(x, y, foregroundDepth);
    graphics::SsaoSettings settings;
    settings.KernelSize = 64;
    settings.Strength = 4.f;
    ssao.Execute(frame, ortho, orthoInverse, settings);
    bool occluded = false;
    for(unsigned y = 4; y + 4 < frame.GetHeight(); ++y) {
        for(unsigned x = 24; x < 33; ++x) occluded = occluded || frame.GetPixel(x, y) != 0xffffffffu;
        for(unsigned x = 33; x < frame.GetWidth(); ++x) CHECK(frame.GetPixel(x, y) == 0xffffffffu);
    }
    CHECK(occluded);

    // Color-only edges on a flat surface respect their own threshold.
    frame.Resize(5, 5);
    FillPlane(frame, projection, 5.f, 0xff000000u);
    frame.SetPixel(2, 2, 0x80404040u);
    graphics::AntiAliasingSettings aa;
    aa.ColorThreshold = 64;
    antialiasing.Execute(frame, aa);
    CHECK(frame.GetPixel(2, 2) == 0x80404040u);
    aa.ColorThreshold = 30;
    antialiasing.Execute(frame, aa);
    CHECK(frame.GetPixel(2, 2) == 0x800c0c0cu);
    CHECK(frame.GetPixel(0, 0) == 0xff000000u);
    aa.ColorThreshold = 256;
    CHECK_THROWS(antialiasing.Execute(frame, aa));

    // Empty background normals do not mark every background pixel as a geometric edge.
    frame.Clear(0xff101010u);
    frame.SetPixel(2, 2, 0xff202020u);
    aa = {};
    aa.ColorThreshold = 255;
    antialiasing.Execute(frame, aa);
    CHECK(frame.GetPixel(2, 2) == 0xff202020u);
    frame.Resize(1, 1);
    frame.Clear(0x11223344u);
    antialiasing.Execute(frame);
    CHECK(frame.GetPixel(0, 0) == 0x11223344u);
    // Normalization is cached per frame, not assumed to have been done by callers.
    frame.Resize(5, 5);
    frame.Clear(0xff000000u);
    frame.SetPixel(2, 2, 0xffffffffu);
    for(unsigned y = 0; y < 5; ++y)
        for(unsigned x = 0; x < 5; ++x) frame.SetNormal(x, y, {0.f, 0.f, 8.f});
    aa = {};
    aa.ColorThreshold = 255;
    antialiasing.Execute(frame, aa);
    CHECK(frame.GetPixel(2, 2) == 0xffffffffu);
    frame.SetNormal(2, 1, {8.f, 0.f, 0.f});
    antialiasing.Execute(frame, aa);
    CHECK(frame.GetPixel(2, 2) == 0xff333333u);

    // A flat RGB region remains unchanged across geometric edges and varying alpha.
    frame.Clear(0xff112233u);
    frame.SetPixel(2, 2, 0x80112233u);
    frame.SetDepth(2, 2, .1f);
    frame.SetDepth(2, 1, .9f);
    frame.SetNormal(2, 2, {1.f, 0.f, 0.f});
    frame.SetNormal(2, 1, {0.f, 0.f, 1.f});
    antialiasing.Execute(frame);
    CHECK(frame.GetPixel(2, 2) == 0x80112233u);

    // Warm caches must agree byte-for-byte with a fresh pass after projection,
    // kernel and shape changes, including equal pixel counts with different shapes.
    ParallelExecutor parallelExecutor(2);
    graphics::SsaoPass parallelSsao(parallelExecutor);
    for(unsigned variant = 0; variant < 3; ++variant) {
        const unsigned width = variant == 1 ? 33u : 65u, height = variant == 1 ? 65u : 33u;
        frame.Resize(width, height);
        graphics::FrameBuffer coldFrame(width, height), parallelFrame(width, height);
        const auto changedProjection = math::CreateOrtho(-2.f - variant, 2.f + variant, -1.f, 1.f, .1f, 20.f + variant);
        const auto changedInverse = changedProjection.Inv();
        const auto populate = [&](graphics::FrameBuffer& target) {
            FillPlane(target, changedProjection, 5.f, 0xffd0b090u);
            const float nearer = (changedProjection * math::Vector(0.f, 0.f, -4.7f, 1.f)).Z;
            for(unsigned y = 0; y < height; ++y)
                for(unsigned x = width / 2; x < width; ++x) target.SetDepth(x, y, nearer);
        };
        populate(frame);
        populate(coldFrame);
        populate(parallelFrame);
        settings.KernelSize = variant == 1 ? 8 : 32;
        graphics::SsaoPass coldSsao(executor);
        ssao.Execute(frame, changedProjection, changedInverse, settings);
        coldSsao.Execute(coldFrame, changedProjection, changedInverse, settings);
        parallelSsao.Execute(parallelFrame, changedProjection, changedInverse, settings);
        CHECK(std::equal(frame.GetColors().begin(), frame.GetColors().end(), coldFrame.GetColors().begin()));
        CHECK(std::equal(frame.GetColors().begin(), frame.GetColors().end(), parallelFrame.GetColors().begin()));
    }
    // Byte fixtures were recorded from the unmasked bilateral blur in both MSVC
    // Debug and Release. They cover masks ending at / crossing bit 63, clipped
    // image neighborhoods, partial words and reuse after differently sized frames.
    struct MaskFixture {
        unsigned Width, Height, SlabX;
        std::uint64_t Expected;
    };
    const std::array<MaskFixture, 11> maskFixtures = {{{1, 1, 0, 0x97f29c2e95e107e8ull},
                                                       {3, 3, 0, 0x7ef3d044df4b9128ull},
                                                       {125, 17, 120, 0x740d6bea559c096eull},
                                                       {127, 17, 123, 0xb7396aa643864ba8ull},
                                                       {128, 17, 124, 0x4a3e3ca2690436abull},
                                                       {129, 17, 125, 0x1ffdd045e93cc913ull},
                                                       {131, 17, 127, 0x4ff38fe1a915fbdeull},
                                                       {255, 17, 126, 0xe25a3b2cf5be045aull},
                                                       {257, 17, 129, 0x65d202986c4792d2ull},
                                                       {129, 3, 125, 0x93b51b9cf51a94cdull},
                                                       {1, 1, 0, 0x97f29c2e95e107e8ull}}};
    const auto maskProjection = math::CreateOrtho(-2.f, 2.f, -1.f, 1.f, .1f, 20.f);
    const auto maskInverse = maskProjection.Inv();
    const float farMaskDepth = (maskProjection * math::Vector(0.f, 0.f, -5.f, 1.f)).Z;
    const float nearMaskDepth = (maskProjection * math::Vector(0.f, 0.f, -4.7f, 1.f)).Z;
    graphics::SsaoSettings maskSettings;
    maskSettings.KernelSize = 32;
    maskSettings.Strength = 4.f;
    for(const auto& fixture : maskFixtures) {
        frame.Resize(fixture.Width, fixture.Height);
        frame.Clear(0xffd0b090u);
        for(unsigned y = 0; y < fixture.Height; ++y)
            for(unsigned x = 0; x < fixture.Width; ++x) {
                frame.SetDepth(x, y, x >= fixture.SlabX && x < fixture.SlabX + 3u ? nearMaskDepth : farMaskDepth);
                frame.SetNormal(x, y, {0.f, 0.f, 1.f});
            }
        parallelSsao.Execute(frame, maskProjection, maskInverse, maskSettings);
        std::uint64_t hash = 14695981039346656037ull;
        for(const auto color : frame.GetColors())
            for(unsigned shift : {0u, 8u, 16u, 24u}) {
                hash ^= (color >> shift) & 255u;
                hash *= 1099511628211ull;
            }
        CHECK(hash == fixture.Expected);
    }
}
