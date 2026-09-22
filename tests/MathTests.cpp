#include "../graphics/Frustum.hpp"
#include "../math/Math.hpp"
#include "TestSupport.hpp"

#include <array>
#include <limits>

void RunMathTests() {
    const math::Quaternion identity;
    CHECK_NEAR(identity.Dot(identity), 1.f, 1e-6);
    CHECK_NEAR(identity.Length(), 1.f, 1e-6);
    CHECK(identity.Slerp(identity, .5f) == identity);
    CHECK(identity.Slerp(identity * -1.f, .5f) == identity);
    CHECK_NEAR(math::Quaternion(0.f, 0.f, 0.f, 0.f).Norm().W, 1.f, 1e-6);
    const auto rotation = math::FromAxisAngle({0.f, 1.f, 0.f}, math::ToRadian(90.f));
    const auto halfway = (identity * 3.f).Slerp(rotation * 7.f, .5f);
    CHECK_NEAR(halfway.Length(), 1.f, 1e-6);
    const auto rotated = halfway.ToMatrix() * math::Vector(1.f, 0.f, 0.f, 0.f);
    CHECK_NEAR(rotated.X, std::sqrt(.5f), 1e-5);
    CHECK_NEAR(rotated.Z, -std::sqrt(.5f), 1e-5);
    CHECK(math::FromAxisAngle({}, 1.f) == identity);

    const auto tangent = math::Vector(.5f, .5f, 0.f, -1.f).NormalizedXYZ();
    CHECK_NEAR(tangent.Length(), 1.f, 1e-6);
    CHECK_NEAR(tangent.W, -1.f, 1e-6);
    CHECK_NEAR(tangent.NormalizedDirection().W, 0.f, 1e-6);
    CHECK_NEAR(math::Vector(std::numeric_limits<float>::infinity(), 0.f, 0.f).Norm().Length(), 0.f, 1e-6);
    CHECK_NEAR(math::Vector{}.Norm().Length(), 0.f, 1e-6);

    const math::Vector shuffled(
        simd::Shuffle<SIMD_MASK(3, 2, 1, 0)>(simd::Set(1.f, 2.f, 3.f, 4.f), simd::Set(10.f, 20.f, 30.f, 40.f)));
    CHECK(shuffled == math::Vector(1.f, 2.f, 30.f, 40.f));
    const math::Vector dot3(simd::HorizonSum<0x71>(simd::Set(1.f, 2.f, 3.f, 4.f), simd::Set(1.f)));
    CHECK(dot3 == math::Vector(6.f, 0.f, 0.f, 0.f));
    CHECK(simd::PackRGBA(simd::Set(.5f, 1.5f, 254.9f, 999.f)) == 0xfffe0100u);
    CHECK(simd::PackRGBA(simd::Set(-10.f, 0.f, 255.f, 255.f)) == 0xffff0000u);

    const auto perspective = math::CreatePerspective(math::ToRadian(60.f), 1.6f, .1f, 100.f);
    const auto inverse = perspective.Inv();
    for(const float distance : {.1f, .5f, 5.f, 45.f, 100.f}) {
        const math::Vector original(.125f * distance, -.2f * distance, -distance, 1.f);
        const auto clip = perspective * original;
        const auto ndc = clip / clip.W;
        const auto homogeneous = inverse * ndc;
        const auto restored = homogeneous / homogeneous.W;
        CHECK_NEAR(restored.X, original.X, .02f);
        CHECK_NEAR(restored.Y, original.Y, .02f);
        CHECK_NEAR(restored.Z, original.Z, .02f);
        CHECK(ndc.Z >= -1e-6f && ndc.Z <= 1.f + 1e-6f);
    }
    const auto nearClip = perspective * math::Vector(0.f, 0.f, -.1f, 1.f);
    const auto farClip = perspective * math::Vector(0.f, 0.f, -100.f, 1.f);
    CHECK_NEAR(nearClip.Z / nearClip.W, 0.f, 1e-6);
    CHECK_NEAR(farClip.Z / farClip.W, 1.f, 1e-6);
    const auto ortho = math::CreateOrtho(-2.f, 6.f, -3.f, 5.f, 2.f, 12.f);
    CHECK(ortho * math::Vector(-2.f, -3.f, -2.f, 1.f) == math::Vector(-1.f, -1.f, 0.f, 1.f));
    CHECK(ortho * math::Vector(6.f, 5.f, -12.f, 1.f) == math::Vector(1.f, 1.f, 1.f, 1.f));
    const auto shadow = math::CreateOrtho(-15.f, 15.f, -15.f, 15.f, -200.f, 200.f);
    CHECK_NEAR((shadow * math::Vector(0.f, 0.f, 200.f, 1.f)).Z, 0.f, 1e-6);
    CHECK_NEAR((shadow * math::Vector(0.f, 0.f, -200.f, 1.f)).Z, 1.f, 1e-6);
    CHECK_THROWS(math::CreatePerspective(0.f, 1.f, .1f, 10.f));
    CHECK_THROWS(math::CreatePerspective(1.f, 0.f, .1f, 10.f));
    CHECK_THROWS(math::CreatePerspective(1.f, 1.f, 10.f, 1.f));
    CHECK_THROWS(math::CreateOrtho(1.f, 1.f, -1.f, 1.f, 0.f, 1.f));

    const auto view = math::CreateLookAt({10.f, 3.f, 6.f}, {10.f, 3.f, 0.f}, {0.f, 1.f, 0.f});
    graphics::Frustum frustum(perspective * view);
    CHECK(frustum.IsSphereInside({10.f, 3.f, 0.f}, .2f));
    CHECK(!frustum.IsSphereInside({10.f, 3.f, 7.f}, .2f));
    CHECK(!frustum.IsSphereInside({100.f, 3.f, 0.f}, .2f));
    CHECK(!frustum.IsSphereInside({10.f, 3.f, -110.f}, .2f));
    CHECK_THROWS(frustum.Update(math::Matrix(0.f)));
    CHECK(frustum.IsSphereInside({10.f, 3.f, 0.f}, .2f));
    graphics::Frustum defaultFrustum;
    CHECK(defaultFrustum.IsSphereInside({0.f, 0.f, .5f}, 0.f));
    CHECK(!defaultFrustum.IsSphereInside({0.f, 0.f, -.1f}, 0.f));
}
