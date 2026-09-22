#include "../AppState.hpp"
#include "TestSupport.hpp"

#include <limits>

void RunAppStateTests() {
    app::AppState first(800, 600);
    app::AppState second(320, 240);
    CHECK(first.IsRunning() && second.IsRunning());
    CHECK_NEAR(first.GetCameraDistance(), 45.f, 1e-6);
    CHECK_NEAR(first.GetLightDirection().Length(), 1.f, 1e-6);
    CHECK_THROWS(app::AppState(0, 600));
    CHECK_THROWS(app::AppState(800, 0));

    first.Apply(app::Action::CyclePrimitive);
    CHECK(first.GetRenderSettings().Primitive == graphics::PrimitiveType::Lines);
    first.Apply(app::Action::CyclePrimitive);
    CHECK(first.GetRenderSettings().Primitive == graphics::PrimitiveType::Points);
    first.Apply(app::Action::CyclePrimitive);
    CHECK(first.GetRenderSettings().Primitive == graphics::PrimitiveType::Triangles);
    CHECK(second.GetRenderSettings().Primitive == graphics::PrimitiveType::Triangles);
    first.Apply(app::Action::ToggleShadows);
    first.Apply(app::Action::ToggleSsao);
    first.Apply(app::Action::ToggleAA);
    first.Apply(app::Action::ToggleToon);
    CHECK(!first.GetRenderSettings().Shadows);
    CHECK(!first.GetRenderSettings().AmbientOcclusion);
    CHECK(!first.GetRenderSettings().AntiAliasing);
    CHECK(first.GetRenderSettings().Toon);
    CHECK(second.GetRenderSettings().Shadows && second.GetRenderSettings().AmbientOcclusion &&
          second.GetRenderSettings().AntiAliasing && !second.GetRenderSettings().Toon);
    first.Apply(app::Action::ToggleShadows);
    CHECK(first.GetRenderSettings().Shadows);

    first.Scroll(1.f);
    CHECK_NEAR(first.GetCameraDistance(), 43.f, 1e-6);
    first.Scroll(-2.f);
    CHECK_NEAR(first.GetCameraDistance(), 47.f, 1e-6);
    first.Scroll(1000.f);
    CHECK_NEAR(first.GetCameraDistance(), 1.f, 1e-6);
    first.Scroll(-1000.f);
    CHECK_NEAR(first.GetCameraDistance(), 200.f, 1e-6);
    first.Scroll(std::numeric_limits<float>::quiet_NaN());
    first.Scroll(std::numeric_limits<float>::infinity());
    CHECK_NEAR(first.GetCameraDistance(), 200.f, 1e-6);
    CHECK_NEAR(second.GetCameraDistance(), 45.f, 1e-6);

    first.SetCameraDistance(10.f);
    CHECK_NEAR(first.GetCameraDistance(), 10.f, 1e-6);
    CHECK_THROWS(first.SetCameraDistance(0.f));
    CHECK_THROWS(first.SetCameraDistance(201.f));
    CHECK_THROWS(first.SetCameraDistance(std::numeric_limits<float>::quiet_NaN()));
    CHECK_NEAR(first.GetCameraDistance(), 10.f, 1e-6);

    // A minimized or transitional zero-sized window retains its last renderable extent.
    first.Resize(127, 65);
    CHECK(first.GetWidth() == 127 && first.GetHeight() == 65);
    first.Resize(0, 0);
    first.Resize(-1, 200);
    first.Resize(100, 0);
    CHECK(first.GetWidth() == 127 && first.GetHeight() == 65);
    CHECK(second.GetWidth() == 320 && second.GetHeight() == 240);
    first.Resize(1, 1);
    CHECK(first.GetWidth() == 1 && first.GetHeight() == 1);

    first.Resize(800, 600);
    const auto originalDirection = first.GetLightDirection();
    first.MovePointer(400, 300);
    CHECK(first.GetLightDirection() == originalDirection);
    first.SetDragging(true);
    first.MovePointer(400, 300);
    CHECK(first.GetLightDirection() == math::Vector(0.f, 0.f, 1.f));
    first.MovePointer(0, 0);
    const float component = 1.f / std::sqrt(3.f);
    CHECK_NEAR(first.GetLightDirection().X, -component, 1e-6);
    CHECK_NEAR(first.GetLightDirection().Y, component, 1e-6);
    CHECK_NEAR(first.GetLightDirection().Z, component, 1e-6);
    const auto draggedDirection = first.GetLightDirection();
    first.SetDragging(false);
    first.MovePointer(800, 600);
    CHECK(first.GetLightDirection() == draggedDirection);
    first.SetDragging(true);
    first.Resize(1600, 1200);
    first.MovePointer(800, 600);
    CHECK(first.GetLightDirection() == math::Vector(0.f, 0.f, 1.f));
    CHECK(second.GetLightDirection() == originalDirection);

    first.Apply(app::Action::Quit);
    CHECK(!first.IsRunning());
    CHECK(second.IsRunning());

    graphics::RenderSettings configured;
    configured.Shadows = false;
    configured.Primitive = graphics::PrimitiveType::Points;
    app::AppState custom(5, 7, configured);
    CHECK(!custom.GetRenderSettings().Shadows);
    CHECK(custom.GetRenderSettings().Primitive == graphics::PrimitiveType::Points);
    custom.Apply(app::Action::CyclePrimitive);
    CHECK(custom.GetRenderSettings().Primitive == graphics::PrimitiveType::Triangles);
}
