#include "TestSupport.hpp"

#include <filesystem>

#include "../math/Math.hpp"
#include "../resources/ModelFactory.hpp"
#include "../scene/MirScene.hpp"

void RunMirSceneTests() {
    const auto assetRoot = std::filesystem::path(__FILE__).parent_path().parent_path() / "assets";
    resources::ModelFactory factory(assetRoot);
    scene::MirScene world;
    const auto cube = factory.CreateCube();
    const auto first = world.Add(cube);
    const auto second = world.Add(cube);

    const auto firstTransform = math::CreateTranslation({2.f, 3.f, -4.f}) * math::CreateScale({2.f, 1.f, .5f});
    const auto secondTransform = math::CreateTranslation({-5.f, 1.f, -2.f});
    CHECK(world.SetTransform(first, firstTransform));
    CHECK(world.SetTransform(second, secondTransform));
    CHECK(world.SetVisible(first, false));
    CHECK(world.SetCastsShadow(first, false));
    CHECK(world.SetCastsShadow(second, false));
    world.Commit();

    const auto objects = world.GetScene().GetObjects();
    CHECK(objects.size() == 2);
    CHECK(objects[0].GetTransform() == firstTransform);
    CHECK(objects[1].GetTransform() == secondTransform);
    CHECK(!objects[0].IsVisible());
    CHECK(!objects[0].CastsShadow());
    CHECK(objects[1].IsVisible());
    CHECK(!objects[1].CastsShadow());
    CHECK(&objects[0].GetModel() == &objects[1].GetModel());

    CHECK(world.Delete(first));
    world.Commit();
    CHECK(!world.GetScene().GetObjects()[0].IsVisible());
    CHECK(!world.SetTransform(first, math::Matrix{}));
}
