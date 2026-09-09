#include "TestSupport.hpp"

#include <filesystem>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "../graphics/Texture.hpp"
#include "../math/Math.hpp"
#include "../resources/ModelFactory.hpp"
#include "../resources/ObjLoader.hpp"
#include "../scene/Scene.hpp"

namespace {
    resources::ObjData Parse(const std::string& source) {
        std::istringstream input(source);
        return resources::ObjLoader::Load(input, "fixture.obj");
    }

    void TextureTests() {
        graphics::Texture image(2, 2, {255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 128, 255, 255, 255, 255});
        CHECK_NEAR(image.Sample(.25f, .25f).X, 1.f, 1e-6);
        CHECK_NEAR(image.Sample(-.25f, 1.25f).Y, 1.f, 1e-6);
        const auto blue = image.Sample(1.25f, .75f);
        CHECK_NEAR(blue.Z, 1.f, 1e-6);
        CHECK_NEAR(blue.W, 128.f / 255.f, 1e-6);
        CHECK_THROWS(graphics::Texture(0, 1, {}));
        CHECK_THROWS(graphics::Texture(1, 1, {0, 0, 0}));
        CHECK_THROWS(image.Sample(std::numeric_limits<float>::infinity(), 0.f));
        CHECK_THROWS(graphics::Texture(std::filesystem::path(__FILE__).parent_path() / "missing-image.tga"));
        CHECK_THROWS(graphics::Texture(std::filesystem::path(__FILE__)));
    }

    void LoaderTests() {
        const auto quad = Parse("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
                                "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nf -4/-4 -3/-3 -2/-2 -1/-1 # quad\n");
        CHECK(quad.Vertices.size() == 4);
        CHECK(quad.Indices.size() == 6);
        for(const auto& vertex : quad.Vertices) {
            CHECK_NEAR(vertex.Normal.Z, 1.f, 1e-6);
            CHECK_NEAR(vertex.Normal.Dot(vertex.Tangent), 0.f, 1e-6);
            CHECK_NEAR(vertex.Tangent.Length(), 1.f, 1e-6);
            CHECK_NEAR(vertex.Tangent.W, 1.f, 1e-6);
        }
        const std::string positions = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n";
        const auto smooth = Parse(positions + "s 1\nf 1 2 3\nf 1 4 2\n");
        CHECK(smooth.Vertices.size() == 4);
        CHECK_NEAR(smooth.Vertices[0].Normal.Y, std::sqrt(.5f), 1e-6);
        CHECK_NEAR(smooth.Vertices[0].Normal.Z, std::sqrt(.5f), 1e-6);
        const auto flat = Parse(positions + "s off\nf 1 2 3\nf 1 4 2\n");
        CHECK(flat.Vertices.size() == 6);
        CHECK_NEAR(flat.Vertices[0].Normal.Y, 0.f, 1e-6);
        CHECK_NEAR(flat.Vertices[0].Normal.Z, 1.f, 1e-6);

        const auto degenerateUV = Parse(positions + "vt 0 0\nvn 0 0 2\nf 1/1/1 2/1/1 3/1/1\n");
        for(const auto& vertex : degenerateUV.Vertices) {
            CHECK_NEAR(vertex.Normal.Length(), 1.f, 1e-6);
            CHECK_NEAR(vertex.Tangent.Length(), 1.f, 1e-6);
            CHECK_NEAR(vertex.Normal.Dot(vertex.Tangent), 0.f, 1e-6);
            CHECK_NEAR(std::abs(vertex.Tangent.W), 1.f, 1e-6);
        }
        CHECK_THROWS(Parse(positions + "f 0 2 3\n"));
        CHECK_THROWS(Parse(positions + "f -5 -3 -2\n"));
        CHECK_THROWS(Parse(positions + "f 1/1 2/1 3/1\n"));
        CHECK_THROWS(Parse(positions + "f 1// 2 3\n"));
        CHECK_THROWS(Parse(positions + "f 1x 2 3\n"));
        CHECK_THROWS(Parse(positions + "f 1 2\n"));
        CHECK_THROWS(Parse(positions + "f 1 1 2\n"));
        CHECK_THROWS(Parse("v invalid 0 0\nf 1 2 3\n"));
        CHECK_THROWS(Parse("# empty file\n"));
        bool sourceIncluded = false;
        try {
            Parse(positions + "f 0 2 3\n");
        }
        catch(const std::runtime_error& error) {
            sourceIncluded = std::string(error.what()).find("fixture.obj:5:") != std::string::npos;
        }
        CHECK(sourceIncluded);
    }

    void ModelAndSceneTests(const std::filesystem::path& assetRoot) {
        resources::ModelFactory factory(assetRoot);
        auto plane = factory.CreatePlane(1.f);
        CHECK(plane->GetMeshes().size() == 1);
        CHECK_NEAR(plane->GetBounds().Radius, std::sqrt(2.f), 1e-5);
        const auto vertices = plane->GetMeshes()[0].GetVertices();
        const auto indices = plane->GetMeshes()[0].GetIndices();
        std::vector<graphics::Vertex> copiedVertices(vertices.begin(), vertices.end());
        std::vector<std::uint32_t> copiedIndices(indices.begin(), indices.end());
        CHECK_THROWS(graphics::Mesh(copiedVertices, {0, 1}));
        CHECK_THROWS(graphics::Mesh(copiedVertices, {0, 1, 99}));
        CHECK_THROWS(graphics::Mesh({}, {}));
        copiedVertices[0].Pos.X = std::numeric_limits<float>::quiet_NaN();
        CHECK_THROWS(graphics::Mesh(copiedVertices, copiedIndices));
        CHECK_THROWS(scene::Model({}));
        graphics::Mesh movable(std::vector<graphics::Vertex>(vertices.begin(), vertices.end()), copiedIndices);
        graphics::Mesh destination(std::move(movable));
        CHECK_THROWS(scene::Model(std::vector<graphics::Mesh>{movable}));
        CHECK_THROWS(scene::SceneObject(nullptr));

        scene::Scene world;
        const auto handle = world.Add(plane);
        const auto second = world.Add(plane);
        CHECK(&world.Get(handle).GetModel() == &world.Get(second).GetModel());
        plane.reset();
        CHECK(world.Get(handle).GetModel().GetMeshes().size() == 1);
        auto& object = world.Get(handle);
        math::Matrix shear;
        shear[1][0] = 1.f;
        shear[3][0] = 4.f;
        object.SetTransform(shear);
        const auto bounds = object.GetWorldBounds();
        for(const auto& mesh : object.GetModel().GetMeshes()) {
            for(const auto& vertex : mesh.GetVertices()) {
                const auto transformed = object.GetTransform() * vertex.Pos;
                CHECK((transformed - bounds.Center).Length() <= bounds.Radius + 1e-5f);
            }
        }
        CHECK_NEAR(world.Get(second).GetTransform()[3][0], 0.f, 1e-6);
        const auto committed = object.GetTransform();
        const auto committedNormal = object.GetNormalTransform();
        CHECK_THROWS(object.SetTransform(math::CreateScale({1.f, 0.f, 1.f})));
        CHECK(object.GetTransform() == committed);
        CHECK(object.GetNormalTransform() == committedNormal);
        CHECK_NEAR(object.GetWorldBounds().Radius, bounds.Radius, 1e-6);
        math::Matrix projective;
        projective[0][3] = .1f;
        CHECK_THROWS(object.SetTransform(projective));
        math::Matrix nonfinite;
        nonfinite[0][0] = std::numeric_limits<float>::infinity();
        CHECK_THROWS(object.SetTransform(nonfinite));

        object.SetTransform(math::CreateScale({2.f, 1.f, 1.f}));
        const auto normal = object.GetNormalTransform() * math::Vector(1.f, 1.f, 0.f, 0.f);
        const auto tangent = object.GetTransform() * math::Vector(1.f, -1.f, 0.f, 0.f);
        CHECK_NEAR(normal.Dot(tangent), 0.f, 1e-6);
        object.SetTransform(math::CreateScale({.0001f, .0001f, .0001f}));
        CHECK_NEAR(object.GetNormalTransform()[0][0], 10000.f, .01);
        object.SetVisible(false);
        object.SetCastsShadow(false);
        CHECK(!object.IsVisible());
        CHECK(!object.CastsShadow());
        CHECK(world.Get(second).IsVisible());
        CHECK_THROWS(world.Get(20));

        const auto cube = factory.CreateCube();
        CHECK(cube->GetMeshes()[0].GetVertices().size() == 24);
        CHECK(cube->GetMeshes()[0].GetIndices().size() == 36);
        const auto sphere = factory.CreateSphere(3.f, 2);
        CHECK(sphere->GetMeshes()[0].GetIndices().size() == 20 * 16 * 3);
        for(const auto& vertex : sphere->GetMeshes()[0].GetVertices()) {
            CHECK_NEAR(vertex.Pos.Length(), 3.f, 1e-5);
            CHECK_NEAR(vertex.Normal.Length(), 1.f, 1e-5);
            CHECK_NEAR(vertex.Normal.Dot(vertex.Tangent), 0.f, 1e-5);
        }
        CHECK_THROWS(factory.CreatePlane(-1.f));
        CHECK_THROWS(factory.CreateSphere(0.f));
        CHECK_THROWS(factory.CreateSphere(1.f, 7));
    }

    void AssetTests(const std::filesystem::path& assetRoot) {
        resources::ModelFactory factory(assetRoot);
        const auto diablo = factory.LoadDiablo();
        const auto anotherDiablo = factory.LoadDiablo();
        CHECK(!diablo->GetMeshes()[0].GetIndices().empty());
        CHECK(diablo->GetMeshes()[0].GetMaterial().DiffuseMap ==
              anotherDiablo->GetMeshes()[0].GetMaterial().DiffuseMap);
        const auto african = factory.LoadAfrican();
        CHECK(african->GetMeshes().size() == 3);
        CHECK(african->GetMeshes()[2].GetMaterial().Alpha == graphics::AlphaMode::Blend);
        CHECK(african->GetMeshes()[0].GetMaterial().SSSMap != nullptr);
        CHECK(african->GetMeshes()[2].GetMaterial().GlossMap != nullptr);
        for(const auto* model : {diablo.get(), african.get()}) {
            for(const auto& mesh : model->GetMeshes()) {
                for(const auto& vertex : mesh.GetVertices()) {
                    CHECK(std::isfinite(vertex.Tangent.X));
                    CHECK_NEAR(vertex.Tangent.Length(), 1.f, 1e-4);
                    CHECK_NEAR(vertex.Normal.Dot(vertex.Tangent), 0.f, 1e-4);
                }
            }
        }
    }
}

void RunSceneTests() {
    const auto assetRoot = std::filesystem::path(__FILE__).parent_path().parent_path() / "assets";
    TextureTests();
    LoaderTests();
    ModelAndSceneTests(assetRoot);
    AssetTests(assetRoot);
}
