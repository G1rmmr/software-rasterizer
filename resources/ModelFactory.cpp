#include "ModelFactory.hpp"

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>

#include "ObjLoader.hpp"

namespace {
    std::shared_ptr<const scene::Model> SingleMesh(std::vector<graphics::Vertex> vertices,
                                                   std::vector<std::uint32_t> indices) {
        std::vector<graphics::Mesh> meshes;
        meshes.emplace_back(std::move(vertices), std::move(indices));
        return std::make_shared<const scene::Model>(std::move(meshes));
    }

    graphics::Vertex SphereVertex(math::Vector normal, float radius) {
        normal = normal.Norm();
        normal.W = 0.f;
        auto position = normal * radius;
        position.W = 1.f;
        math::Vector tangent(-normal.Z, 0.f, normal.X, 0.f);
        if(tangent.Length() < 1e-6f) tangent = {1.f, 0.f, 0.f, 0.f};
        tangent = tangent.Norm();
        tangent.W = -1.f;
        return {position,
                normal,
                {normal.X * .5f + .5f, normal.Y * .5f + .5f, normal.Z * .5f + .5f, 1.f},
                {std::atan2(normal.Z, normal.X) / (2.f * std::numbers::pi_v<float>)+.5f,
                 std::asin(std::clamp(normal.Y, -1.f, 1.f)) / std::numbers::pi_v<float> + .5f, 0.f, 0.f},
                tangent};
    }
}

namespace resources {
    ModelFactory::ModelFactory(std::filesystem::path root)
        : root(std::filesystem::absolute(std::move(root)).lexically_normal()) {}

    std::shared_ptr<const graphics::Texture> ModelFactory::LoadTexture(const std::filesystem::path& relativePath) {
        const auto path = (root / relativePath).lexically_normal();
        if(auto found = textures.find(path); found != textures.end())
            if(auto texture = found->second.lock()) return texture;
        auto texture = std::make_shared<const graphics::Texture>(path);
        textures[path] = texture;
        return texture;
    }

    graphics::Mesh ModelFactory::LoadMesh(const std::filesystem::path& relativePath,
                                          graphics::Material material) const {
        auto data = ObjLoader::Load(root / relativePath);
        return graphics::Mesh(std::move(data.Vertices), std::move(data.Indices), std::move(material));
    }

    std::shared_ptr<const scene::Model> ModelFactory::LoadDiablo() {
        graphics::Material material;
        material.DiffuseMap = LoadTexture("diablo/diablo3_pose_diffuse.tga");
        material.NormalMap = LoadTexture("diablo/diablo3_pose_nm_tangent.tga");
        material.SpecularMap = LoadTexture("diablo/diablo3_pose_spec.tga");
        material.GlowMap = LoadTexture("diablo/diablo3_pose_glow.tga");
        std::vector<graphics::Mesh> meshes;
        meshes.push_back(LoadMesh("diablo/diablo3_pose.obj", std::move(material)));
        return std::make_shared<const scene::Model>(std::move(meshes));
    }

    std::shared_ptr<const scene::Model> ModelFactory::LoadAfrican() {
        std::vector<graphics::Mesh> meshes;
        graphics::Material head;
        head.DiffuseMap = LoadTexture("african/african_head_diffuse.tga");
        head.NormalMap = LoadTexture("african/african_head_nm_tangent.tga");
        head.SpecularMap = LoadTexture("african/african_head_spec.tga");
        head.SSSMap = LoadTexture("african/african_head_SSS.jpg");
        meshes.push_back(LoadMesh("african/african_head.obj", std::move(head)));

        graphics::Material inner;
        inner.DiffuseMap = LoadTexture("african/african_head_eye_inner_diffuse.tga");
        inner.NormalMap = LoadTexture("african/african_head_eye_inner_nm_tangent.tga");
        inner.SpecularMap = LoadTexture("african/african_head_eye_inner_spec.tga");
        meshes.push_back(LoadMesh("african/african_head_eye_inner.obj", std::move(inner)));

        graphics::Material outer;
        outer.DiffuseMap = LoadTexture("african/african_head_eye_outer_diffuse.tga");
        outer.NormalMap = LoadTexture("african/african_head_eye_outer_nm_tangent.tga");
        outer.SpecularMap = LoadTexture("african/african_head_eye_outer_spec.tga");
        outer.GlossMap = LoadTexture("african/african_head_eye_outer_gloss.tga");
        outer.Alpha = graphics::AlphaMode::Blend;
        meshes.push_back(LoadMesh("african/african_head_eye_outer.obj", std::move(outer)));
        return std::make_shared<const scene::Model>(std::move(meshes));
    }

    std::shared_ptr<const scene::Model> ModelFactory::CreatePlane(float halfExtent) const {
        if(!std::isfinite(halfExtent) || halfExtent <= 0.f)
            throw std::invalid_argument("Plane half extent must be finite and positive");
        const math::Vector normal{0.f, 0.f, 1.f, 0.f};
        const math::Vector color{.7f, .7f, .7f, 1.f};
        const math::Vector tangent{1.f, 0.f, 0.f, 1.f};
        std::vector<graphics::Vertex> vertices{
            {{-halfExtent, -halfExtent, 0.f, 1.f}, normal, color, {0.f, 0.f, 0.f, 0.f}, tangent},
            {{halfExtent, -halfExtent, 0.f, 1.f}, normal, color, {1.f, 0.f, 0.f, 0.f}, tangent},
            {{halfExtent, halfExtent, 0.f, 1.f}, normal, color, {1.f, 1.f, 0.f, 0.f}, tangent},
            {{-halfExtent, halfExtent, 0.f, 1.f}, normal, color, {0.f, 1.f, 0.f, 0.f}, tangent}};
        return SingleMesh(std::move(vertices), {0, 1, 2, 0, 2, 3});
    }

    std::shared_ptr<const scene::Model> ModelFactory::CreateCube() const {
        const std::array<math::Vector, 6> normals{
            {{0, 0, 1}, {0, 0, -1}, {0, 1, 0}, {0, -1, 0}, {1, 0, 0}, {-1, 0, 0}}};
        const std::array<math::Vector, 6> tangents{
            {{1, 0, 0}, {-1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {0, 0, -1}, {0, 0, 1}}};
        const std::array<math::Vector, 4> colors{{{1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1}, {1, 1, 1, 1}}};
        std::vector<graphics::Vertex> vertices;
        std::vector<std::uint32_t> indices;
        for(std::size_t face = 0; face < normals.size(); ++face) {
            const auto normal = normals[face], u = tangents[face], v = normal.Cross(u);
            const auto base = static_cast<std::uint32_t>(vertices.size());
            for(int corner = 0; corner < 4; ++corner) {
                const float x = corner == 1 || corner == 2 ? 1.f : -1.f;
                const float y = corner >= 2 ? 1.f : -1.f;
                auto position = normal + u * x + v * y;
                position.W = 1.f;
                vertices.push_back({position,
                                    normal,
                                    colors[corner],
                                    {(x + 1.f) * .5f, (y + 1.f) * .5f, 0.f, 0.f},
                                    {u.X, u.Y, u.Z, 1.f}});
            }
            indices.insert(indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
        }
        return SingleMesh(std::move(vertices), std::move(indices));
    }

    std::shared_ptr<const scene::Model> ModelFactory::CreateSphere(float radius, std::uint32_t subdivisions) const {
        if(!std::isfinite(radius) || radius <= 0.f)
            throw std::invalid_argument("Sphere radius must be finite and positive");
        if(subdivisions > 6) throw std::invalid_argument("Sphere subdivisions must be in [0, 6]");
        const float t = (1.f + std::sqrt(5.f)) * .5f;
        const std::array<math::Vector, 12> positions{{{-1, t, 0},
                                                      {1, t, 0},
                                                      {-1, -t, 0},
                                                      {1, -t, 0},
                                                      {0, -1, t},
                                                      {0, 1, t},
                                                      {0, -1, -t},
                                                      {0, 1, -t},
                                                      {t, 0, -1},
                                                      {t, 0, 1},
                                                      {-t, 0, -1},
                                                      {-t, 0, 1}}};
        std::vector<graphics::Vertex> vertices;
        for(const auto& position : positions) vertices.push_back(SphereVertex(position, radius));
        std::vector<std::uint32_t> indices{0, 11, 5,  0, 5,  1, 0, 1, 7, 0, 7,  10, 0, 10, 11, 1, 5, 9, 5, 11,
                                           4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3,  9,  4, 3,  4,  2, 3, 2, 6, 3,
                                           6, 8,  3,  8, 9,  4, 9, 5, 2, 4, 11, 6,  2, 10, 8,  6, 7, 9, 8, 1};
        for(std::uint32_t level = 0; level < subdivisions; ++level) {
            std::map<std::uint64_t, std::uint32_t> midpoints;
            auto midpoint = [&](std::uint32_t a, std::uint32_t b) {
                const std::uint64_t key = (static_cast<std::uint64_t>(std::min(a, b)) << 32) | std::max(a, b);
                if(auto found = midpoints.find(key); found != midpoints.end()) return found->second;
                const auto index = static_cast<std::uint32_t>(vertices.size());
                const auto normal = (vertices[a].Normal + vertices[b].Normal).Norm();
                vertices.push_back(SphereVertex(normal, radius));
                midpoints.emplace(key, index);
                return index;
            };
            std::vector<std::uint32_t> next;
            next.reserve(indices.size() * 4);
            for(std::size_t i = 0; i < indices.size(); i += 3) {
                const auto v0 = indices[i], v1 = indices[i + 1], v2 = indices[i + 2];
                const auto a = midpoint(v0, v1), b = midpoint(v1, v2), c = midpoint(v2, v0);
                next.insert(next.end(), {v0, a, c, v1, b, a, v2, c, b, a, b, c});
            }
            indices = std::move(next);
        }
        return SingleMesh(std::move(vertices), std::move(indices));
    }
}
