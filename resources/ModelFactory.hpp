#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>

#include "../scene/Model.hpp"

namespace resources {
    class ModelFactory {
    public:
        // root is the assets directory, not the executable or project directory.
        explicit ModelFactory(std::filesystem::path root);
        [[nodiscard]] std::shared_ptr<const scene::Model> LoadDiablo();
        [[nodiscard]] std::shared_ptr<const scene::Model> LoadAfrican();
        [[nodiscard]] std::shared_ptr<const scene::Model> CreatePlane(float halfExtent = 30.f) const;
        [[nodiscard]] std::shared_ptr<const scene::Model> CreateCube() const;
        [[nodiscard]] std::shared_ptr<const scene::Model> CreateSphere(float radius = 2.f,
                                                                       std::uint32_t subdivisions = 3) const;

    private:
        std::filesystem::path root;
        std::map<std::filesystem::path, std::weak_ptr<const graphics::Texture>> textures;
        std::shared_ptr<const graphics::Texture> LoadTexture(const std::filesystem::path& relativePath);
        graphics::Mesh LoadMesh(const std::filesystem::path& relativePath, graphics::Material material) const;
    };
}
