#pragma once

#include <span>
#include <vector>

#include "../graphics/Mesh.hpp"
#include "BoundingSphere.hpp"

namespace scene {
    // Geometry and material bindings have one lifetime, independent of scene instances.
    class Model {
    public:
        explicit Model(std::vector<graphics::Mesh> meshes);
        Model(const Model&) = default;
        Model& operator=(const Model&) = delete;
        [[nodiscard]] std::span<const graphics::Mesh> GetMeshes() const noexcept { return meshes; }
        [[nodiscard]] const BoundingSphere& GetBounds() const noexcept { return bounds; }

    private:
        std::vector<graphics::Mesh> meshes;
        BoundingSphere bounds;
    };
}
