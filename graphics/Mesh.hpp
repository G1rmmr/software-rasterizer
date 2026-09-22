#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "BoundingSphere.hpp"
#include "Material.hpp"
#include "Vertex.hpp"

namespace graphics {
    class Mesh {
    public:
        Mesh(std::vector<Vertex> vertices, std::vector<std::uint32_t> indices, Material material = {});
        [[nodiscard]] std::span<const Vertex> GetVertices() const noexcept { return vertices; }
        [[nodiscard]] std::span<const std::uint32_t> GetIndices() const noexcept { return indices; }
        [[nodiscard]] const Material& GetMaterial() const noexcept { return material; }
        [[nodiscard]] const BoundingSphere& GetBounds() const noexcept { return bounds; }

    private:
        std::vector<Vertex> vertices;
        std::vector<std::uint32_t> indices;
        Material material;
        BoundingSphere bounds;
    };
}
