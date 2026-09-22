#include "Mesh.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace graphics {
    Mesh::Mesh(std::vector<Vertex> vertices, std::vector<std::uint32_t> indices, Material material)
        : vertices(std::move(vertices)), indices(std::move(indices)), material(std::move(material)) {
        if(this->vertices.empty() || this->indices.empty() || this->indices.size() % 3 != 0)
            throw std::invalid_argument("A mesh requires vertices and complete indexed triangles");
        if(!std::isfinite(this->material.AlphaCutoff) || this->material.AlphaCutoff < 0.f ||
           this->material.AlphaCutoff > 1.f)
            throw std::invalid_argument("Material alpha cutoff must be in [0, 1]");
        switch(this->material.Alpha) {
        case AlphaMode::Opaque:
        case AlphaMode::Mask:
        case AlphaMode::Blend: break;
        default: throw std::invalid_argument("Unknown material alpha mode");
        }
        for(auto index : this->indices)
            if(index >= this->vertices.size()) throw std::invalid_argument("Mesh index is outside the vertex array");

        double low[3] = {INFINITY, INFINITY, INFINITY};
        double high[3] = {-INFINITY, -INFINITY, -INFINITY};
        for(const auto& vertex : this->vertices) {
            for(const auto* value : {&vertex.Pos, &vertex.Normal, &vertex.Color, &vertex.UV, &vertex.Tangent})
                if(!std::isfinite(value->X) || !std::isfinite(value->Y) || !std::isfinite(value->Z) ||
                   !std::isfinite(value->W))
                    throw std::invalid_argument("Mesh vertex attributes must be finite");
            if(vertex.Pos.W != 1.f) throw std::invalid_argument("Mesh positions must have homogeneous W = 1");
            const double xyz[] = {vertex.Pos.X, vertex.Pos.Y, vertex.Pos.Z};
            for(int axis = 0; axis < 3; ++axis) {
                low[axis] = std::min(low[axis], xyz[axis]);
                high[axis] = std::max(high[axis], xyz[axis]);
            }
        }
        bounds.Center = {static_cast<float>((low[0] + high[0]) * .5), static_cast<float>((low[1] + high[1]) * .5),
                         static_cast<float>((low[2] + high[2]) * .5), 1.f};
        double radius = 0;
        for(const auto& vertex : this->vertices) {
            const double x = static_cast<double>(vertex.Pos.X) - bounds.Center.X;
            const double y = static_cast<double>(vertex.Pos.Y) - bounds.Center.Y;
            const double z = static_cast<double>(vertex.Pos.Z) - bounds.Center.Z;
            radius = std::max(radius, std::sqrt(x * x + y * y + z * z));
        }
        bounds.Radius = std::nextafter(static_cast<float>(radius), std::numeric_limits<float>::infinity());
        if(!std::isfinite(bounds.Radius)) throw std::invalid_argument("Mesh bounds exceed the supported float range");
    }
}
