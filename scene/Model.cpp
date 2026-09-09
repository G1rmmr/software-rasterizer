#include "Model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace scene {
    Model::Model(std::vector<graphics::Mesh> meshes) : meshes(std::move(meshes)) {
        if(this->meshes.empty()) throw std::invalid_argument("A model requires at least one mesh");
        double low[3] = {INFINITY, INFINITY, INFINITY};
        double high[3] = {-INFINITY, -INFINITY, -INFINITY};
        for(const auto& mesh : this->meshes) {
            if(mesh.GetVertices().empty() || mesh.GetIndices().empty())
                throw std::invalid_argument("A model cannot contain an empty or moved-from mesh");
            for(const auto& vertex : mesh.GetVertices()) {
                const double xyz[] = {vertex.Pos.X, vertex.Pos.Y, vertex.Pos.Z};
                for(int axis = 0; axis < 3; ++axis) {
                    low[axis] = std::min(low[axis], xyz[axis]);
                    high[axis] = std::max(high[axis], xyz[axis]);
                }
            }
        }
        bounds.Center = {static_cast<float>((low[0] + high[0]) * .5), static_cast<float>((low[1] + high[1]) * .5),
                         static_cast<float>((low[2] + high[2]) * .5), 1.f};
        double radius = 0;
        for(const auto& mesh : this->meshes) {
            for(const auto& vertex : mesh.GetVertices()) {
                const double x = static_cast<double>(vertex.Pos.X) - bounds.Center.X;
                const double y = static_cast<double>(vertex.Pos.Y) - bounds.Center.Y;
                const double z = static_cast<double>(vertex.Pos.Z) - bounds.Center.Z;
                radius = std::max(radius, std::sqrt(x * x + y * y + z * z));
            }
        }
        bounds.Radius = std::nextafter(static_cast<float>(radius), std::numeric_limits<float>::infinity());
        if(!std::isfinite(bounds.Center.X) || !std::isfinite(bounds.Center.Y) || !std::isfinite(bounds.Center.Z) ||
           !std::isfinite(bounds.Radius))
            throw std::invalid_argument("Model bounds exceed the supported float range");
    }
}
