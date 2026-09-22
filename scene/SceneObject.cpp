#include "SceneObject.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace scene {
    SceneObject::SceneObject(std::shared_ptr<const Model> model) : model(std::move(model)) {
        if(!this->model) throw std::invalid_argument("A scene object requires a model");
        SetTransform(math::Matrix{});
    }

    void SceneObject::SetTransform(math::Matrix value) {
        for(int column = 0; column < 4; ++column)
            for(int row = 0; row < 4; ++row)
                if(!std::isfinite(value[column][row])) throw std::invalid_argument("Object transform must be finite");
        if(value[0][3] != 0.f || value[1][3] != 0.f || value[2][3] != 0.f || value[3][3] != 1.f)
            throw std::invalid_argument("Object transform must be affine");

        double a[3][3];
        for(int row = 0; row < 3; ++row)
            for(int col = 0; col < 3; ++col) a[row][col] = value[col][row];
        double cofactor[3][3];
        for(int row = 0; row < 3; ++row) {
            for(int col = 0; col < 3; ++col) {
                const int r0 = (row + 1) % 3, r1 = (row + 2) % 3;
                const int c0 = (col + 1) % 3, c1 = (col + 2) % 3;
                cofactor[row][col] = a[r0][c0] * a[r1][c1] - a[r0][c1] * a[r1][c0];
            }
        }
        const double determinant = a[0][0] * cofactor[0][0] + a[0][1] * cofactor[0][1] + a[0][2] * cofactor[0][2];
        if(determinant == 0.0 || !std::isfinite(determinant))
            throw std::invalid_argument("Object transform must have an invertible linear part");
        math::Matrix nextNormal;
        for(int row = 0; row < 3; ++row) {
            for(int col = 0; col < 3; ++col) {
                const float component = static_cast<float>(cofactor[row][col] / determinant);
                if(!std::isfinite(component)) throw std::invalid_argument("Normal transform exceeds the float range");
                nextNormal[col][row] = component;
            }
        }

        // ||A||_2^2 <= ||A^T A||_infinity. Unlike longest-column scaling,
        // this remains a conservative sphere bound for shear and reflection.
        double largestRowSum = 0;
        for(int row = 0; row < 3; ++row) {
            double rowSum = 0;
            for(int col = 0; col < 3; ++col) {
                double gram = 0;
                for(int k = 0; k < 3; ++k) gram += a[k][row] * a[k][col];
                rowSum += std::abs(gram);
            }
            largestRowSum = std::max(largestRowSum, rowSum);
        }
        const auto& local = model->GetBounds();
        const double xyz[] = {local.Center.X, local.Center.Y, local.Center.Z};
        double center[3];
        for(int row = 0; row < 3; ++row) {
            center[row] = value[3][row];
            for(int col = 0; col < 3; ++col) center[row] += a[row][col] * xyz[col];
        }
        BoundingSphere nextBounds;
        nextBounds.Center = {static_cast<float>(center[0]), static_cast<float>(center[1]),
                             static_cast<float>(center[2]), 1.f};
        const double dx = center[0] - nextBounds.Center.X;
        const double dy = center[1] - nextBounds.Center.Y;
        const double dz = center[2] - nextBounds.Center.Z;
        const double radius = local.Radius * std::sqrt(largestRowSum) + std::sqrt(dx * dx + dy * dy + dz * dz);
        nextBounds.Radius = std::nextafter(static_cast<float>(radius), std::numeric_limits<float>::infinity());
        if(!std::isfinite(nextBounds.Center.X) || !std::isfinite(nextBounds.Center.Y) ||
           !std::isfinite(nextBounds.Center.Z) || !std::isfinite(nextBounds.Radius))
            throw std::invalid_argument("World bounds exceed the supported float range");

        // Publish together only after all validation and derived calculations succeed.
        transform = value;
        normalTransform = nextNormal;
        worldBounds = nextBounds;
    }
}
