#pragma once

#include <memory>

#include "../math/Matrix.hpp"
#include "Model.hpp"

namespace scene {
    class SceneObject {
    public:
        explicit SceneObject(std::shared_ptr<const Model> model);
        [[nodiscard]] const Model& GetModel() const noexcept { return *model; }
        [[nodiscard]] const math::Matrix& GetTransform() const noexcept { return transform; }
        [[nodiscard]] const math::Matrix& GetNormalTransform() const noexcept { return normalTransform; }
        void SetTransform(math::Matrix value);
        [[nodiscard]] BoundingSphere GetWorldBounds() const noexcept { return worldBounds; }
        [[nodiscard]] bool IsVisible() const noexcept { return visible; }
        void SetVisible(bool value) noexcept { visible = value; }
        [[nodiscard]] bool CastsShadow() const noexcept { return castsShadow; }
        void SetCastsShadow(bool value) noexcept { castsShadow = value; }

    private:
        std::shared_ptr<const Model> model;
        math::Matrix transform;
        math::Matrix normalTransform;
        BoundingSphere worldBounds;
        bool visible = true;
        bool castsShadow = true;
    };
}
