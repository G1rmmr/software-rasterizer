#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "SceneObject.hpp"

namespace scene {
    class Scene {
    public:
        using Handle = std::size_t;
        void Reserve(std::size_t count) { objects.reserve(count); }
        Handle Add(std::shared_ptr<const Model> model) {
            objects.emplace_back(std::move(model));
            return objects.size() - 1;
        }
        [[nodiscard]] SceneObject& Get(Handle handle) { return objects.at(handle); }
        [[nodiscard]] const SceneObject& Get(Handle handle) const { return objects.at(handle); }
        [[nodiscard]] std::span<const SceneObject> GetObjects() const noexcept { return objects; }

    private:
        std::vector<SceneObject> objects;
    };
}
