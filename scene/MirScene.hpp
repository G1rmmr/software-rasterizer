#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <Config.hpp>
#include <core/Component.hpp>
#include <core/Entity.hpp>
#include <core/Manager.hpp>

#include "../math/Matrix.hpp"
#include "Scene.hpp"

namespace scene {
    // A MIR-backed source of render instances.  MIR owns generation-safe IDs
    // and deferred component writes; the snapshot keeps the renderer's proven
    // SceneObject transform, normal-matrix, and world-bounds contract intact.
    class MirScene final {
    public:
        using Handle = mir::Id;

        static constexpr std::size_t MaxEntities = 4098;

        MirScene();
        ~MirScene();
        MirScene(const MirScene&) = delete;
        MirScene& operator=(const MirScene&) = delete;

        [[nodiscard]] Handle Add(std::shared_ptr<const Model> model);
        [[nodiscard]] bool SetTransform(Handle handle, const math::Matrix& transform) noexcept;
        [[nodiscard]] bool SetVisible(Handle handle, bool visible) noexcept;
        [[nodiscard]] bool SetCastsShadow(Handle handle, bool castsShadow) noexcept;
        [[nodiscard]] bool Delete(Handle handle) noexcept;

        // Apply all deferred MIR mutations, then refresh the immutable-for-a-
        // frame renderer snapshot.  Call this before Renderer::Render().
        void Commit();

        [[nodiscard]] const Scene& GetScene() const noexcept { return snapshot_; }
        [[nodiscard]] std::size_t Size() const noexcept { return bindings_.size(); }

    private:
        struct Binding {
            Handle Entity;
            Scene::Handle Snapshot;
            // Mirror state that has been queued but not yet committed.  MIR
            // components expose committed values only, so this prevents two
            // setters in one frame from overwriting one another.
            bool Visible = true;
            bool CastsShadow = true;
        };

        Scene snapshot_;
        std::vector<Binding> bindings_;
    };
}
