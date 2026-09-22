#include "MirScene.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <type_traits>

namespace scene::detail {
    // CommandBuffer accepts only trivially copyable payloads.  math::Matrix
    // deliberately is not one, so cross the ECS boundary through raw values.
    struct MatrixPayload {
        std::array<float, 16> Elements{};
    };
    static_assert(std::is_trivially_copyable_v<MatrixPayload>);

    struct RenderState {
        bool Visible = true;
        bool CastsShadow = true;
    };
    static_assert(std::is_trivially_copyable_v<RenderState>);

    struct TransformComponent : mir::Component<TransformComponent, MatrixPayload> {};
    struct RenderStateComponent : mir::Component<RenderStateComponent, RenderState> {};

    [[nodiscard]] MatrixPayload ToPayload(const math::Matrix& matrix) noexcept {
        MatrixPayload payload;
        for(std::size_t index = 0; index < payload.Elements.size(); ++index) payload.Elements[index] = matrix.Elements[index];
        return payload;
    }

    [[nodiscard]] math::Matrix FromPayload(const MatrixPayload& payload) noexcept {
        math::Matrix matrix(0.f);
        for(std::size_t index = 0; index < payload.Elements.size(); ++index) matrix.Elements[index] = payload.Elements[index];
        return matrix;
    }
}

namespace scene {
    MirScene::MirScene() {
        static_assert(mir::config::MAX_ENTITY >= MaxEntities);
        snapshot_.Reserve(MaxEntities);
        bindings_.reserve(MaxEntities);
    }

    MirScene::~MirScene() {
        auto& manager = mir::core::Manager::Instance();
        for(const Binding& binding : bindings_) (void)manager.DeleteEntity(binding.Entity);
        manager.UpdateSystem(0.f);
    }

    MirScene::Handle MirScene::Add(std::shared_ptr<const Model> model) {
        if(!model) throw std::invalid_argument("A MIR scene object requires a model");
        if(bindings_.size() == MaxEntities) throw std::length_error("MIR scene entity capacity is exhausted");

        auto& manager = mir::core::Manager::Instance();
        const Handle entity = manager.AddEntity();
        if(!manager.IsValidEntity(entity)) throw std::length_error("MIR entity capacity is exhausted");

        const Scene::Handle object = snapshot_.Add(std::move(model));
        bindings_.push_back({entity, object, true, true});
        if(!detail::TransformComponent::Set(entity, detail::ToPayload(math::Matrix{})) ||
           !detail::RenderStateComponent::Set(entity, {}))
            throw std::runtime_error("MIR command buffer could not create a render instance");
        return entity;
    }

    bool MirScene::SetTransform(const Handle handle, const math::Matrix& transform) noexcept {
        return detail::TransformComponent::Set(handle, detail::ToPayload(transform));
    }

    bool MirScene::SetVisible(const Handle handle, const bool visible) noexcept {
        const auto binding = std::find_if(bindings_.begin(), bindings_.end(),
                                          [handle](const Binding& candidate) { return candidate.Entity == handle; });
        if(binding == bindings_.end()) return false;
        if(!detail::RenderStateComponent::Set(handle, {.Visible = visible, .CastsShadow = binding->CastsShadow})) return false;
        binding->Visible = visible;
        return true;
    }

    bool MirScene::SetCastsShadow(const Handle handle, const bool castsShadow) noexcept {
        const auto binding = std::find_if(bindings_.begin(), bindings_.end(),
                                          [handle](const Binding& candidate) { return candidate.Entity == handle; });
        if(binding == bindings_.end()) return false;
        if(!detail::RenderStateComponent::Set(handle, {.Visible = binding->Visible, .CastsShadow = castsShadow})) return false;
        binding->CastsShadow = castsShadow;
        return true;
    }

    bool MirScene::Delete(const Handle handle) noexcept { return mir::core::Manager::Instance().DeleteEntity(handle); }

    void MirScene::Commit() {
        auto& manager = mir::core::Manager::Instance();
        manager.UpdateSystem(0.f);
        for(const Binding& binding : bindings_) {
            SceneObject& object = snapshot_.Get(binding.Snapshot);
            const detail::MatrixPayload* transform = detail::TransformComponent::TryGet(binding.Entity);
            const detail::RenderState* state = detail::RenderStateComponent::TryGet(binding.Entity);
            if(!manager.IsValidEntity(binding.Entity) || !transform || !state) {
                object.SetVisible(false);
                object.SetCastsShadow(false);
                continue;
            }
            object.SetTransform(detail::FromPayload(*transform));
            object.SetVisible(state->Visible);
            object.SetCastsShadow(state->CastsShadow);
        }
    }
}
