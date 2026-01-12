#pragma once

#include "Object.hpp"

class Wall : public Object {
public:
    inline void Create() {
        std::vector<shader::Vertex> vertices = {
            // Front face (+Z)
            {{-10.f, -10.f, -5.f, 1.f}, {0.f, 0.f, 1.f, 0.f}, {0.1f, 0.1f, 0.1f, 1.f}, {}, {}}, // 0
            {{10.f, -10.f, -5.f, 1.f}, {0.f, 0.f, 1.f, 0.f}, {0.1f, 0.1f, 0.1f, 1.f}, {}, {}},  // 1
            {{10.f, 10.f, -5.f, 1.f}, {0.f, 0.f, 1.f, 0.f}, {0.1f, 0.1f, 0.1f, 1.f}, {}, {}},   // 2
            {{-10.f, 10.f, -5.f, 1.f}, {0.f, 0.f, 1.f, 0.f}, {0.1f, 0.1f, 0.1f, 1.f}, {}, {}},  // 3
        };

        std::vector<std::uint32_t> indices = {
            0, 1, 2, 0, 2, 3, // Front
        };

        this->Meshes.push_back(graphics::Mesh{vertices, indices, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr});
    }
};