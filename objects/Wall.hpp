#pragma once

#include "Object.hpp"

class Wall : public Object {
public:
    inline void Create() {
        const float s = 30.f;

        std::vector<shader::Vertex> vertices = {
            // Front face (+Z)
            {{-s, -s, 0.f, 1.f}, {0.f, 0.f, 1.f, 0.f}, {0.6f, 0.2f, 0.f, 1.f}, {}, {}}, // LB
            {{s, -s, 0.f, 1.f}, {0.f, 0.f, 1.f, 0.f}, {0.6f, 0.2f, 0.f, 1.f}, {}, {}},  // RB
            {{s, s, 0.f, 1.f}, {0.f, 0.f, 1.f, 0.f}, {0.6f, 0.2f, 0.f, 1.f}, {}, {}},   // RT
            {{-s, s, 0.f, 1.f}, {0.f, 0.f, 1.f, 0.f}, {0.6f, 0.2f, 0.f, 1.f}, {}, {}},  // LT
        };

        std::vector<std::uint32_t> indices = {
            0, 1, 2, 0, 2, 3, // Front
        };

        this->Meshes.push_back(graphics::Mesh{vertices, indices, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr});
    }
};