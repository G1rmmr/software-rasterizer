#pragma once

#include <cstdint>
#include <filesystem>
#include <istream>
#include <string>
#include <vector>

#include "../graphics/Vertex.hpp"

namespace resources {
    struct ObjData {
        std::vector<graphics::Vertex> Vertices;
        std::vector<std::uint32_t> Indices;
    };

    class ObjLoader {
    public:
        static ObjData Load(const std::filesystem::path& path);
        // Polygon faces are triangulated as a fan; convex OBJ faces are supported.
        // Missing normals follow OBJ smoothing groups (flat shading by default).
        static ObjData Load(std::istream& input, const std::string& sourceName = "<stream>");
    };
}
