#pragma once

namespace graphics {
    enum class PrimitiveType { Points, Lines, Triangles };
    enum class CullMode { None, Back };

    struct RasterizerOptions {
        PrimitiveType Primitive = PrimitiveType::Triangles;
        CullMode Cull = CullMode::Back;
        bool FlipWinding = false; // Reflected model transforms reverse the front face.
        bool DepthWrite = true;
        bool Blend = false;
    };
}
