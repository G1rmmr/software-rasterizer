#pragma once
#include <cstdint>

namespace config {
    inline constexpr const char* WindowTitle = "software-rasterizer";
    inline constexpr std::uint32_t Width = 1280;
    inline constexpr std::uint32_t Height = 720;
    inline constexpr std::uint32_t ClearColor = 0xff000000u;
    inline constexpr float FieldOfViewDegrees = 45.f;
    inline constexpr float NearPlane = .1f;
    inline constexpr float FarPlane = 300.f; // Covers the full 1..200 camera-distance range and the demo backdrop.
}
