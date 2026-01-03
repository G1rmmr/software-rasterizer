#pragma once

#include "Graphics.h"
#include "Math.h"

#include <cstdint>

namespace world{
    constexpr inline math::Vector3 CAMERA_POS(0.f, 0.f, 1.f);
    constexpr inline math::Vector3 UP_AXIS(0.f, 1.f, 0.f);

    constexpr inline std::uint32_t WIDTH = 1600;
    constexpr inline std::uint32_t HEIGHT = 900;

    inline graphics::FrameBuffer Front(WIDTH, HEIGHT);
    inline graphics::FrameBuffer Back(WIDTH, HEIGHT);
}
