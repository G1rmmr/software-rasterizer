#pragma once
#include "../graphics/FrameBuffer.hpp"
#include <filesystem>

namespace app {
    void WriteBmp(const std::filesystem::path& path, const graphics::FrameBuffer& frame);
}
