#pragma once
#include "../graphics/FrameBuffer.hpp"
#include "../libs/MiniFB.h"
#include <cstdint>

namespace app {
    class Window final {
    public:
        Window(const char* title, std::uint32_t width, std::uint32_t height, bool resizable = false);
        ~Window();
        Window(const Window&) = delete;
        Window& operator=(const Window&) = delete;
        [[nodiscard]] mfb_window* NativeHandle() const noexcept { return handle_; }
        [[nodiscard]] bool IsOpen() const noexcept { return handle_ != nullptr; }
        bool Present(const graphics::FrameBuffer& frame);
        bool Wait() const noexcept;
        void Close() noexcept;

    private:
        mfb_window* handle_ = nullptr;
    };
}
