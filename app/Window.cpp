#include "Window.hpp"
#include <stdexcept>

namespace app {
    Window::Window(const char* title, std::uint32_t width, std::uint32_t height, bool resizable)
        : handle_(mfb_open_ex(title, width, height, resizable ? WF_RESIZABLE : 0)) {
        if(!handle_) throw std::runtime_error("MiniFB could not open a window");
    }
    Window::~Window() {
        Close();
    }
    bool Window::Present(const graphics::FrameBuffer& frame) {
        if(!handle_) return false;
        // MiniFB takes void* for a buffer it only reads. Keep this C API mismatch at the boundary.
        const auto status = mfb_update_ex(handle_, const_cast<std::uint32_t*>(frame.GetColors().data()),
                                          frame.GetWidth(), frame.GetHeight());
        if(status == STATE_EXIT) {
            handle_ = nullptr;
            return false;
        } // MiniFB already freed it.
        if(status != STATE_OK) {
            Close();
            throw std::runtime_error("MiniFB failed to present a frame");
        }
        return true;
    }
    bool Window::Wait() const noexcept {
        return handle_ && mfb_wait_sync(handle_);
    }
    void Window::Close() noexcept {
        if(!handle_) return;
        mfb_set_keyboard_callback(handle_, nullptr);
        mfb_set_resize_callback(handle_, nullptr);
        mfb_set_mouse_button_callback(handle_, nullptr);
        mfb_set_mouse_move_callback(handle_, nullptr);
        mfb_set_mouse_scroll_callback(handle_, nullptr);
        mfb_set_active_callback(handle_, nullptr);
        mfb_set_user_data(handle_, nullptr);
        // mfb_close marks the window; the next event update releases the allocation.
        mfb_close(handle_);
        mfb_update_events(handle_);
        handle_ = nullptr;
    }
}
