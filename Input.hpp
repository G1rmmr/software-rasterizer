#pragma once
#include "AppState.hpp"
#include "libs/MiniFB.h"
#include <array>

namespace app {
    // Adapter from the C window API to instance-specific application actions.
    // The owning Window is destroyed before this controller and its AppState.
    class InputController final {
    public:
        explicit InputController(AppState& state) noexcept : state_(state) {}
        void Attach(mfb_window* window) noexcept {
            mfb_set_user_data(window, this);
            mfb_set_keyboard_callback(window, Keyboard);
            mfb_set_resize_callback(window, Resize);
            mfb_set_mouse_button_callback(window, MouseButton);
            mfb_set_mouse_move_callback(window, MouseMove);
            mfb_set_mouse_scroll_callback(window, MouseScroll);
            mfb_set_active_callback(window, Active);
        }

    private:
        static InputController& Self(mfb_window* window) noexcept {
            return *static_cast<InputController*>(mfb_get_user_data(window));
        }
        static void Keyboard(mfb_window* window, mfb_key key, mfb_key_mod, bool pressed) noexcept {
            auto& self = Self(window);
            const auto index = static_cast<int>(key);
            if(index < 0 || index >= static_cast<int>(self.keys_.size())) return;
            const bool repeated = self.keys_[index];
            self.keys_[index] = pressed;
            if(!pressed || repeated) return;
            switch(key) {
            case KB_KEY_SPACE: self.state_.Apply(Action::CyclePrimitive); break;
            case KB_KEY_Q: self.state_.Apply(Action::ToggleShadows); break;
            case KB_KEY_W: self.state_.Apply(Action::ToggleSsao); break;
            case KB_KEY_E: self.state_.Apply(Action::ToggleAA); break;
            case KB_KEY_R: self.state_.Apply(Action::ToggleToon); break;
            case KB_KEY_ESCAPE: self.state_.Apply(Action::Quit); break;
            default: break;
            }
        }
        static void Resize(mfb_window* window, int width, int height) noexcept {
            Self(window).state_.Resize(width, height);
        }
        static void MouseButton(mfb_window* window, mfb_mouse_button button, mfb_key_mod, bool pressed) noexcept {
            if(button == MOUSE_BTN_1) Self(window).state_.SetDragging(pressed);
        }
        static void MouseMove(mfb_window* window, int x, int y) noexcept { Self(window).state_.MovePointer(x, y); }
        static void MouseScroll(mfb_window* window, mfb_key_mod, float, float y) noexcept {
            Self(window).state_.Scroll(y);
        }
        static void Active(mfb_window* window, bool active) noexcept {
            if(!active) {
                Self(window).keys_.fill(false);
                Self(window).state_.SetDragging(false);
            }
        }
        AppState& state_;
        std::array<bool, KB_KEY_LAST + 1> keys_{};
    };
}
