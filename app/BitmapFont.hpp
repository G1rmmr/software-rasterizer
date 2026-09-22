#pragma once

#include "../graphics/FrameBuffer.hpp"
#include <array>
#include <cstdint>
#include <limits>
#include <string_view>

namespace app {
    // A small fixed 5x7 ASCII font. Scaling uses whole pixels; no font files or allocations.
    class BitmapFont final {
    public:
        using Glyph = std::array<std::uint8_t, 7>;
        static constexpr int GlyphWidth = 5;
        static constexpr int GlyphHeight = 7;
        static constexpr int Advance = 6;

        [[nodiscard]] static constexpr Glyph GetGlyph(char character) noexcept {
            if(character >= 'a' && character <= 'z') character = static_cast<char>(character - 'a' + 'A');
            switch(character) {
            case 'A': return {14, 17, 17, 31, 17, 17, 17};
            case 'B': return {30, 17, 17, 30, 17, 17, 30};
            case 'C': return {14, 17, 16, 16, 16, 17, 14};
            case 'D': return {30, 17, 17, 17, 17, 17, 30};
            case 'E': return {31, 16, 16, 30, 16, 16, 31};
            case 'F': return {31, 16, 16, 30, 16, 16, 16};
            case 'G': return {14, 17, 16, 23, 17, 17, 15};
            case 'H': return {17, 17, 17, 31, 17, 17, 17};
            case 'I': return {14, 4, 4, 4, 4, 4, 14};
            case 'J': return {7, 2, 2, 2, 2, 18, 12};
            case 'K': return {17, 18, 20, 24, 20, 18, 17};
            case 'L': return {16, 16, 16, 16, 16, 16, 31};
            case 'M': return {17, 27, 21, 21, 17, 17, 17};
            case 'N': return {17, 25, 21, 19, 17, 17, 17};
            case 'O': return {14, 17, 17, 17, 17, 17, 14};
            case 'P': return {30, 17, 17, 30, 16, 16, 16};
            case 'Q': return {14, 17, 17, 17, 21, 18, 13};
            case 'R': return {30, 17, 17, 30, 20, 18, 17};
            case 'S': return {15, 16, 16, 14, 1, 1, 30};
            case 'T': return {31, 4, 4, 4, 4, 4, 4};
            case 'U': return {17, 17, 17, 17, 17, 17, 14};
            case 'V': return {17, 17, 17, 17, 17, 10, 4};
            case 'W': return {17, 17, 17, 21, 21, 21, 10};
            case 'X': return {17, 17, 10, 4, 10, 17, 17};
            case 'Y': return {17, 17, 10, 4, 4, 4, 4};
            case 'Z': return {31, 1, 2, 4, 8, 16, 31};
            case '0': return {14, 17, 19, 21, 25, 17, 14};
            case '1': return {4, 12, 4, 4, 4, 4, 14};
            case '2': return {14, 17, 1, 2, 4, 8, 31};
            case '3': return {30, 1, 1, 14, 1, 1, 30};
            case '4': return {2, 6, 10, 18, 31, 2, 2};
            case '5': return {31, 16, 16, 30, 1, 1, 30};
            case '6': return {14, 16, 16, 30, 17, 17, 14};
            case '7': return {31, 1, 2, 4, 8, 8, 8};
            case '8': return {14, 17, 17, 14, 17, 17, 14};
            case '9': return {14, 17, 17, 15, 1, 1, 14};
            case '.': return {0, 0, 0, 0, 0, 6, 6};
            case ':': return {0, 6, 6, 0, 6, 6, 0};
            case '-': return {0, 0, 0, 31, 0, 0, 0};
            case '/': return {1, 2, 2, 4, 8, 8, 16};
            case '+': return {0, 4, 4, 31, 4, 4, 0};
            case '=': return {0, 0, 31, 0, 31, 0, 0};
            case '(': return {2, 4, 8, 8, 8, 4, 2};
            case ')': return {8, 4, 2, 2, 2, 4, 8};
            case '%': return {17, 2, 4, 4, 8, 16, 17};
            case ' ': return {};
            default: return {14, 17, 1, 2, 4, 0, 4};
            }
        }

        [[nodiscard]] static int TextWidth(std::string_view text, int scale = 1) noexcept {
            if(text.empty() || scale < 1 || scale > 8) return 0;
            const auto step = static_cast<std::size_t>(Advance * scale);
            if(text.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) / step)
                return std::numeric_limits<int>::max();
            return static_cast<int>(text.size() * step) - scale;
        }

        static void Draw(graphics::FrameBuffer& target, int x, int y, std::string_view text, std::uint32_t color,
                         int scale = 1) noexcept {
            if(scale < 1 || scale > 8) return;
            const auto width = target.GetWidth(), height = target.GetHeight();
            auto pixels = target.Colors();
            std::int64_t originX = x;
            for(char character : text) {
                if(originX >= static_cast<std::int64_t>(width)) break;
                const auto glyph = GetGlyph(character);
                for(int row = 0; row < GlyphHeight; ++row) {
                    for(int column = 0; column < GlyphWidth; ++column) {
                        if((glyph[row] & (1u << (GlyphWidth - 1 - column))) == 0) continue;
                        for(int dy = 0; dy < scale; ++dy) {
                            const std::int64_t py = static_cast<std::int64_t>(y) + row * scale + dy;
                            if(py < 0 || py >= static_cast<std::int64_t>(height)) continue;
                            for(int dx = 0; dx < scale; ++dx) {
                                const std::int64_t px = originX + column * scale + dx;
                                if(px < 0 || px >= static_cast<std::int64_t>(width)) continue;
                                pixels[static_cast<std::size_t>(py) * width + static_cast<std::size_t>(px)] = color;
                            }
                        }
                    }
                }
                originX += Advance * scale;
            }
        }
    };
}
