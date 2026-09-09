#include "app/Application.hpp"
#include <charconv>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {
    std::uint32_t Number(std::string_view text, const char* option, bool allowZero = false) {
        std::uint32_t value{};
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
        if(error != std::errc{} || end != text.data() + text.size() || (!allowZero && value == 0))
            throw std::invalid_argument(std::string("Invalid value for ") + option);
        return value;
    }
    float Distance(std::string_view text) {
        float value{};
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
        if(error != std::errc{} || end != text.data() + text.size() || !std::isfinite(value) || value < 1.f ||
           value > 200.f)
            throw std::invalid_argument("--distance requires a number within 1..200");
        return value;
    }
}
int main(int argc, char** argv) {
    try {
        app::Options options;
        const auto besideExecutable = std::filesystem::absolute(argv[0]).parent_path() / "assets";
        if(std::filesystem::is_directory(besideExecutable)) options.Assets = besideExecutable;
        for(int i = 1; i < argc; ++i) {
            const std::string_view arg = argv[i];
            const auto value = [&]() -> std::string_view {
                if(++i >= argc) throw std::invalid_argument(std::string("Missing value for ") + std::string(arg));
                return argv[i];
            };
            if(arg == "--headless")
                options.Headless = true;
            else if(arg == "--benchmark") {
                options.Benchmark = true;
                options.Headless = true;
            }
            else if(arg == "--warmup")
                options.WarmupFrames = Number(value(), "--warmup", true);
            else if(arg == "--distance")
                options.CameraDistance = Distance(value());
            else if(arg == "--no-profiler")
                options.Profiler = false;
            else if(arg == "--frames")
                options.Frames = Number(value(), "--frames");
            else if(arg == "--width")
                options.Width = Number(value(), "--width");
            else if(arg == "--height")
                options.Height = Number(value(), "--height");
            else if(arg == "--workers")
                options.Workers = Number(value(), "--workers", true);
            else if(arg == "--model")
                options.Model = value();
            else if(arg == "--assets")
                options.Assets = value();
            else if(arg == "--output")
                options.Output = value();
            else if(arg == "--no-shadows")
                options.Rendering.Shadows = false;
            else if(arg == "--no-ssao")
                options.Rendering.AmbientOcclusion = false;
            else if(arg == "--no-aa")
                options.Rendering.AntiAliasing = false;
            else if(arg == "--toon")
                options.Rendering.Toon = true;
            else if(arg == "--wireframe")
                options.Rendering.Primitive = graphics::PrimitiveType::Lines;
            else if(arg == "--points")
                options.Rendering.Primitive = graphics::PrimitiveType::Points;
            else if(arg == "--help") {
                std::cout << "software-rasterizer [--headless|--benchmark] [--frames N] [--width N] [--height N]\n"
                             "  [--distance 1..200] (default: 45; smaller values zoom in)\n"
                             "  [--warmup N] (benchmark defaults: 30 warmup, 300 measured frames)\n"
                             "  [--model diablo|african|cube|sphere] [--assets DIR] [--output FILE.bmp]\n"
                             "  [--workers N] [--no-shadows] [--no-ssao] [--no-aa] [--no-profiler]\n"
                             "  [--toon] [--wireframe|--points]\n"
                             "Controls: SPACE primitive, Q shadow, W SSAO, E AA, R toon, ESC quit.\n"
                             "Drag left mouse to change light; wheel to zoom.\n";
                return 0;
            }
            else
                throw std::invalid_argument("Unknown option: " + std::string(arg));
        }
        app::Application application(std::move(options));
        return application.Run();
    }
    catch(const std::exception& error) {
        std::cerr << "software-rasterizer: " << error.what() << '\n';
        return 1;
    }
}
