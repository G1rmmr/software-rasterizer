#include <array>
#include <exception>
#include <iostream>
#include <string_view>
void RunMathTests();
void RunRasterizerTests();
void RunSceneTests();
void RunPostProcessTests();
void RunShaderTests();
void RunRendererTests();
void RunAppStateTests();
void RunProfilerTests();
void WriteProfilerSnapshot(const char* path);

int main(int argc, char** argv) {
    struct Group {
        std::string_view Name;
        void (*Run)();
    };
    const std::array groups{Group{"math", RunMathTests},         Group{"rasterizer", RunRasterizerTests},
                            Group{"scene", RunSceneTests},       Group{"postprocess", RunPostProcessTests},
                            Group{"shader", RunShaderTests},     Group{"renderer", RunRendererTests},
                            Group{"appstate", RunAppStateTests}, Group{"profiler", RunProfilerTests}};
    bool found = false;
    for(const auto& group : groups) {
        if(argc > 1 && group.Name != argv[1]) continue;
        found = true;
        try {
            group.Run();
            if(group.Name == "profiler" && argc == 3) WriteProfilerSnapshot(argv[2]);
            std::cout << "[PASS] " << group.Name << '\n';
        }
        catch(const std::exception& error) {
            std::cerr << "[FAIL] " << group.Name << ": " << error.what() << '\n';
            return 1;
        }
    }
    if(!found) {
        std::cerr << "Unknown test group\n";
        return 1;
    }
    return 0;
}
