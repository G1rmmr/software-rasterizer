cmake_minimum_required(VERSION 3.21)
project(software-rasterizer LANGUAGES CXX)

include(FetchContent)
cmake_policy(SET CMP0168 NEW)

# MIR has an Xmake build but its ECS core is header based.  Fetch the exact
# reviewed revisions and let this CMake project own the MSVC ABI and flags.
FetchContent_Declare(zet
    GIT_REPOSITORY https://github.com/G1rmmr/zetcontainer-cpp.git
    GIT_TAG ac876f147417256753d9a1daaf5d96c77940b842
    GIT_SHALLOW TRUE)
FetchContent_Declare(mirengine
    GIT_REPOSITORY https://github.com/G1rmmr/mirengine-cpp.git
    GIT_TAG 42519791c9638c4b09f1af82d46df283548e368b
    GIT_SHALLOW TRUE)
# Neither upstream dependency supplies a CMake project.  MakeAvailable with
# CMP0168 fetches their source directly, without generating an MSBuild
# subproject; this project deliberately compiles only ZET's implementation.
FetchContent_MakeAvailable(zet mirengine)

if(NOT WIN32 OR NOT MSVC OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8
   OR NOT CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "x64")
    message(FATAL_ERROR "The bundled MiniFB library requires Windows x64 with MSVC. Use the windows-msvc preset.")
endif()

# All targets share one ABI and floating-point contract.
add_library(rasterizer-options INTERFACE)
target_compile_features(rasterizer-options INTERFACE cxx_std_20)
target_compile_options(rasterizer-options INTERFACE /utf-8 /arch:AVX2 /fp:precise /W4
    "$<$<CONFIG:Release>:/O2;/Oi>")
target_compile_definitions(rasterizer-options INTERFACE _ITERATOR_DEBUG_LEVEL=0 NOMINMAX)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
set(CMAKE_CXX_EXTENSIONS OFF)

# Only the generation-safe ECS core is linked.  Do not bring MIR's SDL3
# window, Lua, audio, or 2D renderer into the MiniFB/CPU-renderer process.
add_library(mir-ecs STATIC "${zet_SOURCE_DIR}/src/zet.cpp")
target_include_directories(mir-ecs PUBLIC
    "${zet_SOURCE_DIR}/src"
    "${zet_SOURCE_DIR}/src/container"
    "${zet_SOURCE_DIR}/src/memory"
    "${mirengine_SOURCE_DIR}/engine")
target_compile_definitions(mir-ecs PUBLIC
    # The rasterizer scene has two static objects in addition to --instances.
    CONFIG_MAX_ENTITY=4098)
target_link_libraries(mir-ecs PUBLIC rasterizer-options)

add_library(rasterizer-core STATIC
    graphics/Texture.cpp graphics/Mesh.cpp
    scene/Model.cpp scene/SceneObject.cpp scene/MirScene.cpp
    resources/ObjLoader.cpp resources/ModelFactory.cpp
    graphics/Renderer.cpp graphics/passes/ShadowPass.cpp graphics/passes/GeometryPass.cpp
    graphics/post/SsaoPass.cpp graphics/post/AntiAliasingPass.cpp
)
target_include_directories(rasterizer-core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(rasterizer-core PUBLIC rasterizer-options mir-ecs)

add_library(minifb STATIC IMPORTED)
set_target_properties(minifb PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/libs/minifb.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/libs")
target_link_libraries(minifb INTERFACE gdi32 user32 shell32 winmm opengl32)

add_executable(software-rasterizer main.cpp app/Application.cpp app/Window.cpp
    app/ProfilerView.cpp app/ImageWriter.cpp cpp.hint)
target_link_libraries(software-rasterizer PRIVATE rasterizer-core minifb)
set_target_properties(software-rasterizer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin/$<CONFIG>"
    # Visual Studio's CMake folder debugger can read this property literally.
    # Use a configure-time path so both folder mode and generated solutions work.
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT software-rasterizer)

include(CTest)
if(BUILD_TESTING)
    add_executable(rasterizer-tests tests/Main.cpp tests/MathTests.cpp tests/RasterizerTests.cpp
        tests/SceneTests.cpp tests/PostProcessTests.cpp tests/ShaderTests.cpp tests/RendererTests.cpp
        tests/AppStateTests.cpp tests/ProfilerTests.cpp tests/MirSceneTests.cpp app/ProfilerView.cpp app/ImageWriter.cpp)
    target_link_libraries(rasterizer-tests PRIVATE rasterizer-core)
    foreach(group math rasterizer scene postprocess shader renderer appstate profiler mirscene)
        add_test(NAME ${group} COMMAND rasterizer-tests ${group})
        set_tests_properties(${group} PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" TIMEOUT 60)
    endforeach()
    add_test(NAME headless-smoke COMMAND software-rasterizer --headless --frames 2 --width 65 --height 49
        --model cube --workers 2 --output "${CMAKE_CURRENT_BINARY_DIR}/smoke-$<CONFIG>.bmp")
    set_tests_properties(headless-smoke PROPERTIES TIMEOUT 60)
    add_test(NAME headless-instances COMMAND software-rasterizer --headless --frames 2 --width 65 --height 49
        --model cube --instances 4 --workers 2 --output "${CMAKE_CURRENT_BINARY_DIR}/instances-$<CONFIG>.bmp")
    set_tests_properties(headless-instances PROPERTIES TIMEOUT 60)
    add_test(NAME headless-mir-instances COMMAND software-rasterizer --headless --frames 2 --width 65 --height 49
        --model cube --instances 4 --scene-backend mir --workers 2
        --output "${CMAKE_CURRENT_BINARY_DIR}/mir-instances-$<CONFIG>.bmp")
    set_tests_properties(headless-mir-instances PROPERTIES TIMEOUT 60)
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error LANGUAGES CXX)
if(ipo_supported)
    set_property(TARGET rasterizer-core software-rasterizer PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    if(BUILD_TESTING)
        set_property(TARGET rasterizer-tests PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()
else()
    message(STATUS "Release link-time optimization unavailable: ${ipo_error}")
endif()

# Run on every build so asset-only edits are copied without a relink.
add_custom_target(copy_assets
    COMMAND "${CMAKE_COMMAND}" -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/assets"
        "${CMAKE_CURRENT_BINARY_DIR}/bin/$<CONFIG>/assets"
    COMMENT "Copying runtime assets" VERBATIM)
add_dependencies(software-rasterizer copy_assets)
add_custom_target(run COMMAND "$<TARGET_FILE:software-rasterizer>"
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:software-rasterizer>"
    DEPENDS software-rasterizer USES_TERMINAL VERBATIM)

