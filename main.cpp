#include <GLFW/glfw3.h>
#include <gl/gl.h>

#include <cstdio>

#include "World.hpp"
#include "graphics/FrameBuffer.hpp"
#include "graphics/Rasterizer.hpp"

void ErrCallback(int error, const char* description) {
    std::fprintf(stderr, "ERROR : %s\n", description);
}

int main() {
    glfwSetErrorCallback(ErrCallback);

    if(!glfwInit()) return -1;

    GLFWwindow* window = glfwCreateWindow(world::WIDTH, world::HEIGHT, "Software Rasterizer", nullptr, nullptr);
    if(!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    graphics::FrameBuffer frame(world::WIDTH, world::HEIGHT);
    float angle = 0.0f;

    while(!glfwWindowShouldClose(window)) {
        frame.Clear(world::COLOR);

        angle += 0.02f;
        shader::Default shader{world::GetMVP(angle), math::CreateViewport(static_cast<float>(world::WIDTH),
                                                                          static_cast<float>(world::HEIGHT))};

        graphics::Render(frame, shader, world::ModelVertices, world::ModelIndices, graphics::PrimitiveType::Triangles);

        glRasterPos2f(-1, 1);
        glPixelZoom(1, -1);
        glDrawPixels(world::WIDTH, world::HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, frame.GetColor());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
