#include <GLFW/glfw3.h>

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <cstdint>
#include <cstdio>

#include "World.hpp"
#include "graphics/FrameBuffer.hpp"
#include "graphics/Rasterizer.hpp"

void ErrCallback(int error, const char* description) {
    std::fprintf(stderr, "ERROR : %s\n", description);
}

int main() {
    glfwSetErrorCallback(ErrCallback);

    const char* WINDOW_TITLE = "Software Rasterizer";

    if(!glfwInit()) return -1;

    GLFWwindow* window = glfwCreateWindow(world::WIDTH, world::HEIGHT, WINDOW_TITLE, nullptr, nullptr);
    if(!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    graphics::FrameBuffer frame(world::WIDTH, world::HEIGHT);
    float angle = 0.f;

    // world::CreateSphere(10, 20, 20);

    while(!glfwWindowShouldClose(window)) {
        frame.Clear(world::COLOR);

        int fbW = 0, fbH = 0;
        int winW = 0, winH = 0;
        glfwGetFramebufferSize(window, &fbW, &fbH);
        glfwGetWindowSize(window, &winW, &winH);

        glViewport(0, 0, fbW, fbH);

        angle += 0.02f;
        shader::Default shader{
            world::GetMVP(angle),
            math::CreateViewport(static_cast<float>(world::WIDTH), static_cast<float>(world::HEIGHT))
        };

        graphics::Render(frame, shader, world::ModelVertices, world::ModelIndices);

        float scaleX = static_cast<float>(fbW) / world::WIDTH;
        float scaleY = static_cast<float>(fbH) / world::HEIGHT;

        glRasterPos2f(-1.f, 1.f);
        glPixelZoom(scaleX, -scaleY);

        glDrawPixels(world::WIDTH, world::HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, frame.GetColor());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
