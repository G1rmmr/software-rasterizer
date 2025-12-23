#include <GLFW/glfw3.h>

#include <iostream>

void ErrCallback(int error, const char* description){
    std::cerr << "ERROR : " << description << std::endl;
}

int main(){
    glfwSetErrorCallback(ErrCallback);

    if(!glfwInit()) return -1;

    GLFWwindow* window = glfwCreateWindow(640, 480, "Software Rasterizer", nullptr, nullptr);
    if(!window){
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    while (!glfwWindowShouldClose(window))
    {
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}