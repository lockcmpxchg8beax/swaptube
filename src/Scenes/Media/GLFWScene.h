#pragma once

#include <bit>
#include <mutex>
#include <GLFW/glfw3.h>
#include "../Scene.h"

class GLFWInstance
{
    // only hold a weak reference here so glfwTerminate() doesn't crash during static destructor
    inline static std::mutex mutex;
    inline static std::weak_ptr<GLFWInstance> ptr;

public:
    inline GLFWInstance()
    {
        if (!glfwInit()) {
            throw std::runtime_error("glfwInit() failed");
        }
    }

    inline ~GLFWInstance()
    {
        glfwTerminate();
    }

    static std::shared_ptr<GLFWInstance> ref()
    {
        std::lock_guard guard(mutex);

        if (ptr.expired())
        {
            auto instance = std::make_shared<GLFWInstance>();
            ptr = instance;
            return instance;
        }
        else
        {
            return ptr.lock();
        }
    }
};

template<typename Derived>
class GLFWScene
    : public Scene
{
protected:
    std::shared_ptr<GLFWInstance> glfw;
    unsigned tex_width;
    unsigned tex_height;
    GLFWwindow* window;

private:
    inline GLFWwindow* create_window(unsigned width, unsigned height)
    {
        static_cast<Derived*>(this)->before_create_context();

        auto window = glfwCreateWindow(width, height, "", nullptr, nullptr);

        if (!window)
        {
            const char* error;
            glfwGetError(&error);

            throw std::runtime_error("glfwCreateWindow() failed: "s + error);
        }

        return window;
    }

public:
    // some implementations require texture dimensions to be a power of 2, round up to be safe
    GLFWScene(const vec2& dimensions = vec2(1, 1))
        : Scene(dimensions)
        , glfw(GLFWInstance::ref())
        , tex_width(std::bit_ceil(static_cast<unsigned>(get_width())))
        , tex_height(std::bit_ceil(static_cast<unsigned>(get_height())))
        , window(create_window(tex_width, tex_height))
    {
        // potential use of uninitialized member (window) if called at the end of create_window before returning
        static_cast<Derived*>(this)->after_create_context();
    }

    ~GLFWScene()
    {
        glfwDestroyWindow(window);
    }
};
