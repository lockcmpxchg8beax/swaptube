#define GLAD_GLES2_IMPLEMENTATION

#include "GLES2Scene.h"

extern "C" void cuda_copy_pixels_to_device(uint32_t* h_pixels, int size, uint32_t* d_pixels);

GLES2Scene::GLES2Scene(const std::string& glsl_vs, const std::string& glsl_fs, const vec2& dimensions)
    : GLFWScene(dimensions)
    , tex(tex_width, tex_height, GL_BGRA_EXT, GL_BGRA_EXT, GL_UNSIGNED_BYTE)
    , fbo(tex)
    , vs(GL_VERTEX_SHADER, glsl_vs)
    , fs(GL_FRAGMENT_SHADER, glsl_fs)
    , prog(vs, fs)
    , buf(std::make_unique_for_overwrite<uint32_t[]>(get_width() * get_height()))
{
    glViewport(0, 0, tex_width, tex_height);
}

void GLES2Scene::before_create_context()
{
    // EGL for GPU acceleration
    glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);

    // select OpenGL ES 3.0
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    // keep windows hidden for headless rendering
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
}

void GLES2Scene::after_create_context()
{
    glfwMakeContextCurrent(window);

    if (!glfwExtensionSupported("GL_EXT_texture_format_BGRA8888")) {
        throw std::runtime_error("GL_EXT_texture_format_BGRA8888 not supported");
    }

    if (int version = gladLoadGLES2(glfwGetProcAddress); version == 0)
    {
        throw std::runtime_error("gladLoadGLES2() failed");
    }
    else
    {
        std::cout << "Loaded OpenGL ES " << GLAD_VERSION_MAJOR(version) << "." << GLAD_VERSION_MINOR(version) << std::endl;
    }
}

void GLES2Scene::bind_uniform_impl(const std::string& uniform, std::function<void(GLuint)>&& callback)
{
    glfwMakeContextCurrent(window);
    prog.use();

    auto loc = glGetUniformLocation(prog.get(), uniform.c_str());
    uniforms.insert_or_assign(
        uniform,
        [loc, callback = std::move(callback)]{ callback(loc); }
    );
}

#define DEFINE_UNIFORM_TYPE(T, glUniformT, ...) \
    template<> \
    void GLES2Scene::bind_uniform_aux<T>(const std::string& uniform, std::function<T()>&& callback) \
    { \
        bind_uniform_impl(uniform, [callback = std::move(callback)](GLuint loc){ auto v = callback(); glUniformT(loc, __VA_ARGS__); }); \
    }

DEFINE_UNIFORM_TYPE(int, glUniform1i, v);
DEFINE_UNIFORM_TYPE(ivec2, glUniform2i, v.x, v.y);
DEFINE_UNIFORM_TYPE(ivec3, glUniform3i, v.x, v.y, v.z);
DEFINE_UNIFORM_TYPE(ivec4, glUniform4i, v.x, v.y, v.z, v.w);
DEFINE_UNIFORM_TYPE(float, glUniform1f, v);
DEFINE_UNIFORM_TYPE(vec2, glUniform2f, v.x, v.y);
DEFINE_UNIFORM_TYPE(vec3, glUniform3f, v.x, v.y, v.z);
DEFINE_UNIFORM_TYPE(vec4, glUniform4f, v.x, v.y, v.z, v.w);

void GLES2Scene::draw()
{
    glfwMakeContextCurrent(window);
    fbo.bind();
    prog.use();

    for (const auto& [_, func] : uniforms)
    {
        func();
    }

    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();

    glReadPixels(0, 0, get_width(), get_height(), GL_BGRA_EXT, GL_UNSIGNED_BYTE, buf.get());
    cuda_copy_pixels_to_device(buf.get(), get_width() * get_height(), gpu_pix.get_ptr());
}
