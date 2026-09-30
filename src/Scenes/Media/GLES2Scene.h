#pragma once

#include <concepts>
#include <string>
#include "gles2.h"
#include "GLFWScene.h"

template<typename Derived>
class GLObject
{
protected:
    const GLuint obj;

    GLObject(GLuint value)
        : obj(value)
    {
    }

public:
    inline GLuint get() const
    {
        return obj;
    }

    GLObject(const GLObject&) = delete;
    GLObject& operator =(const GLObject&) = delete;
};

class GLTexture
    : public GLObject<GLTexture>
{
    inline static GLuint create_texture(unsigned width, unsigned height, GLint internal_format, GLenum format, GLenum type)
    {
        GLuint tex;
        glGenTextures(1, &tex);

        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, format, type, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        return tex;
    }

public:
    inline GLTexture(unsigned width, unsigned height, GLint internal_format, GLenum format, GLenum type)
        : GLObject(create_texture(width, height, internal_format, format, type))
    {
    }

    inline ~GLTexture()
    {
        glDeleteTextures(1, const_cast<GLuint*>(&obj));
    }

    inline void bind() const
    {
        glBindTexture(GL_TEXTURE_2D, obj);
    }
};

class GLFramebuffer
    : public GLObject<GLFramebuffer>
{
    template<std::size_t... Is>
    inline static GLuint create_framebuffer(std::index_sequence<Is...>, const auto&... textures)
    requires (std::same_as<GLTexture, std::remove_cvref_t<decltype(textures)>> && ...)
    {
        GLuint fbo;
        glGenFramebuffers(1, &fbo);

        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        (glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + Is, GL_TEXTURE_2D, textures.get(), 0), ...);

        if (auto status = glCheckFramebufferStatus(GL_FRAMEBUFFER); status != GL_FRAMEBUFFER_COMPLETE)
        {
            throw std::runtime_error("glCheckFramebufferStatus() returned " + std::to_string(status));
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        return fbo;
    }

public:
    inline GLFramebuffer(const auto&... textures)
    requires (std::same_as<GLTexture, std::remove_cvref_t<decltype(textures)>> && ...)
        : GLObject(create_framebuffer(std::index_sequence_for<decltype(textures)...>{}, textures...))
    {
    }

    inline ~GLFramebuffer()
    {
        glDeleteFramebuffers(1, const_cast<GLuint*>(&obj));
    }

    inline void bind() const
    {
        glBindFramebuffer(GL_FRAMEBUFFER, obj);
    }
};

class GLShader
    : public GLObject<GLShader>
{
    const std::string glsl;

    inline static GLuint try_compile_shader(GLenum type, const char* glsl)
    {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &glsl, nullptr);
        glCompileShader(shader);

        GLint status;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
        if (status != GL_TRUE)
        {
            GLint length;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);

            std::string log;
            log.resize(length);

            glGetShaderInfoLog(shader, length, nullptr, log.data());

            // strip terminating null character
            log.resize(length - 1);

            throw std::runtime_error("glCompileShader() failed:\n" + log + "\n\nShader source:\n" + glsl);
        }

        return shader;
    }

public:
    inline GLShader(GLenum type, const std::string& glsl)
        : GLObject(try_compile_shader(type, glsl.c_str()))
        , glsl(glsl)
    {
    }

    inline ~GLShader()
    {
        glDeleteShader(obj);
    }

    inline const std::string& source() const
    {
        return glsl;
    }
};

class GLProgram
    : public GLObject<GLProgram>
{
    inline static GLuint try_link_program(const GLShader& vs, const GLShader& fs)
    {
        GLuint prog = glCreateProgram();
        glAttachShader(prog, vs.get());
        glAttachShader(prog, fs.get());
        glLinkProgram(prog);

        glDetachShader(prog, vs.get());
        glDetachShader(prog, fs.get());

        GLint status;
        glGetProgramiv(prog, GL_LINK_STATUS, &status);
        if (status != GL_TRUE)
        {
            GLint length;
            glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &length);

            std::string log;
            log.resize(length);

            glGetProgramInfoLog(prog, length, nullptr, log.data());

            // strip terminating null character
            log.resize(length - 1);

            throw std::runtime_error("glLinkProgram() failed:\n" + log + "\n\nVS source:\n" + vs.source() + "\nFS source:\n" + fs.source());
        }

        return prog;
    }

public:
    inline GLProgram(const GLShader& vs, const GLShader& fs)
        : GLObject(try_link_program(vs, fs))
    {
    }

    inline ~GLProgram()
    {
        glDeleteProgram(obj);
    }

    inline void use() const
    {
        glUseProgram(obj);
    }
};

class GLES2Scene
    : public GLFWScene<GLES2Scene>
{
    GLTexture tex;
    GLFramebuffer fbo;
    GLShader vs;
    GLShader fs;
    GLProgram prog;
    std::unique_ptr<uint32_t[]> buf;
    std::unordered_map<std::string, std::function<void()>> uniforms;

    template<typename Type>
    void bind_uniform_aux(const std::string& uniform, std::function<Type()>&& callback);
    void bind_uniform_impl(const std::string& uniform, std::function<void(GLuint)>&& callback);

public:
    GLES2Scene(const std::string& glsl_vs, const std::string& glsl_fs, const vec2& dimensions = vec2(1, 1));

    void before_create_context();
    void after_create_context();

    template<typename Func>
    void bind_uniform(const std::string& uniform, Func&& callback)
    {
        bind_uniform_aux<std::invoke_result_t<Func>>(uniform, std::function{std::forward<Func>(callback)});
    }

    void draw() override;
};
