#pragma once

#include <vector>

#include "Math.h"
#include "SIMD.h"

namespace graphics{
    class FrameBuffer{
    public:
        inline FrameBuffer(const std::uint32_t width, const std::uint32_t height)
            : colorBuffer(width * height, 0)
            , depthBuffer(width * height, 1.0f)
            , width(width)
            , height(height){}

        inline void Clear(const std::uint32_t clearColor = 0) {
            std::fill(colorBuffer.begin(), colorBuffer.end(), clearColor);
            std::fill(depthBuffer.begin(), depthBuffer.end(), 1.0f);
        }

        inline void SetPixel(
            const std::uint32_t x, const std::uint32_t y, const std::uint32_t color) {
            if(x < width && y < height)
                colorBuffer[y * width + x] = color;
        }

    private:
        std::vector<std::uint32_t> colorBuffer;
        std::vector<float> depthBuffer;
        std::uint32_t width;
        std::uint32_t height;
    };

    inline math::Matrix4 CreatePerspective(
        const float fov,
        const float aspect,
        const float near,
        const float far){
        const float h = 1.f / std::tan(fov * 0.5f);
        const float a = far / (far - near);

        math::Matrix4 mat;
        mat[0][0] = h / aspect;
        mat[1][1] = h;
        mat[2][2] = a;
        mat[2][3] = -near * a;
        mat[3][2] = 1.f;
	    return mat;
    }

    inline math::Matrix4 CreateLookAt(
        const math::Vector3& eye,
        const math::Vector3& target,
        const math::Vector3& WorldUp = math::Vector3(0.f, 1.f, 0.f, 0.f)) {
        const math::Vector3 forward = (target - eye).Norm();
        const math::Vector3 right = WorldUp.Cross(forward).Norm();
        const math::Vector3 up = forward.Cross(right).Norm();

        math::Matrix4 mat;
        mat[0] = simd::Set(right.X, up.X, forward.X, 0.f);
        mat[1] = simd::Set(right.Y, up.Y, forward.Y, 0.f);
        mat[2] = simd::Set(right.Z, up.Z, forward.Z, 0.f);
        mat[3] = simd::Set(-right.Dot(eye), -up.Dot(eye), -forward.Dot(eye), 1.f);
        return mat;
    }

    inline math::Matrix4 CreateScale(const math::Vector3& scale){
        math::Matrix4 mat;
        mat[0][0] = scale.X;
        mat[1][1] = scale.Y;
        mat[2][2] = scale.Z;
        return mat;
    }

    inline math::Matrix4 CreateRotation(const math::Vector3& eular){
        math::Quaternion rotation = math::FromAxisAngle(eular);
        return rotation.ToMatrix();
    }

    inline math::Matrix4 CreateTranslation(const math::Vector3& position){
        math::Matrix4 mat;
        mat[3] = simd::Set(position.X, position.Y, position.Z, 1.f);
        return mat;
    }
}
