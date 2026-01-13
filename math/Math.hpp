#pragma once

#include <cmath>
#include <numbers>
#include <random>

#include "Matrix.hpp"
#include "Quaternion.hpp"
#include "SIMD.hpp"
#include "Vector.hpp"

namespace math {
    namespace {
        ENGINE_INLINE std::mt19937& GetRandomEngine() {
            static thread_local std::random_device rd;
            static thread_local std::mt19937 gen(rd());
            return gen;
        }
    }

    ENGINE_INLINE float RandomFloat(const float min, const float max) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(GetRandomEngine());
    }

    ENGINE_INLINE Vector CreateRandomVector(const float min, const float max) {
        return Vector(RandomFloat(min, max), RandomFloat(min, max), RandomFloat(min, max), 1.f);
    }

    ENGINE_INLINE Matrix __vectorcall CreateRandomMatrix(const float min, const float max) {
        Matrix mat;
        for(int i = 0; i < 4; ++i) {
            mat.Cols[i] =
                simd::Set(RandomFloat(min, max), RandomFloat(min, max), RandomFloat(min, max), RandomFloat(min, max));
        }
        return mat;
    }

    ENGINE_INLINE Vector __vectorcall operator*(const Matrix& mat, const Vector& vec) noexcept {
        simd::Floats res = simd::Mul(mat.Cols[0], simd::Set(vec.X));
        res = simd::Add(res, simd::Mul(mat.Cols[1], simd::Set(vec.Y)));
        res = simd::Add(res, simd::Mul(mat.Cols[2], simd::Set(vec.Z)));
        res = simd::Add(res, simd::Mul(mat.Cols[3], simd::Set(vec.W)));

        return Vector(res);
    }

    ENGINE_INLINE Quaternion __vectorcall FromAxisAngle(const Vector& axis, const float radian) noexcept {
        return Quaternion(axis.Norm() * std::sin(radian * 0.5f), std::cos(radian * 0.5f));
    }

    ENGINE_INLINE Vector __vectorcall GetBarycentric(const Vector& pos, const Vector& a, const Vector& b,
                                                     const Vector& c) noexcept {
        const float area = (b - a).Cross2D(c - a);

        if(std::abs(area) < 1e-6f) [[unlikely]]
            return Vector(-1.f, -1.f, -1.f, 0.f);

        const float invArea = 1.f / area;

        const float wA = (b - pos).Cross2D(c - pos) * invArea;
        const float wB = (c - pos).Cross2D(a - pos) * invArea;
        const float wC = 1.f - wA - wB;

        return Vector(wA, wB, wC, 0.f);
    }

    ENGINE_INLINE Matrix __vectorcall CreateViewport(const std::uint32_t screenWidth,
                                                     const std::uint32_t screenHeight) {
        const float width = static_cast<float>(screenWidth);
        const float height = static_cast<float>(screenHeight);

        Matrix mat;
        mat[0][0] = width * 0.5f;
        mat[1][1] = -height * 0.5f;
        mat[2][2] = 1.f;
        mat[3][0] = width * 0.5f;
        mat[3][1] = height * 0.5f;
        mat[3][3] = 1.f;

        return mat;
    }

    ENGINE_INLINE Matrix __vectorcall CreateLookAt(const Vector& eye, const Vector& target, const Vector& up) {
        const Vector z = (eye - target).Norm();
        const Vector x = up.Cross(z).Norm();
        const Vector y = z.Cross(x);

        Matrix mat;
        mat.Cols[0] = simd::Set(x.X, y.X, z.X, 0.f);
        mat.Cols[1] = simd::Set(x.Y, y.Y, z.Y, 0.f);
        mat.Cols[2] = simd::Set(x.Z, y.Z, z.Z, 0.f);
        mat.Cols[3] = simd::Set(-x.Dot(eye), -y.Dot(eye), -z.Dot(eye), 1.f);

        return mat;
    }

    ENGINE_INLINE Matrix __vectorcall CreatePerspective(const float fov, const float aspect, const float near,
                                                        const float far) {
        const float tanHalfFov = std::tan(fov * 0.5f);

        Matrix mat(0.f);
        mat[0][0] = 1.f / (aspect * tanHalfFov);
        mat[1][1] = 1.f / tanHalfFov;
        mat[2][2] = far / (near - far);
        mat[2][3] = -1.f;
        mat[3][2] = (far * near) / (near - far);
        mat[3][3] = 0.f;

        return mat;
    }

    ENGINE_INLINE Matrix __vectorcall CreateOrtho(float left, float right, float bottom, float top, float near,
                                                  float far) {
        Matrix mat(0.f);
        mat[0][0] = 2.f / (right - left);
        mat[1][1] = 2.f / (top - bottom);
        mat[2][2] = -2.f / (far - near);
        mat[0][3] = -(right + left) / (right - left);
        mat[1][3] = -(top + bottom) / (top - bottom);
        mat[2][3] = -(far + near) / (far - near);
        mat[3][3] = 1.f;

        return mat;
    }

    ENGINE_INLINE Matrix __vectorcall CreateScale(const Vector& scale) {
        Matrix mat;
        mat[0][0] = scale.X;
        mat[1][1] = scale.Y;
        mat[2][2] = scale.Z;

        return mat;
    }

    ENGINE_INLINE Matrix __vectorcall CreateRotation(const Vector& axis, const float radian) {
        return FromAxisAngle(axis, radian).ToMatrix();
    }

    ENGINE_INLINE Matrix __vectorcall CreateTranslation(const Vector& position) {
        Matrix mat;
        mat.Cols[3] = simd::Set(position.X, position.Y, position.Z, 1.f);

        return mat;
    }

    ENGINE_INLINE constexpr float ToRadian(const float degree) noexcept {
        return degree * (std::numbers::pi_v<float> / 180.f);
    }

    ENGINE_INLINE constexpr float ToDegree(const float radian) noexcept {
        return radian * (180.f / std::numbers::pi_v<float>);
    }
}