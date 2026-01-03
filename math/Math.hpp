#pragma once

#include <cmath>
#include <numbers>

#include "Matrix.hpp"
#include "Quaternion.hpp"
#include "SIMD.hpp"
#include "Vector.hpp"

namespace math {
    inline Vector operator*(const Matrix& mat, const Vector& vec) noexcept {
        simd::Floats res = simd::Mul(mat.Cols[0], simd::Set(vec.X));
        res = simd::Add(res, simd::Mul(mat.Cols[1], simd::Set(vec.Y)));
        res = simd::Add(res, simd::Mul(mat.Cols[2], simd::Set(vec.Z)));
        res = simd::Add(res, simd::Mul(mat.Cols[3], simd::Set(vec.W)));
        return Vector(res);
    }

    inline Quaternion FromAxisAngle(const Vector& axis, const float radian) noexcept {
        Quaternion result(axis.Norm() * std::sin(radian * 0.5f), std::cos(radian * 0.5f));

        return result;
    }

    inline Vector GetBarycentric(const Vector& pos, const Vector& a, const Vector& b, const Vector& c) noexcept {
        float area = (b - a).Cross2D(c - a);

        if(std::abs(area) < 1e-6f) return Vector(-1.f, -1.f, -1.f, 0.f);

        float wA = (b - pos).Cross2D(c - pos) / area;
        float wB = (c - pos).Cross2D(a - pos) / area;
        float wC = 1.f - wA - wB;

        return Vector(wA, wB, wC, 0.f);
    }

    inline Matrix CreateViewport(const float width, const float height) {
        Matrix mat;
        mat[0][0] = width * 0.5f;
        mat[1][1] = -height * 0.5f;
        mat[2][2] = 1.f;
        mat[3][0] = width * 0.5f;
        mat[3][1] = height * 0.5f;
        mat[3][3] = 1.f;
        return mat;
    }

    inline Matrix CreateLookAt(const Vector& eye, const Vector& target, const Vector& up) {
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

    inline Matrix CreatePerspective(const float fov, const float aspect, const float near, const float far) {
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

    inline Matrix CreateScale(const Vector& scale) {
        Matrix mat;
        mat[0][0] = scale.X;
        mat[1][1] = scale.Y;
        mat[2][2] = scale.Z;
        return mat;
    }

    inline Matrix CreateRotation(const Vector& axis, const float radian) {
        Quaternion rotation = FromAxisAngle(axis, radian);
        return rotation.ToMatrix();
    }

    inline Matrix CreateTranslation(const Vector& position) {
        Matrix mat;
        mat.Cols[3] = simd::Set(position.X, position.Y, position.Z, 1.f);
        return mat;
    }

    inline constexpr float ToRadian(const float degree) noexcept {
        return degree * (std::numbers::pi_v<float> / 180.f);
    }

    inline constexpr float ToDegree(const float radian) noexcept {
        return radian * (180.f / std::numbers::pi_v<float>);
    }
}