#pragma once

#include <cassert>
#include <cmath>

#include "SIMD.hpp"
#include "Vector.hpp"

namespace math {
    struct alignas(16) Matrix {
        union {
            simd::Floats Cols[4];
            float Elements[16];
        };

        ENGINE_INLINE float* __restrict operator[](int index) noexcept {
            return reinterpret_cast<float*>(&Cols[index]);
        }

        ENGINE_INLINE const float* __restrict operator[](int index) const noexcept {
            return reinterpret_cast<const float*>(&Cols[index]);
        }

        ~Matrix() noexcept = default;

        ENGINE_INLINE Matrix() noexcept {
            Cols[0] = simd::Set(1, 0, 0, 0);
            Cols[1] = simd::Set(0, 1, 0, 0);
            Cols[2] = simd::Set(0, 0, 1, 0);
            Cols[3] = simd::Set(0, 0, 0, 1);
        }

        ENGINE_INLINE Matrix(const float val) noexcept {
            for(std::size_t i = 0; i < 4; ++i) Cols[i] = simd::Set(val);
        }

        ENGINE_INLINE Matrix(const Vector& v1, const Vector& v2, const Vector& v3) noexcept {
            Cols[0] = v1.V;
            Cols[1] = v2.V;
            Cols[2] = v3.V;
            Cols[3] = simd::Set(0.f, 0.f, 0.f, 1.f);
        }

        ENGINE_INLINE Matrix(const Matrix& other) noexcept {
            for(std::size_t i = 0; i < 4; ++i) Cols[i] = other.Cols[i];
        }

        ENGINE_INLINE Matrix(Matrix&& other) noexcept {
            for(std::size_t i = 0; i < 4; ++i) Cols[i] = other.Cols[i];
        }

        Matrix& operator=(const Matrix&) = default;
        Matrix& operator=(Matrix&&) = default;

        ENGINE_INLINE Matrix& ENGINE_VECTORCALL operator+=(const Matrix& other) noexcept {
            for(std::size_t i = 0; i < 4; ++i) Cols[i] = simd::Add(Cols[i], other.Cols[i]);
            return *this;
        }

        ENGINE_INLINE Matrix ENGINE_VECTORCALL operator+(Matrix other) const noexcept { return other += *this; }

        ENGINE_INLINE Matrix& ENGINE_VECTORCALL operator-=(const Matrix& other) noexcept {
            for(std::size_t i = 0; i < 4; ++i) Cols[i] = simd::Sub(Cols[i], other.Cols[i]);
            return *this;
        }

        ENGINE_INLINE Matrix ENGINE_VECTORCALL operator-(Matrix other) const noexcept {
            Matrix temp(*this);
            return temp -= other;
        }

        ENGINE_INLINE Matrix& ENGINE_VECTORCALL operator*=(const float val) noexcept {
            const simd::Floats temp = simd::Set(val);

            for(std::size_t i = 0; i < 4; ++i) Cols[i] = simd::Mul(Cols[i], temp);

            return *this;
        }

        ENGINE_INLINE Matrix ENGINE_VECTORCALL operator*(const float val) const noexcept {
            Matrix result(*this);
            return result *= val;
        }

        ENGINE_INLINE Matrix& ENGINE_VECTORCALL operator*=(const Matrix& other) noexcept {
            Matrix original = *this;

            for(int i = 0; i < 4; ++i) {
                simd::Floats col = other.Cols[i];

                simd::Floats rX = simd::Swizzle<SIMD_MASK(0, 0, 0, 0)>(col);
                simd::Floats rY = simd::Swizzle<SIMD_MASK(1, 1, 1, 1)>(col);
                simd::Floats rZ = simd::Swizzle<SIMD_MASK(2, 2, 2, 2)>(col);
                simd::Floats rW = simd::Swizzle<SIMD_MASK(3, 3, 3, 3)>(col);

                simd::Floats res = simd::Mul(original.Cols[0], rX);
                res = simd::Add(res, simd::Mul(original.Cols[1], rY));
                res = simd::Add(res, simd::Mul(original.Cols[2], rZ));
                res = simd::Add(res, simd::Mul(original.Cols[3], rW));

                this->Cols[i] = res;
            }
            return *this;
        }

        ENGINE_INLINE Matrix ENGINE_VECTORCALL operator*(const Matrix& other) const noexcept {
            Matrix res{*this};
            return res *= other;
        }

        ENGINE_INLINE Matrix& ENGINE_VECTORCALL operator/=(const float val) noexcept {
            assert(val != 0.f && "Division by zero!");
            return *this *= 1 / val;
        }

        ENGINE_INLINE Matrix ENGINE_VECTORCALL operator/(const float val) const noexcept {
            Matrix result(*this);
            return result /= val;
        }

        ENGINE_INLINE bool ENGINE_VECTORCALL operator==(const Matrix& other) const noexcept {
            return simd::AllClose(Cols[0], other.Cols[0]) && simd::AllClose(Cols[1], other.Cols[1]) &&
                   simd::AllClose(Cols[2], other.Cols[2]) && simd::AllClose(Cols[3], other.Cols[3]);
        }

        ENGINE_INLINE bool ENGINE_VECTORCALL operator!=(const Matrix& other) const noexcept {
            return !(simd::AllClose(Cols[0], other.Cols[0]) && simd::AllClose(Cols[1], other.Cols[1]) &&
                     simd::AllClose(Cols[2], other.Cols[2]) && simd::AllClose(Cols[3], other.Cols[3]));
        }

        ENGINE_INLINE Matrix ENGINE_VECTORCALL Reciprocal() const noexcept {
            Matrix mat;

            for(std::size_t i = 0; i < 4; ++i) mat.Cols[i] = simd::Reciprocal(Cols[i]);

            return mat;
        }

        ENGINE_INLINE Matrix ENGINE_VECTORCALL Sqrt() const noexcept {
            Matrix mat;

            for(std::size_t i = 0; i < 4; ++i) mat.Cols[i] = simd::Sqrt(Cols[i]);

            return mat;
        }

        ENGINE_INLINE Matrix ENGINE_VECTORCALL Transpose() const noexcept {
            Matrix orig = *this;

            simd::Floats tmp0 = simd::UnpackLow(orig.Cols[0], orig.Cols[1]);
            simd::Floats tmp1 = simd::UnpackHigh(orig.Cols[0], orig.Cols[1]);
            simd::Floats tmp2 = simd::UnpackLow(orig.Cols[2], orig.Cols[3]);
            simd::Floats tmp3 = simd::UnpackHigh(orig.Cols[2], orig.Cols[3]);

            orig.Cols[0] = simd::PackLowHigh(tmp0, tmp2);
            orig.Cols[1] = simd::PackHighLow(tmp2, tmp0);
            orig.Cols[2] = simd::PackLowHigh(tmp1, tmp3);
            orig.Cols[3] = simd::PackHighLow(tmp3, tmp1);

            return orig;
        }

        ENGINE_INLINE Matrix ENGINE_VECTORCALL Inv() const noexcept {
            const float* m = Elements;
            float inv[16];

            inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] +
                     m[13] * m[6] * m[11] - m[13] * m[7] * m[10];

            inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] -
                     m[12] * m[6] * m[11] + m[12] * m[7] * m[10];

            inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] +
                     m[12] * m[5] * m[11] - m[12] * m[7] * m[9];

            inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] -
                      m[12] * m[5] * m[10] + m[12] * m[6] * m[9];

            inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] -
                     m[13] * m[2] * m[11] + m[13] * m[3] * m[10];

            inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] +
                     m[12] * m[2] * m[11] - m[12] * m[3] * m[10];

            inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] -
                     m[12] * m[1] * m[11] + m[12] * m[3] * m[9];

            inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] +
                      m[12] * m[1] * m[10] - m[12] * m[2] * m[9];

            inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] +
                     m[13] * m[2] * m[7] - m[13] * m[3] * m[6];

            inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] -
                     m[12] * m[2] * m[7] + m[12] * m[3] * m[6];

            inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] +
                      m[12] * m[1] * m[7] - m[12] * m[3] * m[5];

            inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] -
                      m[12] * m[1] * m[6] + m[12] * m[2] * m[5];

            inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] -
                     m[9] * m[2] * m[7] + m[9] * m[3] * m[6];

            inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] +
                     m[8] * m[2] * m[7] - m[8] * m[3] * m[6];

            inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] -
                      m[8] * m[1] * m[7] + m[8] * m[3] * m[5];

            inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] +
                      m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

            float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

            if(std::abs(det) < 1e-9f) return Matrix(0.f);

            det = 1.f / det;

            Matrix res;
            for(int i = 0; i < 16; i++) res.Elements[i] = inv[i] * det;
            return res;
        }
    };
}
