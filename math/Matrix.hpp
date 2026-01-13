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

        ENGINE_INLINE Matrix& __vectorcall operator+=(const Matrix& other) noexcept {
            for(std::size_t i = 0; i < 4; ++i) Cols[i] = simd::Add(Cols[i], other.Cols[i]);
            return *this;
        }

        ENGINE_INLINE Matrix __vectorcall operator+(Matrix other) const noexcept { return other += *this; }

        ENGINE_INLINE Matrix& __vectorcall operator-=(const Matrix& other) noexcept {
            for(std::size_t i = 0; i < 4; ++i) Cols[i] = simd::Sub(Cols[i], other.Cols[i]);
            return *this;
        }

        ENGINE_INLINE Matrix __vectorcall operator-(Matrix other) const noexcept {
            Matrix temp(*this);
            return temp -= other;
        }

        ENGINE_INLINE Matrix& __vectorcall operator*=(const float val) noexcept {
            const simd::Floats temp = simd::Set(val);

            for(std::size_t i = 0; i < 4; ++i) Cols[i] = simd::Mul(Cols[i], temp);

            return *this;
        }

        ENGINE_INLINE Matrix __vectorcall operator*(const float val) const noexcept {
            Matrix result(*this);
            return result *= val;
        }

        ENGINE_INLINE Matrix& __vectorcall operator*=(const Matrix& other) noexcept {
            Matrix original = *this;

            for(int i = 0; i < 4; ++i) {
                simd::Floats row1 = simd::Set(other.Elements[i * 4 + 0]);
                simd::Floats row2 = simd::Set(other.Elements[i * 4 + 1]);
                simd::Floats row3 = simd::Set(other.Elements[i * 4 + 2]);
                simd::Floats row4 = simd::Set(other.Elements[i * 4 + 3]);

                simd::Floats temp = simd::Mul(original.Cols[0], row1);
                temp = simd::Add(temp, simd::Mul(original.Cols[1], row2));
                temp = simd::Add(temp, simd::Mul(original.Cols[2], row3));
                temp = simd::Add(temp, simd::Mul(original.Cols[3], row4));

                this->Cols[i] = temp;
            }
            return *this;
        }

        ENGINE_INLINE Matrix __vectorcall operator*(const Matrix& other) const noexcept {
            Matrix res{*this};
            return res *= other;
        }

        ENGINE_INLINE Matrix& __vectorcall operator/=(const float val) noexcept {
            assert(val != 0.f && "Division by zero!");
            return *this *= 1 / val;
        }

        ENGINE_INLINE Matrix __vectorcall operator/(const float val) const noexcept {
            Matrix result(*this);
            return result /= val;
        }

        ENGINE_INLINE bool __vectorcall operator==(const Matrix& other) const noexcept {
            return simd::AllClose(Cols[0], other.Cols[0]) && simd::AllClose(Cols[1], other.Cols[1]) &&
                   simd::AllClose(Cols[2], other.Cols[2]) && simd::AllClose(Cols[3], other.Cols[3]);
        }

        ENGINE_INLINE bool __vectorcall operator!=(const Matrix& other) const noexcept {
            return !(simd::AllClose(Cols[0], other.Cols[0]) && simd::AllClose(Cols[1], other.Cols[1]) &&
                     simd::AllClose(Cols[2], other.Cols[2]) && simd::AllClose(Cols[3], other.Cols[3]));
        }

        ENGINE_INLINE Matrix __vectorcall Reciprocal() const noexcept {
            Matrix mat;

            for(std::size_t i = 0; i < 4; ++i) mat.Cols[i] = simd::Reciprocal(Cols[i]);

            return mat;
        }

        ENGINE_INLINE Matrix __vectorcall Sqrt() const noexcept {
            Matrix mat;

            for(std::size_t i = 0; i < 4; ++i) mat.Cols[i] = simd::Sqrt(Cols[i]);

            return mat;
        }

        ENGINE_INLINE Matrix __vectorcall Transpose() const noexcept {
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

        ENGINE_INLINE Matrix __vectorcall Inv() const noexcept {
            simd::Floats a = simd::PackLowHigh(Cols[0], Cols[1]);
            simd::Floats b = simd::PackHighLow(Cols[0], Cols[1]);
            simd::Floats c = simd::PackLowHigh(Cols[2], Cols[3]);
            simd::Floats d = simd::PackHighLow(Cols[2], Cols[3]);

            const std::uint8_t leftMask = SIMD_MASK(2, 0, 2, 0);
            const std::uint8_t rightMask = SIMD_MASK(3, 1, 3, 1);

            simd::Floats detSub = simd::Sub(
                simd::Mul(simd::Shuffle<leftMask>(Cols[0], Cols[2]), simd::Shuffle<rightMask>(Cols[1], Cols[3])),
                simd::Mul(simd::Shuffle<rightMask>(Cols[0], Cols[2]), simd::Shuffle<leftMask>(Cols[1], Cols[3])));

            simd::Floats detA = simd::Swizzle<SIMD_MASK(0, 0, 0, 0)>(detSub);
            simd::Floats detB = simd::Swizzle<SIMD_MASK(1, 1, 1, 1)>(detSub);
            simd::Floats detC = simd::Swizzle<SIMD_MASK(2, 2, 2, 2)>(detSub);
            simd::Floats detD = simd::Swizzle<SIMD_MASK(3, 3, 3, 3)>(detSub);

            simd::Floats dc = adjMul2(d, c);
            simd::Floats ab = adjMul2(a, b);

            simd::Floats x = simd::Sub(simd::Mul(detD, a), mul2(b, dc));
            simd::Floats w = simd::Sub(simd::Mul(detA, d), mul2(c, ab));
            simd::Floats detM = simd::Mul(detA, detD);

            simd::Floats y = simd::Sub(simd::Mul(detB, c), mulAdj2(d, ab));
            simd::Floats z = simd::Sub(simd::Mul(detC, b), mulAdj2(a, dc));
            detM = simd::Add(detM, simd::Mul(detB, detC));

            simd::Floats tr = simd::Mul(ab, simd::Swizzle<SIMD_MASK(0, 2, 1, 3)>(dc));
            tr = simd::HorizonSum<0xFF>(tr, tr);

            detM = simd::Sub(detM, tr);

            const simd::Floats adjSign = simd::Set(1.f, -1.f, -1.f, 1.f);
            const simd::Floats recDetM = simd::Mul(adjSign, detM);

            x = simd::Mul(x, recDetM);
            y = simd::Mul(y, recDetM);
            z = simd::Mul(z, recDetM);
            w = simd::Mul(w, recDetM);

            Matrix result;

            result.Cols[0] = simd::Shuffle<rightMask>(x, y);
            result.Cols[1] = simd::Shuffle<leftMask>(x, y);
            result.Cols[2] = simd::Shuffle<rightMask>(z, w);
            result.Cols[3] = simd::Shuffle<leftMask>(z, w);

            return result;
        }

    private:
        ENGINE_INLINE simd::Floats __vectorcall mul2(const simd::Floats v1, const simd::Floats v2) const noexcept {
            return simd::Add(
                simd::Mul(v1, simd::Swizzle<SIMD_MASK(0, 3, 0, 3)>(v2)),
                simd::Mul(simd::Swizzle<SIMD_MASK(1, 0, 3, 2)>(v1), simd::Swizzle<SIMD_MASK(2, 1, 2, 1)>(v2)));
        }

        ENGINE_INLINE simd::Floats __vectorcall adjMul2(const simd::Floats v1, const simd::Floats v2) const noexcept {
            return simd::Sub(
                simd::Mul(simd::Swizzle<SIMD_MASK(3, 3, 0, 0)>(v1), v2),
                simd::Mul(simd::Swizzle<SIMD_MASK(1, 1, 2, 2)>(v1), simd::Swizzle<SIMD_MASK(2, 3, 0, 1)>(v2)));
        }

        ENGINE_INLINE simd::Floats __vectorcall mulAdj2(const simd::Floats v1, const simd::Floats v2) const noexcept {
            return simd::Sub(
                simd::Mul(v1, simd::Swizzle<SIMD_MASK(3, 0, 3, 0)>(v2)),
                simd::Mul(simd::Swizzle<SIMD_MASK(1, 0, 3, 2)>(v1), simd::Swizzle<SIMD_MASK(2, 1, 2, 1)>(v2)));
        }
    };
}
