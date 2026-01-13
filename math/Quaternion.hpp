#pragma once

#include <cassert>
#include <cmath>

#include "Matrix.hpp"
#include "SIMD.hpp"
#include "Vector.hpp"

namespace math {
    struct alignas(16) Quaternion {
        union {
            struct {
                float X;
                float Y;
                float Z;
                float W;
            };

            simd::Floats Q;
        };

        ~Quaternion() noexcept = default;

        ENGINE_INLINE Quaternion() noexcept : Q(simd::Set(0.f, 0.f, 0.f, 1.f)) {}
        ENGINE_INLINE Quaternion(const Vector& vec, const float q) noexcept : Q(simd::Set(vec.X, vec.Y, vec.Z, q)) {}
        ENGINE_INLINE Quaternion(const simd::Floats& v) noexcept : Q(v) {}
        ENGINE_INLINE Quaternion(const float x, const float y, const float z, const float w = 0.f) noexcept
            : Q(simd::Set(x, y, z, w)) {}

        Quaternion(const Quaternion&) = default;
        Quaternion(Quaternion&&) = default;
        Quaternion& operator=(const Quaternion&) = default;
        Quaternion& operator=(Quaternion&&) = default;

        ENGINE_INLINE Quaternion& __vectorcall operator+=(const Quaternion& other) noexcept {
            Q = simd::Add(Q, other.Q);
            return *this;
        }

        ENGINE_INLINE Quaternion __vectorcall operator+(Quaternion other) const noexcept { return other += *this; }

        ENGINE_INLINE Quaternion& __vectorcall operator-=(const Quaternion& other) noexcept {
            Q = simd::Sub(Q, other.Q);
            return *this;
        }

        ENGINE_INLINE Quaternion __vectorcall operator-(Quaternion other) const noexcept {
            Quaternion temp(*this);
            return temp -= other;
        }

        ENGINE_INLINE Quaternion& __vectorcall operator*=(const float val) noexcept {
            const simd::Floats temp = simd::Set(val);
            Q = simd::Mul(Q, temp);
            return *this;
        }

        ENGINE_INLINE Quaternion __vectorcall operator*(const float val) const noexcept {
            Quaternion result(*this);
            return result *= val;
        }

        ENGINE_INLINE Quaternion& __vectorcall operator/=(const float val) noexcept {
            assert(val != 0.f && "Division by zero!");
            return *this *= 1 / val;
        }

        ENGINE_INLINE Quaternion __vectorcall operator/(const float val) const noexcept {
            Quaternion result(*this);
            return result /= val;
        }

        ENGINE_INLINE bool __vectorcall operator==(const Quaternion& other) const noexcept {
            return simd::AllClose(Q, other.Q);
        }
        ENGINE_INLINE bool __vectorcall operator!=(const Quaternion& other) const noexcept {
            return !simd::AllClose(Q, other.Q);
        }

        ENGINE_INLINE Quaternion __vectorcall Reciprocal() const noexcept { return Quaternion(simd::Reciprocal(Q)); }
        ENGINE_INLINE Quaternion __vectorcall Sqrt() const noexcept { return Quaternion(simd::Sqrt(Q)); }

        ENGINE_INLINE float __vectorcall Dot(const Quaternion& other) const noexcept {
            return simd::GetFirst(simd::HorizonSum<0x71>(Q, other.Q));
        }

        ENGINE_INLINE Quaternion& __vectorcall operator*=(const Quaternion& other) noexcept {
            Q = simd::Set(W * other.X + X * other.W + Y * other.Z - Z * other.Y,
                          W * other.Y - X * other.Z + Y * other.W + Z * other.X,
                          W * other.Z + X * other.Y - Y * other.X + Z * other.W,
                          W * other.W - X * other.X - Y * other.Y - Z * other.Z);

            return *this;
        }

        ENGINE_INLINE Quaternion __vectorcall operator*(const Quaternion& other) const noexcept {
            Quaternion result(*this);
            return result *= other;
        }

        ENGINE_INLINE float __vectorcall Length() const noexcept {
            float dot = Dot(*this);
            return std::sqrt(dot);
        }

        ENGINE_INLINE Quaternion __vectorcall Norm() const noexcept { return *this / Length(); }

        ENGINE_INLINE Quaternion __vectorcall Conjugate() const noexcept {
            return Quaternion(simd::Mul(Q, simd::Set(-1.f, -1.f, -1.f, 1.f)));
        }

        ENGINE_INLINE Matrix __vectorcall ToMatrix() const noexcept {
            const float xx2 = X * X * 2.f;
            const float yy2 = Y * Y * 2.f;
            const float zz2 = Z * Z * 2.f;
            const float xy2 = X * Y * 2.f;
            const float xz2 = X * Z * 2.f;
            const float yz2 = Y * Z * 2.f;
            const float wx2 = W * X * 2.f;
            const float wy2 = W * Y * 2.f;
            const float wz2 = W * Z * 2.f;

            Matrix result;

            result.Cols[0] = simd::Set(1.f - yy2 - zz2, xy2 + wz2, xz2 - wy2, 0.f);
            result.Cols[1] = simd::Set(xy2 - wz2, 1.f - xx2 - zz2, yz2 + wx2, 0.f);
            result.Cols[2] = simd::Set(xz2 + wy2, yz2 - wx2, 1.f - xx2 - yy2, 0.f);
            result.Cols[3] = simd::Set(0.f, 0.f, 0.f, 1.f);

            return result;
        }

        ENGINE_INLINE Quaternion __vectorcall Slerp(const Quaternion& other, const float t) const noexcept {
            float cosHalfTheta = simd::GetFirst(simd::HorizonSum<0x71>(Q, other.Q));

            Quaternion target = other;
            if(cosHalfTheta < 0.f) {
                target.Q = simd::Mul(other.Q, simd::Set(-1.f));
                cosHalfTheta = -cosHalfTheta;
            }

            if(cosHalfTheta > 0.9995f) {
                return Quaternion(simd::Add(Q, simd::Mul(simd::Sub(target.Q, Q), simd::Set(t))));
            }

            float halfTheta = std::acos(cosHalfTheta);
            float sinHalfTheta = std::sqrt(1.f - cosHalfTheta * cosHalfTheta);

            float ratioA = std::sin((1 - t) * halfTheta) / sinHalfTheta;
            float ratioB = std::sin(t * halfTheta) / sinHalfTheta;

            return Quaternion(simd::Add(simd::Mul(Q, simd::Set(ratioA)), simd::Mul(target.Q, simd::Set(ratioB))));
        }
    };
}
