#pragma once

#include <cassert>
#include <cmath>

#include "SIMD.hpp"

namespace math {
    struct alignas(16) Vector {
        union {
            struct {
                float X;
                float Y;
                float Z;
                float W;
            };

            simd::Floats V;
        };

        ~Vector() noexcept = default;

        ENGINE_INLINE Vector() noexcept : V(simd::Reset()) {}
        ENGINE_INLINE Vector(const float val) noexcept : V(simd::Set(val)) {}
        ENGINE_INLINE Vector(const simd::Floats v) noexcept : V(v) {}
        ENGINE_INLINE Vector(const float x, const float y, const float z, const float w = 0.f) noexcept
            : V(simd::Set(x, y, z, w)) {}

        Vector(const Vector&) = default;
        Vector(Vector&&) = default;
        Vector& operator=(const Vector&) = default;
        Vector& operator=(Vector&&) = default;

        ENGINE_INLINE Vector& __vectorcall operator+=(const Vector& other) noexcept {
            V = simd::Add(V, other.V);
            return *this;
        }

        ENGINE_INLINE Vector __vectorcall operator+(Vector other) const noexcept { return other += *this; }

        ENGINE_INLINE Vector& __vectorcall operator-=(const Vector& other) noexcept {
            V = simd::Sub(V, other.V);
            return *this;
        }

        ENGINE_INLINE Vector __vectorcall operator-(Vector other) const noexcept {
            return Vector(simd::Sub(V, other.V));
        }

        ENGINE_INLINE Vector& __vectorcall operator*=(const float val) noexcept {
            V = simd::Mul(V, simd::Set(val));
            return *this;
        }

        ENGINE_INLINE Vector __vectorcall operator*(const float val) const noexcept {
            return Vector(simd::Mul(V, simd::Set(val)));
        }

        ENGINE_INLINE Vector& __vectorcall operator/=(const float val) noexcept {
            assert(val != 0.f && "Division by zero!");

            const float inv = 1.f / val;
            V = simd::Mul(V, simd::Set(inv));
            return *this;
        }

        ENGINE_INLINE Vector __vectorcall operator/(const float val) const noexcept {
            assert(val != 0.f && "Division by zero!");
            return Vector(simd::Mul(V, simd::Set(1.f / val)));
        }

        ENGINE_INLINE bool __vectorcall operator==(const Vector& other) const noexcept {
            return simd::AllClose(V, other.V);
        }
        ENGINE_INLINE bool __vectorcall operator!=(const Vector& other) const noexcept {
            return !simd::AllClose(V, other.V);
        }

        ENGINE_INLINE Vector __vectorcall Reciprocal() const noexcept { return Vector(simd::Reciprocal(V)); }
        ENGINE_INLINE Vector __vectorcall Sqrt() const noexcept { return Vector(simd::Sqrt(V)); }

        ENGINE_INLINE float __vectorcall Dot(const Vector& other) const noexcept {
            return simd::GetFirst(simd::HorizonSum<0x71>(V, other.V));
        }

        ENGINE_INLINE float __vectorcall Cross2D(const Vector& other) const noexcept {
            return X * other.Y - Y * other.X;
        }

        ENGINE_INLINE Vector __vectorcall Cross(const Vector& other) const noexcept {
            const std::uint8_t leftMask = SIMD_MASK(3, 0, 2, 1);
            const std::uint8_t rightMask = SIMD_MASK(3, 1, 0, 2);

            Vector left{simd::Mul(simd::Shuffle<leftMask>(V, V), simd::Shuffle<rightMask>(other.V, other.V))};
            Vector right{simd::Mul(simd::Shuffle<rightMask>(V, V), simd::Shuffle<leftMask>(other.V, other.V))};

            return left - right;
        }

        ENGINE_INLINE float Length() const noexcept { return std::sqrt(Dot(*this)); }

        ENGINE_INLINE Vector Norm() const noexcept {
            const float len = Length();
            if(len > 1e-6f) return *this / len;
            return Vector(0.f);
        }
    };
}
