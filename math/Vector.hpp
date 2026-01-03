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

        Vector() noexcept : V(simd::Reset()) {}

        Vector(const float val) noexcept : V(simd::Set(val)) {}

        Vector(const simd::Floats& v) noexcept : V(v) {}

        Vector(const float x, const float y, const float z, const float w = 0.f) noexcept : V(simd::Set(x, y, z, w)) {}

        Vector(const Vector& other) noexcept : V(other.V) {}

        Vector(Vector&& other) noexcept : V(other.V) {}

        Vector& operator=(const Vector& other) noexcept {
            if(this != &other) V = other.V;
            return *this;
        }

        Vector& operator=(Vector&& other) noexcept {
            if(this != &other) V = other.V;
            return *this;
        }

        Vector& operator+=(const Vector& other) noexcept {
            V = simd::Add(V, other.V);
            return *this;
        }

        Vector operator+(Vector other) const noexcept {
            other += *this;
            return other;
        }

        Vector& operator-=(const Vector& other) noexcept {
            V = simd::Sub(V, other.V);
            return *this;
        }

        Vector operator-(Vector other) const noexcept {
            Vector temp(*this);
            temp -= other;
            return temp;
        }

        Vector& operator*=(const float val) noexcept {
            const simd::Floats temp = simd::Set(val);
            V = simd::Mul(V, temp);
            return *this;
        }

        Vector operator*(const float val) const noexcept {
            Vector result(*this);
            result *= val;
            return result;
        }

        Vector& operator/=(const float val) noexcept {
            assert(val != 0.f && "Division by zero!");
            *this *= 1 / val;
            return *this;
        }

        Vector operator/(const float val) const noexcept {
            Vector result(*this);
            result /= val;
            return result;
        }

        bool operator==(const Vector& other) const noexcept { return simd::AllClose(V, other.V); }
        bool operator!=(const Vector& other) const noexcept { return !simd::AllClose(V, other.V); }

        Vector Reciprocal() const noexcept { return Vector(simd::Reciprocal(V)); }
        Vector Sqrt() const noexcept { return Vector(simd::Sqrt(V)); }

        float Dot(const Vector& other) const noexcept {
            const Vector temp{simd::HorizonSum<0x71>(V, other.V)};
            return simd::GetFirst(temp.V);
        }

        float Cross2D(const Vector& other) const noexcept { return X * other.Y - Y * other.X; }

        Vector Cross(const Vector& other) const noexcept {
            const std::uint8_t leftMask = _MM_SHUFFLE(3, 0, 2, 1);
            const std::uint8_t rightMask = _MM_SHUFFLE(3, 1, 0, 2);

            Vector left{simd::Mul(simd::Shuffle<leftMask>(V, V), simd::Shuffle<rightMask>(other.V, other.V))};
            Vector right{simd::Mul(simd::Shuffle<rightMask>(V, V), simd::Shuffle<leftMask>(other.V, other.V))};

            return left - right;
        }

        float Length() const noexcept {
            float dot = Dot(*this);
            return std::sqrt(dot);
        }

        Vector Norm() const noexcept { return *this / Length(); }
    };
}