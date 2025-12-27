#pragma once

#include <cmath>
#include <cassert>

#include "SIMD.h"

namespace math{
    struct alignas(16) Vector3{
        union{
            struct{
                float X;
                float Y;
                float Z;
                float W;
            };

            simd::Floats V;
        };

        ~Vector3() noexcept = default;

        Vector3() noexcept : V(simd::Reset()) {}

        Vector3(const float val) noexcept : V(simd::Set(val)) {}

        Vector3(const simd::Floats& v) noexcept : V(v) {}

        Vector3(const float x, const float y, const float z, const float w = 0.f) noexcept 
            : V(simd::Set(x, y, z, w)){}

        Vector3(const Vector3& other) noexcept : V(other.V){}

        Vector3(Vector3&& other) noexcept : V(other.V) {}

        Vector3& operator=(const Vector3& other) noexcept {
            if(this != &other) V = other.V;
            return *this;
        }

        Vector3& operator=(Vector3&& other) noexcept{
            if(this != &other) V = other.V;
            return *this;
        }

        Vector3& operator+=(const Vector3& other) noexcept {
            V = simd::Add(V, other.V);
            return *this;
        }

        Vector3 operator+(Vector3 other) const noexcept {
            other += *this;
            return other;
        }

        Vector3& operator-=(const Vector3& other) noexcept {
            V = simd::Sub(V, other.V);
            return *this;
        }
        
        Vector3 operator-(Vector3 other) const noexcept {
            Vector3 temp(*this);
            temp -= other;
            return temp;
        }

        Vector3& operator*=(const float val) noexcept {
            const simd::Floats temp = simd::Set(val);
            V = simd::Mul(V, temp);
            return *this;
        }

        Vector3 operator*(const float val) const noexcept {
            Vector3 result(*this);
            result *= val;
            return result;
        }

        Vector3& operator/=(const float val) noexcept {
            assert(val != 0.f && "Division by zero!");
            *this *= 1 / val;
            return *this;
        }

        Vector3 operator/(const float val) const noexcept {
            Vector3 result(*this);
            result /= val;
            return result;
        }

        Vector3 Inv() const noexcept { return Vector3(simd::Inv(V)); }

        Vector3 Sqrt() const noexcept { return Vector3(simd::Sqrt(V)); }

        float Dot(const Vector3& other) const noexcept {
            const Vector3 temp{simd::HorizonSum(V, other.V, 0x71)};
            return temp.X;
        }

        bool operator==(const Vector3& other) noexcept {
            const Vector3 diff = *this - other;
            float dist = diff.Dot(diff); 
    
            const float epsilon = 1e-5f;
            return dist < (epsilon * epsilon);
        }

        bool operator!=(const Vector3& other) noexcept {
            const Vector3 diff = *this - other;
            float dist = diff.Dot(diff); 
    
            const float epsilon = 1e-5f;
            return dist >= (epsilon * epsilon);
        }
        
        Vector3 Cross(const Vector3& other) const noexcept {
            const std::uint8_t leftMask = _MM_SHUFFLE(3, 0, 2, 1);
            const std::uint8_t rightMask = _MM_SHUFFLE(3, 1, 0, 2);

            Vector3 left{simd::Mul(
                simd::Shuffle(V, V, leftMask),
                simd::Shuffle(other.V, other.V, rightMask)
            )};

            Vector3 right{simd::Mul(
                simd::Shuffle(V, V, rightMask),
                simd::Shuffle(other.V, other.V, leftMask)
            )};

            return left - right;
        }

        float Length() const noexcept {
            float dot = Dot(*this);
            return std::sqrt(dot);
        }

        Vector3 Norm() const noexcept { return *this / Length(); }
    };

    struct alignas(16) Matrix4{
        union{
            simd::Floats Cols[4];
            float Elements[16];
        };

        inline float* operator[](int index) noexcept {
            return reinterpret_cast<float*>(&Cols[index]);
        }

        inline const float* operator[](int index) const noexcept {
            return reinterpret_cast<const float*>(&Cols[index]);
        }

        ~Matrix4() noexcept = default;

        Matrix4() noexcept {
            Cols[0] = simd::Set(1, 0, 0, 0);
            Cols[1] = simd::Set(0, 1, 0, 0);
            Cols[2] = simd::Set(0, 0, 1, 0);
            Cols[3] = simd::Set(0, 0, 0, 1);
        }

        Matrix4(const float val) noexcept {
            for(std::size_t i = 0; i < 4; ++i)
                Cols[i] = simd::Set(val);
        }
        
        Matrix4(const Matrix4& other) noexcept {
            for(std::size_t i = 0; i < 4; ++i)
                Cols[i] = other.Cols[i];
        }

        Matrix4(Matrix4&& other) noexcept {
            for(std::size_t i = 0; i < 4; ++i)
                Cols[i] = other.Cols[i];
        }

        Matrix4& operator=(const Matrix4& other) noexcept {
            if(this != &other){
                for(std::size_t i = 0; i < 4; ++i)
                    Cols[i] = other.Cols[i];
            }
            return *this;
        }

        Matrix4& operator=(Matrix4&& other) noexcept{
            if(this != &other){
                for(std::size_t i = 0; i < 4; ++i)
                    Cols[i] = other.Cols[i];
            }
            return *this;
        }

        Matrix4& operator+=(const Matrix4& other) noexcept {
            for(std::size_t i = 0; i < 4; ++i)
                Cols[i] = simd::Add(Cols[i], other.Cols[i]);

            return *this;
        }

        Matrix4 operator+(Matrix4 other) const noexcept {
            other += *this;
            return other;
        }

        Matrix4& operator-=(const Matrix4& other) noexcept {
            for(std::size_t i = 0; i < 4; ++i)
                Cols[i] = simd::Sub(Cols[i], other.Cols[i]);

            return *this;
        }
        
        Matrix4 operator-(Matrix4 other) const noexcept {
            Matrix4 temp(*this);
            temp -= other;
            return temp;
        }

        Matrix4& operator*=(const float val) noexcept {
            const simd::Floats temp = simd::Set(val);

            for(std::size_t i = 0; i < 4; ++i)
                Cols[i] = simd::Mul(Cols[i], temp);
            
            return *this;
        }

        Matrix4 operator*(const float val) const noexcept {
            Matrix4 result(*this);
            result *= val;
            return result;
        }

        Matrix4& operator*=(const Matrix4& other) noexcept {
        }

        Matrix4 operator*(const Matrix4 val) const noexcept {
        }

        Matrix4& operator/=(const float val) noexcept {
            assert(val != 0.f && "Division by zero!");
            *this *= 1 / val;
            return *this;
        }

        Matrix4 operator/(const float val) const noexcept {
            Matrix4 result(*this);
            result /= val;
            return result;
        }

        Matrix4 Inv() const noexcept {
            Matrix4 mat;

            for(std::size_t i = 0; i < 4; ++i)
                mat.Cols[i] = simd::Inv(Cols[i]);

            return mat; 
        }

        Matrix4 Sqrt() const noexcept {
            Matrix4 mat;

            for(std::size_t i = 0; i < 4; ++i)
                mat.Cols[i] = simd::Sqrt(Cols[i]);

            return mat;
        }

        bool operator==(const Matrix4& other) noexcept {
        }

        bool operator!=(const Vector3& other) noexcept {
        }
    };
}