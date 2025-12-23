#pragma once

#include <memory>
#include <immintrin.h>

namespace math{
    struct alignas(16) Vector3{
        union{
            struct{
                float X;
                float Y;
                float Z;
                float W;
            };

            __m128 
        };

        Vector3() = default;
        Vector3(const float x, const float y, const float z) : X(x), Y(y), Z(z){}

        Vector3(const Vector3& other) : X(other.X), Y(other.Y), Z(other.Z){}

        Vector3(Vector3&& other) noexcept : X(0.f), Y(0.f), Z(0.f){ *this = std::move(other); }

        Vector3& operator=(const Vector3& other){
            X = other.X;
            Y = other.Y;
            Z = other.Z;
        }

        Vector3& operator=(Vector3&& other) noexcept{
            if(this != &other){
                X = 0.f;
                Y = 0.f;
                Z = 0.f;
                
                *this = std::move(other);
            }
        }

        Vector3 operator+(const Vector3& other){ return {X + other.X, Y + other.Y, Z + other.Z}; }
        Vector3& operator+=(const Vector3& other){
            X += other.X;
            Y += other.Y;
            Z += other.Z;
            return *this;
        }

        Vector3 operator-(const Vector3& other){ return {X - other.X, Y - other.Y, Z - other.Z}; }
        Vector3 operator*(const float val){ return {val * X, val * Y, val * Z}; }
        Vector3 operator/(const float val){ return {X / val, Y / val, Z / val}; }

        float operator*(const Vector3& other){ return X * other.X + Y * other.Y + Z * other.Z; }
    };
}