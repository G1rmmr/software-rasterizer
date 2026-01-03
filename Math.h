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

        bool operator==(const Vector3& other) const noexcept { return simd::AllClose(V, other.V); }
        bool operator!=(const Vector3& other) const noexcept { return !simd::AllClose(V, other.V); }

        Vector3 Reciprocal() const noexcept { return Vector3(simd::Reciprocal(V)); }
        Vector3 Sqrt() const noexcept { return Vector3(simd::Sqrt(V)); }

        float Dot(const Vector3& other) const noexcept {
            const Vector3 temp{simd::HorizonSum(V, other.V, 0x71)};
            return simd::GetFirst(temp);
        }

        float Cross2D(const Vector3& other) const noexcept {
            return X * other.Y - Y * other.X;
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

        Matrix4(const Vector3& v1, const Vector3& v2, const Vector3& v3) noexcept {
            Cols[0] = v1.V;
            Cols[1] = v2.V;
            Cols[2] = v3.V;
            Cols[3] = simd::Set(0.f, 0.f, 0.f, 1.f);
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
            Matrix4 original = *this;

            for(int i = 0; i < 4; ++i) {
                simd::Floats row1 = simd::Set(other[i][0]);
                simd::Floats row2 = simd::Set(other[i][1]);
                simd::Floats row3 = simd::Set(other[i][2]);
                simd::Floats row4 = simd::Set(other[i][3]);

                simd::Floats temp = simd::Mul(original.Cols[0], row1);
                temp = simd::Add(temp, simd::Mul(original.Cols[1], row2));
                temp = simd::Add(temp, simd::Mul(original.Cols[2], row3));
                temp = simd::Add(temp, simd::Mul(original.Cols[3], row4));

                this->Cols[i] = temp;
            }
            return *this;
        }

        Matrix4 operator*(const Matrix4& other) const noexcept {
            Matrix4 res{*this};
            res *= other;
            return res;
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

        bool operator==(const Matrix4& other) const noexcept {
            return simd::AllClose(Cols[0], other.Cols[0]) &&
                simd::AllClose(Cols[1], other.Cols[1]) &&
                simd::AllClose(Cols[2], other.Cols[2]) &&
                simd::AllClose(Cols[3], other.Cols[3]);
        }

        bool operator!=(const Matrix4& other) const noexcept {
            return !(simd::AllClose(Cols[0], other.Cols[0]) &&
                simd::AllClose(Cols[1], other.Cols[1]) &&
                simd::AllClose(Cols[2], other.Cols[2]) &&
                simd::AllClose(Cols[3], other.Cols[3]));
        }

        Matrix4 Reciprocal() const noexcept {
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

        Matrix4 Transpose() const noexcept {
            Matrix4 orig = *this;

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

        Matrix4 Inv() const noexcept {
           	simd::Floats a = simd::PackLowHigh(Cols[0], Cols[1]);
           	simd::Floats b = simd::PackHighLow(Cols[0], Cols[1]);
           	simd::Floats c = simd::PackLowHigh(Cols[2], Cols[3]);
           	simd::Floats d = simd::PackHighLow(Cols[2], Cols[3]);

            const std::uint8_t leftMask = _MM_SHUFFLE(2, 0, 2, 0);
            const std::uint8_t rightMask = _MM_SHUFFLE(3, 1, 3, 1);

           	simd::Floats detSub = simd::Sub(
          		simd::Mul(
                    simd::Shuffle(Cols[0], Cols[2], leftMask),
                    simd::Shuffle(Cols[1], Cols[3], rightMask)),
          		simd::Mul(
                    simd::Shuffle(Cols[0], Cols[2], rightMask),
                    simd::Shuffle(Cols[1], Cols[3], leftMask))
           	);

           	simd::Floats detA = simd::Swizzle(detSub, _MM_SHUFFLE(0, 0, 0, 0));
           	simd::Floats detB = simd::Swizzle(detSub, _MM_SHUFFLE(1, 1, 1, 1));
           	simd::Floats detC = simd::Swizzle(detSub, _MM_SHUFFLE(2, 2, 2, 2));
           	simd::Floats detD = simd::Swizzle(detSub, _MM_SHUFFLE(3, 3, 3, 3));

           	simd::Floats dc = adjMul2(d, c);
            simd::Floats ab = adjMul2(a, b);

            simd::Floats x = simd::Sub(simd::Mul(detD, a), mul2(b, dc));
            simd::Floats w = simd::Sub(simd::Mul(detA, d), mul2(c, ab));
           	simd::Floats detM = simd::Mul(detA, detD);

           	simd::Floats y = simd::Sub(simd::Mul(detB, c), mulAdj2(d, ab));
           	simd::Floats z = simd::Sub(simd::Mul(detC, b), mulAdj2(a, dc));
           	detM = simd::Add(detM, simd::Mul(detB, detC));

           	simd::Floats tr = simd::Mul(ab, simd::Swizzle(dc, _MM_SHUFFLE(0, 2, 1, 3)));
            tr = simd::HorizonSum(tr, tr, 0xFF);

           	detM = simd::Sub(detM, tr);

           	const simd::Floats adjSign = simd::Set(1.f, -1.f, -1.f, 1.f);
            const simd::Floats rec = simd::Reciprocal(detM);
           	const simd::Floats recDetM = simd::Mul(adjSign, detM);

           	x = simd::Mul(x, recDetM);
           	y = simd::Mul(y, recDetM);
           	z = simd::Mul(z, recDetM);
           	w = simd::Mul(w, recDetM);

           	Matrix4 result;

           	result.Cols[0] = simd::Shuffle(x, y, rightMask);
           	result.Cols[1] = simd::Shuffle(x, y, leftMask);
           	result.Cols[2] = simd::Shuffle(z, w, rightMask);
           	result.Cols[3] = simd::Shuffle(z, w, leftMask);

           	return result;
        }

    private:
        simd::Floats mul2(const simd::Floats& v1, const simd::Floats& v2) const noexcept {
           	return simd::Add(
                simd::Mul(v1, simd::Swizzle(v2, _MM_SHUFFLE(0, 3, 0, 3))),
                simd::Mul(
                    simd::Swizzle(v1, _MM_SHUFFLE(1, 0, 3, 2)),
                    simd::Swizzle(v2, _MM_SHUFFLE(2, 1, 2, 1)))
            );
        }

        simd::Floats adjMul2(const simd::Floats& v1, const simd::Floats& v2) const noexcept {
            return simd::Sub(
                simd::Mul(simd::Swizzle(v1, _MM_SHUFFLE(3, 3, 0, 0)), v2),
                simd::Mul(
                    simd::Swizzle(v1, _MM_SHUFFLE(1, 1, 2, 2)),
                    simd::Swizzle(v2, _MM_SHUFFLE(2, 3, 0, 1)))
            );
        }

        simd::Floats mulAdj2(const simd::Floats& v1, const simd::Floats& v2) const noexcept {
            return simd::Sub(
                simd::Mul(v1, simd::Swizzle(v2, _MM_SHUFFLE(3, 0, 3, 0))),
                simd::Mul(
                    simd::Swizzle(v1, _MM_SHUFFLE(1, 0, 3, 2)),
                    simd::Swizzle(v2, _MM_SHUFFLE(2, 1, 2, 1)))
            );
        }
    };

    struct alignas(16) Quaternion{
        union{
            struct{
                float X;
                float Y;
                float Z;
                float W;
            };

            simd::Floats Q;
        };

        ~Quaternion() noexcept = default;

        Quaternion() noexcept : Q(simd::Set(0.f, 0.f, 0.f, 1.f)) {}

        Quaternion(const Vector3& vec, const float q) noexcept
            : Q(simd::Set(vec.X, vec.Y, vec.Z, q)) {}

        Quaternion(const simd::Floats& v) noexcept : Q(v) {}

        Quaternion(const float x, const float y, const float z, const float w = 0.f) noexcept
            : Q(simd::Set(x, y, z, w)){}

        Quaternion(const Quaternion& other) noexcept : Q(other.Q){}

        Quaternion(Quaternion&& other) noexcept : Q(other.Q) {}

        Quaternion& operator=(const Quaternion& other) noexcept {
            if(this != &other) Q = other.Q;
            return *this;
        }

        Quaternion& operator=(Quaternion&& other) noexcept{
            if(this != &other) Q = other.Q;
            return *this;
        }

        Quaternion& operator+=(const Quaternion& other) noexcept {
            Q = simd::Add(Q, other.Q);
            return *this;
        }

        Quaternion operator+(Quaternion other) const noexcept {
            other += *this;
            return other;
        }

        Quaternion& operator-=(const Quaternion& other) noexcept {
            Q = simd::Sub(Q, other.Q);
            return *this;
        }

        Quaternion operator-(Quaternion other) const noexcept {
            Quaternion temp(*this);
            temp -= other;
            return temp;
        }

        Quaternion& operator*=(const float val) noexcept {
            const simd::Floats temp = simd::Set(val);
            Q = simd::Mul(Q, temp);
            return *this;
        }

        Quaternion operator*(const float val) const noexcept {
            Quaternion result(*this);
            result *= val;
            return result;
        }

        Quaternion& operator/=(const float val) noexcept {
            assert(val != 0.f && "Division by zero!");
            *this *= 1 / val;
            return *this;
        }

        Quaternion operator/(const float val) const noexcept {
            Quaternion result(*this);
            result /= val;
            return result;
        }

        bool operator==(const Quaternion& other) const noexcept { return simd::AllClose(Q, other.Q); }
        bool operator!=(const Quaternion& other) const noexcept { return !simd::AllClose(Q, other.Q); }

        Quaternion Reciprocal() const noexcept { return Quaternion(simd::Reciprocal(Q)); }
        Quaternion Sqrt() const noexcept { return Quaternion(simd::Sqrt(Q)); }

        float Dot(const Quaternion& other) const noexcept {
            return simd::GetFirst(simd::HorizonSum(Q, other.Q, 0x71));
        }

        Quaternion& operator*=(const Quaternion& other) noexcept {
            Q = simd::Set(
                W * other.X + X * other.W + Y * other.Z - Z * other.Y,
                W * other.Y - X * other.Z + Y * other.W + Z * other.X,
                W * other.Z + X * other.Y - Y * other.X + Z * other.W,
                W * other.W - X * other.X - Y * other.Y - Z * other.Z);

            return *this;
        }

        Quaternion operator*(const Quaternion& other) const noexcept {
            Quaternion result(*this);
            result *= other;
            return result;
        }

        float Length() const noexcept {
            float dot = Dot(*this);
            return std::sqrt(dot);
        }

        Quaternion Norm() const noexcept { return *this / Length(); }

        Quaternion Conjugate() const noexcept {
            return Quaternion(simd::Mul(Q, simd::Set(-1.f, -1.f, -1.f, 1.f)));
        }

        Matrix4 ToMatrix() const noexcept {
            const float xx2 = X * X * 2.f;
            const float yy2 = Y * Y * 2.f;
            const float zz2 = Z * Z * 2.f;
            const float xy2 = X * Y * 2.f;
            const float xz2 = X * Z * 2.f;
            const float yz2 = Y * Z * 2.f;
            const float wx2 = W * X * 2.f;
            const float wy2 = W * Y * 2.f;
            const float wz2 = W * Z * 2.f;

            Matrix4 result;

            result.Cols[0] = simd::Set(1.f - yy2 - zz2, xy2 + wz2, xz2 - wy2, 0.f);
            result.Cols[1] = simd::Set(xy2 - wz2, 1.f - xx2 - zz2, yz2 + wx2, 0.f);
            result.Cols[2] = simd::Set(xz2 + wy2, yz2 - wx2, 1.f - xx2 - yy2, 0.f);
            result.Cols[3] = simd::Set(0.f, 0.f, 0.f, 1.f);

            return result;
        }

        Quaternion Slerp(const Quaternion& other, const float t) const noexcept {
            float cosHalfTheta = simd::GetFirst(simd::HorizonSum(Q, other.Q, 0x71));

            Quaternion target = other;
            if(cosHalfTheta < 0.f){
                target.Q = simd::Mul(other.Q, simd::Set(-1.f));
                cosHalfTheta = -cosHalfTheta;
            }

            if(cosHalfTheta > 0.9995f){
                return Quaternion(simd::Add(Q, simd::Mul(simd::Sub(target.Q, Q), simd::Set(t))));
            }

            float halfTheta = std::acos(cosHalfTheta);
            float sinHalfTheta = std::sqrt(1.f - cosHalfTheta * cosHalfTheta);

            float ratioA = std::sin((1 - t) * halfTheta) / sinHalfTheta;
            float ratioB = std::sin(t * halfTheta) / sinHalfTheta;

            return Quaternion(simd::Add(simd::Mul(Q, simd::Set(ratioA)), simd::Mul(target.Q, simd::Set(ratioB))));
        }
    };

    inline Vector3 operator*(const Vector3& vec, const Matrix4& mat) noexcept {
        simd::Floats res = simd::Mul(mat.Cols[0], simd::Set(vec.X));
        res = simd::Add(res, simd::Mul(mat.Cols[1], simd::Set(vec.Y)));
        res = simd::Add(res, simd::Mul(mat.Cols[2], simd::Set(vec.Z)));
        res = simd::Add(res, simd::Mul(mat.Cols[3], simd::Set(1.0f)));

        float w = simd::GetFirst(simd::Shuffle(res, res, _MM_SHUFFLE(3, 3, 3, 3)));

        if (std::abs(w) > 1e-6f && std::abs(w - 1.0f) > 1e-6f) {
            const simd::Floats rec = simd::Reciprocal(w);
           	res = simd::Mul(res, rec);
        }

        return Vector3(res);
    }

    inline Quaternion FromAxisAngle(const Vector3& axis, const float radian) const noexcept {
        Quaternion result(
            axis.Norm() * std::sin(radian * 0.5f),
            std::cos(radian * 0.5f));

        return result;
    }

    inline Vector3 GetBarycentric(
        const Vector3& pos, const Vector3& a, const Vector3& b, const Vector3& c) noexcept {
        float area = (b - a).Cross2D(c - a);

        if (std::abs(area) < 1e-6f) return Vector3(-1.f, -1.f, -1.f, 0.f);

        float wA = (b - pos).Cross2D(c - p) / area;
        float wB = (c - p).Cross2D(a - p) / area;
        float wC = 1.f - wA - wB;

        return Vector3(wA, wB, wC, 0.f);
    }
}
