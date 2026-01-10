#pragma once

#include <cstdint>

namespace simd {

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#include <xmmintrin.h>

#define SIMD_MASK(w, z, y, x) _MM_SHUFFLE(w, z, y, x)
#define ENGINE_SIMD_SSE
    typedef __m128 Floats;

    typedef __m128i Int32x4;
    typedef __m128i Int16x8;
    typedef __m128i Int8x16;

    typedef __m128i Uint32x4;
    typedef __m128i Uint16x8;
    typedef __m128i Uint16x4;
    typedef __m128i Uint8x16;
    typedef __m128i Uint8x8;

#elif defined(__arm64__) || defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>

#define SIMD_MASK(w, z, y, x) (((w) << 6) | ((z) << 4) | ((y) << 2) | (x))
#define ENGINE_SIMD_NEON
    typedef float32x4_t Floats;

    typedef int32x4_t Int32x4;
    typedef int16x8_t Int16x8;
    typedef int8x16_t Int8x16;

    typedef uint32x4_t Uint32x4;
    typedef uint16x8_t Uint16x8;
    typedef uint16x4_t Uint16x4;
    typedef uint8x16_t Uint8x16;
    typedef uint8x8_t Uint8x8;

#else
#error "UNDEFINED ARCHITECTURE"
#endif

    // Arithmetics
    template <typename T, typename U> using is_same = std::is_same<T, U>;

    template <typename To, typename From> inline To Cast(From v) noexcept {
        if constexpr(std::is_same<To, From>()) {
            return v;
        }

#if defined(ENGINE_SIMD_SSE)
        if constexpr(std::is_same<To, Floats>()) {
            return _mm_castsi128_ps(v);
        }
        else if constexpr(std::is_same<From, Floats>()) {
            return _mm_castps_si128(v);
        }
        else {
            return v;
        }

#elif defined(ENGINE_SIMD_NEON)
        if constexpr(is_same<To, Floats>::value) {
            if constexpr(is_same<From, Int32s>::value) return vreinterpretq_f32_s32(v);
            if constexpr(is_same<From, Uint32s>::value) return vreinterpretq_f32_u32(v);
            if constexpr(is_same<From, Int16s>::value) return vreinterpretq_f32_s16(v);
            if constexpr(is_same<From, Uint16s>::value) return vreinterpretq_f32_u16(v);
            if constexpr(is_same<From, Int8s>::value) return vreinterpretq_f32_s8(v);
            if constexpr(is_same<From, Uint8s>::value) return vreinterpretq_f32_u8(v);
        }
        else if constexpr(is_same<To, Int32s>::value) {
            if constexpr(is_same<From, Floats>::value) return vreinterpretq_s32_f32(v);
            if constexpr(is_same<From, Uint32s>::value) return vreinterpretq_s32_u32(v);
            return vreinterpretq_s32_u32(vreinterpretq_u32_s32(v));
        }
        else if constexpr(is_same<To, Uint8s>::value) {
            if constexpr(is_same<From, Floats>::value) return vreinterpretq_u8_f32(v);
            if constexpr(is_same<From, Int32s>::value) return vreinterpretq_u8_s32(v);
            if constexpr(is_same<From, Uint32s>::value) return vreinterpretq_u8_u32(v);
        }
        return *(To*)&v;
#endif
    }

    inline Floats Add(const Floats& lhs, const Floats& rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_add_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
        return vaddq_f32(lhs, rhs);
#endif
    }

    inline Floats Sub(const Floats& lhs, const Floats& rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_sub_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
        return vsubq_f32(lhs, rhs);
#endif
    }

    inline Floats Mul(const Floats& lhs, const Floats& rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_mul_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
        return vmulq_f32(lhs, rhs);
#endif
    }

    inline Floats Div(const Floats& lhs, const Floats& rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_div_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
        return vdivq_f32(lhs, rhs);
#endif
    }

    inline Floats Reciprocal(const Floats& val) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_rcp_ps(val);
#elif defined(ENGINE_SIMD_NEON)
        Floats rec = vrecpeq_f32(val);
        return vmulq_f32(vrecpsq_f32(val, rec), rec);
#endif
    }

    inline Floats Sqrt(const Floats& val) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_sqrt_ps(val);
#elif defined(ENGINE_SIMD_NEON)
        return vsqrtq_f32(val);
#endif
    }

    template <std::uint8_t MASK> inline Floats HorizonSum(const Floats& lhs, const Floats& rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_dp_ps(lhs, rhs, MASK);
#elif defined(ENGINE_SIMD_NEON)
        Floats m = vmulq_f32(lhs, rhs);
        const float sum = vaddvq_f32(m);
        return vdupq_n_f32(sum);
#endif
    }

    inline float GetFirst(const Floats& val) {
#ifdef ENGINE_SIMD_SSE
        return _mm_cvtss_f32(val);
#elif defined(ENGINE_SIMD_NEON)
        return vgetq_lane_f32(val, 0);
#endif
    }

    inline bool AllClose(const Floats& a, const Floats& b, float epsilon = 1e-5f) noexcept {
#ifdef ENGINE_SIMD_SSE
        Floats diff = _mm_sub_ps(a, b);
        static const Floats absMask = _mm_set1_ps(-0.0f);
        Floats absDiff = _mm_andnot_ps(absMask, diff);
        Floats eps = _mm_set1_ps(epsilon);
        Floats cmp = _mm_cmplt_ps(absDiff, eps);
        return (_mm_movemask_ps(cmp) == 0xF);
#elif defined(ENGINE_SIMD_NEON)
        Floats diff = vabdq_f32(a, b);
        Uint32s cmp = vcltq_f32(diff, vdupq_n_f32(epsilon));
        return vminvq_u32(cmp) > 0;
#endif
    }

    inline Floats Clamp(const Floats& v, const Floats& minVal, const Floats& maxVal) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_min_ps(_mm_max_ps(v, minVal), maxVal);
#elif defined(ENGINE_SIMD_NEON)
        return vminq_f32(vmaxq_f32(v, minVal), maxVal);
#endif
    }

    inline std::uint32_t PackRGBA(const Floats& val) noexcept {
#ifdef ENGINE_SIMD_SSE
        Int32x4 intVal = _mm_cvtps_epi32(val);
        Int16x8 pack16 = _mm_packus_epi32(intVal, intVal);
        Int8x16 pack8 = _mm_packus_epi16(pack16, pack16);
        return static_cast<std::uint32_t>(_mm_cvtsi128_si32(pack8));

#elif defined(ENGINE_SIMD_NEON)
        Int32s intVal = vcvtq_s32_f32(val);
        Uint16x4 pack16 = vqmovun_s32(intVal);
        Uint8x8 pack8 = vqmovn_u16(vcombine_u16(pack16, pack16));
        return vget_lane_u32(vreinterpret_u32_u8(pack8), 0);
#endif
    }

    // Logicals
    inline Floats Reset() noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_setzero_ps();
#elif defined(ENGINE_SIMD_NEON)
        return vdupq_n_f32(0.f);
#endif
    }

    inline Floats Set(const float val) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_set1_ps(val);
#elif defined(ENGINE_SIMD_NEON)
        return vdupq_n_f32(val);
#endif
    }

    inline Floats Set(const float x, const float y, const float z, const float w) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_set_ps(w, z, y, x);
#elif defined(ENGINE_SIMD_NEON)
        const float temp[4] = {x, y, z, w};
        return vld1q_f32(temp);
#endif
    }

    inline Floats And(const Floats& lhs, const Floats& rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_and_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
        return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(lhs), vreinterpretq_u32_f32(rhs)));
#endif
    }

    inline Floats GreaterEqual(const Floats& lhs, const Floats& rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_cmpge_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
        return vreinterpretq_f32_u32(vcgeq_f32(lhs, rhs));
#endif
    }

    inline Floats LessEqual(const Floats& lhs, const Floats& rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_cmple_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
        return vreinterpretq_f32_u32(vcleq_f32(lhs, rhs));
#endif
    }

    inline int GetMask(const Floats& val) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_movemask_ps(val);
#elif defined(ENGINE_SIMD_NEON)
        static const std::uint8_t __attribute__((aligned(16))) mask[16] = {
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};

        Uint8x16 maskVal = vld1q_u8(mask);
        Uint8x8 res = vshrn_n_u16(vaddq_u16(vreinterpretq_u16_u8(vandq_u8(vreinterpretq_u8_f32(val), maskVal)), 0), 4);

        return vget_lane_u32(vreinterpret_u32_u8(res), 0);
#endif
    }

    template <std::uint8_t MASK> inline Floats Swizzle(const Floats& v) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v), MASK));
#else
        return __builtin_shufflevector(v, v, (MASK & 0x03), ((MASK >> 2) & 0x03), ((MASK >> 4) & 0x03),
                                       ((MASK >> 6) & 0x03));
#endif
    }

    template <std::uint8_t MASK> inline Floats Shuffle(const Floats& lhs, const Floats& rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_shuffle_ps(lhs, rhs, MASK);
#else
        return __builtin_shufflevector(lhs, rhs, (MASK & 0x03), ((MASK >> 2) & 0x03), ((MASK >> 4) & 0x03),
                                       ((MASK >> 6) & 0x03));
#endif
    }

    inline Floats UnpackHigh(const Floats& lhs, const Floats& rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_unpackhi_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
        return vzip2q_f32(lhs, rhs);
#endif
    }

    inline Floats UnpackLow(const Floats& lhs, const Floats& rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_unpacklo_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
        return vzip1q_f32(lhs, rhs);
#endif
    }

    inline Floats PackLowHigh(const Floats& lhs, const Floats& rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_movelh_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
        return vcombine_f32(vget_low_f32(lhs), vget_low_f32(rhs));
#endif
    }

    inline Floats PackHighLow(const Floats& lhs, const Floats& rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_movehl_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
        return vcombine_f32(vget_high_f32(rhs), vget_high_f32(lhs));
#endif
    }
}