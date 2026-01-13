#pragma once

#include <bit>
#include <cstdint>
#include <type_traits>

//__vector-call

#if defined(_MSC_VER)
#define ENGINE_INLINE [[nodiscard]] [[msvc::forceENGINE_INLINE]] inline
#define ENGINE_VECTORCALL __vectorcall
#else
#define ENGINE_INLINE [[nodiscard]] __attribute__((always_ENGINE_INLINE)) inline
#define ENGINE_VECTORCALL
#endif

namespace simd {

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#include <xmmintrin.h>

#define SIMD_MASK(w, z, y, x) _MM_SHUFFLE(w, z, y, x)
#define ENGINE_SIMD_SSE

    typedef __m256 Doubles;
    typedef __m256i Longs;

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

    template <typename To, typename From> ENGINE_INLINE To Cast(From v) noexcept {
        if constexpr(std::is_same<To, From>()) return v;

#if defined(ENGINE_SIMD_SSE)
        if constexpr(std::is_same<To, Floats>()) return _mm_castsi128_ps(v);
        if constexpr(std::is_same<From, Floats>()) return _mm_castps_si128(v);
        return v;

#elif defined(ENGINE_SIMD_NEON)
        if constexpr(is_same<To, Floats>::value) return vreinterpretq_f32_u32((Uint32x4)v);
        if constexpr(is_same<From, Floats>::value) return (To)vreinterpretq_u32_f32(v);
        return (To)v;
#endif
    }

    ENGINE_INLINE Floats ENGINE_VECTORCALL Add(const Floats lhs, const Floats rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_add_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
        return vaddq_f32(lhs, rhs);
#endif
    }

    ENGINE_INLINE Floats ENGINE_VECTORCALL Sub(const Floats lhs, const Floats rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_sub_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
        return vsubq_f32(lhs, rhs);
#endif
    }

    ENGINE_INLINE Floats ENGINE_VECTORCALL Mul(const Floats lhs, const Floats rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_mul_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
        return vmulq_f32(lhs, rhs);
#endif
    }

    ENGINE_INLINE Floats ENGINE_VECTORCALL Div(const Floats lhs, const Floats rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_div_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
        return vdivq_f32(lhs, rhs);
#endif
    }

    ENGINE_INLINE Floats ENGINE_VECTORCALL Reciprocal(const Floats val) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_rcp_ps(val);
#elif defined(ENGINE_SIMD_NEON)
        Floats rec = vrecpeq_f32(val);
        return vmulq_f32(vrecpsq_f32(val, rec), rec);
#endif
    }

    ENGINE_INLINE Floats ENGINE_VECTORCALL Sqrt(const Floats val) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_sqrt_ps(val);
#elif defined(ENGINE_SIMD_NEON)
        return vsqrtq_f32(val);
#endif
    }

    template <std::uint8_t MASK>
    ENGINE_INLINE Floats ENGINE_VECTORCALL HorizonSum(const Floats lhs, const Floats rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_dp_ps(lhs, rhs, MASK);
#elif defined(ENGINE_SIMD_NEON)
        Floats m = vmulq_f32(lhs, rhs);
        static const std::uint32_t maskArr[4] = {
            (MASK & 0x10) ? 0xFFFFFFFF : 0, // X (bit 4)
            (MASK & 0x20) ? 0xFFFFFFFF : 0, // Y (bit 5)
            (MASK & 0x40) ? 0xFFFFFFFF : 0, // Z (bit 6)
            (MASK & 0x80) ? 0xFFFFFFFF : 0  // W (bit 7)
        };

        Uint32x4 bitMask = vld1q_u32(maskArr);
        m = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(m), bitMask));

        const float sum = vaddvq_f32(m);
        return vdupq_n_f32(sum);
#endif
    }

    ENGINE_INLINE float ENGINE_VECTORCALL GetFirst(const Floats val) {
#ifdef ENGINE_SIMD_SSE
        return _mm_cvtss_f32(val);
#elif defined(ENGINE_SIMD_NEON)
        return vgetq_lane_f32(val, 0);
#endif
    }

    ENGINE_INLINE bool ENGINE_VECTORCALL AllClose(const Floats a, const Floats b, float epsilon = 1e-5f) noexcept {
#ifdef ENGINE_SIMD_SSE
        Floats diff = _mm_sub_ps(a, b);

        const Floats absMask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
        Floats absDiff = _mm_and_ps(absMask, diff);

        Floats eps = _mm_set1_ps(epsilon);
        Floats cmp = _mm_cmplt_ps(absDiff, eps);

        return (_mm_movemask_ps(cmp) == 0xF);
#elif defined(ENGINE_SIMD_NEON)
        Floats diff = vabdq_f32(a, b);
        Uint32x4 cmp = vcltq_f32(diff, vdupq_n_f32(epsilon));
        return vminvq_u32(cmp) > 0;
#endif
    }

    ENGINE_INLINE Floats ENGINE_VECTORCALL Clamp(const Floats v, const Floats minVal, const Floats maxVal) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_min_ps(_mm_max_ps(v, minVal), maxVal);
#elif defined(ENGINE_SIMD_NEON)
        return vminq_f32(vmaxq_f32(v, minVal), maxVal);
#endif
    }

    ENGINE_INLINE std::uint32_t ENGINE_VECTORCALL PackRGBA(const Floats val) noexcept {
#ifdef ENGINE_SIMD_SSE
        Int32x4 intVal = _mm_cvtps_epi32(val);
        Int16x8 pack16 = _mm_packus_epi32(intVal, intVal);
        Int8x16 pack8 = _mm_packus_epi16(pack16, pack16);
        return static_cast<std::uint32_t>(_mm_cvtsi128_si32(pack8));

#elif defined(ENGINE_SIMD_NEON)
        Int32x4 intVal = vcvtq_s32_f32(val);
        Uint16x4 pack16 = vqmovun_s32(intVal);
        Uint8x8 pack8 = vqmovn_u16(vcombine_u16(pack16, pack16));
        return vget_lane_u32(vreinterpret_u32_u8(pack8), 0);
#endif
    }

    // Logicals
    ENGINE_INLINE Floats Reset() noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_setzero_ps();
#elif defined(ENGINE_SIMD_NEON)
        return vdupq_n_f32(0.f);
#endif
    }

    ENGINE_INLINE Floats ENGINE_VECTORCALL Set(const float val) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_set1_ps(val);
#elif defined(ENGINE_SIMD_NEON)
        return vdupq_n_f32(val);
#endif
    }

    ENGINE_INLINE Floats ENGINE_VECTORCALL Set(const float x, const float y, const float z, const float w) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_set_ps(w, z, y, x);
#elif defined(ENGINE_SIMD_NEON)
        const float temp[4] = {x, y, z, w};
        return vld1q_f32(temp);
#endif
    }

    ENGINE_INLINE Floats ENGINE_VECTORCALL And(const Floats lhs, const Floats rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_and_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
        return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(lhs), vreinterpretq_u32_f32(rhs)));
#endif
    }

    ENGINE_INLINE Floats ENGINE_VECTORCALL GreaterEqual(const Floats lhs, const Floats rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_cmpge_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
        return vreinterpretq_f32_u32(vcgeq_f32(lhs, rhs));
#endif
    }

    ENGINE_INLINE Floats ENGINE_VECTORCALL LessEqual(const Floats lhs, const Floats rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_cmple_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
        return vreinterpretq_f32_u32(vcleq_f32(lhs, rhs));
#endif
    }

    ENGINE_INLINE int ENGINE_VECTORCALL GetMask(const Floats val) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_movemask_ps(val);
#elif defined(ENGINE_SIMD_NEON)
        Uint32x4 signMask = vshrq_n_u32(vreinterpretq_u32_f32(val), 31);

        const std::uint32_t powers[4] = {1, 2, 4, 8};
        Uint32x4 weighted = vmulq_u32(signMask, vld1q_u32(powers));

        return vaddvq_u32(weighted);
#endif
    }

    template <std::uint8_t MASK> ENGINE_INLINE Floats ENGINE_VECTORCALL Swizzle(const Floats v) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v), MASK));
#else
        return __builtin_shufflevector(v, v, (MASK & 0x03), ((MASK >> 2) & 0x03), ((MASK >> 4) & 0x03),
                                       ((MASK >> 6) & 0x03));
#endif
    }

    template <std::uint8_t MASK>
    ENGINE_INLINE Floats ENGINE_VECTORCALL Shuffle(const Floats lhs, const Floats rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_shuffle_ps(lhs, rhs, MASK);
#else
        return __builtin_shufflevector(lhs, rhs, (MASK & 0x03), ((MASK >> 2) & 0x03), ((MASK >> 4) & 0x03),
                                       ((MASK >> 6) & 0x03));
#endif
    }

    ENGINE_INLINE Floats ENGINE_VECTORCALL UnpackHigh(const Floats lhs, const Floats rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_unpackhi_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
        return vzip2q_f32(lhs, rhs);
#endif
    }

    ENGINE_INLINE Floats ENGINE_VECTORCALL UnpackLow(const Floats lhs, const Floats rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_unpacklo_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
        return vzip1q_f32(lhs, rhs);
#endif
    }

    ENGINE_INLINE Floats ENGINE_VECTORCALL PackLowHigh(const Floats lhs, const Floats rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_movelh_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
        return vcombine_f32(vget_low_f32(lhs), vget_low_f32(rhs));
#endif
    }

    ENGINE_INLINE Floats ENGINE_VECTORCALL PackHighLow(const Floats lhs, const Floats rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_movehl_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
        return vcombine_f32(vget_high_f32(rhs), vget_high_f32(lhs));
#endif
    }
}
