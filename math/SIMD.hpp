#pragma once

#include <cstdint>

namespace simd {

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <immintrin.h>
    #include <xmmintrin.h>

    #define SIMD_MASK(w, z, y, x) _MM_SHUFFLE(w, z, y, x)
    #define ENGINE_SIMD_SSE

    typedef __m128 Floats;

#elif defined(__arm64__) || defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>

    #define SIMD_MASK(w, z, y, x) (((w) << 6) | ((z) << 4) | ((y) << 2) | (x))
    #define ENGINE_SIMD_NEON

    typedef float32x4_t Floats;

#else
    #error "UNDEFINED ARCHITECTURE"
#endif

    // Arithmetics
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
        float32x4_t rec = vrecpeq_f32(val);
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

    template <std::uint8_t MASK>
    inline Floats HorizonSum(const Floats& lhs, const Floats& rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_dp_ps(lhs, rhs, MASK);
#elif defined(ENGINE_SIMD_NEON)
        float32x4_t m = vmulq_f32(lhs, rhs);
        float sum = vaddvq_f32(m);
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
        float32x4_t diff = vabdq_f32(a, b);
        uint32x4_t cmp = vcltq_f32(diff, vdupq_n_f32(epsilon));
        return vminvq_u32(cmp) > 0;
#endif
    }

    // Logicals
    inline Floats Reset() noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_setzero_ps();
#elif defined(ENGINE_SIMD_NEON)
        return vdupq_n_f32(0.0f);
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
        float temp[4] = {x, y, z, w};
        return vld1q_f32(temp);
#endif
    }

    template <std::uint8_t MASK>
    inline Floats Swizzle(const Floats& v) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v), MASK));
#else
        return __builtin_shufflevector(v, v,
            (MASK & 0x03),
            ((MASK >> 2) & 0x03),
            ((MASK >> 4) & 0x03),
            ((MASK >> 6) & 0x03)
        );
#endif
    }

    template <std::uint8_t MASK>
    inline Floats Shuffle(const Floats& lhs, const Floats& rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_shuffle_ps(lhs, rhs, MASK);
#else
        return __builtin_shufflevector(lhs, rhs,
            (MASK & 0x03),
            ((MASK >> 2) & 0x03),
            ((MASK >> 4) & 0x03),
            ((MASK >> 6) & 0x03)
        );
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
        return  _mm_movehl_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
        return vcombine_f32(vget_high_f32(rhs), vget_high_f32(lhs));
#endif
    }
}
