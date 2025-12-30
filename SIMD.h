#pragma once

#include <cstdint>
#include <xmmintrin.h>

namespace simd{

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define ENGINE_SIMD_SSE

    typedef __m128 Floats;

#elif defined(__arm64__) || defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define ENGINE_SIMD_NEON

    typedef float32x4_t SimdFloat;

#else
#error "지원하지 않는 아키텍처입니다."
#endif

// Arithmetics
    inline Floats Add(const Floats& lhs, const Floats& rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_add_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
#endif
    }

    inline Floats Sub(const Floats& lhs, const Floats& rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_sub_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
#endif
    }

    inline Floats Mul(const Floats& lhs, const Floats& rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_mul_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
#endif
    }

    inline Floats Div(const Floats& lhs, const Floats& rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_div_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
#endif
    }

    inline Floats Reciprocal(const Floats& val) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_rcp_ps(val);
#elif defined(ENGINE_SIMD_NEON)
#endif
    }

    inline Floats Sqrt(const Floats& val) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_sqrt_ps(val);
#elif defined(ENGINE_SIMD_NEON)
#endif
    }

    inline Floats HorizonSum(const Floats& lhs, const Floats& rhs, const std::uint8_t mask) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_dp_ps(lhs, rhs, mask);
#elif defined(ENGINE_SIMD_NEON)
#endif
    }

    inline float GetFirst(const Floats& val){
#ifdef ENGINE_SIMD_SSE
        return _mm_cvtss_f32(val);
#elif defined(ENGINE_SIMD_NEON)
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
#endif
    }

// Logicals
    inline Floats Reset() noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_setzero_ps();
#elif defined(ENGINE_SIMD_NEON)
#endif
    }

    inline Floats Set(const float val) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_set1_ps(val);
#elif defined(ENGINE_SIMD_NEON)
#endif
    }

    inline Floats Set(const float x, const float y, const float z, const float w) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_set_ps(w, z, y, x);
#elif defined(ENGINE_SIMD_NEON)
#endif
    }

    inline Floats Shuffle(const Floats& lhs, const Floats& rhs, const ::std::uint8_t mask) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_shuffle_ps(lhs, rhs, mask);
#elif defined(ENGINE_SIMD_NEON)
#endif
    }

    inline Floats Swizzle(const Floats& v, const ::std::uint8_t mask) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v), mask));
#elif defined(ENGINE_SIMD_NEON)
#endif
    }

    inline Floats UnpackHigh(const Floats& lhs, const Floats& rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_unpackhi_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
#endif
    }

    inline Floats UnpackLow(const Floats& lhs, const Floats& rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_unpacklo_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
#endif
    }

    inline Floats PackLowHigh(const Floats& lhs, const Floats& rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_movelh_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
#endif
    }

    inline Floats PackHighLow(const Floats& lhs, const Floats& rhs) noexcept {
#ifdef ENGINE_SIMD_SSE
        return _mm_movehl_ps(lhs, rhs);
#elif defined(ENGINE_SIMD_NEON)
#endif
    }
}
