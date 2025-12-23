#pragma once

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

    inline Floats Add(const Floats& lhs, const Floats& rhs){
#ifdef ENGINE_SIMD_SSE
        return _mm_add_ps(lhs, rhs);

#elif defined(ENGINE_SIMD_NEON)
        return vaddq_f32(lhs, rhs);

#endif
    }

    inline Floats Sub(const Floats& lhs, const Floats& rhs){
#ifdef ENGINE_SIMD_SSE
        return _mm_sub_ps(lhs, rhs);

#elif defined(ENGINE_SIMD_NEON)
        return vsubq_f32(a, b);

#endif
    }

    inline Floats Mul(const Floats& lhs, const Floats& rhs){
#ifdef ENGINE_SIMD_SSE
        return _mm_mul_ps(lhs, rhs);

#elif defined(ENGINE_SIMD_NEON)
        return vmulq_f32(a, b);

#endif
    }

    inline Floats Div(const Floats& lhs, const Floats& rhs){
#ifdef ENGINE_SIMD_SSE
        return _mm_div_ps(lhs, rhs);

#elif defined(ENGINE_SIMD_NEON)
        return vdivq_f32(a, b);

#endif
    }

    inline Floats Zero() {
#ifdef ENGINE_SIMD_SSE
        return _mm_setzero_ps();

#elif defined(ENGINE_SIMD_NEON)
        return vdupq_n_f32(0.0f);

#endif
    }

    inline Floats Load(const float x, const float y, const float z, const float w){
#ifdef ENGINE_SIMD_SSE
        return _mm_set_ps(w, z, y, x);

#elif defined(ENGINE_SIMD_NEON)
        float temp[4] = {x, y, z, w};

        return vld1q_f32(temp);
#endif
    }

    inline float* Store(const Floats& data){
#ifdef ENGINE_SIMD_SSE
        return _mm_set_ps(w, z, y, x);

#elif defined(ENGINE_SIMD_NEON)
        float temp[4] = {x, y, z, w};

        return vld1q_f32(temp);
#endif
    }
}