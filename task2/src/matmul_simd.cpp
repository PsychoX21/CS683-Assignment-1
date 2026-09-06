// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

static inline float hsum(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(vlow, vhigh);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    const int K8 = K - (K % 8);

    int i = 0;
    for (; i + 8 <= M; i += 8) {
        const float* a0 = A + (long)(i + 0) * lda;
        const float* a1 = A + (long)(i + 1) * lda;
        const float* a2 = A + (long)(i + 2) * lda;
        const float* a3 = A + (long)(i + 3) * lda;
        const float* a4 = A + (long)(i + 4) * lda;
        const float* a5 = A + (long)(i + 5) * lda;
        const float* a6 = A + (long)(i + 6) * lda;
        const float* a7 = A + (long)(i + 7) * lda;

        for (int j = 0; j < N; ++j) {
            const float* b = B + (long)j * ldb;

            __m256 s0 = _mm256_setzero_ps();
            __m256 s1 = _mm256_setzero_ps();
            __m256 s2 = _mm256_setzero_ps();
            __m256 s3 = _mm256_setzero_ps();
            __m256 s4 = _mm256_setzero_ps();
            __m256 s5 = _mm256_setzero_ps();
            __m256 s6 = _mm256_setzero_ps();
            __m256 s7 = _mm256_setzero_ps();

            for (int p = 0; p < K8; p += 8) {
                __m256 vb = _mm256_loadu_ps(b + p);
                // _mm_prefetch(reinterpret_cast<const char*>(b + p + 48), _MM_HINT_T0);
                s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + p), vb, s0);
                s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + p), vb, s1);
                s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + p), vb, s2);
                s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a3 + p), vb, s3);
                s4 = _mm256_fmadd_ps(_mm256_loadu_ps(a4 + p), vb, s4);
                s5 = _mm256_fmadd_ps(_mm256_loadu_ps(a5 + p), vb, s5);
                s6 = _mm256_fmadd_ps(_mm256_loadu_ps(a6 + p), vb, s6);
                s7 = _mm256_fmadd_ps(_mm256_loadu_ps(a7 + p), vb, s7);
            }

            float r0 = hsum(s0), r1 = hsum(s1), r2 = hsum(s2), r3 = hsum(s3);
            float r4 = hsum(s4), r5 = hsum(s5), r6 = hsum(s6), r7 = hsum(s7);

            for (int p = K8; p < K; ++p) {
                float bp = b[p];
                r0 += a0[p] * bp; r1 += a1[p] * bp;
                r2 += a2[p] * bp; r3 += a3[p] * bp;
                r4 += a4[p] * bp; r5 += a5[p] * bp;
                r6 += a6[p] * bp; r7 += a7[p] * bp;
            }

            C[(long)(i + 0) * ldc + j] = r0;
            C[(long)(i + 1) * ldc + j] = r1;
            C[(long)(i + 2) * ldc + j] = r2;
            C[(long)(i + 3) * ldc + j] = r3;
            C[(long)(i + 4) * ldc + j] = r4;
            C[(long)(i + 5) * ldc + j] = r5;
            C[(long)(i + 6) * ldc + j] = r6;
            C[(long)(i + 7) * ldc + j] = r7;
        }
    }

    for (; i < M; ++i) {
        const float* a = A + (long)i * lda;
        for (int j = 0; j < N; ++j) {
            const float* b = B + (long)j * ldb;
            float sum = 0.0f;
            for (int p = 0; p < K; ++p) {
                sum += a[p] * b[p];
            }
            C[(long)i * ldc + j] = sum;
        }
    }
}