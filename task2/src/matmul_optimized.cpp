// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

#include <immintrin.h>
#include <algorithm>

#include "matmul.h"

static inline float hsum(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(vlow, vhigh);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    const int BM = 96;
    const int BN = 96;
    const int PREFETCH_DIST = 64;
    const int K8 = K - (K % 8);

    for (int bi = 0; bi < M; bi += BM) {
        const int i_end = std::min(bi + BM, M);
        const int i4_end = bi + ((i_end - bi) / 4) * 4;

        for (int bj = 0; bj < N; bj += BN) {
            const int j_end = std::min(bj + BN, N);
            const int j3_end = bj + ((j_end - bj) / 3) * 3;

            for (int i = bi; i < i4_end; i += 4) {
                const float* a0 = A + (long)(i+0) * lda;
                const float* a1 = A + (long)(i+1) * lda;
                const float* a2 = A + (long)(i+2) * lda;
                const float* a3 = A + (long)(i+3) * lda;

                int j = bj;
                for (; j < j3_end; j += 3) {
                    const float* b0 = B + (long)(j+0) * ldb;
                    const float* b1 = B + (long)(j+1) * ldb;
                    const float* b2 = B + (long)(j+2) * ldb;

                    __m256 s00 = _mm256_setzero_ps(), s01 = _mm256_setzero_ps(), s02 = _mm256_setzero_ps();
                    __m256 s10 = _mm256_setzero_ps(), s11 = _mm256_setzero_ps(), s12 = _mm256_setzero_ps();
                    __m256 s20 = _mm256_setzero_ps(), s21 = _mm256_setzero_ps(), s22 = _mm256_setzero_ps();
                    __m256 s30 = _mm256_setzero_ps(), s31 = _mm256_setzero_ps(), s32 = _mm256_setzero_ps();

                    for (int p = 0; p < K8; p += 8) {
                        _mm_prefetch((const char*)(b0+p+PREFETCH_DIST), _MM_HINT_T0);
                        _mm_prefetch((const char*)(b1+p+PREFETCH_DIST), _MM_HINT_T0);
                        _mm_prefetch((const char*)(b2+p+PREFETCH_DIST), _MM_HINT_T0);

                        __m256 vb0 = _mm256_loadu_ps(b0+p);
                        __m256 vb1 = _mm256_loadu_ps(b1+p);
                        __m256 vb2 = _mm256_loadu_ps(b2+p);

                        __m256 va;
                        va = _mm256_loadu_ps(a0+p);
                        s00 = _mm256_fmadd_ps(va, vb0, s00);
                        s01 = _mm256_fmadd_ps(va, vb1, s01);
                        s02 = _mm256_fmadd_ps(va, vb2, s02);

                        va = _mm256_loadu_ps(a1+p);
                        s10 = _mm256_fmadd_ps(va, vb0, s10);
                        s11 = _mm256_fmadd_ps(va, vb1, s11);
                        s12 = _mm256_fmadd_ps(va, vb2, s12);

                        va = _mm256_loadu_ps(a2+p);
                        s20 = _mm256_fmadd_ps(va, vb0, s20);
                        s21 = _mm256_fmadd_ps(va, vb1, s21);
                        s22 = _mm256_fmadd_ps(va, vb2, s22);

                        va = _mm256_loadu_ps(a3+p);
                        s30 = _mm256_fmadd_ps(va, vb0, s30);
                        s31 = _mm256_fmadd_ps(va, vb1, s31);
                        s32 = _mm256_fmadd_ps(va, vb2, s32);
                    }

                    float r00 = hsum(s00), r01 = hsum(s01), r02 = hsum(s02);
                    float r10 = hsum(s10), r11 = hsum(s11), r12 = hsum(s12);
                    float r20 = hsum(s20), r21 = hsum(s21), r22 = hsum(s22);
                    float r30 = hsum(s30), r31 = hsum(s31), r32 = hsum(s32);

                    for (int p = K8; p < K; ++p) {
                        float bp0 = b0[p], bp1 = b1[p], bp2 = b2[p];
                        r00 += a0[p]*bp0; r01 += a0[p]*bp1; r02 += a0[p]*bp2;
                        r10 += a1[p]*bp0; r11 += a1[p]*bp1; r12 += a1[p]*bp2;
                        r20 += a2[p]*bp0; r21 += a2[p]*bp1; r22 += a2[p]*bp2;
                        r30 += a3[p]*bp0; r31 += a3[p]*bp1; r32 += a3[p]*bp2;
                    }

                    C[(long)(i+0)*ldc+j+0] = r00; C[(long)(i+0)*ldc+j+1] = r01; C[(long)(i+0)*ldc+j+2] = r02;
                    C[(long)(i+1)*ldc+j+0] = r10; C[(long)(i+1)*ldc+j+1] = r11; C[(long)(i+1)*ldc+j+2] = r12;
                    C[(long)(i+2)*ldc+j+0] = r20; C[(long)(i+2)*ldc+j+1] = r21; C[(long)(i+2)*ldc+j+2] = r22;
                    C[(long)(i+3)*ldc+j+0] = r30; C[(long)(i+3)*ldc+j+1] = r31; C[(long)(i+3)*ldc+j+2] = r32;
                }

                for (; j < j_end; ++j) {
                    const float* b = B + (long)j * ldb;
                    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
                    __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
                    for (int p = 0; p < K8; p += 8) {
                        __m256 vb = _mm256_loadu_ps(b+p);
                        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0+p), vb, s0);
                        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1+p), vb, s1);
                        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2+p), vb, s2);
                        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a3+p), vb, s3);
                    }
                    float r0 = hsum(s0), r1 = hsum(s1), r2 = hsum(s2), r3 = hsum(s3);
                    for (int p = K8; p < K; ++p) {
                        float bp = b[p];
                        r0 += a0[p]*bp; r1 += a1[p]*bp; r2 += a2[p]*bp; r3 += a3[p]*bp;
                    }
                    C[(long)(i+0)*ldc+j] = r0; C[(long)(i+1)*ldc+j] = r1;
                    C[(long)(i+2)*ldc+j] = r2; C[(long)(i+3)*ldc+j] = r3;
                }
            }

            for (int i = i4_end; i < i_end; ++i) {
                const float* a = A + (long)i * lda;
                for (int j = bj; j < j_end; ++j) {
                    const float* b = B + (long)j * ldb;
                    __m256 s = _mm256_setzero_ps();
                    for (int p = 0; p < K8; p += 8) {
                        s = _mm256_fmadd_ps(_mm256_loadu_ps(a+p), _mm256_loadu_ps(b+p), s);
                    }
                    float res = hsum(s);
                    for (int p = K8; p < K; ++p) {
                        res += a[p] * b[p];
                    }
                    C[(long)i * ldc + j] = res;
                }
            }
        }
    }
}