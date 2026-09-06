#include <immintrin.h>

#include "matmul.h"

#define PREFETCH_DIST 2    
#define FLOATS_PER_LINE 16

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;

        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            const float* b = B + static_cast<long>(j) * ldb;

            for (int p = 0; p < K; ++p) {
                acc += a[p] * b[p];
                _mm_prefetch(reinterpret_cast<const char*>(&a[p + PREFETCH_DIST*FLOATS_PER_LINE]), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(&b[p + PREFETCH_DIST*FLOATS_PER_LINE]), _MM_HINT_T0);
            }
            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}