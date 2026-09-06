// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

#define PREFETCH_DIST 4    // in terms of j-iterations ahead
#define FLOATS_PER_LINE 16

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;

        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            const float* b = B + static_cast<long>(j) * ldb;

            const int pf_j = j + PREFETCH_DIST;
            if (pf_j < N) {
                const float* pb = B + static_cast<long>(pf_j) * ldb;

                // Manually unrolled prefetch issuance -- no loop overhead.
                // Covers up to 8 cache lines (128 floats) of the future row.
                // Each line individually bounds-checked against K.
                if (0  * FLOATS_PER_LINE < K) _mm_prefetch(reinterpret_cast<const char*>(pb + 0  * FLOATS_PER_LINE), _MM_HINT_T0);
                if (1  * FLOATS_PER_LINE < K) _mm_prefetch(reinterpret_cast<const char*>(pb + 1  * FLOATS_PER_LINE), _MM_HINT_T0);
                if (2  * FLOATS_PER_LINE < K) _mm_prefetch(reinterpret_cast<const char*>(pb + 2  * FLOATS_PER_LINE), _MM_HINT_T0);
                if (3  * FLOATS_PER_LINE < K) _mm_prefetch(reinterpret_cast<const char*>(pb + 3  * FLOATS_PER_LINE), _MM_HINT_T0);
                if (4  * FLOATS_PER_LINE < K) _mm_prefetch(reinterpret_cast<const char*>(pb + 4  * FLOATS_PER_LINE), _MM_HINT_T0);
                if (5  * FLOATS_PER_LINE < K) _mm_prefetch(reinterpret_cast<const char*>(pb + 5  * FLOATS_PER_LINE), _MM_HINT_T0);
                if (6  * FLOATS_PER_LINE < K) _mm_prefetch(reinterpret_cast<const char*>(pb + 6  * FLOATS_PER_LINE), _MM_HINT_T0);
                if (7  * FLOATS_PER_LINE < K) _mm_prefetch(reinterpret_cast<const char*>(pb + 7  * FLOATS_PER_LINE), _MM_HINT_T0);
            }

            for (int p = 0; p < K; ++p) {
                acc += a[p] * b[p];
            }
            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}