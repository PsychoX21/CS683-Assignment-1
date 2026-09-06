// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>
#include <algorithm>

#include "matmul.h"

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    const int BM = 64;
    const int BN = 64;

    for (int bi = 0; bi < M; bi += BM) {
        const int i_end = std::min(bi + BM, M);
        for (int bj = 0; bj < N; bj += BN) {
            const int j_end = std::min(bj + BN, N);
            
            for (int i = bi; i < i_end; ++i) {
                const float* a = A + (long)i * lda;
                for (int j = bj; j < j_end; ++j) {
                    const float* b = B + (long)j * ldb;
                    
                    if (j + 1 < j_end) {
                        const float* b_next = B + (long)(j + 1) * ldb;
                        _mm_prefetch((const char*)(b_next), _MM_HINT_T0);
                        _mm_prefetch((const char*)(b_next + 16), _MM_HINT_T0);
                        _mm_prefetch((const char*)(b_next + 32), _MM_HINT_T0);
                        _mm_prefetch((const char*)(b_next + 48), _MM_HINT_T0);
                    }
                    
                    float acc = 0.0f;
                    for (int p = 0; p < K; ++p) {
                        acc += a[p] * b[p];
                    }
                    C[(long)i * ldc + j] = acc;
                }
            }
        }
    }
}