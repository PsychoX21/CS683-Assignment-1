// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>

#include "convolution.h"

#define TILE_Y 2048
#define TILE_X 2048
#define UNROLL_FACTOR 8

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        int ox = 0;
        for (; ox <= W - UNROLL_FACTOR * 8; ox += UNROLL_FACTOR * 8) {
            __m256 vacc0 = _mm256_setzero_ps();
            __m256 vacc1 = _mm256_setzero_ps();
            __m256 vacc2 = _mm256_setzero_ps();
            __m256 vacc3 = _mm256_setzero_ps();
            __m256 vacc4 = _mm256_setzero_ps();
            __m256 vacc5 = _mm256_setzero_ps();
            __m256 vacc6 = _mm256_setzero_ps();
            __m256 vacc7 = _mm256_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = &in[(oy + ky) * in_stride + ox];
                for (int kx = 0; kx < K; ++kx) {
                    __m256 vk = _mm256_set1_ps(ker[ky * K + kx]);

                    __m256 vin0 = _mm256_loadu_ps(&in_row[kx]);
                    __m256 vin1 = _mm256_loadu_ps(&in_row[kx + 8]);
                    __m256 vin2 = _mm256_loadu_ps(&in_row[kx + 16]);
                    __m256 vin3 = _mm256_loadu_ps(&in_row[kx + 24]);
                    __m256 vin4 = _mm256_loadu_ps(&in_row[kx + 32]);
                    __m256 vin5 = _mm256_loadu_ps(&in_row[kx + 40]);
                    __m256 vin6 = _mm256_loadu_ps(&in_row[kx + 48]);
                    __m256 vin7 = _mm256_loadu_ps(&in_row[kx + 56]);

                    vacc0 = _mm256_fmadd_ps(vin0, vk, vacc0);
                    vacc1 = _mm256_fmadd_ps(vin1, vk, vacc1);
                    vacc2 = _mm256_fmadd_ps(vin2, vk, vacc2);
                    vacc3 = _mm256_fmadd_ps(vin3, vk, vacc3);
                    vacc4 = _mm256_fmadd_ps(vin4, vk, vacc4);
                    vacc5 = _mm256_fmadd_ps(vin5, vk, vacc5);
                    vacc6 = _mm256_fmadd_ps(vin6, vk, vacc6);
                    vacc7 = _mm256_fmadd_ps(vin7, vk, vacc7);
                }
            }

            _mm256_storeu_ps(&out[oy * W + ox], vacc0);
            _mm256_storeu_ps(&out[oy * W + ox + 8], vacc1);
            _mm256_storeu_ps(&out[oy * W + ox + 16], vacc2);
            _mm256_storeu_ps(&out[oy * W + ox + 24], vacc3);
            _mm256_storeu_ps(&out[oy * W + ox + 32], vacc4);
            _mm256_storeu_ps(&out[oy * W + ox + 40], vacc5);
            _mm256_storeu_ps(&out[oy * W + ox + 48], vacc6);
            _mm256_storeu_ps(&out[oy * W + ox + 56], vacc7);
        }

        for (; ox <= W - 8; ox += 8) {
            __m256 vacc = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = &in[(oy + ky) * in_stride + ox];
                for (int kx = 0; kx < K; ++kx) {
                    __m256 vk = _mm256_set1_ps(ker[ky * K + kx]);
                    __m256 vin = _mm256_loadu_ps(&in_row[kx]);
                    vacc = _mm256_fmadd_ps(vin, vk, vacc);
                }
            }
            _mm256_storeu_ps(&out[oy * W + ox], vacc);
        }

        for (; ox < W; ++ox) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
            }
            out[oy * W + ox] = acc;
        }
    }
}
