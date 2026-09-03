// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"
#include <cstring>
#define UNROLL_FACTOR 8

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    // const int p = K / 2;
    // const int in_stride = W + 2 * p;  // padded row stride

    // std::memset(out, 0, sizeof(float) * H * W);

    // for(int ky = 0; ky < K; ++ky) {
    //     for(int kx = 0; kx < K; ++kx) {
    //         const float k_val = ker[ky * K + kx];
    //         for(int oy = 0; oy < H; ++oy) {
    //             int ox = 0;
    //             for(; ox < W - UNROLL_FACTOR; ox += UNROLL_FACTOR) {
    //                 out[oy * W + ox] += in[(oy + ky) * in_stride + (ox + kx)] * k_val;
    //                 out[oy * W + ox + 1] += in[(oy + ky) * in_stride + (ox + kx + 1)] * k_val;
    //                 out[oy * W + ox + 2] += in[(oy + ky) * in_stride + (ox + kx + 2)] * k_val;
    //                 out[oy * W + ox + 3] += in[(oy + ky) * in_stride + (ox + kx + 3)] * k_val;
    //                 out[oy * W + ox + 4] += in[(oy + ky) * in_stride + (ox + kx + 4)] * k_val;
    //                 out[oy * W + ox + 5] += in[(oy + ky) * in_stride + (ox + kx + 5)] * k_val;
    //                 out[oy * W + ox + 6] += in[(oy + ky) * in_stride + (ox + kx + 6)] * k_val;
    //                 out[oy * W + ox + 7] += in[(oy + ky) * in_stride + (ox + kx + 7)] * k_val;
    //             }
    //             for(; ox < W; ++ox) {
    //                 out[oy * W + ox] += in[(oy + ky) * in_stride + (ox + kx)] * k_val;
    //             }
    //         }
    //     }
    // }

    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    
    int oy = 0;
    for (; oy < H - UNROLL_FACTOR; oy += UNROLL_FACTOR) {

        for (int ox = 0; ox < W; ++ox) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
            }
            out[oy * W + ox] = acc;
        }

        for (int ox = 0; ox < W; ++ox) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + 1 + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
            }
            out[(oy + 1) * W + ox] = acc;
        }

        for (int ox = 0; ox < W; ++ox) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + 2 + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
            }
            out[(oy + 2) * W + ox] = acc;
        }

        for (int ox = 0; ox < W; ++ox) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + 3 + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
            }
            out[(oy + 3) * W + ox] = acc;
        }

        for (int ox = 0; ox < W; ++ox) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + 4 + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
            }
            out[(oy + 4) * W + ox] = acc;
        }

        for (int ox = 0; ox < W; ++ox) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + 5 + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
            }
            out[(oy + 5) * W + ox] = acc;
        }

        for (int ox = 0; ox < W; ++ox) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + 6 + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
            }
            out[(oy + 6) * W + ox] = acc;
        }

        for (int ox = 0; ox < W; ++ox) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + 7 + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
            }
            out[(oy + 7) * W + ox] = acc;
        }
    }

    for (; oy < H; ++oy) {
        for (int ox = 0; ox < W; ++ox) {
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