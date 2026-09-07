// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

#ifndef TILE_Y
#define TILE_Y 32
#endif

#ifndef TILE_X
#define TILE_X 128
#endif

    for (int ty = 0; ty < H; ty += TILE_Y) {
        const int h_end = ty + TILE_Y < H ? ty + TILE_Y : H;
        for (int tx = 0; tx < W; tx += TILE_X) {
            const int w_end = tx + TILE_X < W ? tx + TILE_X : W;

            for (int oy = ty; oy < h_end; ++oy) {
                for (int ox = tx; ox < w_end; ++ox) {
                    float sum = 0.0f;
                    for (int ky = 0; ky < K; ++ky) {
                        for (int kx = 0; kx < K; ++kx) {
                            sum += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                        }
                    }
                    out[oy * W + ox] = sum;
                }
            }

        }
    }
}
