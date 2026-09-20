/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* CPU reference of the chroma denoise stage, DESIGN.md 14.4: the colour
 * differences to quarter resolution, a joint bilateral there guided by the
 * quarter resolution G, and that band put back into every pixel, blended
 * by the strength. The same three steps as lib/shaders/chroma_*.comp.glsl,
 * summing in the same order, so that tests/test_pipeline.c can hold the
 * two together. */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ref.h"

/* DESIGN.md 14.4; lib/shaders/chroma_blur.comp.glsl holds the same five. */
#define CRAW_REF_CD_RADIUS    4
#define CRAW_REF_CD_INV_2S2   (1.0f / (2.0f * 2.0f * 2.0f))
#define CRAW_REF_CD_RANGE_K   0.35f
#define CRAW_REF_CD_CHROMA_K  0.35f
#define CRAW_REF_CD_RANGE_EPS 0.02f

/* DESIGN.md 3, on the quarter grid's own coordinates. */
static inline int craw_ref_cd_mirror(int v, int n)
{
    if (v < 0)
        return -v;
    if (v >= n)
        return 2 * n - 2 - v;
    return v;
}

int craw_ref_chroma_denoise(float *rgb, uint32_t width, uint32_t height, float strength)
{
    float *down = NULL, *blur = NULL;
    int w, h, wq, hq;
    int ret = 0;

    if (!rgb || width < 4 * (CRAW_REF_CD_RADIUS + 1) || height < 4 * (CRAW_REF_CD_RADIUS + 1))
        return CRAW_ERROR(EINVAL);
    if (!(strength > 0.0f) || !(strength <= 1.0f))
        return CRAW_ERROR(EINVAL);

    w = (int)width;
    h = (int)height;
    wq = (w + 3) / 4;
    hq = (h + 3) / 4;

    /* (d_r, d_b, G) per quarter texel, twice: the block means and their
     * filtered differences. */
    down = malloc((size_t)wq * hq * 3 * sizeof(*down));
    blur = malloc((size_t)wq * hq * 3 * sizeof(*blur));
    if (!down || !blur) {
        ret = CRAW_ERROR(ENOMEM);
        goto fail;
    }

    /* Pass 1: block means, row by row, left to right, the pixels inside the
     * frame only. */
    for (int qy = 0; qy < hq; qy++) {
        for (int qx = 0; qx < wq; qx++) {
            float sum[3] = { 0.0f, 0.0f, 0.0f }, n = 0.0f;
            float *o = &down[3 * ((size_t)qy * wq + qx)];

            for (int j = 0; j < 4; j++) {
                for (int i = 0; i < 4; i++) {
                    int x = 4 * qx + i, y = 4 * qy + j;
                    const float *px;

                    if (x >= w || y >= h)
                        continue;
                    px = &rgb[4 * ((size_t)y * w + x)];
                    sum[0] += px[0] - px[1];
                    sum[1] += px[2] - px[1];
                    sum[2] += px[1];
                    n += 1.0f;
                }
            }

            for (int c = 0; c < 3; c++)
                o[c] = sum[c] / n;
        }
    }

    /* Pass 2: the joint bilateral, borders mirrored on the quarter grid. */
    for (int qy = 0; qy < hq; qy++) {
        for (int qx = 0; qx < wq; qx++) {
            const float *c = &down[3 * ((size_t)qy * wq + qx)];
            float *o = &blur[3 * ((size_t)qy * wq + qx)];
            float level = CRAW_MAX(c[2], 0.0f) + CRAW_REF_CD_RANGE_EPS;
            float inv_range = 1.0f / (CRAW_REF_CD_RANGE_K * level);
            float inv_chroma = 1.0f / (CRAW_REF_CD_CHROMA_K * level);
            float acc[2] = { 0.0f, 0.0f }, wsum = 0.0f;

            for (int dy = -CRAW_REF_CD_RADIUS; dy <= CRAW_REF_CD_RADIUS; dy++) {
                int sy = craw_ref_cd_mirror(qy + dy, hq);

                for (int dx = -CRAW_REF_CD_RADIUS; dx <= CRAW_REF_CD_RADIUS; dx++) {
                    int sx = craw_ref_cd_mirror(qx + dx, wq);
                    const float *s = &down[3 * ((size_t)sy * wq + sx)];
                    float dg = (s[2] - c[2]) * inv_range;
                    float dd0 = (s[0] - c[0]) * inv_chroma, dd1 = (s[1] - c[1]) * inv_chroma;
                    float wgt = expf(-(float)(dx * dx + dy * dy) * CRAW_REF_CD_INV_2S2 - dg * dg
                                     - dd0 * dd0 - dd1 * dd1);

                    acc[0] += wgt * s[0];
                    acc[1] += wgt * s[1];
                    wsum += wgt;
                }
            }

            o[0] = acc[0] / wsum;
            o[1] = acc[1] / wsum;
            o[2] = c[2];
        }
    }

    /* Pass 3: the band back into every pixel. Quarter texel q holds the
     * mean of full pixels [4q, 4q + 4), centred at 4q + 1.5, so p sits at
     * (p + 0.5) / 4 - 0.5 in texel centre units; the four corners are
     * summed top left first, as the shader does. */
    for (int y = 0; y < h; y++) {
        float fy = ((float)y + 0.5f) * 0.25f - 0.5f;
        float fly = floorf(fy), ty = fy - fly;
        int y0 = CRAW_CLAMP((int)fly, 0, hq - 1), y1 = CRAW_CLAMP((int)fly + 1, 0, hq - 1);

        for (int x = 0; x < w; x++) {
            float fx = ((float)x + 0.5f) * 0.25f - 0.5f;
            float flx = floorf(fx), tx = fx - flx;
            int x0 = CRAW_CLAMP((int)flx, 0, wq - 1), x1 = CRAW_CLAMP((int)flx + 1, 0, wq - 1);
            float w00 = (1.0f - tx) * (1.0f - ty), w10 = tx * (1.0f - ty);
            float w01 = (1.0f - tx) * ty,          w11 = tx * ty;
            const float *d00 = &down[3 * ((size_t)y0 * wq + x0)], *d10 = &down[3 * ((size_t)y0 * wq + x1)];
            const float *d01 = &down[3 * ((size_t)y1 * wq + x0)], *d11 = &down[3 * ((size_t)y1 * wq + x1)];
            const float *b00 = &blur[3 * ((size_t)y0 * wq + x0)], *b10 = &blur[3 * ((size_t)y0 * wq + x1)];
            const float *b01 = &blur[3 * ((size_t)y1 * wq + x0)], *b11 = &blur[3 * ((size_t)y1 * wq + x1)];
            float *px = &rgb[4 * ((size_t)y * w + x)];
            float g = px[1];
            float d[2] = { px[0] - g, px[2] - g };

            for (int c = 0; c < 2; c++) {
                float lo = w00 * d00[c] + w10 * d10[c] + w01 * d01[c] + w11 * d11[c];
                float lf = w00 * b00[c] + w10 * b10[c] + w01 * b01[c] + w11 * b11[c];

                d[c] += strength * (lf - lo);
            }

            px[0] = g + d[0];
            px[2] = g + d[1];
        }
    }

fail:
    free(blur);
    free(down);

    return ret;
}
