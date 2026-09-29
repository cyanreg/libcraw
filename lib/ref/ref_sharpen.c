/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* CPU reference of the capture sharpening stage, DESIGN.md 20: the
 * observed luma, Richardson-Lucy iterations with a Gaussian PSF, and the
 * luma ratio, masked and clamped, applied to the camera RGB. The sequential
 * form of what lib/shaders/sharpen.comp.glsl fuses into one dispatch per
 * iteration, summing every blur in the same order and rounding the planes
 * to fp16 where the GPU stores them, so that tests/test_pipeline.c can hold
 * the two together. The taps and the noise constants come from
 * lib/sharpen.h, as the pipeline's do. */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ref.h"
#include "sharpen.h"

static inline int craw_sh_mirror(int v, int n)
{
    if (v < 0)
        return -v;
    if (v >= n)
        return 2 * n - 2 - v;
    return v;
}

/* dst = the separable blur of src by the taps, along x and then along y,
 * borders mirrored; `tmp` holds the pass along x. */
static void craw_sh_blur(const float *src, float *tmp, float *dst, int w, int h,
                         const float *k, int r)
{
    for (int y = 0; y < h; y++) {
        const float *row = &src[(size_t)y * w];
        for (int x = 0; x < w; x++) {
            float s = k[0] * row[x];
            for (int i = 1; i <= r; i++)
                s += k[i] * (row[craw_sh_mirror(x - i, w)] + row[craw_sh_mirror(x + i, w)]);
            tmp[(size_t)y * w + x] = s;
        }
    }
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float s = k[0] * tmp[(size_t)y * w + x];
            for (int i = 1; i <= r; i++)
                s += k[i] * (tmp[(size_t)craw_sh_mirror(y - i, h) * w + x] +
                             tmp[(size_t)craw_sh_mirror(y + i, h) * w + x]);
            dst[(size_t)y * w + x] = s;
        }
    }
}

int craw_ref_sharpen(float *rgb, uint32_t width, uint32_t height, float radius,
                     uint32_t iterations, float contrast, const CRAWNoiseModel *model,
                     const float wb[3])
{
    const int w = (int)width, h = (int)height;
    const size_t n = (size_t)w * h;
    float k[CRAW_SH_RMAX + 1], n_a = 0.0f, n_b = 0.0f;
    float *d = NULL, *u = NULL, *tmp = NULL, *b = NULL;
    int r, ret = 0;

    if (!rgb || w < CRAW_SH_MIN_SIZE || h < CRAW_SH_MIN_SIZE)
        return CRAW_ERROR(EINVAL);
    if (!(radius > 0.0f && radius <= CRAW_SHARPEN_MAX_RADIUS) || !iterations ||
        iterations > CRAW_SHARPEN_MAX_ITERATIONS || !(contrast >= 0.0f))
        return CRAW_ERROR(EINVAL);
    if (contrast > 0.0f && (!model || !(model->a > 0.0f) || !wb))
        return CRAW_ERROR(EINVAL);

    r = craw_sharpen_kernel(radius, k);
    if (contrast > 0.0f)
        craw_sharpen_noise(model, wb, &n_a, &n_b);

    d = malloc(n * sizeof(*d));
    u = malloc(n * sizeof(*u));
    tmp = malloc(n * sizeof(*tmp));
    b = malloc(n * sizeof(*b));
    if (!d || !u || !tmp || !b) {
        ret = CRAW_ERROR(ENOMEM);
        goto done;
    }

    /* The observed luma, and the first estimate, which is the same. */
    for (size_t i = 0; i < n; i++) {
        const float *px = &rgb[i * 4];
        float y = (px[0] + 2.0f * px[1] + px[2]) * 0.25f;
        d[i] = craw_ref_to_half(fmaxf(y, 0.0f) + CRAW_SH_PEDESTAL);
    }
    memcpy(u, d, n * sizeof(*u));

    for (uint32_t it = 0; it < iterations; it++) {
        /* b = K * u, the ratio in its place, c = K * ratio into b. */
        craw_sh_blur(u, tmp, b, w, h, k, r);
        for (size_t i = 0; i < n; i++)
            b[i] = d[i] / b[i];
        craw_sh_blur(b, tmp, b, w, h, k, r);

        for (size_t i = 0; i < n; i++) {
            float c = CRAW_CLAMP(b[i], 1.0f / CRAW_SH_CORR_MAX, CRAW_SH_CORR_MAX);
            u[i] = fmaxf(u[i] * c, CRAW_SH_PEDESTAL);
            /* The last estimate stays in the shader's registers. */
            if (it + 1 < iterations)
                u[i] = craw_ref_to_half(u[i]);
        }
    }

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t i = (size_t)y * w + x;
            float d0 = d[i], lo = d0, hi = d0, m = 1.0f, rho;

            for (int j = -CRAW_SH_RANGE_R; j <= CRAW_SH_RANGE_R; j++) {
                for (int l = -CRAW_SH_RANGE_R; l <= CRAW_SH_RANGE_R; l++) {
                    float v = d[(size_t)craw_sh_mirror(y + j, h) * w + craw_sh_mirror(x + l, w)];
                    lo = fminf(lo, v);
                    hi = fmaxf(hi, v);
                }
            }

            if (contrast > 0.0f) {
                const float inv_n = 1.0f / (float)((2 * CRAW_SH_MASK_R + 1) * (2 * CRAW_SH_MASK_R + 1));
                float e = 0.0f, l = 0.0f, var, z, t;

                for (int j = -CRAW_SH_MASK_R; j <= CRAW_SH_MASK_R; j++) {
                    for (int q = -CRAW_SH_MASK_R; q <= CRAW_SH_MASK_R; q++) {
                        int yy = craw_sh_mirror(y + j, h), xx = craw_sh_mirror(x + q, w);
                        const float *row = &d[(size_t)yy * w];
                        float gx = row[craw_sh_mirror(x + q + 1, w)] - row[craw_sh_mirror(x + q - 1, w)];
                        float gy = d[(size_t)craw_sh_mirror(y + j + 1, h) * w + xx] -
                                   d[(size_t)craw_sh_mirror(y + j - 1, h) * w + xx];
                        e += gx * gx + gy * gy;
                        l += row[xx];
                    }
                }
                e *= 0.25f * inv_n;
                l = l * inv_n - CRAW_SH_PEDESTAL;

                var = n_a * fmaxf(l, 0.0f) + n_b;
                z = sqrtf(e / fmaxf(var, 1e-30f));
                t = CRAW_CLAMP((z - contrast) / contrast, 0.0f, 1.0f);
                m = t * t * (3.0f - 2.0f * t);
            }

            rho = 1.0f + m * (CRAW_CLAMP(u[i], lo, hi) / d0 - 1.0f);
            if (rho == 1.0f)
                continue;
            for (int c = 0; c < 3; c++)
                rgb[i * 4 + c] *= rho;
        }
    }

done:
    free(b);
    free(tmp);
    free(u);
    free(d);
    return ret;
}
