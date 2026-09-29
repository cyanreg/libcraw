/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* CPU reference of the raw domain denoise pass, DESIGN.md 14.3, and of the
 * temporal average in front of it, 14.7. Same definition as
 * lib/shaders/denoise_nlm.comp.glsl, down to the order the patch distance
 * is accumulated in and the 16-bit re-encoding at the end, so that
 * tests/test_pipeline.c can hold the two to the tolerance of DESIGN.md
 * 14.6 rather than to an eyeball. */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ref.h"

/* Same constants as lib/pipeline.c; DESIGN.md 14.3 is the source of both. */
#define CRAW_REF_DN_SEARCH_R 3
#define CRAW_REF_DN_PATCH_R  1
#define CRAW_REF_DN_GUARD_LO 0.90f
#define CRAW_REF_DN_GUARD_HI 1.00f
/* The least noise variance a sample is taken to carry, DESIGN.md 14.7. */
#define CRAW_REF_DN_VAR_MIN  1.0e-4f

/* The transform of DESIGN.md 14.3 and its constants, held together so the
 * reference cannot drift from the push constant block. */
struct craw_ref_gat {
    float a, c, sc, fwd, inv, off, t0;
};

static void craw_ref_gat_init(struct craw_ref_gat *g, float a, float b)
{
    g->a = a;
    g->c = 3.0f * a * a / 8.0f + CRAW_MAX(b, 0.0f);
    g->sc = sqrtf(g->c);
    g->fwd = 2.0f / a;
    g->inv = a / 4.0f;
    g->off = g->c / a;
    g->t0 = g->fwd * g->sc;
}

static inline float craw_ref_gat(const struct craw_ref_gat *g, float u)
{
    if (u >= 0.0f)
        return g->fwd * sqrtf(g->a * u + g->c);

    /* Below black the square root has no real value and no bounded
     * derivative; its tangent at u = 0 continues it with exactly the slope
     * the read noise there calls for. */
    return g->t0 + u / g->sc;
}

static inline float craw_ref_igat(const struct craw_ref_gat *g, float t)
{
    if (t >= g->t0)
        return g->inv * t * t - g->off;

    return (t - g->t0) * g->sc;
}

/* Mirror in the plane's own coordinates, DESIGN.md 14.3: each CFA plane is
 * an image in its own right and its borders reflect without repeating the
 * edge sample. */
static inline int craw_ref_dn_mirror(int v, int n)
{
    if (n < 2)
        return 0;

    while (v < 0 || v >= n) {
        if (v < 0)
            v = -v;
        if (v >= n)
            v = 2 * n - 2 - v;
    }

    return v;
}

/* The weight of the previous frame in the recursive average of DESIGN.md
 * 14.7 at plane sample (i, k): this frame's stabilised plane `cur` against
 * the previous frame's `prev`, compared over the 3x3 around the sample
 * with the plane mirrored, summed in the shader's order. */
static float craw_ref_alpha(const float *cur, const float *prev, int w2, int h2,
                            int i, int k, float alpha_max, float inv_h2)
{
    float d2 = 0.0f;

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            size_t n = (size_t)craw_ref_dn_mirror(k + ky, h2) * w2 +
                       craw_ref_dn_mirror(i + kx, w2);
            float e = prev[n] - cur[n];
            d2 += e * e;
        }
    }

    return alpha_max * expf(-CRAW_MAX(d2 * (1.0f / 9.0f) - 2.0f, 0.0f) * inv_h2);
}

/* The variance the GPU's state keeps: its square root in eight bits. */
static inline float craw_ref_var_store(float v)
{
    float q = rintf(sqrtf(CRAW_CLAMP(v, 0.0f, 1.0f)) * 255.0f) * (1.0f / 255.0f);

    return q * q;
}

/* The non-local means of DESIGN.md 14.3 at one sample of a stabilised
 * plane `w2` wide. `mx` and `my` point at the mirrored column and row of
 * the sample itself, so that mx[-r] .. mx[r] are its neighbours'. `var`,
 * when there is a temporal history, is the noise variance left in every
 * sample of the plane: the distance loses the two samples' and the
 * strength is taken in units of the centre's (14.7). Without it every
 * sample carries a variance of 1. */
static float craw_ref_nlm(const float *plane, const float *var, int w2, const int *mx,
                          const int *my, float inv_h2)
{
    const int sr = CRAW_REF_DN_SEARCH_R, pr = CRAW_REF_DN_PATCH_R;
    const float inv_np = 1.0f / (float)((2 * pr + 1) * (2 * pr + 1));
    float cp[(2 * CRAW_REF_DN_PATCH_R + 1) * (2 * CRAW_REF_DN_PATCH_R + 1)];
    float acc = 0.0f, wsum = 0.0f;
    int n = 0;

    for (int ky = -pr; ky <= pr; ky++)
        for (int kx = -pr; kx <= pr; kx++)
            cp[n++] = plane[(size_t)my[ky] * w2 + mx[kx]];

    for (int dy = -sr; dy <= sr; dy++) {
        for (int dx = -sr; dx <= sr; dx++) {
            float d2 = 0.0f, wgt, sv;

            n = 0;
            for (int ky = -pr; ky <= pr; ky++) {
                for (int kx = -pr; kx <= pr; kx++) {
                    float e = cp[n++] - plane[(size_t)my[dy + ky] * w2 + mx[dx + kx]];
                    d2 += e * e;
                }
            }

            if (var) {
                float vc = var[(size_t)my[0] * w2 + mx[0]];
                float v2 = vc + var[(size_t)my[dy] * w2 + mx[dx]];

                wgt = expf(-CRAW_MAX(d2 * inv_np - v2, 0.0f) * (inv_h2 / vc));
            } else {
                wgt = expf(-CRAW_MAX(d2 * inv_np - 2.0f, 0.0f) * inv_h2);
            }
            sv = plane[(size_t)my[dy] * w2 + mx[dx]];
            wsum += wgt;
            acc += wgt * sv;
        }
    }

    return acc / wsum;
}

int craw_ref_denoise(const uint16_t *bayer, uint32_t width, uint32_t height,
                     size_t stride, const CRAWProfile *profile,
                     const CRAWNoiseModel *model, float strength,
                     uint16_t *out, size_t out_stride)
{
    if (!(strength > 0.0f))
        return CRAW_ERROR(EINVAL);

    return craw_ref_denoise_temporal(bayer, width, height, stride, profile, model, strength,
                                     0.0f, NULL, NULL, out, out_stride);
}

int craw_ref_denoise_temporal(const uint16_t *bayer, uint32_t width, uint32_t height,
                              size_t stride, const CRAWProfile *profile,
                              const CRAWNoiseModel *model, float strength,
                              float alpha_max, const float *prev, float *next,
                              uint16_t *out, size_t out_stride)
{
    const int sr = CRAW_REF_DN_SEARCH_R, pr = CRAW_REF_DN_PATCH_R;
    const float guard_inv = 1.0f / (CRAW_REF_DN_GUARD_HI - CRAW_REF_DN_GUARD_LO);
    struct craw_ref_gat g;
    float h, inv_h2;
    float *plane = NULL, *cur = NULL, *tprev = NULL, *var = NULL;
    int *mx = NULL, *my = NULL;
    int w2, h2, rad = sr + pr;
    int ret = 0;

    if (!bayer || !profile || !model || !out)
        return CRAW_ERROR(EINVAL);
    if (profile->cfa != CRAW_CFA_RGGB)
        return CRAW_ERROR(ENOTSUP);
    /* The same minimum craw_pipeline_create() enforces: the plane mirror
     * of DESIGN.md 14.1 is good for one reflection, so the stencil radius
     * must fit inside the plane. */
    if ((width & 1) || (height & 1) ||
        width < (uint32_t)(2 * (sr + pr + 2)) || height < (uint32_t)(2 * (sr + pr + 2)))
        return CRAW_ERROR(EINVAL);
    if (stride < width || out_stride < width)
        return CRAW_ERROR(EINVAL);
    if (!(model->a > 0.0f))
        return CRAW_ERROR(EINVAL);
    /* At least one of the two filters, as craw_pipeline_record() never
     * dispatches the pass with neither. */
    if (!(strength >= 0.0f) || !(alpha_max >= 0.0f) || !(alpha_max <= 1.0f) ||
        !(strength > 0.0f || alpha_max > 0.0f))
        return CRAW_ERROR(EINVAL);
    if (alpha_max > 0.0f && !next)
        return CRAW_ERROR(EINVAL);
    if (!(alpha_max > 0.0f))
        prev = NULL;

    for (int s = 0; s < 4; s++) {
        if (!(profile->white_level[s] > profile->black_level[s]))
            return CRAW_ERROR(EINVAL);
    }

    craw_ref_gat_init(&g, model->a, model->b);
    /* One threshold for both filters: the spatial strength when that one
     * runs and the modelled noise level when it does not, as
     * craw_pipeline_denoise_push() sets it. */
    h = strength > 0.0f ? strength : 1.0f;
    inv_h2 = 1.0f / (h * h);

    w2 = (int)(width / 2);
    h2 = (int)(height / 2);

    plane = malloc((size_t)w2 * h2 * sizeof(*plane));
    mx = malloc((size_t)(w2 + 2 * rad) * sizeof(*mx));
    my = malloc((size_t)(h2 + 2 * rad) * sizeof(*my));
    if (prev) {
        cur = malloc((size_t)w2 * h2 * sizeof(*cur));
        tprev = malloc((size_t)w2 * h2 * sizeof(*tprev));
        var = malloc((size_t)w2 * h2 * sizeof(*var));
    }
    if (!plane || !mx || !my || (prev && (!cur || !tprev || !var))) {
        ret = CRAW_ERROR(ENOMEM);
        goto done;
    }

    /* The mirror of every position any stencil can reach, resolved once.
     * The shader gets the same thing for free: its shared tile is loaded
     * through the mirror and then indexed straight. */
    for (int i = 0; i < w2 + 2 * rad; i++)
        mx[i] = craw_ref_dn_mirror(i - rad, w2);
    for (int k = 0; k < h2 + 2 * rad; k++)
        my[k] = craw_ref_dn_mirror(k - rad, h2);

    for (int pl = 0; pl < 4; pl++) {
        int px = pl & 1, py = pl >> 1;
        float black = profile->black_level[pl];
        float range = profile->white_level[pl] - profile->black_level[pl];
        float inv_range = 1.0f / range;

        /* The whole stabilised plane at once, averaged with the previous
         * frame's when the temporal pass has one: the shader keeps a tile
         * of this in shared memory, which is the same values. The state is
         * what the temporal pass made of each sample, before the spatial
         * filter, and it holds the level rather than the stabilised value. */
        for (int k = 0; k < h2; k++) {
            for (int i = 0; i < w2; i++) {
                size_t mp = (size_t)(2 * k + py) * width + 2 * i + px;
                float raw = bayer[(size_t)(2 * k + py) * stride + 2 * i + px] *
                            (1.0f / 65535.0f);
                float t = craw_ref_gat(&g, (raw - black) * inv_range);

                if (prev) {
                    cur[(size_t)k * w2 + i] = t;
                    tprev[(size_t)k * w2 + i] = craw_ref_gat(&g, prev[mp]);
                }
                plane[(size_t)k * w2 + i] = t;
            }
        }
        /* The variance left follows from the weight: this frame brings 1,
         * the previous frame's state what it kept. */
        if (prev) {
            const float *prev_var = prev + (size_t)width * height;

            for (int k = 0; k < h2; k++) {
                for (int i = 0; i < w2; i++) {
                    size_t c = (size_t)k * w2 + i;
                    size_t mp = (size_t)(2 * k + py) * width + 2 * i + px;
                    float a = craw_ref_alpha(cur, tprev, w2, h2, i, k, alpha_max, inv_h2);

                    plane[c] = cur[c] * (1.0f - a) + tprev[c] * a;
                    var[c] = CRAW_MAX((1.0f - a) * (1.0f - a) + a * a * prev_var[mp],
                                      CRAW_REF_DN_VAR_MIN);
                }
            }
        }
        if (alpha_max > 0.0f) {
            float *next_var = next + (size_t)width * height;

            for (int k = 0; k < h2; k++) {
                for (int i = 0; i < w2; i++) {
                    size_t mp = (size_t)(2 * k + py) * width + 2 * i + px;

                    next[mp] = craw_ref_igat(&g, plane[(size_t)k * w2 + i]);
                    next_var[mp] = prev ? craw_ref_var_store(var[(size_t)k * w2 + i]) : 1.0f;
                }
            }
        }

        for (int k = 0; k < h2; k++) {
            for (int i = 0; i < w2; i++) {
                float raw0, u0, u, gr;

                /* The temporal average alone is the plane as it stands. */
                if (strength > 0.0f)
                    u = craw_ref_igat(&g, craw_ref_nlm(plane, prev ? var : NULL, w2,
                                                       mx + rad + i, my + rad + k, inv_h2));
                else
                    u = craw_ref_igat(&g, plane[(size_t)k * w2 + i]);

                raw0 = bayer[(size_t)(2 * k + py) * stride + 2 * i + px] *
                       (1.0f / 65535.0f);
                u0 = (raw0 - black) * inv_range;
                gr = CRAW_CLAMP((u0 - CRAW_REF_DN_GUARD_LO) * guard_inv, 0.0f, 1.0f);
                u = u + (u0 - u) * gr;

                /* The GPU stores into an R16_UNORM image, so the reference
                 * quantizes the same way: round to nearest, ties to even,
                 * over the full 16-bit range. */
                {
                    float raw = CRAW_CLAMP(black + u * range, 0.0f, 1.0f);
                    out[(size_t)(2 * k + py) * out_stride + 2 * i + px] =
                        (uint16_t)rintf(raw * 65535.0f);
                }
            }
        }
    }

done:
    free(var);
    free(tprev);
    free(cur);
    free(my);
    free(mx);
    free(plane);

    return ret;
}
