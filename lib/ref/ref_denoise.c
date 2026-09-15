/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* CPU reference of the raw domain denoise pass, DESIGN.md 14.3. Same
 * definition as lib/shaders/denoise_nlm.comp.glsl, down to the order the
 * patch distance is accumulated in and the 16-bit re-encoding at the end,
 * so that tests/test_pipeline.c can hold the two to the tolerance of
 * DESIGN.md 14.6 rather than to an eyeball. */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ref.h"

/* Same constants as lib/pipeline.c; DESIGN.md 14.3 is the source of both. */
#define CRAW_REF_DN_SEARCH_R 3
#define CRAW_REF_DN_PATCH_R  1
#define CRAW_REF_DN_GUARD_LO 0.90f
#define CRAW_REF_DN_GUARD_HI 1.00f

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

int craw_ref_denoise(const uint16_t *bayer, uint32_t width, uint32_t height,
                     size_t stride, const CRAWProfile *profile,
                     const CRAWNoiseModel *model, float strength,
                     uint16_t *out, size_t out_stride)
{
    const int sr = CRAW_REF_DN_SEARCH_R, pr = CRAW_REF_DN_PATCH_R;
    const int np = (2 * pr + 1) * (2 * pr + 1);
    const float inv_np = 1.0f / (float)np;
    const float guard_inv = 1.0f / (CRAW_REF_DN_GUARD_HI - CRAW_REF_DN_GUARD_LO);
    struct craw_ref_gat g;
    float inv_h2;
    float *plane = NULL;
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
        width < (uint32_t)(2 * (sr + pr + 1)) || height < (uint32_t)(2 * (sr + pr + 1)))
        return CRAW_ERROR(EINVAL);
    if (stride < width || out_stride < width)
        return CRAW_ERROR(EINVAL);
    if (!(model->a > 0.0f))
        return CRAW_ERROR(EINVAL);
    if (!(strength > 0.0f))
        return CRAW_ERROR(EINVAL);

    for (int s = 0; s < 4; s++) {
        if (!(profile->white_level[s] > profile->black_level[s]))
            return CRAW_ERROR(EINVAL);
    }

    craw_ref_gat_init(&g, model->a, model->b);
    inv_h2 = 1.0f / (strength * strength);

    w2 = (int)(width / 2);
    h2 = (int)(height / 2);

    plane = malloc((size_t)w2 * h2 * sizeof(*plane));
    mx = malloc((size_t)(w2 + 2 * rad) * sizeof(*mx));
    my = malloc((size_t)(h2 + 2 * rad) * sizeof(*my));
    if (!plane || !mx || !my) {
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

        /* The whole stabilised plane at once: the shader keeps a tile of
         * this in shared memory, which is the same values. */
        for (int k = 0; k < h2; k++) {
            for (int i = 0; i < w2; i++) {
                float raw = bayer[(size_t)(2 * k + py) * stride + 2 * i + px] *
                            (1.0f / 65535.0f);
                plane[(size_t)k * w2 + i] = craw_ref_gat(&g, (raw - black) * inv_range);
            }
        }

        for (int k = 0; k < h2; k++) {
            for (int i = 0; i < w2; i++) {
                float cp[(2 * CRAW_REF_DN_PATCH_R + 1) * (2 * CRAW_REF_DN_PATCH_R + 1)];
                float acc = 0.0f, wsum = 0.0f;
                float raw0, u0, u, gr;
                int n = 0;

                for (int ky = -pr; ky <= pr; ky++)
                    for (int kx = -pr; kx <= pr; kx++)
                        cp[n++] = plane[(size_t)my[k + ky + rad] * w2 + mx[i + kx + rad]];

                for (int dy = -sr; dy <= sr; dy++) {
                    for (int dx = -sr; dx <= sr; dx++) {
                        float d2 = 0.0f, wgt, sv;

                        n = 0;
                        for (int ky = -pr; ky <= pr; ky++) {
                            for (int kx = -pr; kx <= pr; kx++) {
                                float e = cp[n++] -
                                          plane[(size_t)my[k + dy + ky + rad] * w2 +
                                                mx[i + dx + kx + rad]];
                                d2 += e * e;
                            }
                        }

                        wgt = expf(-CRAW_MAX(d2 * inv_np - 2.0f, 0.0f) * inv_h2);
                        sv = plane[(size_t)my[k + dy + rad] * w2 + mx[i + dx + rad]];
                        wsum += wgt;
                        acc += wgt * sv;
                    }
                }

                u = craw_ref_igat(&g, acc / wsum);

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
    free(my);
    free(mx);
    free(plane);

    return ret;
}
