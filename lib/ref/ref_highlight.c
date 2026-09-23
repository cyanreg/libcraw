/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* CPU reference of the highlight reconstruction, DESIGN.md 18: the colour
 * field's level 0, the pull up its pyramid, the push back down, and the
 * fill into a balanced, unclipped mosaic, rounded to fp16 exactly as the
 * GPU stores it. lib/shaders/highlight_*.comp.glsl implement the same text
 * and are held to this by tests/test_pipeline.c. */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ref.h"

/* DESIGN.md 18.2; lib/shaders/highlight.glsl carries the same numbers. */
#define CRAW_HL_U_LO    0.92f
#define CRAW_HL_U_HI    0.98f
#define CRAW_HL_EPS     1e-4f
#define CRAW_HL_LOG_MAX 2.0f
#define CRAW_HL_W_SAT   1.0f
#define CRAW_HL_DILATE  3

/* Channel each quad site carries, RGGB: R, Gr, Gb, B. */
static const int craw_hl_site_chan[4] = { 0, 1, 1, 2 };

static inline int craw_hl_mirror(int v, int n)
{
    if (v < 0)
        return -v;
    if (v >= n)
        return 2 * n - 2 - v;
    return v;
}

static inline int craw_hl_clampi(int v, int n)
{
    return v < 0 ? 0 : (v >= n ? n - 1 : v);
}

static inline float craw_hl_smoothstep(float e0, float e1, float v)
{
    float t = (v - e0) / (e1 - e0);

    t = CRAW_CLAMP(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

/* Round to the nearest fp16 value, ties to even, and back: the rebuilt
 * mosaic is R16F on the GPU. Only finite values below the fp16 maximum
 * reach here; the stage's own bounds keep them there. */
static float craw_hl_to_half(float f)
{
    uint32_t b, sign, exp, mant, r;
    int e;

    memcpy(&b, &f, sizeof(b));
    sign = b & 0x80000000u;
    b &= 0x7FFFFFFFu;
    exp = b >> 23;
    e = (int)exp - 127;

    if (b == 0)
        return f;

    if (e < -14) {
        /* Subnormal in fp16: a fixed step of 2^-24. */
        float step = 5.9604644775390625e-8f;
        float a = fabsf(f) / step;
        float q = nearbyintf(a) * step;
        return sign ? -q : q;
    }

    /* Keep 10 of the 23 mantissa bits, rounding the other 13 to nearest,
     * ties to even. */
    mant = b & 0x7FFFFFu;
    r = mant & 0x1FFFu;
    mant >>= 13;
    if (r > 0x1000u || (r == 0x1000u && (mant & 1u)))
        mant++;
    if (mant == 0x400u) {
        mant = 0;
        exp++;
    }

    b = sign | (exp << 23) | (mant << 13);
    memcpy(&f, &b, sizeof(f));
    return f;
}

/* One level of the colour field: (S_r, S_b, W) per texel while it is
 * summed, (c_r, c_b, W) once it has been pushed. */
typedef struct CRAWHlLevel {
    int w, h;
    float *t;
} CRAWHlLevel;

/* The level's colour at a texel, clamped to its edge; the top level is
 * still a sum when it is read and is normalized on the way. */
static void craw_hl_color(const CRAWHlLevel *l, int x, int y, bool top, float out[2])
{
    const float *t = &l->t[((size_t)craw_hl_clampi(y, l->h) * l->w + craw_hl_clampi(x, l->w)) * 3];

    if (top) {
        out[0] = t[2] > 0.0f ? t[0] / t[2] : 0.0f;
        out[1] = t[2] > 0.0f ? t[1] / t[2] : 0.0f;
    } else {
        out[0] = t[0];
        out[1] = t[1];
    }
}

/* Bilinear at a position in the level's texel coordinates, mixed along x
 * first, then y, as GLSL's mix() does: a * (1 - t) + b * t. */
static void craw_hl_bilinear(const CRAWHlLevel *l, float px, float py, bool top, float out[2])
{
    float fx0 = floorf(px), fy0 = floorf(py);
    float fx = px - fx0, fy = py - fy0;
    int x0 = (int)fx0, y0 = (int)fy0;
    float c00[2], c10[2], c01[2], c11[2];

    craw_hl_color(l, x0,     y0,     top, c00);
    craw_hl_color(l, x0 + 1, y0,     top, c10);
    craw_hl_color(l, x0,     y0 + 1, top, c01);
    craw_hl_color(l, x0 + 1, y0 + 1, top, c11);

    for (int i = 0; i < 2; i++) {
        float a = c00[i] * (1.0f - fx) + c10[i] * fx;
        float b = c01[i] * (1.0f - fx) + c11[i] * fx;
        out[i] = a * (1.0f - fy) + b * fy;
    }
}

int craw_ref_highlights(const uint16_t *bayer, uint32_t width, uint32_t height,
                        size_t stride, const CRAWProfile *profile,
                        const float wb[3], float *out)
{
    CRAWHlLevel lv[16] = { 0 };
    int nb_levels = 0;
    float *uv = NULL, *av = NULL, *qa = NULL;
    const int w = (int)width, h = (int)height;
    const int qw = w / 2, qh = h / 2;
    int ret = 0;

    if (!bayer || !profile || !wb || !out)
        return CRAW_ERROR(EINVAL);
    if (profile->cfa != CRAW_CFA_RGGB)
        return CRAW_ERROR(ENOTSUP);
    if (width < 8 || height < 8 || (width & 1) || (height & 1) || stride < width)
        return CRAW_ERROR(EINVAL);
    for (int s = 0; s < 4; s++) {
        if (!(profile->white_level[s] > profile->black_level[s]))
            return CRAW_ERROR(EINVAL);
    }

    uv = malloc((size_t)w * h * sizeof(*uv));
    av = malloc((size_t)w * h * sizeof(*av));
    qa = malloc((size_t)qw * qh * sizeof(*qa));
    if (!uv || !av || !qa) {
        ret = CRAW_ERROR(ENOMEM);
        goto fail;
    }

    /* u and a at every site; x is u times the site's gain wherever it is
     * needed, as in the shaders. The range is applied as a reciprocal, as
     * the shaders do. */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int s = (y & 1) * 2 + (x & 1);
            float inv_range = 1.0f / (profile->white_level[s] - profile->black_level[s]);
            float raw = bayer[(size_t)y * stride + x] * (1.0f / 65535.0f);
            float u = CRAW_MAX((raw - profile->black_level[s]) * inv_range, 0.0f);

            uv[(size_t)y * w + x] = u;
            av[(size_t)y * w + x] = craw_hl_smoothstep(CRAW_HL_U_LO, CRAW_HL_U_HI, u);
        }
    }

    /* The largest a of every quad. */
    for (int qy = 0; qy < qh; qy++) {
        for (int qx = 0; qx < qw; qx++) {
            float amax = 0.0f;
            for (int s = 0; s < 4; s++)
                amax = CRAW_MAX(amax, av[(size_t)(2 * qy + (s >> 1)) * w + 2 * qx + (s & 1)]);
            qa[(size_t)qy * qw + qx] = amax;
        }
    }

    /* The pyramid's sizes, level 0 one texel per 4x4 block, down to 1x1. */
    {
        int lw = (w + 3) / 4, lh = (h + 3) / 4;

        while (1) {
            if (nb_levels == (int)CRAW_ARRAY_ELEMS(lv)) {
                ret = CRAW_ERROR(EINVAL);
                goto fail;
            }
            lv[nb_levels].w = lw;
            lv[nb_levels].h = lh;
            lv[nb_levels].t = calloc((size_t)lw * lh * 3, sizeof(float));
            if (!lv[nb_levels].t) {
                ret = CRAW_ERROR(ENOMEM);
                goto fail;
            }
            nb_levels++;
            if (lw == 1 && lh == 1)
                break;
            lw = (lw + 1) / 2;
            lh = (lh + 1) / 2;
        }
    }

    /* Level 0: each block's quads, top row first, left to right, each
     * judged clipped by the largest a over the quads within CRAW_HL_DILATE
     * of it. */
    for (int by = 0; by < lv[0].h; by++) {
        for (int bx = 0; bx < lv[0].w; bx++) {
            float acc[3] = { 0.0f, 0.0f, 0.0f };

            for (int j = 0; j < 2; j++) {
                for (int i = 0; i < 2; i++) {
                    int qx = 2 * bx + i, qy = 2 * by + j;
                    float amax = 0.0f, umax = 0.0f;
                    float rgb[3] = { 0.0f, 0.0f, 0.0f };
                    float m, wq, g, r, b;

                    if (qx >= qw || qy >= qh)
                        continue;

                    for (int dy = -CRAW_HL_DILATE; dy <= CRAW_HL_DILATE; dy++) {
                        for (int dx = -CRAW_HL_DILATE; dx <= CRAW_HL_DILATE; dx++) {
                            int nx = qx + dx, ny = qy + dy;
                            if (nx < 0 || ny < 0 || nx >= qw || ny >= qh)
                                continue;
                            amax = CRAW_MAX(amax, qa[(size_t)ny * qw + nx]);
                        }
                    }

                    for (int s = 0; s < 4; s++) {
                        size_t o = (size_t)(2 * qy + (s >> 1)) * w + 2 * qx + (s & 1);
                        int c = craw_hl_site_chan[s];

                        rgb[c] += uv[o] * wb[c];
                        umax = CRAW_MAX(umax, uv[o]);
                    }
                    rgb[1] *= 0.5f;

                    m = CRAW_MIN(umax, 1.0f);
                    wq = (1.0f - amax) * m * m;
                    g = CRAW_MAX(rgb[1], CRAW_HL_EPS);
                    r = CRAW_CLAMP(logf(CRAW_MAX(rgb[0], CRAW_HL_EPS) / g),
                                   -CRAW_HL_LOG_MAX, CRAW_HL_LOG_MAX);
                    b = CRAW_CLAMP(logf(CRAW_MAX(rgb[2], CRAW_HL_EPS) / g),
                                   -CRAW_HL_LOG_MAX, CRAW_HL_LOG_MAX);

                    acc[0] += wq * r;
                    acc[1] += wq * b;
                    acc[2] += wq;
                }
            }

            memcpy(&lv[0].t[((size_t)by * lv[0].w + bx) * 3], acc, sizeof(acc));
        }
    }

    /* Pull. */
    for (int l = 0; l + 1 < nb_levels; l++) {
        const CRAWHlLevel *f = &lv[l];
        CRAWHlLevel *c = &lv[l + 1];

        for (int y = 0; y < c->h; y++) {
            for (int x = 0; x < c->w; x++) {
                float acc[3] = { 0.0f, 0.0f, 0.0f };

                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        int fx = 2 * x + dx, fy = 2 * y + dy;
                        if (fx >= f->w || fy >= f->h)
                            continue;
                        for (int k = 0; k < 3; k++)
                            acc[k] += f->t[((size_t)fy * f->w + fx) * 3 + k];
                    }
                }

                memcpy(&c->t[((size_t)y * c->w + x) * 3], acc, sizeof(acc));
            }
        }
    }

    /* Push, from the level under the top down to level 0, each level
     * rewritten in place: a texel reads only itself at its own level. */
    for (int l = nb_levels - 2; l >= 0; l--) {
        CRAWHlLevel *f = &lv[l];
        const CRAWHlLevel *c = &lv[l + 1];
        bool top = l + 2 == nb_levels;

        for (int y = 0; y < f->h; y++) {
            for (int x = 0; x < f->w; x++) {
                float *t = &f->t[((size_t)y * f->w + x) * 3];
                float up[2];

                craw_hl_bilinear(c, ((float)x + 0.5f) * 0.5f - 0.5f,
                                 ((float)y + 0.5f) * 0.5f - 0.5f, top, up);

                if (t[2] > 0.0f) {
                    float s = CRAW_MIN(t[2] / CRAW_HL_W_SAT, 1.0f);
                    t[0] = up[0] + s * (t[0] / t[2] - up[0]);
                    t[1] = up[1] + s * (t[1] / t[2] - up[1]);
                } else {
                    t[0] = up[0];
                    t[1] = up[1];
                }
            }
        }
    }

    /* Fill. */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t o = (size_t)y * w + x;
            int c = craw_hl_site_chan[(y & 1) * 2 + (x & 1)];
            float xc = uv[o] * wb[c], a = av[o];
            float res = xc;

            if (a > 0.0f) {
                float sum[3] = { 0.0f, 0.0f, 0.0f }, n[3] = { 0.0f, 0.0f, 0.0f };
                float amax[3] = { 0.0f, 0.0f, 0.0f };
                float lf[2], lrho[3], a1, a2, l1, l2, lc, A, lfull, lpart, L;
                int k1, k2;

                craw_hl_bilinear(&lv[0], (float)(x >> 1) * 0.5f - 0.25f,
                                 (float)(y >> 1) * 0.5f - 0.25f, false, lf);
                lrho[0] = lf[0];
                lrho[1] = 0.0f;
                lrho[2] = lf[1];

                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = craw_hl_mirror(x + dx, w), ny = craw_hl_mirror(y + dy, h);
                        int k = craw_hl_site_chan[(ny & 1) * 2 + (nx & 1)];
                        size_t no = (size_t)ny * w + nx;

                        if (k == c)
                            continue;
                        sum[k] += uv[no] * wb[k];
                        n[k] += 1.0f;
                        amax[k] = CRAW_MAX(amax[k], av[no]);
                    }
                }

                k1 = c == 0 ? 1 : 0;
                k2 = c == 2 ? 1 : 2;
                a1 = 1.0f - amax[k1];
                a2 = 1.0f - amax[k2];
                l1 = logf(CRAW_MAX(sum[k1] / n[k1], CRAW_HL_EPS)) - lrho[k1];
                l2 = logf(CRAW_MAX(sum[k2] / n[k2], CRAW_HL_EPS)) - lrho[k2];
                lc = logf(CRAW_MAX(xc, CRAW_HL_EPS)) - lrho[c];

                A = a1 + a2;
                lfull = expf(CRAW_MAX(lc, CRAW_MAX(l1, l2)));
                lpart = A > 0.0f ? expf((a1 * l1 + a2 * l2) / A) : lfull;
                L = lfull + CRAW_MIN(A, 1.0f) * (lpart - lfull);

                res = xc + a * CRAW_MAX(L * expf(lrho[c]) - xc, 0.0f);
            }

            out[o] = craw_hl_to_half(res);
        }
    }

fail:
    for (int i = 0; i < nb_levels; i++)
        free(lv[i].t);
    free(uv);
    free(av);
    free(qa);
    return ret;
}
