/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* CPU reference of the defringe stage, DESIGN.md 19: the first pass's
 * per-pixel l, c and E, rounded to fp16 as the GPU stores them, then the
 * clamp at every pixel near a strong edge. lib/shaders/defringe*.comp.glsl
 * implement the same text and are held to this by tests/test_pipeline.c. */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ref.h"

/* DESIGN.md 19.2; lib/shaders/defringe.glsl carries the same numbers. */
#define CRAW_DF_YMIN   1e-3f
#define CRAW_DF_Y0     0.02f
#define CRAW_DF_E_LO   0.15f
#define CRAW_DF_E_HI   0.35f
#define CRAW_DF_B_LO   0.005f
#define CRAW_DF_B_HI   0.02f
#define CRAW_DF_D_MIN  2
#define CRAW_DF_D_MAX  6
#define CRAW_DF_DARK   0.1f
#define CRAW_DF_MARGIN 0.03f

static inline int craw_df_mirror(int v, int n)
{
    if (v < 0)
        return -v;
    if (v >= n)
        return 2 * n - 2 - v;
    return v;
}

static inline float craw_df_smoothstep(float e0, float e1, float v)
{
    float t = CRAW_CLAMP((v - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

static inline float craw_df_l(const float *px)
{
    float y = (px[0] + 2.0f * px[1] + px[2]) * 0.25f;
    return logf(CRAW_MAX(y, CRAW_DF_YMIN) + CRAW_DF_Y0);
}

typedef struct CRAWDfImage {
    const float *prep;   /* l, cr, cb, E per pixel */
    int w, h;
} CRAWDfImage;

static inline const float *craw_df_at(const CRAWDfImage *im, int x, int y)
{
    return &im->prep[((size_t)craw_df_mirror(y, im->h) * im->w + craw_df_mirror(x, im->w)) * 4];
}

/* The 3x3 mean of (Y, cr, cb) at a position between pixels, as the 4x4
 * around it weighted (1 - f, 1, 1, f) / 3 on each axis. */
static void craw_df_side(const CRAWDfImage *im, float px, float py, float out[3])
{
    float bx = floorf(px), by = floorf(py);
    float fx = px - bx, fy = py - by;
    int ox = (int)bx - 1, oy = (int)by - 1;
    const float wx[4] = { 1.0f - fx, 1.0f, 1.0f, fx };
    const float wy[4] = { 1.0f - fy, 1.0f, 1.0f, fy };

    out[0] = out[1] = out[2] = 0.0f;
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            const float *v = craw_df_at(im, ox + i, oy + j);
            float wgt = wx[i] * wy[j];

            out[0] += wgt * (expf(v[0]) - CRAW_DF_Y0);
            out[1] += wgt * v[1];
            out[2] += wgt * v[2];
        }
    }
    for (int k = 0; k < 3; k++)
        out[k] *= 1.0f / 9.0f;
}

int craw_ref_defringe(float *rgb, uint32_t width, uint32_t height, float strength)
{
    static const float kv[7] = { 1.0f, 3.0f, 4.0f, 4.0f, 4.0f, 3.0f, 1.0f };
    static const float kd[7] = { -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f };
    const int w = (int)width, h = (int)height;
    CRAWDfImage im;
    float *prep;

    if (!rgb || width < 8 || height < 8)
        return CRAW_ERROR(EINVAL);
    if (!(strength > 0.0f && strength <= 1.0f))
        return CRAW_ERROR(EINVAL);

    prep = malloc((size_t)w * h * 4 * sizeof(*prep));
    if (!prep)
        return CRAW_ERROR(ENOMEM);

    /* The first pass. */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            const float *px = &rgb[((size_t)y * w + x) * 4];
            float l[9], gx, gy, g;

            for (int j = 0; j < 3; j++)
                for (int i = 0; i < 3; i++)
                    l[j * 3 + i] = craw_df_l(&rgb[((size_t)craw_df_mirror(y + j - 1, h) * w +
                                                  craw_df_mirror(x + i - 1, w)) * 4]);

            gx = (l[2] + 2.0f * l[5] + l[8]) - (l[0] + 2.0f * l[3] + l[6]);
            gy = (l[6] + 2.0f * l[7] + l[8]) - (l[0] + 2.0f * l[1] + l[2]);
            g = CRAW_MAX(px[1], CRAW_DF_YMIN);

            float *o = &prep[((size_t)y * w + x) * 4];
            o[0] = craw_ref_to_half(l[4]);
            o[1] = craw_ref_to_half(logf(CRAW_MAX(px[0], CRAW_DF_YMIN) / g));
            o[2] = craw_ref_to_half(logf(CRAW_MAX(px[2], CRAW_DF_YMIN) / g));
            o[3] = craw_ref_to_half(sqrtf(gx * gx + gy * gy) * 0.125f);
        }
    }

    im = (CRAWDfImage) { .prep = prep, .w = w, .h = h };

    /* The second pass, in place: a pixel reads only its own RGB, and the
     * first pass's image around it. */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float en = 0.0f, yx = 0.0f, t, gxs = 0.0f, gys = 0.0f, gl, nx, ny;
            float s1[3], s2[3], ys, lo[2], hi[2], c[2], cl[2], cn[2], yv, er, eb, gn;
            bool ok1, ok2;
            float *px;

            for (int j = -2; j <= 2; j++)
                for (int i = -2; i <= 2; i++)
                    en = CRAW_MAX(en, craw_df_at(&im, x + i, y + j)[3]);
            if (en <= CRAW_DF_E_LO)
                continue;

            for (int j = -1; j <= 1; j++)
                for (int i = -1; i <= 1; i++)
                    yx = CRAW_MAX(yx, expf(craw_df_at(&im, x + i, y + j)[0]) - CRAW_DF_Y0);

            t = strength * craw_df_smoothstep(CRAW_DF_E_LO, CRAW_DF_E_HI, en) *
                craw_df_smoothstep(CRAW_DF_B_LO, CRAW_DF_B_HI, yx);
            if (t <= 0.0f)
                continue;

            for (int j = 0; j < 7; j++) {
                for (int i = 0; i < 7; i++) {
                    float l = craw_df_at(&im, x + i - 3, y + j - 3)[0];
                    gxs += l * (kd[i] * kv[j]);
                    gys += l * (kv[i] * kd[j]);
                }
            }
            gl = sqrtf(gxs * gxs + gys * gys);
            if (!(gl > 0.0f))
                continue;
            nx = gxs / gl;
            ny = gys / gl;

            /* Each side where the edge is weakest along the normal, 2 to 6
             * pixels out, the nearer one on a tie: past the fringe, which
             * must not vote on its own colour, but inside a leaf thinner
             * than that. */
            for (int s = 0; s < 2; s++) {
                float sgn = s ? -1.0f : 1.0f, best = INFINITY;
                int bk = CRAW_DF_D_MIN;

                for (int k = CRAW_DF_D_MIN; k <= CRAW_DF_D_MAX; k++) {
                    float e = craw_df_at(&im, (int)floorf((float)x + sgn * k * nx + 0.5f),
                                         (int)floorf((float)y + sgn * k * ny + 0.5f))[3];
                    if (e < best) {
                        best = e;
                        bk = k;
                    }
                }
                craw_df_side(&im, (float)x + sgn * bk * nx, (float)y + sgn * bk * ny, s ? s2 : s1);
            }
            ys = CRAW_MAX(s1[0], s2[0]);
            ok1 = s1[0] >= CRAW_DF_DARK * ys;
            ok2 = s2[0] >= CRAW_DF_DARK * ys;

            /* A side too dark to have a colour counts as neutral, the
             * colour of the light: a grey pixel beside a red bar stays
             * grey, a blue one beside a white lamp is pulled to white. */
            for (int k = 0; k < 2; k++) {
                float a = ok1 ? s1[k + 1] : 0.0f, b = ok2 ? s2[k + 1] : 0.0f;

                lo[k] = CRAW_MIN(a, b);
                hi[k] = CRAW_MAX(a, b);
                c[k] = craw_df_at(&im, x, y)[k + 1];
                cl[k] = CRAW_CLAMP(c[k], lo[k] - CRAW_DF_MARGIN, hi[k] + CRAW_DF_MARGIN);
            }
            if (cl[0] == c[0] && cl[1] == c[1])
                continue;

            for (int k = 0; k < 2; k++)
                cn[k] = c[k] + t * (cl[k] - c[k]);

            px = &rgb[((size_t)y * w + x) * 4];
            yv = (px[0] + 2.0f * px[1] + px[2]) * 0.25f;
            er = expf(cn[0]);
            eb = expf(cn[1]);
            gn = 4.0f * yv / (er + 2.0f + eb);
            px[0] = gn * er;
            px[1] = gn;
            px[2] = gn * eb;
        }
    }

    free(prep);
    return 0;
}
