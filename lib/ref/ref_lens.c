/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* CPU reference of the lens correction warp, DESIGN.md 16. Same geometry,
 * same Catmull-Rom weights, same border rule and the same order of
 * operations as lib/shaders/lens.comp.glsl, in fp32 throughout: what
 * separates the two on a real frame is the fp16 storage of the image the
 * warp reads, not the arithmetic. The keep-everything mode of 16.8 is here
 * too, scale and black border both, and it gets its scale from the same
 * craw_lens_keep_all_scale() the pipeline calls. */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ref.h"

/* Catmull-Rom weights for taps at -1, 0, +1, +2 around a sample at
 * fractional position `t`. Written the way the shader writes them so that
 * the two round the same way. */
static void craw_ref_catrom(float t, float w[4])
{
    float t2 = t * t;
    float t3 = t2 * t;

    w[0] = -0.5f * t3 + 1.0f * t2 - 0.5f * t;
    w[1] =  1.5f * t3 - 2.5f * t2          + 1.0f;
    w[2] = -1.5f * t3 + 2.0f * t2 + 0.5f * t;
    w[3] =  0.5f * t3 - 0.5f * t2;
}

/* The mirror of DESIGN.md 3, one reflection. */
static int craw_ref_mirror(int i, int n)
{
    if (i < 0)
        i = -i;
    if (i >= n)
        i = 2 * n - 2 - i;

    return i;
}

int craw_ref_lens(const float *src, uint32_t width, uint32_t height,
                  const CRAWLensCorrection *lc, bool keep_all, float *dst)
{
    float cx, cy, hx, hy, half_diag, inv_hd, inv_scale = 1.0f;

    if (!src || !dst || !lc || !width || !height || src == dst)
        return CRAW_ERROR(EINVAL);

    if (!lc->valid) {
        /* The documented way to leave a frame alone. The pipeline skips the
         * pass outright for this; here the frames have to come out equal,
         * so it is a copy. */
        memcpy(dst, src, (size_t)width * height * 4 * sizeof(*dst));
        return 0;
    }

    if (lc->centre[0] < 0.0f || lc->centre[0] > 1.0f ||
        lc->centre[1] < 0.0f || lc->centre[1] > 1.0f)
        return CRAW_ERROR(EINVAL);

    cx = lc->centre[0] * (float)width;
    cy = lc->centre[1] * (float)height;
    hx = CRAW_MAX(lc->centre[0], 1.0f - lc->centre[0]) * (float)width;
    hy = CRAW_MAX(lc->centre[1], 1.0f - lc->centre[1]) * (float)height;
    half_diag = sqrtf(hx * hx + hy * hy);
    if (!(half_diag > 0.0f))
        return CRAW_ERROR(EINVAL);
    inv_hd = 1.0f / half_diag;

    /* DESIGN.md 16.8. The same library call the pipeline makes, so that the
     * two cannot answer differently; filling the frame is a scale of
     * exactly one and a multiply by one changes no bits. */
    if (keep_all) {
        float scale;
        int ret = craw_lens_keep_all_scale(lc, width, height, &scale);

        if (ret < 0)
            return ret;
        inv_scale = 1.0f / scale;
    }

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            float *out = &dst[((size_t)y * width + x) * 4];
            float dx = ((float)x + 0.5f - cx) * inv_scale;
            float dy = ((float)y + 0.5f - cy) * inv_scale;
            float r2 = (dx * dx + dy * dy) * (inv_hd * inv_hd);
            float px[3], py[3];
            bool inside = true;

            for (int c = 0; c < 3; c++) {
                const float *k = lc->k[c];
                float poly = k[0] + r2 * (k[1] + r2 * (k[2] + r2 * k[3]));

                px[c] = cx + dx * poly;
                py[c] = cy + dy * poly;

                /* One boundary for all three channels, so the border is one
                 * shape and not three coloured ones. */
                inside = inside && px[c] >= 0.0f && px[c] <= (float)width &&
                                   py[c] >= 0.0f && py[c] <= (float)height;
            }

            if (keep_all && !inside) {
                /* Black, not transparent: alpha is still the output pixel's
                 * own, and the colour stage writes 1.0 over it. */
                out[0] = out[1] = out[2] = 0.0f;
                out[3] = src[((size_t)y * width + x) * 4 + 3];
                continue;
            }

            for (int c = 0; c < 3; c++) {
                float sx = px[c] - 0.5f;
                float sy = py[c] - 0.5f;
                int bx = (int)floorf(sx);
                int by = (int)floorf(sy);
                float wx[4], wy[4];
                float acc = 0.0f;

                craw_ref_catrom(sx - (float)bx, wx);
                craw_ref_catrom(sy - (float)by, wy);

                for (int j = 0; j < 4; j++) {
                    int qy = craw_ref_mirror(by + j - 1, (int)height);
                    float row = 0.0f;

                    for (int i = 0; i < 4; i++) {
                        int qx = craw_ref_mirror(bx + i - 1, (int)width);

                        row += wx[i] * src[((size_t)qy * width + qx) * 4 + c];
                    }

                    acc += wy[j] * row;
                }

                out[c] = acc;
            }

            /* Alpha is not warped: it comes from the output pixel's own
             * position rather than being the sum of sixteen weights. */
            out[3] = src[((size_t)y * width + x) * 4 + 3];
        }
    }

    return 0;
}
