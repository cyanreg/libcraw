/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* CPU reference of the false colour suppression stage, DESIGN.md 17. The
 * definition is sequential: each pass is a 3x3 median sweep and then a
 * 3x3 box blur sweep over the two colour differences, every sweep reading
 * the whole result of the one before it, borders mirrored.
 * lib/shaders/false_color.comp.glsl fuses the sweeps into one dispatch and
 * is held to this by tests/test_pipeline.c. */

#include <stdlib.h>
#include <string.h>

#include "ref.h"

/* DESIGN.md 3: one reflection, without repeating the edge sample. */
static inline int craw_ref_fc_mirror(int v, int n)
{
    if (v < 0)
        return -v;
    if (v >= n)
        return 2 * n - 2 - v;
    return v;
}

/* The median by an insertion sort of the nine: a selection, so the result
 * is one of the inputs bit for bit, exactly as the shader's network. */
static float craw_ref_median9(float v[9])
{
    for (int i = 1; i < 9; i++) {
        float x = v[i];
        int j = i - 1;

        while (j >= 0 && v[j] > x) {
            v[j + 1] = v[j];
            j--;
        }
        v[j + 1] = x;
    }

    return v[4];
}

/* The box blur: the nine summed in stencil order, top row left to right
 * first, then scaled by the float nearest 1/9, exactly as the shader
 * sums them, so the two agree bit for bit on equal inputs. */
static float craw_ref_mean9(const float v[9])
{
    float s = v[0];

    for (int i = 1; i < 9; i++)
        s += v[i];

    return s * (1.0f / 9.0f);
}

int craw_ref_false_color(float *rgb, uint32_t width, uint32_t height, uint32_t passes)
{
    float *cur = NULL, *nxt = NULL;
    size_t n;
    int w, h;
    int ret = 0;

    if (!rgb || width < 2 || height < 2)
        return CRAW_ERROR(EINVAL);
    if (passes < 1 || passes > CRAW_FALSE_COLOR_MAX_PASSES)
        return CRAW_ERROR(EINVAL);

    w = (int)width;
    h = (int)height;
    n = (size_t)width * height;

    /* The two differences of every pixel, interleaved, twice over: a
     * median cannot run in place. */
    cur = malloc(n * 2 * sizeof(*cur));
    nxt = malloc(n * 2 * sizeof(*nxt));
    if (!cur || !nxt) {
        ret = CRAW_ERROR(ENOMEM);
        goto fail;
    }

    for (size_t i = 0; i < n; i++) {
        cur[2 * i + 0] = rgb[4 * i + 0] - rgb[4 * i + 1];
        cur[2 * i + 1] = rgb[4 * i + 2] - rgb[4 * i + 1];
    }

    /* Sweep sw of 2 * passes: the median of a pass when sw is even, its
     * blur when sw is odd. */
    for (uint32_t sw = 0; sw < 2 * passes; sw++) {
        const bool blur = (sw & 1) != 0;
        float *t;

        for (int y = 0; y < h; y++) {
            const int ys[3] = { craw_ref_fc_mirror(y - 1, h), y, craw_ref_fc_mirror(y + 1, h) };

            for (int x = 0; x < w; x++) {
                const int xs[3] = { craw_ref_fc_mirror(x - 1, w), x, craw_ref_fc_mirror(x + 1, w) };
                float v[2][9];

                for (int j = 0; j < 3; j++) {
                    for (int i = 0; i < 3; i++) {
                        const float *s = &cur[2 * ((size_t)ys[j] * w + xs[i])];
                        v[0][j * 3 + i] = s[0];
                        v[1][j * 3 + i] = s[1];
                    }
                }

                if (blur) {
                    nxt[2 * ((size_t)y * w + x) + 0] = craw_ref_mean9(v[0]);
                    nxt[2 * ((size_t)y * w + x) + 1] = craw_ref_mean9(v[1]);
                } else {
                    nxt[2 * ((size_t)y * w + x) + 0] = craw_ref_median9(v[0]);
                    nxt[2 * ((size_t)y * w + x) + 1] = craw_ref_median9(v[1]);
                }
            }
        }

        t = cur;
        cur = nxt;
        nxt = t;
    }

    for (size_t i = 0; i < n; i++) {
        float g = rgb[4 * i + 1];

        rgb[4 * i + 0] = g + cur[2 * i + 0];
        rgb[4 * i + 2] = g + cur[2 * i + 1];
    }

fail:
    free(nxt);
    free(cur);

    return ret;
}
