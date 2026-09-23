/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ref.h"

/* Guard of the RCD ratios and direction weights, per DESIGN.md 5.3. Large
 * enough that a black region cannot turn a ratio into noise, small enough
 * to be invisible anywhere else. */
#define CRAW_RCD_EPS 1e-5f

/* Channel each quad site carries, RGGB: R, Gr, Gb, B. */
static const int craw_site_chan[4] = { 0, 1, 1, 2 };

/* Mirror without repeating the edge sample, per DESIGN.md section 3. The
 * loop only ever runs more than once on images smaller than the stencil;
 * because the frame dimensions are even, every reflection preserves the
 * parity of the coordinate and therefore the CFA phase. */
static inline int craw_mirror(int v, int n)
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

static inline float craw_fetch(const float *p, uint32_t w, uint32_t h,
                               int x, int y)
{
    return p[(size_t)craw_mirror(y, (int)h) * w + craw_mirror(x, (int)w)];
}

static void craw_ref_prep(const uint16_t *bayer, uint32_t width,
                          uint32_t height, size_t stride,
                          const CRAWProfile *p, const float wb[3], float *v)
{
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            int s = (int)((y & 1) * 2 + (x & 1));
            float raw = bayer[(size_t)y * stride + x] * (1.0f / 65535.0f);
            float t = (raw - p->black_level[s]) /
                      (p->white_level[s] - p->black_level[s]);

            t = CRAW_MAX(t, 0.0f);
            t *= wb[craw_site_chan[s]];

            /* Clipping after the white balance keeps a blown highlight
             * neutral instead of letting the channel with the largest gain
             * run away. The highlight reconstruction of DESIGN.md 18
             * replaces this whole function when it runs. */
            v[(size_t)y * width + x] = CRAW_MIN(t, 1.0f);
        }
    }
}

static void craw_ref_bilinear(const float *v, uint32_t width, uint32_t height,
                              float *rgb)
{
#define V(dx, dy) craw_fetch(v, width, height, x + (dx), y + (dy))
    for (int y = 0; y < (int)height; y++) {
        for (int x = 0; x < (int)width; x++) {
            float *o = &rgb[((size_t)y * width + x) * 4];
            int s = (y & 1) * 2 + (x & 1);
            float edge = 0.25f * (V(0, -1) + V(0, 1) + V(-1, 0) + V(1, 0));
            float corner = 0.25f * (V(-1, -1) + V(1, -1) + V(-1, 1) + V(1, 1));
            float horiz = 0.5f * (V(-1, 0) + V(1, 0));
            float vert = 0.5f * (V(0, -1) + V(0, 1));

            switch (s) {
            case 0:  /* R site */
                o[0] = V(0, 0); o[1] = edge;   o[2] = corner;
                break;
            case 1:  /* G in the R row */
                o[0] = horiz;   o[1] = V(0, 0); o[2] = vert;
                break;
            case 2:  /* G in the B row */
                o[0] = vert;    o[1] = V(0, 0); o[2] = horiz;
                break;
            default: /* B site */
                o[0] = corner;  o[1] = edge;   o[2] = V(0, 0);
                break;
            }
            o[3] = 1.0f;
        }
    }
#undef V
}

/* Malvar-He-Cutler, DESIGN.md 5.2. The four kernels below are the paper's,
 * written out as sums divided by eight. */

#define V(dx, dy) craw_fetch(v, width, height, x + (dx), y + (dy))

static float craw_mhc_green(const float *v, uint32_t width, uint32_t height,
                            int x, int y)
{
    return (4.0f * V(0, 0)
            + 2.0f * (V(0, -1) + V(0, 1) + V(-1, 0) + V(1, 0))
            - (V(0, -2) + V(0, 2) + V(-2, 0) + V(2, 0))) * 0.125f;
}

/* Wanted colour sits at the horizontal neighbours. */
static float craw_mhc_cross_h(const float *v, uint32_t width, uint32_t height,
                              int x, int y)
{
    return (5.0f * V(0, 0)
            + 4.0f * (V(-1, 0) + V(1, 0))
            - (V(-1, -1) + V(1, -1) + V(-1, 1) + V(1, 1) + V(-2, 0) + V(2, 0))
            + 0.5f * (V(0, -2) + V(0, 2))) * 0.125f;
}

/* The transpose of the above: wanted colour at the vertical neighbours. */
static float craw_mhc_cross_v(const float *v, uint32_t width, uint32_t height,
                              int x, int y)
{
    return (5.0f * V(0, 0)
            + 4.0f * (V(0, -1) + V(0, 1))
            - (V(-1, -1) + V(1, -1) + V(-1, 1) + V(1, 1) + V(0, -2) + V(0, 2))
            + 0.5f * (V(-2, 0) + V(2, 0))) * 0.125f;
}

/* Wanted colour sits at the diagonal neighbours. */
static float craw_mhc_diag(const float *v, uint32_t width, uint32_t height,
                           int x, int y)
{
    return (6.0f * V(0, 0)
            + 2.0f * (V(-1, -1) + V(1, -1) + V(-1, 1) + V(1, 1))
            - 1.5f * (V(0, -2) + V(0, 2) + V(-2, 0) + V(2, 0))) * 0.125f;
}

#undef V

static void craw_ref_mhc(const float *v, uint32_t width, uint32_t height,
                         float *rgb)
{
    for (int y = 0; y < (int)height; y++) {
        for (int x = 0; x < (int)width; x++) {
            float *o = &rgb[((size_t)y * width + x) * 4];
            int s = (y & 1) * 2 + (x & 1);
            float c = craw_fetch(v, width, height, x, y);

            switch (s) {
            case 0:  /* R site: green from the cross, blue from the diagonals */
                o[0] = c;
                o[1] = craw_mhc_green(v, width, height, x, y);
                o[2] = craw_mhc_diag(v, width, height, x, y);
                break;
            case 1:  /* G in the R row: R horizontal, B vertical */
                o[0] = craw_mhc_cross_h(v, width, height, x, y);
                o[1] = c;
                o[2] = craw_mhc_cross_v(v, width, height, x, y);
                break;
            case 2:  /* G in the B row: B horizontal, R vertical */
                o[0] = craw_mhc_cross_v(v, width, height, x, y);
                o[1] = c;
                o[2] = craw_mhc_cross_h(v, width, height, x, y);
                break;
            default: /* B site */
                o[0] = craw_mhc_diag(v, width, height, x, y);
                o[1] = craw_mhc_green(v, width, height, x, y);
                o[2] = c;
                break;
            }
            o[3] = 1.0f;
        }
    }
}

/* Mean of a 3x3 neighbourhood, mirrored. */
static float craw_box3(const float *s, uint32_t width, uint32_t height,
                       int x, int y)
{
    float acc = 0.0f;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++)
            acc += craw_fetch(s, width, height, x + dx, y + dy);
    }

    return acc * (1.0f / 9.0f);
}

/* RCD, DESIGN.md 5.3. Three passes, each one a full sweep of the image:
 * directions and low pass, green, then chroma. The chroma pass is split in
 * two because the green sites need the red and blue estimates of their
 * neighbouring red and blue sites, which the first half produces. */
static int craw_ref_rcd(const float *v, uint32_t width, uint32_t height,
                        float *rgb)
{
    size_t n = (size_t)width * height;
    float *lp = NULL, *wv = NULL, *wp = NULL, *gp = NULL, *cp = NULL;
    float *ga = NULL, *gb = NULL;
    int ret = 0;

    lp = malloc(n * sizeof(*lp));
    wv = malloc(n * sizeof(*wv));
    wp = malloc(n * sizeof(*wp));
    gp = malloc(n * sizeof(*gp));
    cp = malloc(n * sizeof(*cp));
    ga = malloc(n * sizeof(*ga));
    gb = malloc(n * sizeof(*gb));
    if (!lp || !wv || !wp || !gp || !cp || !ga || !gb) {
        ret = CRAW_ERROR(ENOMEM);
        goto fail;
    }

#define V(dx, dy) craw_fetch(v, width, height, x + (dx), y + (dy))
#define L(dx, dy) craw_fetch(lp, width, height, x + (dx), y + (dy))
#define G(dx, dy) craw_fetch(gp, width, height, x + (dx), y + (dy))
#define C(dx, dy) craw_fetch(cp, width, height, x + (dx), y + (dy))

/* The published RCD-LPF correction: how much brighter the low pass is here
 * than two samples away, as a signed fraction. */
#define RATIO(a, b) (((a) - (b)) / ((a) + (b) + CRAW_RCD_EPS))

    /* Pass A, low pass. Binomial rather than box so that the ratio the
     * green pass takes of it is not dominated by the CFA pattern itself. */
    for (int y = 0; y < (int)height; y++) {
        for (int x = 0; x < (int)width; x++) {
            lp[(size_t)y * width + x] =
                (4.0f * V(0, 0)
                 + 2.0f * (V(0, -1) + V(0, 1) + V(-1, 0) + V(1, 0))
                 + V(-1, -1) + V(1, -1) + V(-1, 1) + V(1, 1)) * (1.0f / 16.0f);
        }
    }

    /* Pass A, vertical and horizontal discrimination. */
    for (int y = 0; y < (int)height; y++) {
        for (int x = 0; x < (int)width; x++) {
            size_t i = (size_t)y * width + x;

            ga[i] = fabsf(V(0, -1) - V(0, 1))
                  + fabsf(2.0f * V(0, 0) - V(0, -2) - V(0, 2));
            gb[i] = fabsf(V(-1, 0) - V(1, 0))
                  + fabsf(2.0f * V(0, 0) - V(-2, 0) - V(2, 0));
        }
    }
    for (int y = 0; y < (int)height; y++) {
        for (int x = 0; x < (int)width; x++) {
            float vs = craw_box3(ga, width, height, x, y);
            float hs = craw_box3(gb, width, height, x, y);

            /* More horizontal detail means the vertical estimate is the
             * one that does not cross an edge. */
            wv[(size_t)y * width + x] = hs / (hs + vs + CRAW_RCD_EPS);
        }
    }

    /* Pass A, diagonal discrimination: P is NE-SW, Q is NW-SE. */
    for (int y = 0; y < (int)height; y++) {
        for (int x = 0; x < (int)width; x++) {
            size_t i = (size_t)y * width + x;

            ga[i] = fabsf(V(-1, 1) - V(1, -1))
                  + fabsf(2.0f * V(0, 0) - V(-2, 2) - V(2, -2));
            gb[i] = fabsf(V(-1, -1) - V(1, 1))
                  + fabsf(2.0f * V(0, 0) - V(-2, -2) - V(2, 2));
        }
    }
    for (int y = 0; y < (int)height; y++) {
        for (int x = 0; x < (int)width; x++) {
            float ps = craw_box3(ga, width, height, x, y);
            float qs = craw_box3(gb, width, height, x, y);

            wp[(size_t)y * width + x] = qs / (ps + qs + CRAW_RCD_EPS);
        }
    }

    free(ga);
    free(gb);
    ga = gb = NULL;

    /* Pass B, green. At a red or blue site the two neighbouring greens are
     * corrected by the ratio of the low pass across them, which is what
     * keeps a ramp or an edge from flattening out. */
    for (int y = 0; y < (int)height; y++) {
        for (int x = 0; x < (int)width; x++) {
            size_t i = (size_t)y * width + x;
            int s = (y & 1) * 2 + (x & 1);
            float l00, lu, ld, ll, lr, gv, gh, w;

            if (s == 1 || s == 2) {
                gp[i] = v[i];
                continue;
            }

            l00 = L(0, 0);
            lu = L(0, -2); ld = L(0, 2);
            ll = L(-2, 0); lr = L(2, 0);

            gv = 0.5f * (V(0, -1) * (1.0f + RATIO(l00, lu))
                       + V(0,  1) * (1.0f + RATIO(l00, ld)));
            gh = 0.5f * (V(-1, 0) * (1.0f + RATIO(l00, ll))
                       + V( 1, 0) * (1.0f + RATIO(l00, lr)));

            w = wv[i];
            gp[i] = w * gv + (1.0f - w) * gh;
        }
    }

    /* Pass C, first half: the missing colour at the red and blue sites,
     * from the colour differences of the four diagonal neighbours. */
    for (int y = 0; y < (int)height; y++) {
        for (int x = 0; x < (int)width; x++) {
            size_t i = (size_t)y * width + x;
            int s = (y & 1) * 2 + (x & 1);
            float g00, dsw, dne, dnw, dse, c, w;
            float *o = &rgb[i * 4];

            if (s == 1 || s == 2)
                continue;

            g00 = gp[i];
            dsw = V(-1,  1) - G(-1,  1);
            dne = V( 1, -1) - G( 1, -1);
            dnw = V(-1, -1) - G(-1, -1);
            dse = V( 1,  1) - G( 1,  1);

            w = wp[i];
            c = w * (g00 + 0.5f * (dsw + dne))
              + (1.0f - w) * (g00 + 0.5f * (dnw + dse));
            cp[i] = c;

            o[0] = s == 0 ? v[i] : c;
            o[1] = g00;
            o[2] = s == 0 ? c : v[i];
            o[3] = 1.0f;
        }
    }

    /* Pass C, second half: both missing colours at the green sites. One of
     * them has its native sites on the horizontal neighbours and the other
     * on the vertical ones; the estimate in the direction without native
     * sites uses the values the first half wrote there. */
    for (int y = 0; y < (int)height; y++) {
        for (int x = 0; x < (int)width; x++) {
            size_t i = (size_t)y * width + x;
            int s = (y & 1) * 2 + (x & 1);
            float g00, dh, dv, eh, ev, ch, cv, w;
            float *o = &rgb[i * 4];

            if (s == 0 || s == 3)
                continue;

            g00 = gp[i];
            w = wv[i];

            dh = 0.5f * ((V(-1, 0) - G(-1, 0)) + (V(1, 0) - G(1, 0)));
            ev = 0.5f * ((C(0, -1) - G(0, -1)) + (C(0, 1) - G(0, 1)));
            ch = w * (g00 + ev) + (1.0f - w) * (g00 + dh);

            dv = 0.5f * ((V(0, -1) - G(0, -1)) + (V(0, 1) - G(0, 1)));
            eh = 0.5f * ((C(-1, 0) - G(-1, 0)) + (C(1, 0) - G(1, 0)));
            cv = w * (g00 + dv) + (1.0f - w) * (g00 + eh);

            /* Site 1 sits in a red row, site 2 in a blue one. */
            o[0] = s == 1 ? ch : cv;
            o[1] = v[i];
            o[2] = s == 1 ? cv : ch;
            o[3] = 1.0f;
        }
    }

#undef V
#undef L
#undef G
#undef C
#undef RATIO

fail:
    free(lp);
    free(wv);
    free(wp);
    free(gp);
    free(cp);
    free(ga);
    free(gb);

    return ret;
}

int craw_ref_debayer_wb(const uint16_t *bayer, uint32_t width, uint32_t height,
                        size_t stride, const CRAWProfile *profile,
                        const float wb[3], enum CRAWDebayer method, float *rgb)
{
    float *v = NULL;
    int ret;

    if (!bayer || !profile || !wb || !rgb)
        return CRAW_ERROR(EINVAL);
    if ((unsigned)method >= CRAW_DEBAYER_NB)
        return CRAW_ERROR(EINVAL);
    if (profile->cfa != CRAW_CFA_RGGB)
        return CRAW_ERROR(ENOTSUP);
    /* Odd dimensions would make the mirrored borders flip the CFA phase. */
    if (width < 2 || height < 2 || (width & 1) || (height & 1))
        return CRAW_ERROR(EINVAL);
    if (stride < width)
        return CRAW_ERROR(EINVAL);

    for (int s = 0; s < 4; s++) {
        if (!(profile->white_level[s] > profile->black_level[s]))
            return CRAW_ERROR(EINVAL);
    }

    v = malloc((size_t)width * height * sizeof(*v));
    if (!v)
        return CRAW_ERROR(ENOMEM);

    craw_ref_prep(bayer, width, height, stride, profile, wb, v);

    ret = craw_ref_debayer_mosaic(v, width, height, method, rgb);

    free(v);

    return ret;
}

int craw_ref_debayer_mosaic(const float *v, uint32_t width, uint32_t height,
                            enum CRAWDebayer method, float *rgb)
{
    if (!v || !rgb)
        return CRAW_ERROR(EINVAL);
    if (width < 2 || height < 2 || (width & 1) || (height & 1))
        return CRAW_ERROR(EINVAL);

    switch (method) {
    case CRAW_DEBAYER_BILINEAR:
        craw_ref_bilinear(v, width, height, rgb);
        return 0;
    case CRAW_DEBAYER_MHC:
        craw_ref_mhc(v, width, height, rgb);
        return 0;
    case CRAW_DEBAYER_RCD:
        return craw_ref_rcd(v, width, height, rgb);
    default:
        return CRAW_ERROR(EINVAL);
    }
}

int craw_ref_debayer(const uint16_t *bayer, uint32_t width, uint32_t height,
                     size_t stride, const CRAWProfile *profile,
                     enum CRAWDebayer method, float *rgb)
{
    CRAWColorParams as_shot;
    int ret;

    if (!profile)
        return CRAW_ERROR(EINVAL);

    ret = craw_profile_eval(profile, CRAW_WB_AS_SHOT, NULL, &as_shot);
    if (ret < 0)
        return ret;

    return craw_ref_debayer_wb(bayer, width, height, stride, profile, as_shot.wb,
                               method, rgb);
}
