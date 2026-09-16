/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#include <math.h>

#include <craw/utils.h>

#include "gamut.h"

/* ACES 1.3 Reference Gamut Compression.
 *
 * Transcribed from the Academy's own CTL,
 * transforms/ctl/lmt/LMT.Academy.ReferenceGamutCompress.ctl in ampas/aces-dev
 * at the v1.3 tag, functions compress() and gamut_compress(). OpenColorIO's
 * built-in transform ACES-GamutComp-v13 is the same operator with the same
 * seven numbers. Nothing here is a variant of it.
 *
 * The distance limits are how far outside AP1, measured from the achromatic
 * axis, the operator is asked to reach; the Academy picked them to cover the
 * encoding gamuts of the common digital cinema cameras. The thresholds are
 * the fraction of the core gamut left untouched, picked so that every patch
 * of the ColorChecker Classic passes through unchanged. */
#define CRAW_RGC_LIM_CYAN    1.147
#define CRAW_RGC_LIM_MAGENTA 1.264
#define CRAW_RGC_LIM_YELLOW  1.312
#define CRAW_RGC_THR_CYAN    0.815
#define CRAW_RGC_THR_MAGENTA 0.803
#define CRAW_RGC_THR_YELLOW  0.880
#define CRAW_RGC_PWR         1.2

/* The CTL's "scale factor for y = 1 intersect". Solving
 *
 *     thr + scl * nd / (1 + nd^pwr)^(1/pwr) = 1,  nd = (lim - thr) / scl
 *
 * for scl gives what the CTL writes inline, i.e. the scale that makes a
 * distance of exactly `lim` compress to exactly 1.0, which is the AP1 gamut
 * boundary: a colour that far out lands on the hull and anything nearer
 * lands inside it. Done in double here because it runs once per frame. */
static float craw_rgc_scale(double lim, double thr, double pwr)
{
    return (float)((lim - thr) / pow(pow((1.0 - thr) / (lim - thr), -pwr) - 1.0, 1.0 / pwr));
}

void craw_gamut_compress_params(CRAWGamutCompress *g)
{
    static const double lim[3] = {
        CRAW_RGC_LIM_CYAN, CRAW_RGC_LIM_MAGENTA, CRAW_RGC_LIM_YELLOW,
    };
    static const double thr[3] = {
        CRAW_RGC_THR_CYAN, CRAW_RGC_THR_MAGENTA, CRAW_RGC_THR_YELLOW,
    };

    for (int i = 0; i < 3; i++) {
        g->thr[i] = (float)thr[i];
        g->scl[i] = craw_rgc_scale(lim[i], thr[i], CRAW_RGC_PWR);
    }

    g->pwr = (float)CRAW_RGC_PWR;
    g->inv_pwr = (float)(1.0 / CRAW_RGC_PWR);
}

void craw_gamut_compress(const CRAWGamutCompress *g, float rgb[3])
{
    /* The achromatic axis, and the per-channel distance from it. A channel
     * below the achromatic value has a positive distance; the channel that
     * *is* the maximum has a distance of zero and never moves, so the
     * operator only ever pulls the two lesser channels up towards it and
     * cannot change which channel is largest. */
    float ach = CRAW_MAX(rgb[0], CRAW_MAX(rgb[1], rgb[2]));
    float mag;

    /* The CTL's own guard: pure black has no direction to be pulled in. */
    if (ach == 0.0f)
        return;

    mag = fabsf(ach);

    for (int i = 0; i < 3; i++) {
        float d = (ach - rgb[i]) / mag;
        float nd, p;

        /* Below the threshold nothing happens at all, which is what keeps
         * ordinary colours bit for bit what they were. */
        if (d < g->thr[i])
            continue;

        nd = (d - g->thr[i]) / g->scl[i];
        p = powf(nd, g->pwr);
        d = g->thr[i] + g->scl[i] * nd / powf(1.0f + p, g->inv_pwr);

        rgb[i] = ach - d * mag;
    }
}
