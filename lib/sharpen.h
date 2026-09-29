/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#ifndef CRAW_SHARPEN_H
#define CRAW_SHARPEN_H

/* Capture sharpening, DESIGN.md 20: the constants of 20.2 and the two
 * things the host works out per frame, the Gaussian's taps and the noise
 * of the luma, shared by lib/pipeline.c and the CPU reference in
 * lib/ref/ref_sharpen.c so that the two cannot drift apart.
 * lib/shaders/sharpen.glsl carries the same numbers. */

#include <math.h>
#include <stdint.h>

#include <craw/pipeline.h>
#include <craw/noise.h>

/* The widest tap reach, ceil(3 * CRAW_SHARPEN_MAX_RADIUS): the shader's
 * tile is sized for it. An iteration's estimate is read twice that far
 * around a pixel, which one reflection of the mirror of DESIGN.md 3 reaches
 * on a frame at least this wide and high. */
#define CRAW_SH_RMAX     6
#define CRAW_SH_MIN_SIZE (2 * CRAW_SH_RMAX + 2)

/* Added to the luma before the deconvolution, and the floor under the
 * estimate: the smallest normal fp16 value, so that everything the planes
 * hold keeps fp16's full relative precision and no store can flush to
 * zero, and about the read noise of this camera (DESIGN.md 14.2). */
#define CRAW_SH_PEDESTAL 6.103515625e-5f

/* The contrast statistic: squared central differences averaged over the
 * (2 * CRAW_SH_MASK_R + 1)^2 around the pixel. The mask needs the luma
 * CRAW_SH_MASK_R + 1 around it. */
#define CRAW_SH_MASK_R 2

/* What the statistic above reads on noise, over the variance of one
 * sensel of each colour averaged over the quad: the debayer's luma is
 * correlated noise, of which the central differences see only part. The
 * mean over a flat patch of RCD's output of P1000387, whose lower quartile
 * matches, level by level from 5e-4 to 2e-2 of the clip, that of three
 * whole ISO 1000 frames. DESIGN.md 20.2. */
#define CRAW_SH_KAPPA 0.5f

/* The local range clamp's radius. */
#define CRAW_SH_RANGE_R 1

/* The bound on one iteration's multiplicative correction. The estimate is
 * also held at or above the pedestal, which keeps every blur of it
 * positive and every value in fp16's normal range. */
#define CRAW_SH_CORR_MAX 2.0f

/* The Gaussian of standard deviation `sigma` integrated over each pixel's
 * footprint, taps[0] the centre and taps[i] at +-i, cut at ceil(3 sigma)
 * and renormalized so that the 1D kernel sums to exactly one in the order
 * the shader sums it. Returns the reach R; `sigma` must be in
 * (0, CRAW_SHARPEN_MAX_RADIUS]. */
static inline int craw_sharpen_kernel(float sigma, float taps[CRAW_SH_RMAX + 1])
{
    double t[CRAW_SH_RMAX + 1], sum;
    int r = (int)ceil(3.0 * (double)sigma);

    if (r < 1)
        r = 1;
    if (r > CRAW_SH_RMAX)
        r = CRAW_SH_RMAX;

    for (int i = 0; i <= r; i++) {
        double lo = ((double)i - 0.5) / ((double)sigma * sqrt(2.0));
        double hi = ((double)i + 0.5) / ((double)sigma * sqrt(2.0));
        t[i] = 0.5 * (erf(hi) - erf(lo));
    }

    sum = t[0];
    for (int i = 1; i <= r; i++)
        sum += 2.0 * t[i];

    for (int i = 0; i <= CRAW_SH_RMAX; i++)
        taps[i] = i <= r ? (float)(t[i] / sum) : 0.0f;

    return r;
}

/* The modelled variance of the contrast statistic at a luma L,
 * n_a * L + n_b, from the frame's sensor model (a, b) and its white
 * balance gains: a sensel of each colour averaged over the quad, in the
 * balanced units of the intermediate, times CRAW_SH_KAPPA. */
static inline void craw_sharpen_noise(const CRAWNoiseModel *nm, const float wb[3],
                                      float *n_a, float *n_b)
{
    *n_a = CRAW_SH_KAPPA * nm->a * (wb[0] + 2.0f * wb[1] + wb[2]) * 0.25f;
    *n_b = CRAW_SH_KAPPA * fmaxf(nm->b, 0.0f) *
           (wb[0] * wb[0] + 2.0f * wb[1] * wb[1] + wb[2] * wb[2]) * 0.25f;
}

#endif /* CRAW_SHARPEN_H */
