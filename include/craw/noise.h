/*
 * Copyright © 2026, Lynne
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef CRAW_NOISE_H
#define CRAW_NOISE_H

#include <stdint.h>
#include <stdbool.h>

#include <craw/utils.h>

struct CRAWProfile;

/* Sensor noise model and its calibration. See DESIGN.md section 14.
 *
 * The model is the usual affine one: the variance of a sensel is
 *
 *     var(u) = a * u + b
 *
 * with `u` the signal above black, normalized so that the profile's
 * black level is 0.0 and its white level 1.0, i.e. exactly
 * (raw - black_level[s]) / (white_level[s] - black_level[s]) before any
 * white balance gain. `a` is the shot term (Poisson, one unit of variance
 * per `a` of signal), `b` the signal independent read term. Both are in
 * that same normalized domain squared, so a profile's model does not
 * change when a decoder rescales the sample range.
 *
 * Because the model sits before the white balance gains, one pair (a, b)
 * describes all four CFA planes. A stage that filters after a gain g
 * scales it to (a * g, b * g * g). */
typedef struct CRAWNoiseModel {
    float a;
    float b;
} CRAWNoiseModel;

/* The model measured at one exposure index. Knots must be sorted by
 * ascending iso. */
typedef struct CRAWNoiseKnot {
    float iso;            /* the as-shot exposure index it was measured at */
    float a, b;
} CRAWNoiseKnot;

#define CRAW_PROFILE_MAX_NOISE_KNOTS 16

/* Evaluates a knot table at `iso`.
 *
 * Both a and b follow power laws in the sensor gain, so the interpolation
 * is linear in log(iso) of log(a) and log(b), i.e. a power law between
 * knots. Outside the table the model is carried on the way the gain
 * carries it: a in proportion to iso on both sides; b as iso squared above
 * the last knot, where the read noise ahead of the gain is nearly all of
 * it, and unchanged below the first, where the read noise after the gain
 * is most of it. A zero b is carried through as zero, and an iso that is
 * not positive and finite gets the first knot. See DESIGN.md 14.2.
 *
 * Fails with CRAW_ERROR(EINVAL) if the table is empty. */
CRAW_API int craw_noise_model_eval(const CRAWNoiseKnot *knots, uint32_t nb_knots,
                                   float iso, CRAWNoiseModel *out);

/* ------------------------------------------------------------ estimation */

/* Options for craw_noise_estimate(). Zero-initialize and call
 * craw_noise_estimate_params_init() for the defaults. */
typedef struct CRAWNoiseEstimateParams {
    /* Number of intensity bins, 8 to 512 (0 means 128). Bins are equally
     * spaced in sqrt(u), which puts more of them in the shadows where the
     * variance changes fastest. */
    uint32_t nb_bins;

    /* Bins holding fewer samples than this are dropped (0 means 20000). A
     * bin needs enough samples for its median absolute deviation to be
     * stable and for scene texture to be a minority. */
    uint32_t min_bin_samples;

    /* Samples whose local mean is above this are ignored (0 means 0.98):
     * near the clip point the distribution is truncated and the model does
     * not hold. */
    float max_level;

    /* Padding to allow for future options. Must always be set to 0. */
    uint8_t padding[64 - 4*3];
} CRAWNoiseEstimateParams;

/* What craw_noise_estimate() found, model plus the evidence for it. */
typedef struct CRAWNoiseEstimate {
    CRAWNoiseModel model;

    uint32_t nb_bins;       /* bins that had enough samples */
    uint32_t nb_inliers;    /* of those, bins the fit explains */

    /* RMS of ln(measured variance) - ln(modelled variance) over the
     * inliers, i.e. the fit's relative error. Below about 0.15 the model
     * describes the frame well; much above that the frame is textured
     * everywhere and the numbers are an upper bound, not a measurement. */
    float fit_rms;

    /* sqrt(b) expressed as 16-bit code values, which is the read noise in
     * the units a sensor is usually quoted in. At a low exposure index,
     * above roughly 10 the frame almost certainly has no region dark
     * enough to measure b and what was fitted is scene texture; the real
     * value grows with the index (10 at ISO 10000 on the S1M2, DESIGN.md
     * 14.2). Exactly 0 means the frame has no shadows to measure b in. */
    float read_noise_lsb;

    /* Padding to allow for future fields. */
    uint8_t padding[64 - 4*6];
} CRAWNoiseEstimate;

CRAW_API void craw_noise_estimate_params_init(CRAWNoiseEstimateParams *p);

/* Fits var = a * u + b to one Bayer frame.
 *
 * `bayer` is width*height 16-bit UNORM samples with `stride` samples per
 * row; `profile` supplies the per-site black and white levels and the CFA
 * phase. Both dimensions must be even and at least 32.
 *
 * The frame is split into its four CFA planes and each is high-pass
 * filtered with the 3x3 [1 -2 1; -2 4 -2; 1 -2 1] / 6 kernel, whose
 * response to any locally bilinear signal is zero and whose response to
 * white noise of variance V has variance V. Every sample is binned by a
 * 9x9 plane mean, which is exactly uncorrelated with that kernel, and each
 * bin's noise is the median absolute value of the residual over 0.6745.
 * The line through the bins is then fitted robustly: a least trimmed
 * squares seed over bin pairs, then iteratively reweighted Gauss-Newton on
 * ln(variance) with a Tukey biweight, which is what keeps bins full of
 * scene texture (always too noisy, never too quiet) from pulling the fit
 * up. See DESIGN.md 14.2.
 *
 * A flat, evenly lit field fills the bins with nothing but noise and gives
 * the exact model. An ordinary scene gives a good one as long as a
 * majority of the samples in a majority of the bins are smooth; check
 * out->fit_rms and out->read_noise_lsb before trusting it.
 *
 * Fails with CRAW_ERROR(ENOTSUP) for a non-RGGB phase and with
 * CRAW_ERROR(ENODATA) when too few bins survive. */
CRAW_API int craw_noise_estimate(const uint16_t *bayer, uint32_t width, uint32_t height,
                                 size_t stride, const struct CRAWProfile *profile,
                                 const CRAWNoiseEstimateParams *params,
                                 CRAWNoiseEstimate *out);

#endif /* CRAW_NOISE_H */
