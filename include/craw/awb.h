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
#ifndef CRAW_AWB_H
#define CRAW_AWB_H

#include <stdint.h>
#include <stdbool.h>

#include <craw/profile.h>

/* Auto white balance.
 *
 * The GPU side reduces a frame to a two dimensional histogram of raw
 * chromaticity, the CPU side searches it for pixels that could be a
 * neutral surface, using the profile's white balance locus as the prior,
 * and a small temporal model turns per-frame estimates into a smooth
 * illuminant track. Nothing here touches pixels at full resolution on the
 * CPU. */

/* Histogram geometry used by the pipeline. */
#define CRAW_AWB_HIST_BINS     128
#define CRAW_AWB_HIST_LOG_MIN -2.5f
#define CRAW_AWB_HIST_LOG_MAX  1.0f

/* Histogram over ln(R/G) (x axis) and ln(B/G) (y axis) of the raw sensor
 * response, i.e. with the as-shot white balance divided back out. Clipped
 * and near-black samples are excluded. index = iy * bins + ix. */
typedef struct CRAWAwbHistogram {
    uint32_t bins;             /* per axis */
    float log_min, log_max;    /* both axes */
    const uint32_t *count;     /* number of samples per bin */
    const uint32_t *weight;    /* sum of min(g, 0.5) per bin, 16.16 fixed point */
    /* Sum of g^4 per bin, 16.16 fixed point. Emphasises the brightest
     * unclipped samples: light sources and specular glare, which carry the
     * illuminant's own colour and beat a dominant warm surface. */
    const uint32_t *weight_bright;
} CRAWAwbHistogram;

typedef struct CRAWAwbParams {
    /* Samples farther than this from the locus, in ln-ratio units, are not
     * neutral candidates. Default 0.08. */
    float band;
    /* Gaussian falloff of a candidate's weight with its locus distance,
     * ln-ratio units. Default 0.05. */
    float sigma;
    /* Below this share of total sample weight inside the band the estimate
     * is reported invalid. Default 0.02. */
    float min_zone_fraction;
    /* Warmth retention: 0.0 corrects fully to neutral; 1.0 balances any
     * illuminant warmer than ambience_knee_cct as if it were the knee, so
     * the warmth between the two stays in the picture (a 3200 K scene is
     * still corrected down from daylight to 4000 K, and only the last 63
     * mired are kept); values between blend the estimate towards the knee
     * linearly in Kelvin. Default 0.0. */
    float ambience;
    float ambience_knee_cct;   /* default 4000 */
    /* Hard limits on the reported cct; 0 means the profile's table range. */
    float cct_min, cct_max;
    /* Prefer the brightness-weighted candidate when it is valid: a light
     * source or glare in frame is a better neutral than any surface.
     * Default true. */
    bool prefer_bright;
    /* The bright candidate is only trusted when its samples agree along
     * the locus to within this, in mired. Default 60. */
    float bright_max_spread_mired;
} CRAWAwbParams;

CRAW_API void craw_awb_params_init(CRAWAwbParams *p);

/* One reading of the histogram under one weighting. */
typedef struct CRAWAwbCandidate {
    bool valid;
    CRAWIlluminant illuminant; /* before ambience and clamping */
    float in_zone_fraction;    /* share of this weighting's mass inside the band */
    float spread_mired;        /* weighted std deviation along the locus */
} CRAWAwbCandidate;

enum CRAWAwbSource {
    CRAW_AWB_SOURCE_NONE = 0,
    CRAW_AWB_SOURCE_ZONE,      /* the linearly weighted neutral candidates */
    CRAW_AWB_SOURCE_BRIGHT,    /* the g^4 weighted ones */
};

typedef struct CRAWAwbEstimate {
    bool valid;
    CRAWIlluminant illuminant; /* the chosen candidate after ambience and clamping */
    float confidence;          /* 0.0 to 1.0, from in_zone_fraction and spread */
    float in_zone_fraction;    /* of the chosen candidate */
    float spread_mired;        /* of the chosen candidate */
    enum CRAWAwbSource source;
    CRAWAwbCandidate zone;     /* both readings, for diagnostics and UIs */
    CRAWAwbCandidate bright;
} CRAWAwbEstimate;

/* Estimates the illuminant of one frame from its histogram. Requires a
 * profile with a white balance table. `params` may be NULL for defaults. */
CRAW_API int craw_awb_estimate(const CRAWProfile *p,
                               const CRAWAwbHistogram *hist,
                               const CRAWAwbParams *params,
                               CRAWAwbEstimate *out);

/* CPU histogram builder with the same semantics as the GPU pass, for tests
 * and tools. `rgb` is interleaved float camera RGB after white balance
 * with `wb_applied` and clipping at 1.0 (the pipeline's intermediate),
 * `stride` in floats between rows, `channels` 3 or 4. `count` and
 * `weight` and `weight_bright` must hold CRAW_AWB_HIST_BINS squared
 * entries each and are zeroed first. */
CRAW_API int craw_awb_histogram_from_rgb(const float *rgb, uint32_t width,
                                         uint32_t height, size_t stride,
                                         int channels, const float wb_applied[3],
                                         uint32_t *count, uint32_t *weight,
                                         uint32_t *weight_bright);

/* Temporal smoothing of estimates into an illuminant track. */
typedef struct CRAWAwbSmoother CRAWAwbSmoother;

typedef struct CRAWAwbSmootherParams {
    float time_constant;        /* seconds, exponential smoothing in mired; default 1.0 */
    float hysteresis_mired;     /* changes smaller than this are ignored; default 3.0 */
    float max_slew_mired_per_s; /* rate limit; default 30.0 */
    float min_confidence;       /* estimates below this only hold the track; default 0.2 */
} CRAWAwbSmootherParams;

CRAW_API void craw_awb_smoother_params_init(CRAWAwbSmootherParams *p);

CRAW_API int craw_awb_smoother_create(const CRAWAwbSmootherParams *params,
                                      CRAWAwbSmoother **sm);
CRAW_API void craw_awb_smoother_free(CRAWAwbSmoother **sm);

/* Forgets the track; the next push starts from its estimate. */
CRAW_API void craw_awb_smoother_reset(CRAWAwbSmoother *sm);

/* Feeds one frame's estimate at presentation time `pts_s` and returns the
 * smoothed illuminant to apply to that frame. `scene_cut` resets the track
 * to the new estimate immediately. Estimates flagged invalid or below
 * min_confidence hold the current track, and cannot start one or restart
 * it at a cut either: before any track exists such an estimate makes the
 * call return CRAW_ERROR(EAGAIN) and leave `out` untouched, so the caller
 * stays on what it had until a confident estimate arrives. */
CRAW_API int craw_awb_smoother_push(CRAWAwbSmoother *sm, double pts_s,
                                    const CRAWAwbEstimate *est, bool scene_cut,
                                    CRAWIlluminant *out);

#endif /* CRAW_AWB_H */
