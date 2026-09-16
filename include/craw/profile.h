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
#ifndef CRAW_PROFILE_H
#define CRAW_PROFILE_H

#include <stdint.h>
#include <stdbool.h>

#include <craw/utils.h>
#include <craw/noise.h>

/* Bayer colour filter array layout, named by the 2x2 quad read left to
 * right, top to bottom, starting at even (x, y). Only RGGB is implemented
 * in this version; the others are accepted by the profile but rejected by
 * craw_pipeline_create() with CRAW_ERROR(ENOTSUP). */
enum CRAWCfa {
    CRAW_CFA_RGGB = 0,
    CRAW_CFA_BGGR,
    CRAW_CFA_GRBG,
    CRAW_CFA_GBRG,
};

/* Quad site order used by all per-site arrays: index = (y & 1) * 2 + (x & 1). */
#define CRAW_SITE_TL 0
#define CRAW_SITE_TR 1
#define CRAW_SITE_BL 2
#define CRAW_SITE_BR 3

/* An illuminant as a camera profile parameterizes it.
 *
 * cct is a correlated colour temperature in Kelvin along the profile's own
 * white balance locus. tint is a signed offset perpendicular to that locus
 * in log raw-ratio space, i.e. the plane (ln R/G, ln B/G) of a neutral
 * surface's raw response, measured along the "green" direction
 * (-1, -1) / sqrt(2). A positive tint means the neutral sits on the green
 * side of the locus and both R and B gains are raised by exp(tint / sqrt(2))
 * to compensate, which is a magenta correction. +-0.1 is roughly a +-7%
 * change of R and B against G. */
typedef struct CRAWIlluminant {
    float cct;
    float tint;
} CRAWIlluminant;

#define CRAW_PROFILE_MAX_KNOTS 16

/* White balance gains for a neutral under an illuminant of `cct` Kelvin,
 * in camera channel order R, G, B. The green gain is normally 1.0. */
typedef struct CRAWWbKnot {
    float cct;
    float wb[3];
} CRAWWbKnot;

/* Camera RGB (already white balanced with the gains for the same cct) to
 * CIE 1931 XYZ relative to D65, row-major: out[i] = sum_j m[i][j] * in[j].
 * Rows therefore sum to the XYZ of D65 white. */
typedef struct CRAWMatrixKnot {
    float cct;
    float m[3][3];
} CRAWMatrixKnot;

/* Everything libcraw knows about a camera and a clip. Plain data: the
 * caller fills it from decoder side data (e.g. FFmpeg's AVRawColorParams
 * plus the container's per-CCT tables for ProRes RAW), a DNG-style
 * characterization, or its own measurements. Initialize with
 * craw_profile_init() first. */
typedef struct CRAWProfile {
    enum CRAWCfa cfa;

    /* Sensor black and white points per quad site, in the units of the
     * input image samples (UNORM, 0.0 to 1.0). White is the clip point. */
    float black_level[4];
    float white_level[4];

    /* As shot: what the camera itself applied to this clip or frame. */
    float as_shot_wb[3];          /* gains, R G B */
    float as_shot_matrix[3][3];   /* white-balanced camera RGB -> XYZ D65 */
    float as_shot_cct;            /* Kelvin, 0.0 if unknown */

    /* Post-matrix scene-linear scale that maps the sensor range to
     * reflectance with diffuse white at 1.0 (ProRes RAW "GainFactor").
     * 1.0 if the source has no such notion. */
    float gain;

    /* Exposure indices from the container. When both are > 0 the frame is
     * additionally scaled by exposure_index / as_shot_ei. */
    float as_shot_ei;
    float exposure_index;

    /* Per-CCT tables for re-balancing. Knots must be sorted by ascending
     * cct. Either table may be empty; see craw_profile_eval(). */
    CRAWWbKnot     wb_knots[CRAW_PROFILE_MAX_KNOTS];
    uint32_t       nb_wb_knots;
    CRAWMatrixKnot matrix_knots[CRAW_PROFILE_MAX_KNOTS];
    uint32_t       nb_matrix_knots;

    /* Free-form identification, e.g. "Panasonic DC-S1M2". */
    char name[64];

    /* Sensor noise model per exposure index, for the denoise stage. Knots
     * must be sorted by ascending iso; an empty table means the camera has
     * not been characterized and the caller has to supply a model itself.
     * See <craw/noise.h> and DESIGN.md 14.2. */
    CRAWNoiseKnot noise_knots[CRAW_PROFILE_MAX_NOISE_KNOTS];
    uint32_t      nb_noise_knots;
} CRAWProfile;

/* White balance selection for a frame. */
enum CRAWWbMode {
    /* Use as_shot_wb and as_shot_matrix verbatim. Matches what the camera
     * and the vendor's reference decoder produce by default. */
    CRAW_WB_AS_SHOT = 0,
    /* Derive gains and matrix from the profile tables at a chosen
     * illuminant (CRAWIlluminant). */
    CRAW_WB_REBALANCE,
};

/* The per-frame colour parameters the pipeline consumes, as evaluated from
 * a profile. */
typedef struct CRAWColorParams {
    float wb[3];               /* gains applied to raw R, G, B */
    float cam_to_xyz[3][3];    /* for camera RGB balanced with `wb` */
    float gain;                /* scene-linear scale incl. exposure index ratio */
} CRAWColorParams;

/* Sets a profile to identity: RGGB, black 0, white 1, unit gains, identity
 * matrix, gain 1, no tables. */
CRAW_API void craw_profile_init(CRAWProfile *p);

/* Checks levels, table ordering and matrix sanity. Returns 0 or a
 * negative error. */
CRAW_API int craw_profile_validate(const CRAWProfile *p);

/* Evaluates the colour parameters for a frame.
 *
 * CRAW_WB_AS_SHOT ignores `illum`.
 * CRAW_WB_REBALANCE interpolates both tables linearly in Kelvin at
 * illum->cct (clamped to the table range), then multiplies the R and B
 * gains by exp(illum->tint / sqrt(2)). Without a white balance table it
 * fails with CRAW_ERROR(ENOTSUP); without a matrix table it falls back to
 * as_shot_matrix.
 *
 * out->gain is p->gain times exposure_index / as_shot_ei when both are
 * known, else p->gain. */
CRAW_API int craw_profile_eval(const CRAWProfile *p, enum CRAWWbMode mode,
                               const CRAWIlluminant *illum,
                               CRAWColorParams *out);

/* Maps the raw response of a neutral surface (ratios R/G and B/G before
 * any white balance) to the nearest point of the profile's locus. Fills
 * out->cct with that point and out->tint with the signed residual.
 * Requires a white balance table with at least two knots. */
CRAW_API int craw_profile_neutral_to_illuminant(const CRAWProfile *p,
                                                float raw_rg, float raw_bg,
                                                CRAWIlluminant *out);

/* Inverse of the above: the raw ratios a neutral surface produces under
 * `illum`. */
CRAW_API int craw_profile_illuminant_to_neutral(const CRAWProfile *p,
                                                const CRAWIlluminant *illum,
                                                float *raw_rg, float *raw_bg);

/* Output colour spaces. All are linear light, scene referred, with 1.0 at
 * diffuse white. Spaces with a D65 white are reached by a plain XYZ to RGB
 * matrix; the ACES spaces include a Bradford adaptation from D65 (which is
 * what the camera matrix is relative to) to the ACES white point
 * (x 0.32168, y 0.33767).
 *
 * The two ACES spaces are the pair the specification intends:
 * CRAW_CS_ACES_AP0_LINEAR is ACES2065-1, the archival encoding, wide enough
 * that nothing the camera matrix produces falls outside it and therefore
 * never gamut compressed; CRAW_CS_ACES_AP1_LINEAR is ACEScg, the rendering
 * and display hand-off, which ends with the ACES 1.3 Reference Gamut
 * Compression unless the frame sets
 * CRAW_FRAME_FLAG_NO_GAMUT_COMPRESSION. See DESIGN.md 8 and 8.1. */
enum CRAWColorSpace {
    CRAW_CS_XYZ_D65 = 0,
    CRAW_CS_BT709_LINEAR,
    CRAW_CS_BT2020_LINEAR,
    CRAW_CS_P3D65_LINEAR,
    CRAW_CS_V_GAMUT_LINEAR,
    CRAW_CS_ACES_AP0_LINEAR,
    CRAW_CS_ACES_AP1_LINEAR,
    CRAW_CS_NB,
};

/* Evaluates the profile's noise table at `iso`, which is the sensor's own
 * exposure index: the as-shot ISO the frame was recorded at, not the
 * working index a grade rates the clip with. When `iso` is not positive
 * the profile's as_shot_ei is used. Fails with CRAW_ERROR(ENOTSUP) if the
 * profile carries no table. See craw_noise_model_eval() for the
 * interpolation. */
CRAW_API int craw_profile_noise_eval(const CRAWProfile *p, float iso,
                                     CRAWNoiseModel *out);

/* Fills `m` with the row-major matrix taking CIE XYZ (D65 relative) to
 * the given space. Identity for CRAW_CS_XYZ_D65. */
CRAW_API int craw_colorspace_from_xyz_d65(enum CRAWColorSpace cs, float m[3][3]);

/* The inverse: the row-major matrix taking the given space back to CIE XYZ
 * relative to D65. The two compose to the identity to within float
 * rounding, and chaining one with the other is how a caller converts
 * between two output spaces, e.g. the ACEScg the pipeline rendered into
 * back out to the ACES2065-1 an archive wants. */
CRAW_API int craw_colorspace_to_xyz_d65(enum CRAWColorSpace cs, float m[3][3]);

#endif /* CRAW_PROFILE_H */
