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
#ifndef CRAW_PIPELINE_H
#define CRAW_PIPELINE_H

#include <craw/vulkan.h>
#include <craw/profile.h>
#include <craw/awb.h>
#include <craw/lens.h>

typedef struct CRAWPipeline CRAWPipeline;

/* Debayering methods. See DESIGN.md for the exact definitions; the CPU
 * reference and the shaders implement the same text. */
enum CRAWDebayer {
    CRAW_DEBAYER_BILINEAR = 0,  /* 3x3 averages, fastest */
    CRAW_DEBAYER_MHC,           /* Malvar-He-Cutler 5x5 gradient corrected */
    CRAW_DEBAYER_RCD,           /* ratio-corrected directional, see DESIGN.md */
    CRAW_DEBAYER_NB,
};

/* Build an auto white balance histogram every frame; read it back with
 * craw_pipeline_get_awb_histogram(). */
#define CRAW_PIPELINE_FLAG_AWB_HISTOGRAM (1 << 0)

/* Build the raw-domain denoise pass. It costs an extra full resolution
 * R16_UNORM image and, at full strength, a few milliseconds a frame, so it
 * is only created when asked for. Once created it is switched per frame
 * with CRAWFrameParams.denoise_strength, which costs nothing when zero.
 * See DESIGN.md 14. */
#define CRAW_PIPELINE_FLAG_DENOISE (1 << 1)

/* Build the lens correction pass. It costs an extra full resolution image
 * in the pipeline's output format, because the colour stage then writes
 * that instead of the caller's destination and the warp reads it. Once
 * created it is switched per frame with CRAWFrameParams.lens: a correction
 * with valid = false skips the warp entirely and the colour stage writes
 * the caller's destination as it always did, so such a frame is bit for
 * bit what a pipeline built without the flag produces. See DESIGN.md 16. */
#define CRAW_PIPELINE_FLAG_LENS (1 << 2)

/* Build the false colour suppression pass, DESIGN.md 17: a 3x3 median of
 * the colour differences of the debayered intermediate, run up to
 * CRAW_FALSE_COLOR_MAX_PASSES times, which turns the coloured dots that a
 * glint smaller than a CFA quad leaves into neutral ones and touches
 * nothing else. With the RCD debayer it costs no memory: the stage writes
 * RCD's own auxiliary image, which is free by then. With bilinear or MHC
 * it costs an extra full resolution RGBA16F image. Once built it is
 * switched per frame with CRAWFrameParams.false_color_passes, which costs
 * nothing when zero. */
#define CRAW_PIPELINE_FLAG_FALSE_COLOR (1 << 3)

/* The most passes the stage's fused dispatch carries an apron for, and the
 * most any debayer needs: four clear the blob every method makes of a
 * single site glint (bilinear and RCD in three, MHC in four), and on real
 * footage the median has converged by then. DESIGN.md 17.2. */
#define CRAW_FALSE_COLOR_MAX_PASSES 4

/* Fixed configuration of a pipeline. Changing any of it means creating a
 * new pipeline. */
typedef struct CRAWPipelineParams {
    uint32_t width, height;     /* Bayer frame size; both must be even */
    enum CRAWCfa cfa;           /* this version: CRAW_CFA_RGGB only */
    enum CRAWDebayer debayer;

    /* The default output colour space, which any frame can override for
     * itself with CRAWFrameParams.out_space. */
    enum CRAWColorSpace out_space;

    /* VK_FORMAT_R16G16B16A16_SFLOAT or VK_FORMAT_R32G32B32A32_SFLOAT. The
     * output is linear light with values above 1.0 in the highlights and
     * possibly below 0.0 outside the camera's characterized gamut; alpha
     * is written as 1.0. */
    VkFormat out_format;

    uint32_t flags;             /* CRAW_PIPELINE_FLAG_* */

    /* Number of host-visible copies of the auto white balance histogram
     * kept in rotation, 1 to 4 (0 means 1). Each recording writes the next
     * slot, so with n slots the histogram of a frame stays readable while
     * up to n - 1 newer frames are still in flight: a player that keeps
     * submissions queued reads the previous frame's histogram without
     * waiting for the current one. */
    uint32_t nb_awb_slots;

    /* Padding to allow for future options. Must always be set to 0. */
    uint8_t padding[256 - 8*4];
} CRAWPipelineParams;

/* Clamp negative output components to zero. Off by default: negatives are
 * meaningful for values outside the characterized gamut and should be
 * handled by the consumer's gamut mapping. */
#define CRAW_FRAME_FLAG_CLAMP_NEGATIVE (1 << 0)

/* Skip the ACES 1.3 Reference Gamut Compression that CRAW_CS_ACES_AP1_LINEAR
 * output normally ends with, and hand back the unmodified AP1 values, which
 * for a saturated light source means components below zero.
 *
 * It is a frame flag rather than a pipeline parameter because a caller
 * comparing the two wants them frame by frame on one pipeline, and because
 * it costs nothing: it is a bit in the colour shader's push constant and the
 * branch is uniform across the dispatch. It has no effect on any other
 * output space; CRAW_CS_ACES_AP0_LINEAR in particular is never compressed,
 * being the archival encoding that has to keep every value the camera
 * matrix produced. See DESIGN.md 8.1. */
#define CRAW_FRAME_FLAG_NO_GAMUT_COMPRESSION (1 << 2)

/* Lens correction: keep every source pixel instead of filling the frame.
 * The corrected picture is scaled down uniformly, by one factor for all
 * three channels, until the whole warped source fits inside the output,
 * and everything outside it comes out black with alpha still 1.0. Unset,
 * the warp fills the frame and the source's outer corners are lost, which
 * is the trade the camera's own k0 makes. Ignored unless the pipeline was
 * created with CRAW_PIPELINE_FLAG_LENS and the frame carries a valid
 * correction. See DESIGN.md 16.8 and craw_lens_keep_all_scale(). */
#define CRAW_FRAME_FLAG_LENS_KEEP_ALL (1 << 1)

/* Per-frame settings. */
typedef struct CRAWFrameParams {
    const CRAWProfile *profile;

    enum CRAWWbMode wb_mode;
    CRAWIlluminant  illuminant;   /* used for CRAW_WB_REBALANCE */

    /* Extra exposure in stops on top of the profile's gain and exposure
     * index ratio. 0.0 for none. */
    float exposure_stops;

    uint32_t flags;               /* CRAW_FRAME_FLAG_* */

    /* Output colour space for this frame, overriding the pipeline's own
     * CRAWPipelineParams.out_space when out_space_set is true. Nothing in
     * the pipeline's shape depends on the space — it is one 3x3 composed on
     * the host per frame, plus the gamut compression that ACEScg output
     * ends with — so a caller can alternate spaces frame by frame on one
     * pipeline with no rebuild and no extra memory. That is what a viewer
     * showing ACEScg while writing an ACES2065-1 archive of the same frames
     * needs, which is otherwise two pipelines over one clip.
     *
     * The separate bool, rather than a sentinel in the enum: 0 is
     * CRAW_CS_XYZ_D65, a perfectly real space, so the enum has no value
     * free to mean "unset", and a CRAWFrameParams a caller merely zeroed
     * has to keep meaning "whatever the pipeline was created with", which
     * is what it meant before this field existed. */
    enum CRAWColorSpace out_space;
    bool out_space_set;

    /* Denoise strength, in units of the modelled noise level: 1.0 removes
     * noise at exactly the level the model predicts, larger values smooth
     * across real detail, smaller ones leave noise behind. 0.0 skips the
     * pass for this frame without rebuilding the pipeline, and is the
     * default. Ignored unless the pipeline was created with
     * CRAW_PIPELINE_FLAG_DENOISE. */
    float denoise_strength;

    /* Sensor noise model for this frame. When `a` is positive it is used
     * as given; otherwise the profile's table is evaluated at the
     * profile's as_shot_ei, and a profile without a table makes
     * craw_pipeline_record() fail with CRAW_ERROR(ENOTSUP) whenever
     * denoise_strength is positive. */
    CRAWNoiseModel noise_override;

    /* False colour suppression, DESIGN.md 17: the number of 3x3 median
     * passes over the colour differences of the debayered intermediate, 0
     * to CRAW_FALSE_COLOR_MAX_PASSES. 0, the default, skips the pass for
     * this frame without rebuilding the pipeline; more than the maximum
     * makes craw_pipeline_record() fail with CRAW_ERROR(EINVAL). Ignored
     * unless the pipeline was created with CRAW_PIPELINE_FLAG_FALSE_COLOR. */
    uint32_t false_color_passes;

    /* Geometric lens correction for this frame, by value rather than by
     * pointer: the record changes every frame while the lens zooms or
     * focuses, and a frame's parameters have to stay self-contained when a
     * player keeps several frames in flight. `valid` false means "leave
     * this frame alone", which is the default and is what a clip without a
     * record gets. Ignored unless the pipeline was created with
     * CRAW_PIPELINE_FLAG_LENS. See <craw/lens.h> and DESIGN.md 16. */
    CRAWLensCorrection lens;

    /* Padding to allow for future options. Must always be set to 0. The
     * `- 8` after the 64 is out_space, out_space_set and the alignment hole
     * after it, and the `- 4` is false_color_passes, both taken out of the
     * tail so that sizeof stays what it was. */
    uint8_t padding[256 - 4*4 - 8 - 12 - 64 - 8 - 4];
} CRAWFrameParams;

/* Fills `p` with defaults: RGGB, RCD, XYZ D65 output, RGBA16F, no flags.
 * width and height are left at 0 and must be set by the caller. */
CRAW_API void craw_pipeline_params_init(CRAWPipelineParams *p);

/* Fills `fp` with defaults: as-shot white balance, the pipeline's own
 * output colour space, no exposure change, no denoising, no noise model
 * override, no false colour suppression, no lens correction. The profile
 * pointer is left NULL and must be set by the caller. */
CRAW_API void craw_frame_params_init(CRAWFrameParams *fp);

/* Allocates the pipeline's intermediate images and histogram buffers for
 * exactly params->width x params->height. */
CRAW_API int craw_pipeline_create(CRAWContext *ctx,
                                  const CRAWPipelineParams *params,
                                  CRAWPipeline **pipe);

CRAW_API void craw_pipeline_free(CRAWPipeline **pipe);

/* Records the whole chain into `cmd`:
 *
 *   src (VK_FORMAT_R16_UNORM Bayer mosaic, params width x height)
 *     -> [denoise, in the raw domain, on the four CFA planes]
 *     -> black/white normalization, as-shot white balance, clip
 *     -> debayer to full resolution camera RGB
 *     -> [false colour suppression, a median of the colour differences]
 *     -> [auto white balance histogram]
 *     -> white balance re-balance, camera matrix, gain and exposure
 *     -> XYZ to the frame's output space (CRAWFrameParams.out_space when
 *        it is set, else the pipeline's)
 *     -> [ACES 1.3 gamut compression, ACEScg output only]
 *     -> [per-channel lens correction warp, on the linear output]
 *   dst (out_format, same size)
 *
 * src and dst must not alias. Both must already be in their stated layouts
 * with prior work made visible to VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT.
 * The pipeline may be recorded once per command buffer submission; its
 * intermediates are not double-buffered. */
CRAW_API int craw_pipeline_record(CRAWPipeline *pipe, VkCommandBuffer cmd,
                                  const CRAWImage *src, const CRAWImage *dst,
                                  const CRAWFrameParams *fp);

/* Exposes the host-visible histogram filled by the last recorded frame.
 * Only valid after the caller has waited for that submission to complete,
 * and until nb_awb_slots further recordings have been made. Fails with
 * CRAW_ERROR(EINVAL) if the pipeline was created without
 * CRAW_PIPELINE_FLAG_AWB_HISTOGRAM. */
CRAW_API int craw_pipeline_get_awb_histogram(CRAWPipeline *pipe,
                                             CRAWAwbHistogram *out);

/* The same for the recording made `frames_back` recordings ago: 0 is the
 * last one, 1 the one before it, and so on up to nb_awb_slots - 1. The
 * submission that recording belongs to must have completed. */
CRAW_API int craw_pipeline_get_awb_histogram_n(CRAWPipeline *pipe, uint32_t frames_back,
                                               CRAWAwbHistogram *out);

#endif /* CRAW_PIPELINE_H */
