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

/* Fixed configuration of a pipeline. Changing any of it means creating a
 * new pipeline. */
typedef struct CRAWPipelineParams {
    uint32_t width, height;     /* Bayer frame size; both must be even */
    enum CRAWCfa cfa;           /* this version: CRAW_CFA_RGGB only */
    enum CRAWDebayer debayer;
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

/* Per-frame settings. */
typedef struct CRAWFrameParams {
    const CRAWProfile *profile;

    enum CRAWWbMode wb_mode;
    CRAWIlluminant  illuminant;   /* used for CRAW_WB_REBALANCE */

    /* Extra exposure in stops on top of the profile's gain and exposure
     * index ratio. 0.0 for none. */
    float exposure_stops;

    uint32_t flags;               /* CRAW_FRAME_FLAG_* */

    /* Padding to allow for future options. Must always be set to 0. */
    uint8_t padding[256 - 4*4 - 8];
} CRAWFrameParams;

/* Fills `p` with defaults: RGGB, RCD, XYZ D65 output, RGBA16F, no flags.
 * width and height are left at 0 and must be set by the caller. */
CRAW_API void craw_pipeline_params_init(CRAWPipelineParams *p);

/* Fills `fp` with defaults: as-shot white balance, no exposure change. The
 * profile pointer is left NULL and must be set by the caller. */
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
 *     -> black/white normalization, as-shot white balance, clip
 *     -> debayer to full resolution camera RGB
 *     -> [auto white balance histogram]
 *     -> white balance re-balance, camera matrix, gain and exposure
 *     -> XYZ to out_space
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
