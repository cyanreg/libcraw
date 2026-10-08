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

/* Build the lens correction pass, a per-channel warp of the camera RGB
 * intermediate after the debayer. It costs one full resolution RGBA16F
 * image, shared with the other passes that need one. Once created it is
 * switched per frame with CRAWFrameParams.lens: a correction with valid =
 * false skips the warp entirely, so such a frame is bit for bit what a
 * pipeline built without the flag produces. It is also what turns the
 * picture, CRAWFrameParams.rotation, with or without a correction. See
 * DESIGN.md 16. */
#define CRAW_PIPELINE_FLAG_LENS (1 << 2)

/* Build the false colour suppression pass, DESIGN.md 17: a 3x3 median
 * followed by a 3x3 box blur of the colour differences of the debayered
 * intermediate, run up to CRAW_FALSE_COLOR_MAX_PASSES times, which turns
 * the coloured dots that a glint smaller than a CFA quad leaves into
 * neutral ones and touches nothing else. It costs a full resolution
 * RGBA16F image, shared with the other passes that need one. Once built
 * it is switched per frame with CRAWFrameParams.false_color_passes, which
 * costs nothing when zero. */
#define CRAW_PIPELINE_FLAG_FALSE_COLOR (1 << 3)

/* The most passes the stage's fused dispatch carries an apron for, the
 * same range as RawTherapee's steps. One or two are free in a 4:2:0
 * delivery; five is a visible chroma blur at 1:1. DESIGN.md 17.2. */
#define CRAW_FALSE_COLOR_MAX_PASSES 5

/* Build the chroma denoise pass, DESIGN.md 14.4: the colour differences of
 * the debayered intermediate, at quarter resolution, smoothed by a joint
 * bilateral guided by the quarter resolution G, and that band put back
 * into every pixel. It reaches the chroma blotches of a dark scene and the
 * colour of glints two and three sensels wide, which the false colour pass
 * leaves. It costs two quarter resolution RGBA16F images and one full
 * resolution RGBA16F image, shared with the other passes that need one.
 * Once built it is switched per frame with
 * CRAWFrameParams.denoise_chroma_strength, which costs nothing when zero. */
#define CRAW_PIPELINE_FLAG_CHROMA_DENOISE (1 << 4)

/* Build the highlight reconstruction pass, DESIGN.md 18: every sensel
 * that clipped gets its clipped channels rebuilt from the channels that
 * did not clip around it and from the colour of the unclipped light around
 * the blown area, before the debayer, and the result is allowed above the
 * balanced clip point of 1.0 that the prep otherwise cuts everything at. A
 * sky where only green clipped keeps its clouds, and an area blown in every
 * channel keeps the colour of its surroundings instead of going flat white.
 * It costs an extra full resolution R16F mosaic and a quarter resolution
 * RGBA32F colour field with its pyramid, 58 MB at 5.8K. Once built it is
 * switched per frame with CRAW_FRAME_FLAG_HIGHLIGHT_RECONSTRUCTION, which
 * costs nothing when unset. */
#define CRAW_PIPELINE_FLAG_HIGHLIGHT_RECONSTRUCTION (1 << 5)

/* Build the defringe pass, DESIGN.md 19: near a strong edge, the colour of
 * every pixel is pulled back into the range of colours found a few pixels
 * away on either side of that edge, which takes out the purple, blue, cyan
 * and pink rims that the lens's chromatic aberration leaves along bright
 * edges and that highlight reconstruction no longer hides under the clip.
 * The level of every pixel is kept, and a pixel whose colour is already in
 * range is not touched. It costs one full resolution RGBA16F image, shared
 * with the other passes that need one. Once built it is switched per
 * frame with CRAWFrameParams.defringe_strength, which costs nothing when
 * zero. */
#define CRAW_PIPELINE_FLAG_DEFRINGE (1 << 6)

/* Build the temporal denoise pass, DESIGN.md 14.7: in front of the spatial
 * filter of CRAW_PIPELINE_FLAG_DENOISE and in the same variance stabilised
 * domain, every sample of the mosaic is averaged with the previous frame's
 * filtered value at the same place, as far as the 3x3 around it agrees
 * with last frame's within the noise the model predicts. A static scene
 * then averages over the last few frames and anything that moved is left
 * to the spatial filter, which measures what it removes against the noise
 * the average left. It costs two sets of full resolution state, an R32F
 * and an R8 image each, 175 MB at 5.8K, and builds the raw denoise pass
 * and its R16_UNORM image whether or not CRAW_PIPELINE_FLAG_DENOISE is
 * also set. Once built it is switched per frame with
 * CRAWFrameParams.denoise_temporal_strength, which costs nothing when
 * zero.
 *
 * With it the pipeline stops being a function of the frame it is given:
 * each recording reads what the previous one left. See "Temporal state"
 * at craw_pipeline_record() for the rules that makes. */
#define CRAW_PIPELINE_FLAG_TEMPORAL_DENOISE (1 << 7)

/* Build the capture sharpening pass, DESIGN.md 20: Richardson-Lucy
 * deconvolution of the luma of the camera RGB intermediate by a Gaussian
 * of the frame's chosen radius, after every other stage on the camera RGB,
 * with the luma's ratio applied to all three channels so that no colour
 * moves, a local contrast mask in units of the modelled noise that leaves
 * flat and noisy areas alone, and a clamp to the local range that keeps
 * dark rings out of the surroundings of bright lights. It costs three full
 * resolution fp16 planes, 105 MB at 5.8K, and 1 ms plus 0.4 ms an
 * iteration at 5.8K on an RX 6900 XT. Once built it is switched per frame
 * with CRAWFrameParams.sharpen_iterations and sharpen_radius, and costs
 * nothing when either is zero. */
#define CRAW_PIPELINE_FLAG_SHARPEN (1 << 8)

/* The widest Gaussian the sharpening pass takes, sigma in pixels, and the
 * most iterations one frame may ask for. The radius bounds the shader's
 * tile; the count bounds how long one recording can keep the GPU busy. */
#define CRAW_SHARPEN_MAX_RADIUS     2.0f
#define CRAW_SHARPEN_MAX_ITERATIONS 100

/* Time every dispatch the pipeline records, for
 * craw_pipeline_get_stage_times(): a pair of GPU timestamps around each one,
 * in a query pool the pipeline owns that holds the last
 * CRAW_PIPELINE_PROFILE_SLOTS recordings. Without the flag no pool exists
 * and a recording is exactly what it is on a pipeline that never heard of
 * profiling. With it every dispatch starts after the one before it has
 * finished, which the barriers between passes mostly demand anyway; the
 * auto white balance histogram and the colour pass, which could overlap,
 * no longer do, so the recording as a whole can take a little longer.
 * Needs timestamps on the queue family the context was created with
 * (timestampValidBits above zero); craw_pipeline_create() fails with
 * CRAW_ERROR(ENOTSUP) on one without. DESIGN.md 21. */
#define CRAW_PIPELINE_FLAG_PROFILE (1 << 9)

/* How many recordings' timings a profiling pipeline keeps. */
#define CRAW_PIPELINE_PROFILE_SLOTS 4

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
 * turned when CRAWFrameParams.rotation turns it, and everything outside it
 * comes out black with alpha still 1.0. Unset, the warp fills the frame
 * and the source's outer corners are lost, which is the trade the camera's
 * own k0 makes. Ignored unless the pipeline was created with
 * CRAW_PIPELINE_FLAG_LENS and the frame carries a valid correction or a
 * turn. See DESIGN.md 16.8, 16.9 and craw_lens_rotation_scale(). */
#define CRAW_FRAME_FLAG_LENS_KEEP_ALL (1 << 1)

/* Rebuild the clipped highlights of this frame, DESIGN.md 18. Unset, the
 * frame is clipped at the balanced clip point exactly as on a pipeline
 * built without the stage, and the pass is not dispatched. Ignored unless
 * the pipeline was created with CRAW_PIPELINE_FLAG_HIGHLIGHT_RECONSTRUCTION.
 * A flag rather than a strength because there is nothing to dial: a
 * rebuilt channel is either the estimate or the clip. */
#define CRAW_FRAME_FLAG_HIGHLIGHT_RECONSTRUCTION (1 << 3)

/* The frame before this one was not the previous frame of the same scene:
 * a seek, a loop back to the start, a new clip, a cut. The temporal
 * denoise ignores its state for this frame, as if the pipeline had just
 * been created, and fills it again from this frame on. It is the
 * denoiser's `scene_cut` (craw_awb_smoother_push()), and stricter: the
 * state is last frame's picture sample by sample, so any jump breaks it,
 * where an illuminant track may well survive a seek within one scene. No
 * effect without CRAW_PIPELINE_FLAG_TEMPORAL_DENOISE. DESIGN.md 14.7. */
#define CRAW_FRAME_FLAG_TEMPORAL_RESET (1 << 4)

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

    /* False colour suppression, DESIGN.md 17: the number of median and
     * blur passes over the colour differences of the debayered
     * intermediate, 0 to CRAW_FALSE_COLOR_MAX_PASSES. 0, the default, skips the pass for
     * this frame without rebuilding the pipeline; more than the maximum
     * makes craw_pipeline_record() fail with CRAW_ERROR(EINVAL). Ignored
     * unless the pipeline was created with CRAW_PIPELINE_FLAG_FALSE_COLOR. */
    uint32_t false_color_passes;

    /* Chroma denoise, DESIGN.md 14.4: how much of the quarter resolution
     * chroma band is replaced by its bilateral smoothing, 0.0 to 1.0. 0.0,
     * the default, skips the pass for this frame without rebuilding the
     * pipeline; above 1.0 makes craw_pipeline_record() fail with
     * CRAW_ERROR(EINVAL). Ignored unless the pipeline was created with
     * CRAW_PIPELINE_FLAG_CHROMA_DENOISE. */
    float denoise_chroma_strength;

    /* Geometric lens correction for this frame, by value rather than by
     * pointer: the record changes every frame while the lens zooms or
     * focuses, and a frame's parameters have to stay self-contained when a
     * player keeps several frames in flight. `valid` false means "leave
     * this frame alone", which is the default and is what a clip without a
     * record gets. Ignored unless the pipeline was created with
     * CRAW_PIPELINE_FLAG_LENS. See <craw/lens.h> and DESIGN.md 16. */
    CRAWLensCorrection lens;

    /* Defringe, DESIGN.md 19: how far a fringe's colour is pulled back into
     * the range its edge's two sides span, 0.0 to 1.0. 0.0, the default,
     * skips the pass for this frame without rebuilding the pipeline; above
     * 1.0 makes craw_pipeline_record() fail with CRAW_ERROR(EINVAL). Ignored
     * unless the pipeline was created with CRAW_PIPELINE_FLAG_DEFRINGE. */
    float defringe_strength;

    /* Temporal denoise, DESIGN.md 14.7: alpha_max, the most of the previous
     * frame's filtered value that a sample can take on, 0.0 to 1.0. On a
     * static scene the average then reaches back about 1 / (1 - alpha_max)
     * frames and the noise falls towards sqrt((1 - a) / (1 + a)) of a
     * frame's alone, a third at 0.8; where the 3x3 around a sample differs
     * from last frame's by more than the model's noise it takes less, and
     * where it has moved it takes nothing. How much more counts as moved
     * is the spatial strength h of denoise_strength when that pass runs
     * and 1.0 when it does not; the spatial filter then measures what it
     * removes against the noise the average left. 0.8 is a good value;
     * 1.0 is allowed and useless, a static sample then keeps last frame's
     * value for ever and never takes in another. It needs the noise model
     * exactly as denoise_strength does, and fails the same way without
     * one. 0.0, the default, skips the pass for this frame without
     * rebuilding the pipeline; above 1.0 makes craw_pipeline_record() fail
     * with CRAW_ERROR(EINVAL). Ignored unless the pipeline was created
     * with CRAW_PIPELINE_FLAG_TEMPORAL_DENOISE.
     *
     * denoise_temporal_replay says that this recording develops the same
     * frame as the last one did, again: a grade changed while paused, a
     * picker that needs the frame without the gamut compression. It reads
     * the state that recording read, and replaces what that recording
     * wrote, so a frame developed any number of times comes out the same
     * as the first time and the next frame still follows it. Unset, the
     * recording is the next frame. See "Temporal state" at
     * craw_pipeline_record(). */
    float denoise_temporal_strength;
    bool denoise_temporal_replay;

    /* Capture sharpening, DESIGN.md 20. Ignored unless the pipeline was
     * created with CRAW_PIPELINE_FLAG_SHARPEN; a radius or an iteration
     * count of zero, the defaults, skips the pass for this frame without
     * rebuilding the pipeline.
     *
     * sharpen_radius is the sigma of the Gaussian blur being undone, in
     * pixels, above 0 and at most CRAW_SHARPEN_MAX_RADIUS.
     * sharpen_iterations is the number of Richardson-Lucy iterations, at
     * most CRAW_SHARPEN_MAX_ITERATIONS: each costs about the same, and a
     * viewer can run fewer while playing than it does for a still.
     * sharpen_contrast is the local contrast below which nothing is
     * sharpened, in units of the modelled noise the way denoise_strength
     * is: 0 sharpens everything, noise included, and a positive value
     * needs a noise model exactly as the denoise does (noise_override, or
     * the profile's table), failing with CRAW_ERROR(ENOTSUP) without one.
     * Anything out of range makes craw_pipeline_record() fail with
     * CRAW_ERROR(EINVAL). */
    float    sharpen_radius;
    uint32_t sharpen_iterations;
    float    sharpen_contrast;

    /* Turn the picture by this many degrees, clockwise as it is looked at,
     * about the optical centre, DESIGN.md 16.9. Less than
     * CRAW_LENS_MAX_ROTATION either way; more, or not a number, makes
     * craw_pipeline_record() fail with CRAW_ERROR(EINVAL). The frame keeps
     * its size: filling it, the picture is magnified by
     * craw_lens_rotation_scale() until the turned frame has a source
     * everywhere, measured against what the lens correction makes of the
     * readout rather than against the readout's rectangle, so a turn uses
     * the source the correction's own frame would have thrown away before
     * it magnifies anything. With CRAW_FRAME_FLAG_LENS_KEEP_ALL the picture
     * is shrunk until the whole turned source fits instead.
     *
     * It is the lens warp's own resample, one Catmull-Rom for both, so it
     * needs CRAW_PIPELINE_FLAG_LENS and is ignored without it; a frame with
     * no record is warped with no correction, only turned. 0.0, the
     * default, is the frame the record or the readout gives, with no pass
     * at all when there is no record either. */
    float rotation;

    /* Padding to allow for future options. Must always be set to 0. The
     * `- 8` after the 64 is out_space, out_space_set and the alignment hole
     * after it, the three `- 4` after that are false_color_passes,
     * denoise_chroma_strength and defringe_strength, the `- 4 - 4` after
     * those are denoise_temporal_strength, then denoise_temporal_replay
     * with the three bytes after it, the `- 12` the three sharpen_ fields
     * and the last `- 4` rotation, all taken out of the tail so that sizeof
     * stays what it was. */
    uint8_t padding[256 - 4*4 - 8 - 12 - 64 - 8 - 4 - 4 - 4 - 4 - 4 - 12 - 4];
} CRAWFrameParams;

/* Fills `p` with defaults: RGGB, RCD, XYZ D65 output, RGBA16F, no flags.
 * width and height are left at 0 and must be set by the caller. */
CRAW_API void craw_pipeline_params_init(CRAWPipelineParams *p);

/* Fills `fp` with defaults: as-shot white balance, the pipeline's own
 * output colour space, no exposure change, no denoising of any kind, no
 * noise model override, no highlight reconstruction, no false colour
 * suppression, no defringe, no lens correction, no sharpening, no turn.
 * The profile pointer is left NULL and must be set by the caller. */
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
 *     -> [temporal denoise, each sample averaged with the previous frame's]
 *     -> [denoise, in the raw domain, on the four CFA planes]
 *     -> black/white normalization, as-shot white balance, and either the
 *        clip or [highlight reconstruction, clipped channels rebuilt]
 *     -> debayer to full resolution camera RGB
 *     -> [false colour suppression, a median of the colour differences]
 *     -> [per-channel lens correction warp, and the turn]
 *     -> [defringe, colour along strong edges held to what both sides have]
 *     -> [chroma denoise, the quarter resolution chroma band smoothed]
 *     -> [capture sharpening, Richardson-Lucy on the luma, as a ratio]
 *     -> [auto white balance histogram]
 *     -> white balance re-balance, camera matrix, gain and exposure
 *     -> XYZ to the frame's output space (CRAWFrameParams.out_space when
 *        it is set, else the pipeline's)
 *     -> [ACES 1.3 gamut compression, ACEScg output only]
 *   dst (out_format, same size)
 *
 * src and dst must not alias. Both must already be in their stated layouts
 * with prior work made visible to VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT.
 * The pipeline may be recorded once per command buffer submission; its
 * intermediates are not double-buffered. Recordings must be submitted to
 * one queue in the order they were made: the barriers that order one
 * recording's use of the intermediates after the last one's are recorded
 * into the later command buffer and only reach back in submission order.
 * Any number of them may be in flight at once.
 *
 * Temporal state. With CRAW_PIPELINE_FLAG_TEMPORAL_DENOISE the pipeline
 * keeps two sets of state and remembers which holds the last developed
 * frame's filtered planes and which the one before it; a recording is no
 * longer a function of its arguments alone. The rules:
 *
 *  - Nothing is valid on creation, and CRAW_FRAME_FLAG_TEMPORAL_RESET
 *    forgets whatever was: the frame is developed as the first one, and
 *    the average refills from it.
 *  - A recording without denoise_temporal_replay is the next frame: it
 *    reads the state the last recording wrote and writes the other set.
 *    A frame the caller skipped (a player dropping one to keep up) is
 *    not a cut; the next one simply moved further, and the pass sees that
 *    as motion.
 *  - A recording with it develops the last recorded frame again: it reads
 *    the set that recording read and writes the one that recording
 *    wrote. With the same parameters the output is bit for bit the first
 *    development's, however often it is repeated, and a change of
 *    parameters is a change to that frame alone.
 *  - A next frame recorded with the pass off (a strength of zero) leaves
 *    no state behind, so the frame after it starts again; a replay with
 *    the pass off leaves the state as it was.
 *  - A recording that fails has changed nothing. One that succeeds but is
 *    never submitted has: follow it with a reset.
 *
 * The state images are read and written only by recordings, in the order
 * above, so the barriers that order consecutive recordings on one queue
 * order them too, however many submissions are in flight. DESIGN.md 14.7
 * and 11b. */
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

/* One pass of a profiled recording, CRAW_PIPELINE_FLAG_PROFILE. */
typedef struct CRAWStageTime {
    /* The pass's shader, "rcd", "denoise_nlm", "sharpen" and so on.
     * Owned by the pipeline and valid until it is freed. */
    const char *name;
    uint32_t count;   /* dispatches of it in the recording */
    double   ms;      /* their GPU time, summed */
} CRAWStageTime;

/* The GPU time of every pass of the last recording, one entry per pass in
 * the order the passes first ran: a pass dispatched more than once (the
 * sharpening's iterations, the highlight pyramid's levels) is one entry
 * with its dispatches summed. Returns how many entries the recording has,
 * of which the first `max` are written to `out` (which may be NULL when
 * `max` is 0). Only valid once the submission that recording belongs to
 * has completed; before that it fails with CRAW_ERROR(EAGAIN). Fails with
 * CRAW_ERROR(EINVAL) on a pipeline created without
 * CRAW_PIPELINE_FLAG_PROFILE. The times do not include the barriers
 * between passes; the gap between their sum and a timestamp pair around
 * the whole recording is what those cost. */
CRAW_API int craw_pipeline_get_stage_times(CRAWPipeline *pipe, CRAWStageTime *out,
                                           uint32_t max);

/* The same for the recording made `frames_back` recordings ago, 0 to
 * CRAW_PIPELINE_PROFILE_SLOTS - 1, whose submission must have completed. */
CRAW_API int craw_pipeline_get_stage_times_n(CRAWPipeline *pipe, uint32_t frames_back,
                                             CRAWStageTime *out, uint32_t max);

#endif /* CRAW_PIPELINE_H */
