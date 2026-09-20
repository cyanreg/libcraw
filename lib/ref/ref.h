/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#ifndef CRAW_REF_H
#define CRAW_REF_H

/* CPU reference implementation of the pipeline, float32, used by tests and
 * tools to verify the shaders. Not exported. Follows DESIGN.md to the
 * letter; when the shaders and this disagree, DESIGN.md decides. */

#include <stdint.h>
#include <craw/profile.h>
#include <craw/pipeline.h>

/* Raw domain denoise, DESIGN.md 14.3: reads the mosaic and writes a
 * denoised mosaic in the same 16-bit UNORM raw encoding, which is exactly
 * what the GPU pass produces. `strength` must be positive; `model` is the
 * (a, b) of the frame, already resolved from the profile or an override. */
int craw_ref_denoise(const uint16_t *bayer, uint32_t width, uint32_t height,
                     size_t stride, const CRAWProfile *profile,
                     const CRAWNoiseModel *model, float strength,
                     uint16_t *out, size_t out_stride);

/* Resolves the noise model the pipeline would use for a frame: the
 * override when it carries a positive `a`, else the profile's table at its
 * as-shot exposure index. */
int craw_ref_noise_model(const CRAWFrameParams *fp, CRAWNoiseModel *out);

/* Runs prep + debayer only. `bayer` is width*height 16-bit UNORM samples
 * with `stride` samples per row. `rgb` receives width*height*4 floats
 * (camera RGB after white balance with `wb` and clipping, alpha 1.0). */
int craw_ref_debayer_wb(const uint16_t *bayer, uint32_t width, uint32_t height,
                        size_t stride, const CRAWProfile *profile,
                        const float wb[3], enum CRAWDebayer method, float *rgb);

/* The same with the profile's as-shot gains. */
int craw_ref_debayer(const uint16_t *bayer, uint32_t width, uint32_t height,
                     size_t stride, const CRAWProfile *profile,
                     enum CRAWDebayer method, float *rgb);

/* False colour suppression, DESIGN.md 17, in place on a camera RGB buffer
 * of width*height*4 floats: `passes` 3x3 medians of R - G and B - G with
 * mirrored borders, G and alpha untouched. `passes` must be 1 to
 * CRAW_FALSE_COLOR_MAX_PASSES; zero is the caller's decision not to call. */
int craw_ref_false_color(float *rgb, uint32_t width, uint32_t height, uint32_t passes);

/* Runs the colour stage on a camera RGB buffer in place: matrix, gain and
 * exposure for the frame's illuminant, then XYZ to `out_space`, which is
 * the *effective* space — the caller has already resolved the frame's own
 * override against the pipeline's, with craw_ref_out_space(). The buffer
 * must have been debayered with the gains of that same illuminant. */
int craw_ref_color(float *rgb, uint32_t width, uint32_t height,
                   const CRAWFrameParams *fp, enum CRAWColorSpace out_space);

/* The space a frame actually lands in: its own when it set one, else the
 * pipeline's. The same choice craw_pipeline_record() makes. */
static inline enum CRAWColorSpace craw_ref_out_space(const CRAWPipelineParams *params,
                                                     const CRAWFrameParams *fp)
{
    return fp->out_space_set ? fp->out_space : params->out_space;
}

/* Lens correction, DESIGN.md 16: the per-channel radial warp of the linear
 * output. `src` and `dst` are width*height*4 floats and must not be the
 * same buffer. A correction with valid = false is a copy, which is what
 * the pipeline's skipping the pass amounts to.
 *
 * `keep_all` is CRAW_FRAME_FLAG_LENS_KEEP_ALL (16.8): the scale comes from
 * craw_lens_keep_all_scale(), the same call the pipeline makes, so the two
 * cannot drift apart. */
int craw_ref_lens(const float *src, uint32_t width, uint32_t height,
                  const CRAWLensCorrection *lc, bool keep_all, float *dst);

/* Whole chain: [denoise,] debayer, [false colour,] colour, [lens]. `out`
 * receives width*height*4 floats. The denoise runs when `params` carries
 * CRAW_PIPELINE_FLAG_DENOISE and `fp->denoise_strength` is positive, the
 * false colour suppression when it carries CRAW_PIPELINE_FLAG_FALSE_COLOR
 * and `fp->false_color_passes` is positive, and the warp when it carries
 * CRAW_PIPELINE_FLAG_LENS and `fp->lens` is valid; all three are the
 * conditions craw_pipeline_record() applies. */
int craw_ref_process(const uint16_t *bayer, uint32_t width, uint32_t height,
                     size_t stride, const CRAWPipelineParams *params,
                     const CRAWFrameParams *fp, float *out);

#endif /* CRAW_REF_H */
