/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "gamut.h"
#include "ref.h"

int craw_ref_color(float *rgb, uint32_t width, uint32_t height,
                   const CRAWFrameParams *fp, enum CRAWColorSpace out_space)
{
    CRAWColorParams par;
    CRAWGamutCompress gc;
    float m_out[3][3];
    float scale;
    bool clamp_negative, compress;
    int ret;

    if (!rgb || !fp || !fp->profile)
        return CRAW_ERROR(EINVAL);

    ret = craw_profile_eval(fp->profile, fp->wb_mode, &fp->illuminant, &par);
    if (ret < 0)
        return ret;

    ret = craw_colorspace_from_xyz_d65(out_space, m_out);
    if (ret < 0)
        return ret;

    scale = par.gain * exp2f(fp->exposure_stops);
    clamp_negative = (fp->flags & CRAW_FRAME_FLAG_CLAMP_NEGATIVE) != 0;

    /* DESIGN.md 8.1: only ACEScg, and only when the frame has not vetoed
     * it. ACES2065-1 is the archive and keeps whatever the matrix made. */
    compress = out_space == CRAW_CS_ACES_AP1_LINEAR &&
               !(fp->flags & CRAW_FRAME_FLAG_NO_GAMUT_COMPRESSION);
    craw_gamut_compress_params(&gc);

    for (size_t i = 0; i < (size_t)width * height; i++) {
        float *px = &rgb[i * 4];
        float cam[3], xyz[3], out[3];

        for (int c = 0; c < 3; c++)
            cam[c] = px[c];

        for (int r = 0; r < 3; r++) {
            xyz[r] = (par.cam_to_xyz[r][0] * cam[0]
                    + par.cam_to_xyz[r][1] * cam[1]
                    + par.cam_to_xyz[r][2] * cam[2]) * scale;
        }

        for (int r = 0; r < 3; r++) {
            out[r] = m_out[r][0] * xyz[0]
                   + m_out[r][1] * xyz[1]
                   + m_out[r][2] * xyz[2];
        }

        if (compress)
            craw_gamut_compress(&gc, out);

        /* Negatives are normal outside the characterized gamut and are
         * only removed when the caller asks for it. */
        if (clamp_negative) {
            for (int r = 0; r < 3; r++)
                out[r] = CRAW_MAX(out[r], 0.0f);
        }

        px[0] = out[0];
        px[1] = out[1];
        px[2] = out[2];
        px[3] = 1.0f;
    }

    return 0;
}

int craw_ref_noise_model(const CRAWFrameParams *fp, CRAWNoiseModel *out)
{
    if (!fp || !fp->profile || !out)
        return CRAW_ERROR(EINVAL);

    if (fp->noise_override.a > 0.0f) {
        *out = fp->noise_override;
        return 0;
    }

    return craw_profile_noise_eval(fp->profile, fp->profile->as_shot_ei, out);
}

int craw_ref_process(const uint16_t *bayer, uint32_t width, uint32_t height,
                     size_t stride, const CRAWPipelineParams *params,
                     const CRAWFrameParams *fp, float *out)
{
    uint16_t *clean = NULL;
    float *state = NULL;
    float strength, alpha_max;
    int ret;

    if (!params || !fp || !fp->profile)
        return CRAW_ERROR(EINVAL);
    if (params->width && (params->width != width || params->height != height))
        return CRAW_ERROR(EINVAL);
    if (params->cfa != fp->profile->cfa)
        return CRAW_ERROR(EINVAL);

    /* A single frame has no previous one, so the temporal average is
     * what it is on a pipeline's first frame or after a reset: the frame
     * taken as it stands, spatial filter or not. DESIGN.md 14.7. */
    strength = (params->flags & CRAW_PIPELINE_FLAG_DENOISE) ? fp->denoise_strength : 0.0f;
    alpha_max = (params->flags & CRAW_PIPELINE_FLAG_TEMPORAL_DENOISE)
              ? fp->denoise_temporal_strength : 0.0f;
    strength = CRAW_MAX(strength, 0.0f);
    alpha_max = CRAW_MAX(alpha_max, 0.0f);
    if (strength > 0.0f || alpha_max > 0.0f) {
        CRAWNoiseModel nm;

        ret = craw_ref_noise_model(fp, &nm);
        if (ret < 0)
            return ret;

        clean = malloc((size_t)width * height * sizeof(*clean));
        if (alpha_max > 0.0f)
            state = malloc(2 * (size_t)width * height * sizeof(*state));
        if (!clean || (alpha_max > 0.0f && !state)) {
            ret = CRAW_ERROR(ENOMEM);
            goto done;
        }

        ret = craw_ref_denoise_temporal(bayer, width, height, stride, fp->profile, &nm,
                                        strength, alpha_max, NULL, state, clean, width);
        if (ret < 0)
            goto done;

        bayer = clean;
        stride = width;
    }

    {
        CRAWColorParams par;

        ret = craw_profile_eval(fp->profile, fp->wb_mode, &fp->illuminant, &par);
        if (ret < 0)
            goto done;

        /* The rebuilt mosaic replaces the prep and its clip, DESIGN.md
         * 18.3. */
        if ((params->flags & CRAW_PIPELINE_FLAG_HIGHLIGHT_RECONSTRUCTION) &&
            (fp->flags & CRAW_FRAME_FLAG_HIGHLIGHT_RECONSTRUCTION)) {
            float *v = malloc((size_t)width * height * sizeof(*v));

            if (!v) {
                ret = CRAW_ERROR(ENOMEM);
                goto done;
            }
            ret = craw_ref_highlights(bayer, width, height, stride, fp->profile, par.wb, v);
            if (ret >= 0)
                ret = craw_ref_debayer_mosaic(v, width, height, params->debayer, out);
            free(v);
        } else {
            ret = craw_ref_debayer_wb(bayer, width, height, stride, fp->profile, par.wb,
                                      params->debayer, out);
        }
        if (ret < 0)
            goto done;
    }

    /* On the intermediate, before anything reads it: the false colour,
     * DESIGN.md 17.3, the lens warp, 16.3, the defringe, 19.3, the chroma
     * denoise on what they left, 14.4, and the sharpening, 20.3. */
    if ((params->flags & CRAW_PIPELINE_FLAG_FALSE_COLOR) && fp->false_color_passes > 0) {
        ret = craw_ref_false_color(out, width, height, fp->false_color_passes);
        if (ret < 0)
            goto done;
    }
    /* The warp cannot run in place, so the intermediate is moved aside and
     * the result written back over `out`. DESIGN.md 16.3: on the camera
     * RGB, before the defringe. */
    if ((params->flags & CRAW_PIPELINE_FLAG_LENS) && fp->lens.valid) {
        size_t n = (size_t)width * height * 4;
        float *moved = malloc(n * sizeof(*moved));

        if (!moved) {
            ret = CRAW_ERROR(ENOMEM);
            goto done;
        }

        memcpy(moved, out, n * sizeof(*moved));
        ret = craw_ref_lens(moved, width, height, &fp->lens,
                            (fp->flags & CRAW_FRAME_FLAG_LENS_KEEP_ALL) != 0, out);
        free(moved);
        if (ret < 0)
            goto done;
    }
    if ((params->flags & CRAW_PIPELINE_FLAG_DEFRINGE) && fp->defringe_strength > 0.0f) {
        ret = craw_ref_defringe(out, width, height, fp->defringe_strength);
        if (ret < 0)
            goto done;
    }
    if ((params->flags & CRAW_PIPELINE_FLAG_CHROMA_DENOISE) && fp->denoise_chroma_strength > 0.0f) {
        ret = craw_ref_chroma_denoise(out, width, height, fp->denoise_chroma_strength);
        if (ret < 0)
            goto done;
    }
    /* After every other stage on the camera RGB, DESIGN.md 20.3, with the
     * mask's noise under the frame's own gains. */
    if ((params->flags & CRAW_PIPELINE_FLAG_SHARPEN) && fp->sharpen_iterations > 0 &&
        fp->sharpen_radius > 0.0f) {
        CRAWNoiseModel nm = { 0 };
        CRAWColorParams par;

        ret = craw_profile_eval(fp->profile, fp->wb_mode, &fp->illuminant, &par);
        if (ret >= 0 && fp->sharpen_contrast > 0.0f)
            ret = craw_ref_noise_model(fp, &nm);
        if (ret >= 0)
            ret = craw_ref_sharpen(out, width, height, fp->sharpen_radius,
                                   fp->sharpen_iterations, fp->sharpen_contrast, &nm, par.wb);
        if (ret < 0)
            goto done;
    }

    ret = craw_ref_color(out, width, height, fp, craw_ref_out_space(params, fp));

done:
    free(state);
    free(clean);

    return ret;
}
