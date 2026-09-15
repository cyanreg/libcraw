/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ref.h"

int craw_ref_color(float *rgb, uint32_t width, uint32_t height,
                   const CRAWFrameParams *fp, enum CRAWColorSpace out_space)
{
    CRAWColorParams par;
    float m_out[3][3];
    float scale;
    bool clamp_negative;
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
            /* Negatives are normal outside the characterized gamut and are
             * only removed when the caller asks for it. */
            if (clamp_negative)
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
    int ret;

    if (!params || !fp || !fp->profile)
        return CRAW_ERROR(EINVAL);
    if (params->width && (params->width != width || params->height != height))
        return CRAW_ERROR(EINVAL);
    if (params->cfa != fp->profile->cfa)
        return CRAW_ERROR(EINVAL);

    if ((params->flags & CRAW_PIPELINE_FLAG_DENOISE) && fp->denoise_strength > 0.0f) {
        CRAWNoiseModel nm;

        ret = craw_ref_noise_model(fp, &nm);
        if (ret < 0)
            return ret;

        clean = malloc((size_t)width * height * sizeof(*clean));
        if (!clean)
            return CRAW_ERROR(ENOMEM);

        ret = craw_ref_denoise(bayer, width, height, stride, fp->profile, &nm,
                               fp->denoise_strength, clean, width);
        if (ret < 0) {
            free(clean);
            return ret;
        }

        bayer = clean;
        stride = width;
    }

    {
        CRAWColorParams par;

        ret = craw_profile_eval(fp->profile, fp->wb_mode, &fp->illuminant, &par);
        if (ret < 0)
            goto done;

        ret = craw_ref_debayer_wb(bayer, width, height, stride, fp->profile, par.wb,
                                  params->debayer, out);
        if (ret < 0)
            goto done;
    }

    ret = craw_ref_color(out, width, height, fp, params->out_space);

done:
    free(clean);

    return ret;
}
