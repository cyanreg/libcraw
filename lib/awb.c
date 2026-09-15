/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#include <math.h>
#include <string.h>

#include <craw/awb.h>

/* Share of the sample weight that has to sit inside the neutral band before
 * the estimate is trusted completely, and the spread along the locus at
 * which it is trusted not at all. Both are properties of the estimator
 * rather than of a scene, so they are not exposed as parameters. */
#define CRAW_AWB_CONF_ZONE_FRACTION 0.3
#define CRAW_AWB_CONF_SPREAD_MIRED  60.0

void craw_awb_params_init(CRAWAwbParams *p)
{
    if (!p)
        return;

    *p = (CRAWAwbParams) {
        .band              = 0.08f,
        .sigma             = 0.05f,
        .min_zone_fraction = 0.02f,
        .ambience          = 0.0f,
        .ambience_knee_cct = 4000.0f,
        .cct_min           = 0.0f,
        .cct_max           = 0.0f,
        .prefer_bright     = true,
        .bright_max_spread_mired = 60.0f,
    };
}

/* Accumulators of one weighting over the bins inside the band. */
typedef struct CRAWAwbAccum {
    double total;      /* mass of the weighting over every bin */
    double zone;       /* mass inside the band */
    double w, wm, wmm, wt;
} CRAWAwbAccum;

static void craw_awb_accum(CRAWAwbAccum *a, double mass, bool in_zone, double falloff,
                           double mired, double tint)
{
    a->total += mass;
    if (!in_zone)
        return;
    a->zone += mass;

    {
        double w = mass * falloff;
        if (!(w > 0.0))
            return;
        a->w   += w;
        a->wm  += w * mired;
        a->wmm += w * mired * mired;
        a->wt  += w * tint;
    }
}

static void craw_awb_candidate(const CRAWAwbAccum *a, double min_zone_fraction,
                               double max_spread, CRAWAwbCandidate *c)
{
    double in_zone, mired, spread;

    *c = (CRAWAwbCandidate) { .valid = false };
    if (!(a->total > 0.0) || !(a->w > 0.0))
        return;

    in_zone = a->zone / a->total;
    mired = a->wm / a->w;

    /* Variance of the candidates along the locus. Clamped at zero because
     * the sum of squares form is one cancellation away from a tiny
     * negative number when every candidate sits in the same bin. */
    spread = a->wmm / a->w - mired * mired;
    spread = spread > 0.0 ? sqrt(spread) : 0.0;

    *c = (CRAWAwbCandidate) {
        .valid            = in_zone >= min_zone_fraction &&
                            (max_spread <= 0.0 || spread <= max_spread),
        .illuminant       = { .cct = (float)(1e6 / mired), .tint = (float)(a->wt / a->w) },
        .in_zone_fraction = (float)in_zone,
        .spread_mired     = (float)spread,
    };
}

int craw_awb_estimate(const CRAWProfile *p, const CRAWAwbHistogram *hist,
                      const CRAWAwbParams *params, CRAWAwbEstimate *out)
{
    CRAWAwbParams defaults;
    CRAWAwbAccum zone = { 0 }, bright = { 0 };
    const CRAWAwbCandidate *chosen;
    double span, cct, conf, cct_min, cct_max;
    int ret;

    if (!p || !hist || !out)
        return CRAW_ERROR(EINVAL);
    if (!hist->count || !hist->weight || hist->bins < 2)
        return CRAW_ERROR(EINVAL);
    if (!(hist->log_max > hist->log_min))
        return CRAW_ERROR(EINVAL);
    if (p->nb_wb_knots < 2)
        return CRAW_ERROR(ENOTSUP);

    if (!params) {
        craw_awb_params_init(&defaults);
        params = &defaults;
    }
    if (!(params->band > 0.0f) || !(params->sigma > 0.0f))
        return CRAW_ERROR(EINVAL);

    *out = (CRAWAwbEstimate) {
        .valid = false,
        .illuminant = { .cct = 0.0f, .tint = 0.0f },
        .source = CRAW_AWB_SOURCE_NONE,
    };

    span = (double)hist->log_max - hist->log_min;

    for (uint32_t iy = 0; iy < hist->bins; iy++) {
        double y = hist->log_min + span * (iy + 0.5) / hist->bins;

        for (uint32_t ix = 0; ix < hist->bins; ix++) {
            uint32_t idx = iy * hist->bins + ix;
            double x, d, falloff, mired;
            float sample_rg, sample_bg;
            bool in_zone;
            CRAWIlluminant il;

            if (!hist->count[idx])
                continue;

            x = hist->log_min + span * (ix + 0.5) / hist->bins;

            /* The bin centre as a raw neutral: its nearest locus sample
             * gives the Kelvin it would be, and the residual its distance
             * from anything neutral at all. The sample is recovered from
             * the Kelvin instead of being returned directly, which keeps
             * this to the public profile API. */
            ret = craw_profile_neutral_to_illuminant(p, (float)exp(x),
                                                     (float)exp(y), &il);
            if (ret < 0)
                return ret;

            ret = craw_profile_illuminant_to_neutral(p,
                    &(CRAWIlluminant) { .cct = il.cct, .tint = 0.0f },
                    &sample_rg, &sample_bg);
            if (ret < 0)
                return ret;

            {
                double dx = x - log((double)sample_rg);
                double dy = y - log((double)sample_bg);
                d = sqrt(dx * dx + dy * dy);
            }

            in_zone = d < params->band;
            falloff = exp(-(d / params->sigma) * (d / params->sigma));
            mired = 1e6 / il.cct;

            craw_awb_accum(&zone, hist->weight[idx] / 65536.0, in_zone, falloff,
                           mired, il.tint);
            if (hist->weight_bright)
                craw_awb_accum(&bright, hist->weight_bright[idx] / 65536.0, in_zone,
                               falloff, mired, il.tint);
        }
    }

    craw_awb_candidate(&zone, params->min_zone_fraction, 0.0, &out->zone);
    craw_awb_candidate(&bright, params->min_zone_fraction,
                       params->bright_max_spread_mired, &out->bright);

    /* A light source or its glare in frame is the illuminant itself; a
     * surface is only ever a guess about it. */
    if (params->prefer_bright && out->bright.valid) {
        chosen = &out->bright;
        out->source = CRAW_AWB_SOURCE_BRIGHT;
    } else if (out->zone.in_zone_fraction > 0.0f || out->zone.valid) {
        chosen = &out->zone;
        out->source = CRAW_AWB_SOURCE_ZONE;
    } else {
        return 0;
    }

    if (!(chosen->illuminant.cct > 0.0f))
        return 0;

    conf = CRAW_CLAMP(chosen->in_zone_fraction / CRAW_AWB_CONF_ZONE_FRACTION, 0.0, 1.0) *
           CRAW_CLAMP(1.0 - chosen->spread_mired / CRAW_AWB_CONF_SPREAD_MIRED, 0.0, 1.0);

    cct = chosen->illuminant.cct;

    /* Ambience: leave some of the warmth of a warm light in place rather
     * than correcting a candle to daylight. */
    if (params->ambience > 0.0f && cct < params->ambience_knee_cct)
        cct += (double)params->ambience * (params->ambience_knee_cct - cct);

    cct_min = params->cct_min > 0.0f ? params->cct_min : p->wb_knots[0].cct;
    cct_max = params->cct_max > 0.0f ? params->cct_max :
              p->wb_knots[p->nb_wb_knots - 1].cct;
    if (cct_max < cct_min)
        cct_max = cct_min;
    cct = CRAW_CLAMP(cct, cct_min, cct_max);

    out->valid            = chosen->valid;
    out->illuminant       = (CRAWIlluminant) { .cct = (float)cct, .tint = chosen->illuminant.tint };
    out->confidence       = (float)conf;
    out->in_zone_fraction = chosen->in_zone_fraction;
    out->spread_mired     = chosen->spread_mired;

    return 0;
}

int craw_awb_histogram_from_rgb(const float *rgb, uint32_t width,
                                uint32_t height, size_t stride,
                                int channels, const float wb_applied[3],
                                uint32_t *count, uint32_t *weight,
                                  uint32_t *weight_bright)
{
    const uint32_t bins = CRAW_AWB_HIST_BINS;
    const float lmin = CRAW_AWB_HIST_LOG_MIN;
    const float lmax = CRAW_AWB_HIST_LOG_MAX;

    if (!rgb || !wb_applied || !count || !weight || !weight_bright)
        return CRAW_ERROR(EINVAL);
    if (channels != 3 && channels != 4)
        return CRAW_ERROR(EINVAL);
    if (stride < (size_t)width * (size_t)channels)
        return CRAW_ERROR(EINVAL);
    if (!(wb_applied[0] > 0.0f) || !(wb_applied[2] > 0.0f))
        return CRAW_ERROR(EINVAL);

    memset(count, 0, (size_t)bins * bins * sizeof(*count));
    memset(weight, 0, (size_t)bins * bins * sizeof(*weight));
    memset(weight_bright, 0, (size_t)bins * bins * sizeof(*weight_bright));

    /* One sample per 8x8 block, exactly like the GPU pass, which reduces a
     * block per invocation. Blocks that hang over the right or bottom edge
     * average only the pixels that exist. */
    for (uint32_t by = 0; by < height; by += 8) {
        uint32_t y1 = CRAW_MIN(by + 8, height);

        for (uint32_t bx = 0; bx < width; bx += 8) {
            uint32_t x1 = CRAW_MIN(bx + 8, width);
            float acc[3] = { 0.0f, 0.0f, 0.0f };
            float n = (float)((y1 - by) * (x1 - bx));
            float r, g, b, rr, bb, x, y, w;
            uint32_t ix, iy;

            for (uint32_t py = by; py < y1; py++) {
                const float *row = rgb + (size_t)py * stride;

                for (uint32_t px = bx; px < x1; px++) {
                    acc[0] += row[(size_t)px * channels + 0];
                    acc[1] += row[(size_t)px * channels + 1];
                    acc[2] += row[(size_t)px * channels + 2];
                }
            }

            r = acc[0] / n;
            g = acc[1] / n;
            b = acc[2] / n;

            /* Clipped samples have lost their chromaticity and near black
             * ones never had any. */
            if (CRAW_MAX(CRAW_MAX(r, g), b) >= 0.98f || g < 0.002f)
                continue;

            rr = (r / wb_applied[0]) / g;
            bb = (b / wb_applied[2]) / g;

            /* Written so that a NaN from a negative debayer overshoot lands
             * in the first bin instead of indexing out of the array. */
            x = logf(rr);
            y = logf(bb);
            x = x > lmin ? (x < lmax ? x : lmax) : lmin;
            y = y > lmin ? (y < lmax ? y : lmax) : lmin;

            {
                float span = lmax - lmin;
                int bin_x = (int)floorf((x - lmin) / span * bins);
                int bin_y = (int)floorf((y - lmin) / span * bins);

                ix = (uint32_t)CRAW_CLAMP(bin_x, 0, (int)bins - 1);
                iy = (uint32_t)CRAW_CLAMP(bin_y, 0, (int)bins - 1);
            }

            w = CRAW_MIN(g, 0.5f);

            count[iy * bins + ix]  += 1;
            weight[iy * bins + ix] += (uint32_t)roundf(w * 65536.0f);
            weight_bright[iy * bins + ix] += (uint32_t)roundf(g * g * g * g * 65536.0f);
        }
    }

    return 0;
}
