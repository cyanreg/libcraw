/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <craw/awb.h>

/* The track is kept in mired rather than in Kelvin: a fixed step in mired
 * is a roughly constant visual change of white balance, so one time
 * constant and one slew limit behave the same at 3000 K and at 8000 K. */
struct CRAWAwbSmoother {
    CRAWAwbSmootherParams params;

    bool have_state;
    double mired;
    double tint;
    double last_pts;
};

void craw_awb_smoother_params_init(CRAWAwbSmootherParams *p)
{
    if (!p)
        return;

    *p = (CRAWAwbSmootherParams) {
        .time_constant        = 1.0f,
        .hysteresis_mired     = 3.0f,
        .max_slew_mired_per_s = 30.0f,
        .min_confidence       = 0.2f,
    };
}

int craw_awb_smoother_create(const CRAWAwbSmootherParams *params,
                             CRAWAwbSmoother **sm)
{
    CRAWAwbSmoother *s;
    int ret = 0;

    if (!sm)
        return CRAW_ERROR(EINVAL);

    *sm = NULL;

    s = calloc(1, sizeof(*s));
    if (!s)
        return CRAW_ERROR(ENOMEM);

    if (params)
        s->params = *params;
    else
        craw_awb_smoother_params_init(&s->params);

    if (!(s->params.time_constant > 0.0f) ||
        !(s->params.max_slew_mired_per_s > 0.0f) ||
        !(s->params.hysteresis_mired >= 0.0f) ||
        !isfinite(s->params.hysteresis_mired) ||
        !isfinite(s->params.time_constant) ||
        !isfinite(s->params.max_slew_mired_per_s) ||
        !isfinite(s->params.min_confidence)) {
        ret = CRAW_ERROR(EINVAL);
        goto fail;
    }

    *sm = s;
    return 0;

fail:
    free(s);
    return ret;
}

void craw_awb_smoother_free(CRAWAwbSmoother **sm)
{
    if (!sm || !*sm)
        return;

    free(*sm);
    *sm = NULL;
}

void craw_awb_smoother_reset(CRAWAwbSmoother *sm)
{
    if (!sm)
        return;

    sm->have_state = false;
    sm->mired = 0.0;
    sm->tint = 0.0;
    sm->last_pts = 0.0;
}

void craw_awb_smoother_set(CRAWAwbSmoother *sm, double pts_s,
                           const CRAWIlluminant *illum)
{
    if (!sm || !illum || !isfinite(illum->cct) || !(illum->cct > 0.0f) ||
        !isfinite(illum->tint))
        return;

    sm->mired = 1e6 / (double)illum->cct;
    sm->tint = illum->tint;
    sm->have_state = true;
    sm->last_pts = pts_s;
}

int craw_awb_smoother_push(CRAWAwbSmoother *sm, double pts_s,
                           const CRAWAwbEstimate *est, bool scene_cut,
                           CRAWIlluminant *out)
{
    double dt, target_mired = 0.0, target_tint = 0.0;
    bool usable;

    if (!sm || !est || !out)
        return CRAW_ERROR(EINVAL);

    usable = est->valid && isfinite(est->illuminant.cct) &&
             est->illuminant.cct > 0.0f && isfinite(est->illuminant.tint);
    if (usable) {
        target_mired = 1e6 / (double)est->illuminant.cct;
        target_tint = est->illuminant.tint;
    }

    /* Clamped to a second: a long gap (a seek, a dropped stretch) should
     * not turn into an unbounded jump, and a non monotonic pts should not
     * run the filter backwards. */
    dt = sm->have_state ? pts_s - sm->last_pts : 0.0;
    if (!(dt > 0.0))
        dt = 0.0;
    if (dt > 1.0)
        dt = 1.0;

    /* The first usable estimate is the track, confident or not: there is
     * nothing better to show until a confident one arrives, and the gate
     * below keeps the unsure readings after it from moving the track
     * about (on a night scene they swing by thousands of kelvin from one
     * frame to the next). A cut restarts it only from a confident one. A
     * track that waited for a confident start was never there at all on a
     * clip the estimator is unsure of throughout. */
    if (usable && (!sm->have_state ||
                   (scene_cut && est->confidence >= sm->params.min_confidence))) {
        sm->mired = target_mired;
        sm->tint = target_tint;
        sm->have_state = true;
    } else if (!sm->have_state) {
        /* Nothing to report yet: no track and no estimate to start one. */
        return CRAW_ERROR(EAGAIN);
    } else if (usable && est->confidence >= sm->params.min_confidence) {
        double delta = target_mired - sm->mired;

        if (fabs(delta) >= (double)sm->params.hysteresis_mired) {
            double frac = 1.0 - exp(-dt / (double)sm->params.time_constant);
            double limit = (double)sm->params.max_slew_mired_per_s * dt;
            double step = delta * frac;

            /* The rate limit is applied as a scale on the fraction so that
             * the tint moves by the same share of its own error as the
             * temperature does. */
            if (fabs(step) > limit)
                frac *= limit / fabs(step);

            sm->mired += delta * frac;
            sm->tint += (target_tint - sm->tint) * frac;
        }
    }

    sm->last_pts = pts_s;

    out->cct = (float)(1e6 / sm->mired);
    out->tint = (float)sm->tint;

    return 0;
}
