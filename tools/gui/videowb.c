/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* The white balance of an already developed clip.
 *
 * A raw clip is balanced by libcraw, from the sensor's own ratios and the
 * camera's tables. A developed clip was balanced in the camera, and what is
 * left to do is undo that balance and apply another, on a picture that is
 * RGB in some set of primaries - BT.709, BT.2020, V-Gamut - with the
 * camera's tone curve on it. The model is three steps.
 *
 * The camera adapted the scene's white to the space's. Which white it
 * adapted from is the colour temperature the clip records: the maker note's
 * (pana.c), which on an auto clip is the one the camera settled on, or the
 * XML's in Kelvin mode. A colour temperature names a point on the Planckian
 * locus, so that is where it is taken from. A clip that records none is
 * taken as balanced for the space's own white, and the correction is then
 * relative to that.
 *
 * So under an illuminant of T Kelvin the camera renders a neutral surface as
 * the Planckian white of T, Bradford-adapted from the recorded white to the
 * space's, in the linear RGB libplacebo's LINEAR stage holds. Sampled from
 * 2000 to 12000 K, that is this clip's white balance locus, and written as
 * a CRAWProfile's white balance table it gives the developed clip the same
 * notion of an illuminant a raw clip has: a temperature along the locus and
 * a tint across it, in the same log-ratio units, with libcraw's own locus
 * mapping for the spot pick and its auto white balance estimator working
 * on it unchanged. The linear RGB plays the part of the sensor's.
 *
 * Correcting to an illuminant is then a Bradford adaptation in XYZ, from
 * the colour the camera gave that illuminant's neutral to the space's
 * white: a 3x3 on the linear picture, the identity for the illuminant the
 * clip recorded, which gpu.c folds the video exposure into and applies at
 * the LINEAR stage.
 *
 * What a picture style does beyond the balance - its tone curve, its
 * saturation - is not undone, because nothing records it: on anything but
 * a log recording this is the correction any grading tool makes on a
 * developed picture, not a development. */

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libplacebo/colorspace.h>

#include <craw/awb.h>
#include <craw/profile.h>

#include "view.h"

/* The locus is sampled evenly in mired between these. libcraw interpolates
 * its white balance table linearly in Kelvin, and sixteen knots spaced in
 * mired keep that within a fraction of a mired of the curve. */
#define VIEW_VWB_CCT_MIN 2000.0
#define VIEW_VWB_CCT_MAX 12000.0

/* A space's RGB to XYZ relative to its own white, the inverse, and the
 * white. */
struct view_vwb_space {
    double to_xyz[3][3];
    double from_xyz[3][3];
    float white[2];
};

static void view_vwb_mul(const double a[3][3], const double b[3][3],
                         double out[3][3])
{
    double t[3][3];

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            t[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];

    memcpy(out, t, sizeof(t));
}

static void view_vwb_apply(const double m[3][3], const double v[3], double out[3])
{
    for (int i = 0; i < 3; i++)
        out[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
}

static bool view_vwb_invert(const double m[3][3], double out[3][3])
{
    double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
               - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
               + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if (!(fabs(det) > 1e-12))
        return false;

    out[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det;
    out[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det;
    out[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det;
    out[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det;
    out[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det;
    out[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det;
    out[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det;
    out[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det;
    out[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det;

    return true;
}

static void view_vwb_cat(const float cat[3][3], double out[3][3])
{
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            out[i][j] = cat[i][j];
}

static bool view_vwb_space(enum pl_color_primaries prim, struct view_vwb_space *sp)
{
    const struct pl_raw_primaries *rp = pl_raw_primaries_get(prim);
    pl_matrix3x3 m;

    if (!rp)
        return false;

    m = pl_get_rgb2xyz_matrix(rp);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            sp->to_xyz[i][j] = m.m[i][j];

    sp->white[0] = rp->white.x;
    sp->white[1] = rp->white.y;

    return view_vwb_invert(sp->to_xyz, sp->from_xyz);
}

/* The temperatures the locus is sampled at: evenly in mired, plus the
 * recorded one itself, so that the recorded illuminant is a knot and
 * evaluates to exactly the space's white - between two knots libcraw's
 * interpolation is a few tenths of a per cent off the curve, which is
 * nothing anywhere but at the one point that has to be the identity. A
 * grid point within a few mired of it gives way. Ascending in Kelvin. */
static int view_vwb_ccts(int recorded, double *cct)
{
    const int grid = CRAW_PROFILE_MAX_KNOTS - 1;
    const double warm = 1.0e6 / VIEW_VWB_CCT_MIN, cool = 1.0e6 / VIEW_VWB_CCT_MAX;
    const double rec = recorded > 0 ? 1.0e6 / recorded : 0.0;
    bool placed = recorded <= 0;
    int n = 0;

    for (int i = 0; i < grid; i++) {
        double mired = warm - (warm - cool) * i / (grid - 1);

        if (!placed && rec > mired) {
            cct[n++] = recorded;
            placed = true;
        }
        if (recorded > 0 && fabs(rec - mired) < 4.0)
            continue;
        cct[n++] = 1.0e6 / mired;
    }
    if (!placed)
        cct[n++] = recorded;

    return n;
}

/* The white balance table of the clip in `prim`, as the header comment
 * derives it. A knot whose neutral falls outside the space - a very warm
 * light seen from a very cool recording, in BT.709 - has no gains and is
 * left out; the ones either side of the recorded white always remain. */
static bool view_vwb_profile_in(const struct view_state *st,
                                const struct view_vwb_space *sp,
                                CRAWProfile *p)
{
    double adapt[3][3], m[3][3], ccts[CRAW_PROFILE_MAX_KNOTS];
    float rec[2], cat[3][3];
    int nb;

    if (st->dec.wb_kelvin > 0) {
        if (craw_planckian_xy((float)st->dec.wb_kelvin, rec) < 0)
            return false;
    } else {
        rec[0] = sp->white[0];
        rec[1] = sp->white[1];
    }

    if (craw_bradford_adaptation(rec, sp->white, cat) < 0)
        return false;
    view_vwb_cat(cat, adapt);
    view_vwb_mul(sp->from_xyz, adapt, m);

    craw_profile_init(p);
    snprintf(p->name, sizeof(p->name), "developed clip");

    nb = view_vwb_ccts(st->dec.wb_kelvin, ccts);
    for (int i = 0; i < nb; i++) {
        double cct = ccts[i], xyz[3], rgb[3];
        float xy[2];
        CRAWWbKnot *k;

        if (craw_planckian_xy((float)cct, xy) < 0)
            return false;

        xyz[0] = xy[0] / xy[1];
        xyz[1] = 1.0;
        xyz[2] = (1.0 - xy[0] - xy[1]) / xy[1];
        view_vwb_apply(m, xyz, rgb);

        if (!(rgb[0] > 0.0 && rgb[1] > 0.0 && rgb[2] > 0.0))
            continue;

        k = &p->wb_knots[p->nb_wb_knots++];
        k->cct = (float)cct;
        k->wb[0] = (float)(rgb[1] / rgb[0]);
        k->wb[1] = 1.0f;
        k->wb[2] = (float)(rgb[1] / rgb[2]);
    }

    /* The recorded white renders as the space's, which is (1, 1, 1). */
    p->as_shot_wb[0] = p->as_shot_wb[1] = p->as_shot_wb[2] = 1.0f;
    p->as_shot_cct = (float)st->dec.wb_kelvin;

    return p->nb_wb_knots >= 2;
}

bool view_vwb_profile(const struct view_state *st, enum pl_color_primaries prim,
                      CRAWProfile *p)
{
    struct view_vwb_space sp;

    return view_vwb_space(prim, &sp) && view_vwb_profile_in(st, &sp, p);
}

bool view_vwb_correction(const struct view_state *st, enum pl_color_primaries prim,
                         const CRAWIlluminant *il, pl_matrix3x3 *out)
{
    struct view_vwb_space sp;
    CRAWProfile p;
    double rgb[3], xyz[3], cat_d[3][3], m[3][3], sum;
    float rg, bg, xy[2], cat[3][3];

    if (!view_vwb_space(prim, &sp) || !view_vwb_profile_in(st, &sp, &p) ||
        craw_profile_illuminant_to_neutral(&p, il, &rg, &bg) < 0)
        return false;

    /* The colour the camera gave this illuminant's neutral. */
    rgb[0] = rg;
    rgb[1] = 1.0;
    rgb[2] = bg;
    view_vwb_apply(sp.to_xyz, rgb, xyz);
    sum = xyz[0] + xyz[1] + xyz[2];
    if (!(sum > 0.0) || !(xyz[1] > 0.0))
        return false;
    xy[0] = (float)(xyz[0] / sum);
    xy[1] = (float)(xyz[1] / sum);

    if (craw_bradford_adaptation(xy, sp.white, cat) < 0)
        return false;
    view_vwb_cat(cat, cat_d);

    view_vwb_mul(cat_d, sp.to_xyz, m);
    view_vwb_mul(sp.from_xyz, m, m);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            out->m[i][j] = (float)m[i][j];

    return true;
}

bool view_vwb_as_shot(const struct view_state *st, CRAWIlluminant *out)
{
    enum pl_color_primaries prim;
    CRAWProfile p;

    return view_gpu_source_space(&st->gpu, &prim, NULL) &&
           view_vwb_profile(st, prim, &p) &&
           craw_profile_neutral_to_illuminant(&p, 1.0f, 1.0f, out) >= 0;
}

bool view_vwb_target(const struct view_state *st, CRAWIlluminant *out)
{
    if (st->dec.raw || !st->sections.wb)
        return false;

    switch (st->look.wb) {
    case VIEW_WB_MANUAL:
        *out = st->look.manual;
        return true;
    case VIEW_WB_AUTO:
        if (!st->awb.have_illum)
            return false;
        *out = st->awb.illum;
        return true;
    default:
        return false;
    }
}

bool view_vwb_linear(const struct view_state *st, pl_matrix3x3 *out)
{
    float gain = view_video_exposure(st);
    enum pl_color_primaries prim;
    CRAWIlluminant il;
    bool wb;

    wb = view_vwb_target(st, &il) &&
         view_gpu_source_space(&st->gpu, &prim, NULL) &&
         view_vwb_correction(st, prim, &il, out);

    /* The recorded illuminant is a knot of the locus and comes back as the
     * identity to within float rounding. That is no correction at all, and
     * is left off rather than attached: a hook at the LINEAR stage costs a
     * pass through an intermediate, which is a picture of its own, and
     * picking Manual - which starts from the balance on screen - must not
     * change what is on screen. */
    if (wb) {
        float dev = 0.0f;

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                dev = fmaxf(dev, fabsf(out->m[i][j] - (i == j ? 1.0f : 0.0f)));
        wb = dev > 1e-5f;
    }
    if (!wb)
        *out = pl_matrix3x3_identity;

    if (gain != 1.0f)
        pl_matrix3x3_scale(out, gain);

    return wb || gain != 1.0f;
}

/* ------------------------------------------------- the auto white balance */

/* One estimate from a collected probe: the histogram libcraw's GPU pass
 * would have built, from the linear picture scaled so that the source's
 * own peak - where a channel clips - sits at the histogram's clip of 1.0,
 * then the estimator on this clip's locus and the smoother the raw path
 * uses. True when the track moved. */
static bool view_vwb_estimate(struct view_state *st, float *rgba, int w, int h)
{
    struct view_vwb *v = &st->vwb;
    const size_t bins = (size_t)CRAW_AWB_HIST_BINS * CRAW_AWB_HIST_BINS;
    static const float unity[3] = { 1.0f, 1.0f, 1.0f };
    enum pl_color_primaries prim;
    CRAWAwbHistogram hist;
    CRAWIlluminant out, before = st->awb.illum;
    bool had = st->awb.have_illum;
    CRAWProfile p;
    float peak;
    int ret;

    if (!view_gpu_source_space(&st->gpu, &prim, &peak) || !(peak > 0.0f) ||
        !view_vwb_profile(st, prim, &p))
        return false;

    if (!v->hist) {
        v->hist = calloc(3 * bins, sizeof(*v->hist));
        if (!v->hist)
            return false;
    }

    for (size_t i = 0; i < (size_t)w * h * 4; i++)
        rgba[i] /= peak;

    if (craw_awb_histogram_from_rgb(rgba, (uint32_t)w, (uint32_t)h,
                                    (size_t)w * 4, 4, unity, v->hist,
                                    v->hist + bins, v->hist + 2 * bins) < 0)
        return false;

    hist = (CRAWAwbHistogram) {
        .bins          = CRAW_AWB_HIST_BINS,
        .log_min       = CRAW_AWB_HIST_LOG_MIN,
        .log_max       = CRAW_AWB_HIST_LOG_MAX,
        .count         = v->hist,
        .weight        = v->hist + bins,
        .weight_bright = v->hist + 2 * bins,
    };

    if (craw_awb_estimate(&p, &hist, &st->awb.params, &st->awb.last) < 0)
        return false;

    st->awb.clock += view_frame_duration(st);
    ret = craw_awb_smoother_push(st->awb.smoother, st->awb.clock,
                                 &st->awb.last, false, &out);
    if (ret >= 0) {
        st->awb.illum = out;
        st->awb.have_illum = true;
    }

    if (st->cfg.verbose)
        view_log(st, "awb: frame %" PRId64 " est %s cct %5.0f tint %+.4f conf %.2f"
                     " -> track %s cct %5.0f tint %+.4f\n",
                 st->index, st->awb.last.valid ? "ok" : "invalid",
                 st->awb.last.illuminant.cct, st->awb.last.illuminant.tint,
                 st->awb.last.confidence, ret >= 0 ? "moved" : "held",
                 st->awb.have_illum ? st->awb.illum.cct : 0.0f,
                 st->awb.have_illum ? st->awb.illum.tint : 0.0f);

    return st->awb.have_illum &&
           (!had || fabsf(out.cct - before.cct) >= 0.5f ||
            fabsf(out.tint - before.tint) >= 1e-4f);
}

void view_vwb_step(struct view_state *st, bool wait)
{
    struct view_vwb *v = &st->vwb;
    bool on = !st->dec.raw && st->sections.wb && st->look.wb == VIEW_WB_AUTO;
    float *rgba;
    int w, h, ret;

    ret = view_gpu_probe_poll(&st->gpu, wait, &rgba, &w, &h);
    if (ret < 0) {
        fprintf(stderr, "cannot read the auto white balance probe back\n");
        v->converging = false;
    } else if (ret > 0) {
        bool mine = v->probe_gen == v->gen && on;

        v->converging = false;
        /* A probe of the picture a seek left behind, like the raw path's
         * one stale histogram (struct view_awb). */
        if (mine && st->awb.skip > 0) {
            st->awb.skip--;
            mine = false;
        }
        if (mine && view_vwb_estimate(st, rgba, w, h)) {
            /* While paused the same picture is probed again until the
             * track settles, which is how a still converges; while playing
             * the next picture does it anyway. */
            v->converging = true;
            st->present_due = true;
        }
    }

    if (!on || st->gpu.probe_busy || !st->gpu.have_mapped)
        return;
    if (st->gpu.pic_serial == v->probed_serial && v->probe_gen == v->gen &&
        !v->converging)
        return;

    if (view_gpu_probe_start(&st->gpu) < 0)
        return;

    v->probed_serial = st->gpu.pic_serial;
    v->probe_gen = v->gen;
}
