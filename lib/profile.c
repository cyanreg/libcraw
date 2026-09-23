/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#include <math.h>
#include <string.h>

#include <craw/profile.h>

/* 1 / sqrt(2): the tint axis is the unit vector (-1, -1) / sqrt(2) in the
 * (ln R/G, ln B/G) plane, so a tint of t moves a neutral by t / sqrt(2) on
 * each axis and is compensated by a gain of exp(t / sqrt(2)) on R and B. */
#define CRAW_INV_SQRT2 0.70710678118654752440

/* Step of the locus resampling, in mired. Equal steps in mired rather than
 * in Kelvin because perceived white balance and the sensor ratios are both
 * close to linear in mired, so this spends samples where they matter. */
#define CRAW_LOCUS_STEP_MIRED 5.0

void craw_profile_init(CRAWProfile *p)
{
    if (!p)
        return;

    *p = (CRAWProfile) {
        .cfa = CRAW_CFA_RGGB,
        .black_level = { 0.0f, 0.0f, 0.0f, 0.0f },
        .white_level = { 1.0f, 1.0f, 1.0f, 1.0f },
        .as_shot_wb = { 1.0f, 1.0f, 1.0f },
        .as_shot_matrix = {
            { 1.0f, 0.0f, 0.0f },
            { 0.0f, 1.0f, 0.0f },
            { 0.0f, 0.0f, 1.0f },
        },
        .as_shot_cct = 0.0f,
        .gain = 1.0f,
        .as_shot_ei = 0.0f,
        .exposure_index = 0.0f,
        .nb_wb_knots = 0,
        .nb_matrix_knots = 0,
    };
}

static bool craw_finite_n(const float *v, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(v[i]))
            return false;
    }
    return true;
}

int craw_profile_validate(const CRAWProfile *p)
{
    if (!p)
        return CRAW_ERROR(EINVAL);

    if ((unsigned)p->cfa > CRAW_CFA_GBRG)
        return CRAW_ERROR(EINVAL);

    for (int s = 0; s < 4; s++) {
        if (!isfinite(p->black_level[s]) || !isfinite(p->white_level[s]))
            return CRAW_ERROR(EINVAL);
        if (p->black_level[s] < 0.0f || p->white_level[s] > 1.0f)
            return CRAW_ERROR(EINVAL);
        if (!(p->white_level[s] > p->black_level[s]))
            return CRAW_ERROR(EINVAL);
    }

    if (!craw_finite_n(p->as_shot_wb, 3))
        return CRAW_ERROR(EINVAL);
    for (int c = 0; c < 3; c++) {
        if (!(p->as_shot_wb[c] > 0.0f))
            return CRAW_ERROR(EINVAL);
    }

    if (!craw_finite_n(&p->as_shot_matrix[0][0], 9))
        return CRAW_ERROR(EINVAL);

    /* A singular camera matrix would make the whole colour stage collapse
     * to a plane, and is always a filling error rather than a real
     * characterization. */
    {
        const float (*m)[3] = p->as_shot_matrix;
        double det = (double)m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                   - (double)m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                   + (double)m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        if (!(fabs(det) > 1e-9))
            return CRAW_ERROR(EINVAL);
    }

    if (!isfinite(p->gain) || !(p->gain > 0.0f))
        return CRAW_ERROR(EINVAL);
    if (!isfinite(p->as_shot_cct) || p->as_shot_cct < 0.0f)
        return CRAW_ERROR(EINVAL);
    if (!isfinite(p->as_shot_ei) || p->as_shot_ei < 0.0f)
        return CRAW_ERROR(EINVAL);
    if (!isfinite(p->exposure_index) || p->exposure_index < 0.0f)
        return CRAW_ERROR(EINVAL);

    if (p->nb_wb_knots > CRAW_PROFILE_MAX_KNOTS ||
        p->nb_matrix_knots > CRAW_PROFILE_MAX_KNOTS)
        return CRAW_ERROR(EINVAL);

    for (uint32_t i = 0; i < p->nb_wb_knots; i++) {
        const CRAWWbKnot *k = &p->wb_knots[i];
        if (!isfinite(k->cct) || !(k->cct > 0.0f))
            return CRAW_ERROR(EINVAL);
        if (i && !(k->cct > p->wb_knots[i - 1].cct))
            return CRAW_ERROR(EINVAL);
        if (!craw_finite_n(k->wb, 3))
            return CRAW_ERROR(EINVAL);
        for (int c = 0; c < 3; c++) {
            if (!(k->wb[c] > 0.0f))
                return CRAW_ERROR(EINVAL);
        }
    }

    for (uint32_t i = 0; i < p->nb_matrix_knots; i++) {
        const CRAWMatrixKnot *k = &p->matrix_knots[i];
        if (!isfinite(k->cct) || !(k->cct > 0.0f))
            return CRAW_ERROR(EINVAL);
        if (i && !(k->cct > p->matrix_knots[i - 1].cct))
            return CRAW_ERROR(EINVAL);
        if (!craw_finite_n(&k->m[0][0], 9))
            return CRAW_ERROR(EINVAL);
    }

    if (p->name[sizeof(p->name) - 1] != '\0')
        return CRAW_ERROR(EINVAL);

    return 0;
}

/* Index of the interval of a knot table containing `cct`, plus the
 * interpolation factor inside it, clamped to the ends of the table. Both
 * tables are interpolated linearly in Kelvin because that is what the
 * camera and the vendor decoder do; interpolating in mired would be
 * smoother but would not reproduce the per-frame values the container
 * carries. */
static void craw_knot_interval(double cct_a, double cct_b, double cct,
                               double *frac)
{
    double f = 0.0;

    if (cct_b > cct_a)
        f = (cct - cct_a) / (cct_b - cct_a);

    *frac = CRAW_CLAMP(f, 0.0, 1.0);
}

static void craw_wb_at(const CRAWProfile *p, double cct, double wb[3])
{
    uint32_t i = 0;
    double f;

    if (p->nb_wb_knots == 1) {
        for (int c = 0; c < 3; c++)
            wb[c] = p->wb_knots[0].wb[c];
        return;
    }

    while (i + 2 < p->nb_wb_knots && cct >= p->wb_knots[i + 1].cct)
        i++;

    craw_knot_interval(p->wb_knots[i].cct, p->wb_knots[i + 1].cct, cct, &f);

    for (int c = 0; c < 3; c++) {
        double a = p->wb_knots[i].wb[c];
        double b = p->wb_knots[i + 1].wb[c];
        wb[c] = a + (b - a) * f;
    }
}

static void craw_matrix_at(const CRAWProfile *p, double cct, float m[3][3])
{
    uint32_t i = 0;
    double f;

    if (p->nb_matrix_knots == 1) {
        memcpy(m, p->matrix_knots[0].m, sizeof(float) * 9);
        return;
    }

    while (i + 2 < p->nb_matrix_knots && cct >= p->matrix_knots[i + 1].cct)
        i++;

    craw_knot_interval(p->matrix_knots[i].cct, p->matrix_knots[i + 1].cct,
                       cct, &f);

    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            double a = p->matrix_knots[i].m[r][c];
            double b = p->matrix_knots[i + 1].m[r][c];
            m[r][c] = (float)(a + (b - a) * f);
        }
    }
}

int craw_profile_eval(const CRAWProfile *p, enum CRAWWbMode mode,
                      const CRAWIlluminant *illum, CRAWColorParams *out)
{
    if (!p || !out)
        return CRAW_ERROR(EINVAL);

    switch (mode) {
    case CRAW_WB_AS_SHOT:
        memcpy(out->wb, p->as_shot_wb, sizeof(out->wb));
        memcpy(out->cam_to_xyz, p->as_shot_matrix, sizeof(out->cam_to_xyz));
        break;
    case CRAW_WB_REBALANCE: {
        double wb[3], cct, tint_gain;

        if (!illum)
            return CRAW_ERROR(EINVAL);
        if (!p->nb_wb_knots)
            return CRAW_ERROR(ENOTSUP);
        if (!isfinite(illum->cct) || !isfinite(illum->tint))
            return CRAW_ERROR(EINVAL);

        cct = CRAW_CLAMP((double)illum->cct,
                         (double)p->wb_knots[0].cct,
                         (double)p->wb_knots[p->nb_wb_knots - 1].cct);

        craw_wb_at(p, cct, wb);

        tint_gain = exp((double)illum->tint * CRAW_INV_SQRT2);
        wb[0] *= tint_gain;
        wb[2] *= tint_gain;

        for (int c = 0; c < 3; c++)
            out->wb[c] = (float)wb[c];

        if (p->nb_matrix_knots) {
            uint32_t nb = p->nb_matrix_knots;
            double mcct = CRAW_CLAMP(cct, (double)p->matrix_knots[0].cct,
                                     (double)p->matrix_knots[nb - 1].cct);

            craw_matrix_at(p, mcct, out->cam_to_xyz);
        } else {
            memcpy(out->cam_to_xyz, p->as_shot_matrix, sizeof(out->cam_to_xyz));
        }
        break;
    }
    default:
        return CRAW_ERROR(EINVAL);
    }

    out->gain = p->gain;
    if (p->as_shot_ei > 0.0f && p->exposure_index > 0.0f)
        out->gain *= p->exposure_index / p->as_shot_ei;

    return 0;
}

/* How far (lx, ly) - the logarithms of a neutral's raw R/G and B/G - is
 * from the locus at `cct` across the green direction: the residual, point
 * minus locus, dotted with (1, -1). Zero where the point lies from the
 * locus along (1, 1), the only direction a tint moves a neutral. */
static double craw_locus_across(const CRAWProfile *p, double lx, double ly,
                                double cct, double res[2])
{
    double wb[3];

    craw_wb_at(p, cct, wb);
    res[0] = lx + log(wb[0]);
    res[1] = ly + log(wb[2]);

    return res[0] - res[1];
}

/* Finds the point of the locus that (lx, ly) lies from along the green
 * direction, and the residual from it, which is then along that direction
 * too: exactly the displacement craw_profile_illuminant_to_neutral() gives
 * a tint, so the two functions invert each other wherever the locus
 * reaches. The nearest point would not do that - its residual is
 * perpendicular to the locus, which is only the green direction where the
 * locus happens to run at right angles to it, and whatever part of the
 * residual is not along green is lost when it becomes a tint.
 *
 * The locus is walked from warm to cool at the knots and in steps of at
 * most CRAW_LOCUS_STEP_MIRED between them, for the first sample pair the
 * crossing lies between, and the crossing is then found by bisection on
 * the profile's own interpolation of the gains. Every knot is a sample, so
 * a neutral measured at a knot maps back to that knot's Kelvin. A point
 * beyond either end, across the green direction, goes to that end. */
static int craw_locus_green(const CRAWProfile *p, double lx, double ly,
                            double *cct_out, double res[2])
{
    double a_cct = 0.0, a_g = 0.0, b_cct = 0.0, m_a, m_b;
    bool have_a = false, found = false;

    if (!p)
        return CRAW_ERROR(EINVAL);
    if (p->nb_wb_knots < 2)
        return CRAW_ERROR(ENOTSUP);

    for (uint32_t i = 0; i < p->nb_wb_knots && !found; i++) {
        const bool last = i + 1 == p->nb_wb_knots;
        const double mired_a = 1e6 / p->wb_knots[i].cct;
        const double span = last ? 0.0 : mired_a - 1e6 / p->wb_knots[i + 1].cct;
        int steps = last ? 1 : (int)ceil(span / CRAW_LOCUS_STEP_MIRED);

        if (steps < 1)
            steps = 1;

        /* [knot i, knot i + 1), and the last knot on its own. */
        for (int j = 0; j < steps; j++) {
            double cct = 1e6 / (mired_a - span * ((double)j / steps));
            double g = craw_locus_across(p, lx, ly, cct, res);

            if (g == 0.0) {
                *cct_out = cct;
                return 0;
            }
            if (have_a && (g > 0.0) != (a_g > 0.0)) {
                b_cct = cct;
                found = true;
                break;
            }

            a_cct = cct;
            a_g = g;
            have_a = true;
        }
    }

    /* No crossing: the point is past an end of the locus, across the green
     * direction, and goes to the end it is nearer to in that measure. */
    if (!found) {
        double r_warm[2], r_cool[2];
        double warm = p->wb_knots[0].cct;
        double cool = p->wb_knots[p->nb_wb_knots - 1].cct;
        bool w = fabs(craw_locus_across(p, lx, ly, warm, r_warm)) <=
                 fabs(craw_locus_across(p, lx, ly, cool, r_cool));

        *cct_out = w ? warm : cool;
        res[0] = w ? r_warm[0] : r_cool[0];
        res[1] = w ? r_warm[1] : r_cool[1];
        return 0;
    }

    /* Halved in mired, the way the samples were spaced; sixty halvings of
     * a five mired interval are past the double's own resolution. */
    m_a = 1e6 / a_cct;
    m_b = 1e6 / b_cct;
    for (int k = 0; k < 60; k++) {
        double m = 0.5 * (m_a + m_b);
        double g = craw_locus_across(p, lx, ly, 1e6 / m, res);

        if (g == 0.0) {
            m_a = m_b = m;
            break;
        }
        if ((g > 0.0) == (a_g > 0.0)) {
            m_a = m;
            a_g = g;
        } else {
            m_b = m;
        }
    }

    *cct_out = 1e6 / (0.5 * (m_a + m_b));
    craw_locus_across(p, lx, ly, *cct_out, res);

    return 0;
}

int craw_profile_neutral_to_illuminant(const CRAWProfile *p,
                                       float raw_rg, float raw_bg,
                                       CRAWIlluminant *out)
{
    double cct, res[2];
    int ret;

    if (!out)
        return CRAW_ERROR(EINVAL);
    if (!(raw_rg > 0.0f) || !(raw_bg > 0.0f) ||
        !isfinite(raw_rg) || !isfinite(raw_bg))
        return CRAW_ERROR(EINVAL);

    ret = craw_locus_green(p, log((double)raw_rg), log((double)raw_bg),
                           &cct, res);
    if (ret < 0)
        return ret;

    out->cct = (float)cct;
    out->tint = (float)(-(res[0] + res[1]) * CRAW_INV_SQRT2);

    return 0;
}

int craw_profile_illuminant_to_neutral(const CRAWProfile *p,
                                       const CRAWIlluminant *illum,
                                       float *raw_rg, float *raw_bg)
{
    double wb[3], cct, tint_gain;

    if (!p || !illum || !raw_rg || !raw_bg)
        return CRAW_ERROR(EINVAL);
    if (!p->nb_wb_knots)
        return CRAW_ERROR(ENOTSUP);
    if (!isfinite(illum->cct) || !isfinite(illum->tint))
        return CRAW_ERROR(EINVAL);

    cct = CRAW_CLAMP((double)illum->cct,
                     (double)p->wb_knots[0].cct,
                     (double)p->wb_knots[p->nb_wb_knots - 1].cct);

    craw_wb_at(p, cct, wb);

    /* The inverse of the gains craw_profile_eval() would apply: a neutral
     * under this illuminant is exactly what those gains undo. */
    tint_gain = exp(-(double)illum->tint * CRAW_INV_SQRT2);

    *raw_rg = (float)(tint_gain / wb[0]);
    *raw_bg = (float)(tint_gain / wb[2]);

    return 0;
}
