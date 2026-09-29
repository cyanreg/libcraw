/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Sensor noise model: evaluation of a per-ISO table and calibration of one
 * from a frame. See DESIGN.md section 14.2. Nothing here touches the GPU;
 * the model it produces is what the denoise pass consumes. */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <craw/noise.h>
#include <craw/profile.h>

/* Median absolute value of a zero-mean Gaussian over its standard
 * deviation: Phi^-1(0.75). */
#define CRAW_NOISE_MAD_SCALE 0.674489750196081746f

/* Tukey biweight cutoff in units of the robust scale of the residuals. The
 * textbook 4.685 gives 95% efficiency on clean Gaussian residuals. */
#define CRAW_NOISE_TUKEY_C 4.685

/* Floor on that scale, in log-variance units. Without it a run of bins
 * that happen to agree to 1% would reject every other bin as an outlier;
 * 0.02 is well below the 10-15% a real frame reaches. */
#define CRAW_NOISE_SCALE_FLOOR 0.02

#define CRAW_NOISE_MAX_BINS 512

/* The padding tails of the public option structs are written as arithmetic
 * on the fields in front of them, so a field added without adjusting them
 * changes the size and breaks the ABI silently. These catch that. */
static_assert(sizeof(CRAWNoiseEstimateParams) == 64, "CRAWNoiseEstimateParams size");
static_assert(sizeof(CRAWNoiseEstimate) == 64, "CRAWNoiseEstimate size");

/* ------------------------------------------------------------ evaluation */

int craw_noise_model_eval(const CRAWNoiseKnot *knots, uint32_t nb_knots,
                          float iso, CRAWNoiseModel *out)
{
    const CRAWNoiseKnot *lo, *hi;
    uint32_t i;
    float t, la, lb, g;

    if (!knots || !nb_knots || !out)
        return CRAW_ERROR(EINVAL);

    lo = &knots[0];
    hi = &knots[nb_knots - 1];

    /* No exposure index to scale by: the lowest knot, as it always was. */
    if (!(iso > 0.0f) || !isfinite(iso)) {
        *out = (CRAWNoiseModel) { .a = lo->a, .b = lo->b };
        return 0;
    }

    /* Outside the table the model moves the way a gain moves it, which is
     * what an exposure index is on a sensor that writes its raw after the
     * gain. The shot term is the signal's own count scaled by the gain, so
     * a ~ iso. The read term is two noises, one ahead of the gain, which
     * the gain squares, and one after it, which the gain does not touch:
     * b = b0 + c * iso^2. At the top of a table the first is nearly all of
     * b, so above it b ~ iso^2; towards the bottom b tends to b0, so below
     * it b is kept. DESIGN.md 14.2 has the S1M2's numbers for both. */
    if (iso <= lo->iso) {
        *out = (CRAWNoiseModel) { .a = lo->a * (iso / lo->iso), .b = lo->b };
        return 0;
    }
    if (iso >= hi->iso) {
        g = iso / hi->iso;
        *out = (CRAWNoiseModel) { .a = hi->a * g, .b = hi->b * g * g };
        return 0;
    }

    for (i = 0; i + 1 < nb_knots; i++) {
        if (iso <= knots[i + 1].iso)
            break;
    }

    /* Between knots both terms follow power laws in the sensor's gain, so
     * the straight line to interpolate along is log against log. That also
     * bridges a change of gain mode, where the table steps instead of
     * following one law, without pretending to know where in the interval
     * the camera switches. A knot with a zero term has no logarithm; there
     * the neighbouring knot's own zero is the only sensible answer, and the
     * term is interpolated linearly instead so that it reaches zero
     * continuously. */
    if (!(knots[i + 1].iso > knots[i].iso)) {
        *out = (CRAWNoiseModel) { .a = knots[i].a, .b = knots[i].b };
        return 0;
    }

    t = (logf(iso) - logf(knots[i].iso)) /
        (logf(knots[i + 1].iso) - logf(knots[i].iso));

    if (knots[i].a > 0.0f && knots[i + 1].a > 0.0f) {
        la = logf(knots[i].a) + t * (logf(knots[i + 1].a) - logf(knots[i].a));
        out->a = expf(la);
    } else {
        out->a = knots[i].a + t * (knots[i + 1].a - knots[i].a);
    }

    if (knots[i].b > 0.0f && knots[i + 1].b > 0.0f) {
        lb = logf(knots[i].b) + t * (logf(knots[i + 1].b) - logf(knots[i].b));
        out->b = expf(lb);
    } else {
        out->b = knots[i].b + t * (knots[i + 1].b - knots[i].b);
    }

    return 0;
}

int craw_profile_noise_eval(const CRAWProfile *p, float iso, CRAWNoiseModel *out)
{
    if (!p || !out)
        return CRAW_ERROR(EINVAL);
    if (!p->nb_noise_knots)
        return CRAW_ERROR(ENOTSUP);

    if (!(iso > 0.0f))
        iso = p->as_shot_ei;

    return craw_noise_model_eval(p->noise_knots, p->nb_noise_knots, iso, out);
}

/* ------------------------------------------------------------ estimation */

void craw_noise_estimate_params_init(CRAWNoiseEstimateParams *p)
{
    if (!p)
        return;

    *p = (CRAWNoiseEstimateParams) {
        .nb_bins         = 128,
        .min_bin_samples = 20000,
        .max_level       = 0.98f,
    };
}

/* Radius of the box mean a sample is binned by. The high-pass kernel sums
 * to zero over its 3x3 support, so any box mean whose window contains that
 * support is exactly uncorrelated with it; 9x9 makes the mean's own noise
 * a ninth of a sample's, which keeps the binning from dragging samples
 * into neighbouring bins. */
#define CRAW_NOISE_MEAN_R 4

struct craw_noise_bin {
    float *res;        /* |high-pass residual| of every sample in the bin */
    size_t nb, cap;
    double mean_sum;
};

static int craw_noise_bin_push(struct craw_noise_bin *b, float v)
{
    if (b->nb == b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 4096;
        float *p = realloc(b->res, cap * sizeof(*p));

        if (!p)
            return CRAW_ERROR(ENOMEM);

        b->res = p;
        b->cap = cap;
    }

    b->res[b->nb++] = v;

    return 0;
}

static int craw_noise_cmp_float(const void *a, const void *b)
{
    float x = *(const float *)a, y = *(const float *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

static int craw_noise_cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

/* Least trimmed squares over every pair of bins: the line through two of
 * them whose two thirds smallest log residuals are smallest overall. This
 * only has to land in the right basin; the Gauss-Newton below does the
 * rest. Physically impossible candidates (a <= 0 or b < 0) are skipped, so
 * the seed always starts the refinement inside the valid quadrant. */
static bool craw_noise_lts_seed(const double *m, const double *v, int nb,
                                double *a_out, double *b_out, double *scale)
{
    double best = INFINITY, ba = 0.0, bb = 0.0;
    double *work = malloc((size_t)nb * sizeof(*work));
    int keep = (nb * 2) / 3 + 1;

    if (!work)
        return false;

    for (int i = 0; i < nb; i++) {
        for (int j = i + 1; j < nb; j++) {
            double a, b, cost = 0.0;

            if (m[j] - m[i] < 1e-5)
                continue;

            a = (v[j] - v[i]) / (m[j] - m[i]);
            b = v[i] - a * m[i];
            if (!(a > 0.0) || b < 0.0)
                continue;

            for (int k = 0; k < nb; k++) {
                double d = log(v[k]) - log(a * m[k] + b);
                work[k] = d * d;
            }
            qsort(work, (size_t)nb, sizeof(*work), craw_noise_cmp_double);
            for (int k = 0; k < keep; k++)
                cost += work[k];

            if (cost < best) {
                best = cost;
                ba = a;
                bb = b;
            }
        }
    }

    free(work);

    if (!(ba > 0.0))
        return false;

    *a_out = ba;
    *b_out = bb;
    *scale = sqrt(best / keep);

    return true;
}

/* Iteratively reweighted Gauss-Newton on ln(a * m + b), with a and b
 * carried as their logarithms so that neither can leave the physical
 * quadrant, and a Tukey biweight on the residual.
 *
 * The residual is taken in the log domain because the measured variances
 * span three decades across a frame and what matters is that the model is
 * right by a percentage everywhere, not that it is right in absolute terms
 * in the highlights. Scene texture only ever adds variance, so the bins it
 * contaminates sit above the line and the biweight discards them; that
 * one-sidedness is why a robust loss works here at all.
 *
 * Bins are weighted by log(1 + samples) rather than by the sample count:
 * a bin with fifty times the samples of another is worth more, but not
 * fifty times more, because its error is dominated by how smooth that part
 * of the scene is and not by counting statistics. */
static void craw_noise_refine(const double *m, const double *v, const double *cnt,
                              int nb, double scale, double *a_io, double *b_io)
{
    double la = log(*a_io);
    double lb = log(*b_io > 1e-16 ? *b_io : 1e-16);
    double *res = malloc((size_t)nb * sizeof(*res));

    if (!res)
        return;

    if (scale < CRAW_NOISE_SCALE_FLOOR)
        scale = CRAW_NOISE_SCALE_FLOOR;

    for (int it = 0; it < 40; it++) {
        double ea = exp(la), eb = exp(lb);
        double jtj[3] = { 0.0, 0.0, 0.0 };   /* [aa, ab, bb] */
        double jtr[2] = { 0.0, 0.0 };
        double det, d0, d1, cut;

        for (int k = 0; k < nb; k++)
            res[k] = log(v[k]) - log(ea * m[k] + eb);

        /* Re-estimate the scale of the residuals every few steps: the
         * first one comes from the seed and is usually too generous. */
        if ((it & 7) == 0) {
            double *ar = malloc((size_t)nb * sizeof(*ar));

            if (ar) {
                for (int k = 0; k < nb; k++)
                    ar[k] = fabs(res[k]);
                qsort(ar, (size_t)nb, sizeof(*ar), craw_noise_cmp_double);
                scale = 1.4826 * ar[nb / 2];
                free(ar);
                if (scale < CRAW_NOISE_SCALE_FLOOR)
                    scale = CRAW_NOISE_SCALE_FLOOR;
            }
        }

        cut = CRAW_NOISE_TUKEY_C * scale;

        for (int k = 0; k < nb; k++) {
            double den = ea * m[k] + eb;
            double ja = ea * m[k] / den;    /* d ln(den) / d ln(a) */
            double jb = eb / den;           /* d ln(den) / d ln(b) */
            double u = res[k] / cut;
            double w;

            w = fabs(u) < 1.0 ? (1.0 - u * u) * (1.0 - u * u) : 0.0;
            w *= log(1.0 + cnt[k]);

            jtj[0] += w * ja * ja;
            jtj[1] += w * ja * jb;
            jtj[2] += w * jb * jb;
            jtr[0] += w * ja * res[k];
            jtr[1] += w * jb * res[k];
        }

        /* A frame with no shadows leaves b completely undetermined, which
         * makes the normal equations singular; the damping term turns that
         * into "leave b where it was" instead of a division by zero. */
        det = (jtj[0] + 1e-9) * (jtj[2] + 1e-9) - jtj[1] * jtj[1];
        if (!(fabs(det) > 1e-30))
            break;

        d0 = ( (jtj[2] + 1e-9) * jtr[0] - jtj[1] * jtr[1]) / det;
        d1 = (-jtj[1] * jtr[0] + (jtj[0] + 1e-9) * jtr[1]) / det;

        d0 = CRAW_CLAMP(d0, -0.5, 0.5);
        d1 = CRAW_CLAMP(d1, -0.5, 0.5);

        la += d0;
        lb += d1;
        /* exp(-40) is zero for every purpose the model has. */
        if (lb < -40.0)
            lb = -40.0;

        if (fabs(d0) < 1e-9 && fabs(d1) < 1e-9)
            break;
    }

    free(res);

    *a_io = exp(la);
    *b_io = lb <= -40.0 ? 0.0 : exp(lb);
}

int craw_noise_estimate(const uint16_t *bayer, uint32_t width, uint32_t height,
                        size_t stride, const CRAWProfile *profile,
                        const CRAWNoiseEstimateParams *params,
                        CRAWNoiseEstimate *out)
{
    CRAWNoiseEstimateParams def;
    struct craw_noise_bin *bins = NULL;
    float *plane = NULL;
    double *bm = NULL, *bv = NULL, *bc = NULL;
    double a = 0.0, b = 0.0, scale = 0.0, rms = 0.0, range0;
    uint32_t nb_bins, min_samples;
    float max_level;
    int w2, h2, nb = 0, nin = 0;
    int ret = 0;

    if (!bayer || !profile || !out)
        return CRAW_ERROR(EINVAL);
    if (profile->cfa != CRAW_CFA_RGGB)
        return CRAW_ERROR(ENOTSUP);
    if ((width & 1) || (height & 1) || width < 32 || height < 32)
        return CRAW_ERROR(EINVAL);
    if (stride < width)
        return CRAW_ERROR(EINVAL);

    if (!params) {
        craw_noise_estimate_params_init(&def);
        params = &def;
    }

    nb_bins = params->nb_bins ? params->nb_bins : 128;
    min_samples = params->min_bin_samples ? params->min_bin_samples : 20000;
    max_level = params->max_level > 0.0f ? params->max_level : 0.98f;
    if (nb_bins < 8 || nb_bins > CRAW_NOISE_MAX_BINS)
        return CRAW_ERROR(EINVAL);

    for (int s = 0; s < 4; s++) {
        if (!(profile->white_level[s] > profile->black_level[s]))
            return CRAW_ERROR(EINVAL);
    }

    w2 = (int)(width / 2);
    h2 = (int)(height / 2);
    if (w2 < 2 * CRAW_NOISE_MEAN_R + 1 || h2 < 2 * CRAW_NOISE_MEAN_R + 1)
        return CRAW_ERROR(EINVAL);

    plane = malloc((size_t)w2 * h2 * sizeof(*plane));
    bins = calloc(nb_bins, sizeof(*bins));
    bm = malloc(nb_bins * sizeof(*bm));
    bv = malloc(nb_bins * sizeof(*bv));
    bc = malloc(nb_bins * sizeof(*bc));
    if (!plane || !bins || !bm || !bv || !bc) {
        ret = CRAW_ERROR(ENOMEM);
        goto done;
    }

    for (int s = 0; s < 4; s++) {
        int px = s & 1, py = s >> 1;
        double black = profile->black_level[s];
        double range = profile->white_level[s] - profile->black_level[s];

        for (int k = 0; k < h2; k++) {
            for (int i = 0; i < w2; i++) {
                double raw = bayer[(size_t)(2 * k + py) * stride + 2 * i + px] *
                             (1.0 / 65535.0);
                plane[(size_t)k * w2 + i] = (float)((raw - black) / range);
            }
        }

        for (int k = CRAW_NOISE_MEAN_R; k < h2 - CRAW_NOISE_MEAN_R; k++) {
            for (int i = CRAW_NOISE_MEAN_R; i < w2 - CRAW_NOISE_MEAN_R; i++) {
                const float *p = plane + (size_t)k * w2 + i;
                float r, mean = 0.0f;
                int bi;

                /* [1 -2 1; -2 4 -2; 1 -2 1] / 6: zero on any bilinear
                 * signal, unit gain on white noise variance. */
                r = (p[-w2 - 1] + p[-w2 + 1] + p[w2 - 1] + p[w2 + 1]
                     - 2.0f * (p[-w2] + p[w2] + p[-1] + p[1])
                     + 4.0f * p[0]) * (1.0f / 6.0f);

                for (int dy = -CRAW_NOISE_MEAN_R; dy <= CRAW_NOISE_MEAN_R; dy++)
                    for (int dx = -CRAW_NOISE_MEAN_R; dx <= CRAW_NOISE_MEAN_R; dx++)
                        mean += p[(ptrdiff_t)dy * w2 + dx];
                mean *= 1.0f / ((2 * CRAW_NOISE_MEAN_R + 1) *
                                (2 * CRAW_NOISE_MEAN_R + 1));

                if (!(mean > 0.0f) || mean > max_level)
                    continue;

                /* Bins equally spaced in sqrt(mean): the noise standard
                 * deviation is what they have to resolve and it is the
                 * square root of the level. */
                bi = (int)(sqrtf(mean) * (float)nb_bins);
                bi = CRAW_CLAMP(bi, 0, (int)nb_bins - 1);

                ret = craw_noise_bin_push(&bins[bi], fabsf(r));
                if (ret < 0)
                    goto done;
                bins[bi].mean_sum += mean;
            }
        }
    }

    for (uint32_t i = 0; i < nb_bins; i++) {
        double sigma;

        if (bins[i].nb < min_samples)
            continue;

        qsort(bins[i].res, bins[i].nb, sizeof(*bins[i].res), craw_noise_cmp_float);
        sigma = bins[i].res[bins[i].nb / 2] / CRAW_NOISE_MAD_SCALE;
        if (!(sigma > 0.0))
            continue;

        bm[nb] = bins[i].mean_sum / (double)bins[i].nb;
        bv[nb] = sigma * sigma;
        bc[nb] = (double)bins[i].nb;
        nb++;
    }

    if (nb < 3) {
        ret = CRAW_ERROR(ENODATA);
        goto done;
    }

    if (!craw_noise_lts_seed(bm, bv, nb, &a, &b, &scale)) {
        ret = CRAW_ERROR(ENODATA);
        goto done;
    }

    craw_noise_refine(bm, bv, bc, nb, scale, &a, &b);

    for (int k = 0; k < nb; k++) {
        double d = log(bv[k]) - log(a * bm[k] + b);

        /* A bin the model is out by more than about 40% on is not
         * describing sensor noise; it is describing the scene. */
        if (fabs(d) < 0.35) {
            rms += d * d;
            nin++;
        }
    }
    rms = sqrt(rms / (nin ? nin : 1));

    range0 = profile->white_level[0] - profile->black_level[0];

    *out = (CRAWNoiseEstimate) {
        .model          = { .a = (float)a, .b = (float)b },
        .nb_bins        = (uint32_t)nb,
        .nb_inliers     = (uint32_t)nin,
        .fit_rms        = (float)rms,
        .read_noise_lsb = (float)(sqrt(b) * range0 * 65535.0),
    };

done:
    if (bins) {
        for (uint32_t i = 0; i < nb_bins; i++)
            free(bins[i].res);
        free(bins);
    }
    free(bc);
    free(bv);
    free(bm);
    free(plane);

    return ret;
}
