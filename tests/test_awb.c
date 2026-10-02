/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* The auto white balance estimator on synthetic histograms built by the
 * reference histogram builder, and the temporal smoother on synthetic
 * tracks. DESIGN.md sections 4.4 and 7. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <craw/awb.h>

static int failures;

#define CHECK(cond, ...)                                        \
    do {                                                        \
        if (!(cond)) {                                          \
            printf("FAIL %s:%d: ", __func__, __LINE__);         \
            printf(__VA_ARGS__);                                \
            printf("\n");                                       \
            failures++;                                         \
        }                                                       \
    } while (0)

#define CHECK_NEAR(got, want, tol, ...)                                 \
    do {                                                                \
        double g_ = (got), w_ = (want), t_ = (tol);                     \
        if (!(fabs(g_ - w_) <= t_)) {                                   \
            printf("FAIL %s:%d: ", __func__, __LINE__);                 \
            printf(__VA_ARGS__);                                        \
            printf(": got %.9g, want %.9g, off by %.3g (tol %.3g)\n",   \
                   g_, w_, fabs(g_ - w_), t_);                          \
            failures++;                                                 \
        }                                                               \
    } while (0)

#define BINS CRAW_AWB_HIST_BINS

/* Panasonic DC-S1M2, DESIGN.md section 9. Only the white balance table and
 * the as-shot gains matter here. */
static void craw_test_profile_init(CRAWProfile *p)
{
    static const CRAWWbKnot wb[] = {
        {  2000.0f, { 0.8454f, 1.0f, 3.9482f } },
        {  3200.0f, { 1.3763f, 1.0f, 2.4707f } },
        {  3400.0f, { 1.4477f, 1.0f, 2.2881f } },
        {  5000.0f, { 1.9499f, 1.0f, 1.7195f } },
        {  5500.0f, { 2.0253f, 1.0f, 1.6676f } },
        {  6300.0f, { 2.2051f, 1.0f, 1.5348f } },
        {  7300.0f, { 2.3299f, 1.0f, 1.4186f } },
        { 11500.0f, { 2.6469f, 1.0f, 1.2318f } },
    };

    craw_profile_init(p);

    for (int s = 0; s < 4; s++) {
        p->black_level[s] = 256.0f / 65535.0f;
        p->white_level[s] = 63743.0f / 65535.0f;
    }

    p->as_shot_wb[0] = 2.02539f;
    p->as_shot_wb[1] = 1.0f;
    p->as_shot_wb[2] = 1.66797f;
    p->as_shot_cct = 5500.0f;
    p->gain = 23.04f;

    memcpy(p->wb_knots, wb, sizeof(wb));
    p->nb_wb_knots = CRAW_ARRAY_ELEMS(wb);
}

static uint32_t craw_test_rand(uint32_t *s)
{
    uint32_t x = *s;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;

    return x;
}

static double craw_test_uniform(uint32_t *s, double lo, double hi)
{
    return lo + (hi - lo) * (craw_test_rand(s) / 4294967296.0);
}

/* Builds the pipeline's intermediate for a scene that is neutral under
 * `cct`, with `sat_fraction` of its area covered by strongly coloured
 * patches. The patches are aligned to the 8x8 blocks the histogram pass
 * reduces, because a colour smeared over every block would leave no
 * neutral samples at all rather than a neutral population plus a coloured
 * one, which is what a real scene looks like. */
static int craw_test_scene(const CRAWProfile *p, double cct, double tint,
                           double sat_fraction, uint32_t width,
                           uint32_t height, float *rgb)
{
    CRAWIlluminant il = { .cct = (float)cct, .tint = (float)tint };
    static const float sat[][3] = {
        { 0.90f, 0.12f, 0.08f },
        { 0.10f, 0.85f, 0.15f },
        { 0.12f, 0.20f, 0.90f },
        { 0.85f, 0.75f, 0.10f },
    };
    uint32_t seed = 0xC0FFEEu;
    float rg, bg;
    int ret;

    ret = craw_profile_illuminant_to_neutral(p, &il, &rg, &bg);
    if (ret < 0)
        return ret;

    for (uint32_t by = 0; by < height; by += 8) {
        for (uint32_t bx = 0; bx < width; bx += 8) {
            float col[3];
            bool coloured = craw_test_uniform(&seed, 0.0, 1.0) < sat_fraction;

            if (coloured) {
                uint32_t k = craw_test_rand(&seed) % CRAW_ARRAY_ELEMS(sat);
                memcpy(col, sat[k], sizeof(col));
            } else {
                /* A neutral surface at a random luminance, with a small
                 * chromaticity jitter so that the population spans a few
                 * histogram bins instead of landing in exactly one. */
                double lum = craw_test_uniform(&seed, 0.05, 0.55);
                double jx = craw_test_uniform(&seed, -0.02, 0.02);
                double jy = craw_test_uniform(&seed, -0.02, 0.02);

                col[0] = (float)(lum * rg * exp(jx) * p->as_shot_wb[0]);
                col[1] = (float)(lum * p->as_shot_wb[1]);
                col[2] = (float)(lum * bg * exp(jy) * p->as_shot_wb[2]);
            }

            for (uint32_t y = by; y < CRAW_MIN(by + 8, height); y++) {
                for (uint32_t x = bx; x < CRAW_MIN(bx + 8, width); x++) {
                    float *px = &rgb[((size_t)y * width + x) * 4];

                    /* Per pixel shading noise: it moves the luminance, not
                     * the chromaticity, so the block mean stays on the
                     * same ratio. */
                    float n = (float)craw_test_uniform(&seed, 0.97, 1.03);

                    px[0] = CRAW_MIN(col[0] * n, 1.0f);
                    px[1] = CRAW_MIN(col[1] * n, 1.0f);
                    px[2] = CRAW_MIN(col[2] * n, 1.0f);
                    px[3] = 1.0f;
                }
            }
        }
    }

    return 0;
}

static void craw_test_histogram(void)
{
    const uint32_t w = 320, h = 256;
    CRAWProfile p;
    float *rgb = malloc((size_t)w * h * 4 * sizeof(*rgb));
    uint32_t *count = malloc(BINS * BINS * sizeof(*count));
    uint32_t *weight = malloc(BINS * BINS * sizeof(*weight));
    uint32_t *bright = malloc(BINS * BINS * sizeof(*bright));
    uint64_t total = 0;
    int ret;

    craw_test_profile_init(&p);

    /* A single flat neutral: the whole frame has to land in the one bin
     * that contains ln(R/G), ln(B/G) of that neutral. The values are the
     * intermediate a 3200 K grey of 0.25 produces after the as-shot gains,
     * which are for 5500 K, so the raw ratios are the 3200 K knot's. */
    for (size_t i = 0; i < (size_t)w * h; i++) {
        rgb[i * 4 + 0] = 0.25f * (2.02539f / 1.3763f);
        rgb[i * 4 + 1] = 0.25f;
        rgb[i * 4 + 2] = 0.25f * (1.66797f / 2.4707f);
        rgb[i * 4 + 3] = 1.0f;
    }

    ret = craw_awb_histogram_from_rgb(rgb, w, h, (size_t)w * 4, 4,
                                      p.as_shot_wb, count, weight, bright);
    CHECK(ret == 0, "histogram failed: %d", ret);

    {
        int nonzero = 0, ix = -1, iy = -1;
        double lo = CRAW_AWB_HIST_LOG_MIN;
        double sc = BINS / (CRAW_AWB_HIST_LOG_MAX - CRAW_AWB_HIST_LOG_MIN);
        double x = log(1.0 / 1.3763), y = log(1.0 / 2.4707);
        int want_ix = (int)floor((x - lo) * sc);
        int want_iy = (int)floor((y - lo) * sc);

        for (int i = 0; i < BINS * BINS; i++) {
            total += count[i];
            if (count[i]) {
                nonzero++;
                ix = i % BINS;
                iy = i / BINS;
            }
        }

        CHECK(nonzero == 1, "a flat neutral must fill exactly one bin, got %d",
              nonzero);
        CHECK(total == (uint64_t)(w / 8) * (h / 8),
              "one sample per 8x8 block: got %llu, want %u",
              (unsigned long long)total, (w / 8) * (h / 8));
        CHECK(ix == want_ix && iy == want_iy,
              "a 3200 K neutral must land in bin (%d, %d), got (%d, %d)",
              want_ix, want_iy, ix, iy);
        CHECK(weight[(size_t)iy * BINS + ix] ==
              (uint32_t)(w / 8) * (h / 8) * (uint32_t)roundf(0.25f * 65536.0f),
              "the weight of a 0.25 grey block is min(g, 0.5) in 16.16");
    }

    /* Clipped and near black samples are skipped. */
    for (size_t i = 0; i < (size_t)w * h; i++) {
        rgb[i * 4 + 0] = 0.99f;
        rgb[i * 4 + 1] = 0.99f;
        rgb[i * 4 + 2] = 0.99f;
    }
    ret = craw_awb_histogram_from_rgb(rgb, w, h, (size_t)w * 4, 4,
                                      p.as_shot_wb, count, weight, bright);
    CHECK(ret == 0, "histogram failed: %d", ret);
    total = 0;
    for (int i = 0; i < BINS * BINS; i++)
        total += count[i];
    CHECK(total == 0, "clipped samples must be skipped, got %llu",
          (unsigned long long)total);

    for (size_t i = 0; i < (size_t)w * h; i++) {
        rgb[i * 4 + 0] = 0.001f;
        rgb[i * 4 + 1] = 0.001f;
        rgb[i * 4 + 2] = 0.001f;
    }
    ret = craw_awb_histogram_from_rgb(rgb, w, h, (size_t)w * 4, 4,
                                      p.as_shot_wb, count, weight, bright);
    CHECK(ret == 0, "histogram failed: %d", ret);
    total = 0;
    for (int i = 0; i < BINS * BINS; i++)
        total += count[i];
    CHECK(total == 0, "near black samples must be skipped, got %llu",
          (unsigned long long)total);

    /* Partial blocks at the right and bottom edge still produce a sample. */
    ret = craw_test_scene(&p, 3200.0, 0.0, 0.0, 12, 4, rgb);
    CHECK(ret == 0, "scene build failed: %d", ret);
    ret = craw_awb_histogram_from_rgb(rgb, 12, 4, 12 * 4, 4, p.as_shot_wb,
                                      count, weight, bright);
    CHECK(ret == 0, "histogram failed: %d", ret);
    total = 0;
    for (int i = 0; i < BINS * BINS; i++)
        total += count[i];
    CHECK(total == 2, "a 12x4 frame is two partial blocks, got %llu",
          (unsigned long long)total);

    /* Three channel input, tightly packed, must behave identically. */
    for (size_t i = 0; i < 64; i++) {
        rgb[i * 3 + 0] = 0.25f * (2.02539f / 1.3763f);
        rgb[i * 3 + 1] = 0.25f;
        rgb[i * 3 + 2] = 0.25f * (1.66797f / 2.4707f);
    }
    ret = craw_awb_histogram_from_rgb(rgb, 8, 8, 8 * 3, 3, p.as_shot_wb,
                                      count, weight, bright);
    CHECK(ret == 0, "three channel histogram failed: %d", ret);
    total = 0;
    for (int i = 0; i < BINS * BINS; i++)
        total += count[i];
    CHECK(total == 1, "an 8x8 three channel frame is one block, got %llu",
          (unsigned long long)total);

    CHECK(craw_awb_histogram_from_rgb(rgb, 8, 8, 8 * 4, 2, p.as_shot_wb,
                                      count, weight, bright) == CRAW_ERROR(EINVAL),
          "an unsupported channel count must fail with EINVAL");
    CHECK(craw_awb_histogram_from_rgb(rgb, 8, 8, 8, 4, p.as_shot_wb,
                                      count, weight, bright) == CRAW_ERROR(EINVAL),
          "a stride below the row size must fail with EINVAL");

    free(rgb);
    free(count);
    free(weight);
    free(bright);
}


/* A dominant warm surface under a cool light, the case that defeats a
 * surface-only estimator: 90% of the blocks are a warm surface that reads
 * as a 3000 K neutral at low luminance, 10% are the glare of a 6000 K light
 * at high luminance. The bright reading must win, and switching it off
 * must fall back to the surface reading. */
static void craw_test_bright_source(void)
{
    const uint32_t w = 320, h = 256;
    CRAWProfile p;
    CRAWAwbParams params;
    CRAWAwbEstimate est;
    CRAWAwbHistogram hist;
    float *rgb = malloc((size_t)w * h * 4 * sizeof(*rgb));
    uint32_t *count = malloc(BINS * BINS * sizeof(*count));
    uint32_t *weight = malloc(BINS * BINS * sizeof(*weight));
    uint32_t *bright = malloc(BINS * BINS * sizeof(*bright));
    uint32_t seed = 0xBEEFu;
    float rg_s, bg_s, rg_l, bg_l;
    int ret;

    craw_test_profile_init(&p);
    craw_awb_params_init(&params);
    CHECK(params.prefer_bright, "prefer_bright must default to true");
    CHECK_NEAR(params.bright_max_spread_mired, 60.0, 1e-6, "default bright spread limit");

    ret = craw_profile_illuminant_to_neutral(&p, &(CRAWIlluminant) { .cct = 3000.0f },
                                             &rg_s, &bg_s);
    CHECK(ret == 0, "surface neutral failed: %d", ret);
    ret = craw_profile_illuminant_to_neutral(&p, &(CRAWIlluminant) { .cct = 6000.0f },
                                             &rg_l, &bg_l);
    CHECK(ret == 0, "light neutral failed: %d", ret);

    for (uint32_t by = 0; by < h; by += 8) {
        for (uint32_t bx = 0; bx < w; bx += 8) {
            bool light = craw_test_uniform(&seed, 0.0, 1.0) < 0.10;
            double lum = light ? craw_test_uniform(&seed, 0.45, 0.70)
                               : craw_test_uniform(&seed, 0.06, 0.20);
            double jx = craw_test_uniform(&seed, -0.015, 0.015);
            double jy = craw_test_uniform(&seed, -0.015, 0.015);
            float rg = light ? rg_l : rg_s, bg = light ? bg_l : bg_s;
            float col[3] = {
                (float)(lum * rg * exp(jx) * p.as_shot_wb[0]),
                (float)(lum * p.as_shot_wb[1]),
                (float)(lum * bg * exp(jy) * p.as_shot_wb[2]),
            };

            for (uint32_t y = by; y < by + 8; y++) {
                for (uint32_t x = bx; x < bx + 8; x++) {
                    float *px = &rgb[((size_t)y * w + x) * 4];
                    float n = (float)craw_test_uniform(&seed, 0.97, 1.03);

                    px[0] = CRAW_MIN(col[0] * n, 1.0f);
                    px[1] = CRAW_MIN(col[1] * n, 1.0f);
                    px[2] = CRAW_MIN(col[2] * n, 1.0f);
                    px[3] = 1.0f;
                }
            }
        }
    }

    hist = (CRAWAwbHistogram) {
        .bins = BINS,
        .log_min = CRAW_AWB_HIST_LOG_MIN,
        .log_max = CRAW_AWB_HIST_LOG_MAX,
        .count = count,
        .weight = weight,
        .weight_bright = bright,
    };

    ret = craw_awb_histogram_from_rgb(rgb, w, h, (size_t)w * 4, 4, p.as_shot_wb,
                                      count, weight, bright);
    CHECK(ret == 0, "histogram failed: %d", ret);

    ret = craw_awb_estimate(&p, &hist, &params, &est);
    CHECK(ret == 0, "estimate failed: %d", ret);
    CHECK(est.valid, "the glare scene must be valid");
    CHECK(est.source == CRAW_AWB_SOURCE_BRIGHT, "the bright reading must be chosen, got %d",
          (int)est.source);
    CHECK_NEAR(est.illuminant.cct, 6000.0, 300.0, "the light's colour must win");
    CHECK(est.bright.valid, "the bright candidate must be valid");
    CHECK(est.zone.valid && est.zone.illuminant.cct < 4200.0f,
          "the surface reading must still be dragged down by the wood, got %g",
          (double)est.zone.illuminant.cct);
    printf("  glare scene: chosen %.0f K (bright %.0f K spread %.1f, zone %.0f K spread %.1f)\n",
           (double)est.illuminant.cct, (double)est.bright.illuminant.cct,
           (double)est.bright.spread_mired, (double)est.zone.illuminant.cct,
           (double)est.zone.spread_mired);

    params.prefer_bright = false;
    ret = craw_awb_estimate(&p, &hist, &params, &est);
    CHECK(ret == 0, "estimate failed: %d", ret);
    CHECK(est.source == CRAW_AWB_SOURCE_ZONE, "with prefer_bright off the zone reading is used");
    CHECK(est.illuminant.cct < 4200.0f, "zone reading must be warm, got %g",
          (double)est.illuminant.cct);

    free(rgb);
    free(count);
    free(weight);
    free(bright);
}

static void craw_test_estimate(void)
{
    const uint32_t w = 320, h = 256;
    CRAWProfile p;
    CRAWAwbParams params;
    CRAWAwbEstimate est;
    CRAWAwbHistogram hist;
    float *rgb = malloc((size_t)w * h * 4 * sizeof(*rgb));
    uint32_t *count = malloc(BINS * BINS * sizeof(*count));
    uint32_t *weight = malloc(BINS * BINS * sizeof(*weight));
    uint32_t *bright = malloc(BINS * BINS * sizeof(*bright));
    int ret;

    craw_test_profile_init(&p);
    craw_awb_params_init(&params);

    CHECK_NEAR(params.band, 0.08, 1e-6, "default band");
    CHECK_NEAR(params.sigma, 0.05, 1e-6, "default sigma");
    CHECK_NEAR(params.min_zone_fraction, 0.02, 1e-6, "default min zone");
    CHECK_NEAR(params.ambience, 0.0, 1e-6, "default ambience");
    CHECK_NEAR(params.ambience_knee_cct, 4000.0, 1e-6, "default knee");

    hist = (CRAWAwbHistogram) {
        .bins = BINS,
        .log_min = CRAW_AWB_HIST_LOG_MIN,
        .log_max = CRAW_AWB_HIST_LOG_MAX,
        .count = count,
        .weight = weight,
        .weight_bright = bright,
    };

    /* The headline case from DESIGN.md section 9: a neutral scene at
     * 3200 K with 30% of its area strongly coloured. */
    ret = craw_test_scene(&p, 3200.0, 0.0, 0.3, w, h, rgb);
    CHECK(ret == 0, "scene build failed: %d", ret);
    ret = craw_awb_histogram_from_rgb(rgb, w, h, (size_t)w * 4, 4,
                                      p.as_shot_wb, count, weight, bright);
    CHECK(ret == 0, "histogram failed: %d", ret);

    ret = craw_awb_estimate(&p, &hist, &params, &est);
    CHECK(ret == 0, "estimate failed: %d", ret);
    CHECK(est.valid, "a scene with 70%% neutral area must be valid");
    CHECK_NEAR(est.illuminant.cct, 3200.0, 100.0, "estimated cct at 3200 K");
    CHECK_NEAR(est.illuminant.tint, 0.0, 0.02, "estimated tint at 3200 K");
    CHECK(est.confidence > 0.5f, "confidence must be high, got %g",
          (double)est.confidence);
    /* The bright reading is chosen when valid, and the coloured blocks are
     * bright, so its own zone share is modest; the surface reading is the
     * one that must see most of the frame as neutral. */
    CHECK(est.zone.in_zone_fraction > 0.3f, "zone in-zone fraction must be high, got %g",
          (double)est.zone.in_zone_fraction);
    CHECK(est.spread_mired < 20.0f, "spread must be small, got %g",
          (double)est.spread_mired);

    printf("  3200 K scene: cct %.1f tint %+.4f conf %.3f zone %.3f "
           "spread %.2f mired\n", (double)est.illuminant.cct,
           (double)est.illuminant.tint, (double)est.confidence,
           (double)est.in_zone_fraction, (double)est.spread_mired);

    /* Ambience leaves part of the warmth in place. */
    params.ambience = 0.5f;
    ret = craw_awb_estimate(&p, &hist, &params, &est);
    CHECK(ret == 0, "estimate failed: %d", ret);
    CHECK_NEAR(est.illuminant.cct, 3200.0 + 0.5 * (4000.0 - 3200.0), 60.0,
               "ambience 0.5 at a 4000 K knee");
    params.ambience = 0.0f;

    /* Hard limits win over the estimate. */
    params.cct_min = 4000.0f;
    ret = craw_awb_estimate(&p, &hist, &params, &est);
    CHECK(ret == 0, "estimate failed: %d", ret);
    CHECK_NEAR(est.illuminant.cct, 4000.0, 1e-3, "cct_min clamp");
    params.cct_min = 0.0f;

    /* Other illuminants, including tinted ones. */
    for (int i = 0; i < 4; i++) {
        static const double ccts[] = { 2600.0, 4300.0, 5500.0, 8000.0 };
        double tint = i == 1 ? 0.05 : (i == 3 ? -0.04 : 0.0);

        ret = craw_test_scene(&p, ccts[i], tint, 0.25, w, h, rgb);
        CHECK(ret == 0, "scene build failed: %d", ret);
        ret = craw_awb_histogram_from_rgb(rgb, w, h, (size_t)w * 4, 4,
                                          p.as_shot_wb, count, weight, bright);
        CHECK(ret == 0, "histogram failed: %d", ret);
        ret = craw_awb_estimate(&p, &hist, &params, &est);
        CHECK(ret == 0, "estimate failed: %d", ret);

        CHECK(est.valid, "a %g K scene must be valid", ccts[i]);
        /* A fixed tolerance in mired: the locus is sampled in mired, so the
         * error in Kelvin grows with the square of the temperature. */
        CHECK_NEAR(1e6 / est.illuminant.cct, 1e6 / ccts[i], 6.0,
                   "estimated cct at %g K (mired)", ccts[i]);
        CHECK_NEAR(est.illuminant.tint, tint, 0.02,
                   "estimated tint at %g K", ccts[i]);
    }

    /* Nothing neutral in frame at all. */
    ret = craw_test_scene(&p, 3200.0, 0.0, 1.0, w, h, rgb);
    CHECK(ret == 0, "scene build failed: %d", ret);
    ret = craw_awb_histogram_from_rgb(rgb, w, h, (size_t)w * 4, 4,
                                      p.as_shot_wb, count, weight, bright);
    CHECK(ret == 0, "histogram failed: %d", ret);
    ret = craw_awb_estimate(&p, &hist, &params, &est);
    CHECK(ret == 0, "estimate failed: %d", ret);
    CHECK(!est.valid, "a scene with no neutral must be invalid");

    /* An empty histogram is not an error either. */
    memset(count, 0, BINS * BINS * sizeof(*count));
    memset(weight, 0, BINS * BINS * sizeof(*weight));
    ret = craw_awb_estimate(&p, &hist, &params, &est);
    CHECK(ret == 0, "an empty histogram must not be an error: %d", ret);
    CHECK(!est.valid, "an empty histogram must be invalid");

    /* A profile without a locus cannot be searched. */
    p.nb_wb_knots = 1;
    CHECK(craw_awb_estimate(&p, &hist, &params, &est) == CRAW_ERROR(ENOTSUP),
          "a profile without a locus must fail with ENOTSUP");

    free(rgb);
    free(count);
    free(weight);
    free(bright);
}

static CRAWAwbEstimate craw_test_est(double cct, double tint, double conf)
{
    return (CRAWAwbEstimate) {
        .valid            = true,
        .illuminant       = { .cct = (float)cct, .tint = (float)tint },
        .confidence       = (float)conf,
        .in_zone_fraction = 0.5f,
        .spread_mired     = 5.0f,
    };
}

static void craw_test_smoother(void)
{
    CRAWAwbSmootherParams sp;
    CRAWAwbSmoother *sm = NULL;
    CRAWAwbEstimate est;
    CRAWIlluminant out;
    const double frame = 1.0 / 30.0;
    double t = 0.0, prev;
    int ret;

    craw_awb_smoother_params_init(&sp);
    CHECK_NEAR(sp.time_constant, 1.0, 1e-6, "default time constant");
    CHECK_NEAR(sp.hysteresis_mired, 3.0, 1e-6, "default hysteresis");
    CHECK_NEAR(sp.max_slew_mired_per_s, 30.0, 1e-6, "default slew limit");
    CHECK_NEAR(sp.min_confidence, 0.2, 1e-6, "default min confidence");

    ret = craw_awb_smoother_create(&sp, &sm);
    CHECK(ret == 0 && sm, "smoother create failed: %d", ret);
    if (!sm)
        return;

    /* The first estimate is adopted as is. */
    est = craw_test_est(5500.0, 0.0, 1.0);
    ret = craw_awb_smoother_push(sm, t, &est, false, &out);
    CHECK(ret == 0, "push failed: %d", ret);
    CHECK_NEAR(out.cct, 5500.0, 1e-2, "the first push adopts the estimate");

    /* A step to 3200 K: rate limited first, then exponential, and it stops
     * inside the hysteresis band. */
    est = craw_test_est(3200.0, 0.05, 1.0);
    prev = 1e6 / 5500.0;
    for (int i = 0; i < 30; i++) {
        t += frame;
        ret = craw_awb_smoother_push(sm, t, &est, false, &out);
        CHECK(ret == 0, "push failed: %d", ret);
        CHECK(1e6 / out.cct >= prev - 1e-6, "the track must not go backwards");
        prev = 1e6 / out.cct;
    }
    CHECK_NEAR(prev - 1e6 / 5500.0, 30.0, 0.5,
               "one second of a large step is the slew limit");

    for (int i = 0; i < 30 * 12; i++) {
        t += frame;
        ret = craw_awb_smoother_push(sm, t, &est, false, &out);
        CHECK(ret == 0, "push failed: %d", ret);
    }
    CHECK_NEAR(1e6 / out.cct, 1e6 / 3200.0, sp.hysteresis_mired + 0.5,
               "a step must converge within a few seconds (mired)");
    CHECK_NEAR(out.tint, 0.05, 0.01, "the tint follows the same fractions");

    printf("  step 5500 K -> 3200 K settles at %.1f K, tint %+.4f\n",
           (double)out.cct, (double)out.tint);

    /* Wiggles smaller than the hysteresis change nothing at all. */
    {
        float held_cct = out.cct, held_tint = out.tint;

        for (int i = 0; i < 60; i++) {
            double mired = 1e6 / held_cct + (i & 1 ? 1.0 : -1.0);

            t += frame;
            est = craw_test_est(1e6 / mired, 0.2, 1.0);
            ret = craw_awb_smoother_push(sm, t, &est, false, &out);
            CHECK(ret == 0, "push failed: %d", ret);
        }
        CHECK(out.cct == held_cct && out.tint == held_tint,
              "a 1 mired wiggle must not move the track");
    }

    /* Low confidence and invalid estimates hold the track. */
    {
        float held_cct = out.cct, held_tint = out.tint;

        for (int i = 0; i < 60; i++) {
            t += frame;
            est = craw_test_est(8000.0, 0.4, 0.1);
            ret = craw_awb_smoother_push(sm, t, &est, false, &out);
            CHECK(ret == 0, "push failed: %d", ret);
        }
        CHECK(out.cct == held_cct && out.tint == held_tint,
              "an estimate below min_confidence must hold the track");

        for (int i = 0; i < 60; i++) {
            t += frame;
            est = craw_test_est(8000.0, 0.4, 1.0);
            est.valid = false;
            ret = craw_awb_smoother_push(sm, t, &est, false, &out);
            CHECK(ret == 0, "push failed: %d", ret);
        }
        CHECK(out.cct == held_cct && out.tint == held_tint,
              "an invalid estimate must hold the track");
    }

    /* A scene cut is immediate. */
    t += frame;
    est = craw_test_est(8000.0, -0.03, 1.0);
    ret = craw_awb_smoother_push(sm, t, &est, true, &out);
    CHECK(ret == 0, "push failed: %d", ret);
    CHECK_NEAR(out.cct, 8000.0, 1e-2, "a scene cut adopts the estimate");
    CHECK_NEAR(out.tint, -0.03, 1e-6, "a scene cut adopts the tint");

    /* After a reset the first usable estimate starts the track whatever
     * its confidence - a clip the estimator is unsure of throughout is
     * still balanced - and the unsure readings after it hold the track
     * rather than move it about. */
    craw_awb_smoother_reset(sm);
    t += frame;
    est = craw_test_est(3850.0, 0.0, 0.1);
    ret = craw_awb_smoother_push(sm, t, &est, false, &out);
    CHECK(ret == 0, "a low-confidence estimate must start the track: %d", ret);
    CHECK_NEAR(out.cct, 3850.0, 1e-2, "and the track is that estimate");
    for (int i = 0; i < 30; i++) {
        t += frame;
        est = craw_test_est(8000.0 - 100.0 * i, 0.3, 0.1);
        ret = craw_awb_smoother_push(sm, t, &est, false, &out);
        CHECK(ret == 0, "push failed: %d", ret);
    }
    CHECK_NEAR(out.cct, 3850.0, 1e-2, "unsure readings after the start hold it");

    /* An invalid one cannot start a track: EAGAIN, output untouched. */
    craw_awb_smoother_reset(sm);
    t += frame;
    out.cct = 1234.0f;
    est = craw_test_est(3850.0, 0.0, 1.0);
    est.valid = false;
    ret = craw_awb_smoother_push(sm, t, &est, false, &out);
    CHECK(ret == CRAW_ERROR(EAGAIN), "an invalid estimate must not start a track: %d", ret);
    CHECK_NEAR(out.cct, 1234.0, 1e-6, "EAGAIN leaves the output untouched");

    /* The first push after a reset is adopted as is. */
    t += frame;
    est = craw_test_est(2900.0, 0.0, 1.0);
    ret = craw_awb_smoother_push(sm, t, &est, false, &out);
    CHECK(ret == 0, "push failed: %d", ret);
    CHECK_NEAR(out.cct, 2900.0, 1e-2, "the first push after a reset adopts");

    /* With a track, a low-confidence scene cut holds rather than jumps. */
    t += frame;
    est = craw_test_est(8000.0, 0.0, 0.1);
    ret = craw_awb_smoother_push(sm, t, &est, true, &out);
    CHECK(ret == 0, "push failed: %d", ret);
    CHECK_NEAR(out.cct, 2900.0, 1e-2, "a low-confidence scene cut holds the track");

    /* A long gap must not turn into an unbounded jump: dt is clamped to a
     * second, so at most one second worth of slew happens. */
    est = craw_test_est(7000.0, 0.0, 1.0);
    ret = craw_awb_smoother_push(sm, t + 60.0, &est, false, &out);
    CHECK(ret == 0, "push failed: %d", ret);
    CHECK(1e6 / 2900.0 - 1e6 / out.cct <= 30.0 + 1e-3,
          "a 60 second gap must still respect the slew limit");

    craw_awb_smoother_free(&sm);
    CHECK(sm == NULL, "free must NULL the caller's handle");

    /* Nothing to report before the first usable estimate. */
    ret = craw_awb_smoother_create(NULL, &sm);
    CHECK(ret == 0 && sm, "smoother create with defaults failed: %d", ret);
    est = craw_test_est(5500.0, 0.0, 1.0);
    est.valid = false;
    CHECK(craw_awb_smoother_push(sm, 0.0, &est, false, &out) ==
          CRAW_ERROR(EAGAIN),
          "an invalid first estimate has nothing to report");
    craw_awb_smoother_free(&sm);

    sp.time_constant = 0.0f;
    CHECK(craw_awb_smoother_create(&sp, &sm) == CRAW_ERROR(EINVAL),
          "a zero time constant must be rejected");
    CHECK(sm == NULL, "a failed create must not leave a handle");
}

int main(void)
{
    craw_test_histogram();
    craw_test_estimate();
    craw_test_bright_source();
    craw_test_smoother();

    if (failures) {
        printf("test_awb: %d failure(s)\n", failures);
        return 1;
    }

    printf("test_awb: all checks passed\n");
    return 0;
}
