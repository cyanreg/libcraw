/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* The film grain parameter model in tools/gui/grain.c: that every preset
 * produces data the AV1 and H.274 specifications allow, that the controls do
 * what their names say, and that the AV1 grain template matches an
 * independent implementation. CPU only - what the shader does with the data
 * is measured by tools/grain/grain_preview.c on a real GPU. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "grain.h"

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

static struct craw_grain_params base(int width)
{
    struct craw_grain_params p = craw_grain_default_params();
    p.width = width;
    p.height = width * 9 / 16;
    p.seed = 0x1234;
    return p;
}

/* Every combination the UI can reach has to come out valid. That is a lot of
 * combinations, which is the point: the model has to be total. */
static void craw_test_presets(void)
{
    static const int widths[] = { 1920, 3840, 5760 };
    int built = 0;

    for (size_t w = 0; w < sizeof(widths) / sizeof(*widths); w++) {
        for (int g = 0; g < CRAW_GRAIN_GAUGE_NB; g++) {
            for (int s = 0; s < CRAW_GRAIN_STOCK_NB; s++) {
                for (int r = 0; r < CRAW_GRAIN_RESPONSE_NB; r++) {
                    for (int m = 0; m < CRAW_GRAIN_MODEL_NB; m++) {
                        for (int pa = 0; pa < CRAW_GRAIN_PATH_NB; pa++) {
                            struct craw_grain_params p = base(widths[w]);
                            struct craw_grain_out o;
                            p.gauge = g;
                            p.stock = s;
                            p.response = r;
                            p.model = m;
                            p.path = pa;
                            int ret = craw_grain_build(&p, &o);
                            CHECK(ret == 0, "%s/%s/%s/%s at %d: build "
                                  "returned %d",
                                  craw_grain_gauge_key(g),
                                  craw_grain_stock_key(s),
                                  craw_grain_response_name(r),
                                  craw_grain_model_name(m), widths[w], ret);
                            if (ret)
                                continue;
                            const char *bad = craw_grain_validate(&o.data);
                            CHECK(!bad, "%s/%s/%s/%s at %d: %s",
                                  craw_grain_gauge_key(g),
                                  craw_grain_stock_key(s),
                                  craw_grain_response_name(r),
                                  craw_grain_model_name(m), widths[w],
                                  bad ? bad : "");
                            CHECK(o.enabled, "%s/%s: grain came out disabled",
                                  craw_grain_gauge_key(g),
                                  craw_grain_stock_key(s));
                            /* The amplitude has to survive quantisation.
                             * AV1 interpolates its scaling function, so it
                             * only loses rounding; H.274's is a staircase of
                             * sixteen steps, one of which is centred on mid
                             * grey - but only in the display-encoded domain,
                             * so on the source path a steep response can be
                             * a sixth of a step out. See the open question
                             * in DESIGN.md about placing H.274's intervals
                             * perceptually. */
                            double tol = o.model_used == CRAW_GRAIN_MODEL_H274
                                       ? 0.20 : 0.06;
                            CHECK_NEAR(o.sigma_built, o.sigma_signal,
                                       tol * o.sigma_signal + 1e-5,
                                       "%s/%s/%s/%s at %d: amplitude",
                                       craw_grain_gauge_key(g),
                                       craw_grain_stock_key(s),
                                       craw_grain_response_name(r),
                                       craw_grain_model_name(m), widths[w]);
                            built++;
                        }
                    }
                }
            }
        }
    }
    printf("  %d preset combinations built and validated\n", built);
}

static void craw_test_amount(void)
{
    struct craw_grain_params p = base(1920);
    struct craw_grain_out o;

    p.amount = 0.0f;
    CHECK(craw_grain_build(&p, &o) == 0, "amount 0 must not be an error");
    CHECK(!o.enabled, "amount 0 must disable grain");
    CHECK(o.data.type == PL_FILM_GRAIN_NONE,
          "amount 0 must leave the type at NONE, got %d", o.data.type);
    CHECK(craw_grain_sigma_at(&o, 0.5f) == 0.0f,
          "amount 0 must produce no amplitude");
    CHECK(!craw_grain_validate(&o.data), "NONE must validate");

    /* Amplitude has to be linear in `amount` where it is not clipped. */
    float got[3];
    for (int i = 0; i < 3; i++) {
        p.amount = 0.25f * (i + 1);
        CHECK(craw_grain_build(&p, &o) == 0, "amount %.2f", p.amount);
        got[i] = o.sigma_built;
    }
    CHECK_NEAR(got[1] / got[0], 2.0, 0.08, "amount 0.5 against 0.25");
    CHECK_NEAR(got[2] / got[0], 3.0, 0.12, "amount 0.75 against 0.25");
}

/* The size control moves the blob and nothing else, until it runs into the
 * model's ceiling. */
static void craw_test_size(void)
{
    struct craw_grain_params p = base(1920);
    struct craw_grain_out small, large;

    p.gauge = CRAW_GRAIN_GAUGE_SUPER35;
    p.size = 1.0f;
    CHECK(craw_grain_build(&p, &small) == 0, "size 1");
    p.size = 3.0f;
    CHECK(craw_grain_build(&p, &large) == 0, "size 3");

    CHECK_NEAR(large.blob_px / small.blob_px, 3.0, 0.01,
               "the size control must scale the asked-for blob");
    CHECK(large.blob_clamped > small.blob_clamped + 1.0f,
          "a 3x size must actually build a bigger blob: %.2f against %.2f",
          large.blob_clamped, small.blob_clamped);

    /* Past the ceiling the blob stops growing but the amplitude must not
     * run away with it. */
    p.size = 16.0f;
    struct craw_grain_out huge, huger;
    CHECK(craw_grain_build(&p, &huge) == 0, "size 16");
    p.size = 32.0f;
    CHECK(craw_grain_build(&p, &huger) == 0, "size 32");
    /* The reported figure is the CPU field simulation's, which runs high
     * past about six pixels; grain_preview measures 7.7 px on the GPU where
     * this says 10.7. What matters is that it stops. */
    CHECK(huge.blob_clamped < 12.0f,
          "the AV1 model must have a ceiling, got %.2f", huge.blob_clamped);
    CHECK_NEAR(huger.blob_clamped, huge.blob_clamped, 0.01,
               "past the ceiling, asking for more must change nothing");
    CHECK_NEAR(huge.sigma_built, huge.sigma_signal,
               0.06 * huge.sigma_signal, "amplitude past the size ceiling");
}

/* A smaller gauge is magnified more, so its grain must come out bigger at
 * the same output resolution. */
static void craw_test_gauge_order(void)
{
    static const int order[] = {
        CRAW_GRAIN_GAUGE_IMAX, CRAW_GRAIN_GAUGE_65MM,
        CRAW_GRAIN_GAUGE_SUPER35, CRAW_GRAIN_GAUGE_16MM,
        CRAW_GRAIN_GAUGE_SUPER8, CRAW_GRAIN_GAUGE_REGULAR8,
    };
    float prev = 0.0f;
    for (size_t i = 0; i < sizeof(order) / sizeof(*order); i++) {
        struct craw_grain_params p = base(1920);
        struct craw_grain_out o;
        p.gauge = order[i];
        p.stock = CRAW_GRAIN_STOCK_V3_500T;
        CHECK(craw_grain_build(&p, &o) == 0, "%s",
              craw_grain_gauge_key(order[i]));
        CHECK(o.blob_px > prev, "%s must ask for a bigger blob than the "
              "gauge before it: %.3f against %.3f",
              craw_grain_gauge_key(order[i]), o.blob_px, prev);
        prev = o.blob_px;
    }
}

/* What each response promises, read back off the built scaling function the
 * same way the shader will read it. */
static void craw_test_response(void)
{
    struct craw_grain_params p = base(1920);
    struct craw_grain_out o;
    const int n = 33;

    /* Negative: no grain at black, rising with exposure. */
    p.response = CRAW_GRAIN_RESPONSE_NEGATIVE;
    CHECK(craw_grain_build(&p, &o) == 0, "negative");
    CHECK(craw_grain_sigma_at(&o, 0.0f) == 0.0f,
          "the negative response must leave pure black clean, got %g",
          craw_grain_sigma_at(&o, 0.0f));
    for (int i = 1; i < n; i++) {
        float a = craw_grain_sigma_at(&o, (i - 1) / (float)(n - 1));
        float b = craw_grain_sigma_at(&o, i / (float)(n - 1));
        CHECK(b >= a - 1e-7f, "the negative response must not fall: "
              "%.5f at %.3f then %.5f at %.3f",
              a, (i - 1) / (float)(n - 1), b, i / (float)(n - 1));
    }

    /* Flat: the same everywhere. */
    p.response = CRAW_GRAIN_RESPONSE_FLAT;
    CHECK(craw_grain_build(&p, &o) == 0, "flat");
    float ref = craw_grain_sigma_at(&o, 0.5f);
    for (int i = 0; i < n; i++) {
        float v = craw_grain_sigma_at(&o, i / (float)(n - 1));
        CHECK_NEAR(v, ref, 0.02 * ref, "the flat response must not vary, "
                   "at level %.3f", i / (float)(n - 1));
    }

    /* Print: a mid-tone peak, weaker at both ends. */
    p.response = CRAW_GRAIN_RESPONSE_PRINT;
    CHECK(craw_grain_build(&p, &o) == 0, "print");
    float mid = craw_grain_sigma_at(&o, 0.5f);
    CHECK(craw_grain_sigma_at(&o, 0.0f) < mid,
          "print must be weaker at black than at mid grey");
    CHECK(craw_grain_sigma_at(&o, 1.0f) < mid,
          "print must be weaker at white than at mid grey");

    /* Reversal: peak below mid grey, near-clean highlights. */
    p.response = CRAW_GRAIN_RESPONSE_REVERSAL;
    CHECK(craw_grain_build(&p, &o) == 0, "reversal");
    CHECK(craw_grain_sigma_at(&o, 0.3f) > craw_grain_sigma_at(&o, 0.6f),
          "reversal must peak below mid grey");
    CHECK(craw_grain_sigma_at(&o, 1.0f) < 0.1f * craw_grain_sigma_at(&o, 0.3f),
          "reversal must leave the highlights nearly clean");

    /* Every response has to deliver the physical amplitude at mid grey,
     * which is the whole point of normalising them there. */
    for (int r = 0; r < CRAW_GRAIN_RESPONSE_NB; r++) {
        p.response = r;
        CHECK(craw_grain_build(&p, &o) == 0, "%s",
              craw_grain_response_name(r));
        CHECK_NEAR(craw_grain_sigma_at(&o, 0.5f), o.sigma_signal,
                   0.06 * o.sigma_signal, "%s at mid grey",
                   craw_grain_response_name(r));
    }
}

static void craw_test_chroma(void)
{
    struct craw_grain_params p = base(1920);
    struct craw_grain_out o;

    p.chroma = 0.0f;
    CHECK(craw_grain_build(&p, &o) == 0, "chroma 0");
    CHECK(o.data.type == PL_FILM_GRAIN_AV1, "expected AV1");
    CHECK(o.data.params.av1.num_points_uv[0] == 0 &&
          o.data.params.av1.num_points_uv[1] == 0,
          "chroma 0 must leave the chroma channels without points");

    p.chroma = 0.5f;
    CHECK(craw_grain_build(&p, &o) == 0, "chroma 0.5");
    CHECK(o.data.params.av1.num_points_uv[0] > 0,
          "chroma 0.5 must give the chroma channels points");
    /* Chroma grain below luma grain: the points say so directly, since both
     * are solved against their own template at the same scaling shift. */
    const struct pl_av1_grain_data *d = &o.data.params.av1;
    int mid_y = d->points_y[d->num_points_y / 2][1];
    int mid_u = d->points_uv[0][d->num_points_uv[0] / 2][1];
    CHECK(mid_u < mid_y, "chroma 0.5 must be quieter than luma: %d against %d",
          mid_u, mid_y);
}

/* Seeds change the grain and nothing else. */
static void craw_test_seed(void)
{
    struct craw_grain_params p = base(1920);
    struct craw_grain_out a, b;

    CHECK(craw_grain_build(&p, &a) == 0, "seed A");
    p.seed = 0x99887766ULL;
    CHECK(craw_grain_build(&p, &b) == 0, "seed B");

    CHECK(a.data.seed != b.data.seed, "a new seed must reach the grain data");
    CHECK(a.data.params.av1.scaling_shift == b.data.params.av1.scaling_shift &&
          a.data.params.av1.ar_coeff_lag == b.data.params.av1.ar_coeff_lag &&
          !memcmp(a.data.params.av1.ar_coeffs_y,
                  b.data.params.av1.ar_coeffs_y, 24) &&
          !memcmp(a.data.params.av1.points_y, b.data.params.av1.points_y,
                  sizeof(a.data.params.av1.points_y)),
          "only the seed may differ between two frames of the same look");
    CHECK_NEAR(a.sigma_built, b.sigma_built, 1e-6, "amplitude across seeds");

    /* The AV1 template only takes sixteen bits of it. */
    CHECK(b.data.seed == (0x99887766ULL & 0xFFFF),
          "AV1 seeds are 16 bit, got %llu", (unsigned long long)b.data.seed);
}

static void craw_test_validator(void)
{
    struct craw_grain_params p = base(1920);
    struct craw_grain_out o;
    CHECK(craw_grain_build(&p, &o) == 0, "build");

    /* Each of these is a constraint the specification places on the data,
     * so each has to be caught. */
    struct pl_film_grain_data d = o.data;
    d.params.av1.points_y[2][0] = d.params.av1.points_y[1][0];
    CHECK(craw_grain_validate(&d) != NULL, "equal point values must fail");

    d = o.data;
    d.params.av1.scaling_shift = 12;
    CHECK(craw_grain_validate(&d) != NULL, "scaling_shift 12 must fail");

    d = o.data;
    d.params.av1.ar_coeff_lag = 4;
    CHECK(craw_grain_validate(&d) != NULL, "ar_coeff_lag 4 must fail");

    d = o.data;
    d.params.av1.ar_coeff_shift = 5;
    CHECK(craw_grain_validate(&d) != NULL, "ar_coeff_shift 5 must fail");

    d = o.data;
    d.params.av1.grain_scale_shift = 4;
    CHECK(craw_grain_validate(&d) != NULL, "grain_scale_shift 4 must fail");

    d = o.data;
    d.params.av1.num_points_y = 15;
    CHECK(craw_grain_validate(&d) != NULL, "15 luma points must fail");

    d = o.data;
    d.params.av1.ar_coeffs_y[23] = 1;
    CHECK(craw_grain_validate(&d) != NULL,
          "coefficients past the lag must fail");

    /* H.274's own constraints, including the one that is libplacebo's
     * rather than the specification's. */
    p.model = CRAW_GRAIN_MODEL_H274;
    CHECK(craw_grain_build(&p, &o) == 0, "h274 build");
    CHECK(!craw_grain_validate(&o.data), "h274 must validate");

    d = o.data;
    d.params.h274.model_id = 1;
    CHECK(craw_grain_validate(&d) != NULL,
          "model 1 must fail: libplacebo does not implement it");

    d = o.data;
    d.params.h274.log2_scale_factor = 16;
    CHECK(craw_grain_validate(&d) != NULL, "log2_scale_factor 16 must fail");

    d = o.data;
    d.params.h274.component_model_present[0] = false;
    d.params.h274.component_model_present[1] = false;
    d.params.h274.component_model_present[2] = false;
    CHECK(craw_grain_validate(&d) != NULL, "no component present must fail");

    CHECK(craw_grain_validate(NULL) != NULL, "NULL must fail");
}

static void craw_test_keys(void)
{
    for (int g = 0; g < CRAW_GRAIN_GAUGE_NB; g++)
        CHECK(craw_grain_gauge_from_key(craw_grain_gauge_key(g)) == g,
              "gauge key round trip for %s", craw_grain_gauge_key(g));
    for (int s = 0; s < CRAW_GRAIN_STOCK_NB; s++)
        CHECK(craw_grain_stock_from_key(craw_grain_stock_key(s)) == s,
              "stock key round trip for %s", craw_grain_stock_key(s));
    CHECK(craw_grain_gauge_from_key("SUPER35") == CRAW_GRAIN_GAUGE_SUPER35,
          "keys must be case insensitive");
    CHECK(craw_grain_gauge_from_key("nonsense") == -1, "unknown key");
    CHECK(craw_grain_stock_from_key(NULL) == -1, "NULL key");
    CHECK(craw_grain_response_from_key("print") == CRAW_GRAIN_RESPONSE_PRINT,
          "response key");
}

static void craw_test_bad_params(void)
{
    struct craw_grain_params p = base(1920);
    struct craw_grain_out o;

    p.width = 0;
    CHECK(craw_grain_build(&p, &o) < 0, "a zero width must be rejected");
    p = base(1920);
    p.gauge = CRAW_GRAIN_GAUGE_NB;
    CHECK(craw_grain_build(&p, &o) < 0, "an out of range gauge");
    p = base(1920);
    p.stock = CRAW_GRAIN_STOCK_NB;
    CHECK(craw_grain_build(&p, &o) < 0, "an out of range stock");
    CHECK(craw_grain_build(NULL, &o) < 0, "NULL params");
    CHECK(craw_grain_build(&p, NULL) < 0, "NULL out");
}

/* The AV1 grain template, against dav1d.
 *
 * dav1d is an independent AV1 decoder, so its generate_grain_y_c and its
 * dav1d_gaussian_sequence are a real second opinion rather than a copy of
 * the same source. Two things were checked to produce these vectors:
 * libplacebo's gaussian_sequence (which grain.c carries) is byte for byte
 * dav1d's dav1d_gaussian_sequence, all 2048 entries; and dav1d's generator,
 * compiled standalone at 8 bits against its own table, produces exactly the
 * numbers below. See the scratchpad's ref.c. */
static void craw_test_template(void)
{
    static const struct {
        const char *name;
        uint16_t seed;
        int gss, lag, shift;
        int8_t coeffs[24];
        int16_t row0[16];       /* buf[9][9..24], the top left of the LUT */
        int16_t mid[16];        /* buf[49][49..64] */
        double sd;              /* over the 64x64 window libplacebo uploads */
    } vectors[] = {
        {
            .name = "lag 0, grain_scale_shift 0", .seed = 0x1234,
            .gss = 0, .lag = 0, .shift = 6,
            .row0 = { -39,  -5,  16,  27, -19,   0,  24,  12,
                       24,  56,  25,  33,   3, -13,   8, -38 },
            .mid  = {  67, -50,  43,   9,  31,  -4, -38, -36,
                       46,  38, -42, -40,  41,   4,  48,  -1 },
            .sd = 32.088024,
        }, {
            .name = "lag 0, grain_scale_shift 2", .seed = 0x4321,
            .gss = 2, .lag = 0, .shift = 6,
            .row0 = {   6,   3,   3, -14,   6,  -3,  12,   0,
                       -5,   7, -15,   9,  -9,   2,   3, -18 },
            .mid  = { -10,   5,   4, -11, -16,   1,   5,   4,
                        3, -10,  11,  12,  -6,  -2,  18,   0 },
            .sd = 7.989577,
        }, {
            .name = "lag 1, separable a = 92/128", .seed = 0x1234,
            .gss = 1, .lag = 1, .shift = 7,
            .coeffs = { -66, 92, 0, 92 },
            .row0 = { -15, -41, -11, -11,  21,   9,  29,  43,
                       47,  69,  37,  71,  42,  41,  19, -19 },
            .mid  = {  31, -11,   8,   8,  10,  -7, -23,  -5,
                       28,  50,   4, -60, -44, -61, -24,  -6 },
            .sd = 31.948087,
        },
    };

    static int16_t buf[73][82];
    for (size_t v = 0; v < sizeof(vectors) / sizeof(*vectors); v++) {
        struct pl_av1_grain_data d = {0};
        d.grain_scale_shift = vectors[v].gss;
        d.ar_coeff_lag = vectors[v].lag;
        d.ar_coeff_shift = vectors[v].shift;
        memcpy(d.ar_coeffs_y, vectors[v].coeffs, 24);

        float sd = craw_grain_template_y(buf, &d, 8, vectors[v].seed);

        for (int i = 0; i < 16; i++) {
            CHECK(buf[9][9 + i] == vectors[v].row0[i],
                  "%s: buf[9][%d] is %d, dav1d says %d",
                  vectors[v].name, 9 + i, buf[9][9 + i], vectors[v].row0[i]);
            CHECK(buf[49][49 + i] == vectors[v].mid[i],
                  "%s: buf[49][%d] is %d, dav1d says %d",
                  vectors[v].name, 49 + i, buf[49][49 + i], vectors[v].mid[i]);
        }
        CHECK_NEAR(sd, vectors[v].sd, 1e-4, "%s: template sd",
                   vectors[v].name);
    }

    /* Nothing may leave the range the specification allows at 8 bits. */
    struct pl_av1_grain_data d = {0};
    d.ar_coeff_lag = 1;
    d.ar_coeff_shift = 7;
    d.ar_coeffs_y[0] = -110;
    d.ar_coeffs_y[1] = 119;
    d.ar_coeffs_y[3] = 119;
    craw_grain_template_y(buf, &d, 8, 0x1234);
    for (int y = 0; y < 73; y++)
        for (int x = 0; x < 82; x++)
            if (buf[y][x] < -128 || buf[y][x] > 127) {
                CHECK(0, "template value %d at [%d][%d] is outside "
                      "[-128, 127]", buf[y][x], y, x);
                return;
            }
}

int main(void)
{
    craw_test_template();
    craw_test_presets();
    craw_test_amount();
    craw_test_size();
    craw_test_gauge_order();
    craw_test_response();
    craw_test_chroma();
    craw_test_seed();
    craw_test_validator();
    craw_test_keys();
    craw_test_bad_params();

    if (failures) {
        printf("test_grain: %d failure(s)\n", failures);
        return 1;
    }

    printf("test_grain: all checks passed\n");
    return 0;
}
