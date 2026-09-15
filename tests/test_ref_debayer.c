/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* The three CPU reference debayers on synthetic mosaics with known
 * analytic results, per DESIGN.md sections 5 and 11. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <craw/profile.h>
#include <craw/pipeline.h>
#include <ref/ref.h>

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

static const char *craw_test_method_name(enum CRAWDebayer m)
{
    switch (m) {
    case CRAW_DEBAYER_BILINEAR: return "bilinear";
    case CRAW_DEBAYER_MHC:      return "MHC";
    case CRAW_DEBAYER_RCD:      return "RCD";
    default:                    return "?";
    }
}

/* Unit profile: no black level, no clipping headroom, no gains, so the
 * prepped value of a sample is exactly sample / 65535 and every analytic
 * expectation below can be written in those terms. */
static void craw_test_unit_profile(CRAWProfile *p)
{
    craw_profile_init(p);
    p->cfa = CRAW_CFA_RGGB;
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

/* Every finite, alpha 1, nothing silently NaN at a mirrored border. */
static void craw_test_finite(const float *rgb, uint32_t w, uint32_t h,
                             enum CRAWDebayer m, const char *what)
{
    int bad = 0;

    for (size_t i = 0; i < (size_t)w * h; i++) {
        for (int c = 0; c < 4; c++) {
            if (!isfinite(rgb[i * 4 + c]))
                bad++;
        }
        if (rgb[i * 4 + 3] != 1.0f)
            bad++;
    }

    CHECK(bad == 0, "%s on %s: %d non-finite or non-unit-alpha components",
          craw_test_method_name(m), what, bad);
}

static void craw_test_constant(void)
{
    /* Arbitrary, distinct, well away from the clip. */
    static const uint16_t site[4] = { 12000, 34000, 34000, 51000 };
    const uint32_t w = 24, h = 16;
    const float want[3] = { 12000.0f / 65535.0f, 34000.0f / 65535.0f,
                            51000.0f / 65535.0f };
    CRAWProfile p;
    uint16_t *bayer = malloc((size_t)w * h * sizeof(*bayer));
    float *rgb = malloc((size_t)w * h * 4 * sizeof(*rgb));

    craw_test_unit_profile(&p);

    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++)
            bayer[y * w + x] = site[(y & 1) * 2 + (x & 1)];
    }

    for (int m = 0; m < CRAW_DEBAYER_NB; m++) {
        double worst = 0.0;
        int worst_x = -1, worst_y = -1, worst_c = -1;
        int ret = craw_ref_debayer(bayer, w, h, w, &p, m, rgb);

        CHECK(ret == 0, "%s failed: %d", craw_test_method_name(m), ret);
        craw_test_finite(rgb, w, h, m, "a constant colour");

        for (uint32_t y = 0; y < h; y++) {
            for (uint32_t x = 0; x < w; x++) {
                for (int c = 0; c < 3; c++) {
                    double d = fabs(rgb[((size_t)y * w + x) * 4 + c] - want[c]);
                    if (d > worst) {
                        worst = d;
                        worst_x = (int)x; worst_y = (int)y; worst_c = c;
                    }
                }
            }
        }

        /* Exact up to float rounding, borders included: every kernel of
         * every method sums to one over the colour it reconstructs. */
        CHECK(worst <= 1e-6, "%s on a constant colour: worst error %.3g at "
              "(%d, %d) channel %d", craw_test_method_name(m), worst,
              worst_x, worst_y, worst_c);
    }

    free(bayer);
    free(rgb);
}

/* A grey linear ramp: every site of the mosaic samples the same linear
 * function of x (or of y), so a correct reconstruction is that function
 * again in all three channels. */
static void craw_test_ramp(bool vertical)
{
    const uint32_t w = 32, h = 32;
    const int base = 3000, step = 800;
    CRAWProfile p;
    uint16_t *bayer = malloc((size_t)w * h * sizeof(*bayer));
    float *rgb = malloc((size_t)w * h * 4 * sizeof(*rgb));

    craw_test_unit_profile(&p);

    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++)
            bayer[y * w + x] =
                (uint16_t)(base + step * (int)(vertical ? y : x));
    }

    for (int m = 0; m < CRAW_DEBAYER_NB; m++) {
        /* Mirroring a ramp does not produce a ramp, so the comparison is
         * only valid where the stencil stays inside the image: 2 pixels for
         * bilinear and MHC, 4 for RCD (its chroma pass reaches a green
         * plane that was itself built from a 5x5 low pass). */
        int margin = m == CRAW_DEBAYER_RCD ? 4 : 2;
        double tol = m == CRAW_DEBAYER_RCD ? 1e-3 : 1e-5;
        double worst = 0.0;
        int worst_x = -1, worst_y = -1, worst_c = -1;
        int ret = craw_ref_debayer(bayer, w, h, w, &p, m, rgb);

        CHECK(ret == 0, "%s failed: %d", craw_test_method_name(m), ret);
        craw_test_finite(rgb, w, h, m, vertical ? "a vertical ramp"
                                                : "a horizontal ramp");

        for (uint32_t y = margin; y < h - margin; y++) {
            for (uint32_t x = margin; x < w - margin; x++) {
                float want = (float)(base + step * (int)(vertical ? y : x)) /
                             65535.0f;

                for (int c = 0; c < 3; c++) {
                    double d = fabs(rgb[((size_t)y * w + x) * 4 + c] - want);
                    if (d > worst) {
                        worst = d;
                        worst_x = (int)x; worst_y = (int)y; worst_c = c;
                    }
                }
            }
        }

        CHECK(worst <= tol, "%s on a %s ramp: worst error %.3g at (%d, %d) "
              "channel %d (tol %.3g)", craw_test_method_name(m),
              vertical ? "vertical" : "horizontal", worst, worst_x, worst_y,
              worst_c, tol);
    }

    free(bayer);
    free(rgb);
}

/* Nothing analytic to assert here, but a checkerboard is the worst case for
 * every directional decision and a random mosaic exercises the ratio
 * guards; both must stay finite everywhere. */
static void craw_test_hostile(void)
{
    const uint32_t w = 40, h = 24;
    CRAWProfile p;
    uint32_t seed = 0x12345678u;
    uint16_t *checker = malloc((size_t)w * h * sizeof(*checker));
    uint16_t *noise = malloc((size_t)w * h * sizeof(*noise));
    uint16_t *dark = calloc((size_t)w * h, sizeof(*dark));
    float *rgb = malloc((size_t)w * h * 4 * sizeof(*rgb));

    craw_test_unit_profile(&p);

    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            checker[y * w + x] = ((x >> 1) + (y >> 1)) & 1 ? 65535 : 0;
            noise[y * w + x] = (uint16_t)(craw_test_rand(&seed) & 0xFFFF);
        }
    }

    for (int m = 0; m < CRAW_DEBAYER_NB; m++) {
        int ret;

        ret = craw_ref_debayer(checker, w, h, w, &p, m, rgb);
        CHECK(ret == 0, "%s failed: %d", craw_test_method_name(m), ret);
        craw_test_finite(rgb, w, h, m, "a checkerboard");

        ret = craw_ref_debayer(noise, w, h, w, &p, m, rgb);
        CHECK(ret == 0, "%s failed: %d", craw_test_method_name(m), ret);
        craw_test_finite(rgb, w, h, m, "noise");

        /* All zero is the case where every RCD ratio divides by its eps. */
        ret = craw_ref_debayer(dark, w, h, w, &p, m, rgb);
        CHECK(ret == 0, "%s failed: %d", craw_test_method_name(m), ret);
        craw_test_finite(rgb, w, h, m, "black");
    }

    free(checker);
    free(noise);
    free(dark);
    free(rgb);
}

/* The smallest frame the stencils can be evaluated on: every fetch of a 5x5
 * kernel is mirrored, some of them twice. */
static void craw_test_tiny(void)
{
    static const uint16_t site[4] = { 9000, 22000, 22000, 41000 };
    const uint32_t w = 4, h = 4;
    const float want[3] = { 9000.0f / 65535.0f, 22000.0f / 65535.0f,
                            41000.0f / 65535.0f };
    CRAWProfile p;
    uint16_t bayer[16];
    float rgb[16 * 4];

    craw_test_unit_profile(&p);

    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++)
            bayer[y * w + x] = site[(y & 1) * 2 + (x & 1)];
    }

    for (int m = 0; m < CRAW_DEBAYER_NB; m++) {
        int ret = craw_ref_debayer(bayer, w, h, w, &p, m, rgb);
        double worst = 0.0;

        CHECK(ret == 0, "%s on 4x4 failed: %d", craw_test_method_name(m), ret);
        craw_test_finite(rgb, w, h, m, "a 4x4 frame");

        for (size_t i = 0; i < (size_t)w * h; i++) {
            for (int c = 0; c < 3; c++)
                worst = fmax(worst, fabs(rgb[i * 4 + c] - want[c]));
        }

        CHECK(worst <= 1e-6, "%s on a 4x4 constant frame: worst error %.3g",
              craw_test_method_name(m), worst);
    }
}

/* DESIGN.md 4.1: black subtraction, normalization, as-shot gains, clip. */
static void craw_test_prep(void)
{
    const uint32_t w = 8, h = 8;
    const float black = 256.0f / 65535.0f, white = 63743.0f / 65535.0f;
    const float wb[3] = { 2.02539f, 1.0f, 1.66797f };
    CRAWProfile p;
    uint16_t bayer[64];
    float rgb[64 * 4];
    int ret;

    craw_profile_init(&p);
    for (int s = 0; s < 4; s++) {
        p.black_level[s] = black;
        p.white_level[s] = white;
    }
    memcpy(p.as_shot_wb, wb, sizeof(wb));

    /* R below black, G mid scale, B above the white balanced clip. */
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            static const uint16_t site[4] = { 100, 32000, 32000, 60000 };
            bayer[y * w + x] = site[(y & 1) * 2 + (x & 1)];
        }
    }

    ret = craw_ref_debayer(bayer, w, h, w, &p, CRAW_DEBAYER_BILINEAR, rgb);
    CHECK(ret == 0, "prep test debayer failed: %d", ret);

    {
        float mid = (32000.0f / 65535.0f - black) / (white - black);
        float hi = (60000.0f / 65535.0f - black) / (white - black) * wb[2];
        double worst_r = 0.0, worst_g = 0.0, worst_b = 0.0;

        CHECK(hi > 1.0f, "the blue sample must exceed the clip to be a test");

        for (size_t i = 0; i < (size_t)w * h; i++) {
            worst_r = fmax(worst_r, fabs(rgb[i * 4 + 0] - 0.0f));
            worst_g = fmax(worst_g, fabs(rgb[i * 4 + 1] - mid));
            worst_b = fmax(worst_b, fabs(rgb[i * 4 + 2] - 1.0f));
        }

        CHECK(worst_r <= 1e-7, "a sample below black must clamp to 0 (%.3g)",
              worst_r);
        CHECK(worst_g <= 1e-6, "a mid scale sample must normalize (%.3g)",
              worst_g);
        CHECK(worst_b <= 1e-7, "a sample above the balanced clip must clamp "
              "to 1 (%.3g)", worst_b);
    }
}

static void craw_test_errors(void)
{
    const uint32_t w = 8, h = 8;
    CRAWProfile p;
    uint16_t bayer[64] = { 0 };
    float rgb[64 * 4];

    craw_test_unit_profile(&p);

    p.cfa = CRAW_CFA_BGGR;
    CHECK(craw_ref_debayer(bayer, w, h, w, &p, CRAW_DEBAYER_RCD, rgb) ==
          CRAW_ERROR(ENOTSUP), "a non-RGGB CFA must fail with ENOTSUP");
    p.cfa = CRAW_CFA_RGGB;

    CHECK(craw_ref_debayer(bayer, w, h, w, &p, CRAW_DEBAYER_NB, rgb) ==
          CRAW_ERROR(EINVAL), "an unknown method must fail with EINVAL");
    CHECK(craw_ref_debayer(NULL, w, h, w, &p, CRAW_DEBAYER_RCD, rgb) ==
          CRAW_ERROR(EINVAL), "a NULL mosaic must fail with EINVAL");
    CHECK(craw_ref_debayer(bayer, 7, h, 7, &p, CRAW_DEBAYER_RCD, rgb) ==
          CRAW_ERROR(EINVAL), "an odd width must fail with EINVAL");
    CHECK(craw_ref_debayer(bayer, w, h, 4, &p, CRAW_DEBAYER_RCD, rgb) ==
          CRAW_ERROR(EINVAL), "a stride below the width must fail with EINVAL");
}

/* The whole chain on a flat neutral frame: a neutral under the as-shot
 * illuminant has to come out neutral, at the luminance the gain says. */
static void craw_test_process(void)
{
    const uint32_t w = 16, h = 16;
    CRAWProfile p;
    /* Zero initialized rather than through craw_pipeline_params_init(): the
     * CPU reference must not drag in the GPU pipeline. */
    CRAWPipelineParams pp = { 0 };
    CRAWFrameParams fp = { 0 };
    uint16_t bayer[256];
    float rgb[256 * 4];
    double worst = 0.0;
    int ret;

    /* Rows of a camera matrix sum to the XYZ of D65 white, so a balanced
     * neutral of v comes out as v times that white. */
    static const float cam[3][3] = {
        { 0.9505f / 3.0f, 0.9505f / 3.0f, 0.9505f / 3.0f },
        { 1.0000f / 3.0f, 1.0000f / 3.0f, 1.0000f / 3.0f },
        { 1.0890f / 3.0f, 1.0890f / 3.0f, 1.0890f / 3.0f },
    };

    craw_profile_init(&p);
    p.as_shot_wb[0] = 2.0f;
    p.as_shot_wb[2] = 1.6f;
    memcpy(p.as_shot_matrix, cam, sizeof(cam));
    p.gain = 2.0f;

    pp.width = w;
    pp.height = h;
    pp.cfa = CRAW_CFA_RGGB;
    pp.debayer = CRAW_DEBAYER_MHC;
    pp.out_space = CRAW_CS_BT709_LINEAR;

    fp.profile = &p;
    fp.wb_mode = CRAW_WB_AS_SHOT;

    /* A neutral of 0.25 in raw units under the as-shot gains. */
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            static const float chan[4] = { 2.0f, 1.0f, 1.0f, 1.6f };
            int s = (int)((y & 1) * 2 + (x & 1));
            bayer[y * w + x] = (uint16_t)(0.25f * 65535.0f / chan[s] + 0.5f);
        }
    }

    ret = craw_ref_process(bayer, w, h, w, &pp, &fp, rgb);
    CHECK(ret == 0, "craw_ref_process failed: %d", ret);

    for (size_t i = 0; i < (size_t)w * h; i++) {
        for (int c = 0; c < 3; c++)
            worst = fmax(worst, fabs(rgb[i * 4 + c] - 0.5));
        CHECK(rgb[i * 4 + 3] == 1.0f, "alpha must be 1");
    }

    CHECK(worst <= 2e-4, "a neutral must stay neutral through the chain "
          "(worst %.3g)", worst);

    /* An extra stop doubles it. */
    fp.exposure_stops = 1.0f;
    ret = craw_ref_process(bayer, w, h, w, &pp, &fp, rgb);
    CHECK(ret == 0, "craw_ref_process failed: %d", ret);
    worst = 0.0;
    for (size_t i = 0; i < (size_t)w * h; i++) {
        for (int c = 0; c < 3; c++)
            worst = fmax(worst, fabs(rgb[i * 4 + c] - 1.0));
    }
    CHECK(worst <= 4e-4, "an extra stop must double the output (worst %.3g)",
          worst);
}

/* RCD is a directional method: on a hard vertical edge it must not average
 * across the edge the way the two fixed kernels do. This is the only
 * property that separates it from them at the same cost, so a degenerate
 * RCD that quietly fell back to an isotropic estimate would still pass
 * every analytic check above. */
static void craw_test_directional(void)
{
    const uint32_t w = 32, h = 32;
    CRAWProfile p;
    uint16_t *bayer = malloc((size_t)w * h * sizeof(*bayer));
    float *rgb = malloc((size_t)w * h * 4 * sizeof(*rgb));
    double err[CRAW_DEBAYER_NB] = { 0.0 };

    craw_test_unit_profile(&p);

    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++)
            bayer[y * w + x] = x < w / 2 ? 6000 : 50000;
    }

    for (int m = 0; m < CRAW_DEBAYER_NB; m++) {
        int ret = craw_ref_debayer(bayer, w, h, w, &p, m, rgb);

        CHECK(ret == 0, "%s failed: %d", craw_test_method_name(m), ret);
        craw_test_finite(rgb, w, h, m, "a vertical step");

        for (uint32_t y = 4; y < h - 4; y++) {
            for (uint32_t x = 4; x < w - 4; x++) {
                float want = (x < w / 2 ? 6000.0f : 50000.0f) / 65535.0f;

                for (int c = 0; c < 3; c++)
                    err[m] += fabs(rgb[((size_t)y * w + x) * 4 + c] - want);
            }
        }
    }

    CHECK(err[CRAW_DEBAYER_RCD] < 0.01 * err[CRAW_DEBAYER_BILINEAR],
          "RCD must resolve a vertical step far better than bilinear "
          "(%.4f against %.4f)", err[CRAW_DEBAYER_RCD],
          err[CRAW_DEBAYER_BILINEAR]);
    CHECK(err[CRAW_DEBAYER_RCD] < 0.01 * err[CRAW_DEBAYER_MHC],
          "RCD must resolve a vertical step far better than MHC "
          "(%.4f against %.4f)", err[CRAW_DEBAYER_RCD],
          err[CRAW_DEBAYER_MHC]);

    free(bayer);
    free(rgb);
}

int main(void)
{
    craw_test_constant();
    craw_test_ramp(false);
    craw_test_ramp(true);
    craw_test_directional();
    craw_test_hostile();
    craw_test_tiny();
    craw_test_prep();
    craw_test_errors();
    craw_test_process();

    if (failures) {
        printf("test_ref_debayer: %d failure(s)\n", failures);
        return 1;
    }

    printf("test_ref_debayer: all checks passed\n");
    return 0;
}
