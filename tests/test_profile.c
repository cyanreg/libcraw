/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Profile evaluation, locus mapping and output colour space matrices,
 * against the values in DESIGN.md sections 8 and 9, and the noise model's
 * evaluation and the S1M2 table of section 14.2. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <craw/noise.h>
#include <craw/profile.h>

#if CRAW_TOOL_HAVE_LAVU
#include <libavutil/dict.h>
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#include <libavutil/raw_color_params.h>
#endif

#include "profiles.h"

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

/* Panasonic DC-S1M2, clip P1000387.MOV, per DESIGN.md section 9. */
static const float craw_test_frame_matrix[3][3] = {
    { 0.695646f, 0.152540f,  0.102269f },
    { 0.303238f, 0.760187f, -0.063425f },
    { 0.005748f, 0.031326f,  1.051984f },
};

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
    static const CRAWMatrixKnot mat[] = {
        {  2300.0f, { { 0.6535f, 0.1947f, 0.1022f },
                      { 0.2831f, 0.6975f, 0.0193f },
                      { 0.0113f, -0.1074f, 1.1852f } } },
        {  2800.0f, { { 0.6601f, 0.1881f, 0.1022f },
                      { 0.2862f, 0.7073f, 0.0064f },
                      { 0.0104f, -0.0857f, 1.1644f } } },
        {  4000.0f, { { 0.6759f, 0.1723f, 0.1023f },
                      { 0.2938f, 0.7308f, -0.0246f },
                      { 0.0083f, -0.0337f, 1.1144f } } },
        {  5000.0f, { { 0.6891f, 0.1591f, 0.1023f },
                      { 0.3001f, 0.7504f, -0.0505f },
                      { 0.0066f, 0.0096f, 1.0728f } } },
        {  6500.0f, { { 0.7088f, 0.1394f, 0.1023f },
                      { 0.3095f, 0.7798f, -0.0893f },
                      { 0.0040f, 0.0747f, 1.0104f } } },
        {  7500.0f, { { 0.7218f, 0.1263f, 0.1023f },
                      { 0.3158f, 0.7991f, -0.1149f },
                      { 0.0023f, 0.1176f, 0.9692f } } },
        { 10000.0f, { { 0.7218f, 0.1263f, 0.1023f },
                      { 0.3158f, 0.7991f, -0.1149f },
                      { 0.0023f, 0.1176f, 0.9692f } } },
        { 15000.0f, { { 0.7218f, 0.1263f, 0.1023f },
                      { 0.3158f, 0.7991f, -0.1149f },
                      { 0.0023f, 0.1176f, 0.9692f } } },
    };

    craw_profile_init(p);

    for (int s = 0; s < 4; s++) {
        p->black_level[s] = 256.0f / 65535.0f;
        p->white_level[s] = 63743.0f / 65535.0f;
    }

    p->as_shot_wb[0] = 2.02539f;
    p->as_shot_wb[1] = 1.0f;
    p->as_shot_wb[2] = 1.66797f;
    memcpy(p->as_shot_matrix, craw_test_frame_matrix,
           sizeof(p->as_shot_matrix));
    p->as_shot_cct = 5500.0f;
    p->gain = 23.04f;

    memcpy(p->wb_knots, wb, sizeof(wb));
    p->nb_wb_knots = CRAW_ARRAY_ELEMS(wb);
    memcpy(p->matrix_knots, mat, sizeof(mat));
    p->nb_matrix_knots = CRAW_ARRAY_ELEMS(mat);

    snprintf(p->name, sizeof(p->name), "Panasonic DC-S1M2");
}

static void craw_test_validate(void)
{
    CRAWProfile p;

    craw_test_profile_init(&p);
    CHECK(craw_profile_validate(&p) == 0,
          "the section 9 profile must validate");

    craw_test_profile_init(&p);
    p.white_level[2] = p.black_level[2];
    CHECK(craw_profile_validate(&p) == CRAW_ERROR(EINVAL),
          "a white level at the black level must be rejected");

    craw_test_profile_init(&p);
    p.as_shot_wb[1] = 0.0f;
    CHECK(craw_profile_validate(&p) == CRAW_ERROR(EINVAL),
          "a zero white balance gain must be rejected");

    craw_test_profile_init(&p);
    memset(p.as_shot_matrix, 0, sizeof(p.as_shot_matrix));
    CHECK(craw_profile_validate(&p) == CRAW_ERROR(EINVAL),
          "a singular camera matrix must be rejected");

    craw_test_profile_init(&p);
    p.wb_knots[3].cct = p.wb_knots[4].cct;
    CHECK(craw_profile_validate(&p) == CRAW_ERROR(EINVAL),
          "unsorted white balance knots must be rejected");

    craw_test_profile_init(&p);
    p.gain = -1.0f;
    CHECK(craw_profile_validate(&p) == CRAW_ERROR(EINVAL),
          "a negative gain must be rejected");
}

static void craw_test_eval(void)
{
    CRAWProfile p;
    CRAWColorParams cp;
    CRAWIlluminant il = { .cct = 5500.0f, .tint = 0.0f };
    int ret;

    craw_test_profile_init(&p);

    /* As shot returns the header values untouched. */
    ret = craw_profile_eval(&p, CRAW_WB_AS_SHOT, NULL, &cp);
    CHECK(ret == 0, "as-shot eval failed: %d", ret);
    CHECK(cp.wb[0] == p.as_shot_wb[0] && cp.wb[1] == p.as_shot_wb[1] &&
          cp.wb[2] == p.as_shot_wb[2], "as-shot gains must be verbatim");
    CHECK(memcmp(cp.cam_to_xyz, craw_test_frame_matrix,
                 sizeof(craw_test_frame_matrix)) == 0,
          "as-shot matrix must be verbatim");
    CHECK_NEAR(cp.gain, 23.04, 1e-6, "as-shot gain");

    /* Re-balancing to the frame's own illuminant has to reproduce what the
     * camera wrote into the frame header, which is the whole point of the
     * per-CCT tables. */
    ret = craw_profile_eval(&p, CRAW_WB_REBALANCE, &il, &cp);
    CHECK(ret == 0, "rebalance eval failed: %d", ret);
    CHECK_NEAR(cp.wb[0], 2.02539, 1e-3, "rebalanced R gain at 5500 K");
    CHECK_NEAR(cp.wb[1], 1.0, 1e-6, "rebalanced G gain at 5500 K");
    CHECK_NEAR(cp.wb[2], 1.66797, 1e-3, "rebalanced B gain at 5500 K");

    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            CHECK_NEAR(cp.cam_to_xyz[r][c], craw_test_frame_matrix[r][c], 5e-4,
                       "rebalanced matrix [%d][%d] at 5500 K", r, c);
        }
    }

    /* Tint raises R and B by exp(tint / sqrt(2)) and leaves G alone. */
    il.tint = 0.1f;
    ret = craw_profile_eval(&p, CRAW_WB_REBALANCE, &il, &cp);
    CHECK(ret == 0, "tinted eval failed: %d", ret);
    CHECK_NEAR(cp.wb[0], 2.0253 * exp(0.1 / sqrt(2.0)), 1e-5, "tinted R gain");
    CHECK_NEAR(cp.wb[1], 1.0, 1e-6, "tinted G gain");
    CHECK_NEAR(cp.wb[2], 1.6676 * exp(0.1 / sqrt(2.0)), 1e-5, "tinted B gain");
    il.tint = 0.0f;

    /* Outside the table the ends are held. */
    il.cct = 100.0f;
    ret = craw_profile_eval(&p, CRAW_WB_REBALANCE, &il, &cp);
    CHECK(ret == 0, "cold eval failed: %d", ret);
    CHECK_NEAR(cp.wb[0], 0.8454, 1e-6, "gains clamp to the first knot");
    il.cct = 1e6f;
    ret = craw_profile_eval(&p, CRAW_WB_REBALANCE, &il, &cp);
    CHECK(ret == 0, "hot eval failed: %d", ret);
    CHECK_NEAR(cp.wb[0], 2.6469, 1e-6, "gains clamp to the last knot");
    il.cct = 5500.0f;

    /* Gain picks up the exposure index ratio when both are known. */
    p.as_shot_ei = 640.0f;
    p.exposure_index = 1280.0f;
    ret = craw_profile_eval(&p, CRAW_WB_AS_SHOT, NULL, &cp);
    CHECK(ret == 0, "exposure index eval failed: %d", ret);
    CHECK_NEAR(cp.gain, 23.04 * 2.0, 1e-4, "gain times exposure index ratio");
    p.exposure_index = 320.0f;
    ret = craw_profile_eval(&p, CRAW_WB_REBALANCE, &il, &cp);
    CHECK(ret == 0, "exposure index eval failed: %d", ret);
    CHECK_NEAR(cp.gain, 23.04 * 0.5, 1e-4, "gain times exposure index ratio");
    p.as_shot_ei = 0.0f;
    ret = craw_profile_eval(&p, CRAW_WB_AS_SHOT, NULL, &cp);
    CHECK(ret == 0, "eval without an as-shot EI failed: %d", ret);
    CHECK_NEAR(cp.gain, 23.04, 1e-6, "gain without a known as-shot EI");
    p.exposure_index = 0.0f;

    /* Without a matrix table the as-shot matrix stands in; without a white
     * balance table there is nothing to re-balance to. */
    p.nb_matrix_knots = 0;
    ret = craw_profile_eval(&p, CRAW_WB_REBALANCE, &il, &cp);
    CHECK(ret == 0, "eval without a matrix table failed: %d", ret);
    CHECK(memcmp(cp.cam_to_xyz, craw_test_frame_matrix,
                 sizeof(craw_test_frame_matrix)) == 0,
          "an empty matrix table must fall back to the as-shot matrix");

    p.nb_wb_knots = 0;
    ret = craw_profile_eval(&p, CRAW_WB_REBALANCE, &il, &cp);
    CHECK(ret == CRAW_ERROR(ENOTSUP),
          "rebalance without a wb table must fail with ENOTSUP, got %d", ret);
    ret = craw_profile_eval(&p, CRAW_WB_AS_SHOT, NULL, &cp);
    CHECK(ret == 0, "as-shot must work without any table: %d", ret);
}

static void craw_test_locus(void)
{
    CRAWProfile p;
    CRAWIlluminant il;
    float rg, bg;
    int ret;

    craw_test_profile_init(&p);

    /* The raw ratios of the 5500 K knot must map back to exactly 5500 K
     * with no tint: the knots are locus samples. */
    ret = craw_profile_neutral_to_illuminant(&p, 1.0f / 2.0253f,
                                             1.0f / 1.6676f, &il);
    CHECK(ret == 0, "neutral_to_illuminant failed: %d", ret);
    CHECK_NEAR(il.cct, 5500.0, 1e-3, "cct of the 5500 K knot's ratios");
    CHECK_NEAR(il.tint, 0.0, 1e-3, "tint of the 5500 K knot's ratios");

    /* Every knot, both directions. */
    for (uint32_t i = 0; i < p.nb_wb_knots; i++) {
        CRAWIlluminant knot = { .cct = p.wb_knots[i].cct, .tint = 0.0f };

        ret = craw_profile_illuminant_to_neutral(&p, &knot, &rg, &bg);
        CHECK(ret == 0, "illuminant_to_neutral failed: %d", ret);
        CHECK_NEAR(rg, 1.0 / p.wb_knots[i].wb[0], 1e-6, "knot %u R/G", i);
        CHECK_NEAR(bg, 1.0 / p.wb_knots[i].wb[2], 1e-6, "knot %u B/G", i);

        ret = craw_profile_neutral_to_illuminant(&p, rg, bg, &il);
        CHECK(ret == 0, "neutral_to_illuminant failed: %d", ret);
        CHECK_NEAR(il.cct, p.wb_knots[i].cct, 1e-3,
                   "knot %u round trip cct", i);
        CHECK_NEAR(il.tint, 0.0, 1e-3, "knot %u round trip tint", i);
    }

    /* Points between the knots come back to within the sampling step. */
    for (double cct = 2100.0; cct < 11400.0; cct += 137.0) {
        CRAWIlluminant want = { .cct = (float)cct, .tint = 0.0f };

        ret = craw_profile_illuminant_to_neutral(&p, &want, &rg, &bg);
        CHECK(ret == 0, "illuminant_to_neutral failed: %d", ret);
        ret = craw_profile_neutral_to_illuminant(&p, rg, bg, &il);
        CHECK(ret == 0, "neutral_to_illuminant failed: %d", ret);

        /* Half of the 5 mired sampling step, in Kelvin. */
        CHECK_NEAR(1e6 / il.cct, 1e6 / cct, 2.5 + 1e-3,
                   "round trip of %g K in mired", cct);
        CHECK_NEAR(il.tint, 0.0, 2e-3, "round trip tint of %g K", cct);
    }

    /* The two mappings are each other's inverse off the locus too, at any
     * tint: the tint moves a neutral along the green direction and nothing
     * else, and the mapping back measures along the same direction. */
    for (double cct = 2300.0; cct < 11000.0; cct += 911.0) {
        for (double tint = -0.15; tint <= 0.15; tint += 0.05) {
            CRAWIlluminant want = { .cct = (float)cct, .tint = (float)tint };

            ret = craw_profile_illuminant_to_neutral(&p, &want, &rg, &bg);
            CHECK(ret == 0, "illuminant_to_neutral failed: %d", ret);
            ret = craw_profile_neutral_to_illuminant(&p, rg, bg, &il);
            CHECK(ret == 0, "neutral_to_illuminant failed: %d", ret);
            CHECK_NEAR(il.cct, cct, 0.05, "cct of %g K tint %+g", cct, tint);
            CHECK_NEAR(il.tint, tint, 1e-5, "tint of %g K tint %+g", cct, tint);
        }
    }

    /* A neutral moved off the locus by a known tint comes back with that
     * tint, and still at the temperature it was moved from. */
    for (int i = 0; i < 2; i++) {
        double tint = i ? -0.03 : 0.02;
        CRAWIlluminant want = { .cct = 5500.0f, .tint = (float)tint };

        ret = craw_profile_illuminant_to_neutral(&p, &want, &rg, &bg);
        CHECK(ret == 0, "illuminant_to_neutral failed: %d", ret);
        CHECK_NEAR(rg, exp(-tint / sqrt(2.0)) / 2.0253, 1e-6, "tinted R/G");
        CHECK_NEAR(bg, exp(-tint / sqrt(2.0)) / 1.6676, 1e-6, "tinted B/G");

        ret = craw_profile_neutral_to_illuminant(&p, rg, bg, &il);
        CHECK(ret == 0, "neutral_to_illuminant failed: %d", ret);
        CHECK_NEAR(il.tint, tint, 1e-3, "tint of a point moved by %g", tint);
        CHECK_NEAR(il.cct, 5500.0, 1e-3, "cct of a point moved by %g", tint);
    }

    /* Re-balancing to what a measured neutral says must cancel that
     * neutral's ratios, which is the property the AWB loop relies on. */
    {
        CRAWColorParams cp;
        double rg_in = 1.0 / 1.62, bg_in = 1.0 / 1.93;

        ret = craw_profile_neutral_to_illuminant(&p, (float)rg_in,
                                                 (float)bg_in, &il);
        CHECK(ret == 0, "neutral_to_illuminant failed: %d", ret);
        ret = craw_profile_eval(&p, CRAW_WB_REBALANCE, &il, &cp);
        CHECK(ret == 0, "rebalance eval failed: %d", ret);
        CHECK_NEAR(rg_in * cp.wb[0], 1.0, 2e-3,
                   "a measured neutral balances R to 1");
        CHECK_NEAR(bg_in * cp.wb[2], 1.0, 2e-3,
                   "a measured neutral balances B to 1");
    }

    /* A table with fewer than two knots has no locus. */
    p.nb_wb_knots = 1;
    ret = craw_profile_neutral_to_illuminant(&p, 0.5f, 0.6f, &il);
    CHECK(ret == CRAW_ERROR(ENOTSUP),
          "a one knot locus must fail with ENOTSUP, got %d", ret);
}

static void craw_test_colorspaces(void)
{
    /* DESIGN.md section 8, from the OCIO studio config. */
    static const float ap0[3][3] = {
        {  1.063496f,  0.006409f, -0.015807f },
        { -0.492074f,  1.368223f,  0.091337f },
        { -0.002816f,  0.004644f,  0.916419f },
    };
    static const float ap1[3][3] = {
        {  1.660585f, -0.315296f, -0.241509f },
        { -0.659926f,  1.608392f,  0.017299f },
        {  0.009003f, -0.003567f,  0.913643f },
    };
    /* The sRGB/BT.709 to ACEScg matrix as it is published everywhere,
     * Bradford adapted D65 to the ACES white. The library never stores it:
     * it comes out of XYZ_to_AP1 * BT709_to_XYZ, so matching it is a check
     * that both halves and the adaptation between them are right. */
    static const float srgb_to_ap1[3][3] = {
        { 0.6130974f, 0.3395231f, 0.0473795f },
        { 0.0701937f, 0.9163538f, 0.0134524f },
        { 0.0206156f, 0.1095698f, 0.8698146f },
    };
    /* ACEScg to ACES2065-1, from the ACES documentation. Both spaces share
     * the ACES white, so the two adaptations in the chain have to cancel
     * exactly; this is what catches an adaptation applied in the wrong
     * direction, which the published matrices above would not. */
    static const float ap1_to_ap0[3][3] = {
        {  0.6954522414f, 0.1406786965f, 0.1638690622f },
        {  0.0447945634f, 0.8596711185f, 0.0955343182f },
        { -0.0055258826f, 0.0040252103f, 1.0015006723f },
    };
    static const enum CRAWColorSpace d65_rgb[] = {
        CRAW_CS_BT709_LINEAR, CRAW_CS_BT2020_LINEAR,
        CRAW_CS_P3D65_LINEAR, CRAW_CS_V_GAMUT_LINEAR,
    };
    const double wx = 0.3127, wy = 0.3290;
    const double white[3] = { wx / wy, 1.0, (1.0 - wx - wy) / wy };
    float m[3][3], n[3][3];
    double worst;
    int ret;

    worst = 0.0;
    ret = craw_colorspace_from_xyz_d65(CRAW_CS_ACES_AP0_LINEAR, m);
    CHECK(ret == 0, "AP0 matrix failed: %d", ret);
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            CHECK_NEAR(m[r][c], ap0[r][c], 1e-3, "XYZ to AP0 [%d][%d]", r, c);
            worst = fmax(worst, fabs(m[r][c] - ap0[r][c]));
        }
    }
    printf("  XYZ(D65) to AP0: max error %.3e against DESIGN.md 8\n", worst);

    worst = 0.0;
    ret = craw_colorspace_from_xyz_d65(CRAW_CS_ACES_AP1_LINEAR, m);
    CHECK(ret == 0, "AP1 matrix failed: %d", ret);
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            CHECK_NEAR(m[r][c], ap1[r][c], 1e-3, "XYZ to AP1 [%d][%d]", r, c);
            worst = fmax(worst, fabs(m[r][c] - ap1[r][c]));
        }
    }
    printf("  XYZ(D65) to AP1: max error %.3e against DESIGN.md 8\n", worst);

    /* Every space's two matrices must undo each other. */
    for (int cs = 0; cs < CRAW_CS_NB; cs++) {
        double id_err = 0.0;

        ret = craw_colorspace_from_xyz_d65((enum CRAWColorSpace)cs, m);
        CHECK(ret == 0, "from_xyz %d failed: %d", cs, ret);
        ret = craw_colorspace_to_xyz_d65((enum CRAWColorSpace)cs, n);
        CHECK(ret == 0, "to_xyz %d failed: %d", cs, ret);

        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                double o = (double)m[r][0] * n[0][c] + (double)m[r][1] * n[1][c]
                         + (double)m[r][2] * n[2][c];
                CHECK_NEAR(o, r == c ? 1.0 : 0.0, 1e-5,
                           "space %d round trip [%d][%d]", cs, r, c);
                id_err = fmax(id_err, fabs(o - (r == c ? 1.0 : 0.0)));
            }
        }
        printf("  space %d round trip: max error %.3e\n", cs, id_err);
    }

    /* BT.709 through XYZ into AP1, against the published composed matrix,
     * and the white of one landing on the white of the other. */
    {
        double comp[3][3], sum[3] = { 0.0, 0.0, 0.0 };

        ret = craw_colorspace_to_xyz_d65(CRAW_CS_BT709_LINEAR, n);
        CHECK(ret == 0, "BT.709 to XYZ failed: %d", ret);
        ret = craw_colorspace_from_xyz_d65(CRAW_CS_ACES_AP1_LINEAR, m);
        CHECK(ret == 0, "XYZ to AP1 failed: %d", ret);

        worst = 0.0;
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                comp[r][c] = (double)m[r][0] * n[0][c] + (double)m[r][1] * n[1][c]
                           + (double)m[r][2] * n[2][c];
                CHECK_NEAR(comp[r][c], srgb_to_ap1[r][c], 1e-3,
                           "sRGB to ACEScg [%d][%d]", r, c);
                worst = fmax(worst, fabs(comp[r][c] - srgb_to_ap1[r][c]));
                sum[r] += comp[r][c];
            }
        }
        printf("  sRGB to ACEScg: max error %.3e against the published matrix\n", worst);

        /* sRGB white is (1, 1, 1), so the row sums are what it becomes. */
        worst = 0.0;
        for (int r = 0; r < 3; r++) {
            CHECK_NEAR(sum[r], 1.0, 1e-4, "sRGB white in ACEScg, channel %d", r);
            worst = fmax(worst, fabs(sum[r] - 1.0));
        }
        printf("  sRGB white in ACEScg: max error %.3e from (1, 1, 1)\n", worst);
    }

    /* AP1 to AP0, where the two adaptations have to cancel. */
    {
        ret = craw_colorspace_to_xyz_d65(CRAW_CS_ACES_AP1_LINEAR, n);
        CHECK(ret == 0, "AP1 to XYZ failed: %d", ret);
        ret = craw_colorspace_from_xyz_d65(CRAW_CS_ACES_AP0_LINEAR, m);
        CHECK(ret == 0, "XYZ to AP0 failed: %d", ret);

        worst = 0.0;
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                double o = (double)m[r][0] * n[0][c] + (double)m[r][1] * n[1][c]
                         + (double)m[r][2] * n[2][c];
                CHECK_NEAR(o, ap1_to_ap0[r][c], 1e-4, "ACEScg to ACES2065-1 [%d][%d]",
                           r, c);
                worst = fmax(worst, fabs(o - ap1_to_ap0[r][c]));
            }
        }
        printf("  ACEScg to ACES2065-1: max error %.3e against the published matrix\n",
               worst);
    }

    /* D65 at unit luminance is white in every D65 space. */
    for (size_t i = 0; i < CRAW_ARRAY_ELEMS(d65_rgb); i++) {
        ret = craw_colorspace_from_xyz_d65(d65_rgb[i], m);
        CHECK(ret == 0, "colour space %d failed: %d", (int)d65_rgb[i], ret);

        for (int r = 0; r < 3; r++) {
            double o = m[r][0] * white[0] + m[r][1] * white[1]
                     + m[r][2] * white[2];
            CHECK_NEAR(o, 1.0, 1e-4, "D65 white in space %d, channel %d",
                       (int)d65_rgb[i], r);
        }
    }

    /* The ACES spaces are adapted, so D65 lands on their own white, which
     * is equal energy in RGB terms too. */
    for (int i = 0; i < 2; i++) {
        enum CRAWColorSpace cs = i ? CRAW_CS_ACES_AP1_LINEAR
                                   : CRAW_CS_ACES_AP0_LINEAR;

        ret = craw_colorspace_from_xyz_d65(cs, m);
        CHECK(ret == 0, "colour space %d failed: %d", (int)cs, ret);

        for (int r = 0; r < 3; r++) {
            double o = m[r][0] * white[0] + m[r][1] * white[1]
                     + m[r][2] * white[2];
            CHECK_NEAR(o, 1.0, 1e-3, "D65 white in ACES space %d, channel %d",
                       (int)cs, r);
        }
    }

    ret = craw_colorspace_from_xyz_d65(CRAW_CS_XYZ_D65, m);
    CHECK(ret == 0, "XYZ matrix failed: %d", ret);
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++)
            CHECK(m[r][c] == (r == c ? 1.0f : 0.0f),
                  "XYZ must be the identity");
    }

    ret = craw_colorspace_from_xyz_d65(CRAW_CS_NB, m);
    CHECK(ret == CRAW_ERROR(EINVAL), "an unknown space must fail, got %d", ret);

    ret = craw_colorspace_to_xyz_d65(CRAW_CS_NB, m);
    CHECK(ret == CRAW_ERROR(EINVAL), "an unknown space must fail, got %d", ret);
}

/* The locus against the one Planckian radiator every colour table carries,
 * CIE illuminant A at 2856 K, and the tabulated 5000 K and 6500 K points.
 * Krystek's approximation is good to about 1e-4 in uv - it measures 1.1e-4
 * at 5000 K - and y moves three times as far as v does, so the tolerance
 * is in xy what that is; a just noticeable difference is ten times it. */
static void craw_test_planckian(void)
{
    static const struct { float cct; double x, y; } pts[] = {
        { 2856.0f, 0.44757, 0.40745 },
        { 5000.0f, 0.34510, 0.35162 },
        { 6500.0f, 0.31352, 0.32363 },
    };
    float xy[2], lo[2], hi[2];
    int ret;

    for (int i = 0; i < (int)(sizeof(pts) / sizeof(pts[0])); i++) {
        ret = craw_planckian_xy(pts[i].cct, xy);
        CHECK(ret == 0, "craw_planckian_xy(%.0f) failed: %d", pts[i].cct, ret);
        CHECK_NEAR(xy[0], pts[i].x, 5e-4, "x at %.0f K", pts[i].cct);
        CHECK_NEAR(xy[1], pts[i].y, 5e-4, "y at %.0f K", pts[i].cct);
    }

    /* Clamped, not extrapolated. */
    craw_planckian_xy(15000.0f, lo);
    craw_planckian_xy(40000.0f, hi);
    CHECK(lo[0] == hi[0] && lo[1] == hi[1], "40000 K must clamp to 15000 K");

    ret = craw_planckian_xy(NAN, xy);
    CHECK(ret == CRAW_ERROR(EINVAL), "a NaN temperature must fail, got %d", ret);
}

/* The published D65 to D50 Bradford matrix (Lindbloom), the whites exact
 * both ways, and the identity for one white. */
static void craw_test_bradford(void)
{
    static const float d65[2] = { 0.312727f, 0.329023f };
    static const float d50[2] = { 0.345669f, 0.358496f };
    static const float want[3][3] = {
        {  1.0478112f, 0.0228866f, -0.0501270f },
        {  0.0295424f, 0.9904844f, -0.0170491f },
        { -0.0092345f, 0.0150436f,  0.7521316f },
    };
    float m[3][3], w[3], out[3];
    int ret;

    ret = craw_bradford_adaptation(d65, d50, m);
    CHECK(ret == 0, "craw_bradford_adaptation failed: %d", ret);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            CHECK_NEAR(m[i][j], want[i][j], 2e-4, "D65 to D50 [%d][%d]", i, j);

    w[0] = d65[0] / d65[1];
    w[1] = 1.0f;
    w[2] = (1.0f - d65[0] - d65[1]) / d65[1];
    for (int i = 0; i < 3; i++)
        out[i] = m[i][0] * w[0] + m[i][1] * w[1] + m[i][2] * w[2];
    CHECK_NEAR(out[0], d50[0] / d50[1], 1e-5, "D65 white to D50 X");
    CHECK_NEAR(out[1], 1.0, 1e-5, "D65 white to D50 Y");
    CHECK_NEAR(out[2], (1.0 - d50[0] - d50[1]) / d50[1], 1e-5, "D65 white to D50 Z");

    ret = craw_bradford_adaptation(d65, d65, m);
    CHECK(ret == 0, "craw_bradford_adaptation to itself failed: %d", ret);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            CHECK_NEAR(m[i][j], i == j ? 1.0 : 0.0, 1e-6, "identity [%d][%d]", i, j);
}

#define CHECK_REL(got, want, rel, ...) \
    CHECK_NEAR(got, want, (rel) * fabs((double)(want)), __VA_ARGS__)

/* The evaluation rules of DESIGN.md 14.2 on a made-up table: the knots
 * themselves, a power law between them, a ~ iso on both sides of the
 * table, b ~ iso^2 above it and b kept below it, continuity where the
 * rules meet, and the first knot when there is no exposure index. */
static void craw_test_noise_eval(void)
{
    static const CRAWNoiseKnot kn[] = {
        {  400.0f, 1.0e-5f, 4.0e-8f },
        { 1600.0f, 2.0e-5f, 1.0e-9f },
        { 6400.0f, 8.0e-5f, 1.6e-8f },
    };
    static const CRAWNoiseKnot zero_b[] = {
        { 100.0f, 1.0e-5f, 0.0f    },
        { 400.0f, 4.0e-5f, 4.0e-9f },
    };
    const uint32_t nb = CRAW_ARRAY_ELEMS(kn);
    CRAWNoiseModel m, m2;
    int ret;

    for (uint32_t i = 0; i < nb; i++) {
        ret = craw_noise_model_eval(kn, nb, kn[i].iso, &m);
        CHECK(ret == 0, "eval at knot %u failed: %d", i, ret);
        CHECK_REL(m.a, kn[i].a, 1e-5, "a at the knot at %.0f", kn[i].iso);
        CHECK_REL(m.b, kn[i].b, 1e-5, "b at the knot at %.0f", kn[i].iso);
    }

    /* Halfway in log(iso) is the geometric mean of both terms, and a
     * quarter of the way the power law's own value. */
    craw_noise_model_eval(kn, nb, 800.0f, &m);
    CHECK_REL(m.a, sqrt(1.0e-5 * 2.0e-5), 1e-5, "a halfway from 400 to 1600");
    CHECK_REL(m.b, sqrt(4.0e-8 * 1.0e-9), 1e-5, "b halfway from 400 to 1600");
    craw_noise_model_eval(kn, nb, 1600.0f * sqrt(2.0), &m);
    CHECK_REL(m.a, 2.0e-5 * pow(4.0, 0.25), 1e-5, "a a quarter of the way to 6400");
    CHECK_REL(m.b, 1.0e-9 * pow(16.0, 0.25), 1e-5, "b a quarter of the way to 6400");

    /* Above the table: a ~ iso, b ~ iso^2. */
    craw_noise_model_eval(kn, nb, 25600.0f, &m);
    CHECK_REL(m.a, 8.0e-5 * 4.0, 1e-5, "a four times past the last knot");
    CHECK_REL(m.b, 1.6e-8 * 16.0, 1e-5, "b four times past the last knot");

    /* Below it: a ~ iso, b kept. */
    craw_noise_model_eval(kn, nb, 100.0f, &m);
    CHECK_REL(m.a, 1.0e-5 * 0.25, 1e-5, "a a quarter of the first knot");
    CHECK_REL(m.b, 4.0e-8, 1e-5, "b a quarter of the first knot");

    /* No step where interpolation hands over to extrapolation. */
    for (int e = 0; e < 2; e++) {
        float iso = e ? kn[nb - 1].iso : kn[0].iso;

        craw_noise_model_eval(kn, nb, iso * 0.999f, &m);
        craw_noise_model_eval(kn, nb, iso * 1.001f, &m2);
        CHECK_REL(m2.a, m.a, 3e-3, "a across the knot at %.0f", iso);
        CHECK_REL(m2.b, m.b, 5e-3, "b across the knot at %.0f", iso);
    }

    /* Without an exposure index there is nothing to scale by. */
    {
        const float bad[] = { 0.0f, -100.0f, NAN, INFINITY };

        for (int i = 0; i < (int)CRAW_ARRAY_ELEMS(bad); i++) {
            ret = craw_noise_model_eval(kn, nb, bad[i], &m);
            CHECK(ret == 0 && m.a == kn[0].a && m.b == kn[0].b,
                  "iso %g must give the first knot", (double)bad[i]);
        }
    }

    /* A zero read term has no logarithm: linear to it inside the table,
     * zero below it. */
    craw_noise_model_eval(zero_b, 2, 200.0f, &m);
    CHECK_REL(m.a, 2.0e-5, 1e-5, "a with a zero b knot");
    CHECK_REL(m.b, 2.0e-9, 1e-5, "b interpolated linearly to a zero b");
    craw_noise_model_eval(zero_b, 2, 50.0f, &m);
    CHECK(m.b == 0.0f, "a zero b stays zero below the table, got %g", (double)m.b);

    /* A single knot extrapolates both ways. */
    craw_noise_model_eval(kn, 1, 800.0f, &m);
    CHECK_REL(m.a, 2.0e-5, 1e-5, "a from a one knot table");
    CHECK_REL(m.b, 1.6e-7, 1e-5, "b from a one knot table");

    ret = craw_noise_model_eval(kn, 0, 800.0f, &m);
    CHECK(ret == CRAW_ERROR(EINVAL), "an empty table must fail, got %d", ret);
}

/* The built-in S1M2 table of tools/profiles.c against the measurements of
 * DESIGN.md 14.2: sorted, a ~ 12e-9 * iso from 1000 up and b on
 * 6.1e-10 + 2.3e-16 * iso^2 there, the ISO 640 mode's step, the ISO 10000
 * clip's own fit, and a model that never falls as the index rises. */
static void craw_test_noise_s1m2(void)
{
    CRAWProfile p;
    CRAWNoiseModel m, prev = { 0.0f, 0.0f }, lo, hi;
    const CRAWNoiseKnot *kn;
    float prev_iso = 0.0f;
    uint32_t nb;
    int ret;

    craw_test_profile_init(&p);
    ret = craw_tool_profile_apply_builtin(&p, "s1m2");
    CHECK(ret == 0, "the s1m2 built-in failed: %d", ret);
    CHECK(craw_profile_validate(&p) == 0, "the s1m2 built-in must validate");

    kn = p.noise_knots;
    nb = p.nb_noise_knots;
    CHECK(nb >= 2 && nb <= CRAW_PROFILE_MAX_NOISE_KNOTS, "%u noise knots", nb);

    for (uint32_t i = 0; i < nb; i++) {
        CHECK(kn[i].a > 0.0f && kn[i].b > 0.0f, "knot %.0f has a zero term", kn[i].iso);
        if (i)
            CHECK(kn[i].iso > kn[i - 1].iso, "knots out of order at %.0f", kn[i].iso);
        if (kn[i].iso < 1000.0f)
            continue;

        CHECK_REL(kn[i].a / kn[i].iso, 12e-9, 0.12, "a / iso at %.0f", kn[i].iso);
        CHECK(kn[i].b > (6.1e-10 + 2.3e-16 * kn[i].iso * kn[i].iso) / 1.5 &&
              kn[i].b < (6.1e-10 + 2.3e-16 * kn[i].iso * kn[i].iso) * 1.5,
              "b at %.0f is %g, off the law", kn[i].iso, (double)kn[i].b);
    }

    /* ISO 640 is the other gain mode: the a of 1000, twenty times the b. */
    craw_noise_model_eval(kn, nb, 640.0f, &lo);
    craw_noise_model_eval(kn, nb, 1000.0f, &hi);
    CHECK_REL(lo.a, hi.a, 0.1, "a at 640 against 1000");
    CHECK(lo.b > 10.0f * hi.b, "b at 640 %g against 1000 %g", (double)lo.b, (double)hi.b);

    /* P1094353's own fit, which DESIGN.md 14.7 and 20.6 measured with. */
    p.as_shot_ei = 10000.0f;
    ret = craw_profile_noise_eval(&p, 0.0f, &m);
    CHECK(ret == 0, "the profile eval at 10000 failed: %d", ret);
    CHECK_REL(m.a, 1.19e-4, 0.05, "a at ISO 10000");
    CHECK_REL(m.b, 2.95e-8, 0.15, "b at ISO 10000");
    CHECK(m.a > 8.0f * hi.a, "a at 10000 must be the gain's, not a clamp's");

    /* Past the top, the law: twice the index, twice a, four times b. */
    craw_noise_model_eval(kn, nb, kn[nb - 1].iso, &lo);
    craw_noise_model_eval(kn, nb, 2.0f * kn[nb - 1].iso, &hi);
    CHECK_REL(hi.a, 2.0 * lo.a, 1e-5, "a at twice the last knot");
    CHECK_REL(hi.b, 4.0 * lo.b, 1e-5, "b at twice the last knot");

    /* Monotonic from 100 to 102400: a never falls, nor does b from 1000,
     * where the one gain mode starts. */
    for (int s = 0; s <= 100; s++) {
        float iso = 100.0f * powf(2.0f, 10.0f * (float)s / 100.0f);

        craw_noise_model_eval(kn, nb, iso, &m);
        CHECK(m.a > 0.0f && isfinite(m.a) && m.b > 0.0f && isfinite(m.b),
              "model at %.0f: %g %g", iso, (double)m.a, (double)m.b);
        if (s) {
            CHECK(m.a >= prev.a, "a falls at %.0f: %g after %g", iso,
                  (double)m.a, (double)prev.a);
            if (prev_iso >= 1000.0f)
                CHECK(m.b >= prev.b, "b falls at %.0f: %g after %g", iso,
                      (double)m.b, (double)prev.b);
        }
        prev = m;
        prev_iso = iso;
    }
}

#if CRAW_TOOL_HAVE_LAVU

/* A decoded frame as far as craw_tool_profile_from_frame() looks at one: a
 * pixel format, the colour record and the Exif keys. No pixels. */
static AVFrame *craw_test_raw_frame(enum AVPixelFormat fmt,
                                    enum AVRawColorParamsType type,
                                    AVRawColorParams **rc)
{
    AVFrame *f = av_frame_alloc();

    if (!f)
        return NULL;

    f->format = fmt;
    f->width = 6008;
    f->height = 4008;

    *rc = av_raw_color_params_create_side_data(f);
    if (!*rc) {
        av_frame_free(&f);
        return NULL;
    }
    (*rc)->type = type;

    return f;
}

/* The ProRes RAW branch is the frame header's record copied over, as both
 * tools did on their own before they shared the builder. */
static void craw_test_frame_prores_raw(void)
{
    AVRawColorParams *rc = NULL;
    AVFrame *f;
    CRAWProfile p;
    int ret;

    f = craw_test_raw_frame(AV_PIX_FMT_BAYER_RGGB16LE, AV_RAW_COLOR_PARAMS_PRORES_RAW, &rc);
    CHECK(f, "cannot allocate a frame");
    if (!f)
        return;

    rc->black_level = av_make_q(256, 65535);
    rc->white_level = av_make_q(63743, 65535);
    rc->wb_cct = 5500;
    rc->codec.prores_raw.wb_red = av_d2q(2.02539, 100000);
    rc->codec.prores_raw.wb_blue = av_d2q(1.66797, 100000);
    rc->codec.prores_raw.gain = av_d2q(23.04, 100000);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            rc->codec.prores_raw.color_matrix[i][j] =
                av_d2q(craw_test_frame_matrix[i][j], 1000000);

    craw_profile_init(&p);
    craw_tool_profile_apply_builtin(&p, "s1m2");
    ret = craw_tool_profile_from_frame(&p, f, false);
    CHECK(ret == 0, "a ProRes RAW frame failed: %d", ret);
    CHECK(p.cfa == CRAW_CFA_RGGB, "ProRes RAW cfa %d", (int)p.cfa);
    CHECK_NEAR(p.black_level[3], 256.0 / 65535.0, 1e-9, "ProRes RAW black");
    CHECK_NEAR(p.white_level[0], 63743.0 / 65535.0, 1e-9, "ProRes RAW white");
    CHECK_NEAR(p.as_shot_wb[0], 2.02539, 1e-6, "ProRes RAW R gain");
    CHECK_NEAR(p.as_shot_wb[2], 1.66797, 1e-6, "ProRes RAW B gain");
    CHECK_NEAR(p.gain, 23.04, 1e-6, "ProRes RAW gain");
    CHECK_NEAR(p.as_shot_cct, 5500.0, 0.0, "ProRes RAW cct");
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            CHECK_NEAR(p.as_shot_matrix[i][j], craw_test_frame_matrix[i][j], 1e-6,
                       "ProRes RAW matrix [%d][%d]", i, j);

    av_frame_free(&f);
}

#if CRAW_TOOL_HAVE_RW2

/* An RW2 still the way the rw2 decoder exports one: 14-bit levels in the
 * 16-bit sample range, the as-shot gains and nothing else, and the ISO in
 * the Exif, space padded as the decoder prints it. */
static AVFrame *craw_test_rw2_frame(enum AVPixelFormat fmt, double wb_r, double wb_b,
                                    const char *iso)
{
    AVRawColorParams *rc = NULL;
    AVFrame *f = craw_test_raw_frame(fmt, AV_RAW_COLOR_PARAMS_RW2, &rc);

    if (!f)
        return NULL;

    rc->black_level = av_make_q(512, 65535);
    rc->white_level = av_make_q(16383, 65535);
    rc->codec.rw2.wb_red = av_d2q(wb_r, 100000);
    rc->codec.rw2.wb_blue = av_d2q(wb_b, 100000);

    av_dict_set(&f->metadata, "Make", "Panasonic", 0);
    av_dict_set(&f->metadata, "Model", "DC-S1M2", 0);
    if (iso)
        av_dict_set(&f->metadata, "ExifIFD/ISOSpeedRatings", iso, 0);

    return f;
}

static void craw_test_frame_rw2(void)
{
    static const struct {
        enum AVPixelFormat fmt;
        enum CRAWCfa cfa;
    } phases[] = {
        { AV_PIX_FMT_BAYER_RGGB16LE, CRAW_CFA_RGGB },
        { AV_PIX_FMT_BAYER_GRBG16LE, CRAW_CFA_GRBG },
        { AV_PIX_FMT_BAYER_GBRG16LE, CRAW_CFA_GBRG },
        { AV_PIX_FMT_BAYER_BGGR16LE, CRAW_CFA_BGGR },
    };
    /* The 5500 K knot's pair, which must land on 5500 K exactly, and a
     * balance between knots with a tint on it, 4200 K and +0.02. */
    const double knot_r = 2.0253, knot_b = 1.6676;
    double r42, b42;
    CRAWProfile p, tab;
    CRAWIlluminant il, want;
    CRAWColorParams cp;
    AVFrame *f;
    float rg, bg;
    int ret;

    craw_profile_init(&tab);
    ret = craw_tool_profile_apply_builtin(&tab, "s1m2");
    CHECK(ret == 0, "the s1m2 built-in failed: %d", ret);

    il = (CRAWIlluminant) { .cct = 4200.0f, .tint = 0.02f };
    ret = craw_profile_illuminant_to_neutral(&tab, &il, &rg, &bg);
    CHECK(ret == 0, "illuminant_to_neutral failed: %d", ret);
    r42 = 1.0 / rg;
    b42 = 1.0 / bg;

    /* The phase is the pixel format's, whichever of the four it is. */
    for (size_t i = 0; i < CRAW_ARRAY_ELEMS(phases); i++) {
        f = craw_test_rw2_frame(phases[i].fmt, knot_r, knot_b, "    640");
        CHECK(f, "cannot allocate a frame");
        if (!f)
            return;
        p = tab;
        ret = craw_tool_profile_from_frame(&p, f, false);
        CHECK(ret == 0, "an RW2 frame failed: %d", ret);
        CHECK(p.cfa == phases[i].cfa, "%s: cfa %d, want %d",
              av_get_pix_fmt_name(phases[i].fmt), (int)p.cfa, (int)phases[i].cfa);
        av_frame_free(&f);
    }

    /* Levels, gains, the colour temperature and the matrix at the knot. */
    f = craw_test_rw2_frame(AV_PIX_FMT_BAYER_RGGB16LE, knot_r, knot_b, "    640");
    p = tab;
    ret = craw_tool_profile_from_frame(&p, f, false);
    CHECK(ret == 0, "an RW2 frame failed: %d", ret);
    for (int i = 0; i < 4; i++) {
        CHECK_NEAR(p.black_level[i], 512.0 / 65535.0, 1e-9, "RW2 black, site %d", i);
        CHECK_NEAR(p.white_level[i], 16383.0 / 65535.0, 1e-9, "RW2 white, site %d", i);
    }
    CHECK_NEAR(p.as_shot_wb[0], knot_r, 1e-6, "RW2 R gain");
    CHECK_NEAR(p.as_shot_wb[1], 1.0, 0.0, "RW2 G gain");
    CHECK_NEAR(p.as_shot_wb[2], knot_b, 1e-6, "RW2 B gain");
    CHECK_NEAR(p.as_shot_cct, 5500.0, 1e-2, "RW2 cct at the 5500 K knot's gains");
    CHECK(p.nb_wb_knots == tab.nb_wb_knots && p.nb_matrix_knots == tab.nb_matrix_knots &&
          p.nb_noise_knots == tab.nb_noise_knots && !strcmp(p.name, tab.name),
          "the camera's tables must be left as they were");
    av_frame_free(&f);

    /* Between knots and off the locus: the temperature is the one the
     * locus mapping gives for the reciprocal gains, and the matrix the
     * table's at that point. */
    f = craw_test_rw2_frame(AV_PIX_FMT_BAYER_RGGB16LE, r42, b42, "    100");
    p = tab;
    ret = craw_tool_profile_from_frame(&p, f, false);
    CHECK(ret == 0, "an RW2 frame failed: %d", ret);
    ret = craw_profile_neutral_to_illuminant(&tab, 1.0f / p.as_shot_wb[0],
                                             1.0f / p.as_shot_wb[2], &want);
    CHECK(ret == 0, "neutral_to_illuminant failed: %d", ret);
    CHECK_NEAR(p.as_shot_cct, want.cct, 0.0, "RW2 cct between knots");
    CHECK_NEAR(want.cct, 4200.0, 0.05, "the 4200 K balance's temperature");
    CHECK_NEAR(want.tint, 0.02, 1e-4, "the 4200 K balance's tint");
    ret = craw_profile_eval(&tab, CRAW_WB_REBALANCE, &want, &cp);
    CHECK(ret == 0, "eval failed: %d", ret);
    CHECK(!memcmp(p.as_shot_matrix, cp.cam_to_xyz, sizeof(cp.cam_to_xyz)),
          "the RW2 matrix must be the table's at its temperature");
    av_frame_free(&f);

    /* The gain mode by ISO: the low mode's factor at 640 and below, the
     * high one's from 1000, each over eight because a still is rated three
     * stops closer to the clip than a V-Log recording (DESIGN.md 9c), and
     * the ISO the as-shot index, so that a working index re-rates a still
     * as it does a clip. */
    {
        static const struct {
            const char *iso;
            float want_iso, want_gain;
        } modes[] = {
            { "    100",   100.0f, 14.51f / 8.0f },
            { "    640",   640.0f, 14.51f / 8.0f },
            { "    800",   800.0f, 14.51f / 8.0f },
            { "   1000",  1000.0f, 23.04f / 8.0f },
            { "  12800", 12800.0f, 23.04f / 8.0f },
            { NULL,          0.0f, 14.51f / 8.0f },
        };

        for (size_t i = 0; i < CRAW_ARRAY_ELEMS(modes); i++) {
            f = craw_test_rw2_frame(AV_PIX_FMT_BAYER_RGGB16LE, knot_r, knot_b,
                                    modes[i].iso);
            p = tab;
            ret = craw_tool_profile_from_frame(&p, f, false);
            CHECK(ret == 0, "an RW2 frame failed: %d", ret);
            CHECK_NEAR(p.as_shot_ei, modes[i].want_iso, 0.0, "as-shot ISO of `%s`",
                       modes[i].iso ? modes[i].iso : "(none)");
            CHECK_NEAR(p.gain, modes[i].want_gain, 1e-6, "gain at ISO `%s`",
                       modes[i].iso ? modes[i].iso : "(none)");
            av_frame_free(&f);
        }

        f = craw_test_rw2_frame(AV_PIX_FMT_BAYER_RGGB16LE, knot_r, knot_b, "    640");
        p = tab;
        craw_tool_profile_from_frame(&p, f, false);
        p.exposure_index = 1000.0f;
        ret = craw_profile_eval(&p, CRAW_WB_AS_SHOT, NULL, &cp);
        CHECK(ret == 0, "eval failed: %d", ret);
        CHECK_NEAR(cp.gain, 14.51 / 8.0 * 1000.0 / 640.0, 1e-4, "an ISO 640 still rated at 1000");
        av_frame_free(&f);
    }

    /* No tables, no matrix: refused, never an identity. */
    f = craw_test_rw2_frame(AV_PIX_FMT_BAYER_RGGB16LE, knot_r, knot_b, "    640");
    craw_profile_init(&p);
    printf("(an error message about missing tables is expected here)\n");
    fflush(stdout);
    ret = craw_tool_profile_from_frame(&p, f, false);
    CHECK(ret == CRAW_ERROR(ENOENT), "an RW2 without tables must be refused: %d", ret);
    av_frame_free(&f);

    /* Not a mosaic at all. */
    f = craw_test_rw2_frame(AV_PIX_FMT_RGB48LE, knot_r, knot_b, "    640");
    p = tab;
    ret = craw_tool_profile_from_frame(&p, f, false);
    CHECK(ret == CRAW_ERROR(ENOTSUP), "an RGB frame must be refused: %d", ret);
    av_frame_free(&f);
}

#endif /* CRAW_TOOL_HAVE_RW2 */

#endif /* CRAW_TOOL_HAVE_LAVU */

int main(void)
{
    craw_test_validate();
    craw_test_eval();
    craw_test_locus();
    craw_test_colorspaces();
    craw_test_planckian();
    craw_test_bradford();
    craw_test_noise_eval();
    craw_test_noise_s1m2();
#if CRAW_TOOL_HAVE_LAVU
    craw_test_frame_prores_raw();
#if CRAW_TOOL_HAVE_RW2
    craw_test_frame_rw2();
#endif
#endif

    if (failures) {
        printf("test_profile: %d failure(s)\n", failures);
        return 1;
    }

    printf("test_profile: all checks passed\n");
    return 0;
}
