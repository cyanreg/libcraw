/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Profile evaluation, locus mapping and output colour space matrices,
 * against the values in DESIGN.md sections 8 and 9. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <craw/profile.h>

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

int main(void)
{
    craw_test_validate();
    craw_test_eval();
    craw_test_locus();
    craw_test_colorspaces();

    if (failures) {
        printf("test_profile: %d failure(s)\n", failures);
        return 1;
    }

    printf("test_profile: all checks passed\n");
    return 0;
}
