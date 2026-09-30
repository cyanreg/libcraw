/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* The window's colour choice in tools/gui/colour_map.c: for what a
 * compositor can do and what it prefers, what libplacebo is asked to render
 * and which parametric description the surface is tagged with. CPU only;
 * the protocol itself is exercised by running cineview under a compositor
 * (DESIGN.md 22). Every choice is also checked for being something the
 * compositor cannot answer with a protocol error. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <craw/utils.h>

#include "colour.h"

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

#define BIT(i) (1ull << (i))

/* What sway 1.13 with the Vulkan renderer advertises: version 2,
 * parametric and mastering primaries only, four curves, two sets of
 * primaries and the one intent every compositor must have. */
static struct view_colour_caps sway_caps(void)
{
    return (struct view_colour_caps) {
        .version = 2,
        .parametric = true,
        .mastering_display_primaries = true,
        .tf_named = BIT(VIEW_WP_TF_GAMMA22) | BIT(VIEW_WP_TF_ST2084_PQ) |
                    BIT(VIEW_WP_TF_EXT_LINEAR) | BIT(VIEW_WP_TF_BT1886),
        .primaries_named = BIT(VIEW_WP_PRIM_SRGB) | BIT(VIEW_WP_PRIM_BT2020),
        .intents = 1u << VIEW_WP_INTENT_PERCEPTUAL,
    };
}

/* A preferred description the way a compositor's info events deliver it:
 * the xy always, the name when there is one. */
static struct view_colour_desc pref(enum pl_color_primaries prim, int wp_prim,
                                    int tf, float min, float max, float ref,
                                    enum pl_color_primaries target,
                                    float tmin, float tmax)
{
    return (struct view_colour_desc) {
        .primaries       = *pl_raw_primaries_get(prim),
        .primaries_named = wp_prim,
        .tf_named        = tf,
        .min_lum         = min,
        .max_lum         = max,
        .ref_lum         = ref,
        .target          = *pl_raw_primaries_get(target),
        .target_min      = tmin,
        .target_max      = tmax,
    };
}

static bool prim_eq(const struct pl_raw_primaries *a, enum pl_color_primaries b)
{
    return pl_raw_primaries_similar(a, pl_raw_primaries_get(b));
}

static bool xy_near(struct pl_cie_xy a, struct pl_cie_xy b, float tol)
{
    return fabsf(a.x - b.x) <= tol && fabsf(a.y - b.y) <= tol;
}

/* Display P3 as it can be sent inside BT.2020: its red lies on the
 * spectral locus a hair outside BT.2020's red-green edge (0.0017 in y) and
 * is clipped onto it, and all three are pulled 1e-4 of the way to white. */
static bool is_p3_in_2020(const struct pl_raw_primaries *a)
{
    const struct pl_raw_primaries *p3 = pl_raw_primaries_get(PL_COLOR_PRIM_DISPLAY_P3);

    return xy_near(a->green, p3->green, 1e-4f) && xy_near(a->blue, p3->blue, 1e-4f) &&
           xy_near(a->red, p3->red, 2e-3f) && xy_near(a->white, p3->white, 1e-6f) &&
           pl_primaries_superset(pl_raw_primaries_get(PL_COLOR_PRIM_BT_2020), a);
}

/* Nothing in a choice may be something the protocol answers with an error:
 * a name the compositor did not advertise, a request behind a feature it
 * lacks, a luminance pair upside down once it is the integers on the
 * wire, a target volume outside the container, an intent it does not do. */
static void legal(const char *name, const struct view_colour_caps *caps,
                  const struct view_colour_choice *c)
{
    const struct pl_raw_primaries *container = pl_raw_primaries_get(c->target.primaries);

    CHECK(caps->primaries_named & BIT(c->primaries_named),
          "%s: primaries %d not advertised", name, c->primaries_named);
    if (c->tf_named)
        CHECK(caps->tf_named & BIT(c->tf_named),
              "%s: tf %d not advertised", name, c->tf_named);
    else
        CHECK(caps->set_tf_power && c->tf_power >= 1.0f && c->tf_power <= 10.0f,
              "%s: power %g without set_tf_power or out of range", name, c->tf_power);
    CHECK(!c->set_luminances || caps->set_luminances,
          "%s: set_luminances without the feature", name);
    if (c->set_luminances) {
        CHECK(roundf(c->max_lum) * 10000.0f > roundf(c->min_lum * 10000.0f),
              "%s: max_lum %g not above min_lum %g", name, c->max_lum, c->min_lum);
        CHECK(roundf(c->ref_lum) * 10000.0f > roundf(c->min_lum * 10000.0f),
              "%s: ref_lum %g not above min_lum %g", name, c->ref_lum, c->min_lum);
    }
    CHECK(!c->set_mastering || caps->mastering_display_primaries,
          "%s: mastering without the feature", name);
    if (c->set_mastering) {
        CHECK(pl_primaries_superset(container, &c->mastering),
              "%s: mastering primaries outside the container", name);
        CHECK(pl_raw_primaries_equal(&c->mastering, &c->target.hdr.prim),
              "%s: mastering primaries are not what libplacebo maps into", name);
        if (c->mastering_max > 0.0f)
            CHECK(roundf(c->mastering_max) * 10000.0f >
                  roundf(c->mastering_min * 10000.0f),
                  "%s: mastering max %g not above min %g", name,
                  c->mastering_max, c->mastering_min);
    }
    CHECK(caps->intents & (1u << c->intent), "%s: intent %d not advertised",
          name, c->intent);
    CHECK(c->depth == (pl_color_transfer_is_hdr(c->target.transfer) ? 10 : 8),
          "%s: depth %d", name, c->depth);
    CHECK(c->exact == (c->candidate == 1), "%s: exact %d but candidate %d",
          name, c->exact, c->candidate);
}

static int resolve(const char *name, const struct view_colour_caps *caps,
                   const struct view_colour_desc *d, struct view_colour_choice *c)
{
    int ret = view_colour_resolve(caps, d, c);

    if (ret >= 0)
        legal(name, caps, c);

    return ret;
}

/* 1. sway's SDR output: the preferred is exactly what can be rendered and
 * tagged, and nothing beyond the two names needs sending. */
static void test_sway_sdr(void)
{
    struct view_colour_caps caps = sway_caps();
    struct view_colour_desc d = pref(PL_COLOR_PRIM_BT_709, VIEW_WP_PRIM_SRGB,
                                     VIEW_WP_TF_GAMMA22, 0.2f, 80.0f, 80.0f,
                                     PL_COLOR_PRIM_BT_709, 0.2f, 80.0f);
    struct view_colour_choice c;

    CHECK(resolve(__func__, &caps, &d, &c) == 0, "refused");
    CHECK(c.target.primaries == PL_COLOR_PRIM_BT_709, "primaries %d", c.target.primaries);
    CHECK(c.target.transfer == PL_COLOR_TRC_GAMMA22, "transfer %d", c.target.transfer);
    CHECK_NEAR(c.target.hdr.max_luma, 203.0, 1e-4, "max_luma");
    CHECK_NEAR(c.target.hdr.min_luma, 203.0 * 0.2 / 80.0, 1e-5, "min_luma");
    CHECK(prim_eq(&c.target.hdr.prim, PL_COLOR_PRIM_BT_709), "hdr.prim is not BT.709");
    CHECK(c.depth == 8, "depth %d", c.depth);
    CHECK(c.primaries_named == VIEW_WP_PRIM_SRGB, "primaries_named %d", c.primaries_named);
    CHECK(c.tf_named == VIEW_WP_TF_GAMMA22, "tf_named %d", c.tf_named);
    CHECK(!c.set_luminances, "set_luminances without the feature");
    CHECK(!c.set_mastering, "mastering sent for SDR in the container's own gamut");
    CHECK(c.intent == VIEW_WP_INTENT_PERCEPTUAL, "intent %d", c.intent);
    CHECK(c.exact, "not exact");
}

/* 2. sway's HDR output: BT.2020 PQ with the display's target volume, which
 * is what libplacebo tone maps to and what the mastering request says. */
static void test_sway_hdr(void)
{
    struct view_colour_caps caps = sway_caps();
    struct view_colour_desc d = pref(PL_COLOR_PRIM_BT_2020, VIEW_WP_PRIM_BT2020,
                                     VIEW_WP_TF_ST2084_PQ, 0.005f, 10000.0f, 203.0f,
                                     PL_COLOR_PRIM_BT_2020, 0.005f, 1000.0f);
    struct view_colour_choice c;

    CHECK(resolve(__func__, &caps, &d, &c) == 0, "refused");
    CHECK(c.target.primaries == PL_COLOR_PRIM_BT_2020, "primaries %d", c.target.primaries);
    CHECK(c.target.transfer == PL_COLOR_TRC_PQ, "transfer %d", c.target.transfer);
    CHECK_NEAR(c.target.hdr.min_luma, 0.005, 1e-7, "min_luma");
    CHECK_NEAR(c.target.hdr.max_luma, 1000.0, 1e-3, "max_luma");
    CHECK(prim_eq(&c.target.hdr.prim, PL_COLOR_PRIM_BT_2020), "hdr.prim is not BT.2020");
    CHECK(c.depth == 10, "depth %d", c.depth);
    CHECK(!c.set_luminances, "set_luminances without the feature");
    CHECK(c.set_mastering, "no mastering for HDR");
    CHECK(prim_eq(&c.mastering, PL_COLOR_PRIM_BT_2020), "mastering primaries");
    CHECK_NEAR(c.mastering_min, 0.005, 1e-7, "mastering_min");
    CHECK_NEAR(c.mastering_max, 1000.0, 1e-3, "mastering_max");
    CHECK(!c.peak_assumed, "peak assumed");
    CHECK(c.exact, "not exact");
}

/* 3. A target gamut inside BT.2020 (a P3 panel), a compositor with
 * set_luminances and the relative intent: libplacebo gamut maps into the
 * panel, the compositor is told so, and the intent is relative. */
static void test_hdr_p3_target(void)
{
    struct view_colour_caps caps = sway_caps();
    struct view_colour_desc d = pref(PL_COLOR_PRIM_BT_2020, VIEW_WP_PRIM_BT2020,
                                     VIEW_WP_TF_ST2084_PQ, 0.005f, 10000.0f, 203.0f,
                                     PL_COLOR_PRIM_DISPLAY_P3, 0.005f, 800.0f);
    struct view_colour_choice c;

    caps.set_luminances = true;
    caps.intents |= 1u << VIEW_WP_INTENT_RELATIVE;

    CHECK(resolve(__func__, &caps, &d, &c) == 0, "refused");
    CHECK(c.target.primaries == PL_COLOR_PRIM_BT_2020, "primaries %d", c.target.primaries);
    CHECK(is_p3_in_2020(&c.target.hdr.prim), "hdr.prim is not P3");
    CHECK(c.set_mastering && is_p3_in_2020(&c.mastering),
          "mastering primaries are not P3");
    /* Clipped, and not for nothing. */
    CHECK(!pl_primaries_superset(pl_raw_primaries_get(PL_COLOR_PRIM_BT_2020),
                                 pl_raw_primaries_get(PL_COLOR_PRIM_DISPLAY_P3)),
          "P3 is inside BT.2020 after all");
    CHECK_NEAR(c.mastering_max, 800.0, 1e-3, "mastering_max");
    CHECK_NEAR(c.target.hdr.max_luma, 800.0, 1e-3, "max_luma");
    /* PQ's minimum is an offset under the whole curve in the protocol, and
     * libplacebo's PQ is absolute: the smallest step above zero. */
    CHECK(c.set_luminances, "no luminances with the feature");
    CHECK(c.min_lum > 0.0f && c.min_lum <= 0.0001f, "min_lum %g", c.min_lum);
    CHECK(c.max_lum > c.min_lum, "max_lum %g", c.max_lum);
    CHECK_NEAR(c.ref_lum, 203.0, 1e-4, "ref_lum");
    CHECK(c.intent == VIEW_WP_INTENT_RELATIVE, "intent %d", c.intent);
    CHECK(c.exact, "not exact");
}

/* 4. A target peak of 10000 is PQ's reach, not a display's. */
static void test_hdr_no_peak(void)
{
    struct view_colour_caps caps = sway_caps();
    struct view_colour_desc d = pref(PL_COLOR_PRIM_BT_2020, VIEW_WP_PRIM_BT2020,
                                     VIEW_WP_TF_ST2084_PQ, 0.005f, 10000.0f, 203.0f,
                                     PL_COLOR_PRIM_BT_2020, 0.005f, 10000.0f);
    struct view_colour_choice c;

    CHECK(resolve(__func__, &caps, &d, &c) == 0, "refused");
    CHECK_NEAR(c.target.hdr.max_luma, 1000.0, 1e-3, "max_luma");
    CHECK_NEAR(c.mastering_max, 1000.0, 1e-3, "mastering_max");
    CHECK(c.peak_assumed, "peak not flagged as assumed");

    /* And a range upside down is no better. */
    d.target_max = 0.001f;
    CHECK(resolve(__func__, &caps, &d, &c) == 0, "refused");
    CHECK_NEAR(c.target.hdr.max_luma, 1000.0, 1e-3, "max_luma, upside down");
    CHECK(c.peak_assumed, "peak not flagged as assumed, upside down");
}

/* 5. An HDR preference in extended linear, which a UNORM image cannot
 * carry: BT.2020 PQ instead, into the compositor's sRGB target gamut. */
static void test_ext_linear_hdr(void)
{
    struct view_colour_caps caps = sway_caps();
    struct view_colour_desc d = pref(PL_COLOR_PRIM_BT_709, VIEW_WP_PRIM_SRGB,
                                     VIEW_WP_TF_EXT_LINEAR, 0.0f, 1000.0f, 203.0f,
                                     PL_COLOR_PRIM_BT_709, 0.0f, 1000.0f);
    struct view_colour_choice c;

    CHECK(resolve(__func__, &caps, &d, &c) == 0, "refused");
    CHECK(c.candidate == 2, "candidate %d", c.candidate);
    CHECK(!c.exact, "exact");
    CHECK(c.why && strstr(c.why, "cannot be rendered"), "why: %s", c.why ? c.why : "(null)");
    CHECK(c.target.primaries == PL_COLOR_PRIM_BT_2020 &&
          c.target.transfer == PL_COLOR_TRC_PQ, "not BT.2020 PQ");
    CHECK(c.primaries_named == VIEW_WP_PRIM_BT2020 &&
          c.tf_named == VIEW_WP_TF_ST2084_PQ, "not tagged BT.2020 PQ");
    CHECK(prim_eq(&c.target.hdr.prim, PL_COLOR_PRIM_BT_709), "gamut is not the sRGB target");
    CHECK(c.set_mastering && prim_eq(&c.mastering, PL_COLOR_PRIM_BT_709),
          "mastering is not the sRGB target");
    CHECK_NEAR(c.target.hdr.min_luma, PL_COLOR_HDR_BLACK, 1e-9, "min_luma");
    CHECK_NEAR(c.target.hdr.max_luma, 1000.0, 1e-3, "max_luma");
    CHECK(c.depth == 10, "depth %d", c.depth);
}

/* 6. Extended linear with SDR luminances: the sRGB display in gamma 2.2. */
static void test_ext_linear_sdr(void)
{
    struct view_colour_caps caps = sway_caps();
    struct view_colour_desc d = pref(PL_COLOR_PRIM_BT_709, VIEW_WP_PRIM_SRGB,
                                     VIEW_WP_TF_EXT_LINEAR, 0.2f, 80.0f, 80.0f,
                                     PL_COLOR_PRIM_BT_709, 0.2f, 80.0f);
    struct view_colour_choice c;

    CHECK(resolve(__func__, &caps, &d, &c) == 0, "refused");
    CHECK(c.candidate == 3 && !c.exact, "candidate %d", c.candidate);
    CHECK(c.primaries_named == VIEW_WP_PRIM_SRGB &&
          c.tf_named == VIEW_WP_TF_GAMMA22, "not tagged sRGB gamma 2.2");
    CHECK(c.target.primaries == PL_COLOR_PRIM_BT_709 &&
          c.target.transfer == PL_COLOR_TRC_GAMMA22, "not BT.709 gamma 2.2");
    CHECK(!c.set_luminances, "luminances sent for a fallback");
    /* gamma22's implied 0.2/80/80, since that is what the surface will be
     * read with */
    CHECK_NEAR(c.target.hdr.min_luma, 203.0 * 0.2 / 80.0, 1e-5, "min_luma");
    CHECK_NEAR(c.target.hdr.max_luma, 203.0, 1e-4, "max_luma");
}

/* 7. A version 1 compositor that knows the sRGB curve only by the name
 * version 2 deprecated. */
static void test_v1_srgb(void)
{
    struct view_colour_caps caps = {
        .version = 1,
        .parametric = true,
        .tf_named = BIT(VIEW_WP_TF_SRGB),
        .primaries_named = BIT(VIEW_WP_PRIM_SRGB),
        .intents = 1u << VIEW_WP_INTENT_PERCEPTUAL,
    };
    struct view_colour_desc d = pref(PL_COLOR_PRIM_BT_709, VIEW_WP_PRIM_SRGB,
                                     VIEW_WP_TF_SRGB, 0.2f, 80.0f, 80.0f,
                                     PL_COLOR_PRIM_BT_709, 0.2f, 80.0f);
    struct view_colour_choice c;

    CHECK(resolve(__func__, &caps, &d, &c) == 0, "refused");
    CHECK(c.exact, "not exact");
    CHECK(c.target.primaries == PL_COLOR_PRIM_BT_709 &&
          c.target.transfer == PL_COLOR_TRC_SRGB, "not BT.709 sRGB");
    CHECK(c.tf_named == VIEW_WP_TF_SRGB, "tf_named %d", c.tf_named);

    /* The same compositor preferring a curve it does not advertise lands on
     * the last candidate. */
    d.tf_named = VIEW_WP_TF_GAMMA22;
    CHECK(resolve(__func__, &caps, &d, &c) == 0, "refused");
    CHECK(c.candidate == 5 && !c.exact, "candidate %d", c.candidate);
    CHECK(c.target.transfer == PL_COLOR_TRC_SRGB, "transfer %d", c.target.transfer);
}

/* 8. Nothing this program can render is something the compositor takes. */
static void test_nothing(void)
{
    struct view_colour_caps caps = sway_caps();
    struct view_colour_desc d = pref(PL_COLOR_PRIM_BT_709, VIEW_WP_PRIM_SRGB,
                                     VIEW_WP_TF_EXT_LINEAR, 0.2f, 80.0f, 80.0f,
                                     PL_COLOR_PRIM_BT_709, 0.2f, 80.0f);
    struct view_colour_choice c;

    caps.tf_named = BIT(VIEW_WP_TF_EXT_LINEAR);
    CHECK(view_colour_resolve(&caps, &d, &c) == CRAW_ERROR(ENOTSUP), "not refused");

    /* HDR luminances make no difference without PQ. */
    d.max_lum = 1000.0f;
    CHECK(view_colour_resolve(&caps, &d, &c) == CRAW_ERROR(ENOTSUP), "not refused, HDR");

    /* Nor do primaries it does not advertise. */
    caps = sway_caps();
    caps.primaries_named = BIT(VIEW_WP_PRIM_DCI_P3);
    d = pref(PL_COLOR_PRIM_BT_709, VIEW_WP_PRIM_SRGB, VIEW_WP_TF_GAMMA22,
             0.2f, 80.0f, 80.0f, PL_COLOR_PRIM_BT_709, 0.2f, 80.0f);
    CHECK(view_colour_resolve(&caps, &d, &c) == CRAW_ERROR(ENOTSUP),
          "not refused, primaries");
}

/* 9. Primaries sent as xy only are taken by the name they match. */
static void test_unnamed_primaries(void)
{
    struct view_colour_caps caps = sway_caps();
    struct view_colour_desc d = pref(PL_COLOR_PRIM_BT_709, 0,
                                     VIEW_WP_TF_GAMMA22, 0.2f, 80.0f, 80.0f,
                                     PL_COLOR_PRIM_BT_709, 0.2f, 80.0f);
    struct view_colour_choice c;

    CHECK(resolve(__func__, &caps, &d, &c) == 0, "refused");
    CHECK(c.exact, "not exact");
    CHECK(c.primaries_named == VIEW_WP_PRIM_SRGB, "primaries_named %d", c.primaries_named);
    CHECK(view_colour_primaries_match(&d.primaries) == VIEW_WP_PRIM_SRGB, "match");

    /* And xy that match nothing refuse the first candidate, not all. */
    d.primaries.red.x = 0.7f;
    CHECK(resolve(__func__, &caps, &d, &c) == 0, "refused, custom");
    CHECK(c.candidate == 3, "candidate %d, custom", c.candidate);
}

/* 10. A power curve: sent as a power where the compositor takes one, the
 * named gamma22 otherwise. */
static void test_power(void)
{
    struct view_colour_caps caps = sway_caps();
    struct view_colour_desc d = pref(PL_COLOR_PRIM_BT_709, VIEW_WP_PRIM_SRGB,
                                     0, 0.2f, 80.0f, 80.0f,
                                     PL_COLOR_PRIM_BT_709, 0.2f, 80.0f);
    struct view_colour_choice c;

    d.tf_power = 2.2f;
    caps.set_tf_power = true;
    CHECK(resolve(__func__, &caps, &d, &c) == 0, "refused");
    CHECK(c.exact, "not exact");
    CHECK(c.tf_named == 0 && fabsf(c.tf_power - 2.2f) < 1e-6f,
          "tf_named %d power %g", c.tf_named, c.tf_power);
    CHECK(c.target.transfer == PL_COLOR_TRC_GAMMA22, "transfer %d", c.target.transfer);

    caps.set_tf_power = false;
    CHECK(resolve(__func__, &caps, &d, &c) == 0, "refused, no set_tf_power");
    CHECK(c.candidate == 3 && !c.exact, "candidate %d", c.candidate);
    CHECK(c.tf_named == VIEW_WP_TF_GAMMA22, "tf_named %d", c.tf_named);

    /* A power libplacebo has no curve for is not rendered as another. */
    caps.set_tf_power = true;
    d.tf_power = 2.35f;
    CHECK(resolve(__func__, &caps, &d, &c) == 0, "refused, 2.35");
    CHECK(c.candidate == 3, "candidate %d, 2.35", c.candidate);
}

/* The black level of an SDR fallback is the one its curve implies, not the
 * preferred description's: BT.1886's curve is built from it, and the
 * compositor decodes the surface with 0.01/100/100 when nothing says
 * otherwise. */
static void test_bt1886_black(void)
{
    struct view_colour_caps caps = sway_caps();
    struct view_colour_desc d = pref(PL_COLOR_PRIM_BT_709, VIEW_WP_PRIM_SRGB,
                                     VIEW_WP_TF_GAMMA28, 0.2f, 80.0f, 80.0f,
                                     PL_COLOR_PRIM_BT_709, 0.2f, 80.0f);
    struct view_colour_choice c;

    caps.tf_named = BIT(VIEW_WP_TF_BT1886);
    CHECK(resolve(__func__, &caps, &d, &c) == 0, "refused");
    CHECK(c.candidate == 4, "candidate %d", c.candidate);
    CHECK(c.target.transfer == PL_COLOR_TRC_BT_1886, "transfer %d", c.target.transfer);
    CHECK_NEAR(c.target.hdr.min_luma, 203.0 * 0.01 / 100.0, 1e-6, "min_luma");
    CHECK_NEAR(c.target.hdr.max_luma, 203.0, 1e-4, "max_luma");
}

/* An SDR preference with headroom above its reference white, on a
 * compositor that can be told the luminances: sent unchanged, and rendered
 * with 1.0 at the maximum, the reference white still at 203. */
static void test_sdr_headroom(void)
{
    struct view_colour_caps caps = sway_caps();
    struct view_colour_desc d = pref(PL_COLOR_PRIM_BT_709, VIEW_WP_PRIM_SRGB,
                                     VIEW_WP_TF_GAMMA22, 0.1f, 400.0f, 200.0f,
                                     PL_COLOR_PRIM_BT_709, 0.1f, 400.0f);
    struct view_colour_choice c;

    caps.set_luminances = true;
    CHECK(resolve(__func__, &caps, &d, &c) == 0, "refused");
    CHECK(c.exact, "not exact");
    CHECK(c.set_luminances, "luminances not sent");
    CHECK(c.min_lum == 0.1f && c.max_lum == 400.0f && c.ref_lum == 200.0f,
          "luminances %g/%g/%g", c.min_lum, c.max_lum, c.ref_lum);
    CHECK_NEAR(c.target.hdr.max_luma, 203.0 * 400.0 / 200.0, 1e-3, "max_luma");
    CHECK_NEAR(c.target.hdr.min_luma, 203.0 * 0.1 / 200.0, 1e-6, "min_luma");
    CHECK(c.depth == 8, "depth %d", c.depth);

    /* Without the feature the curve's defaults are what the surface is
     * read with, and so what is rendered. */
    caps.set_luminances = false;
    CHECK(resolve(__func__, &caps, &d, &c) == 0, "refused, no feature");
    CHECK(!c.set_luminances, "luminances sent without the feature");
    CHECK_NEAR(c.target.hdr.max_luma, 203.0, 1e-4, "max_luma, no feature");
}

/* HLG is encoded for the display the description names: the nominal 1000
 * cd/m² one when the luminances cannot be sent, the compositor's target
 * when they can, and then they say so. */
static void test_hlg(void)
{
    struct view_colour_caps caps = sway_caps();
    struct view_colour_desc d = pref(PL_COLOR_PRIM_BT_2020, VIEW_WP_PRIM_BT2020,
                                     VIEW_WP_TF_HLG, 0.005f, 1000.0f, 203.0f,
                                     PL_COLOR_PRIM_BT_2020, 0.001f, 600.0f);
    struct view_colour_choice c;

    caps.tf_named |= BIT(VIEW_WP_TF_HLG);
    CHECK(resolve(__func__, &caps, &d, &c) == 0, "refused");
    CHECK(c.exact && c.target.transfer == PL_COLOR_TRC_HLG, "not exact HLG");
    CHECK(!c.set_luminances, "luminances without the feature");
    CHECK_NEAR(c.target.hdr.max_luma, 1000.0, 1e-3, "max_luma");
    CHECK_NEAR(c.target.hdr.min_luma, 0.005, 1e-7, "min_luma");
    CHECK(c.depth == 10, "depth %d", c.depth);

    caps.set_luminances = true;
    CHECK(resolve(__func__, &caps, &d, &c) == 0, "refused, set_luminances");
    CHECK(c.set_luminances, "luminances not sent");
    CHECK_NEAR(c.target.hdr.max_luma, 600.0, 1e-3, "max_luma");
    CHECK_NEAR(c.target.hdr.min_luma, 0.001, 1e-7, "min_luma");
    CHECK(c.max_lum == c.target.hdr.max_luma && c.min_lum == c.target.hdr.min_luma &&
          c.ref_lum == 203.0f, "luminances %g/%g/%g", c.min_lum, c.max_lum, c.ref_lum);
}

/* The protocol's cie1931_xyz is ST 428's XYZ, not libplacebo's CIE RGB. */
static void test_xyz(void)
{
    struct view_colour_caps caps = sway_caps();
    struct view_colour_desc d = pref(PL_COLOR_PRIM_BT_709, VIEW_WP_PRIM_CIE1931_XYZ,
                                     VIEW_WP_TF_GAMMA22, 0.2f, 80.0f, 80.0f,
                                     PL_COLOR_PRIM_BT_709, 0.2f, 80.0f);
    struct view_colour_choice c;

    caps.primaries_named |= BIT(VIEW_WP_PRIM_CIE1931_XYZ);
    CHECK(resolve(__func__, &caps, &d, &c) == 0, "refused");
    CHECK(c.candidate == 3, "candidate %d", c.candidate);
    CHECK(c.why && strstr(c.why, "libplacebo"), "why: %s", c.why ? c.why : "(null)");
}

/* compound_power_2_4 is the piecewise sRGB curve, which libplacebo has. */
static void test_compound_power(void)
{
    struct view_colour_caps caps = sway_caps();
    struct view_colour_desc d = pref(PL_COLOR_PRIM_BT_709, VIEW_WP_PRIM_SRGB,
                                     VIEW_WP_TF_COMPOUND_POWER_2_4, 0.2f, 80.0f, 80.0f,
                                     PL_COLOR_PRIM_BT_709, 0.2f, 80.0f);
    struct view_colour_choice c;

    caps.tf_named |= BIT(VIEW_WP_TF_COMPOUND_POWER_2_4);
    CHECK(resolve(__func__, &caps, &d, &c) == 0, "refused");
    CHECK(c.exact && c.target.transfer == PL_COLOR_TRC_SRGB, "not exact sRGB");
}

/* What decides whether a changed preference re-tags the surface. */
static void test_equal(void)
{
    struct view_colour_caps caps = sway_caps();
    struct view_colour_desc d = pref(PL_COLOR_PRIM_BT_2020, VIEW_WP_PRIM_BT2020,
                                     VIEW_WP_TF_ST2084_PQ, 0.005f, 10000.0f, 203.0f,
                                     PL_COLOR_PRIM_BT_2020, 0.005f, 1000.0f);
    struct view_colour_choice a, b;

    CHECK(resolve(__func__, &caps, &d, &a) == 0, "refused");
    CHECK(resolve(__func__, &caps, &d, &b) == 0, "refused");
    CHECK(view_colour_choice_equal(&a, &b), "the same preference differs");

    d.target_max = 1200.0f;
    CHECK(resolve(__func__, &caps, &d, &b) == 0, "refused");
    CHECK(!view_colour_choice_equal(&a, &b), "a new peak compares equal");

    /* Another SDR preference is another choice, too. */
    d = pref(PL_COLOR_PRIM_BT_709, VIEW_WP_PRIM_SRGB, VIEW_WP_TF_GAMMA22,
             0.2f, 80.0f, 80.0f, PL_COLOR_PRIM_BT_709, 0.2f, 80.0f);
    CHECK(resolve(__func__, &caps, &d, &b) == 0, "refused");
    CHECK(!view_colour_choice_equal(&a, &b), "SDR compares equal to HDR");
}

/* The log lines never overflow and always terminate. */
static void test_describe(void)
{
    struct view_colour_caps caps = sway_caps();
    struct view_colour_desc d = pref(PL_COLOR_PRIM_BT_2020, VIEW_WP_PRIM_BT2020,
                                     VIEW_WP_TF_ST2084_PQ, 0.005f, 10000.0f, 203.0f,
                                     PL_COLOR_PRIM_DISPLAY_P3, 0.005f, 10000.0f);
    struct view_colour_choice c;
    char buf[512], small[8];

    CHECK(resolve(__func__, &caps, &d, &c) == 0, "refused");

    view_colour_describe_caps(&caps, buf, sizeof(buf));
    CHECK(strstr(buf, "parametric") && strstr(buf, "st2084_pq") &&
          strstr(buf, "bt2020") && strstr(buf, "perceptual"), "caps: %s", buf);
    view_colour_describe_desc(&d, buf, sizeof(buf));
    CHECK(strstr(buf, "bt2020 st2084_pq") && strstr(buf, "custom (display_p3)"),
          "desc: %s", buf);
    view_colour_describe_choice(&c, buf, sizeof(buf));
    CHECK(strstr(buf, "BT.2020 PQ") && strstr(buf, "assumed") &&
          strstr(buf, "10-bit") && strstr(buf, "exact"), "choice: %s", buf);

    memset(small, 'x', sizeof(small));
    view_colour_describe_choice(&c, small, sizeof(small));
    CHECK(memchr(small, 0, sizeof(small)), "not terminated");
}

int main(void)
{
    test_sway_sdr();
    test_sway_hdr();
    test_hdr_p3_target();
    test_hdr_no_peak();
    test_ext_linear_hdr();
    test_ext_linear_sdr();
    test_v1_srgb();
    test_nothing();
    test_unnamed_primaries();
    test_power();
    test_bt1886_black();
    test_sdr_headroom();
    test_hlg();
    test_xyz();
    test_compound_power();
    test_equal();
    test_describe();

    if (failures) {
        printf("test_colour: %d failure(s)\n", failures);
        return 1;
    }

    printf("test_colour: all checks passed\n");
    return 0;
}
