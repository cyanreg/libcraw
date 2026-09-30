/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* The window's colour, the pure half: what to render and what to tell the
 * compositor, out of what the compositor can do and what it prefers for
 * the surface. Nothing in here speaks Wayland or knows about SDL, so that
 * tests/test_colour.c can put it through every case without a compositor.
 * colour.c reads the protocol's events into the structs of colour.h and
 * hands them here (DESIGN.md 22). */

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <craw/utils.h>

#include "colour.h"

/* --------------------------------------------------------------- names */

static const char *const view_colour_tf_names[] = {
    [VIEW_WP_TF_BT1886]             = "bt1886",
    [VIEW_WP_TF_GAMMA22]            = "gamma22",
    [VIEW_WP_TF_GAMMA28]            = "gamma28",
    [VIEW_WP_TF_ST240]              = "st240",
    [VIEW_WP_TF_EXT_LINEAR]         = "ext_linear",
    [VIEW_WP_TF_LOG_100]            = "log_100",
    [VIEW_WP_TF_LOG_316]            = "log_316",
    [VIEW_WP_TF_XVYCC]              = "xvycc",
    [VIEW_WP_TF_SRGB]               = "srgb",
    [VIEW_WP_TF_EXT_SRGB]           = "ext_srgb",
    [VIEW_WP_TF_ST2084_PQ]          = "st2084_pq",
    [VIEW_WP_TF_ST428]              = "st428",
    [VIEW_WP_TF_HLG]                = "hlg",
    [VIEW_WP_TF_COMPOUND_POWER_2_4] = "compound_power_2_4",
};

static const char *const view_colour_prim_names[] = {
    [VIEW_WP_PRIM_SRGB]         = "srgb",
    [VIEW_WP_PRIM_PAL_M]        = "pal_m",
    [VIEW_WP_PRIM_PAL]          = "pal",
    [VIEW_WP_PRIM_NTSC]         = "ntsc",
    [VIEW_WP_PRIM_GENERIC_FILM] = "generic_film",
    [VIEW_WP_PRIM_BT2020]       = "bt2020",
    [VIEW_WP_PRIM_CIE1931_XYZ]  = "cie1931_xyz",
    [VIEW_WP_PRIM_DCI_P3]       = "dci_p3",
    [VIEW_WP_PRIM_DISPLAY_P3]   = "display_p3",
    [VIEW_WP_PRIM_ADOBE_RGB]    = "adobe_rgb",
};

static const char *const view_colour_intent_names[] = {
    [VIEW_WP_INTENT_PERCEPTUAL]             = "perceptual",
    [VIEW_WP_INTENT_RELATIVE]               = "relative",
    [VIEW_WP_INTENT_SATURATION]             = "saturation",
    [VIEW_WP_INTENT_ABSOLUTE]               = "absolute",
    [VIEW_WP_INTENT_RELATIVE_BPC]           = "relative_bpc",
    [VIEW_WP_INTENT_ABSOLUTE_NO_ADAPTATION] = "absolute_no_adaptation",
};

#define VIEW_COLOUR_NAME(table, i)                                         \
    ((i) > 0 && (size_t)(i) < CRAW_ARRAY_ELEMS(table) && (table)[i]        \
     ? (table)[i] : "unknown")

const char *view_colour_tf_name(int wp_tf)
{
    return VIEW_COLOUR_NAME(view_colour_tf_names, wp_tf);
}

const char *view_colour_primaries_name(int wp_prim)
{
    return VIEW_COLOUR_NAME(view_colour_prim_names, wp_prim);
}

const char *view_colour_intent_name(int wp_intent)
{
    /* Perceptual is 0, which the macro above would call unknown. */
    if (wp_intent == VIEW_WP_INTENT_PERCEPTUAL)
        return view_colour_intent_names[VIEW_WP_INTENT_PERCEPTUAL];
    return VIEW_COLOUR_NAME(view_colour_intent_names, wp_intent);
}

/* ------------------------------------------------------------- mapping */

/* The named transfer functions libplacebo can encode into the swapchain's
 * image. That image is RGBA16 UNORM (pass-through, see colour.c), which
 * holds 0 to 1 and nothing outside it, so the extended ones are out:
 * ext_linear needs a float format, and a compositor's HDR is had through
 * PQ or HLG instead. st428 is XYZ, which is a colour system and not a
 * curve; the logs, st240, xvYCC and ext_srgb have no libplacebo curve.
 *
 * compound_power_2_4 is the piecewise IEC 61966-2-1 curve, the sRGB
 * encoding's exact inverse, which is precisely libplacebo's
 * PL_COLOR_TRC_SRGB (and what the deprecated srgb name meant, as wlroots
 * treats them); it can be rendered, so it is. */
static enum pl_color_transfer view_colour_tf_pl(int wp_tf)
{
    switch (wp_tf) {
    case VIEW_WP_TF_BT1886:             return PL_COLOR_TRC_BT_1886;
    case VIEW_WP_TF_GAMMA22:            return PL_COLOR_TRC_GAMMA22;
    case VIEW_WP_TF_GAMMA28:            return PL_COLOR_TRC_GAMMA28;
    case VIEW_WP_TF_SRGB:
    case VIEW_WP_TF_COMPOUND_POWER_2_4: return PL_COLOR_TRC_SRGB;
    case VIEW_WP_TF_ST2084_PQ:          return PL_COLOR_TRC_PQ;
    case VIEW_WP_TF_HLG:                return PL_COLOR_TRC_HLG;
    default:                            return PL_COLOR_TRC_UNKNOWN;
    }
}

/* A power curve is renderable when it is one of libplacebo's pure powers.
 * The protocol carries the exponent to four decimals, so 1e-3 is loose
 * enough for any compositor's rounding and tight enough to tell 2.2 from
 * 2.4. */
static enum pl_color_transfer view_colour_power_pl(float e)
{
    static const struct {
        float e;
        enum pl_color_transfer trc;
    } powers[] = {
        { 1.8f, PL_COLOR_TRC_GAMMA18 },
        { 2.0f, PL_COLOR_TRC_GAMMA20 },
        { 2.2f, PL_COLOR_TRC_GAMMA22 },
        { 2.4f, PL_COLOR_TRC_GAMMA24 },
        { 2.6f, PL_COLOR_TRC_GAMMA26 },
        { 2.8f, PL_COLOR_TRC_GAMMA28 },
    };

    for (size_t i = 0; i < CRAW_ARRAY_ELEMS(powers); i++)
        if (fabsf(e - powers[i].e) <= 1e-3f)
            return powers[i].trc;

    return PL_COLOR_TRC_UNKNOWN;
}

/* The named primaries libplacebo has a set of raw primaries for, with the
 * same xy. cie1931_xyz is not among them: the protocol's name is SMPTE ST
 * 428-1's XYZ "primaries", (1, 0), (0, 1), (0, 0) with the equal-energy
 * white, while PL_COLOR_PRIM_CIE_1931 is CIE 1931 *RGB*, 700/546.1/435.8
 * nm, a different triangle altogether. The two only share a name (and
 * libplacebo's libav mapping of AVCOL_PRI_SMPTE428), and XYZ goes with the
 * st428 curve, which cannot be rendered here anyway. */
static enum pl_color_primaries view_colour_prim_pl(int wp_prim)
{
    switch (wp_prim) {
    case VIEW_WP_PRIM_SRGB:         return PL_COLOR_PRIM_BT_709;
    case VIEW_WP_PRIM_PAL_M:        return PL_COLOR_PRIM_BT_470M;
    case VIEW_WP_PRIM_PAL:          return PL_COLOR_PRIM_BT_601_625;
    case VIEW_WP_PRIM_NTSC:         return PL_COLOR_PRIM_BT_601_525;
    case VIEW_WP_PRIM_GENERIC_FILM: return PL_COLOR_PRIM_FILM_C;
    case VIEW_WP_PRIM_BT2020:       return PL_COLOR_PRIM_BT_2020;
    case VIEW_WP_PRIM_DCI_P3:       return PL_COLOR_PRIM_DCI_P3;
    case VIEW_WP_PRIM_DISPLAY_P3:   return PL_COLOR_PRIM_DISPLAY_P3;
    case VIEW_WP_PRIM_ADOBE_RGB:    return PL_COLOR_PRIM_ADOBE;
    default:                        return PL_COLOR_PRIM_UNKNOWN;
    }
}

int view_colour_primaries_match(const struct pl_raw_primaries *xy)
{
    for (int p = VIEW_WP_PRIM_SRGB; p <= VIEW_WP_PRIM_ADOBE_RGB; p++) {
        enum pl_color_primaries pl = view_colour_prim_pl(p);

        if (pl != PL_COLOR_PRIM_UNKNOWN &&
            pl_raw_primaries_similar(xy, pl_raw_primaries_get(pl)))
            return p;
    }

    return 0;
}

/* The luminances a description has when set_luminances is not sent: the
 * ones its transfer function implies. BT.1886's come from BT.2035, PQ's
 * and HLG's from BT.2408, and everything else, pure powers included, has
 * the sRGB display's of srgb.xalter, which set_luminances names as the
 * protocol's default. */
static void view_colour_tf_defaults(int wp_tf, float *min, float *max, float *ref)
{
    switch (wp_tf) {
    case VIEW_WP_TF_BT1886:
        *min = 0.01f;  *max = 100.0f;   *ref = 100.0f;
        break;
    case VIEW_WP_TF_ST2084_PQ:
        *min = 0.005f; *max = 10000.0f; *ref = 203.0f;
        break;
    case VIEW_WP_TF_HLG:
        *min = 0.005f; *max = 1000.0f;  *ref = 203.0f;
        break;
    default:
        *min = 0.2f;   *max = 80.0f;    *ref = 80.0f;
        break;
    }
}

/* The protocol carries a minimum luminance in units of 1e-4 cd/m² and a
 * maximum or reference one in whole cd/m², and refuses (with a protocol
 * error, which ends the connection) any pair where the latter is not above
 * the former. Judged on the integers that will actually be sent. */
static bool view_colour_lum_above(float min, float above)
{
    return isfinite(min) && isfinite(above) && min >= 0.0f &&
           roundf(above) * 10000.0f > roundf(min * 10000.0f) &&
           roundf(above) <= (float)UINT32_MAX;
}

/* ---------------------------------------------------------- the choice */

struct view_colour_cand {
    int num;            /* its place in the list, 1 to 5 */
    int prim;           /* wp primaries, 0: none */
    int tf;             /* wp transfer function, 0: the power below */
    float power;
};

static bool view_colour_bit(uint64_t set, int i)
{
    return i > 0 && i < 64 && (set >> i) & 1;
}

/* NULL when the compositor can be told this pair and libplacebo can render
 * it, else why not. */
static const char *view_colour_refuse(const struct view_colour_caps *caps,
                                      const struct view_colour_cand *c,
                                      enum pl_color_primaries *prim,
                                      enum pl_color_transfer *trc)
{
    if (!c->prim)
        return "the primaries have no name";
    *prim = view_colour_prim_pl(c->prim);
    if (*prim == PL_COLOR_PRIM_UNKNOWN)
        return "libplacebo has no such primaries";
    if (!view_colour_bit(caps->primaries_named, c->prim))
        return "the primaries are not advertised";

    if (c->tf) {
        *trc = view_colour_tf_pl(c->tf);
        if (*trc == PL_COLOR_TRC_UNKNOWN)
            return "the transfer function cannot be rendered into a UNORM image";
        if (!view_colour_bit(caps->tf_named, c->tf))
            return "the transfer function is not advertised";
    } else {
        if (c->power <= 0.0f)
            return "no transfer function";
        *trc = view_colour_power_pl(c->power);
        if (*trc == PL_COLOR_TRC_UNKNOWN)
            return "libplacebo has no such power curve";
        if (!caps->set_tf_power)
            return "a power curve needs set_tf_power";
    }

    return NULL;
}

/* The compositor's target gamut as something that can be rendered into and
 * sent: false when there is no point, i.e. when it is not a valid triangle
 * or is the container anyway.
 *
 * A target that reaches outside the container cannot be encoded in a UNORM
 * image, and cannot be sent without extended_target_volume. It is clipped
 * into the container rather than dropped, because the one that matters
 * most is: a P3 panel behind a BT.2020 container. Both reds lie on the
 * spectral locus, P3's at about 615 nm and BT.2020's at 630, and the
 * locus bulges outwards between them, so P3's red is outside BT.2020's
 * red-green edge by 0.0017 in y. Refusing it would gamut map into all of
 * BT.2020 and leave the panel to clip; clipped, P3's red moves onto that
 * edge and the rest of the triangle stays. The clipped points are then
 * pulled a ten-thousandth of the way towards white, which puts them
 * strictly inside the edge they were moved onto and keeps them there once
 * they are rounded to the protocol's millionths. */
static bool view_colour_gamut(const struct pl_raw_primaries *container,
                              const struct pl_raw_primaries *target,
                              struct pl_raw_primaries *out)
{
    struct pl_raw_primaries g;
    struct pl_cie_xy *p[3] = { &g.red, &g.green, &g.blue };

    if (!pl_primaries_valid(target) || pl_raw_primaries_similar(container, target))
        return false;

    if (pl_primaries_superset(container, target)) {
        *out = *target;
        return true;
    }

    if (!pl_primaries_compatible(target, container))
        return false;

    g = pl_primaries_clip(target, container);
    for (int i = 0; i < 3; i++) {
        p[i]->x = g.white.x + (p[i]->x - g.white.x) * (1.0f - 1e-4f);
        p[i]->y = g.white.y + (p[i]->y - g.white.y) * (1.0f - 1e-4f);
    }

    if (!pl_primaries_valid(&g) || !pl_primaries_superset(container, &g) ||
        pl_raw_primaries_similar(container, &g))
        return false;

    *out = g;
    return true;
}

int view_colour_resolve(const struct view_colour_caps *caps,
                        const struct view_colour_desc *pref,
                        struct view_colour_choice *out)
{
    struct view_colour_cand cands[5];
    const struct pl_raw_primaries *container;
    struct pl_raw_primaries gamut;
    enum pl_color_primaries prim = PL_COLOR_PRIM_UNKNOWN;
    enum pl_color_transfer trc = PL_COLOR_TRC_UNKNOWN;
    const char *why = NULL;
    float min, max, ref;
    int nb = 0, win = -1;
    bool hdr, hdr_trc;

    *out = (struct view_colour_choice) { 0 };

    /* What the compositor would like best, as it is; then, for a
     * compositor that prefers something HDR, the one HDR encoding a UNORM
     * image can carry; then the three SDR encodings of the sRGB display,
     * the last of them for a version 1 compositor that has only the name
     * version 2 deprecated. An unnamed set of primaries is taken as the
     * named one it matches, so that a compositor that only sends xy is not
     * refused for it. */
    cands[nb++] = (struct view_colour_cand) {
        .num   = 1,
        .prim  = pref->primaries_named ? pref->primaries_named
                                       : view_colour_primaries_match(&pref->primaries),
        .tf    = pref->tf_named,
        .power = pref->tf_named ? 0.0f : pref->tf_power,
    };
    hdr = pref->tf_named == VIEW_WP_TF_ST2084_PQ || pref->tf_named == VIEW_WP_TF_HLG ||
          pref->max_lum > pref->ref_lum;
    if (hdr) {
        cands[nb++] = (struct view_colour_cand) {
            .num = 2, .prim = VIEW_WP_PRIM_BT2020, .tf = VIEW_WP_TF_ST2084_PQ };
    }
    cands[nb++] = (struct view_colour_cand) {
        .num = 3, .prim = VIEW_WP_PRIM_SRGB, .tf = VIEW_WP_TF_GAMMA22 };
    cands[nb++] = (struct view_colour_cand) {
        .num = 4, .prim = VIEW_WP_PRIM_SRGB, .tf = VIEW_WP_TF_BT1886 };
    cands[nb++] = (struct view_colour_cand) {
        .num = 5, .prim = VIEW_WP_PRIM_SRGB, .tf = VIEW_WP_TF_SRGB };

    for (int i = 0; i < nb && win < 0; i++) {
        const char *r = view_colour_refuse(caps, &cands[i], &prim, &trc);

        if (!r)
            win = i;
        else if (!i)
            why = r;
    }

    out->why = why;
    if (win < 0)
        return CRAW_ERROR(ENOTSUP);

    out->candidate = cands[win].num;
    out->exact = cands[win].num == 1;
    out->primaries_named = cands[win].prim;
    out->tf_named = cands[win].tf;
    out->tf_power = cands[win].tf ? 0.0f : cands[win].power;
    out->depth = pl_color_transfer_is_hdr(trc) ? 10 : 8;
    out->intent = caps->intents & (1u << VIEW_WP_INTENT_RELATIVE)
                ? VIEW_WP_INTENT_RELATIVE : VIEW_WP_INTENT_PERCEPTUAL;
    hdr_trc = pl_color_transfer_is_hdr(trc);

    /* The gamut libplacebo maps into, and the target volume the compositor
     * is told about: the compositor's own target where it is smaller than
     * the container, the container otherwise. Set whichever it is, so that
     * the primaries sent below are exactly the ones libplacebo mapped
     * into. */
    container = pl_raw_primaries_get(prim);
    if (!view_colour_gamut(container, &pref->target, &gamut))
        gamut = *container;

    if (trc == PL_COLOR_TRC_PQ) {
        /* PQ is absolute: libplacebo encodes nits, and the luminances
         * decide nothing but where the tone mapper aims, which is the
         * compositor's target volume. A peak of 10000 is the curve's own
         * reach and not a display's - wlroots sends the transfer
         * function's defaults as the target, 10000 included - so it and a
         * range upside down both mean the compositor has no idea, and a
         * 1000 cd/m² display is assumed. */
        min = pref->target_min > 0.0f ? pref->target_min : PL_COLOR_HDR_BLACK;
        max = pref->target_max;
        if (!(max < 10000.0f && max > min)) {
            max = PL_COLOR_HLG_PEAK;
            out->peak_assumed = true;
        }

        /* The primary volume. The protocol takes PQ's maximum as the
         * minimum plus 10000, i.e. the minimum is an offset under the
         * whole curve, while libplacebo's PQ is the absolute one: the
         * pair only agrees at a minimum of zero. The smallest the protocol
         * can carry above it, 1e-4 cd/m², keeps any compositor that
         * divides by the minimum out of trouble at no visible cost; the
         * display's real black is the target volume's business, below.
         * The reference is BT.2408's 203, which is also PQ's default and
         * libplacebo's SDR white, so nothing needs scaling either way. */
        if (caps->set_luminances) {
            out->set_luminances = true;
            out->min_lum = 0.0001f;
            out->max_lum = 10000.0f;
            out->ref_lum = PL_COLOR_SDR_WHITE;
        }
    } else if (trc == PL_COLOR_TRC_HLG) {
        /* HLG is relative, and libplacebo's encoding of it depends on the
         * display it is encoded for: the system gamma follows the peak and
         * the black lift the minimum. So the signal is encoded for the
         * display the description names. Without set_luminances that is
         * the protocol's nominal HLG display, 0.005 to 1000 cd/m², and the
         * compositor maps it onto its own; with it, the compositor's
         * target, which it is then told. */
        if (caps->set_luminances) {
            min = pref->target_min > 0.0f ? pref->target_min : PL_COLOR_HDR_BLACK;
            max = pref->target_max;
            if (!(max < 10000.0f && max > min)) {
                max = PL_COLOR_HLG_PEAK;
                out->peak_assumed = true;
            }
            out->set_luminances = true;
            out->min_lum = min;
            out->max_lum = max;
            out->ref_lum = PL_COLOR_SDR_WHITE;
        } else {
            view_colour_tf_defaults(VIEW_WP_TF_HLG, &min, &max, &ref);
        }
    } else {
        /* An SDR curve is relative: libplacebo puts its reference white,
         * 1.0, at PL_COLOR_SDR_WHITE by construction, maps the target's
         * [min, max] onto the signal's [0, 1] and, for BT.1886, derives the
         * curve's black lift from the same two. So the luminances that
         * matter are the ones of the description the surface will carry,
         * scaled so that its reference white lands on 203: the
         * compositor's own when they can be sent - which for sway's
         * 0.2/80/80 is a contrast of 400:1 and no headroom - and else the
         * ones the chosen curve implies, since those are what the
         * compositor will read the surface with. A preferred description
         * whose maximum is above its reference white gets that headroom as
         * a target above 203. */
        if (out->exact && caps->set_luminances &&
            pref->ref_lum > 0.0f &&
            view_colour_lum_above(pref->min_lum, pref->max_lum) &&
            view_colour_lum_above(pref->min_lum, pref->ref_lum)) {
            out->set_luminances = true;
            out->min_lum = pref->min_lum;
            out->max_lum = pref->max_lum;
            out->ref_lum = pref->ref_lum;
            min = pref->min_lum;
            max = pref->max_lum;
            ref = pref->ref_lum;
        } else {
            view_colour_tf_defaults(out->tf_named, &min, &max, &ref);
        }

        max = PL_COLOR_SDR_WHITE * max / ref;
        min = CRAW_MAX(PL_COLOR_SDR_WHITE * min / ref, PL_COLOR_HDR_BLACK);
    }

    out->target = (struct pl_color_space) {
        .primaries = prim,
        .transfer  = trc,
        .hdr = {
            .prim     = gamut,
            .min_luma = min,
            .max_luma = max,
        },
    };

    /* The target volume: for HDR always, since it is the one place the
     * display's peak and black reach the compositor, and for SDR only when
     * the gamut is not the container's (its luminances are the primary
     * volume's there, which is the default). Never outside the container:
     * extended_target_volume is not used. */
    if (caps->mastering_display_primaries &&
        (hdr_trc || !pl_raw_primaries_similar(&gamut, container))) {
        out->set_mastering = true;
        out->mastering = gamut;
        if (hdr_trc && view_colour_lum_above(min, max)) {
            out->mastering_min = min;
            out->mastering_max = max;
        }
    }

    return 0;
}

bool view_colour_choice_equal(const struct view_colour_choice *a,
                              const struct view_colour_choice *b)
{
    return pl_color_space_equal(&a->target, &b->target) &&
           a->depth == b->depth &&
           a->primaries_named == b->primaries_named &&
           a->tf_named == b->tf_named &&
           a->tf_power == b->tf_power &&
           a->set_luminances == b->set_luminances &&
           a->min_lum == b->min_lum &&
           a->max_lum == b->max_lum &&
           a->ref_lum == b->ref_lum &&
           a->set_mastering == b->set_mastering &&
           pl_raw_primaries_equal(&a->mastering, &b->mastering) &&
           a->mastering_min == b->mastering_min &&
           a->mastering_max == b->mastering_max &&
           a->intent == b->intent;
}

/* ---------------------------------------------------------------- logs */

static const char *view_colour_pl_prim_name(enum pl_color_primaries prim)
{
    switch (prim) {
    case PL_COLOR_PRIM_BT_709:      return "BT.709";
    case PL_COLOR_PRIM_BT_470M:     return "BT.470M";
    case PL_COLOR_PRIM_BT_601_625:  return "BT.601-625";
    case PL_COLOR_PRIM_BT_601_525:  return "BT.601-525";
    case PL_COLOR_PRIM_FILM_C:      return "film C";
    case PL_COLOR_PRIM_BT_2020:     return "BT.2020";
    case PL_COLOR_PRIM_DCI_P3:      return "DCI-P3";
    case PL_COLOR_PRIM_DISPLAY_P3:  return "Display P3";
    case PL_COLOR_PRIM_ADOBE:       return "Adobe RGB";
    default:                        return "?";
    }
}

static const char *view_colour_pl_trc_name(enum pl_color_transfer trc)
{
    switch (trc) {
    case PL_COLOR_TRC_BT_1886:  return "BT.1886";
    case PL_COLOR_TRC_SRGB:     return "sRGB";
    case PL_COLOR_TRC_GAMMA18:  return "gamma 1.8";
    case PL_COLOR_TRC_GAMMA20:  return "gamma 2.0";
    case PL_COLOR_TRC_GAMMA22:  return "gamma 2.2";
    case PL_COLOR_TRC_GAMMA24:  return "gamma 2.4";
    case PL_COLOR_TRC_GAMMA26:  return "gamma 2.6";
    case PL_COLOR_TRC_GAMMA28:  return "gamma 2.8";
    case PL_COLOR_TRC_PQ:       return "PQ";
    case PL_COLOR_TRC_HLG:      return "HLG";
    default:                    return "?";
    }
}

/* snprintf that keeps appending, and stops quietly at the end of `buf`. */
static size_t view_colour_cat(char *buf, size_t size, size_t at, const char *fmt, ...)
    craw_printf_format(4, 5);

static size_t view_colour_cat(char *buf, size_t size, size_t at, const char *fmt, ...)
{
    va_list args;
    int n;

    if (at >= size)
        return at;

    va_start(args, fmt);
    n = vsnprintf(buf + at, size - at, fmt, args);
    va_end(args);

    return n < 0 ? at : CRAW_MIN(at + (size_t)n, size);
}

static size_t view_colour_cat_prim(char *buf, size_t size, size_t at, int named,
                                   const struct pl_raw_primaries *xy)
{
    int match;

    if (named)
        return view_colour_cat(buf, size, at, "%s", view_colour_primaries_name(named));

    match = view_colour_primaries_match(xy);
    if (match)
        return view_colour_cat(buf, size, at, "custom (%s)",
                               view_colour_primaries_name(match));

    /* Near one, which is what a target clipped into its container is: P3
     * in BT.2020 is a few thousandths off P3. For the log only. */
    for (int p = VIEW_WP_PRIM_SRGB; p <= VIEW_WP_PRIM_ADOBE_RGB; p++) {
        enum pl_color_primaries pl = view_colour_prim_pl(p);
        const struct pl_raw_primaries *r;
        float d;

        if (pl == PL_COLOR_PRIM_UNKNOWN)
            continue;
        r = pl_raw_primaries_get(pl);
        d = fabsf(xy->red.x - r->red.x)     + fabsf(xy->red.y - r->red.y) +
            fabsf(xy->green.x - r->green.x) + fabsf(xy->green.y - r->green.y) +
            fabsf(xy->blue.x - r->blue.x)   + fabsf(xy->blue.y - r->blue.y) +
            fabsf(xy->white.x - r->white.x) + fabsf(xy->white.y - r->white.y);
        if (d < 0.01f)
            return view_colour_cat(buf, size, at, "custom (near %s)",
                                   view_colour_primaries_name(p));
    }

    return view_colour_cat(buf, size, at, "custom");
}

void view_colour_describe_caps(const struct view_colour_caps *caps,
                               char *buf, size_t size)
{
    static const struct {
        size_t off;
        const char *name;
    } feats[] = {
        { offsetof(struct view_colour_caps, parametric),     "parametric" },
        { offsetof(struct view_colour_caps, set_primaries),  "set_primaries" },
        { offsetof(struct view_colour_caps, set_tf_power),   "set_tf_power" },
        { offsetof(struct view_colour_caps, set_luminances), "set_luminances" },
        { offsetof(struct view_colour_caps, mastering_display_primaries),
          "set_mastering_display_primaries" },
        { offsetof(struct view_colour_caps, extended_target_volume),
          "extended_target_volume" },
    };
    size_t at = 0;

    at = view_colour_cat(buf, size, at, "v%u,", caps->version);
    for (size_t i = 0; i < CRAW_ARRAY_ELEMS(feats); i++)
        if (*(const bool *)((const char *)caps + feats[i].off))
            at = view_colour_cat(buf, size, at, " %s", feats[i].name);

    at = view_colour_cat(buf, size, at, "; tf");
    for (int i = 1; i < 64; i++)
        if (view_colour_bit(caps->tf_named, i))
            at = view_colour_cat(buf, size, at, " %s", view_colour_tf_name(i));

    at = view_colour_cat(buf, size, at, "; primaries");
    for (int i = 1; i < 64; i++)
        if (view_colour_bit(caps->primaries_named, i))
            at = view_colour_cat(buf, size, at, " %s", view_colour_primaries_name(i));

    at = view_colour_cat(buf, size, at, "; intents");
    for (int i = 0; i < 32; i++)
        if (caps->intents & (1u << i))
            at = view_colour_cat(buf, size, at, " %s", view_colour_intent_name(i));
    (void)at;
}

void view_colour_describe_desc(const struct view_colour_desc *d,
                               char *buf, size_t size)
{
    size_t at = 0;

    at = view_colour_cat_prim(buf, size, at, d->primaries_named, &d->primaries);
    if (d->tf_named)
        at = view_colour_cat(buf, size, at, " %s", view_colour_tf_name(d->tf_named));
    else
        at = view_colour_cat(buf, size, at, " power %.5g", d->tf_power);
    at = view_colour_cat(buf, size, at, " %.5g/%.5g/%.5g cd/m², target ",
                         d->min_lum, d->max_lum, d->ref_lum);
    at = view_colour_cat_prim(buf, size, at, 0, &d->target);
    at = view_colour_cat(buf, size, at, " %.5g/%.5g", d->target_min, d->target_max);
    if (d->max_cll > 0.0f || d->max_fall > 0.0f)
        at = view_colour_cat(buf, size, at, ", cll %.5g fall %.5g",
                             d->max_cll, d->max_fall);
    (void)at;
}

void view_colour_describe_choice(const struct view_colour_choice *c,
                                 char *buf, size_t size)
{
    const struct pl_raw_primaries *container = pl_raw_primaries_get(c->target.primaries);
    size_t at = 0;

    at = view_colour_cat(buf, size, at, "%s %s %.5g to %.5g cd/m²",
                         view_colour_pl_prim_name(c->target.primaries),
                         view_colour_pl_trc_name(c->target.transfer),
                         c->target.hdr.min_luma, c->target.hdr.max_luma);
    if (c->peak_assumed)
        at = view_colour_cat(buf, size, at, " (peak assumed: the compositor gave none)");
    if (!pl_raw_primaries_similar(&c->target.hdr.prim, container)) {
        at = view_colour_cat(buf, size, at, ", gamut ");
        at = view_colour_cat_prim(buf, size, at, 0, &c->target.hdr.prim);
    }
    at = view_colour_cat(buf, size, at, ", %d-bit dither, ", c->depth);
    if (c->exact)
        at = view_colour_cat(buf, size, at, "exact");
    else
        at = view_colour_cat(buf, size, at, "fallback %d (%s)", c->candidate,
                             c->why ? c->why : "?");
    at = view_colour_cat(buf, size, at, "; tagged %s ",
                         view_colour_primaries_name(c->primaries_named));
    if (c->tf_named)
        at = view_colour_cat(buf, size, at, "%s", view_colour_tf_name(c->tf_named));
    else
        at = view_colour_cat(buf, size, at, "power %.5g", c->tf_power);
    if (c->set_luminances)
        at = view_colour_cat(buf, size, at, " %.5g/%.5g/%.5g cd/m²",
                             c->min_lum, c->max_lum, c->ref_lum);
    if (c->set_mastering) {
        at = view_colour_cat(buf, size, at, ", mastering ");
        at = view_colour_cat_prim(buf, size, at, 0, &c->mastering);
        if (c->mastering_max > 0.0f)
            at = view_colour_cat(buf, size, at, " %.5g/%.5g",
                                 c->mastering_min, c->mastering_max);
    }
    at = view_colour_cat(buf, size, at, ", intent %s", view_colour_intent_name(c->intent));
    (void)at;
}
