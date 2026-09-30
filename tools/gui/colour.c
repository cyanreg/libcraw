/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* The window's colour, the Wayland half (DESIGN.md 22).
 *
 * The surface asks wp_color_manager_v1 for its preferred image description,
 * colour_map.c decides what of it can be rendered, libplacebo renders into
 * exactly that and the surface is tagged with a parametric description
 * saying so. When the preference changes - the window moved to another
 * output, HDR switched on - the same happens again while the window keeps
 * running.
 *
 * Everything here runs on a wl_event_queue of its own, made off SDL's
 * wl_display: the manager is bound through a registry created on a proxy
 * wrapper of the display that sits on that queue, and every object made
 * from the manager inherits it. SDL, Mesa's WSI and this module then read
 * the one socket, each dispatching only its own queue, and none of them
 * sees the others' events. Opening waits on roundtrips of this queue; at
 * run time nothing waits: the loop dispatches what has arrived once per
 * iteration, and each step sends its requests and returns.
 *
 * The order at opening matters because a wl_surface can have only one
 * wp_color_management_surface_v1, and a second is a protocol error, which
 * ends the connection and with it the window. Mesa's Wayland WSI creates
 * one of its own for every swapchain colour space but pass-through (and,
 * on a compositor that still advertises the deprecated srgb curve, sRGB),
 * parented to the swapchain. So the swapchain is switched to pass-through
 * first - libplacebo does that for an unknown/unknown hint, recreating the
 * swapchain on the spot and destroying the retired one at once, and with
 * it anything Mesa had on the surface - and only then is ours created.
 * Every other request is guarded by the feature it needs for the same
 * reason: a missing guard is not a graceful failure but a dead window. */

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include "view.h"
#include "colour.h"

#if CINEVIEW_HAVE_WAYLAND_COLOUR

#include <wayland-client.h>
#include "color-management-v1-client-protocol.h"

/* colour.h repeats the protocol's numbers so that the resolver and its test
 * need no generated header; here is where they are held to them. */
#define VIEW_COLOUR_SAME(ours, theirs) \
    static_assert((int)(ours) == (int)(theirs), #ours " is not " #theirs)

VIEW_COLOUR_SAME(VIEW_WP_TF_BT1886, WP_COLOR_MANAGER_V1_TRANSFER_FUNCTION_BT1886);
VIEW_COLOUR_SAME(VIEW_WP_TF_GAMMA22, WP_COLOR_MANAGER_V1_TRANSFER_FUNCTION_GAMMA22);
VIEW_COLOUR_SAME(VIEW_WP_TF_GAMMA28, WP_COLOR_MANAGER_V1_TRANSFER_FUNCTION_GAMMA28);
VIEW_COLOUR_SAME(VIEW_WP_TF_ST240, WP_COLOR_MANAGER_V1_TRANSFER_FUNCTION_ST240);
VIEW_COLOUR_SAME(VIEW_WP_TF_EXT_LINEAR, WP_COLOR_MANAGER_V1_TRANSFER_FUNCTION_EXT_LINEAR);
VIEW_COLOUR_SAME(VIEW_WP_TF_LOG_100, WP_COLOR_MANAGER_V1_TRANSFER_FUNCTION_LOG_100);
VIEW_COLOUR_SAME(VIEW_WP_TF_LOG_316, WP_COLOR_MANAGER_V1_TRANSFER_FUNCTION_LOG_316);
VIEW_COLOUR_SAME(VIEW_WP_TF_XVYCC, WP_COLOR_MANAGER_V1_TRANSFER_FUNCTION_XVYCC);
VIEW_COLOUR_SAME(VIEW_WP_TF_SRGB, WP_COLOR_MANAGER_V1_TRANSFER_FUNCTION_SRGB);
VIEW_COLOUR_SAME(VIEW_WP_TF_EXT_SRGB, WP_COLOR_MANAGER_V1_TRANSFER_FUNCTION_EXT_SRGB);
VIEW_COLOUR_SAME(VIEW_WP_TF_ST2084_PQ, WP_COLOR_MANAGER_V1_TRANSFER_FUNCTION_ST2084_PQ);
VIEW_COLOUR_SAME(VIEW_WP_TF_ST428, WP_COLOR_MANAGER_V1_TRANSFER_FUNCTION_ST428);
VIEW_COLOUR_SAME(VIEW_WP_TF_HLG, WP_COLOR_MANAGER_V1_TRANSFER_FUNCTION_HLG);
VIEW_COLOUR_SAME(VIEW_WP_TF_COMPOUND_POWER_2_4, WP_COLOR_MANAGER_V1_TRANSFER_FUNCTION_COMPOUND_POWER_2_4);
VIEW_COLOUR_SAME(VIEW_WP_PRIM_SRGB, WP_COLOR_MANAGER_V1_PRIMARIES_SRGB);
VIEW_COLOUR_SAME(VIEW_WP_PRIM_PAL_M, WP_COLOR_MANAGER_V1_PRIMARIES_PAL_M);
VIEW_COLOUR_SAME(VIEW_WP_PRIM_PAL, WP_COLOR_MANAGER_V1_PRIMARIES_PAL);
VIEW_COLOUR_SAME(VIEW_WP_PRIM_NTSC, WP_COLOR_MANAGER_V1_PRIMARIES_NTSC);
VIEW_COLOUR_SAME(VIEW_WP_PRIM_GENERIC_FILM, WP_COLOR_MANAGER_V1_PRIMARIES_GENERIC_FILM);
VIEW_COLOUR_SAME(VIEW_WP_PRIM_BT2020, WP_COLOR_MANAGER_V1_PRIMARIES_BT2020);
VIEW_COLOUR_SAME(VIEW_WP_PRIM_CIE1931_XYZ, WP_COLOR_MANAGER_V1_PRIMARIES_CIE1931_XYZ);
VIEW_COLOUR_SAME(VIEW_WP_PRIM_DCI_P3, WP_COLOR_MANAGER_V1_PRIMARIES_DCI_P3);
VIEW_COLOUR_SAME(VIEW_WP_PRIM_DISPLAY_P3, WP_COLOR_MANAGER_V1_PRIMARIES_DISPLAY_P3);
VIEW_COLOUR_SAME(VIEW_WP_PRIM_ADOBE_RGB, WP_COLOR_MANAGER_V1_PRIMARIES_ADOBE_RGB);
VIEW_COLOUR_SAME(VIEW_WP_INTENT_PERCEPTUAL, WP_COLOR_MANAGER_V1_RENDER_INTENT_PERCEPTUAL);
VIEW_COLOUR_SAME(VIEW_WP_INTENT_RELATIVE, WP_COLOR_MANAGER_V1_RENDER_INTENT_RELATIVE);
VIEW_COLOUR_SAME(VIEW_WP_INTENT_SATURATION, WP_COLOR_MANAGER_V1_RENDER_INTENT_SATURATION);
VIEW_COLOUR_SAME(VIEW_WP_INTENT_ABSOLUTE, WP_COLOR_MANAGER_V1_RENDER_INTENT_ABSOLUTE);
VIEW_COLOUR_SAME(VIEW_WP_INTENT_RELATIVE_BPC, WP_COLOR_MANAGER_V1_RENDER_INTENT_RELATIVE_BPC);
VIEW_COLOUR_SAME(VIEW_WP_INTENT_ABSOLUTE_NO_ADAPTATION, WP_COLOR_MANAGER_V1_RENDER_INTENT_ABSOLUTE_NO_ADAPTATION);

/* The newest version this code speaks: ready2 and preferred_changed2 are
 * version 2's, and version 3 only adds the Windows scRGB and BT.2100
 * descriptions, which are nothing this program renders. */
#define VIEW_COLOUR_VERSION 2

/* Roundtrips an opening step may take before the compositor is deemed not
 * to answer. Every step here is answered within the first. */
#define VIEW_COLOUR_TRIES 4

/* Where a wp_image_description_v1 of the preferred kind is: asked for,
 * ready, failed; its information asked for and then all read. */
enum view_colour_state {
    VIEW_COLOUR_NONE,
    VIEW_COLOUR_WAITING,
    VIEW_COLOUR_READY,
    VIEW_COLOUR_FAILED,
    VIEW_COLOUR_INFO,
    VIEW_COLOUR_DONE,
};

struct view_wl_colour {
    bool verbose;
    /* The surface carries our description and the renderer targets it. */
    bool active;
    /* The swapchain came back as something other than pass-through: the
     * next dispatch takes everything down again. Set inside a frame, which
     * is no place to do it (see view_colour_target()). */
    bool lost;
    /* The target colour moved since the last dispatch returned. */
    bool changed;

    /* SDL's; borrowed, never destroyed here. */
    struct wl_display *display;
    struct wl_surface *surface;

    struct wl_event_queue *queue;
    struct wl_display *wrapper;
    struct wl_registry *registry;
    uint32_t manager_name, manager_version;

    struct wp_color_manager_v1 *manager;
    struct view_colour_caps caps;
    bool caps_done;

    struct wp_color_management_surface_feedback_v1 *feedback;
    struct wp_color_management_surface_v1 *surface_cm;

    /* The preferred description being asked about. Its identity de-duplicates
     * a preferred_changed that names the description already held; the last
     * announced identity one that repeats itself. A compositor that
     * numbers every copy afresh (wlroots does: "TODO: de-duplicate
     * identity") defeats both, and the comparison of the choices below is
     * then what keeps a repeat from re-tagging the surface. */
    struct wp_image_description_v1 *pref;
    enum view_colour_state pref_state;
    uint64_t pref_identity;
    uint64_t announced;
    bool refetch;
    uint32_t pref_cause;
    char pref_msg[128];

    struct wp_image_description_info_v1 *info;
    struct view_colour_desc info_desc;
    struct view_colour_desc preferred;
    const char *refused;        /* why the last preferred could not be rendered */

    /* Ours: the description on the surface and what it says, and the next
     * one while the compositor has not yet made it ready. */
    struct wp_image_description_v1 *desc;
    struct view_colour_choice choice;
    struct wp_image_description_v1 *pending;
    enum view_colour_state pending_state;
    struct view_colour_choice pending_choice;
    uint32_t pending_cause;
    char pending_msg[128];
};

static void view_colour_log(const struct view_wl_colour *c, const char *fmt, ...)
    craw_printf_format(2, 3);

static void view_colour_log(const struct view_wl_colour *c, const char *fmt, ...)
{
    va_list args;

    if (!c->verbose)
        return;

    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

static const char *view_colour_cause_name(uint32_t cause)
{
    switch (cause) {
    case WP_IMAGE_DESCRIPTION_V1_CAUSE_LOW_VERSION:      return "low_version";
    case WP_IMAGE_DESCRIPTION_V1_CAUSE_UNSUPPORTED:      return "unsupported";
    case WP_IMAGE_DESCRIPTION_V1_CAUSE_OPERATING_SYSTEM: return "operating_system";
    case WP_IMAGE_DESCRIPTION_V1_CAUSE_NO_OUTPUT:        return "no_output";
    default:                                             return "unknown";
    }
}

/* True, with the reason printed, once the connection has failed. A
 * protocol error is fatal to the whole connection, SDL's window included;
 * saying which request drew it is the most this can still do. */
static bool view_colour_broken(const struct view_wl_colour *c, const char *what)
{
    const struct wl_interface *iface = NULL;
    uint32_t id = 0, code;
    int err = wl_display_get_error(c->display);

    if (!err)
        return false;

    if (err == EPROTO) {
        code = wl_display_get_protocol_error(c->display, &iface, &id);
        fprintf(stderr, "colour: %s: protocol error %u on %s@%u, the Wayland "
                        "connection is gone\n", what, code,
                iface ? iface->name : "?", id);
    } else {
        fprintf(stderr, "colour: %s: the Wayland connection failed: %s\n",
                what, strerror(err));
    }

    return true;
}

static int view_colour_roundtrip(struct view_wl_colour *c, const char *what)
{
    if (wl_display_roundtrip_queue(c->display, c->queue) < 0) {
        if (!view_colour_broken(c, what))
            fprintf(stderr, "colour: %s: the roundtrip failed\n", what);
        return CRAW_ERROR(EPROTO);
    }

    return view_colour_broken(c, what) ? CRAW_ERROR(EPROTO) : 0;
}

/* ------------------------------------------------------------ listeners */

static void view_colour_global(void *data, struct wl_registry *registry,
                               uint32_t name, const char *interface,
                               uint32_t version)
{
    struct view_wl_colour *c = data;

    (void)registry;
    if (!strcmp(interface, wp_color_manager_v1_interface.name)) {
        c->manager_name = name;
        c->manager_version = version;
    }
}

/* The protocol asks compositors never to remove the manager, and nothing
 * else of the registry is used. */
static void view_colour_global_remove(void *data, struct wl_registry *registry,
                                      uint32_t name)
{
    (void)data;
    (void)registry;
    (void)name;
}

static const struct wl_registry_listener view_colour_registry_listener = {
    .global        = view_colour_global,
    .global_remove = view_colour_global_remove,
};

static void view_colour_intent(void *data, struct wp_color_manager_v1 *m,
                               uint32_t intent)
{
    struct view_wl_colour *c = data;

    (void)m;
    if (intent < 32)
        c->caps.intents |= 1u << intent;
}

static void view_colour_feature(void *data, struct wp_color_manager_v1 *m,
                                uint32_t feature)
{
    struct view_wl_colour *c = data;

    (void)m;
    switch (feature) {
    case WP_COLOR_MANAGER_V1_FEATURE_PARAMETRIC:
        c->caps.parametric = true;
        break;
    case WP_COLOR_MANAGER_V1_FEATURE_SET_PRIMARIES:
        c->caps.set_primaries = true;
        break;
    case WP_COLOR_MANAGER_V1_FEATURE_SET_TF_POWER:
        c->caps.set_tf_power = true;
        break;
    case WP_COLOR_MANAGER_V1_FEATURE_SET_LUMINANCES:
        c->caps.set_luminances = true;
        break;
    case WP_COLOR_MANAGER_V1_FEATURE_SET_MASTERING_DISPLAY_PRIMARIES:
        c->caps.mastering_display_primaries = true;
        break;
    case WP_COLOR_MANAGER_V1_FEATURE_EXTENDED_TARGET_VOLUME:
        c->caps.extended_target_volume = true;
        break;
    default:
        break;
    }
}

static void view_colour_tf(void *data, struct wp_color_manager_v1 *m, uint32_t tf)
{
    struct view_wl_colour *c = data;

    (void)m;
    if (tf < 64)
        c->caps.tf_named |= 1ull << tf;
}

static void view_colour_prim(void *data, struct wp_color_manager_v1 *m, uint32_t prim)
{
    struct view_wl_colour *c = data;

    (void)m;
    if (prim < 64)
        c->caps.primaries_named |= 1ull << prim;
}

static void view_colour_caps_done(void *data, struct wp_color_manager_v1 *m)
{
    struct view_wl_colour *c = data;

    (void)m;
    c->caps_done = true;
}

static const struct wp_color_manager_v1_listener view_colour_manager_listener = {
    .supported_intent          = view_colour_intent,
    .supported_feature         = view_colour_feature,
    .supported_tf_named        = view_colour_tf,
    .supported_primaries_named = view_colour_prim,
    .done                      = view_colour_caps_done,
};

/* A new preferred description: asked for at the next step, unless it is
 * the one already held or the one last announced. */
static void view_colour_announce(struct view_wl_colour *c, uint64_t identity)
{
    if (identity && (identity == c->pref_identity || identity == c->announced))
        return;

    c->announced = identity;
    c->refetch = true;
}

static void view_colour_changed(void *data,
                                struct wp_color_management_surface_feedback_v1 *f,
                                uint32_t identity)
{
    (void)f;
    view_colour_announce(data, identity);
}

static void view_colour_changed2(void *data,
                                 struct wp_color_management_surface_feedback_v1 *f,
                                 uint32_t hi, uint32_t lo)
{
    (void)f;
    view_colour_announce(data, (uint64_t)hi << 32 | lo);
}

static const struct wp_color_management_surface_feedback_v1_listener
view_colour_feedback_listener = {
    .preferred_changed  = view_colour_changed,
    .preferred_changed2 = view_colour_changed2,
};

/* The preferred description's events. Only the current object's count: a
 * replaced one has been destroyed, and libwayland drops what was still on
 * its way to it, but the check costs nothing. */
static void view_colour_pref_failed(void *data, struct wp_image_description_v1 *d,
                                    uint32_t cause, const char *msg)
{
    struct view_wl_colour *c = data;

    if (d != c->pref)
        return;
    c->pref_state = VIEW_COLOUR_FAILED;
    c->pref_cause = cause;
    snprintf(c->pref_msg, sizeof(c->pref_msg), "%s", msg ? msg : "");
}

static void view_colour_pref_ready2(void *data, struct wp_image_description_v1 *d,
                                    uint32_t hi, uint32_t lo)
{
    struct view_wl_colour *c = data;

    if (d != c->pref)
        return;
    c->pref_state = VIEW_COLOUR_READY;
    c->pref_identity = (uint64_t)hi << 32 | lo;
}

static void view_colour_pref_ready(void *data, struct wp_image_description_v1 *d,
                                   uint32_t identity)
{
    view_colour_pref_ready2(data, d, 0, identity);
}

static const struct wp_image_description_v1_listener view_colour_pref_listener = {
    .failed = view_colour_pref_failed,
    .ready  = view_colour_pref_ready,
    .ready2 = view_colour_pref_ready2,
};

/* The same for ours. */
static void view_colour_desc_failed(void *data, struct wp_image_description_v1 *d,
                                    uint32_t cause, const char *msg)
{
    struct view_wl_colour *c = data;

    if (d != c->pending)
        return;
    c->pending_state = VIEW_COLOUR_FAILED;
    c->pending_cause = cause;
    snprintf(c->pending_msg, sizeof(c->pending_msg), "%s", msg ? msg : "");
}

static void view_colour_desc_ready2(void *data, struct wp_image_description_v1 *d,
                                    uint32_t hi, uint32_t lo)
{
    struct view_wl_colour *c = data;

    (void)hi;
    (void)lo;
    if (d != c->pending)
        return;
    c->pending_state = VIEW_COLOUR_READY;
}

static void view_colour_desc_ready(void *data, struct wp_image_description_v1 *d,
                                   uint32_t identity)
{
    view_colour_desc_ready2(data, d, 0, identity);
}

static const struct wp_image_description_v1_listener view_colour_desc_listener = {
    .failed = view_colour_desc_failed,
    .ready  = view_colour_desc_ready,
    .ready2 = view_colour_desc_ready2,
};

/* The preferred description's information, one event per property. */
static struct pl_raw_primaries view_colour_xy(int32_t rx, int32_t ry, int32_t gx,
                                              int32_t gy, int32_t bx, int32_t by,
                                              int32_t wx, int32_t wy)
{
    return (struct pl_raw_primaries) {
        .red   = { rx / 1e6f, ry / 1e6f },
        .green = { gx / 1e6f, gy / 1e6f },
        .blue  = { bx / 1e6f, by / 1e6f },
        .white = { wx / 1e6f, wy / 1e6f },
    };
}

static void view_colour_info_icc(void *data, struct wp_image_description_info_v1 *i,
                                 int32_t fd, uint32_t size)
{
    (void)data;
    (void)i;
    (void)size;
    /* Parametric descriptions carry none; should one come anyway, the fd
     * is ours to close. */
    close(fd);
}

static void view_colour_info_prim(void *data, struct wp_image_description_info_v1 *i,
                                  int32_t rx, int32_t ry, int32_t gx, int32_t gy,
                                  int32_t bx, int32_t by, int32_t wx, int32_t wy)
{
    struct view_wl_colour *c = data;

    (void)i;
    c->info_desc.primaries = view_colour_xy(rx, ry, gx, gy, bx, by, wx, wy);
}

static void view_colour_info_prim_named(void *data, struct wp_image_description_info_v1 *i,
                                        uint32_t prim)
{
    struct view_wl_colour *c = data;

    (void)i;
    c->info_desc.primaries_named = (int)prim;
}

static void view_colour_info_power(void *data, struct wp_image_description_info_v1 *i,
                                   uint32_t eexp)
{
    struct view_wl_colour *c = data;

    (void)i;
    c->info_desc.tf_power = (float)eexp / 10000.0f;
}

static void view_colour_info_tf(void *data, struct wp_image_description_info_v1 *i,
                                uint32_t tf)
{
    struct view_wl_colour *c = data;

    (void)i;
    c->info_desc.tf_named = (int)tf;
}

static void view_colour_info_lum(void *data, struct wp_image_description_info_v1 *i,
                                 uint32_t min, uint32_t max, uint32_t ref)
{
    struct view_wl_colour *c = data;

    (void)i;
    c->info_desc.min_lum = (float)min / 10000.0f;
    c->info_desc.max_lum = (float)max;
    c->info_desc.ref_lum = (float)ref;
}

static void view_colour_info_target_prim(void *data, struct wp_image_description_info_v1 *i,
                                         int32_t rx, int32_t ry, int32_t gx, int32_t gy,
                                         int32_t bx, int32_t by, int32_t wx, int32_t wy)
{
    struct view_wl_colour *c = data;

    (void)i;
    c->info_desc.target = view_colour_xy(rx, ry, gx, gy, bx, by, wx, wy);
}

static void view_colour_info_target_lum(void *data, struct wp_image_description_info_v1 *i,
                                        uint32_t min, uint32_t max)
{
    struct view_wl_colour *c = data;

    (void)i;
    c->info_desc.target_min = (float)min / 10000.0f;
    c->info_desc.target_max = (float)max;
}

static void view_colour_info_cll(void *data, struct wp_image_description_info_v1 *i,
                                 uint32_t cll)
{
    struct view_wl_colour *c = data;

    (void)i;
    c->info_desc.max_cll = (float)cll;
}

static void view_colour_info_fall(void *data, struct wp_image_description_info_v1 *i,
                                  uint32_t fall)
{
    struct view_wl_colour *c = data;

    (void)i;
    c->info_desc.max_fall = (float)fall;
}

/* A destructor event: the compositor is done with the object, and the
 * client frees its proxy. */
static void view_colour_info_done(void *data, struct wp_image_description_info_v1 *i)
{
    struct view_wl_colour *c = data;

    wp_image_description_info_v1_destroy(i);
    if (i != c->info)
        return;
    c->info = NULL;
    c->pref_state = VIEW_COLOUR_DONE;
}

static const struct wp_image_description_info_v1_listener view_colour_info_listener = {
    .done             = view_colour_info_done,
    .icc_file         = view_colour_info_icc,
    .primaries        = view_colour_info_prim,
    .primaries_named  = view_colour_info_prim_named,
    .tf_power         = view_colour_info_power,
    .tf_named         = view_colour_info_tf,
    .luminances       = view_colour_info_lum,
    .target_primaries = view_colour_info_target_prim,
    .target_luminance = view_colour_info_target_lum,
    .target_max_cll   = view_colour_info_cll,
    .target_max_fall  = view_colour_info_fall,
};

/* ------------------------------------------------------------- requests */

static uint32_t view_colour_min_lum(float v)
{
    return (uint32_t)lroundf(v * 10000.0f);
}

static uint32_t view_colour_lum(float v)
{
    return (uint32_t)lroundf(v);
}

static int32_t view_colour_coord(float v)
{
    return (int32_t)lroundf(v * 1e6f);
}

/* Whether a maximum or reference luminance is above a minimum, as the
 * compositor will judge it: on the integers sent. */
static bool view_colour_above(float min, float above)
{
    return (uint64_t)view_colour_lum(above) * 10000 > view_colour_min_lum(min);
}

/* Everything the creator will be asked, checked against what the manager
 * advertised before a single request goes out: a property skipped half way
 * would leave the set incomplete, and create() on an incomplete set is a
 * protocol error. colour_map.c only ever chooses what passes this; this is
 * the guard, not the decision. */
static const char *view_colour_unsendable(const struct view_colour_caps *caps,
                                          const struct view_colour_choice *ch)
{
    if (!caps->parametric)
        return "no parametric descriptions";
    if (ch->primaries_named <= 0 || ch->primaries_named >= 64 ||
        !(caps->primaries_named & (1ull << ch->primaries_named)))
        return "primaries not advertised";
    if (ch->tf_named) {
        if (ch->tf_named < 0 || ch->tf_named >= 64 ||
            !(caps->tf_named & (1ull << ch->tf_named)))
            return "transfer function not advertised";
    } else if (!caps->set_tf_power || !(ch->tf_power >= 1.0f && ch->tf_power <= 10.0f)) {
        return "power curve not settable";
    }
    if (ch->set_luminances &&
        (!caps->set_luminances || !view_colour_above(ch->min_lum, ch->max_lum) ||
         !view_colour_above(ch->min_lum, ch->ref_lum)))
        return "luminances not settable";
    if (ch->set_mastering &&
        (!caps->mastering_display_primaries ||
         (ch->mastering_max > 0.0f &&
          !view_colour_above(ch->mastering_min, ch->mastering_max))))
        return "target volume not settable";
    if (ch->intent < 0 || ch->intent >= 32 || !(caps->intents & (1u << ch->intent)))
        return "intent not advertised";

    return NULL;
}

/* Makes the description for `ch`, replacing one still in the making. Ready
 * or failed arrives later; view_colour_step() takes it from there. */
static int view_colour_create(struct view_wl_colour *c, const struct view_colour_choice *ch)
{
    struct wp_image_description_creator_params_v1 *cr;
    const char *why = view_colour_unsendable(&c->caps, ch);
    const struct pl_raw_primaries *m = &ch->mastering;

    if (why) {
        fprintf(stderr, "colour: not sending the description: %s\n", why);
        return CRAW_ERROR(EINVAL);
    }

    if (c->pending) {
        wp_image_description_v1_destroy(c->pending);
        c->pending = NULL;
        c->pending_state = VIEW_COLOUR_NONE;
    }

    cr = wp_color_manager_v1_create_parametric_creator(c->manager);
    if (!cr)
        return CRAW_ERROR(ENOMEM);

    wp_image_description_creator_params_v1_set_primaries_named(cr, (uint32_t)ch->primaries_named);
    if (ch->tf_named)
        wp_image_description_creator_params_v1_set_tf_named(cr, (uint32_t)ch->tf_named);
    else
        wp_image_description_creator_params_v1_set_tf_power(cr,
            (uint32_t)lroundf(ch->tf_power * 10000.0f));

    if (ch->set_luminances)
        wp_image_description_creator_params_v1_set_luminances(cr,
            view_colour_min_lum(ch->min_lum), view_colour_lum(ch->max_lum),
            view_colour_lum(ch->ref_lum));

    /* Never max_cll or max_fall: this is a window, not a mastered stream,
     * and the picture's peak changes with every frame and every grade. */
    if (ch->set_mastering) {
        wp_image_description_creator_params_v1_set_mastering_display_primaries(cr,
            view_colour_coord(m->red.x),   view_colour_coord(m->red.y),
            view_colour_coord(m->green.x), view_colour_coord(m->green.y),
            view_colour_coord(m->blue.x),  view_colour_coord(m->blue.y),
            view_colour_coord(m->white.x), view_colour_coord(m->white.y));
        if (ch->mastering_max > 0.0f)
            wp_image_description_creator_params_v1_set_mastering_luminance(cr,
                view_colour_min_lum(ch->mastering_min),
                view_colour_lum(ch->mastering_max));
    }

    /* create is the creator's destructor. */
    c->pending = wp_image_description_creator_params_v1_create(cr);
    if (!c->pending)
        return CRAW_ERROR(ENOMEM);

    wp_image_description_v1_add_listener(c->pending, &view_colour_desc_listener, c);
    c->pending_state = VIEW_COLOUR_WAITING;
    c->pending_choice = *ch;

    return 0;
}

static void view_colour_drop_pref(struct view_wl_colour *c)
{
    if (c->info) {
        wp_image_description_info_v1_destroy(c->info);
        c->info = NULL;
    }
    if (c->pref) {
        wp_image_description_v1_destroy(c->pref);
        c->pref = NULL;
    }
    c->pref_state = VIEW_COLOUR_NONE;
}

static void view_colour_describe(const struct view_wl_colour *c, const char *what)
{
    char caps[384], pref[256], choice[512];

    if (!c->verbose)
        return;

    view_colour_describe_caps(&c->caps, caps, sizeof(caps));
    view_colour_describe_desc(&c->preferred, pref, sizeof(pref));
    view_colour_describe_choice(&c->choice, choice, sizeof(choice));
    fprintf(stderr, "colour: %s: wp_color_manager_v1 %s; preferred %s (identity %" PRIu64 "); "
                    "rendering %s\n", what, caps, pref, c->pref_identity, choice);
}

/* Advances whatever is in flight as far as the events so far allow. Sends
 * requests and never waits for an answer, so the run loop can call it every
 * iteration; opening drives it with roundtrips in between. */
static void view_colour_step(struct view_wl_colour *c)
{
    bool sent = false;

    /* A new preferred description was announced: ask for it, dropping
     * whatever was still being asked about the last one. */
    if (c->refetch) {
        c->refetch = false;
        view_colour_drop_pref(c);
        c->pref = wp_color_management_surface_feedback_v1_get_preferred_parametric(c->feedback);
        if (c->pref) {
            wp_image_description_v1_add_listener(c->pref, &view_colour_pref_listener, c);
            c->pref_state = VIEW_COLOUR_WAITING;
        }
        sent = true;
    }

    if (c->pref_state == VIEW_COLOUR_FAILED) {
        fprintf(stderr, "colour: the compositor has no preferred description to "
                        "give (%s: %s); the window keeps the one it has\n",
                view_colour_cause_name(c->pref_cause), c->pref_msg);
        view_colour_drop_pref(c);
    }

    if (c->pref_state == VIEW_COLOUR_READY) {
        c->info_desc = (struct view_colour_desc) { 0 };
        c->info = wp_image_description_v1_get_information(c->pref);
        if (c->info) {
            wp_image_description_info_v1_add_listener(c->info, &view_colour_info_listener, c);
            c->pref_state = VIEW_COLOUR_INFO;
        } else {
            c->pref_state = VIEW_COLOUR_NONE;
        }
        sent = true;
    }

    /* All of it read: decide, and make a description unless the decision
     * is the one the surface already carries or is about to. The preferred
     * object is kept until the next one replaces it. */
    if (c->pref_state == VIEW_COLOUR_DONE) {
        struct view_colour_choice ch;
        const struct view_colour_choice *have = c->pending ? &c->pending_choice
                                              : c->desc    ? &c->choice : NULL;

        c->pref_state = VIEW_COLOUR_NONE;
        c->preferred = c->info_desc;

        if (view_colour_resolve(&c->caps, &c->preferred, &ch) < 0) {
            char pref[256];

            /* Opening says so itself, and stays untagged. */
            c->refused = ch.why;
            view_colour_describe_desc(&c->preferred, pref, sizeof(pref));
            if (c->active)
                fprintf(stderr, "colour: the compositor now prefers %s, which cannot "
                                "be rendered (%s); the window keeps the description "
                                "it has\n", pref, ch.why ? ch.why : "?");
        } else if (have && view_colour_choice_equal(&ch, have)) {
            view_colour_log(c, "colour: preferred description %" PRIu64 " changes "
                               "nothing\n", c->pref_identity);
        } else if (view_colour_create(c, &ch) >= 0) {
            sent = true;
        }
    }

    if (c->pending_state == VIEW_COLOUR_FAILED) {
        fprintf(stderr, "colour: the compositor refused the description (%s: %s); "
                        "the window keeps the one it has\n",
                view_colour_cause_name(c->pending_cause), c->pending_msg);
        wp_image_description_v1_destroy(c->pending);
        c->pending = NULL;
        c->pending_state = VIEW_COLOUR_NONE;
        sent = true;
    }

    /* Ready, and there is a surface to put it on (not yet while opening).
     * set_image_description is double-buffered: it lands with the next
     * wl_surface.commit, which is normally the present of the frame the
     * caller renders next against the new target, so the new colour and
     * the new tag arrive together. A commit of SDL's in between - a
     * configure being acknowledged - would show one frame of the old
     * picture under the new tag; that is the price of not owning the
     * commits, and one frame is all it can cost. Setting copies, so the
     * old description can go at once. */
    if (c->pending_state == VIEW_COLOUR_READY && c->surface_cm) {
        wp_color_management_surface_v1_set_image_description(c->surface_cm, c->pending,
                                                              (uint32_t)c->pending_choice.intent);
        if (c->desc)
            wp_image_description_v1_destroy(c->desc);
        c->desc = c->pending;
        c->choice = c->pending_choice;
        c->pending = NULL;
        c->pending_state = VIEW_COLOUR_NONE;
        c->changed = true;
        sent = true;
        if (c->active)
            view_colour_describe(c, "changed");
    }

    if (sent)
        wl_display_flush(c->display);
}

/* Whether the surface offers a pass-through format at all. Mesa always
 * does, and libplacebo only picks one when asked; a driver that has none
 * would leave the swapchain as it is and the surface tagged for a picture
 * it is not getting, so the module does not start there. */
static bool view_colour_pass_through(const struct view_gpu *g)
{
    VkSurfaceFormatKHR *fmts;
    uint32_t nb = 0;
    bool found = false;
    VkResult res;

    res = vkGetPhysicalDeviceSurfaceFormatsKHR(g->vk->phys_device, g->surface, &nb, NULL);
    if (res != VK_SUCCESS || !nb)
        return false;

    fmts = calloc(nb, sizeof(*fmts));
    if (!fmts)
        return false;

    res = vkGetPhysicalDeviceSurfaceFormatsKHR(g->vk->phys_device, g->surface, &nb, fmts);
    for (uint32_t i = 0; (res == VK_SUCCESS || res == VK_INCOMPLETE) && i < nb; i++)
        found |= fmts[i].colorSpace == VK_COLOR_SPACE_PASS_THROUGH_EXT;

    free(fmts);

    return found;
}

/* ---------------------------------------------------------- entry points */

int view_colour_open(struct view_gpu *g, void *sdl_window, bool verbose)
{
    SDL_Window *win = sdl_window;
    SDL_PropertiesID props;
    struct wl_display *dpy;
    struct wl_surface *surf;
    struct view_wl_colour *c;
    const char *why = NULL;
    int ret;

    if (!win || !g->sw)
        return 0;

    /* NULL on X11, XWayland included, which leaves the window untagged sRGB
     * exactly as before. */
    props = SDL_GetWindowProperties(win);
    dpy = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER, NULL);
    surf = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, NULL);
    if (!dpy || !surf) {
        if (verbose)
            fprintf(stderr, "colour: not a Wayland window, the window stays sRGB\n");
        return 0;
    }

    if (!view_colour_pass_through(g)) {
        if (verbose)
            fprintf(stderr, "colour: the surface has no pass-through format, the "
                            "window stays sRGB\n");
        return 0;
    }

    c = calloc(1, sizeof(*c));
    if (!c)
        return CRAW_ERROR(ENOMEM);
    g->wl_colour = c;
    c->verbose = verbose;
    c->display = dpy;
    c->surface = surf;

#if WAYLAND_VERSION_MAJOR > 1 || WAYLAND_VERSION_MINOR >= 23
    c->queue = wl_display_create_queue_with_name(dpy, "cineview colour");
#else
    c->queue = wl_display_create_queue(dpy);
#endif
    c->wrapper = wl_proxy_create_wrapper(dpy);
    if (!c->queue || !c->wrapper)
        goto nomem;
    wl_proxy_set_queue((struct wl_proxy *)c->wrapper, c->queue);

    c->registry = wl_display_get_registry(c->wrapper);
    if (!c->registry)
        goto nomem;
    wl_registry_add_listener(c->registry, &view_colour_registry_listener, c);
    if (view_colour_roundtrip(c, "registry") < 0)
        goto inactive;
    if (!c->manager_name) {
        why = "the compositor has no wp_color_manager_v1";
        goto inactive;
    }

    /* Version 2 at most: see VIEW_COLOUR_VERSION. The features, curves,
     * primaries and intents arrive right after the bind, and done closes
     * them. */
    c->caps.version = CRAW_MIN(c->manager_version, VIEW_COLOUR_VERSION);
    c->manager = wl_registry_bind(c->registry, c->manager_name,
                                  &wp_color_manager_v1_interface, c->caps.version);
    if (!c->manager)
        goto nomem;
    wp_color_manager_v1_add_listener(c->manager, &view_colour_manager_listener, c);
    for (int i = 0; i < VIEW_COLOUR_TRIES && !c->caps_done; i++)
        if (view_colour_roundtrip(c, "wp_color_manager_v1") < 0)
            goto inactive;
    if (!c->caps_done) {
        why = "wp_color_manager_v1 never said done";
        goto inactive;
    }
    if (!c->caps.parametric) {
        why = "the compositor takes no parametric descriptions";
        goto inactive;
    }

    /* A feedback object of our own; SDL has one on the same surface for its
     * HDR window properties, which the protocol allows in any number. */
    c->feedback = wp_color_manager_v1_get_surface_feedback(c->manager, surf);
    if (!c->feedback)
        goto nomem;
    wp_color_management_surface_feedback_v1_add_listener(c->feedback,
                                                         &view_colour_feedback_listener, c);

    /* The preferred description, and then everything it says. */
    c->refetch = true;
    view_colour_step(c);
    for (int i = 0; i < VIEW_COLOUR_TRIES && c->pref_state == VIEW_COLOUR_WAITING; i++)
        if (view_colour_roundtrip(c, "get_preferred_parametric") < 0)
            goto inactive;
    if (c->pref_state == VIEW_COLOUR_FAILED) {
        fprintf(stderr, "colour: the compositor has no preferred description to give "
                        "(%s: %s), the window stays sRGB\n",
                view_colour_cause_name(c->pref_cause), c->pref_msg);
        goto inactive;
    }
    if (c->pref_state != VIEW_COLOUR_READY) {
        why = "no answer to get_preferred_parametric";
        goto inactive;
    }

    view_colour_step(c);
    for (int i = 0; i < VIEW_COLOUR_TRIES && c->pref_state == VIEW_COLOUR_INFO; i++)
        if (view_colour_roundtrip(c, "get_information") < 0)
            goto inactive;
    if (c->pref_state != VIEW_COLOUR_DONE) {
        why = "the preferred description's information never ended";
        goto inactive;
    }

    /* The decision, and the description that says it. */
    view_colour_step(c);
    if (!c->pending) {
        char pref[256];

        view_colour_describe_desc(&c->preferred, pref, sizeof(pref));
        fprintf(stderr, "colour: nothing the compositor accepts can be rendered for "
                        "its preferred %s (%s), the window stays sRGB\n", pref,
                c->refused ? c->refused : "the description could not be sent");
        goto inactive;
    }
    for (int i = 0; i < VIEW_COLOUR_TRIES && c->pending_state == VIEW_COLOUR_WAITING; i++)
        if (view_colour_roundtrip(c, "create") < 0)
            goto inactive;
    if (c->pending_state == VIEW_COLOUR_FAILED) {
        fprintf(stderr, "colour: the compositor refused the description (%s: %s), "
                        "the window stays sRGB\n",
                view_colour_cause_name(c->pending_cause), c->pending_msg);
        goto inactive;
    }
    if (c->pending_state != VIEW_COLOUR_READY) {
        why = "no answer to create";
        goto inactive;
    }

    /* Only now the swapchain: the description exists and is ready, so
     * nothing after this point can refuse, and a refusal before it has
     * left the window as it was. Pass-through first, which takes Mesa's
     * own colour surface down with the old swapchain; then ours. This is
     * outside any frame, as it has to be: the hint takes the swapchain's
     * lock, which pl_swapchain_start_frame() holds until the submit. */
    pl_swapchain_colorspace_hint(g->sw, &(struct pl_color_space) { 0 });

    c->surface_cm = wp_color_manager_v1_get_surface(c->manager, surf);
    if (!c->surface_cm)
        goto nomem;
    view_colour_step(c);
    if (view_colour_broken(c, "set_image_description"))
        goto inactive;

    c->active = true;
    c->changed = false;
    view_colour_describe(c, "on");

    return 0;

inactive:
    if (why && verbose)
        fprintf(stderr, "colour: %s, the window stays sRGB\n", why);
    view_colour_close(g);
    return 0;

nomem:
    ret = CRAW_ERROR(ENOMEM);
    view_colour_close(g);
    return ret;
}

bool view_colour_dispatch(struct view_gpu *g)
{
    struct view_wl_colour *c = g->wl_colour;
    bool changed;

    if (!c)
        return false;

    /* The swapchain never became pass-through. The surface goes back to
     * untagged - destroying the surface object unsets its description -
     * and the swapchain is left as it is, which is what it was before. */
    if (c->lost) {
        view_colour_close(g);
        return true;
    }

    if (wl_display_dispatch_queue_pending(c->display, c->queue) < 0 ||
        view_colour_broken(c, "dispatch")) {
        view_colour_close(g);
        return false;
    }

    view_colour_step(c);

    changed = c->changed;
    c->changed = false;

    return changed;
}

bool view_colour_active(const struct view_gpu *g)
{
    return g->wl_colour && g->wl_colour->active && !g->wl_colour->lost;
}

void view_colour_target(struct view_gpu *g, struct pl_color_space *csp,
                        struct pl_bit_encoding *bits)
{
    struct view_wl_colour *c = g->wl_colour;

    if (!c || !c->active || c->lost)
        return;

    /* Pass-through comes back from libplacebo as unknown/unknown, which is
     * the whole point: the caller says what the image holds. Anything else
     * is a swapchain that did not take the hint. Taking the module down
     * means hinting the swapchain and destroying protocol objects, and this
     * runs between pl_swapchain_start_frame() and the submit, with the
     * swapchain's lock held; so the frame is rendered as the swapchain
     * says and the rest waits for the next dispatch. */
    if (csp->primaries != PL_COLOR_PRIM_UNKNOWN || csp->transfer != PL_COLOR_TRC_UNKNOWN) {
        fprintf(stderr, "colour: the swapchain is not pass-through, the window "
                        "goes back to untagged sRGB\n");
        c->lost = true;
        return;
    }

    /* The image is 16-bit UNORM (pass-through's best format), but what
     * reaches the screen is the compositor's framebuffer, eight bits for
     * an SDR output and ten for HDR, into which it rounds without
     * dithering. So the renderer is told the picture holds that many bits
     * in sixteen: it dithers to that grid and floors onto it, and the
     * sixteen-bit texel holds an exact eight- or ten-bit code that the
     * compositor's rounding reproduces. */
    *csp = c->choice.target;
    *bits = (struct pl_bit_encoding) {
        .sample_depth = c->choice.depth,
        .color_depth  = c->choice.depth,
        .bit_shift    = 0,
    };
}

void view_colour_overlay(const struct view_gpu *g, struct pl_color_space *csp)
{
    const struct view_wl_colour *c = g->wl_colour;

    if (!c || !c->active || c->lost) {
        *csp = pl_color_space_srgb;
        return;
    }

    /* The panel's code values are drawn for an sRGB-ish display with its
     * white at the display's white. On an SDR target that is the target
     * itself, so they pass straight through as they did before; only a
     * target with headroom above its reference white has its peak brought
     * down to that white, where an SDR element belongs. On an HDR target
     * the panel is what the compositor would make of an SDR window there:
     * BT.709 gamma 2.2 with 1.0 at 203 cd/m². */
    if (pl_color_transfer_is_hdr(c->choice.target.transfer)) {
        *csp = (struct pl_color_space) {
            .primaries = PL_COLOR_PRIM_BT_709,
            .transfer  = PL_COLOR_TRC_GAMMA22,
        };
        return;
    }

    *csp = c->choice.target;
    if (csp->hdr.max_luma > PL_COLOR_SDR_WHITE)
        csp->hdr.max_luma = PL_COLOR_SDR_WHITE;
}

void view_colour_close(struct view_gpu *g)
{
    struct view_wl_colour *c = g->wl_colour;

    if (!c)
        return;

    /* Destroying the surface object also unsets its description. */
    if (c->surface_cm)
        wp_color_management_surface_v1_destroy(c->surface_cm);
    if (c->pending)
        wp_image_description_v1_destroy(c->pending);
    if (c->desc)
        wp_image_description_v1_destroy(c->desc);
    view_colour_drop_pref(c);
    if (c->feedback)
        wp_color_management_surface_feedback_v1_destroy(c->feedback);
    if (c->manager)
        wp_color_manager_v1_destroy(c->manager);
    if (c->registry)
        wl_registry_destroy(c->registry);
    if (c->wrapper)
        wl_proxy_wrapper_destroy(c->wrapper);
    if (c->queue)
        wl_event_queue_destroy(c->queue);
    if (!wl_display_get_error(c->display))
        wl_display_flush(c->display);

    if (c->active)
        view_colour_log(c, "colour: off, the window is untagged sRGB\n");

    free(c);
    g->wl_colour = NULL;
}

#else /* !CINEVIEW_HAVE_WAYLAND_COLOUR */

/* Built without wayland-client, wayland-protocols or wayland-scanner: the
 * window is untagged sRGB, as it always was. */

int view_colour_open(struct view_gpu *g, void *sdl_window, bool verbose)
{
    (void)g;
    (void)sdl_window;
    if (verbose)
        fprintf(stderr, "colour: built without Wayland colour management, the "
                        "window stays sRGB\n");
    return 0;
}

bool view_colour_dispatch(struct view_gpu *g)
{
    (void)g;
    return false;
}

bool view_colour_active(const struct view_gpu *g)
{
    (void)g;
    return false;
}

void view_colour_target(struct view_gpu *g, struct pl_color_space *csp,
                        struct pl_bit_encoding *bits)
{
    (void)g;
    (void)csp;
    (void)bits;
}

void view_colour_overlay(const struct view_gpu *g, struct pl_color_space *csp)
{
    (void)g;
    *csp = pl_color_space_srgb;
}

void view_colour_close(struct view_gpu *g)
{
    (void)g;
}

#endif /* CINEVIEW_HAVE_WAYLAND_COLOUR */
