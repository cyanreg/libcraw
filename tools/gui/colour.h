/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#ifndef CINEVIEW_COLOUR_H
#define CINEVIEW_COLOUR_H

/* The window's colour (DESIGN.md 22).
 *
 * On a Wayland compositor with wp_color_manager_v1 the window is not an
 * untagged sRGB surface any more: it asks the compositor which image
 * description the surface would best be in, renders libplacebo's target
 * into exactly that, and tags the surface with a parametric description
 * saying so, so that the compositor has nothing left to convert.
 *
 * This header has two halves. The first is pure: the protocol's numbers,
 * what was read from the compositor, and view_colour_resolve(), which
 * decides what to render and what to tell the compositor. It needs nothing
 * but libplacebo's colour space types, so tests/test_colour.c builds
 * colour_map.c on its own. The second is the window's side, colour.c,
 * which speaks the protocol; it is declared here against an incomplete
 * struct view_gpu so that the pure half never sees SDL or Wayland. */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <libplacebo/colorspace.h>

/* wp_color_manager_v1's enums, by the protocol's own numbers. They are
 * repeated here rather than taken from the generated header so that the
 * resolver and its test need no wayland-scanner; colour.c checks every one
 * of them against the generated values at compile time. */
enum view_wp_tf {
    VIEW_WP_TF_BT1886             = 1,
    VIEW_WP_TF_GAMMA22            = 2,
    VIEW_WP_TF_GAMMA28            = 3,
    VIEW_WP_TF_ST240              = 4,
    VIEW_WP_TF_EXT_LINEAR         = 5,
    VIEW_WP_TF_LOG_100            = 6,
    VIEW_WP_TF_LOG_316            = 7,
    VIEW_WP_TF_XVYCC              = 8,
    VIEW_WP_TF_SRGB               = 9,
    VIEW_WP_TF_EXT_SRGB           = 10,
    VIEW_WP_TF_ST2084_PQ          = 11,
    VIEW_WP_TF_ST428              = 12,
    VIEW_WP_TF_HLG                = 13,
    VIEW_WP_TF_COMPOUND_POWER_2_4 = 14,
};

enum view_wp_primaries {
    VIEW_WP_PRIM_SRGB             = 1,
    VIEW_WP_PRIM_PAL_M            = 2,
    VIEW_WP_PRIM_PAL              = 3,
    VIEW_WP_PRIM_NTSC             = 4,
    VIEW_WP_PRIM_GENERIC_FILM     = 5,
    VIEW_WP_PRIM_BT2020           = 6,
    VIEW_WP_PRIM_CIE1931_XYZ      = 7,
    VIEW_WP_PRIM_DCI_P3           = 8,
    VIEW_WP_PRIM_DISPLAY_P3       = 9,
    VIEW_WP_PRIM_ADOBE_RGB        = 10,
};

enum view_wp_intent {
    VIEW_WP_INTENT_PERCEPTUAL             = 0,
    VIEW_WP_INTENT_RELATIVE               = 1,
    VIEW_WP_INTENT_SATURATION             = 2,
    VIEW_WP_INTENT_ABSOLUTE               = 3,
    VIEW_WP_INTENT_RELATIVE_BPC           = 4,
    VIEW_WP_INTENT_ABSOLUTE_NO_ADAPTATION = 5,
};

/* What the compositor can do, from wp_color_manager_v1's events. */
struct view_colour_caps {
    uint32_t version;              /* bound interface version */
    bool parametric, set_primaries, set_tf_power, set_luminances,
         mastering_display_primaries, extended_target_volume;
    uint64_t tf_named;             /* bit i: named TF i supported (enum values < 64) */
    uint64_t primaries_named;      /* bit i: named primaries i supported */
    uint32_t intents;              /* bit i: render intent i supported */
};

/* A parsed image description, from wp_image_description_info_v1. */
struct view_colour_desc {
    struct pl_raw_primaries primaries;   /* always sent */
    int primaries_named;                 /* wp enum, 0 when none */
    int tf_named;                        /* wp enum, 0 when none */
    float tf_power;                      /* exponent, 0 when none */
    float min_lum, max_lum, ref_lum;     /* luminances event */
    struct pl_raw_primaries target;      /* target_primaries */
    float target_min, target_max;        /* target_luminance */
    float max_cll, max_fall;             /* 0 when not sent */
};

/* What was decided: how to render and what to tell the compositor. */
struct view_colour_choice {
    struct pl_color_space target;        /* libplacebo's target colour */
    int depth;                           /* dither depth: 8 or 10 */
    /* The parametric description to set, in protocol terms. */
    int primaries_named;                 /* wp enum */
    int tf_named;                        /* wp enum, or 0 with tf_power */
    float tf_power;
    bool set_luminances;                 /* whether to send set_luminances */
    float min_lum, max_lum, ref_lum;
    bool set_mastering;                  /* whether to send mastering primaries (+ luminance when HDR) */
    struct pl_raw_primaries mastering;   /* == target.hdr.prim */
    float mastering_min, mastering_max;
    int intent;                          /* wp render intent */
    bool exact;                          /* true when the preferred was matched as is */
    /* Which of the candidates won, 1 to 5 (1 is `exact`), and, when it was
     * not the first, why the first was refused; for the log line. */
    int candidate;
    const char *why;
    /* The compositor's target peak was no use (10000 cd/m² or nothing)
     * and 1000 was assumed in its place; for the log line. */
    bool peak_assumed;
};

/* Returns 0 with *out filled, or CRAW_ERROR(ENOTSUP) when nothing this
 * program can render is something the compositor can accept. `why` is set
 * on failure too, to the first candidate's reason. */
int view_colour_resolve(const struct view_colour_caps *caps,
                        const struct view_colour_desc *preferred,
                        struct view_colour_choice *out);

/* Whether two choices would render the same and put the same description
 * on the surface: every protocol-side field and the whole target colour.
 * The bookkeeping (`exact`, `candidate`, `why`) does not count. */
bool view_colour_choice_equal(const struct view_colour_choice *a,
                              const struct view_colour_choice *b);

/* The named set `xy` is, within libplacebo's pl_raw_primaries_similar(),
 * out of those this program can render into; 0 for none. */
int view_colour_primaries_match(const struct pl_raw_primaries *xy);

const char *view_colour_tf_name(int wp_tf);           /* for logs */
const char *view_colour_primaries_name(int wp_prim);  /* for logs */
const char *view_colour_intent_name(int wp_intent);   /* for logs */

/* One line's worth of each, for the -v log: what the compositor offers,
 * what it prefers, and what was made of it. */
void view_colour_describe_caps(const struct view_colour_caps *caps,
                               char *buf, size_t size);
void view_colour_describe_desc(const struct view_colour_desc *desc,
                               char *buf, size_t size);
void view_colour_describe_choice(const struct view_colour_choice *choice,
                                 char *buf, size_t size);

/* ----------------------------------------------------- the window side */

struct view_gpu;

/* Right after the swapchain's first pl_swapchain_resize(). Returns 0
 * unless out of memory: no Wayland window, no manager, a compositor
 * without parametric descriptions or one that prefers nothing this program
 * can render all leave the module inactive, which is not an error. When it
 * comes back active the swapchain has been switched to pass-through and
 * the caller resizes it again. */
int  view_colour_open(struct view_gpu *g, void *sdl_window, bool verbose);
/* Once per loop iteration, before SDL_PollEvent(); never blocks. True when
 * the window's target colour moved, so that the caller draws. */
bool view_colour_dispatch(struct view_gpu *g);
bool view_colour_active(const struct view_gpu *g);
/* Between pl_frame_from_swapchain() and the render: replaces the frame's
 * colour and bit encoding with the negotiated ones. `csp` is the frame's
 * colour as the swapchain gave it, which has to be pass-through's
 * unknown; anything else means the pass-through never took, and the
 * module leaves both alone and gives up at the next dispatch. */
void view_colour_target(struct view_gpu *g, struct pl_color_space *csp,
                        struct pl_bit_encoding *bits);
/* What the panel overlay is tagged as. */
void view_colour_overlay(const struct view_gpu *g, struct pl_color_space *csp);
/* Before the swapchain is destroyed. */
void view_colour_close(struct view_gpu *g);

#endif /* CINEVIEW_COLOUR_H */
