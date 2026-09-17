/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#ifndef CINEVIEW_GRAIN_UI_H
#define CINEVIEW_GRAIN_UI_H

#include <stddef.h>

#include <libplacebo/shaders/custom.h>

#include "view.h"

/* Only ui.c has nuklear's header; for everyone else the tag has to exist at
 * file scope or the prototype below would declare it inside its own
 * parameter list. */
struct nk_context;

/* The viewer's side of the film grain model.
 *
 * grain.c holds the model itself and knows nothing about this program;
 * everything here is the part that belongs to cineview: the panel section,
 * the command line option, and the hook handed to pl_render_params. It is a
 * file of its own so that ui.c, main.c and gpu.c each need one line. */

/* The state a fresh viewer starts with: grain off, and behind that a Super
 * 35 / VISION3 250D print look ready for when it is switched on. */
void view_grain_defaults(struct view_grain *g);

/* Parses `--grain GAUGE[,STOCK][:AMOUNT]`, e.g. `super8`, `16mm,500t` or
 * `super35,50d:0.6`. Turns grain on. Returns 0 or a negative errno, having
 * printed what was wrong. */
int view_grain_parse(struct view_grain *g, const char *arg);

/* Rebuilds the grain for this frame and returns the hook to put in
 * pl_render_params.hooks, or NULL when grain is off or the build failed.
 * `export_path` picks which renderer's shader state to use - the display
 * and the export each own one, the way they each own a pl_renderer - and
 * with it the seed policy. The returned pointer stays valid until the next
 * call for the same path. */
const struct pl_hook *view_grain_hook(struct view_state *st, bool export_path);

/* Releases both hooks' shader state. Call before the pl_gpu goes away. */
void view_grain_close(struct view_grain *g);

/* One line for the panel's readout: what the model made of the current
 * settings at the current magnification. Returns `buf`. */
const char *view_grain_summary(const struct view_grain *g, char *buf, size_t n);

/* And one for what was asked for rather than what was built: the gauge, the
 * stock and the amount, which is the part that is a setting and carries
 * from clip to clip. Returns `buf`. */
const char *view_grain_setting(const struct view_grain *g, char *buf, size_t n);

/* The body of the panel's "Film grain" section, laid out by ui.c inside the
 * section header it owns - the switch on that header is grain.on, so there
 * is none in here. Worked with the pointer, like the rest of the panel. */
void view_panel_grain(struct nk_context *ctx, struct view_state *st);

#endif /* CINEVIEW_GRAIN_UI_H */
