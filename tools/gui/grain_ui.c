/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* The viewer's film grain section: the panel rows, the command line option
 * and the hook the renderer runs.
 *
 * The model lives in grain.c and knows nothing about craw-view. What this
 * file adds is where the grain is applied from the viewer's point of view.
 * The hook sits at PL_HOOK_PRE_OUTPUT, so the grain lands on the picture
 * after scaling and colour management - which means what is graded is what
 * is on screen, at the size it is on screen, zoomed or cropped or not. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vulkan/vulkan.h>

#include <SDL3/SDL.h>

/* nuklear is a single header whose struct layout depends on which
 * NK_INCLUDE_* switches were set, so this file has to set exactly the ones
 * ui.c does - minus the two implementation switches, which belong to ui.c
 * alone. Keep the two lists in step. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wundef"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wmissing-prototypes"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#pragma GCC diagnostic ignored "-Wparentheses"
#pragma GCC diagnostic ignored "-Wtype-limits"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wpointer-arith"

#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT

#include "nuklear/nuklear.h"

#pragma GCC diagnostic pop

#include "grain.h"
#include "grain_ui.h"

/* The row height and margin ui.c lays the panel out with. */
#define GRAIN_ROW 24.0f

/* ------------------------------------------------------- panel plumbing */

/* These three mirror ui.c's view_row(), view_step() and view_adjust_*(),
 * which are static to that file. They are copied rather than shared because
 * this section is meant to cost ui.c exactly one line; they are eleven lines
 * of nuklear's public API and no state of their own, but if ui.c's focus
 * ring ever changes - a different colour, a different inset, a modifier on
 * the step - these have to change with it. */
static bool grain_row(struct view_state *st, struct nk_context *ctx)
{
    bool focused = st->ui_row == st->focus;

    if (focused) {
        struct nk_rect b = nk_widget_bounds(ctx);
        struct nk_rect c = nk_window_get_content_region(ctx);
        struct nk_command_buffer *out = nk_window_get_canvas(ctx);

        b.x = c.x - 2.0f;
        b.w = c.w + 4.0f;
        b.y -= 2.0f;
        b.h += 4.0f;
        nk_fill_rect(out, b, 2.0f, nk_rgba(46, 64, 96, 255));
    }

    st->ui_row++;

    return focused;
}

static int grain_step(struct view_state *st, bool focused)
{
    int adj = focused ? st->adjust : 0;

    if (adj)
        st->adjust = 0;

    return adj;
}

static void grain_adjust_float(struct view_state *st, bool focused, float *v,
                               float step, float lo, float hi)
{
    int adj = grain_step(st, focused);

    if (adj) {
        *v = CRAW_CLAMP(*v + (float)adj * step, lo, hi);
        if (fabsf(*v) < step * 1e-3f)
            *v = 0.0f;
    }
}

static void grain_adjust_cycle(struct view_state *st, bool focused, int *v,
                               int n)
{
    int adj = grain_step(st, focused);

    if (adj)
        *v = (*v + adj + n) % n;
}

/* ------------------------------------------------------------ defaults */

void view_grain_defaults(struct view_grain *g)
{
    struct craw_grain_params p = craw_grain_default_params();

    *g = (struct view_grain) {
        .on       = false,
        .gauge    = p.gauge,
        .stock    = p.stock,
        .response = p.response,
        .amount   = p.amount,
        .size     = p.size,
        /* Negative means "whatever the stock says", which is what the
         * model wants and what the panel shows as `auto`. */
        .chroma   = -1.0f,
        .boil     = true,
        .boil_flag = 1,
    };
}

/* ------------------------------------------------------- command line */

int view_grain_parse(struct view_grain *g, const char *arg)
{
    char buf[128];
    char *comma, *colon;
    int gauge, stock;

    snprintf(buf, sizeof(buf), "%s", arg);

    colon = strchr(buf, ':');
    if (colon) {
        char *end;
        float amount;

        *colon++ = '\0';
        amount = strtof(colon, &end);
        if (end == colon || *end || !(amount >= 0.0f)) {
            fprintf(stderr, "--grain: `%s` is not an amount\n", colon);
            return CRAW_ERROR(EINVAL);
        }
        g->amount = amount;
    }

    comma = strchr(buf, ',');
    if (comma) {
        *comma++ = '\0';
        stock = craw_grain_stock_from_key(comma);
        if (stock < 0) {
            fprintf(stderr, "--grain: `%s` is not a stock\n", comma);
            goto list;
        }
        g->stock = stock;
    }

    gauge = craw_grain_gauge_from_key(buf);
    if (gauge < 0) {
        fprintf(stderr, "--grain: `%s` is not a gauge\n", buf);
        goto list;
    }
    g->gauge = gauge;
    g->on = true;
    return 0;

list:
    fprintf(stderr, "  gauges:");
    for (int i = 0; i < CRAW_GRAIN_GAUGE_NB; i++)
        fprintf(stderr, " %s", craw_grain_gauge_key(i));
    fprintf(stderr, "\n  stocks:");
    for (int i = 0; i < CRAW_GRAIN_STOCK_NB; i++)
        fprintf(stderr, " %s", craw_grain_stock_key(i));
    fprintf(stderr, "\n");
    return CRAW_ERROR(EINVAL);
}

/* ----------------------------------------------------------- the hook */

/* The grain is a property of the film, so what it looks like on screen
 * depends on how much the frame is magnified. At fit, the whole frame fills
 * the window and one screen pixel covers `width / window` of it; at 1:1 it
 * covers one source pixel. Handing the model the frame's width *as it would
 * be at this zoom* gets both right with one number: it comes out as the
 * window width when fitted and as the source width at 1:1, so zooming in on
 * a still magnifies the grain exactly as a loupe on a scan would. */
static int grain_model_width(const struct view_state *st, bool export_path)
{
    float w;

    if (export_path)
        return (int)st->gpu.width;      /* the export target is the frame */

    w = (float)st->gpu.width * view_zoom_scale(st);
    if (!(w >= 16.0f))
        w = 16.0f;
    /* Rounded to a coarse grid. Solving for the filter costs a few dozen
     * template generations, and the solver only caches one answer, so a
     * width that moves by a pixel - a window being dragged to a new size -
     * would re-solve every frame. A sixteen pixel grid changes the blob
     * size by well under a percent and makes that impossible. */
    return ((int)lrintf(w) + 8) / 16 * 16;
}

/* A seed per frame, from the frame's own index.
 *
 * Deriving it from the index rather than from a counter buys two things at
 * once: an export of the same range with the same settings is bit for bit
 * the same twice over, and a paused viewer holds one grain still instead of
 * boiling under a colourist who is trying to look at it. The mix is
 * splitmix64's finaliser, because the AV1 path keeps only the low sixteen
 * bits and consecutive indices have to land far apart in those. */
static uint64_t grain_seed_for(const struct view_state *st)
{
    uint64_t x;

    if (!st->grain.boil)
        return 0x9E3779B97F4A7C15ULL;

    x = (uint64_t)st->index + 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

const struct pl_hook *view_grain_hook(struct view_state *st, bool export_path)
{
    struct view_grain *g = &st->grain;
    craw_grain_hook **slot = export_path ? &g->hook_export : &g->hook;
    struct craw_grain_params p = craw_grain_default_params();
    struct craw_grain_out out;
    const struct pl_hook *hook;

    if (!g->on || g->amount <= 0.0f || !st->gpu.gpu)
        return NULL;

    if (!*slot) {
        *slot = craw_grain_hook_create(st->gpu.gpu);
        if (!*slot)
            return NULL;
    }

    p.gauge    = g->gauge;
    p.stock    = g->stock;
    p.response = g->response;
    p.amount   = g->amount;
    p.size     = g->size;
    p.chroma   = g->chroma;
    p.path     = CRAW_GRAIN_PATH_OUTPUT;
    p.width    = grain_model_width(st, export_path);
    p.height   = (int)((int64_t)p.width * st->gpu.height / st->gpu.width);
    if (p.height < 1)
        p.height = 1;
    p.seed     = grain_seed_for(st);

    hook = craw_grain_hook_update(*slot, &p, &out);

    /* The readout is the display path's, so that switching to an export
     * does not rewrite what the panel is showing. */
    if (!export_path) {
        g->blob_px    = out.blob_px;
        g->blob_built = out.blob_clamped;
        g->sigma      = out.sigma_built;
        g->model_used = out.model_used;
        g->subsample  = out.subsample;
    }

    return hook;
}

void view_grain_close(struct view_grain *g)
{
    craw_grain_hook_destroy(&g->hook);
    craw_grain_hook_destroy(&g->hook_export);
}

const char *view_grain_summary(const struct view_grain *g, char *buf, size_t n)
{
    if (!g->on || g->amount <= 0.0f)
        snprintf(buf, n, "off");
    else
        /* Both blob sizes, because the gauge routinely asks for more than
         * the model can build and the panel should not hide that, and the
         * amplitude as a percentage because four decimal places of a
         * fraction do not fit in the column. */
        snprintf(buf, n, "%s %.1f/%.1f %.2f%%%s",
                 craw_grain_model_name(g->model_used),
                 (double)g->blob_built, (double)g->blob_px,
                 100.0 * (double)g->sigma,
                 /* Said out loud because it changes how the grain is made,
                  * not just how it looks: above a few pixels the field is
                  * synthesised smaller and upsampled, or the template runs
                  * out of blobs and the picture tiles. */
                 g->subsample == 2 ? " 1/2" : g->subsample == 4 ? " 1/4" : "");
    return buf;
}

/* ------------------------------------------------------------- panel */

void view_panel_grain(struct nk_context *ctx, struct view_state *st)
{
    struct view_grain *g = &st->grain;
    static const char *responses[CRAW_GRAIN_RESPONSE_NB] = {
        "negative", "print", "reversal", "flat",
    };
    static const char *gauges[CRAW_GRAIN_GAUGE_NB];
    static const char *stocks[CRAW_GRAIN_STOCK_NB];
    static bool names_done;
    char buf[128];
    int on;

    if (!names_done) {
        for (int i = 0; i < CRAW_GRAIN_GAUGE_NB; i++)
            gauges[i] = craw_grain_gauge_name(i);
        for (int i = 0; i < CRAW_GRAIN_STOCK_NB; i++)
            stocks[i] = craw_grain_stock_name(i);
        names_done = true;
    }

    /* Open when grain is already on, which after --grain it is: somebody
     * who asked for grain on the command line wants to see the controls.
     * nuklear remembers the state per title from the first frame on, so
     * this only decides where the section starts. */
    if (!nk_tree_push(ctx, NK_TREE_TAB, "Film grain",
                      g->on ? NK_MAXIMIZED : NK_MINIMIZED))
        return;

    nk_layout_row_dynamic(ctx, GRAIN_ROW, 2);
    on = g->on ? 1 : 0;
    grain_adjust_cycle(st, grain_row(st, ctx), &on, 2);
    g->on = on != 0;
    {
        int flag = g->on ? 1 : 0;
        nk_checkbox_label(ctx, "grain", &flag);
        g->on = flag != 0;
    }
    nk_labelf(ctx, NK_TEXT_LEFT, "%s",
              view_grain_summary(g, buf, sizeof(buf)));

    nk_layout_row_dynamic(ctx, GRAIN_ROW, 2);
    grain_adjust_cycle(st, grain_row(st, ctx), &g->gauge,
                       CRAW_GRAIN_GAUGE_NB);
    nk_label(ctx, "gauge", NK_TEXT_LEFT);
    g->gauge = nk_combo(ctx, gauges, CRAW_GRAIN_GAUGE_NB, g->gauge, 20,
                        nk_vec2(180, 260));

    nk_layout_row_dynamic(ctx, GRAIN_ROW, 2);
    grain_adjust_cycle(st, grain_row(st, ctx), &g->stock,
                       CRAW_GRAIN_STOCK_NB);
    nk_label(ctx, "stock", NK_TEXT_LEFT);
    g->stock = nk_combo(ctx, stocks, CRAW_GRAIN_STOCK_NB, g->stock, 20,
                        nk_vec2(180, 280));

    nk_layout_row_dynamic(ctx, GRAIN_ROW, 2);
    grain_adjust_cycle(st, grain_row(st, ctx), &g->response,
                       CRAW_GRAIN_RESPONSE_NB);
    nk_label(ctx, "response", NK_TEXT_LEFT);
    g->response = nk_combo(ctx, responses, CRAW_GRAIN_RESPONSE_NB,
                           g->response, 20, nk_vec2(160, 120));

    nk_layout_row_dynamic(ctx, GRAIN_ROW, 1);
    grain_adjust_float(st, grain_row(st, ctx), &g->amount, 0.05f, 0.0f, 3.0f);
    nk_property_float(ctx, "#amount", 0.0f, &g->amount, 3.0f, 0.05f, 0.01f);

    grain_adjust_float(st, grain_row(st, ctx), &g->size, 0.05f, 0.25f, 4.0f);
    nk_property_float(ctx, "#size", 0.25f, &g->size, 4.0f, 0.05f, 0.01f);

    /* Chroma is the one control with a meaningful "unset": below zero the
     * stock's own figure is used, which is what most people want, so the
     * slider starts one step above that and stepping down off the bottom
     * goes back to it. */
    nk_layout_row_dynamic(ctx, GRAIN_ROW, 2);
    {
        bool f = grain_row(st, ctx);
        float c = g->chroma < 0.0f ? -0.05f : g->chroma;

        grain_adjust_float(st, f, &c, 0.05f, -0.05f, 2.0f);
        g->chroma = c < 0.0f ? -1.0f : c;
        if (g->chroma < 0.0f)
            nk_label(ctx, "chroma: auto", NK_TEXT_LEFT);
        else
            nk_labelf(ctx, NK_TEXT_LEFT, "chroma: %.2f", (double)g->chroma);
        nk_labelf(ctx, NK_TEXT_LEFT, "%s",
                  craw_grain_response_name(g->response));
    }

    nk_layout_row_dynamic(ctx, GRAIN_ROW, 2);
    {
        int boil = g->boil ? 1 : 0;

        grain_adjust_cycle(st, grain_row(st, ctx), &boil, 2);
        g->boil_flag = boil;
        nk_checkbox_label(ctx, "boil", &g->boil_flag);
        g->boil = g->boil_flag != 0;
        nk_label(ctx, g->boil ? "new grain each frame" : "held still",
                 NK_TEXT_LEFT);
    }

    /* Where the numbers came from, since half the stocks are estimates. */
    nk_layout_row_dynamic(ctx, GRAIN_ROW, 1);
    nk_labelf(ctx, NK_TEXT_LEFT, "%.72s", craw_grain_stock_source(g->stock));

    nk_tree_pop(ctx);
}
