/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* The nuklear panel.
 *
 * The UI is drawn by nuklear's native Vulkan backend into a libplacebo
 * texture the size of the window, and composited by the renderer as a
 * pl_overlay in the same pl_render_image call as the picture. libplacebo
 * hands the image over with pl_vulkan_hold_ex, nuklear waits on the
 * semaphore that fires, and the semaphore nuklear signals when its own
 * submission is done is what libplacebo waits on again. */

#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vulkan/vulkan.h>

#include <libavutil/time.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <fontconfig/fontconfig.h>

/* The vendored headers are third party and do not build clean under this
 * project's warning flags; the suppressions stop at the end of the include
 * block so that everything below is held to the same standard as the rest
 * of the tool. */
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
#define NK_IMPLEMENTATION
#define NK_SDL_VULKAN_IMPLEMENTATION

#include "nuklear/nuklear.h"
#include "nuklear/nuklear_sdl_vulkan.h"

#pragma GCC diagnostic pop

#include <libplacebo/gpu.h>

#include "view.h"
#include "grain_ui.h"

/* The embedded film simulation collection is optional: without it the row
 * and its menu compile out and the panel says so. */
#ifndef CINEVIEW_HAVE_LUTS
#define CINEVIEW_HAVE_LUTS 0
#endif

#if CINEVIEW_HAVE_LUTS
#include "luts.h"
#endif

#define VIEW_UI_WIDTH  VIEW_PANEL_WIDTH
#define VIEW_UI_MARGIN VIEW_PANEL_MARGIN

/* The panel's face is whatever fontconfig calls sans-serif on this machine,
 * at this many pixels: the size at which the panel's longest readouts
 * still fit its 360 px width and a row of the film menu still fits a
 * modest window. Every row height below is sized for it; nuklear's own 13
 * px bitmap face is only what the panel falls back to when fontconfig has
 * nothing or the file will not load. */
#define VIEW_FONT_PX   15.0f
#define VIEW_ROW       26.0f
/* A row in a combo's dropdown, which nuklear takes as an int. */
#define VIEW_COMBO_ROW 24

/* The cascading menu's own metrics. Its rows are tighter than the panel's:
 * a film category can hold twenty-six stocks and the column still has to
 * fit in a modest window. */
#define VIEW_MENU_ROW   24.0f
#define VIEW_MENU_PAD    4.0f
#define VIEW_MENU_XPAD   8.0f
/* How long the pointer has to rest on an entry before its submenu opens,
 * and how long an already open submenu survives the pointer sweeping
 * across its siblings on the way over to it. */
#define VIEW_MENU_OPEN_US  150000
/* How long a film has to be under the pointer before it previews: long
 * enough that the films a hand crosses on its way into an open column
 * (fifty milliseconds each at a stroll) do not each flash their look. */
#define VIEW_MENU_PEEK_US   80000
#define VIEW_MENU_GUARD_US 300000
#define VIEW_MENU_MIN_W 110.0f
#define VIEW_MENU_MAX_W 330.0f
#define VIEW_MENU_ARROW 16.0f

#define VIEW_NK_VERTEX_BUFFER (512 * 1024)
#define VIEW_NK_INDEX_BUFFER  (128 * 1024)

static struct nk_context *view_nk;
static VkSemaphore view_ui_hold_sem[VIEW_SLOTS];

/* What the panel cuts a value that will not fit with: the typographic
 * ellipsis once a face that has it is loaded, three dots until then, since
 * the built-in bitmap face has no U+2026. */
static const char *view_ellipsis = "...";

/* ------------------------------------------------------------ lifetime */

/* The face fontconfig names for "sans-serif", the way every desktop
 * application on the machine finds its UI font, so that the panel gets the
 * one the person already reads everything else in: no option, no
 * environment variable, nothing to configure. `path` is the file and
 * `index` the face inside it for a collection; `ellipsis` says whether
 * the face has U+2026, which fontconfig knows from its charset and which
 * decides what long values are cut with. False when fontconfig has
 * nothing to offer, in which case the built-in face stays.
 *
 * fontconfig's whole lifetime is inside this call: nothing else in the
 * program uses it, and the atlas reads the file into memory of its own. */
static bool view_font_find(char *path, size_t n, int *index, bool *ellipsis,
                           char *family, size_t family_n)
{
    FcPattern *pat, *match = NULL;
    FcResult res = FcResultNoMatch;
    FcChar8 *str;
    FcCharSet *cs;
    bool ok = false;

    if (!FcInit())
        return false;

    pat = FcNameParse((const FcChar8 *)"sans-serif");
    if (pat) {
        FcConfigSubstitute(NULL, pat, FcMatchPattern);
        FcDefaultSubstitute(pat);
        match = FcFontMatch(NULL, pat, &res);
    }

    if (match && res == FcResultMatch &&
        FcPatternGetString(match, FC_FILE, 0, &str) == FcResultMatch) {
        snprintf(path, n, "%s", (const char *)str);
        if (FcPatternGetInteger(match, FC_INDEX, 0, index) != FcResultMatch)
            *index = 0;
        *ellipsis = FcPatternGetCharSet(match, FC_CHARSET, 0, &cs) == FcResultMatch &&
                    FcCharSetHasChar(cs, 0x2026);
        if (FcPatternGetString(match, FC_FAMILY, 0, &str) == FcResultMatch)
            snprintf(family, family_n, "%s", (const char *)str);
        else
            family[0] = '\0';
        ok = true;
    }

    if (match)
        FcPatternDestroy(match);
    if (pat)
        FcPatternDestroy(pat);
    FcFini();

    return ok;
}

/* Adds that face to the atlas, or nothing: nuklear bakes its own bitmap
 * face when the atlas is empty, which is the fallback. Returns the font to
 * set as the style's, or NULL to leave nuklear's. */
static struct nk_font *view_font_load(struct view_state *st,
                                      struct nk_font_atlas *atlas)
{
    /* Latin-1, which nuklear's default range is and which the panel's
     * degree sign needs, plus the ellipsis when the face has it. A
     * collection's inner faces are beyond nuklear's loader, which always
     * reads the first, so only a plain file at index 0 is taken. */
    static const nk_rune latin1[] = { 0x0020, 0x00FF, 0 };
    static const nk_rune latin1_ellipsis[] = { 0x0020, 0x00FF, 0x2026, 0x2026, 0 };
    struct nk_font_config cfg = nk_font_config(VIEW_FONT_PX);
    struct nk_font *font;
    char path[VIEW_MAX_PATH], family[128];
    bool ellipsis = false;
    int index = 0;

    if (!view_font_find(path, sizeof(path), &index, &ellipsis,
                        family, sizeof(family))) {
        view_log(st, "panel font: fontconfig has no sans-serif face, "
                     "keeping the built-in one\n");
        return NULL;
    }
    if (index != 0) {
        view_log(st, "panel font: %s is face %d of a collection, which the "
                     "atlas cannot take; keeping the built-in one\n",
                 path, index);
        return NULL;
    }

    /* Horizontal oversampling is what makes a small face read cleanly: the
     * glyphs are rasterized at three times the width and sampled back, so
     * their stems land between pixels without smearing. Vertically the
     * rows are pixel aligned already. */
    cfg.oversample_h = 3;
    cfg.oversample_v = 1;
    cfg.range = ellipsis ? latin1_ellipsis : latin1;

    font = nk_font_atlas_add_from_file(atlas, path, VIEW_FONT_PX, &cfg);
    if (!font) {
        view_log(st, "panel font: %s would not load, keeping the built-in "
                     "one\n", path);
        return NULL;
    }

    view_ellipsis = ellipsis ? "\xe2\x80\xa6" : "...";
    view_log(st, "panel font: %s (%s) at %.0f px\n", path, family, VIEW_FONT_PX);

    return font;
}

static void view_ui_toggle_style(struct nk_context *ctx);

int view_ui_open(struct view_state *st, void *sdl_window)
{
    struct view_gpu *g = &st->gpu;
    struct nk_font_atlas *atlas;
    struct nk_font *font;
    int w = 0, h = 0, aw = 0, ah = 0;
    int ret;

    SDL_GetWindowSizeInPixels(sdl_window, &w, &h);

    ret = view_gpu_ui_resize(g, w, h);
    if (ret < 0)
        return ret;

    for (int i = 0; i < VIEW_SLOTS; i++) {
        view_ui_hold_sem[i] = pl_vulkan_sem_create(g->gpu, pl_vulkan_sem_params(
            .type = VK_SEMAPHORE_TYPE_BINARY,
        ));
        if (view_ui_hold_sem[i] == VK_NULL_HANDLE)
            return CRAW_ERROR(ENOMEM);
    }

    /* The render pass has to agree with whatever VkFormat libplacebo gave
     * the overlay texture, which is why view_gpu_ui_resize reports it. */
    view_nk = nk_sdl_init(sdl_window, g->vk->device, g->vk->phys_device, g->qf,
                          g->ui_views, VIEW_SLOTS, g->ui_format,
                          NK_SDL_DEFAULT, VIEW_NK_VERTEX_BUFFER,
                          VIEW_NK_INDEX_BUFFER);
    if (!view_nk)
        return CRAW_ERROR(ENOMEM);

    nk_sdl_font_stash_begin(&atlas);
    font = view_font_load(st, atlas);
    nk_sdl_font_stash_end(g->queue, &aw, &ah);
    /* The backend only installs the built-in face by itself. */
    if (font)
        nk_style_set_font(view_nk, &font->handle);
    view_log(st, "panel font atlas: %dx%d\n", aw, ah);

    /* An opaque panel keeps the alpha the backend writes at 1.0 wherever it
     * matters; only the outermost antialiased border pixels are partially
     * covered, and those are where its src-alpha blend against a cleared
     * target is not exactly premultiplied. */
    view_nk->style.window.fixed_background = nk_style_item_color(nk_rgba(24, 24, 28, 255));
    view_ui_toggle_style(view_nk);

    return 0;
}

void view_ui_close(struct view_state *st)
{
    if (view_nk) {
        nk_sdl_shutdown();
        view_nk = NULL;
    }
    for (int i = 0; st->gpu.gpu && i < VIEW_SLOTS; i++)
        if (view_ui_hold_sem[i] != VK_NULL_HANDLE)
            pl_vulkan_sem_destroy(st->gpu.gpu, &view_ui_hold_sem[i]);
}

bool view_ui_wants_keys(const struct view_state *st)
{
    /* Only a text field being typed into takes the keys. nuklear's own
     * notion of an active item includes the pointer merely hovering the
     * panel, which must not switch the shortcuts off. */
    return st->ui_edit_active;
}

bool view_ui_locked(const struct view_state *st)
{
    return st->video.active;
}

int view_ui_resize(struct view_state *st, int w, int h)
{
    int ret = view_gpu_ui_resize(&st->gpu, w, h);

    if (ret < 0)
        return ret;

    /* nuklear kept the address of g->ui_view, which view_gpu_ui_resize has
     * just refilled, so recreating its framebuffers picks up the new one;
     * only then can the ones they used to point at go away. */
    nk_sdl_resize((uint32_t)w, (uint32_t)h);
    view_gpu_ui_drop_old(&st->gpu);

    return 0;
}

/* A click whose press and release both land in one input cycle is lost to
 * nuklear: a button fires on a press it sees down when the panel is built,
 * and by then this one is up again. A panel that is built a few times a
 * second - under a video export developing 5.8K frames, which holds the
 * lock the panel waits for - loses most clicks that way, the stop button's
 * among them. So the release of a button pressed in the same cycle is held
 * back and handed over when the next one opens, after the panel has been
 * built with the press. */
static bool view_ui_down[4];            /* SDL's left, middle, right */
static SDL_Event view_ui_held[8];
static int view_ui_nb_held;

bool view_ui_event(void *sdl_event)
{
    SDL_Event *e = sdl_event;
    int b = e->button.button;

    if (e->type == SDL_EVENT_MOUSE_BUTTON_DOWN && b > 0 && b < 4) {
        view_ui_down[b] = true;
    } else if (e->type == SDL_EVENT_MOUSE_BUTTON_UP && b > 0 && b < 4 &&
               view_ui_down[b] &&
               view_ui_nb_held < (int)CRAW_ARRAY_ELEMS(view_ui_held)) {
        view_ui_held[view_ui_nb_held++] = *e;
        return true;
    }

    return nk_sdl_handle_event(sdl_event) != 0;
}

void view_ui_input_begin(void)
{
    int held = view_ui_nb_held;

    nk_input_begin(view_nk);
    memset(view_ui_down, 0, sizeof(view_ui_down));
    view_ui_nb_held = 0;
    for (int i = 0; i < held; i++)
        view_ui_event(&view_ui_held[i]);
}

void view_ui_input_end(void)
{
    nk_sdl_handle_grab();
    nk_input_end(view_nk);
}

/* -------------------------------------------------------------- panel */

static const char *view_wb_source_name(enum CRAWAwbSource s)
{
    switch (s) {
    case CRAW_AWB_SOURCE_ZONE:   return "surfaces";
    case CRAW_AWB_SOURCE_BRIGHT: return "bright";
    case CRAW_AWB_SOURCE_NONE:   break;
    }
    return "none";
}

/* A text cut to fit `w` pixels of the panel's font, with the ellipsis on
 * the end when it had to be: a value that does not fit its column is shown
 * as far as it goes rather than clipped mid-glyph by the widget. The cut
 * steps back over whole UTF-8 sequences, since the values carry a degree
 * sign and a face may put anything in a camera's model name. */
static void view_fit_text(const struct nk_user_font *f, float w,
                          const char *text, char *out, size_t n)
{
    size_t len = strlen(text);
    float ell;

    if (len >= n)
        len = n - 1;

    if (f->width(f->userdata, f->height, text, (int)len) <= w) {
        memcpy(out, text, len);
        out[len] = '\0';
        return;
    }

    ell = f->width(f->userdata, f->height, view_ellipsis,
                   (int)strlen(view_ellipsis));
    while (len > 0) {
        len--;
        /* Not on a continuation byte: back to the start of the sequence. */
        while (len > 0 && ((unsigned char)text[len] & 0xC0) == 0x80)
            len--;
        if (f->width(f->userdata, f->height, text, (int)len) + ell <= w)
            break;
    }

    if (len + strlen(view_ellipsis) >= n)
        len = n - 1 - strlen(view_ellipsis);
    memcpy(out, text, len);
    strcpy(out + len, view_ellipsis);
}

/* The three sections a scene-referred export takes out of the picture: the
 * look, the adjustments and the grain. All three are display-referred
 * operators on values in [0,1], and a scene-referred export is written
 * straight out of the pipeline with none of them on it - so they are drawn
 * at half strength and take no input. The same hooks are left off the
 * display too, which is the point: what is on screen is what the file
 * holds, up to the display transform, so a live control here would be
 * lying about the file and about the window at once.
 *
 * The section's own tree tab is deliberately left alive, because a section
 * you cannot open is a section that cannot tell you why it is greyed. */
/* nk_widget_disable_begin() and _end() for a stretch of the panel, with
 * one rule on top: while the whole panel is locked by a running video
 * export, the end of an inner stretch is not an end. nuklear's pair does
 * not nest - the end re-enables everything - and a section that greyed
 * three rows for its own reasons would otherwise unlock the rest of the
 * panel behind it. */
static void view_disable_begin(struct nk_context *ctx)
{
    nk_widget_disable_begin(ctx);
}

static void view_disable_end(struct view_state *st, struct nk_context *ctx)
{
    if (view_ui_locked(st))
        return;

    nk_widget_disable_end(ctx);
}

static bool view_grey_begin(struct view_state *st, struct nk_context *ctx,
                            const char *why)
{
    if (!view_export_is_scene(st))
        return false;

    /* At full strength, before the disable: the one line that explains the
     * rest is the one line that has to be readable. Three rows of height,
     * because nk_label_wrap() wraps inside the rectangle it is given and
     * clips what does not fit rather than growing. */
    if (why) {
        nk_layout_row_dynamic(ctx, VIEW_ROW * 3.0f, 1);
        nk_label_wrap(ctx, why);
    }

    view_disable_begin(ctx);

    return true;
}

static void view_grey_end(struct view_state *st, struct nk_context *ctx, bool on)
{
    if (!on)
        return;

    view_disable_end(st, ctx);
}

/* ------------------------------------------------- the section headers */

/* A section header is a power switch and an expander side by side, the way
 * RawTherapee puts an enable button at the left of every tool's title: off
 * bypasses the whole section while every parameter in it is kept, so
 * switching it back on returns the picture it had.
 *
 * nuklear's tree header cannot host it. A tree header is one widget from
 * the triangle to the far edge of the row, and it owns every click that
 * lands on it, so there is nowhere to put a switch that is not also the
 * expander. The row is therefore built here out of a template - a square
 * cell for the switch, then a button over the rest of the row - and that
 * button is drawn as the NK_TREE_TAB it replaces: the same height, the same
 * background and border, the triangle at the left of a left-aligned title,
 * and the body indented under it the way nuklear indents a tree's.
 *
 * Which sections are open lives in view state rather than in nuklear's own
 * per-title table, because nothing but a tree widget writes that table.
 * Both halves are clicked; nothing on the keyboard reaches them. */

/* Lit when on, an outline when off: down a column of ten of these the
 * state has to be readable without reading anything. */
#define VIEW_SWITCH_ON  nk_rgb(122, 182, 255)
#define VIEW_SWITCH_OFF nk_rgb(104, 104, 112)

/* Checkboxes and radio buttons in the same language as the switches: off
 * is an empty dark well with the switch's grey outline, on is the
 * switch's lit colour inside it. nuklear's default draws a set box as a
 * dark hole in a grey square and an unset one as the plain grey square,
 * which reads the wrong way round. */
static void view_ui_toggle_style(struct nk_context *ctx)
{
    struct nk_style_toggle *t[2] = { &ctx->style.checkbox, &ctx->style.option };

    for (int i = 0; i < 2; i++) {
        t[i]->normal        = nk_style_item_color(nk_rgb(30, 30, 36));
        t[i]->hover         = nk_style_item_color(nk_rgb(48, 48, 56));
        t[i]->active        = t[i]->hover;
        t[i]->border        = 1.0f;
        t[i]->border_color  = VIEW_SWITCH_OFF;
        t[i]->padding       = nk_vec2(3.0f, 3.0f);
        t[i]->cursor_normal = nk_style_item_color(VIEW_SWITCH_ON);
        t[i]->cursor_hover  = nk_style_item_color(VIEW_SWITCH_ON);
    }
}

/* The body's indent, which is the one thing a tree does to the panel that
 * has no call of its own: `sign` +1 on the way in and -1 on the way out, so
 * the two are exact inverses the way nk_tree_push() and nk_tree_pop() are. */
static void view_section_indent(struct nk_context *ctx, float sign)
{
    struct nk_panel *p = nk_window_get_panel(ctx);
    float indent = ctx->style.tab.indent;

    p->at_x += sign * indent;
    p->bounds.w -= sign * (indent + ctx->style.window.padding.x);
}

/* `on` is the section's switch, or NULL for a section with nothing to
 * bypass; the cell is spent either way, so every title in the column lines
 * up. `disabled` greys the switch without changing it, which is what a
 * scene-referred export does to the three display-referred sections.
 *
 * Returns true when the body is to be laid out, and then view_section_end()
 * closes it. */
static bool view_section_header(struct nk_context *ctx, struct view_state *st,
                                enum view_section id, const char *title,
                                bool *on, bool disabled)
{
    const struct nk_style *style = &ctx->style;
    float h = style->font->height + 2.0f * style->tab.padding.y;
    struct nk_style_button bs;
    struct nk_rect b;

    /* A square switch cell as tall as the header, then the expander. */
    nk_layout_row_template_begin(ctx, h);
    nk_layout_row_template_push_static(ctx, h);
    nk_layout_row_template_push_dynamic(ctx);
    nk_layout_row_template_end(ctx);

    if (on) {
        bs = style->tab.tab_minimize_button;
        bs.border = 1.0f;
        bs.border_color = style->tab.border_color;
        bs.padding = style->tab.padding;
        /* The colour as well as the dot follows what is shown: a greyed
         * switch is off, in the off colour, not a dimmed blue outline. */
        bs.text_normal = bs.text_hover = bs.text_active =
            *on && !disabled ? VIEW_SWITCH_ON : VIEW_SWITCH_OFF;
        /* This style is a copy, so the dimming nk_widget_disable_begin()
         * does to ctx->style never reaches it: a greyed switch has to be
         * given the factor here to look greyed as well as refuse the
         * click. The symbol's colour only: nuklear draws the outline as a
         * disc of the button's background colour inside a disc of the
         * symbol's, and takes that background undimmed, so dimming the
         * button's own background left a lighter disc inside a dark rim,
         * which read as a filled dot. */
        if (disabled)
            bs.color_factor_text = bs.disabled_factor;

        if (disabled)
            view_disable_begin(ctx);
        /* A disabled widget takes no input at all, so the click below
         * cannot happen while the section is greyed. A greyed switch shows
         * off whatever the setting is: nothing is being applied, and a
         * solid dot that cannot be clicked read as forced on (Lynne, of
         * Lens on a still with no record). The setting itself is kept for
         * the next clip the section can work on. */
        if (nk_button_symbol_styled(ctx, &bs, *on && !disabled
                                                  ? NK_SYMBOL_CIRCLE_SOLID
                                                  : NK_SYMBOL_CIRCLE_OUTLINE))
            *on = !*on;
        if (disabled)
            view_disable_end(st, ctx);
    } else {
        nk_spacing(ctx, 1);
    }

    bs = style->tab.tab_minimize_button;
    /* The expander fills its whole cell and is drawn as the tree tab it
     * replaces. */
    bs.normal = bs.hover = bs.active = style->tab.background;
    bs.border = style->tab.border;
    bs.border_color = style->tab.border_color;
    bs.rounding = style->tab.rounding;
    bs.text_normal = bs.text_hover = bs.text_active = style->tab.text;
    bs.text_alignment = NK_TEXT_LEFT;
    /* Padding that leaves the triangle its square, where the tree draws
     * its own. */
    bs.padding = nk_vec2(style->tab.padding.x + style->font->height +
                         style->window.spacing.x, style->tab.padding.y);

    b = nk_widget_bounds(ctx);
    if (nk_button_label_styled(ctx, &bs, title))
        st->sections.open[id] = !st->sections.open[id];

    /* The triangle, drawn after the button so that it lands on the
     * background the button has just filled, in the place and the shape
     * nuklear's own tree gives it: a square of the font's height inset by
     * the tab padding, pointing right while the section is shut and down
     * while it is open. */
    {
        struct nk_command_buffer *out = nk_window_get_canvas(ctx);
        struct nk_color c = style->tab.text;
        float x = b.x + style->tab.padding.x, y = b.y + style->tab.padding.y;
        float s = style->font->height;

        if (st->sections.open[id])
            nk_fill_triangle(out, x, y, x + s, y, x + s * 0.5f, y + s, c);
        else
            nk_fill_triangle(out, x, y, x + s, y + s * 0.5f, x, y + s, c);
    }

    if (!st->sections.open[id])
        return false;

    view_section_indent(ctx, 1.0f);

    return true;
}

static void view_section_end(struct nk_context *ctx)
{
    view_section_indent(ctx, -1.0f);
}

/* ------------------------------------------------- the cascading menus */

/* Three rows use this: the film simulation, the video export's codec
 * preset and the export's colour space. The geometry, the hover
 * hysteresis and the drawing are one implementation; what differs is only
 * what a column holds, which the five accessors below answer per kind.
 *
 * nuklear has popups and groups but nothing cascading, and its windows
 * hand input to whichever one the pointer last activated, which is not how
 * a menu behaves: every column has to keep reacting while the panel
 * underneath is still the window nuklear considers active. So the columns
 * are NK_WINDOW_NO_INPUT windows — nuklear draws, borders and clips them,
 * and skips them entirely when it decides who owns the pointer — and the
 * hover, click and scroll logic below works straight off the pointer
 * position against rectangles this file computes. That also keeps the
 * columns in creation order, which is drawing order, so they always sit on
 * top of the panel built before them. */

#if CINEVIEW_HAVE_LUTS
/* The film at entry `i` of the films column, under the category the path
 * has open, or NULL. */
static const ViewLutFilm *view_menu_film_at(const struct view_state *st, int i)
{
    const struct view_lut_menu *m = &st->menu;
    const ViewLutFilmCategory *c;

    if (m->sel[0] != 0 || m->sel[1] < 0 || i < 0)
        return NULL;
    c = view_lut_film_category(m->sel[1]);
    if (!c || i >= c->n_films)
        return NULL;

    return view_lut_film(c->first_film + i);
}

/* The film whose renderings column 3 lists, or NULL when the path does not
 * reach that far. */
static const ViewLutFilm *view_menu_film(const struct view_state *st)
{
    return view_menu_film_at(st, st->menu.sel[2]);
}

/* The rendering that stands for a film while the film itself is what the
 * highlight is on: the one labelled "base", or failing that the first
 * whose label mentions it ("NC, base"), then the first that is neither a
 * push nor a pull, then the first there is. Six films have no base at
 * all - the print stocks are three gamut mappings each, Velvia 100 a
 * single generic profile - and the Polaroids call theirs "cold" and
 * "warm", which the third rule reaches. */
static int view_menu_film_base(const ViewLutFilm *f)
{
    int found = -1;

    if (!f || f->n_luts <= 0)
        return -1;

    for (int i = f->first_lut; i < f->first_lut + f->n_luts; i++) {
        const ViewLut *l = view_lut_get(i);

        if (!l)
            continue;
        if (!strcmp(l->label, "base"))
            return i;
        if (found < 0 && strstr(l->label, "base"))
            found = i;
    }
    if (found >= 0)
        return found;

    for (int i = f->first_lut; i < f->first_lut + f->n_luts; i++) {
        const ViewLut *l = view_lut_get(i);

        if (l && !strstr(l->label, "stop"))
            return i;
    }

    return f->first_lut;
}
#endif

/* The colour spaces are grouped by their first column entry, so that the
 * three BT.2020 transfers hang off one row. */
static int view_menu_colour_groups(void)
{
    int n = 0;

    for (int i = 0; i < VIEW_NB_COLOURS; i++)
        if (!i || strcmp(view_colours[i].group, view_colours[i - 1].group))
            n++;

    return n;
}

/* The first and the count of the colours in group `g`. */
static int view_menu_colour_group(int g, int *count)
{
    int seen = -1;

    for (int i = 0; i < VIEW_NB_COLOURS; i++) {
        if (!i || strcmp(view_colours[i].group, view_colours[i - 1].group))
            seen++;
        if (seen != g)
            continue;

        *count = 0;
        for (int j = i; j < VIEW_NB_COLOURS &&
                        !strcmp(view_colours[j].group, view_colours[i].group);
             j++)
            (*count)++;

        return i;
    }

    *count = 0;

    return -1;
}

/* A leaf column's entries select rather than opening another column. */
/* The pixel formats the export's format row offers, and how many.
 *
 * A scene-referred export has exactly one - the family's float format,
 * which is the only thing the direct path writes - and it is shown rather
 * than hidden so that the row still says what the file will hold. Anything
 * else is what the probe found the family's encoder opens with; a format
 * it does not open with is never offered at all, and one it opens with
 * that the preset's profile forbids is offered greyed. */
static int view_menu_fmts(const struct view_state *st,
                          const enum AVPixelFormat **out)
{
    const struct view_preset_sel *s = &st->look.video_preset;
    const struct view_preset_family *f;

    if (s->family < 0 || s->family >= view_nb_preset_families) {
        *out = NULL;
        return 0;
    }

    f = &view_preset_families[s->family];
    if (view_colours[CRAW_CLAMP(s->colour, 0, VIEW_NB_COLOURS - 1)].scene) {
        *out = &f->scene_fmt;
        return f->scene_fmt == AV_PIX_FMT_NONE ? 0 : 1;
    }

    return view_presets_fmts(s->family, out);
}

static bool view_menu_leaf(const struct view_state *st, int col)
{
    const struct view_lut_menu *m = &st->menu;
    int first, n;

    switch (m->kind) {
    case VIEW_MENU_PIXFMT:
        return true;
    case VIEW_MENU_PRESET:
        return col == 1;
    case VIEW_MENU_COLOUR:
        /* A group with one entry in it is its own leaf: BT.709 and the two
         * ACES spaces are one row each, BT.2020 opens its transfers. */
        if (col == 1)
            return true;
        first = m->sel[0] >= 0 ? view_menu_colour_group(m->sel[0], &n) : -1;
        return first < 0 || n <= 1;
    default:
        return (m->sel[0] == 1 && col == 2) || (m->sel[0] == 0 && col == 3);
    }
}

static int view_menu_count(const struct view_state *st, int col)
{
    const struct view_lut_menu *m = &st->menu;
#if CINEVIEW_HAVE_LUTS
    const ViewLutFilmCategory *fc;
    const ViewLutLookCategory *lc;
    const ViewLutFilm *f;
#endif
    int n;

    if (m->kind == VIEW_MENU_PIXFMT) {
        const enum AVPixelFormat *fmts;

        return col == 0 ? view_menu_fmts(st, &fmts) : 0;
    }

    if (m->kind == VIEW_MENU_PRESET) {
        if (col == 0)
            return view_nb_preset_families;
        if (col == 1 && m->sel[0] >= 0 &&
            m->sel[0] < view_nb_preset_families)
            return view_preset_families[m->sel[0]].nb_presets;
        return 0;
    }

    if (m->kind == VIEW_MENU_COLOUR) {
        if (col == 0)
            return view_menu_colour_groups();
        if (col == 1 && m->sel[0] >= 0) {
            view_menu_colour_group(m->sel[0], &n);
            return n > 1 ? n : 0;
        }
        return 0;
    }

#if CINEVIEW_HAVE_LUTS
    switch (col) {
    case 0:
        return 2;
    case 1:
        if (m->sel[0] < 0)
            return 0;
        return m->sel[0] == 0 ? view_luts_film_category_count()
                              : view_luts_look_category_count();
    case 2:
        if (m->sel[0] < 0 || m->sel[1] < 0)
            return 0;
        if (m->sel[0] == 0) {
            fc = view_lut_film_category(m->sel[1]);
            return fc ? fc->n_films : 0;
        }
        lc = view_lut_look_category(m->sel[1]);
        return lc ? lc->n_luts : 0;
    case 3:
        f = view_menu_film(st);
        return f ? f->n_luts : 0;
    default:
        return 0;
    }
#else
    (void)col;
    return 0;
#endif
}

/* What an entry of a leaf column stands for, or -1: a rendering for the
 * film menu, a preset for the codec one, a colour space for the third. */
static int view_menu_lut(const struct view_state *st, int col, int i)
{
    const struct view_lut_menu *m = &st->menu;
#if CINEVIEW_HAVE_LUTS
    const ViewLutLookCategory *lc;
    const ViewLutFilm *f;
#endif
    int first, n;

    if (!view_menu_leaf(st, col) || i < 0 || i >= view_menu_count(st, col))
        return -1;

    if (m->kind == VIEW_MENU_PRESET || m->kind == VIEW_MENU_PIXFMT)
        return i;

    if (m->kind == VIEW_MENU_COLOUR) {
        first = view_menu_colour_group(col == 0 ? i : m->sel[0], &n);
        if (first < 0)
            return -1;
        return col == 0 ? first : first + i;
    }

#if CINEVIEW_HAVE_LUTS
    if (col == 2) {
        lc = view_lut_look_category(m->sel[1]);
        return lc && lc->luts ? (int)lc->luts[i] : -1;
    }

    f = view_menu_film(st);
    return f ? f->first_lut + i : -1;
#else
    return -1;
#endif
}

/* Why the entry cannot be chosen, or NULL. A greyed row still shows and
 * still highlights: what it says is the point. */
static const char *view_menu_why_not(const struct view_state *st, int col, int i)
{
    const struct view_lut_menu *m = &st->menu;
    const struct view_preset_sel *s = &st->look.video_preset;
    const enum AVPixelFormat *fmts;
    int ix;

    if (m->kind == VIEW_MENU_PIXFMT)
        return i >= 0 && i < view_menu_fmts(st, &fmts)
             ? view_presets_fmt_why_not(s->family, s->preset, fmts[i]) : NULL;

    if (m->kind == VIEW_MENU_PRESET)
        return col == 0 ? view_presets_family_why_not(i)
                        : view_presets_preset_why_not(m->sel[0], i);

    /* A colour the codec in use cannot write is not greyed: picking it
     * takes the codec that can with it. Only one no codec here can write
     * is. */
    if (m->kind == VIEW_MENU_COLOUR) {
        struct view_preset_sel t = *s;

        ix = view_menu_lut(st, col, i);
        if (ix < 0 && col == 0)
            ix = view_menu_colour_group(i, &(int) { 0 });
        return ix < 0 || view_presets_take_colour(&t, ix)
             ? NULL : view_presets_colour_why_not(s->family, s->preset, ix);
    }

    return NULL;
}

static void view_menu_label(const struct view_state *st, int col, int i,
                            char *buf, size_t n)
{
    const struct view_lut_menu *m = &st->menu;
#if CINEVIEW_HAVE_LUTS
    const ViewLutFilmCategory *fc;
    const ViewLutLookCategory *lc;
    const ViewLutFilm *f;
    const ViewLut *l;
#endif
    const enum AVPixelFormat *fmts;
    int ix;

    buf[0] = '\0';

    if (m->kind == VIEW_MENU_PIXFMT) {
        if (i >= 0 && i < view_menu_fmts(st, &fmts))
            view_presets_fmt_label(fmts[i], buf, n);
        return;
    }

    if (m->kind == VIEW_MENU_PRESET) {
        if (col == 0 && i < view_nb_preset_families)
            snprintf(buf, n, "%s", view_preset_families[i].name);
        else if (col == 1 && m->sel[0] >= 0 &&
                 m->sel[0] < view_nb_preset_families &&
                 i < view_preset_families[m->sel[0]].nb_presets)
            snprintf(buf, n, "%s",
                     view_preset_families[m->sel[0]].presets[i].name);
        return;
    }

    if (m->kind == VIEW_MENU_COLOUR) {
        if (col == 0) {
            int cnt;

            ix = view_menu_colour_group(i, &cnt);
            if (ix >= 0)
                snprintf(buf, n, "%s", view_colours[ix].group);
        } else {
            ix = view_menu_lut(st, col, i);
            if (ix >= 0)
                snprintf(buf, n, "%s", view_colours[ix].name);
        }
        return;
    }

#if CINEVIEW_HAVE_LUTS
    switch (col) {
    case 0:
        snprintf(buf, n, "%s", i == 0 ? "Films" : "Looks");
        return;

    case 1:
        if (m->sel[0] == 0) {
            fc = view_lut_film_category(i);
            if (fc)
                snprintf(buf, n, "%s", fc->name);
        } else {
            lc = view_lut_look_category(i);
            if (lc)
                snprintf(buf, n, "%s", lc->name);
        }
        return;

    case 2:
        if (m->sel[0] == 0) {
            fc = view_lut_film_category(m->sel[1]);
            f = fc ? view_lut_film(fc->first_film + i) : NULL;
            if (f)
                snprintf(buf, n, "%s", f->name);
            return;
        }
        /* A look cuts across films, so without the stock on the row the
         * rendering's name says nothing about what it is emulating. */
        l = view_lut_get(view_menu_lut(st, col, i));
        f = l ? view_lut_film(l->film) : NULL;
        if (l && f)
            snprintf(buf, n, "%s \xc2\xb7 %s", l->name, f->name);
        return;

    case 3:
        l = view_lut_get(view_menu_lut(st, col, i));
        if (l && l->label[0])
            snprintf(buf, n, "%s \xc2\xb7 %s", l->name, l->label);
        else if (l)
            snprintf(buf, n, "%s", l->name);
        return;

    default:
        return;
    }
#else
    (void)col;
#endif
}

/* What choosing entry `i` of leaf column `col` does. */
static void view_menu_select(struct view_state *st, int col, int i)
{
    struct view_lut_menu *m = &st->menu;
    struct view_preset_sel *s = &st->look.video_preset;
    const enum AVPixelFormat *fmts;
    int ix = view_menu_lut(st, col, i);

    if (ix < 0 || view_menu_why_not(st, col, i))
        return;

    switch (m->kind) {
    case VIEW_MENU_PIXFMT:
        if (ix >= view_menu_fmts(st, &fmts))
            break;
        s->pix_fmt = fmts[ix];
        /* Stamped with the family it was chosen under: the row falls back
         * to the preset's own pin the moment the family moves, since the
         * next family's encoder need not take this format at all. */
        s->pix_fmt_family = s->family;
        s->set = true;
        break;
    case VIEW_MENU_PRESET:
        s->family = m->sel[0];
        s->preset = ix;
        s->set = true;
        view_presets_family_defaults(s);
        break;
    case VIEW_MENU_COLOUR:
        view_presets_take_colour(s, ix);
        break;
    default:
#if CINEVIEW_HAVE_LUTS
        view_lut_select_film(st, ix);
#endif
        break;
    }
}

/* The entry that is in use, for the mark the drawing puts behind it. */
static int view_menu_current(const struct view_state *st, int col)
{
    const struct view_lut_menu *m = &st->menu;
    const struct view_preset_sel *s = &st->look.video_preset;
    const enum AVPixelFormat *fmts;
    int first, n;

    switch (m->kind) {
    case VIEW_MENU_PIXFMT:
        n = view_menu_fmts(st, &fmts);
        /* A scene-referred export's one format is the one in use. */
        if (view_colours[CRAW_CLAMP(s->colour, 0, VIEW_NB_COLOURS - 1)].scene)
            return n ? 0 : -1;
        for (int i = 0; i < n; i++)
            if (fmts[i] == view_presets_sel_fmt(s))
                return i;
        return -1;
    case VIEW_MENU_PRESET:
        return col == 0 ? s->family : (m->sel[0] == s->family ? s->preset : -1);
    case VIEW_MENU_COLOUR:
        if (col == 0) {
            for (int g = 0; g < view_menu_colour_groups(); g++) {
                first = view_menu_colour_group(g, &n);
                if (first >= 0 && s->colour >= first && s->colour < first + n)
                    return g;
            }
            return -1;
        }
        first = m->sel[0] >= 0 ? view_menu_colour_group(m->sel[0], &n) : -1;
        return first >= 0 && s->colour >= first && s->colour < first + n
             ? s->colour - first : -1;
    default:
        /* A rendering is only findable by walking the leaf column, since
         * the index says nothing about which entry holds it. */
        if (st->look.film_sim < 0 || !view_menu_leaf(st, col))
            return -1;
        n = view_menu_count(st, col);
        for (int i = 0; i < n; i++)
            if (view_menu_lut(st, col, i) == st->look.film_sim)
                return i;
        return -1;
    }
}

static float view_menu_width(struct nk_context *ctx,
                             const struct view_state *st, int col)
{
    const struct nk_user_font *font = ctx->style.font;
    int n = view_menu_count(st, col);
    float text = 0.0f, w;

    for (int i = 0; i < n; i++) {
        char buf[192];
        float tw;

        view_menu_label(st, col, i, buf, sizeof(buf));
        tw = font->width(font->userdata, font->height, buf, (int)strlen(buf));
        if (tw > text)
            text = tw;
    }

    w = text + 2 * VIEW_MENU_XPAD + 6.0f +
        (view_menu_leaf(st, col) ? 0.0f : VIEW_MENU_ARROW);

    return CRAW_CLAMP(w, VIEW_MENU_MIN_W, VIEW_MENU_MAX_W);
}

/* Places the columns. The first hangs off the panel row, each next one
 * beside the entry that opened it, and none of them may leave the window:
 * the cascade runs left when there is no room on the right, and a column
 * taller than the window scrolls instead of growing. */
static void view_menu_layout(struct nk_context *ctx, struct view_state *st)
{
    struct view_lut_menu *m = &st->menu;
    float winw = (float)st->gpu.ui_w;
    float winh = (float)st->gpu.ui_h;
    const float margin = VIEW_UI_MARGIN;
    int dir;

    /* How far the highlighted path reaches. A column opens only once its
     * parent entry is highlighted and actually has children. */
    m->cols = 1;
    for (int c = 0; c + 1 < VIEW_MENU_COLS; c++) {
        if (m->sel[c] < 0 || view_menu_leaf(st, c) ||
            view_menu_count(st, c + 1) <= 0)
            break;
        m->cols = c + 2;
    }

    dir = (m->ax + m->aw + view_menu_width(ctx, st, 0) <= winw - margin) ? 1 : -1;
    /* The pointer pass needs it too: "towards the submenu" is whichever
     * way the cascade runs. */
    m->dir = dir;

    for (int c = 0; c < m->cols; c++) {
        float w = view_menu_width(ctx, st, c);
        float full = (float)view_menu_count(st, c) * VIEW_MENU_ROW +
                     2 * VIEW_MENU_PAD;
        float h = CRAW_MIN(full, winh - 2 * margin);
        float maxs = CRAW_MAX(0.0f, full - h);
        float s, x, y;

        if (c == 0) {
            x = dir > 0 ? m->ax + m->aw : m->ax - w;
            y = m->ay;
        } else {
            x = dir > 0 ? m->x[c - 1] + m->w[c - 1] : m->x[c - 1] - w;
            /* Line the column up with the row that opened it, the way a
             * submenu hangs off its parent entry. */
            y = m->y[c - 1] + VIEW_MENU_PAD - m->scroll[c - 1] +
                (float)m->sel[c - 1] * VIEW_MENU_ROW;
        }

        m->x[c] = CRAW_CLAMP(x, margin, CRAW_MAX(margin, winw - margin - w));
        m->y[c] = CRAW_CLAMP(y, margin, CRAW_MAX(margin, winh - margin - h));
        m->w[c] = w;
        m->h[c] = h;

        s = CRAW_CLAMP(m->scroll[c], 0.0f, maxs);
        /* A column opened on the current selection has to show it, or the
         * menu opens scrolled to the top of a list twenty-six entries long
         * with the highlighted one somewhere below the fold. */
        if (m->sel[c] >= 0) {
            float top = (float)m->sel[c] * VIEW_MENU_ROW;
            float inner = h - 2 * VIEW_MENU_PAD;

            if (top < s)
                s = top;
            else if (top + VIEW_MENU_ROW > s + inner)
                s = top + VIEW_MENU_ROW - inner;
            s = CRAW_CLAMP(s, 0.0f, maxs);
        }
        m->scroll[c] = s;
    }
}

/* The rectangle of entry `i` of column `c`, in window pixels. */
static struct nk_rect view_menu_entry(const struct view_lut_menu *m, int c, int i)
{
    return nk_rect(m->x[c] + 1.0f,
                   m->y[c] + VIEW_MENU_PAD +
                   (float)i * VIEW_MENU_ROW - m->scroll[c],
                   m->w[c] - 2.0f, VIEW_MENU_ROW);
}

static void view_menu_descend(struct view_state *st, int col, int i)
{
    struct view_lut_menu *m = &st->menu;

    if (m->sel[col] == i)
        return;

    m->sel[col] = i;
    /* A different parent invalidates everything below it. */
    for (int k = col + 1; k < VIEW_MENU_COLS; k++) {
        m->sel[k] = -1;
        m->scroll[k] = 0.0f;
    }
}

/* Pointer pass: hover moves the highlight and opens the next column, a
 * click selects a rendering or opens a submenu at once, the wheel scrolls
 * the column under the pointer. Returns the rendering to select, or -1.
 *
 * Hover is deliberately not immediate. Sweeping the pointer diagonally
 * from an entry to something in the submenu it opened crosses that entry's
 * siblings, and a menu that switched on every crossing would collapse the
 * column being aimed at - which is what made this feel finicky. So an
 * entry has to be rested on before its submenu opens, and for a moment
 * after one opens the siblings are ignored while the pointer is still
 * travelling towards it. */
/* A click on a row that opens another column: the film menu just walks
 * into it, but the export's two treat the parent as a choice of its own -
 * clicking a family takes that family's default preset, clicking a colour
 * group takes the first entry under it - which is what the row above the
 * menu says it does. Returns true when that settled it. */
static bool view_menu_click_parent(struct view_state *st, int col, int i)
{
    struct view_lut_menu *m = &st->menu;
    struct view_preset_sel *s = &st->look.video_preset;
    int first, n;

    if (m->kind == VIEW_MENU_PRESET && col == 0) {
        if (view_presets_family_why_not(i))
            return true;            /* greyed: eat the click, change nothing */
        s->family = i;
        s->preset = view_preset_families[i].def;
        s->set = true;
        view_presets_family_defaults(s);
        return true;
    }

    if (m->kind == VIEW_MENU_COLOUR && col == 0) {
        first = view_menu_colour_group(i, &n);
        if (first >= 0)
            view_presets_take_colour(s, first);
        return true;
    }

    return false;
}

static int view_menu_pointer(struct nk_context *ctx, struct view_state *st,
                             bool *over)
{
    struct view_lut_menu *m = &st->menu;
    struct nk_input *in = &ctx->input;
    struct nk_vec2 p = in->mouse.pos;
    int64_t now = av_gettime_relative();
    float dx = p.x - m->prev_x;
    int hit_col = -1, hit_entry = -1;
    bool clicked = false;
    int pick = -1;

    *over = false;

    for (int c = 0; c < m->cols; c++) {
        struct nk_rect r = nk_rect(m->x[c], m->y[c], m->w[c], m->h[c]);
        int n = view_menu_count(st, c);

        if (!NK_INBOX(p.x, p.y, r.x, r.y, r.w, r.h))
            continue;
        *over = true;

        /* A column taller than the window takes the wheel itself, so the
         * picture underneath neither zooms nor scrolls. */
        if (in->mouse.scroll_delta.y != 0.0f) {
            float full = (float)n * VIEW_MENU_ROW + 2 * VIEW_MENU_PAD;

            m->scroll[c] = CRAW_CLAMP(m->scroll[c] -
                                      in->mouse.scroll_delta.y * VIEW_MENU_ROW * 2.0f,
                                      0.0f, CRAW_MAX(0.0f, full - m->h[c]));
            in->mouse.scroll_delta.y = 0.0f;
        }

        for (int i = 0; i < n; i++) {
            struct nk_rect e = view_menu_entry(m, c, i);

            if (p.y < e.y || p.y >= e.y + e.h)
                continue;
            /* Only rows fully inside the column count, or the half row
             * peeking out of a scrolled list would take clicks. */
            if (e.y < r.y || e.y + e.h > r.y + r.h)
                break;

            hit_col = c;
            hit_entry = i;
            clicked = nk_input_is_mouse_click_in_rect(in, NK_BUTTON_LEFT, e) != 0;
            break;
        }
    }

    /* How long the pointer has been where it is. */
    if (hit_col != m->hover_col || hit_entry != m->hover_entry) {
        m->hover_col = hit_col;
        m->hover_entry = hit_entry;
        m->hover_since = now;
    }
    m->guarded = false;

    if (hit_col >= 0) {
        bool leaf = view_menu_leaf(st, hit_col);

        if (clicked) {
            /* A click is unambiguous: act on it at once, whatever the
             * hover timers were about to do. */
            if (leaf) {
                view_menu_select(st, hit_col, hit_entry);
                pick = 1;
            } else if (view_menu_click_parent(st, hit_col, hit_entry)) {
                pick = 1;
            } else {
                view_menu_descend(st, hit_col, hit_entry);
                m->guard_col = hit_col;
                m->guard_until = now + VIEW_MENU_GUARD_US;
            }
        } else if (m->sel[hit_col] != hit_entry) {
            /* Still travelling towards the submenu this column opened:
             * the siblings on the way do not get to close it. */
            bool guarded = hit_col == m->guard_col && now < m->guard_until &&
                           dx * (float)m->dir > 0.0f;

            m->guarded = guarded;
            if (leaf) {
                /* Nothing to protect and nothing to open, so a leaf's
                 * highlight simply follows the pointer. */
                view_menu_descend(st, hit_col, hit_entry);
            } else if (!guarded && now - m->hover_since >= VIEW_MENU_OPEN_US) {
                view_menu_descend(st, hit_col, hit_entry);
                m->guard_col = hit_col;
                m->guard_until = now + VIEW_MENU_GUARD_US;
            }
        }
    }

    m->prev_x = p.x;
    m->prev_y = p.y;

    return pick;
}

bool view_ui_menu_due(const struct view_state *st, int64_t now)
{
    const struct view_lut_menu *m = &st->menu;

    if (!m->open || m->hover_col < 0 || m->hover_entry < 0 ||
        view_menu_leaf(st, m->hover_col))
        return false;

#if CINEVIEW_HAVE_LUTS
    /* A film's preview, due a moment after the pointer stopped on it and
     * not yet shown. */
    if (m->kind == VIEW_MENU_FILM && m->sel[0] == 0 && m->hover_col == 2 &&
        !m->guarded && now - m->hover_since >= VIEW_MENU_PEEK_US) {
        int ix = view_menu_film_base(view_menu_film_at(st, m->hover_entry));

        if (ix >= 0 && ix != st->preview_sim && ix != st->look.film_sim)
            return true;
    }
#endif

    if (m->sel[m->hover_col] == m->hover_entry)
        return false;
    if (m->guarded)
        return now >= m->guard_until;
    return now - m->hover_since >= VIEW_MENU_OPEN_US;
}

static void view_menu_draw(struct nk_context *ctx, struct view_state *st)
{
    struct view_lut_menu *m = &st->menu;
    const struct nk_user_font *font = ctx->style.font;

    nk_style_push_vec2(ctx, &ctx->style.window.padding, nk_vec2(0.0f, 0.0f));
    nk_style_push_style_item(ctx, &ctx->style.window.fixed_background,
                             nk_style_item_color(nk_rgba(34, 34, 40, 255)));

    for (int c = 0; c < m->cols; c++) {
        struct nk_rect r = nk_rect(m->x[c], m->y[c], m->w[c], m->h[c]);
        struct nk_command_buffer *out;
        int n = view_menu_count(st, c);
        bool leaf = view_menu_leaf(st, c);
        char name[32];

        snprintf(name, sizeof(name), "cineview-lutmenu-%d", c);
        if (!nk_begin(ctx, name, r,
                      NK_WINDOW_BORDER | NK_WINDOW_NO_SCROLLBAR |
                      NK_WINDOW_NO_INPUT)) {
            nk_end(ctx);
            continue;
        }
        out = nk_window_get_canvas(ctx);

        int cur = view_menu_current(st, c);

        for (int i = 0; i < n; i++) {
            struct nk_rect e = view_menu_entry(m, c, i);
            struct nk_rect t;
            char buf[192];
            bool on = m->sel[c] == i;
            bool grey = view_menu_why_not(st, c, i) != NULL;

            if (e.y + e.h <= r.y || e.y >= r.y + r.h)
                continue;           /* scrolled out of the column */

            view_menu_label(st, c, i, buf, sizeof(buf));
            /* The entry actually in use gets a mark of its own, so it
             * stays findable while the highlight is off wandering. */
            if (i == cur)
                nk_fill_rect(out, e, 2.0f, nk_rgba(40, 62, 52, 255));
            if (on)
                nk_fill_rect(out, e, 2.0f, nk_rgba(52, 66, 96, 255));

            t = nk_rect(e.x + VIEW_MENU_XPAD, e.y, e.w - 2 * VIEW_MENU_XPAD, e.h);
            nk_draw_text(out, t, buf, (int)strlen(buf), font,
                         nk_rgba(0, 0, 0, 0),
                         grey ? nk_rgba(126, 126, 132, 255)
                              : nk_rgba(226, 226, 232, 255));

            /* The mark that says this entry opens another column. Plain
             * ASCII: the atlas bakes Latin-1 and the ellipsis, whichever
             * face it holds, so a typographic arrow would come out as a
             * missing glyph. */
            if (!leaf) {
                const char *arrow = ">";

                t = nk_rect(e.x + e.w - VIEW_MENU_ARROW, e.y,
                            VIEW_MENU_ARROW - VIEW_MENU_PAD, e.h);
                nk_draw_text(out, t, arrow, (int)strlen(arrow), font,
                             nk_rgba(0, 0, 0, 0), nk_rgba(150, 150, 160, 255));
            }
        }

        /* A scrolled column says so, or a list cut off at the window edge
         * looks like the whole list. */
        if ((float)n * VIEW_MENU_ROW + 2 * VIEW_MENU_PAD > r.h) {
            float full = (float)n * VIEW_MENU_ROW + 2 * VIEW_MENU_PAD;
            float frac = r.h / full;
            struct nk_rect bar = nk_rect(r.x + r.w - 3.0f,
                                         r.y + r.h * (m->scroll[c] / full),
                                         2.0f, r.h * frac);

            nk_fill_rect(out, bar, 1.0f, nk_rgba(110, 110, 124, 255));
        }

        nk_end(ctx);
    }

    nk_style_pop_style_item(ctx);
    nk_style_pop_vec2(ctx);
}

void view_ui_menu_open(struct view_state *st, enum view_menu_kind kind)
{
    struct view_lut_menu *m = &st->menu;
#if CINEVIEW_HAVE_LUTS
    const ViewLut *l = view_lut_get(st->look.film_sim);
#endif

    if (m->open)
        return;

    /* The film simulation is one of the three things a scene-referred
     * export takes away, and a cascade that opened over a greyed row would
     * be offering a choice that changes neither the file nor the screen.
     * A locked panel has no rows a cascade could change anything from. */
    if (kind == VIEW_MENU_FILM && view_export_is_scene(st))
        return;
    if (view_ui_locked(st))
        return;

    m->open = true;
    m->kind = kind;
    m->cols = 1;
    for (int c = 0; c < VIEW_MENU_COLS; c++) {
        m->sel[c] = -1;
        m->scroll[c] = 0.0f;
    }

    /* No stale hover timers from the last time it was open, and no guard
     * protecting a column that no longer exists. */
    m->hover_col = -1;
    m->hover_entry = -1;
    m->hover_since = av_gettime_relative();
    m->guard_col = -1;
    m->guard_until = 0;

    /* Open on what is already selected, so the columns come up showing
     * the current setting rather than the top of the list. */
    if (kind == VIEW_MENU_PRESET) {
        m->sel[0] = st->look.video_preset.family;
        m->sel[1] = st->look.video_preset.preset;
        return;
    }
    if (kind == VIEW_MENU_COLOUR) {
        m->sel[0] = view_menu_current(st, 0);
        m->sel[1] = m->sel[0] >= 0 ? view_menu_current(st, 1) : -1;
        return;
    }
    if (kind == VIEW_MENU_PIXFMT) {
        m->sel[0] = view_menu_current(st, 0);
        return;
    }

#if CINEVIEW_HAVE_LUTS
    if (l) {
        const ViewLutFilm *f = view_lut_film(l->film);
        const ViewLutFilmCategory *c = f ? view_lut_film_category(f->category) : NULL;

        if (f && c) {
            m->sel[0] = 0;
            m->sel[1] = f->category;
            m->sel[2] = 0;
            for (int i = 0; i < c->n_films; i++) {
                const ViewLutFilm *g = view_lut_film(c->first_film + i);

                if (g == f) {
                    m->sel[2] = i;
                    break;
                }
            }
            m->sel[3] = st->look.film_sim - f->first_lut;
        }
    }
#endif
}

void view_ui_menu_close(struct view_state *st)
{
    st->menu.open = false;
    st->menu.cols = 0;
    /* A pick has already taken the preview over; anything else closing
     * the menu puts the chosen rendering back. */
    view_lut_preview(st, -1);
}

bool view_ui_menu_covers(const struct view_state *st, float x, float y)
{
    const struct view_lut_menu *m = &st->menu;
    float sx = st->win_w > 0 ? (float)st->gpu.ui_w / (float)st->win_w : 1.0f;
    float sy = st->win_h > 0 ? (float)st->gpu.ui_h / (float)st->win_h : 1.0f;

    if (!m->open)
        return false;

    /* SDL reports window points, the panel and the menu live in drawable
     * pixels; the two differ under a fractional scale. */
    x *= sx;
    y *= sy;

    for (int c = 0; c < m->cols; c++)
        if (NK_INBOX(x, y, m->x[c], m->y[c], m->w[c], m->h[c]))
            return true;

    return false;
}

/* One panel row that opens a cascade: a label, a button showing what is
 * selected, and the anchor the columns hang off. The menu itself is built
 * after the panel so that it draws on top. */
static void view_menu_row(struct nk_context *ctx, struct view_state *st,
                          enum view_menu_kind kind, const char *label,
                          const char *value)
{
    struct view_lut_menu *m = &st->menu;
    struct nk_rect content = nk_window_get_content_region(ctx);
    struct nk_rect b;

    nk_label(ctx, label, NK_TEXT_LEFT);

    b = nk_widget_bounds(ctx);

    /* Click to toggle, never hover: a menu that dropped open whenever the
     * pointer crossed the row on its way somewhere else was the single
     * most irritating thing about it. */
    if (nk_button_label(ctx, value)) {
        if (m->open && m->kind == kind) {
            view_ui_menu_close(st);
        } else {
            view_ui_menu_close(st);
            view_ui_menu_open(st, kind);
        }
    }

    /* The anchor is the whole row rather than the button, so the cascade
     * opens flush with the panel's edge instead of halfway across it. It
     * is only written by whichever row owns the open menu, or two rows
     * would fight over where the columns go. */
    if (m->open && m->kind == kind) {
        m->ax = content.x;
        m->aw = content.w;
        m->ay = b.y;
        m->ah = b.h;
        m->anchored = true;
    }
}

static void view_panel_film_sim(struct nk_context *ctx, struct view_state *st)
{
    nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
    view_menu_row(ctx, st, VIEW_MENU_FILM, "film simulation",
#if CINEVIEW_HAVE_LUTS
                  view_lut_film_name(st));
#else
                  "not built in");
#endif
}

/* Runs the whole menu for this frame: pointer first so that the cascade
 * reacts in the same frame the pointer moved, then the layout it implies,
 * then the drawing. */
static void view_menu_frame(struct nk_context *ctx, struct view_state *st)
{
    struct view_lut_menu *m = &st->menu;
    bool over = false;
    int pick;

    if (!m->open)
        return;

    /* The row the menu hangs off was not emitted this frame - its section
     * has been collapsed under it - so there is nothing to hang off and
     * nothing to draw. Closing rather than hiding, or Escape would go on
     * being swallowed by a menu nobody can see. A locked panel closes
     * it for the same reason: the columns do their own hit testing, and a
     * choice made in one would land on a setting the run has copied. */
    if (!m->anchored || view_ui_locked(st)) {
        view_ui_menu_close(st);
        return;
    }

    pick = view_menu_pointer(ctx, st, &over);
    if (pick > 0) {
        view_ui_menu_close(st);
        return;
    }

    /* Merely moving the pointer off the columns does not close anything:
     * the menu is opened by a click and it stays until something says
     * otherwise. A click outside it is one of those things, and so is the
     * row itself, which toggles and is therefore left alone here. */
    if (!over && ctx->input.mouse.buttons[NK_BUTTON_LEFT].clicked &&
        !NK_INBOX(ctx->input.mouse.buttons[NK_BUTTON_LEFT].clicked_pos.x,
                  ctx->input.mouse.buttons[NK_BUTTON_LEFT].clicked_pos.y,
                  m->ax, m->ay, m->aw, m->ah)) {
        view_ui_menu_close(st);
        return;
    }

    view_menu_layout(ctx, st);

    /* The film menu's highlight is a preview: whatever rendering the leaf
     * column's rests on is shown while it does. The leaf is the last
     * column open, and view_menu_lut() answers -1 for anything that is
     * not a rendering - a parent column, or a leaf with nothing
     * highlighted yet. A film with none of its renderings highlighted
     * previews as its base, so the films column can be walked without
     * going into every submenu; a look category has no such rendering,
     * and previews as nothing. */
    if (m->kind == VIEW_MENU_FILM) {
        int ix = view_menu_lut(st, m->cols - 1, m->sel[m->cols - 1]);

#if CINEVIEW_HAVE_LUTS
        /* A film under the pointer previews as its base once the pointer
         * has been on it a moment (VIEW_MENU_PEEK_US), ahead of the rest
         * that opens its column: the preview is what the hover is for, and
         * a rendering unpacks in a few milliseconds. Waiting for the column
         * meant the rest, then the idle tick that noticed it, then a
         * present - a third of a second between the pointer stopping on a
         * film and its look, where a variant showed in a tenth. The moment
         * is what keeps the films a hand crosses on its way into an open
         * column from flashing by, as is the guard. */
        if (m->sel[0] == 0 && m->hover_col == 2 && m->hover_entry >= 0 &&
            !m->guarded &&
            av_gettime_relative() - m->hover_since >= VIEW_MENU_PEEK_US)
            ix = view_menu_film_base(view_menu_film_at(st, m->hover_entry));
        else if (ix < 0)
            ix = view_menu_film_base(view_menu_film(st));
#endif
        view_lut_preview(st, ix);
    }

    view_menu_draw(ctx, st);
}

/* A slider over `n` stops that goes where it is pressed, as the scrub bar
 * does: a press anywhere on it is the stop under the pointer, and the
 * pointer moving while the button is held takes it along. nuklear's own has
 * to be caught by its knob, and in this nuklear the place it is caught is
 * half a knob to the left of where the knob is drawn - at the left end a
 * press on the knob mostly missed, so the rate row's knob, once there,
 * could hardly be picked up again; with every other frame that is where
 * the clip's own rate starts. Drawn by nuklear's slider drawing, so it
 * looks, hovers and greys out as a slider does; one stop is drawn at the
 * right end. True when the stop changed. */
static bool view_stop_slider(struct nk_context *ctx, int *pos, int n)
{
    struct nk_window *win = ctx->current;
    const struct nk_style_slider *style = &ctx->style.slider;
    struct nk_rect bounds, bar, knob;
    struct nk_input *in;
    nk_flags state = NK_WIDGET_STATE_INACTIVE;
    int at = n > 1 ? CRAW_CLAMP(*pos, 0, n - 1) : 0;
    bool moved = false;
    float t;

    if (!nk_widget(&bounds, ctx))
        return false;
    in = win->widgets_disabled || (win->layout->flags & NK_WINDOW_ROM)
         ? NULL : &ctx->input;

    bar = bounds;
    bar.x += style->padding.x;
    bar.y += style->padding.y;
    bar.w = CRAW_MAX(bar.w, 2.0f * style->padding.x + style->cursor_size.x)
          - 2.0f * style->padding.x;
    bar.h = CRAW_MAX(bar.h, 2.0f * style->padding.y) - 2.0f * style->padding.y;

    if (n > 1 && in &&
        nk_input_has_mouse_click_down_in_rect(in, NK_BUTTON_LEFT, bounds, nk_true)) {
        float u = CRAW_CLAMP((in->mouse.pos.x - bar.x) / bar.w, 0.0f, 1.0f);
        int want = (int)lrintf(u * (float)(n - 1));

        state = NK_WIDGET_STATE_ACTIVE;
        if (want != at) {
            at = want;
            moved = true;
        }
    } else if (in && nk_input_is_mouse_hovering_rect(in, bounds)) {
        state = NK_WIDGET_STATE_HOVERED;
    }

    t = n > 1 ? (float)at / (float)(n - 1) : 1.0f;
    knob.w = style->cursor_size.x;
    knob.h = style->cursor_size.y;
    knob.x = bar.x + bar.w * t - knob.w * 0.5f;
    knob.y = bar.y + bar.h * 0.5f - knob.h * 0.5f;

    if (style->draw_begin)
        style->draw_begin(&win->buffer, style->userdata);
    nk_draw_slider(&win->buffer, state, style, &bar, &knob, 0.0f, t, 1.0f);
    if (style->draw_end)
        style->draw_end(&win->buffer, style->userdata);

    *pos = at;
    return moved;
}

/* A checkbox with three states, which nuklear does not have: empty for 0,
 * and for 1 and 2 the box half filled along its diagonal, the upper left
 * triangle for 1 and the lower right for 2, in the colour a checked box is
 * filled with. Made of nuklear's own checkbox - its geometry, copied from
 * nk_do_toggle(), its click (nk_toggle_behavior()) and its drawing
 * (nk_draw_checkbox()), which the implementation in this file makes
 * available - so it looks, hovers and greys out as every other checkbox of
 * the panel does. `text` may be empty, for the box alone. A click moves
 * the state on, 0, 1, 2 and round. True when it did. */
static bool view_check3(struct nk_context *ctx, const char *text, int *state)
{
    struct nk_window *win = ctx->current;
    const struct nk_style_toggle *ts = &ctx->style.checkbox;
    const struct nk_user_font *font = ctx->style.font;
    const struct nk_input *in;
    struct nk_rect r, bounds, select, cursor, label;
    enum nk_widget_layout_states ws;
    bool clicked;

    ws = nk_widget(&r, ctx);
    if (!ws)
        return false;
    in = ws == NK_WIDGET_ROM || ws == NK_WIDGET_DISABLED ||
         win->layout->flags & NK_WINDOW_ROM ? NULL : &ctx->input;

    r.w = NK_MAX(r.w, font->height + 2 * ts->padding.x);
    r.h = NK_MAX(r.h, font->height + 2 * ts->padding.y);
    bounds = nk_rect(r.x - ts->touch_padding.x, r.y - ts->touch_padding.y,
                     r.w + 2 * ts->touch_padding.x, r.h + 2 * ts->touch_padding.y);
    select = nk_rect(r.x, r.y + r.h / 2.0f - font->height / 2.0f,
                     font->height, font->height);
    label = nk_rect(select.x + select.w + ts->spacing, select.y, 0.0f, select.w);
    label.w = NK_MAX(r.x + r.w, label.x) - label.x;
    cursor = nk_rect(select.x + ts->padding.x + ts->border,
                     select.y + ts->padding.y + ts->border,
                     select.w - (2 * ts->padding.x + 2 * ts->border),
                     select.h - (2 * ts->padding.y + 2 * ts->border));

    clicked = nk_toggle_behavior(in, bounds, &ctx->last_widget_state, nk_false);
    if (clicked)
        *state = (*state + 1) % 3;

    nk_draw_checkbox(&win->buffer, ctx->last_widget_state, ts, nk_false,
                     &label, &select, &cursor, text, (int)strlen(text), font,
                     NK_TEXT_LEFT);
    if (*state) {
        const struct nk_style_item *c =
            ctx->last_widget_state & (NK_WIDGET_STATE_HOVER | NK_WIDGET_STATE_ACTIVED)
            ? &ts->cursor_hover : &ts->cursor_normal;
        float x0 = cursor.x, y0 = cursor.y;
        float x1 = cursor.x + cursor.w, y1 = cursor.y + cursor.h;

        if (c->type == NK_STYLE_ITEM_COLOR) {
            struct nk_color col = nk_rgb_factor(c->data.color, ts->color_factor);

            if (*state == 1)
                nk_fill_triangle(&win->buffer, x0, y0, x1, y0, x0, y1, col);
            else
                nk_fill_triangle(&win->buffer, x1, y0, x1, y1, x0, y1, col);
        }
    }

    return clicked;
}

/* The rate row under the scrub bar, DESIGN.md 23: the clip shown and
 * written at a lower fixed rate, frame for frame, so that a 59.94 clip at
 * 23.976 plays two and a half times slower and exports as a 23.976 file two
 * and a half times as long. A slider over the stops - the fixed rates below
 * the clip's own, slowest at the left, and the clip's own at the right -
 * with its label at the left and what the stop means at the right, in one
 * row. The readout's column is as wide as the widest of this clip's stops,
 * so the slider does not change length under the pointer while it is
 * dragged; at 15 px Noto Sans a 59.94 clip's eight stops leave it 153 px,
 * 22 a stop.
 *
 * Moving it chooses the rate and nothing else: no seek, no pause. The run
 * loop makes the choice the rate, and does what that means for the sound
 * and the clock, in one step. A clip at 23.976 has nothing slower to go
 * to, and its row is greyed with the knob at the right, where the clip's
 * own rate is. With every other frame the stops stay the clip's, and the
 * half rate it then plays at its own speed sits among them, so the knob
 * can go either side of it (DESIGN.md 23.6). The every-other-frame box is
 * the row's last column. */
static void view_panel_rate(struct nk_context *ctx, struct view_state *st)
{
    const struct nk_user_font *f = ctx->style.font;
    AVRational stops[VIEW_RATE_MAX_STOPS];
    char text[64], shown[64];
    int own = 0, n = view_rate_stops(st, stops, VIEW_RATE_MAX_STOPS, &own);
    int cur = own, pos, half = st->half;
    float label_w, read_w = 0.0f, gap, avail, box_w;

    if (view_slowed(st))
        for (int i = 0; i < n; i++)
            if (i != own && !av_cmp_q(stops[i], st->rate))
                cur = i;

    for (int i = 0; i < n; i++) {
        view_rate_label(st, stops[i], text, sizeof(text));
        read_w = CRAW_MAX(read_w, f->width(f->userdata, f->height, text,
                                           (int)strlen(text)));
    }
    label_w = f->width(f->userdata, f->height, "rate", 4) + 12.0f;
    /* The knob at either end hangs half its width past the slider's cell,
     * which at the right end is over the readout's first digits; a column
     * of that width between the two keeps them apart. */
    gap = ctx->style.slider.cursor_size.x * 0.5f;
    read_w += 4.0f;
    /* The every-other-frame box at the right end of the row: a checkbox's
     * own square and its padding. */
    box_w = f->height + 2.0f * ctx->style.checkbox.padding.x + 4.0f;
    /* A face wider than this one still leaves the slider something to be
     * dragged along; the readout is cut with an ellipsis instead. */
    avail = nk_window_get_content_region(ctx).w
          - 2.0f * ctx->style.window.padding.x - 4.0f * ctx->style.window.spacing.x;
    read_w = CRAW_MAX(CRAW_MIN(read_w, avail - label_w - gap - box_w - 96.0f), 32.0f);

    nk_layout_row_template_begin(ctx, VIEW_ROW);
    nk_layout_row_template_push_static(ctx, label_w);
    nk_layout_row_template_push_dynamic(ctx);
    nk_layout_row_template_push_static(ctx, gap);
    nk_layout_row_template_push_static(ctx, read_w);
    nk_layout_row_template_push_static(ctx, box_w);
    nk_layout_row_template_end(ctx);

    if (n < 2)
        view_disable_begin(ctx);

    nk_label(ctx, "rate", NK_TEXT_LEFT);

    /* The choice, and only the choice: the run loop makes it the rate
     * before the clock is next read (view_rate_follow()). One stop alone
     * is drawn at the right end, greyed. */
    pos = cur;
    if (view_stop_slider(ctx, &pos, n) && pos != cur)
        st->rate_choice = pos == own ? (AVRational) { 0, 0 } : stops[pos];
    nk_spacing(ctx, 1);

    view_rate_label(st, stops[pos], text, sizeof(text));
    view_fit_text(f, nk_widget_bounds(ctx).w - 2.0f, text, shown, sizeof(shown));
    nk_label(ctx, shown, NK_TEXT_LEFT);

    if (n < 2)
        view_disable_end(st, ctx);

    /* Every frame, or every other one: the odd-numbered or the
     * even-numbered, counted from frame 0 as the readout counts. A
     * checkbox with three states, moved on by each click as by `f`: empty
     * for all frames, its upper left half filled for the odd ones and its
     * lower right half for the even ones. The readout beside it says which
     * ("14.99 fps, odd frames"). The choice and nothing else: the run loop
     * takes it up with the rate (view_rate_follow()). Live whatever the
     * slider is, since leaving frames out is what gives a clip with
     * nothing slower than its own rate stops to move to. */
    if (view_check3(ctx, "", &half))
        st->half = (enum view_half)half;
}

/* The scrub bar. Not nuklear's slider, whose knob has to be caught before
 * anything moves and then goes in whole steps of the drag: a press anywhere
 * on the bar is the frame under the pointer, the picture stays on it for as
 * long as the button is held, and the pointer moving while held takes it
 * along. Playback stops while the bar is held and goes on from wherever it
 * is let go if the clip was playing before. Drawn with nuklear's own slider
 * drawing, so it looks like every other slider on the panel, in nuklear's
 * own geometry: the knob's centre runs from the bar's left edge to its
 * right, as every slider's does. */
static void view_scrub_bar(struct nk_context *ctx, struct view_state *st,
                           int64_t total)
{
    struct nk_window *win = ctx->current;
    const struct nk_style_slider *style = &ctx->style.slider;
    struct nk_rect bounds, bar, knob;
    struct nk_input *in;
    nk_flags state = NK_WIDGET_STATE_INACTIVE;
    bool held = false;
    int64_t at = CRAW_CLAMP(st->index, 0, total - 1);

    if (!nk_widget(&bounds, ctx))
        return;
    in = win->widgets_disabled || (win->layout->flags & NK_WINDOW_ROM)
         ? NULL : &ctx->input;

    bar = bounds;
    bar.x += style->padding.x;
    bar.y += style->padding.y;
    bar.w = CRAW_MAX(bar.w, 2.0f * style->padding.x + style->cursor_size.x)
          - 2.0f * style->padding.x;
    bar.h = CRAW_MAX(bar.h, 2.0f * style->padding.y) - 2.0f * style->padding.y;

    /* Held: the button is down and went down on the bar, wherever the
     * pointer is now - a drag that wanders off the bar keeps scrubbing. */
    if (in && nk_input_has_mouse_click_down_in_rect(in, NK_BUTTON_LEFT, bounds,
                                                    nk_true)) {
        double t = CRAW_CLAMP((in->mouse.pos.x - bar.x) / bar.w, 0.0, 1.0);
        int64_t want = (int64_t)llrint(t * (double)(total - 1));

        held = true;
        state = NK_WIDGET_STATE_ACTIVE;
        if (!st->scrubbing) {
            st->scrubbing = true;
            st->scrub_resume = st->playing;
            st->scrub_want = st->seek_to >= 0 ? st->seek_to : st->index;
            st->playing = false;
            view_decoder_readahead(&st->dec, 2);
        }
        if (want != st->scrub_want) {
            st->scrub_want = want;
            view_log(st, "scrub: frame %" PRId64 "\n", want);
            st->seek_to = want;
            st->need_frame = true;
        }
        at = want;
    } else if (in && nk_input_is_mouse_hovering_rect(in, bounds)) {
        state = NK_WIDGET_STATE_HOVERED;
    }

    /* Let go. A drag was a stream of seeks, and each would have paid for
     * the temporal denoise's pre-roll (view_step_frame()), which halves the
     * rate the picture follows the pointer at; they started the average
     * cold instead, and the frame let go on is sought once more, with its
     * pre-roll. */
    if (st->scrubbing && !held) {
        st->scrubbing = false;
        view_decoder_readahead(&st->dec, VIEW_PKTQ_MAX);
        if (st->scrub_resume)
            st->playing = true;
        st->scrub_resume = false;
        if (st->seek_to < 0 && view_temporal_strength(st) > 0.0f) {
            st->seek_to = st->index;
            st->seek_cold = true;
            st->need_frame = true;
        }
    }

    knob.w = style->cursor_size.x;
    knob.h = style->cursor_size.y;
    knob.x = bar.x + bar.w * (float)((double)at / (double)(total - 1)) - knob.w * 0.5f;
    knob.y = bar.y + bar.h * 0.5f - knob.h * 0.5f;

    if (style->draw_begin)
        style->draw_begin(&win->buffer, style->userdata);
    nk_draw_slider(&win->buffer, state, style, &bar, &knob, 0.0f, (float)at,
                   (float)(total - 1));
    if (style->draw_end)
        style->draw_end(&win->buffer, style->userdata);
}

static void view_panel_transport(struct nk_context *ctx, struct view_state *st)
{
    int64_t total = st->dec.nb_frames;

    /* No transport buttons: the keyboard is the transport. Space plays and
     * pauses, Shift with Page Up and Page Down are the ends of the clip,
     * Left and Right step five seconds and Shift with them one. The scrub
     * bar stays, because going to a frame by eye is the one thing keys are
     * bad at, and under it the rate, which no key has. */
    if (total > 1) {
        nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
        view_scrub_bar(ctx, st, total);
    }

    /* A still is not played, so there is nothing for it to be slowed to,
     * nor frames for it to leave out. */
    if (st->dec.fmt && !st->dec.still)
        view_panel_rate(ctx, st);
}

/* A labelled slider with its value beside it, the rate row's shape: label,
 * the slider, half a knob of air, the readout. Returns true when the value
 * changed. */
static bool view_slider_row(struct nk_context *ctx, const char *label,
                            float *value, float min, float max, float step,
                            const char *fmt)
{
    const struct nk_user_font *f = ctx->style.font;
    char text[32];
    float label_w, read_w, gap;
    float before = *value;

    snprintf(text, sizeof(text), fmt, max);
    read_w = f->width(f->userdata, f->height, text, (int)strlen(text));
    snprintf(text, sizeof(text), fmt, min);
    read_w = CRAW_MAX(read_w, f->width(f->userdata, f->height, text,
                                       (int)strlen(text))) + 4.0f;
    label_w = f->width(f->userdata, f->height, label, (int)strlen(label)) + 12.0f;
    gap = ctx->style.slider.cursor_size.x * 0.5f;

    nk_layout_row_template_begin(ctx, VIEW_ROW);
    nk_layout_row_template_push_static(ctx, label_w);
    nk_layout_row_template_push_dynamic(ctx);
    nk_layout_row_template_push_static(ctx, gap);
    nk_layout_row_template_push_static(ctx, read_w);
    nk_layout_row_template_end(ctx);

    nk_label(ctx, label, NK_TEXT_LEFT);
    nk_slider_float(ctx, min, value, max, step);
    nk_spacing(ctx, 1);
    nk_labelf(ctx, NK_TEXT_LEFT, fmt, *value);

    return *value != before;
}

/* One section for both kinds of clip. A raw clip is balanced by libcraw
 * from the sensor's ratios and the camera's tables; a developed one is
 * rebalanced from the temperature it records, on a locus of its own in its
 * own linear RGB (videowb.c), which gives it the same three modes, the same
 * pick and the same two numbers in the same units. Only the developed one
 * has a switch - a raw clip's balance is always applied - and it lights
 * when anything here is touched. */
static void view_panel_wb(struct nk_context *ctx, struct view_state *st)
{
    struct view_look *lk = &st->look;
    const enum view_wb mode = lk->wb;
    const CRAWIlluminant manual = lk->manual;
    const float ambience = st->awb.params.ambience;

    nk_layout_row_dynamic(ctx, VIEW_ROW, 3);
    if (nk_option_label(ctx, "As shot", lk->wb == VIEW_WB_AS_SHOT))
        lk->wb = VIEW_WB_AS_SHOT;
    if (nk_option_label(ctx, "Auto", lk->wb == VIEW_WB_AUTO))
        lk->wb = VIEW_WB_AUTO;
    /* Manual starts from the balance on screen, whichever mode put it
     * there, so that picking it changes nothing until a number moves. */
    if (nk_option_label(ctx, "Manual", lk->wb == VIEW_WB_MANUAL) &&
        lk->wb != VIEW_WB_MANUAL) {
        CRAWIlluminant cur;

        /* In the rows' own resolution, whole Kelvin and a ten-thousandth
         * of tint, so that the fields read what was picked up rather than
         * 3099.99 and -0.00; the recorded temperature is whole already. */
        if (view_wb_applied(st, &cur)) {
            lk->manual.cct = roundf(cur.cct);
            lk->manual.tint = roundf(cur.tint * 1e4f) / 1e4f + 0.0f;
        }
        lk->wb = VIEW_WB_MANUAL;
    }

    /* The spot white balance, a manual balance's tool and so a row of
     * Manual alone; leaving Manual with a pick armed calls it off. The
     * button arms it and the next click on the picture samples, so while
     * armed it is lit - its title and border in the colour the section
     * switches light up in - and the pointer over the picture is a
     * crosshair; a second press or Escape disarms. What a pick finds goes
     * into the manual rows below, which is the readout; what it could not
     * find is the one word beside the button, and nothing else on the
     * panel says anything about it. */
    if (lk->wb != VIEW_WB_MANUAL) {
        if (st->pick.armed)
            view_pick_arm(st, false);
    } else {
        struct nk_style_button bs = ctx->style.button;

        if (st->pick.armed) {
            bs.border_color = VIEW_SWITCH_ON;
            bs.text_normal = bs.text_hover = bs.text_active = VIEW_SWITCH_ON;
        }

        nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
        if (nk_button_label_styled(ctx, &bs, "pick white"))
            view_pick_arm(st, !st->pick.armed);
        nk_label(ctx, st->pick.refused, NK_TEXT_LEFT);
    }

    nk_layout_row_dynamic(ctx, VIEW_ROW, 1);

    switch (lk->wb) {
    case VIEW_WB_AS_SHOT:
        if (st->dec.raw)
            nk_labelf(ctx, NK_TEXT_LEFT, "camera: %.0f K, gains %.4f / 1 / %.4f",
                      st->prof.as_shot_cct, st->prof.as_shot_wb[0],
                      st->prof.as_shot_wb[2]);
        else if (st->dec.wb_kelvin > 0)
            nk_labelf(ctx, NK_TEXT_LEFT, "camera: %d K", st->dec.wb_kelvin);
        else
            nk_label(ctx, "camera: none recorded, taken as the space's white",
                     NK_TEXT_LEFT);
        break;

    case VIEW_WB_AUTO: {
        const CRAWAwbEstimate *e = &st->awb.last;

        if (st->awb.have_illum)
            nk_labelf(ctx, NK_TEXT_LEFT, "track: %.0f K, tint %+.4f",
                      st->awb.illum.cct, st->awb.illum.tint);
        else
            nk_label(ctx, "track: none yet, staying as shot", NK_TEXT_LEFT);

        nk_labelf(ctx, NK_TEXT_LEFT, "estimate: %s %.0f K, tint %+.4f",
                  e->valid ? "ok" : "bad", e->illuminant.cct, e->illuminant.tint);
        nk_labelf(ctx, NK_TEXT_LEFT, "conf %.2f, from %s, spread %.0f mired",
                  e->confidence, view_wb_source_name(e->source), e->spread_mired);

        view_slider_row(ctx, "ambience", &st->awb.params.ambience, 0.0f, 1.0f,
                        0.01f, "%.2f");
        break;
    }

    case VIEW_WB_MANUAL:
        nk_property_float(ctx, "#CCT (K)", 2000.0f, &lk->manual.cct, 11500.0f,
                          50.0f, 5.0f);
        nk_slider_float(ctx, 2000.0f, &lk->manual.cct, 11500.0f, 10.0f);
        nk_property_float(ctx, "#tint", -0.2f, &lk->manual.tint, 0.2f, 0.005f, 0.0005f);
        nk_slider_float(ctx, -0.2f, &lk->manual.tint, 0.2f, 0.002f);
        break;
    }

    if (!st->dec.raw &&
        (lk->wb != mode || lk->manual.cct != manual.cct ||
         lk->manual.tint != manual.tint || st->awb.params.ambience != ambience))
        st->sections.wb = true;
}

static void view_panel_exposure(struct nk_context *ctx, struct view_state *st)
{
    struct view_look *lk = &st->look;
    static char labels[64][8];
    static const char *items[64];
    static int nb_items;
    int sel = 0;

    if (!nb_items) {
        nb_items = CRAW_MIN(view_nb_iso, 64);
        for (int i = 0; i < nb_items; i++) {
            snprintf(labels[i], sizeof(labels[i]), "%d", (int)view_iso_list[i]);
            items[i] = labels[i];
        }
    }

    /* The working index defaults to the as-shot ISO, which is what rates a
     * camera original at 1.0; anything else re-rates the clip. */
    for (int i = 0; i < nb_items; i++)
        if (fabsf(view_iso_list[i] - (lk->ei > 0.0f ? lk->ei : st->prof.as_shot_ei)) < 0.5f)
            sel = i;

    /* Re-rating a clip needs an as-shot index to rate it against; without
     * one - which is every developed clip this camera writes - only the
     * bias below means anything. */
    if (st->dec.raw || st->dec.meta_as_shot_ei > 0.0f) {
        int chosen;

        nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
        nk_label(ctx, "exposure index", NK_TEXT_LEFT);
        chosen = nk_combo(ctx, items, nb_items, sel, VIEW_COMBO_ROW,
                          nk_vec2(140, 300));
        if (chosen != sel) {
            lk->ei = view_iso_list[chosen];
            st->sections.exposure = true;
        }
    }

    nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
    {
        /* A touched control switches the section on, RawTherapee's way:
         * the switch starts off and a lit one means the bias is in use. */
        float before = lk->exposure;

        nk_property_float(ctx, "#bias (stops)", -3.0f, &lk->exposure, 3.0f, 0.1f, 0.01f);
        nk_slider_float(ctx, -3.0f, &lk->exposure, 3.0f, 0.05f);
        if (lk->exposure != before)
            st->sections.exposure = true;
    }
    /* The gain the picture really has, which a bypassed section takes back
     * to the as-shot rating while the two controls above keep their
     * numbers. */
    nk_labelf(ctx, NK_TEXT_LEFT, "gain %.2f, ceiling %.0f nits",
              view_total_gain(st) * exp2f(view_exposure_bias(st)),
              view_peak_ceiling(st));
}

/* What the record says the lens is doing, in one word: the sign of the
 * correction at the corner. Barrel distortion is corrected by pulling the
 * corners in, pincushion by pushing them out. */
static const char *view_lens_kind(const struct view_lens *ln)
{
    float g = ln->corner[1];

    return g < -5.0f ? "barrel" : g > 5.0f ? "pincushion" : "near-neutral";
}

/* The "Lens" section: the geometric correction of DESIGN.md 16, driven by
 * the record every frame of a Panasonic ProRes RAW clip carries in its own
 * vendor metadata. The pipeline is always built with the pass (gpu.c), so
 * the switch is live: it reaches the next frame while playing and puts the
 * held picture through again while paused.
 *
 * The correction is the camera's, not the panel's. All this shows is what
 * the frame on screen brought with it, because a clip whose frames carry no
 * record cannot be corrected at all and the reason has to be visible rather
 * than inferred from nothing happening. On a clip that zooms while
 * recording the readout moves frame by frame, which is the record doing
 * exactly what it is for. The warp's own switch is the header's, greyed
 * there on a clip that carries no record; what is left here is the mode
 * and the readout. */
static void view_panel_lens(struct nk_context *ctx, struct view_state *st)
{
    struct view_lens *ln = &st->lens;
    bool have = ln->record.valid;

    nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
    nk_label(ctx, "lens correction", NK_TEXT_LEFT);
    nk_labelf(ctx, NK_TEXT_LEFT, "%s",
              !st->dec.raw       ? "not a raw clip"
            : !have              ? "no record"
            : ln->on             ? "on"
                                 : "off");

    /* What the warp does with the frame, DESIGN.md 16.8: fill it and lose
     * the source's corners, which is the trade the camera's own crop makes,
     * or shrink the corrected picture until all of the source fits and
     * paint the rest black. On a clip with no record the row is drawn but
     * cannot be moved. */
    {
        int sel = ln->keep_sel;

        nk_layout_row_dynamic(ctx, VIEW_ROW, 3);
        nk_label(ctx, "frame", NK_TEXT_LEFT);
        if (have) {
            if (nk_option_label(ctx, "fill", sel == 0))
                sel = 0;
            if (nk_option_label(ctx, "keep all", sel == 1))
                sel = 1;
        } else {
            nk_label(ctx, "fill", NK_TEXT_LEFT);
            nk_label(ctx, "keep all", NK_TEXT_LEFT);
        }

        ln->keep_sel = sel;
        ln->keep = sel != 0;
    }

    nk_layout_row_dynamic(ctx, VIEW_ROW, 1);

    if (have) {
        /* Short enough for the panel's width: the corner shift per channel
         * and the chromatic difference are what a glance wants. */
        nk_labelf(ctx, NK_TEXT_LEFT, "%s: corner %+.0f/%+.0f/%+.0f px RGB",
                  view_lens_kind(ln), (double)ln->corner[0],
                  (double)ln->corner[1], (double)ln->corner[2]);
        nk_labelf(ctx, NK_TEXT_LEFT, "CA R-B %+.2f px, centre %.3f, %.3f",
                  (double)(ln->corner[0] - ln->corner[2]),
                  (double)ln->record.centre[0], (double)ln->record.centre[1]);

        /* The scale is the price of keeping the corners, so it is only
         * shown while the mode that charges it is on. It is the turned
         * frame's when the Rotate section turns it. */
        if (view_lens_keep_all(st)) {
            float scale = view_warp_scale(st);

            if (scale > 0.0f)
                nk_labelf(ctx, NK_TEXT_LEFT, "keep all: scaled %.3f", (double)scale);
            else
                nk_label(ctx, "keep all: this record cannot be inverted",
                         NK_TEXT_LEFT);
        }
    } else {
        nk_label(ctx, "no lens record in this clip", NK_TEXT_LEFT);
    }

    /* Two smoothed averages of libcraw's own pass time, the way the Denoise
     * section measures its own, so the cost shown is a measurement. */
    if (st->cfg.verbose) {
        if (ln->ms_on > 0.0 && ln->ms_off > 0.0) {
            nk_labelf(ctx, NK_TEXT_LEFT, "libcraw %.2f ms on, %.2f off",
                      ln->ms_on, ln->ms_off);
            nk_labelf(ctx, NK_TEXT_LEFT, "the warp costs %.2f ms",
                      ln->ms_on - ln->ms_off);
        } else {
            nk_labelf(ctx, NK_TEXT_LEFT, "libcraw %.2f ms; toggle to measure",
                      st->gpu.last_gpu_ms);
        }
    }
}

/* The "Rotate" section: the turn of DESIGN.md 16.9, clockwise for a
 * positive angle. It is the lens warp's own resample, so it is offered for
 * raw clips, with or without a lens record, and the picture is magnified
 * just enough that the turned frame stays filled - using the corners the
 * lens correction would otherwise cut off before magnifying at all. The
 * number under the slider is how much that is, the only thing the turn
 * costs; with the Lens section keeping everything it is the shrink
 * instead. */
static void view_panel_rotate(struct nk_context *ctx, struct view_state *st)
{
    struct view_rotate *r = &st->rotate;
    float before = r->degrees;
    float scale;

    nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
    nk_property_float(ctx, "#degrees", -VIEW_ROTATE_MAX, &r->degrees, VIEW_ROTATE_MAX,
                      0.1f, 0.01f);
    nk_slider_float(ctx, -VIEW_ROTATE_MAX, &r->degrees, VIEW_ROTATE_MAX, 0.05f);

    /* A touched control switches the section on, RawTherapee's way. */
    if (r->degrees != before)
        r->on = true;

    if (view_rotation(st) == 0.0f)
        return;

    scale = view_warp_scale(st);
    if (!(scale > 0.0f))
        nk_label(ctx, "this record cannot be inverted", NK_TEXT_LEFT);
    else if (view_lens_keep_all(st))
        nk_labelf(ctx, NK_TEXT_LEFT, "scaled %.3f to keep everything", (double)scale);
    else if (scale == 1.0f)
        nk_label(ctx, "fills the frame without magnifying", NK_TEXT_LEFT);
    else
        nk_labelf(ctx, NK_TEXT_LEFT, "magnified %.3f to fill the frame", (double)scale);
}

/* The "Sharpen" section: the capture sharpening of DESIGN.md 20, the last
 * stage on the camera RGB. The pipeline is always built with it (gpu.c), so
 * every row is live. The iteration count is the one a paused picture and an
 * export get; while the clip plays the pass runs at most
 * VIEW_SHARPEN_PLAY_ITERATIONS, and pausing puts the picture through again
 * at the full count. The contrast row greys on a camera with no noise
 * model, where there is nothing to measure contrast against and the stage
 * sharpens everything. */
static void view_panel_sharpen(struct nk_context *ctx, struct view_state *st)
{
    struct view_sharpen *sh = &st->sharpen;
    bool grey = !view_denoise_model(st, NULL, NULL);

    nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
    nk_property_float(ctx, "#radius", 0.3f, &sh->radius, CRAW_SHARPEN_MAX_RADIUS, 0.05f, 0.005f);
    nk_slider_float(ctx, 0.3f, &sh->radius, CRAW_SHARPEN_MAX_RADIUS, 0.05f);
    nk_property_int(ctx, "#iterations", 1, &sh->iterations, CRAW_SHARPEN_MAX_ITERATIONS, 1, 0.2f);
    nk_slider_int(ctx, 1, &sh->iterations, CRAW_SHARPEN_MAX_ITERATIONS, 1);

    if (grey)
        view_disable_begin(ctx);
    nk_property_float(ctx, "#contrast", 0.0f, &sh->contrast, 4.0f, 0.1f, 0.01f);
    nk_slider_float(ctx, 0.0f, &sh->contrast, 4.0f, 0.05f);
    if (grey)
        view_disable_end(st, ctx);
}

/* The "Denoise" section: the raw domain stage of DESIGN.md 14, which runs
 * on the CFA planes before the debayer. The pipeline is always built with
 * the pass (gpu.c), so everything here is live: the strength reaches the
 * next frame and a change while paused puts the picture through again.
 *
 * The model is the camera's, not the panel's. All this shows is which one
 * was found for the frame's own ISO, because a clip the camera has not been
 * characterized at cannot be denoised at all and the reason has to be
 * visible rather than inferred from nothing happening. The stage's own
 * switch is the header's; what is left here is the strength, what the
 * frame on screen resolved, the temporal average in front of the stage
 * and the chroma stage after the debayer. */
static void view_panel_denoise(struct nk_context *ctx, struct view_state *st)
{
    struct view_denoise *d = &st->denoise;

    nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
    nk_labelf(ctx, NK_TEXT_LEFT, "%s",
              !d->have_model ? "no model: raw stage off"
                             : d->on ? "on, before the debayer" : "off");

    nk_property_float(ctx, "#strength", 0.0f, &d->strength, 3.0f, 0.25f, 0.01f);
    nk_slider_float(ctx, 0.0f, &d->strength, 3.0f, 0.05f);

    /* 1.0 is "remove noise at exactly the modelled level", so the numbers
     * either side of it are worth naming. Zero leaves the spatial filter
     * out, for the temporal average or the chroma stage alone. */
    nk_labelf(ctx, NK_TEXT_LEFT, "%.2f: %s", (double)d->strength,
              d->strength <= 0.0f  ? "spatial filter off"
            : d->strength < 0.95f  ? "under the modelled level"
            : d->strength < 1.05f  ? "the modelled level"
            : d->strength < 2.05f  ? "smooths across real detail"
                                   : "heavy");

    if (!d->have_model)
        nk_labelf(ctx, NK_TEXT_LEFT, "no noise model for ISO %.0f: off", (double)d->iso);
    else
        nk_labelf(ctx, NK_TEXT_LEFT, "noise a %.3g, b %.2g at ISO %.0f",
                  (double)d->model.a, (double)d->model.b, (double)d->iso);

    /* The temporal average in front of the raw stage, DESIGN.md 14.7: how
     * much of the previous frame a sample keeps where the two agree within
     * the model's noise. It needs the model, like the strength above. The
     * row stops short of 1, at which a static sample would keep its first
     * frame for ever. */
    nk_property_float(ctx, "#temporal", 0.0f, &d->temporal, 0.95f, 0.05f, 0.01f);
    nk_slider_float(ctx, 0.0f, &d->temporal, 0.95f, 0.02f);

    /* The chroma denoise after the debayer, DESIGN.md 14.4: the quarter
     * resolution chroma band replaced by its luma-guided smoothing. It
     * needs no model, so it runs whenever the section is on and this is
     * above zero. */
    nk_property_float(ctx, "#chroma", 0.0f, &d->chroma, 1.0f, 0.1f, 0.005f);
    nk_slider_float(ctx, 0.0f, &d->chroma, 1.0f, 0.02f);

    /* Two smoothed averages of libcraw's own pass time, one for the frames
     * that ran the stage and one for the frames that did not, so the cost
     * shown is a measurement and not an estimate. It only appears once the
     * clip has been played both ways. */
    if (st->cfg.verbose) {
        if (d->ms_on > 0.0 && d->ms_off > 0.0) {
            nk_labelf(ctx, NK_TEXT_LEFT, "libcraw %.2f ms on, %.2f off",
                      d->ms_on, d->ms_off);
            nk_labelf(ctx, NK_TEXT_LEFT, "the pass costs %.2f ms",
                      d->ms_on - d->ms_off);
        } else {
            nk_labelf(ctx, NK_TEXT_LEFT, "libcraw %.2f ms; toggle to measure",
                      st->gpu.last_gpu_ms);
        }
    }
}

/* The "Adjustments" section: soft light and vibrance, the two RawTherapee
 * tools adjust.c reimplements. It sits between Look and Film grain because
 * that is where the hooks sit in the render: after the film simulation the
 * Look section chooses, before the grain. The header above it is the
 * section's switch: off leaves both hooks unattached at the amounts they
 * hold. */
static void view_panel_adjust(struct nk_context *ctx, struct view_state *st)
{
    struct view_adjust *a = &st->adjustments;
    char buf[128];
    bool grey;

    grey = view_grey_begin(st, ctx,
        "not applied: the export is scene referred, and these two are "
        "display-referred operators on values in [0,1]");

    nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
    nk_labelf(ctx, NK_TEXT_LEFT, "%s",
              view_adjust_summary(a, buf, sizeof(buf)));

    /* A touched control switches the section on, RawTherapee's way: the
     * switch starts off and a lit one means an adjustment is in use. */
    const struct view_adjust before = *a;

    /* Soft light first, even though it runs last: it is the one control
     * most people came for, and the panel is not the pipeline. */
    nk_property_float(ctx, "#soft light", 0.0f, &a->softlight, 100.0f,
                      5.0f, 0.25f);
    nk_slider_float(ctx, 0.0f, &a->softlight, 100.0f, 1.0f);
    nk_property_float(ctx, "#pastels", -100.0f, &a->pastels, 100.0f,
                      5.0f, 0.25f);
    nk_slider_float(ctx, -100.0f, &a->pastels, 100.0f, 1.0f);
    nk_property_float(ctx, "#saturated", -100.0f, &a->saturated, 100.0f,
                      5.0f, 0.25f);
    nk_slider_float(ctx, -100.0f, &a->saturated, 100.0f, 1.0f);
    nk_property_float(ctx, "#threshold", 0.0f, &a->threshold, 100.0f,
                      5.0f, 0.25f);
    nk_slider_float(ctx, 0.0f, &a->threshold, 100.0f, 1.0f);

    nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
    nk_checkbox_label(ctx, "protect skin", &a->protect_skin_flag);
    a->protect_skin = a->protect_skin_flag != 0;
    nk_label(ctx, a->protect_skin ? "warm hues held back" : "every hue alike",
             NK_TEXT_LEFT);

    nk_checkbox_label(ctx, "avoid colour shift", &a->avoid_shift_flag);
    a->avoid_shift = a->avoid_shift_flag != 0;
    nk_label(ctx, a->avoid_shift ? "chroma kept in gamut" : "chroma clipped",
             NK_TEXT_LEFT);

    if (a->softlight != before.softlight || a->pastels != before.pastels ||
        a->saturated != before.saturated || a->threshold != before.threshold ||
        a->protect_skin != before.protect_skin ||
        a->avoid_shift != before.avoid_shift)
        st->sections.adjust = true;

    view_grey_end(st, ctx, grey);
}

/* How many lines a status takes in the panel's width, and how tall a row
 * that holds them is. */
static int view_status_lines(struct nk_context *ctx, const char *text,
                             float *height)
{
    const struct nk_user_font *f = ctx->style.font;
    float avail = nk_window_get_content_region(ctx).w
                - 2.0f * ctx->style.window.padding.x
                - 2.0f * ctx->style.text.padding.x;
    float tw = f->width(f->userdata, f->height, text, (int)strlen(text));
    float line = f->height + 2.0f * ctx->style.text.padding.y;
    int lines = 1;

    if (avail > 1.0f && tw > avail)
        lines = CRAW_MIN((int)ceilf(tw * 1.15f / avail), 4);
    *height = lines == 1 ? VIEW_ROW
                         : CRAW_MAX(VIEW_ROW, (float)lines * line + 4.0f);

    return lines;
}

/* A status line: one row when it fits, wrapped over as many as it takes
 * when it does not. A refusal is a sentence, and a sentence cut at the
 * panel's edge is not one anybody can act on. */
static void view_status_rows(struct nk_context *ctx, const char *text)
{
    float height;
    int lines = view_status_lines(ctx, text, &height);

    nk_layout_row_dynamic(ctx, height, 1);
    if (lines == 1)
        nk_label(ctx, text, NK_TEXT_LEFT);
    else
        nk_label_wrap(ctx, text);
}

/* The sound row of the Video block: a combo of the six modes, with the
 * ones the file's container will not carry greyed and labelled with the
 * muxer's own reason, and under it the one number the mode takes. What the
 * container is comes from the file field: the popup, which only opens once
 * the field is left, greys opus and flac under a `.mov`.
 *
 * A slowed clip is exported without sound (DESIGN.md 23.3), so while it is
 * both rows are greyed and the line under the combo says that instead of
 * what the mode does. The mode is left as it was chosen, on the combo where
 * it can be seen, and it is what the next export at the clip's own rate
 * writes. */
static void view_panel_export_audio(struct nk_context *ctx, struct view_state *st,
                                    const char *path)
{
    struct view_preset_sel *sel = &st->look.video_preset;
    const struct view_audio_spec *m;
    char label[160];
    int mode = CRAW_CLAMP(sel->audio_mode, 0, VIEW_NB_AUDIO_MODES - 1);
    bool slowed = view_slowed(st);

    if (slowed)
        view_disable_begin(ctx);

    nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
    nk_label(ctx, "audio", NK_TEXT_LEFT);

    m = &view_audio_modes[mode];
    if (m->field)
        snprintf(label, sizeof(label), "%s %d %s", m->name, sel->audio_q[mode],
                 m->field);
    else
        snprintf(label, sizeof(label), "%s", m->name);

    /* Wide enough for a refusal: the popup may run past the combo's own
     * width, and "mov: opus only supported in MP4" is what it has to say. */
    if (nk_combo_begin_label(ctx, label, nk_vec2(330, 200))) {
        nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
        for (int i = 0; i < VIEW_NB_AUDIO_MODES; i++) {
            const char *no = view_encode_audio_why_not(st, path, i);

            if (no) {
                snprintf(label, sizeof(label), "%s: %s",
                         view_audio_modes[i].name, no);
                view_disable_begin(ctx);
                nk_combo_item_label(ctx, label, NK_TEXT_LEFT);
                view_disable_end(st, ctx);
            } else if (nk_combo_item_label(ctx, view_audio_modes[i].name,
                                           NK_TEXT_LEFT)) {
                mode = i;
            }
        }
        nk_combo_end(ctx);
    }
    sel->audio_mode = mode;
    m = &view_audio_modes[mode];

    /* The number the mode takes, or what the mode is instead of one. */
    nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
    if (m->field) {
        int step = !strcmp(m->field, "level") ? 1 : 16;

        snprintf(label, sizeof(label), "#%s", m->field);
        nk_property_int(ctx, label, m->min, &sel->audio_q[mode], m->max, step,
                        (float)step * 0.2f);
        nk_label(ctx, slowed                  ? "no sound at this rate"
                    : mode == VIEW_AUDIO_OPUS ? "120 ms frames" : "", NK_TEXT_LEFT);
    } else {
        /* The second column is simply left empty: nk_spacing() over the
         * last column of a row opens a whole new row, which is a blank
         * line in the panel. */
        nk_label(ctx, slowed                 ? "no sound at this rate"
               : mode == VIEW_AUDIO_COPY ? "the clip's own track"
               : mode == VIEW_AUDIO_WAV  ? "PCM at the clip's depth"
               : "no sound", NK_TEXT_LEFT);
    }

    if (slowed)
        view_disable_end(st, ctx);
}

/* ------------------------------------------------- the export file fields */

/* A path's extension - the last dot after the last slash, dot included -
 * and where it starts; "" and the whole path when there is none. */
static const char *view_field_ext(const char *path, size_t *stem)
{
    const char *slash = strrchr(path, '/');
    const char *dot = strrchr(slash ? slash + 1 : path, '.');

    *stem = dot ? (size_t)(dot - path) : strlen(path);
    return dot ? dot : "";
}

/* The file field of an export section follows its default, `want` - the
 * clip's name and the frame or the range on screen, in ~/Processed, under
 * the section's own extension - except for what was typed into it, which
 * is told from what was put there by `def`, the default the field was
 * last given. The two parts of a name are kept differently. The
 * extension, once typed, is the user's format and stays: on this clip,
 * the next and every one after, until it is typed back. The name before
 * it, once typed, is kept on the clip it was typed on and on no other:
 * the next clip gets its own name again, under the user's extension. The
 * name used to be kept across clips as well, so a name typed for one
 * clip was the next one's too, and exporting the next wrote over the
 * first (Lynne). A name typed without an extension gets the default's
 * when the button is pressed and is not a choice of extension.
 *
 * `clip` is the clip the field was last given a default on, which is how
 * the next one is told. Called while the field is not being typed into. */
static void view_field_follow(const struct view_state *st, char *field,
                              size_t size, char *def, size_t def_size,
                              int *clip, const char *want)
{
    char out[VIEW_MAX_PATH];
    const char *fext = "", *dext, *wext;
    size_t fstem = 0, dstem, wstem;
    bool stem_typed = false, ext_typed = false;

    if (field[0]) {
        fext = view_field_ext(field, &fstem);
        dext = view_field_ext(def, &dstem);
        stem_typed = fstem != dstem || strncmp(field, def, fstem) != 0;
        ext_typed = fext[0] && strcmp(fext, dext) != 0;
    }
    wext = view_field_ext(want, &wstem);

    if (stem_typed && *clip == st->clip)
        /* The user's name on the user's clip: left alone. */
        snprintf(out, sizeof(out), "%s", field);
    else
        snprintf(out, sizeof(out), "%.*s%s", (int)wstem, want,
                 ext_typed ? fext : wext);

    snprintf(field, size, "%s", out);
    snprintf(def, def_size, "%s", want);
    *clip = st->clip;
}

/* The Export video section: the codec preset and the colour
 * space cascades, the pixel format, the size, whatever knobs the chosen
 * family declares, the sound, the threads, a frame range, the file, one
 * button, a progress bar and a way to stop. The selection itself lives in
 * the look, so it survives the export and carries from clip to clip; the
 * file field follows its default until it is typed into, and then it is
 * the user's and carries too. */
static void view_panel_export_video(struct nk_context *ctx, struct view_state *st)
{
    struct view_video *v = &st->video;
    struct view_preset_sel *sel = &st->look.video_preset;
    const struct view_preset_family *fam;
    char want[VIEW_MAX_PATH], label[128];
    const char *codec, *why;
    bool busy = v->active;
    bool scene, vulkan;
    /* The last frame there is, which is the most either field can hold:
     * a range cannot reach past the clip. A clip whose length FFmpeg
     * cannot tell has no such bound, and the engine finds the end by
     * running into it. */
    int last = st->dec.nb_frames > 0
             ? (int)CRAW_MIN(st->dec.nb_frames - 1, INT_MAX - 1) : INT_MAX - 1;

    /* The whole clip until a range has been set: Ctrl+1 and Ctrl+2 mark
     * the start and the end at the frame on screen, and the two fields
     * below take numbers. A range set on a longer clip is cut to this one
     * without a word: the fields' own bounds are what say how far it can
     * go. */
    if (!busy) {
        if (!v->range_set) {
            v->start_ui = 0;
            v->end_ui = last;
        } else {
            v->start_ui = CRAW_CLAMP(v->start_ui, 0, last);
            v->end_ui = CRAW_CLAMP(v->end_ui, v->start_ui, last);
        }
    }

    if (sel->family < 0 || sel->family >= view_nb_preset_families)
        view_presets_default(sel);

    fam = &view_preset_families[sel->family];
    vulkan = view_presets_family_is_vulkan(sel->family);

    /* --codec still wins, for an encoder that is not one of the ten. */
    codec = st->cfg.export_codec;
    if (!codec || !codec[0])
        codec = fam->encoder;

    /* The file field follows its default - the clip's name, the range and
     * the family's container, in ~/Processed - the way view_field_follow()
     * says: a typed extension stays, a typed name stays on its clip. Not
     * while a run is on: the name the run is writing is the one to show.
     * And not while the field is being typed into: a field cleared to type
     * a fresh name into would otherwise be refilled under the cursor. */
    if (!busy && !v->field_active) {
        view_encode_default_path(st, v->start_ui, v->end_ui, sel, codec,
                                 want, sizeof(want));
        view_field_follow(st, v->field, sizeof(v->field), v->field_def,
                          sizeof(v->field_def), &v->field_clip, want);
    }

    nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
    view_menu_row(ctx, st, VIEW_MENU_PRESET, "video",
                  view_presets_label(sel, label, sizeof(label)));

    nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
    view_menu_row(ctx, st, VIEW_MENU_COLOUR, "colour",
                  view_colours[CRAW_CLAMP(sel->colour, 0,
                                          VIEW_NB_COLOURS - 1)].name);

    scene = view_colours[CRAW_CLAMP(sel->colour, 0, VIEW_NB_COLOURS - 1)].scene;

    /* The pixel format the picture is written in. The list is what the
     * encoder was found to accept - nothing else is offered - and the entry
     * in use is the preset's own pin until the row is touched, after which
     * the choice outlives a preset change and dies with a family one. */
    {
        /* A scene-referred export is written in the family's float format
         * whatever the row held before: say that one. */
        enum AVPixelFormat f = scene && fam->scene_fmt != AV_PIX_FMT_NONE
                             ? fam->scene_fmt : view_presets_sel_fmt(sel);

        nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
        /* The format is known from the table the moment the preset is,
         * while the *list* has to be asked of the encoder one format at a
         * time; so the row says what the file will be written as straight
         * away and the menu under it fills in behind. A preset that pins
         * nothing leaves the choice to the engine until the row is used. */
        if (f != AV_PIX_FMT_NONE)
            view_presets_fmt_label(f, label, sizeof(label));
        else
            snprintf(label, sizeof(label), "chosen by the engine");
        view_menu_row(ctx, st, VIEW_MENU_PIXFMT, "pixel format", label);
    }

    /* Why the selection cannot be used, said once and in full rather than
     * left for the encoder to discover. */
    why = view_presets_preset_why_not(sel->family, sel->preset);
    if (!why)
        why = view_presets_colour_why_not(sel->family, sel->preset,
                                          sel->colour);
    /* A format the row still holds from a preset whose profile allowed it,
     * under one whose profile does not: the menu greys such an entry, but a
     * preset change can walk the selection into one all the same, and the
     * profile is the thing that has to win. */
    if (!why && !scene)
        why = view_presets_fmt_why_not(sel->family, sel->preset,
                                       view_presets_sel_fmt(sel));
    /* The size row and a scene-referred space cannot both have their way:
     * the direct path has nothing to resample with, and inventing a scaler
     * for linear light is a decision nobody asked this program to make. */
    if (!why && scene && sel->size != VIEW_SIZE_FULL)
        why = "a scene-referred export is written straight from the "
              "pipeline, at the source's own size; set the size row to full";
    if (!why && scene && !st->dec.raw)
        why = "a scene-referred export comes out of the raw pipeline, and "
              "this clip is already developed";
    /* A hardware engine's largest picture, against what the size row
     * makes of the crop: H.264 on RADV stops at 4096, and a 5.8K clip at
     * full size is past it. */
    if (!why && !scene) {
        int x, y, cw, ch, w, h;

        view_crop_disp_rect(st, &x, &y, &cw, &ch);
        view_presets_size(sel, cw, ch, &w, &h);
        why = view_presets_extent_why_not(avcodec_find_encoder_by_name(codec),
                                          w, h);
    }
    /* What the file's name decides - the container, whether it carries the
     * encoder and the sound - is judged when the button is pressed, by
     * view_encode_begin(), and never while the name is being typed: every
     * name on the way from `.mkv` to `.mp4` is a name that picks no muxer,
     * and a row that appeared above the field for as long as that lasted
     * pushed the field down the panel, out of the window and out of
     * nuklear's hands, which drops a text field it cannot see. A row that
     * comes and goes under the cursor is not a refusal, it is a fight. */
    /* Three rows of height: a wrapped label clips at the bottom of the
     * rectangle it is given, and a refusal that ends mid-sentence is not a
     * refusal anybody can act on. */
    if (why) {
        nk_layout_row_dynamic(ctx, VIEW_ROW * 3.0f, 1);
        nk_labelf_wrap(ctx, "cannot export: %s", why);
    } else if (scene) {
        nk_layout_row_dynamic(ctx, VIEW_ROW * 3.0f, 1);
        nk_label_wrap(ctx, "scene referred: written straight from the "
                           "pipeline as float, with no tone curve, no gamut "
                           "mapping, no film simulation and no grain");
    }

    /* The size, which every export has and the direct one refuses. The
     * rectangle is the rotated one, because the file is written upright:
     * what the row halves and what the readout says are the picture as it
     * comes out, not the readout behind it. */
    {
        int x, y, cw, ch;

        view_crop_disp_rect(st, &x, &y, &cw, &ch);

        nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
        nk_label(ctx, "size", NK_TEXT_LEFT);
        sel->size = nk_combo(ctx, view_size_names, VIEW_NB_SIZES, sel->size,
                             VIEW_COMBO_ROW, nk_vec2(160, 220));

        if (sel->size == VIEW_SIZE_TYPED) {
            nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
            nk_property_int(ctx, "#width", 2, &sel->size_w, 16384, 2, 1.0f);
            nk_property_int(ctx, "#height", 2, &sel->size_h, 16384, 2, 1.0f);
        }

        nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
        nk_labelf(ctx, NK_TEXT_LEFT, "writes %s",
                  view_presets_size_label(sel, cw, ch, label, sizeof(label)));
    }

    /* The target peak of a PQ export, which is part of what the target is.
     * How the picture is mapped into it is not a row here any more: a
     * display-referred export is mapped exactly as the window is, with
     * libplacebo's own defaults, so what the file holds is what is on
     * screen. */
    if (view_colours[sel->colour].hdr &&
        sel->colour == VIEW_COLOUR_2020_PQ) {
        nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
        nk_property_int(ctx, "#peak nits", 100, &sel->hdr_peak, 10000,
                        50, 5.0f);
    }

    /* Only the rows the family declares. */
    if (fam->ctrl & VIEW_CTRL_ASYNC) {
        nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
        nk_property_int(ctx, "#async depth", fam->async_min, &sel->async,
                        fam->async_max, 1, 0.05f);
    }
    if (fam->ctrl & VIEW_CTRL_QP) {
        nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
        nk_property_int(ctx, "#qp", fam->qp_min, &sel->qp, fam->qp_max,
                        1, 0.2f);
    }
    if (fam->ctrl & VIEW_CTRL_CPU) {
        nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
        nk_property_int(ctx, "#cpu preset", fam->cpu_min, &sel->cpu,
                        fam->cpu_max, 1, 0.05f);
    }
    if (fam->ctrl & VIEW_CTRL_RC && fam->nb_rc > 0) {
        const struct view_rc_spec *rc;

        nk_layout_row_dynamic(ctx, VIEW_ROW, fam->nb_rc);
        for (int i = 0; i < fam->nb_rc; i++)
            if (nk_option_label(ctx, fam->rc[i].label, sel->rc == i))
                sel->rc = i;

        rc = &fam->rc[CRAW_CLAMP(sel->rc, 0, fam->nb_rc - 1)];
        nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
        if (rc->bitrate) {
            int mbit = CRAW_MAX(sel->kbit / 1000, 1);

            /* Mbit/s on the panel, kbit/s underneath: a bitrate field that
             * counted in kilobits would need five digits to say anything
             * useful about a 5.8K export. */
            nk_property_int(ctx, "#Mbit/s", 1, &mbit, 2000, 1, 0.5f);
            sel->kbit = mbit * 1000;
        } else {
            char name[32];

            snprintf(name, sizeof(name), "#%s", rc->label);
            nk_property_int(ctx, name, rc->qmin, &sel->quant, rc->qmax,
                            1, 0.2f);
        }
    }

    /* The threads, which only a software encoder has: greyed under a
     * Vulkan family, with the note saying why. */
    {
        int max = view_presets_max_threads();

        if (vulkan)
            view_disable_begin(ctx);
        nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
        nk_property_int(ctx, "#threads", 0, &sel->threads, max, 1, 0.2f);
        nk_label(ctx, vulkan ? "none: a Vulkan encoder"
                             : sel->threads ? "" : "0: FFmpeg's auto", NK_TEXT_LEFT);
        if (vulkan)
            view_disable_end(st, ctx);
    }

    view_panel_export_audio(ctx, st, v->field);

    nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
    if (nk_property_int(ctx, "#start", 0, &v->start_ui, last, 1, 1.0f))
        v->range_set = true;
    if (nk_property_int(ctx, "#end", v->start_ui, &v->end_ui, last, 1, 1.0f))
        v->range_set = true;

    nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
    nk_label(ctx, "file (the extension picks the container)", NK_TEXT_LEFT);
    v->field_active = !!(nk_edit_string_zero_terminated(ctx, NK_EDIT_FIELD,
                                                        v->field,
                                                        sizeof(v->field),
                                                        nk_filter_default)
                         & NK_EDIT_ACTIVE);
    if (v->field_active)
        st->ui_edit_active = true;

    /* The one stretch of a locked panel that stays live: the stop button,
     * the progress bar and the status under them. Everything else is
     * disabled from the top of the panel down while the run is on. */
    if (busy)
        nk_widget_disable_end(ctx);

    nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
    if (nk_button_label(ctx, busy ? "stop" : "export video")) {
        if (busy) {
            view_encode_cancel(st);
        } else if (!why) {
            struct view_encode_opts o = {
                .start = v->start_ui,
                .end   = v->end_ui,
                .path  = v->field,
                .sel   = sel,
                .codec = codec,
                .enc   = st->cfg.export_enc,
            };

            /* A name without an extension gets the family's container,
             * in the field itself, so that what the panel shows is what
             * is written. Everything else about the name - a container
             * that will not carry the encoder or the sound - is refused by
             * the engine, on the status line under this button. */
            view_encode_complete_path(sel, codec, v->field, sizeof(v->field));
            v->range_set = true;
            view_encode_begin(st, &o);
        }
    }
    {
        nk_size cur = (nk_size)(v->progress * 1000.0);

        nk_progress(ctx, &cur, 1000, nk_false);
    }

    if (v->status[0])
        view_status_rows(ctx, v->status);
    if (v->path[0]) {
        const char *name = strrchr(v->path, '/');

        nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
        nk_labelf(ctx, NK_TEXT_LEFT, "%.46s", name ? name + 1 : v->path);
    }

    if (busy)
        nk_widget_disable_begin(ctx);
}

/* The Export image section: the frame on screen, one image, the format
 * picked by the file field's extension. The field follows the frame on
 * screen, a .jpg under the clip's name, the way view_field_follow() says:
 * a typed extension stays, a typed name stays on its clip. */
static void view_panel_export_still(struct nk_context *ctx, struct view_state *st)
{
    struct view_export *e = &st->exp;
    char want[VIEW_MAX_PATH];
    int x, y, w, h;

    /* Not while the field is being typed into: a field cleared to type a
     * fresh name into would otherwise be refilled under the cursor. */
    if (!e->field_active) {
        view_export_default_path(st, st->index, want, sizeof(want));
        view_field_follow(st, e->field, sizeof(e->field), e->field_def,
                          sizeof(e->field_def), &e->field_clip, want);
    }

    nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
    nk_label(ctx, "file (the extension picks the format)", NK_TEXT_LEFT);
    e->field_active = !!(nk_edit_string_zero_terminated(ctx, NK_EDIT_FIELD,
                                                        e->field,
                                                        sizeof(e->field),
                                                        nk_filter_default)
                         & NK_EDIT_ACTIVE);
    if (e->field_active)
        st->ui_edit_active = true;

    nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
    if (nk_button_label(ctx, "export image") && !e->active) {
        /* The current frame, and nothing but it: a range of stills is a
         * video encode, which this is not. The encoder is the one the
         * name implies; there is no codec box, the extension is the
         * choice. */
        st->playing = false;
        /* A name without an extension gets the .jpg the default has, in
         * the field itself, so that what the panel shows is what is
         * written. */
        view_export_complete_path(e->field, sizeof(e->field));
        view_export_begin(st, st->index, e->field, NULL);
    }
    /* What the file will be, which is the rotated size: the still is
     * written upright like everything else. */
    if (view_crop_disp_rect(st, &x, &y, &w, &h))
        nk_labelf(ctx, NK_TEXT_LEFT, "%dx%d cropped", w, h);
    else
        nk_labelf(ctx, NK_TEXT_LEFT, "%dx%d full", w, h);

    if (e->status[0])
        view_status_rows(ctx, e->status);
}

/* A block's title with a rule over it, so that the block reads as a thing
 * of its own under the rows above it. */
static void view_panel_block(struct nk_context *ctx, const char *title)
{
    nk_layout_row_dynamic(ctx, 6.0f, 1);
    nk_rule_horizontal(ctx, nk_rgba(80, 80, 92, 255), nk_false);
    nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
    nk_label(ctx, title, NK_TEXT_LEFT);
}

/* The "Crop" section: what a left drag has drawn. The mode the drag draws
 * in is the header's switch, which `c` is the other hand on, and switching
 * it on is what clears a rectangle - so there is no clear button, only the
 * readout. The rectangle is held in source pixels and reported on the
 * picture as it is shown, so the readout says which region the export will
 * write whatever the window is doing with it. On a rotated clip those are
 * two different rectangles and the one worth printing is the one that was
 * drawn - `--crop` takes the source's, which is that one turned back. */
static void view_panel_crop(struct nk_context *ctx, struct view_state *st)
{
    int x, y, w, h;

    nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
    nk_label(ctx, st->crop.on ? "drag to draw one; c or the switch keeps it"
                              : "crop mode is off (c); on again clears",
             NK_TEXT_LEFT);

    /* "(clip)" while the rectangle is the one the clip's own frame header
     * marked rather than one that was drawn or asked for: it says why a
     * picture that was never cropped by hand exports smaller than its
     * readout, and that the rectangle is the picture on screen rather than
     * a shade over it. */
    if (view_crop_disp_rect(st, &x, &y, &w, &h))
        nk_labelf(ctx, NK_TEXT_LEFT, "crop: %dx%d at %d,%d%s", w, h, x, y,
                  st->crop.from_clip ? " (clip)" : "");
    else
        nk_label(ctx, "crop: none, drag one over the picture", NK_TEXT_LEFT);
}

/* The "Clip" block: what the file says about the recording, as one label
 * and one value per row, and only the rows the file has something for. It
 * is information and nothing else - nothing here is clicked - so the
 * labels are dimmed and the values carry the row. A value too long for its
 * column is cut with an ellipsis rather than let run under the scrollbar:
 * the panel's width is fixed and a camera's model name is not.
 *
 * It sits under everything live - the readouts, the scrub bar, the mute
 * and the volume - and above the first section, rather than straight
 * under the clip line, because its height is the one thing in the panel's
 * top that changes with the clip: two rows for a phone's MP4, eleven for
 * a camera's HEVC take. Put above the scrub bar it would move the bar under
 * the pointer at every clip switch; here the live rows keep their places
 * and the block reads as the heading of the sections that follow. */
static void view_panel_clip(struct nk_context *ctx, struct view_state *st)
{
    const struct nk_user_font *font = ctx->style.font;
    struct view_info_row rows[VIEW_INFO_MAX];
    int n = view_decoder_info(&st->dec, rows, VIEW_INFO_MAX);
    float label_w = 0.0f;

    if (!n)
        return;

    view_panel_block(ctx, "Clip");

    /* The label column is as wide as its widest label, measured in the
     * face that is loaded, plus a gap; the value gets the rest. */
    for (int i = 0; i < n; i++) {
        float w = font->width(font->userdata, font->height, rows[i].label,
                              (int)strlen(rows[i].label));

        if (strcmp(rows[i].label, "camera"))
            label_w = CRAW_MAX(label_w, w);
    }
    label_w += 12.0f;

    for (int i = 0; i < n; i++) {
        struct nk_rect b;
        char value[80];

        /* The camera is in the log's clip line and the lens row says
         * which system it is; on the panel it was a row saying the same
         * thing every time (Lynne). */
        if (!strcmp(rows[i].label, "camera"))
            continue;

        nk_layout_row_template_begin(ctx, VIEW_ROW);
        nk_layout_row_template_push_static(ctx, label_w);
        nk_layout_row_template_push_dynamic(ctx);
        nk_layout_row_template_end(ctx);

        nk_label_colored(ctx, rows[i].label, NK_TEXT_LEFT,
                         nk_rgba(158, 158, 170, 255));

        b = nk_widget_bounds(ctx);
        view_fit_text(font, b.w - 2.0f, rows[i].value, value, sizeof(value));
        nk_label(ctx, value, NK_TEXT_LEFT);
    }
}

/* The bottom of the panel, while files are marked for deletion
 * (Shift+Alt+Delete, struct view_bin), down at its bottom edge while the
 * sections above leave room and after the last of them when they do not:
 * the button that deletes them, and
 * beside it whether the file on screen is one of them - with its JPEG
 * when a still has one. Under them, what went wrong with the last press,
 * if anything did. Nothing at all while nothing is marked. */
static void view_panel_bin(struct nk_context *ctx, struct view_state *st)
{
    const struct view_bin *bin = &st->bin;
    const struct view_bin_entry *b;
    char label[48];

    if (!bin->entries || (!bin->nb_marked && !bin->status[0]))
        return;
    b = &bin->entries[st->clip];

    /* Pushed down by a row as tall as the room left under the sections:
     * where the next row would start is measured on a row of one pixel,
     * and every row costs the window's spacing after it. Short of the
     * bottom by a few pixels, so that the panel never comes out a pixel
     * taller than its window and scrolls for nothing. The rows are filled
     * with nk_spacer(), not nk_spacing(): spacing that fills a row starts
     * the next one at the same height, which doubled the push. */
    {
        float sp = ctx->style.window.spacing.y;
        float need = 6.0f + sp, top, gap, status_h;
        struct nk_rect region = nk_window_get_content_region(ctx);

        if (bin->nb_marked)
            need += VIEW_ROW + sp;
        if (bin->status[0]) {
            view_status_lines(ctx, bin->status, &status_h);
            need += status_h + sp;
        }

        nk_layout_row_dynamic(ctx, 1.0f, 1);
        top = nk_widget_bounds(ctx).y;
        nk_spacer(ctx);

        gap = region.y + region.h - 4.0f - (top + 1.0f + 2.0f * sp) - need;
        if (gap >= 1.0f) {
            nk_layout_row_dynamic(ctx, gap, 1);
            nk_spacer(ctx);
        }
    }

    nk_layout_row_dynamic(ctx, 6.0f, 1);
    nk_rule_horizontal(ctx, nk_rgba(80, 80, 92, 255), nk_false);

    if (bin->nb_marked) {
        snprintf(label, sizeof(label), "delete %d marked", bin->nb_marked);
        nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
        if (nk_button_label(ctx, label))
            st->bin.go = true;
        if (b->marked)
            nk_label_colored(ctx, b->jpg ? "this one, and its JPEG" : "this one",
                             NK_TEXT_LEFT, nk_rgba(235, 110, 100, 255));
        else
            nk_label(ctx, "", NK_TEXT_LEFT);
    }

    if (bin->status[0])
        view_status_rows(ctx, bin->status);
}

static void view_panel(struct nk_context *ctx, struct view_state *st)
{
    struct view_look *lk = &st->look;
    const char *file = strrchr(st->cfg.input, '/');
    bool locked = view_ui_locked(st);
    int sel;

    file = file ? file + 1 : st->cfg.input;

    /* A running video export locks the panel from here to the end: every
     * row is drawn and none takes input, by mouse, wheel or key. The engine
     * owns the decoder - it seeks it and steps it - and has copied the
     * settings, so a slider that moved would be changing a picture nobody
     * is looking at and nothing about the file. The Video block's stop
     * button, progress bar and status are the one stretch left live, which
     * view_panel_export_video() re-enables for exactly those three. The
     * pair does not nest, which is what the lock-aware view_disable_end()
     * is for. */
    if (locked)
        view_disable_begin(ctx);

    nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
    /* Which of how many, whenever there is more than one: a playlist is
     * the only thing that makes the question worth the room. The rotation
     * joins the codec inside the brackets when the clip carries one, since
     * it is the other thing about the clip that decides what is on screen
     * and it is otherwise invisible - a portrait picture looks like a
     * portrait clip. */
    {
        char rot[12] = "";

        if (st->dec.rotation)
            snprintf(rot, sizeof(rot), ", %d°", st->dec.rotation_deg);

        if (st->cfg.nb_inputs > 1)
            nk_labelf(ctx, NK_TEXT_LEFT, "%d/%d %s [%s %s%s]",
                      st->clip + 1, st->cfg.nb_inputs, file,
                      st->dec.codec_name ? st->dec.codec_name : "?",
                      st->dec.raw ? "raw"
                                  : st->dec.hwaccel ? "video, gpu" : "video, cpu",
                      rot);
        else
            nk_labelf(ctx, NK_TEXT_LEFT, "%s  [%s %s%s]", file,
                      st->dec.codec_name ? st->dec.codec_name : "?",
                      st->dec.raw ? "raw"
                                  : st->dec.hwaccel ? "video, gpu" : "video, cpu",
                      rot);
    }
    /* Whatever the playlist has to report: a file that would not open, a
     * switch refused. Under the clip line, because that is what it is
     * about, and never behind a fold. */
    if (st->clip_status[0])
        nk_labelf(ctx, NK_TEXT_LEFT, "%.46s", st->clip_status);
    nk_labelf(ctx, NK_TEXT_LEFT, "frame %" PRId64 " / %" PRId64 "  %.3f s",
              st->index, st->dec.nb_frames ? st->dec.nb_frames - 1 : 0, st->pts);
    nk_labelf(ctx, NK_TEXT_LEFT, "%.0f/%.0f fps  decode %.1f ms  craw %.2f ms",
              st->proc_fps, st->fps, st->decode_ms, st->gpu.last_gpu_ms);

    view_panel_transport(ctx, st);

    if (st->audio.available) {
        nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
        nk_checkbox_label(ctx, "mute", &st->mute_flag);
        st->mute = st->mute_flag != 0;
        nk_slider_float(ctx, 0.0f, &st->volume, 1.0f, 0.01f);

        nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
        nk_labelf(ctx, NK_TEXT_LEFT, "audio: %s", st->audio.mapping);
        nk_labelf(ctx, NK_TEXT_LEFT, "%.0f%%  buffered %3.0f ms  a/v %+.1f ms  "
                                     "drops %" PRId64 "  under %" PRIu64,
                  st->volume * 100.0f, view_audio_buffered(&st->audio) * 1000.0,
                  st->sync.diff * 1000.0, st->sync.dropped, st->audio.underruns);
    }

    view_panel_clip(ctx, st);

    /* Every section from here down is a header - a switch and an expander -
     * and a body under it.
     *
     * The white balance is offered for both kinds of clip, but only a
     * developed clip's can be switched off: a raw clip's is always applied,
     * so its header carries the empty switch slot that keeps the column
     * lined up. A developed clip has no mosaic, so Debayer, Denoise, Lens,
     * Rotate and Sharpen are not offered for one at all. */
    if (view_section_header(ctx, st, VIEW_SEC_WB, "White balance",
                            st->dec.raw ? NULL : &st->sections.wb, false)) {
        view_panel_wb(ctx, st);
        view_section_end(ctx);
    }

    if (view_section_header(ctx, st, VIEW_SEC_EXPOSURE, "Exposure",
                            &st->sections.exposure, false)) {
        view_panel_exposure(ctx, st);
        view_section_end(ctx);
    }

    if (st->dec.raw &&
        view_section_header(ctx, st, VIEW_SEC_DEBAYER, "Debayer", NULL, false)) {
        nk_layout_row_dynamic(ctx, VIEW_ROW, 3);
        for (int i = 0; i < CRAW_DEBAYER_NB; i++)
            if (nk_option_label(ctx, view_debayer_names[i], lk->debayer == i))
                lk->debayer = (enum CRAWDebayer)i;
        /* False colour suppression, DESIGN.md 17: median passes over the
         * colour differences after the debayer, which is why it is this
         * section's row and not a section of its own. Zero is off and
         * costs nothing; the pipeline always carries the stage. */
        nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
        nk_property_int(ctx, "#false colour passes", 0, &lk->false_color,
                        CRAW_FALSE_COLOR_MAX_PASSES, 1, 0.05f);
        nk_slider_int(ctx, 0, &lk->false_color, CRAW_FALSE_COLOR_MAX_PASSES, 1);
        /* Highlight reconstruction, DESIGN.md 18: how the mosaic's clipped
         * samples come off, before the debayer, so a row here too. */
        nk_checkbox_label(ctx, "rebuild clipped highlights", &lk->highlights);
        /* Defringe, DESIGN.md 19, after the debayer like the false colour
         * passes. */
        nk_property_float(ctx, "#defringe", 0.0f, &lk->defringe, 1.0f, 0.05f, 0.01f);
        nk_slider_float(ctx, 0.0f, &lk->defringe, 1.0f, 0.05f);
        view_section_end(ctx);
    }

    if (st->dec.raw &&
        view_section_header(ctx, st, VIEW_SEC_DENOISE, "Denoise",
                            &st->denoise.on, false)) {
        view_panel_denoise(ctx, st);
        view_section_end(ctx);
    }

    /* The lens switch is the one that can be refused: a clip whose frames
     * carry no correction record cannot be corrected, so the switch greys
     * out, showing off, rather than lying about what it would do. */
    if (st->dec.raw &&
        view_section_header(ctx, st, VIEW_SEC_LENS, "Lens", &st->lens.on,
                            !st->lens.record.valid)) {
        view_panel_lens(ctx, st);
        view_section_end(ctx);
    }

    if (st->dec.raw &&
        view_section_header(ctx, st, VIEW_SEC_ROTATE, "Rotate", &st->rotate.on, false)) {
        view_panel_rotate(ctx, st);
        view_section_end(ctx);
    }

    if (st->dec.raw &&
        view_section_header(ctx, st, VIEW_SEC_SHARPEN, "Sharpen", &st->sharpen.on, false)) {
        view_panel_sharpen(ctx, st);
        view_section_end(ctx);
    }

    if (view_section_header(ctx, st, VIEW_SEC_LOOK, "Look", &st->sections.look,
                            view_export_is_scene(st))) {
        /* Tone mapping, peak detection and gamut mapping used to be here.
         * They are not a look: the window is mapped with libplacebo's own
         * defaults for whatever the compositor negotiated, and choosing is
         * the export's business. The Export section has them. */
        bool grey;

        /* A readout and not a control: what libplacebo measured for the
         * window against what the pipeline can have produced. */
        nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
        nk_label(ctx, "display peak", NK_TEXT_LEFT);
        if (st->gpu.measured_peak > 0.0)
            nk_labelf(ctx, NK_TEXT_LEFT, "%.0f / %.0f nits",
                      st->gpu.measured_peak, view_peak_ceiling(st));
        else
            nk_labelf(ctx, NK_TEXT_LEFT, "ceiling %.0f nits", view_peak_ceiling(st));

        /* The curve the window is tone mapped with, out of libplacebo's
         * own list. A display setting rather than a look: it is above the
         * greying and outside the section's switch, because the window is
         * mapped through it whatever else is bypassed, and every
         * display-referred export follows it (gpu.c). */
        {
            static const char *names[32];
            static int nb_names;

            if (!nb_names) {
                nb_names = CRAW_MIN(view_tone_map_count(), 32);
                for (int i = 0; i < nb_names; i++)
                    names[i] = view_tone_map_name(i);
            }

            nk_label(ctx, "tone map", NK_TEXT_LEFT);
            lk->tone_map = nk_combo(ctx, names, nb_names,
                                    CRAW_CLAMP(lk->tone_map, 0, nb_names - 1),
                                    VIEW_COMBO_ROW, nk_vec2(160, 340));
        }

        grey = view_grey_begin(st, ctx,
            "not applied: the export is scene referred, so the film "
            "simulation is off and the screen shows what the file holds");

        view_panel_film_sim(ctx, st);

        /* A user .cube comes from --lut on the command line; the panel only
         * shows it, and its two settings, while one is loaded. */
        if (st->lut && lk->film_sim < 0) {
            const char *base = strrchr(lk->lut_path, '/');
            static const char *lut_types[] = { "native", "normalized" };

            nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
            nk_labelf(ctx, NK_TEXT_LEFT, "cube: %s", base ? base + 1 : lk->lut_path);

            nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
            sel = lk->lut_type == PL_LUT_NORMALIZED ? 1 : 0;
            nk_label(ctx, "LUT type", NK_TEXT_LEFT);
            sel = nk_combo(ctx, lut_types, 2, sel, VIEW_COMBO_ROW, nk_vec2(140, 70));
            lk->lut_type = sel ? PL_LUT_NORMALIZED : PL_LUT_NATIVE;

            nk_label(ctx, "LUT input", NK_TEXT_LEFT);
            sel = nk_combo(ctx, view_lut_space_names, view_nb_lut_spaces,
                           lk->lut_space, VIEW_COMBO_ROW, nk_vec2(160, 140));
            if (sel != lk->lut_space) {
                lk->lut_space = sel;
                view_lut_apply_space(st->lut, sel);
            }
        }

        if (st->lut_error[0]) {
            nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
            nk_labelf(ctx, NK_TEXT_LEFT, "%s", st->lut_error);
        }

        view_grey_end(st, ctx, grey);
        view_section_end(ctx);
    }

    if (view_section_header(ctx, st, VIEW_SEC_ADJUST, "Adjustments",
                            &st->sections.adjust, view_export_is_scene(st))) {
        view_panel_adjust(ctx, st);
        view_section_end(ctx);
    }

    if (view_section_header(ctx, st, VIEW_SEC_GRAIN, "Film grain",
                            &st->grain.on, view_export_is_scene(st))) {
        view_panel_grain(ctx, st);
        view_section_end(ctx);
    }

    /* The crop mode is a switch like any other, and `c` is the same
     * switch. Switching it on clears whatever rectangle there was, so the
     * whole frame is back and a new one can be drawn; off keeps the one
     * that was drawn, shaded, for the export. Checked before the body is
     * laid out so that the readout under the header never shows the old
     * rectangle for a frame. */
    {
        bool was = st->crop.on;
        bool open = view_section_header(ctx, st, VIEW_SEC_CROP, "Crop",
                                        &st->crop.on, false);

        if (st->crop.on && !was)
            view_crop_clear(st);
        if (open) {
            view_panel_crop(ctx, st);
            view_section_end(ctx);
        }
    }

    /* Two exports, two sections, two files: the image of the frame on
     * screen, and the video of the range. Neither field names the other's
     * file, and each keeps what was typed into it while its section is
     * folded: the fields are state, and a body that is not laid out simply
     * is not looked at. */
    if (view_section_header(ctx, st, VIEW_SEC_EXPORT_IMAGE, "Export image",
                            NULL, false)) {
        view_panel_export_still(ctx, st);
        view_section_end(ctx);
    }

    /* A running video export holds its section open: the panel is locked
     * from the top down while the run is on, so the expander takes no
     * click, and the stop button and the progress are in this body. */
    if (st->video.active)
        st->sections.open[VIEW_SEC_EXPORT_VIDEO] = true;
    if (view_section_header(ctx, st, VIEW_SEC_EXPORT_VIDEO, "Export video",
                            NULL, false)) {
        view_panel_export_video(ctx, st);
        view_section_end(ctx);
    }

    view_panel_bin(ctx, st);

    /* The end of the lock. nuklear's disable rewrites the style in place
     * and only its end puts the colours back, so this has to be the real
     * end whenever the top of the panel began one - not the lock-aware
     * helper, which skips the end while locked so that the sections
     * inside cannot undo the lock. Skipped here, the style stayed
     * disabled after the export was over, with nothing left to restore
     * it: everything greyed out and dead until the program was restarted. */
    if (locked)
        nk_widget_disable_end(ctx);
}

/* ------------------------------------------------------- the crop shade */

/* A band from whole pixel edges. They have to be whole: nk_fill_rect takes
 * a float rectangle but stores it as a short origin and an unsigned short
 * size, each truncated on its own, so a band that starts at 693.09 and ends
 * at 846.18 is drawn from 693 with a height of 153 and stops at 845 - the
 * fraction is dropped twice and the far edge falls a pixel short. Two bands
 * that meet on such an edge leave the pixel between them unshaded, which is
 * a bright one pixel line all the way across the picture. */
static void view_crop_band(struct nk_command_buffer *out, int x0, int y0,
                           int x1, int y1, struct nk_color c)
{
    if (x1 > x0 && y1 > y0)
        nk_fill_rect(out, nk_rect((float)x0, (float)y0, (float)(x1 - x0),
                                  (float)(y1 - y0)), 0.0f, c);
}

/* Everything outside the crop, darkened, with a thin light border on the
 * rectangle itself. It is drawn on the nuklear canvas rather than anywhere
 * in the render path, so the picture is not touched and no shader changes;
 * the geometry comes from the same view transform that gives libplacebo
 * its crop rects, which is what keeps the shade on the same source pixels
 * at every zoom, pan and window size.
 *
 * The window is the picture exactly as it lands on screen, so nuklear's
 * own clipping cuts the bands and the border at its edges, and it is built
 * before the panel so that the panel and the menu draw on top of it. */
static void view_crop_draw(struct nk_context *ctx, struct view_state *st)
{
    struct nk_command_buffer *out;
    /* Black at 153 of 255, which is not the 60 % it looks like: the
     * backend blends into a cleared, fully transparent target with
     * src_alpha for the colour and for the alpha both, so what lands in
     * the overlay is 0.36 coverage, and the renderer composites that on
     * the swapchain in the swapchain's own encoding. What is left outside
     * the rectangle is 0.64 of the encoded picture - 82 where the picture
     * is 128, and 41 where it is 64, both measured: dark enough to read as
     * excluded without hiding what is in it. */
    struct nk_color shade = nk_rgba(0, 0, 0, 153);
    pl_rect2df src, dst, crop;
    int dx0, dy0, dx1, dy1, cx0, cy0, cx1, cy1, rx0, ry0, rx1, ry1;

    /* A clip's own rectangle is the picture on screen (view_disp_rect()),
     * so there is nothing outside it to shade and its border would be a
     * light line round the edge of the picture. */
    if (!st->crop.have || st->crop.from_clip || !view_crop_screen(st, &crop))
        return;

    view_zoom_rects(st, &src, &dst);
    if (pl_rect_w(dst) < 4.0f || pl_rect_h(dst) < 4.0f)
        return;

    /* The edges every band is built from, as whole pixels, worked out once
     * here rather than per band: two bands that share an edge have to be
     * handed the very same number, or the pixel on it is shaded twice or
     * not at all. The crop's own edges are also held to the picture, so
     * that a crop scrolled off an edge collapses the bands on that side
     * instead of asking for a rectangle whose size no longer fits in the
     * unsigned short nuklear keeps it in. */
    dx0 = (int)lroundf(dst.x0);
    dy0 = (int)lroundf(dst.y0);
    dx1 = (int)lroundf(dst.x1);
    dy1 = (int)lroundf(dst.y1);
    rx0 = (int)lroundf(crop.x0);
    ry0 = (int)lroundf(crop.y0);
    rx1 = (int)lroundf(crop.x1);
    ry1 = (int)lroundf(crop.y1);
    cx0 = CRAW_CLAMP(rx0, dx0, dx1);
    cy0 = CRAW_CLAMP(ry0, dy0, dy1);
    cx1 = CRAW_CLAMP(rx1, cx0, dx1);
    cy1 = CRAW_CLAMP(ry1, cy0, dy1);

    nk_style_push_vec2(ctx, &ctx->style.window.padding, nk_vec2(0.0f, 0.0f));
    nk_style_push_style_item(ctx, &ctx->style.window.fixed_background,
                             nk_style_item_color(nk_rgba(0, 0, 0, 0)));

    /* The window is the picture on the same whole pixels the bands are cut
     * from. It has to be whole for the same reason they do: nuklear turns
     * the window's rectangle into the clip it hands the backend by
     * truncating the origin and the size on their own, so a picture that
     * starts at 233.82 and is 612.36 tall is clipped to rows 233 to 844 and
     * the last row of the bottom band is cut off. */
    if (nk_begin(ctx, "cineview-crop",
                 nk_rect((float)dx0, (float)dy0,
                         (float)(dx1 - dx0), (float)(dy1 - dy0)),
                 NK_WINDOW_NO_SCROLLBAR | NK_WINDOW_NO_INPUT |
                 NK_WINDOW_BACKGROUND)) {
        out = nk_window_get_canvas(ctx);

        /* Four bands that tile the picture minus the rectangle. Any of them
         * can come out empty - the crop scrolled off an edge collapses the
         * two bands on that side - and the helper drops those. */
        view_crop_band(out, dx0, dy0, dx1, cy0, shade);
        view_crop_band(out, dx0, cy1, dx1, dy1, shade);
        view_crop_band(out, dx0, cy0, cx0, cy1, shade);
        view_crop_band(out, cx1, cy0, dx1, cy1, shade);

        /* The border on the same whole pixels, so that the line straddles
         * the edge of the shade instead of landing a pixel inside it, and
         * on the crop's own edges rather than the clamped ones: a side
         * that has scrolled off the picture has to stay off it rather than
         * come to rest on the edge of the view. */
        nk_stroke_rect(out, nk_rect((float)rx0, (float)ry0,
                                    (float)(rx1 - rx0), (float)(ry1 - ry0)),
                       0.0f, 1.0f, nk_rgba(235, 235, 240, 220));
    }
    nk_end(ctx);

    nk_style_pop_style_item(ctx);
    nk_style_pop_vec2(ctx);
}

/* A red frame round the picture while the file on screen is marked for
 * deletion (struct view_bin), so that a marked file cannot be looked at
 * without it being plain. Drawn inside the picture's own edges, on the
 * same whole pixels the crop shade uses: outside them it would fall in the
 * letterbox of a fitted picture but off the screen of a zoomed one, and
 * the frame has to be there whichever. */
static void view_bin_draw(struct nk_context *ctx, struct view_state *st)
{
    const float t = 4.0f;
    pl_rect2df src, dst;
    int x0, y0, x1, y1;

    if (!st->bin.entries || !st->bin.entries[st->clip].marked || !st->have_frame)
        return;

    view_zoom_rects(st, &src, &dst);
    x0 = (int)lroundf(dst.x0);
    y0 = (int)lroundf(dst.y0);
    x1 = (int)lroundf(dst.x1);
    y1 = (int)lroundf(dst.y1);
    if (x1 - x0 < 4 * (int)t || y1 - y0 < 4 * (int)t)
        return;

    nk_style_push_vec2(ctx, &ctx->style.window.padding, nk_vec2(0.0f, 0.0f));
    nk_style_push_style_item(ctx, &ctx->style.window.fixed_background,
                             nk_style_item_color(nk_rgba(0, 0, 0, 0)));
    if (nk_begin(ctx, "cineview-bin",
                 nk_rect((float)x0, (float)y0, (float)(x1 - x0), (float)(y1 - y0)),
                 NK_WINDOW_NO_SCROLLBAR | NK_WINDOW_NO_INPUT |
                 NK_WINDOW_BACKGROUND)) {
        struct nk_command_buffer *out = nk_window_get_canvas(ctx);
        struct nk_color red = nk_rgba(225, 40, 30, 255);

        /* Four bands rather than a stroke, for the same reason the crop
         * shade is bands: whole pixels, and edges that meet exactly. */
        nk_fill_rect(out, nk_rect((float)x0, (float)y0, (float)(x1 - x0), t), 0.0f, red);
        nk_fill_rect(out, nk_rect((float)x0, (float)y1 - t, (float)(x1 - x0), t), 0.0f, red);
        nk_fill_rect(out, nk_rect((float)x0, (float)y0 + t, t,
                                  (float)(y1 - y0) - 2.0f * t), 0.0f, red);
        nk_fill_rect(out, nk_rect((float)x1 - t, (float)y0 + t, t,
                                  (float)(y1 - y0) - 2.0f * t), 0.0f, red);
    }
    nk_end(ctx);
    nk_style_pop_style_item(ctx);
    nk_style_pop_vec2(ctx);
}

void view_ui_build(struct view_state *st)
{
    float h = (float)st->gpu.ui_h - 2 * VIEW_UI_MARGIN;

    st->ui_edit_active = false;
    st->menu.anchored = false;

    /* Before the panel, which is what puts the panel on top of them. */
    view_crop_draw(view_nk, st);
    view_bin_draw(view_nk, st);

    if (nk_begin(view_nk, "cineview",
                 nk_rect((float)st->gpu.ui_w - VIEW_UI_WIDTH - VIEW_UI_MARGIN,
                         VIEW_UI_MARGIN, VIEW_UI_WIDTH, h),
                 NK_WINDOW_BORDER | NK_WINDOW_TITLE | NK_WINDOW_SCROLL_AUTO_HIDE)) {
        const struct nk_window *win = nk_window_find(view_nk, "cineview");

        view_panel(view_nk, st);

        /* A number field being typed into takes the keyboard the way a
         * text field does. nuklear keeps a property's edit in the window
         * rather than reporting it from the widget, and a field that did
         * not take the keys would hand a typed frame number to the
         * shortcuts - `1` is the 1:1 zoom - and never see it itself, since
         * SDL3 only sends text to a window that has asked for it. */
        if (win && win->property.active)
            st->ui_edit_active = true;
    }

    nk_end(view_nk);

    /* The menu's columns are built after the panel, which is what puts
     * them on top of it: nuklear draws its windows in list order and
     * NK_WINDOW_NO_INPUT keeps that order from ever being shuffled. The
     * row scrolling out of view, or its whole section collapsing, takes
     * the menu with it rather than leaving it hanging off nothing. */
    if (st->menu.open && !st->menu.anchored)
        view_ui_menu_close(st);
    view_menu_frame(view_nk, st);

    /* The layout pass is what discovers whether an edit field took the
     * keyboard, and SDL3 sends no text events to a window that has not
     * started text input, so the answer goes straight back to SDL here. */
    nk_sdl_text_input(st->ui_edit_active);
}

/* -------------------------------------------------------------- render */

int view_ui_render(struct view_state *st)
{
    struct view_gpu *g = &st->gpu;
    VkSemaphore done;
    int64_t t0 = av_gettime_relative();
    int slot = g->ui_slot;

    /* One overlay image, command buffer and hold semaphore per frame in
     * flight, so the backend's single command buffer per index is never
     * re-recorded while it is still running and the renderer needs no
     * drain between frames. What orders libplacebo's read of the image is
     * the semaphore the backend signals when its own submission is done. */
    if (!pl_vulkan_hold_ex(g->gpu, pl_vulkan_hold_params(
            .tex       = g->ui_tex[slot],
            /* Any layout will do: the backend's render pass declares an
             * undefined initial layout and clears, so the contents are
             * discarded either way. It must still be a real layout, since
             * libplacebo records a barrier to reach it. */
            .layout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .qf        = g->qf,
            .semaphore = (pl_vulkan_sem) { view_ui_hold_sem[slot], 0 },
        ))) {
        fprintf(stderr, "cannot hold the overlay texture\n");
        return CRAW_ERROR(EBUSY);
    }

    done = nk_sdl_render(g->queue, (uint32_t)slot, view_ui_hold_sem[slot],
                         NK_ANTI_ALIASING_ON);

    pl_vulkan_release_ex(g->gpu, pl_vulkan_release_params(
        .tex       = g->ui_tex[slot],
        /* What the backend's render pass leaves behind. */
        .layout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .qf        = g->qf,
        .semaphore = (pl_vulkan_sem) { done, 0 },
    ));

    st->timing.ui += (double)(av_gettime_relative() - t0) / 1000.0;

    return 0;
}

/* Called once the frame that used the current overlay has been submitted. */
void view_ui_rotate(struct view_gpu *g)
{
    g->ui_slot = (g->ui_slot + 1) % VIEW_SLOTS;
}
