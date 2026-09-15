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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vulkan/vulkan.h>

#include <libavutil/time.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

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
#define VIEW_ROW       24.0f

/* The cascading menu's own metrics. Its rows are tighter than the panel's:
 * a film category can hold twenty-six stocks and the column still has to
 * fit in a modest window. */
#define VIEW_MENU_ROW   22.0f
#define VIEW_MENU_PAD    4.0f
#define VIEW_MENU_XPAD   8.0f
/* How long the pointer has to rest on an entry before its submenu opens,
 * and how long an already open submenu survives the pointer sweeping
 * across its siblings on the way over to it. */
#define VIEW_MENU_OPEN_US  150000
#define VIEW_MENU_GUARD_US 300000
#define VIEW_MENU_MIN_W 110.0f
#define VIEW_MENU_MAX_W 330.0f
#define VIEW_MENU_ARROW 16.0f

#define VIEW_NK_VERTEX_BUFFER (512 * 1024)
#define VIEW_NK_INDEX_BUFFER  (128 * 1024)

static struct nk_context *view_nk;
static VkSemaphore view_ui_hold_sem[VIEW_SLOTS];

/* ------------------------------------------------------------ lifetime */

int view_ui_open(struct view_state *st, void *sdl_window)
{
    struct view_gpu *g = &st->gpu;
    struct nk_font_atlas *atlas;
    int w = 0, h = 0;
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
    view_gpu_lock(g);
    nk_sdl_font_stash_end(g->queue);
    view_gpu_unlock(g);

    /* An opaque panel keeps the alpha the backend writes at 1.0 wherever it
     * matters; only the outermost antialiased border pixels are partially
     * covered, and those are where its src-alpha blend against a cleared
     * target is not exactly premultiplied. */
    view_nk->style.window.fixed_background = nk_style_item_color(nk_rgba(24, 24, 28, 255));

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

bool view_ui_event(void *sdl_event)
{
    return nk_sdl_handle_event(sdl_event) != 0;
}

void view_ui_input_begin(void)
{
    nk_input_begin(view_nk);
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

/* Marks the next row as one Up and Down can reach, and draws the focus ring
 * behind it. nuklear has no focus model of its own, so the list is simply
 * the order the rows are emitted in. */
static bool view_row(struct view_state *st, struct nk_context *ctx)
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

/* The pending Left/Right step for the focused row, consumed on read. */
static int view_step(struct view_state *st, bool focused)
{
    int adj = focused ? st->adjust : 0;

    if (adj)
        st->adjust = 0;

    return adj;
}

static void view_adjust_float(struct view_state *st, bool focused, float *v,
                              float step, float lo, float hi)
{
    int adj = view_step(st, focused);

    if (adj) {
        *v = CRAW_CLAMP(*v + (float)adj * step, lo, hi);
        /* Three sixths of a stop back from a half is not -0.00. */
        if (fabsf(*v) < step * 1e-3f)
            *v = 0.0f;
    }
}

static void view_adjust_cycle(struct view_state *st, bool focused, int *v, int n)
{
    int adj = view_step(st, focused);

    if (adj)
        *v = (*v + adj + n) % n;
}

/* ------------------------------------------- the film simulation menu */

#if CINEVIEW_HAVE_LUTS

/* nuklear has popups and groups but nothing cascading, and its windows
 * hand input to whichever one the pointer last activated, which is not how
 * a menu behaves: every column has to keep reacting while the panel
 * underneath is still the window nuklear considers active. So the columns
 * are NK_WINDOW_NO_INPUT windows — nuklear draws, borders and clips them,
 * and skips them entirely when it decides who owns the pointer — and the
 * hover, click and scroll logic below works straight off the pointer
 * position against rectangles this file computes. That also keeps the
 * columns in creation order, which is drawing order, so they always sit on
 * top of the panel built before them. */

/* The film whose renderings column 3 lists, or NULL when the path does not
 * reach that far. */
static const ViewLutFilm *view_menu_film(const struct view_state *st)
{
    const struct view_lut_menu *m = &st->menu;
    const ViewLutFilmCategory *c;

    if (m->sel[0] != 0 || m->sel[1] < 0 || m->sel[2] < 0)
        return NULL;
    c = view_lut_film_category(m->sel[1]);
    if (!c || m->sel[2] >= c->n_films)
        return NULL;

    return view_lut_film(c->first_film + m->sel[2]);
}

/* A leaf column's entries are renderings, so clicking one selects. */
static bool view_menu_leaf(const struct view_state *st, int col)
{
    return (st->menu.sel[0] == 1 && col == 2) ||
           (st->menu.sel[0] == 0 && col == 3);
}

static int view_menu_count(const struct view_state *st, int col)
{
    const struct view_lut_menu *m = &st->menu;
    const ViewLutFilmCategory *fc;
    const ViewLutLookCategory *lc;
    const ViewLutFilm *f;

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
}

/* The rendering an entry of a leaf column stands for, or -1. */
static int view_menu_lut(const struct view_state *st, int col, int i)
{
    const ViewLutLookCategory *lc;
    const ViewLutFilm *f;

    if (!view_menu_leaf(st, col) || i < 0 || i >= view_menu_count(st, col))
        return -1;

    if (col == 2) {
        lc = view_lut_look_category(st->menu.sel[1]);
        return lc && lc->luts ? (int)lc->luts[i] : -1;
    }

    f = view_menu_film(st);
    return f ? f->first_lut + i : -1;
}

static void view_menu_label(const struct view_state *st, int col, int i,
                            char *buf, size_t n)
{
    const struct view_lut_menu *m = &st->menu;
    const ViewLutFilmCategory *fc;
    const ViewLutLookCategory *lc;
    const ViewLutFilm *f;
    const ViewLut *l;

    buf[0] = '\0';

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
        /* Keyboard navigation must keep its highlight on screen. */
        if (m->kb == c && m->sel[c] >= 0) {
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

    if (hit_col >= 0) {
        bool leaf = view_menu_leaf(st, hit_col);

        if (clicked) {
            /* A click is unambiguous: act on it at once, whatever the
             * hover timers were about to do. */
            m->kb = -1;
            if (leaf) {
                pick = view_menu_lut(st, hit_col, hit_entry);
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

            if (leaf) {
                /* Nothing to protect and nothing to open, so a leaf's
                 * highlight simply follows the pointer. */
                m->kb = -1;
                view_menu_descend(st, hit_col, hit_entry);
            } else if (!guarded && now - m->hover_since >= VIEW_MENU_OPEN_US) {
                m->kb = -1;
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

        for (int i = 0; i < n; i++) {
            struct nk_rect e = view_menu_entry(m, c, i);
            struct nk_rect t;
            char buf[192];
            bool on = m->sel[c] == i;

            if (e.y + e.h <= r.y || e.y >= r.y + r.h)
                continue;           /* scrolled out of the column */

            view_menu_label(st, c, i, buf, sizeof(buf));
            /* The rendering actually in use gets a mark of its own, so it
             * stays findable while the highlight is off wandering. */
            if (view_menu_lut(st, c, i) == st->look.film_sim &&
                st->look.film_sim >= 0)
                nk_fill_rect(out, e, 2.0f, nk_rgba(40, 62, 52, 255));
            if (on)
                nk_fill_rect(out, e, 2.0f,
                             m->kb == c ? nk_rgba(64, 92, 140, 255)
                                        : nk_rgba(52, 66, 96, 255));

            t = nk_rect(e.x + VIEW_MENU_XPAD, e.y, e.w - 2 * VIEW_MENU_XPAD, e.h);
            nk_draw_text(out, t, buf, (int)strlen(buf), font,
                         nk_rgba(0, 0, 0, 0), nk_rgba(226, 226, 232, 255));

            /* The mark that says this entry opens another column. Plain
             * ASCII: nuklear's built-in font atlas only bakes Latin-1, so
             * a typographic arrow would come out as a missing glyph. */
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

void view_ui_menu_open(struct view_state *st)
{
    struct view_lut_menu *m = &st->menu;
    const ViewLut *l = view_lut_get(st->look.film_sim);

    if (m->open)
        return;

    m->open = true;
    m->cols = 1;
    m->kb = 0;
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

    /* Open on what is already selected, so Enter then the arrows walks on
     * from the current look rather than from the top of the list. */
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
            m->kb = 3;
        }
    }
}

void view_ui_menu_close(struct view_state *st)
{
    st->menu.open = false;
    st->menu.cols = 0;
    st->menu.kb = -1;
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

bool view_ui_menu_key(struct view_state *st, int key)
{
    struct view_lut_menu *m = &st->menu;
    int c, n;

    if (!m->open)
        return false;

    if (key == SDLK_ESCAPE) {
        view_ui_menu_close(st);
        return true;
    }

    /* The pointer may have been driving; a key takes it back at whatever
     * depth the columns are already open to. */
    if (m->kb < 0)
        m->kb = CRAW_MAX(m->cols - 1, 0);
    c = CRAW_CLAMP(m->kb, 0, m->cols - 1);
    n = view_menu_count(st, c);

    switch (key) {
    case SDLK_UP:
    case SDLK_DOWN:
        if (n > 0) {
            int i = m->sel[c];

            if (i < 0)
                i = key == SDLK_DOWN ? 0 : n - 1;
            else
                i = (i + (key == SDLK_DOWN ? 1 : -1) + n) % n;
            view_menu_descend(st, c, i);
            m->kb = c;
        }
        return true;

    case SDLK_RIGHT:
    case SDLK_RETURN:
    case SDLK_KP_ENTER:
        if (m->sel[c] < 0 && n > 0)
            view_menu_descend(st, c, 0);
        if (view_menu_leaf(st, c)) {
            int ix = view_menu_lut(st, c, m->sel[c]);

            /* Right on a leaf has nowhere to go; only Enter selects. */
            if (key != SDLK_RIGHT && ix >= 0) {
                view_lut_select_film(st, ix);
                view_ui_menu_close(st);
            }
            return true;
        }
        /* Step into the column this entry opens, highlighting its first
         * row so the next key has something to move from. */
        if (view_menu_count(st, c + 1) > 0) {
            if (m->sel[c + 1] < 0)
                m->sel[c + 1] = 0;
            m->kb = c + 1;
        }
        return true;

    case SDLK_LEFT:
        if (c > 0) {
            m->kb = c - 1;
        } else {
            view_ui_menu_close(st);
        }
        return true;

    default:
        return false;
    }
}

/* Left and Right on the focused row step through the renderings of the
 * film category the current selection belongs to. Films are contiguous
 * inside a category and renderings inside a film, so that whole category
 * is one run of indices and a step is one index inside it. */
static void view_menu_step(struct view_state *st, int adj)
{
    int n = view_luts_count();
    int cur = st->look.film_sim;
    int lo = 0, hi = n - 1;
    const ViewLut *l;
    const ViewLutFilm *f;
    const ViewLutFilmCategory *c;

    if (!adj || n <= 0)
        return;

    if (cur < 0 || cur >= n) {
        view_lut_select_film(st, adj > 0 ? 0 : n - 1);
        return;
    }

    l = view_lut_get(cur);
    f = l ? view_lut_film(l->film) : NULL;
    c = f ? view_lut_film_category(f->category) : NULL;
    if (c && c->n_films > 0) {
        const ViewLutFilm *a = view_lut_film(c->first_film);
        const ViewLutFilm *z = view_lut_film(c->first_film + c->n_films - 1);

        if (a && z) {
            lo = a->first_lut;
            hi = z->first_lut + z->n_luts - 1;
        }
    }

    view_lut_select_film(st, CRAW_CLAMP(cur + adj, lo, hi));
}

/* The panel row. Clicking it toggles the menu; the menu itself is built
 * after the panel so that it draws on top. */
static void view_panel_film_sim(struct nk_context *ctx, struct view_state *st)
{
    struct view_lut_menu *m = &st->menu;
    struct nk_rect content = nk_window_get_content_region(ctx);
    struct nk_rect b;
    bool focused;

    nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
    focused = view_row(st, ctx);
    m->row = st->ui_row - 1;        /* view_row() has already stepped it */
    view_menu_step(st, view_step(st, focused));

    nk_label(ctx, "film simulation", NK_TEXT_LEFT);

    b = nk_widget_bounds(ctx);
    /* The menu hangs off the whole row rather than the button, so that it
     * opens flush with the panel's edge instead of halfway across it. */
    m->ax = content.x;
    m->aw = content.w;
    m->ay = b.y;
    m->ah = b.h;
    m->anchored = true;

    /* Click to toggle, never hover: a menu that dropped open whenever the
     * pointer crossed the row on its way somewhere else was the single
     * most irritating thing about it. */
    if (nk_button_label(ctx, view_lut_film_name(st))) {
        if (m->open)
            view_ui_menu_close(st);
        else
            view_ui_menu_open(st);
        m->kb = -1;
    }
}

/* Runs the whole menu for this frame: pointer first so that the cascade
 * reacts in the same frame the pointer moved, then the layout it implies,
 * then the drawing. */
static void view_menu_frame(struct nk_context *ctx, struct view_state *st)
{
    struct view_lut_menu *m = &st->menu;
    bool over = false;
    int pick;

    if (!m->open || !m->anchored)
        return;

    pick = view_menu_pointer(ctx, st, &over);
    if (pick >= 0) {
        view_lut_select_film(st, pick);
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
    view_menu_draw(ctx, st);
}

#else /* !CINEVIEW_HAVE_LUTS */

void view_ui_menu_open(struct view_state *st) { (void)st; }
void view_ui_menu_close(struct view_state *st) { (void)st; }
bool view_ui_menu_key(struct view_state *st, int key) { (void)st; (void)key; return false; }

bool view_ui_menu_covers(const struct view_state *st, float x, float y)
{
    (void)st; (void)x; (void)y;
    return false;
}

static void view_panel_film_sim(struct nk_context *ctx, struct view_state *st)
{
    (void)st;
    nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
    nk_label(ctx, "film simulation", NK_TEXT_LEFT);
    nk_label(ctx, "not built in", NK_TEXT_LEFT);
}

static void view_menu_frame(struct nk_context *ctx, struct view_state *st)
{
    (void)ctx; (void)st;
}

#endif /* CINEVIEW_HAVE_LUTS */

static void view_panel_transport(struct nk_context *ctx, struct view_state *st)
{
    int64_t total = st->dec.nb_frames;
    int pos;

    /* No transport buttons: the keyboard is the transport. Space plays and
     * pauses, Shift with Page Up and Page Down are the ends of the clip and
     * Shift with Left and Right steps a second. The scrub bar stays,
     * because dragging to a frame is the one thing keys are bad at. */
    if (total > 1) {
        bool f;
        int adj;

        pos = (int)CRAW_CLAMP(st->index, 0, total - 1);
        nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
        f = view_row(st, ctx);
        adj = view_step(st, f);
        if (adj) {
            pos = (int)CRAW_CLAMP((int64_t)pos + adj, 0, total - 1);
            st->playing = false;
            st->seek_to = pos;
            st->need_frame = true;
        } else if (nk_slider_int(ctx, 0, &pos, (int)(total - 1), 1) &&
                   pos != (int)st->index) {
            st->seek_to = pos;
            st->need_frame = true;
        }
    }
}

static void view_panel_wb(struct nk_context *ctx, struct view_state *st)
{
    struct view_look *lk = &st->look;

    int mode = (int)lk->wb;

    nk_layout_row_dynamic(ctx, VIEW_ROW, 3);
    view_adjust_cycle(st, view_row(st, ctx), &mode, 3);
    lk->wb = (enum view_wb)mode;
    if (nk_option_label(ctx, "As shot", lk->wb == VIEW_WB_AS_SHOT))
        lk->wb = VIEW_WB_AS_SHOT;
    if (nk_option_label(ctx, "Auto", lk->wb == VIEW_WB_AUTO))
        lk->wb = VIEW_WB_AUTO;
    if (nk_option_label(ctx, "Manual", lk->wb == VIEW_WB_MANUAL))
        lk->wb = VIEW_WB_MANUAL;

    nk_layout_row_dynamic(ctx, VIEW_ROW, 1);

    switch (lk->wb) {
    case VIEW_WB_AS_SHOT:
        nk_labelf(ctx, NK_TEXT_LEFT, "camera: %.0f K, gains %.4f / 1 / %.4f",
                  st->prof.as_shot_cct, st->prof.as_shot_wb[0], st->prof.as_shot_wb[2]);
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

        view_adjust_float(st, view_row(st, ctx), &st->awb.params.ambience,
                          0.05f, 0.0f, 1.0f);
        nk_property_float(ctx, "#ambience", 0.0f, &st->awb.params.ambience, 1.0f,
                          0.05f, 0.005f);
        break;
    }

    case VIEW_WB_MANUAL:
        view_adjust_float(st, view_row(st, ctx), &lk->manual.cct,
                          100.0f, 2000.0f, 11500.0f);
        nk_property_float(ctx, "#CCT (K)", 2000.0f, &lk->manual.cct, 11500.0f,
                          50.0f, 5.0f);
        nk_slider_float(ctx, 2000.0f, &lk->manual.cct, 11500.0f, 10.0f);
        view_adjust_float(st, view_row(st, ctx), &lk->manual.tint,
                          0.01f, -0.2f, 0.2f);
        nk_property_float(ctx, "#tint", -0.2f, &lk->manual.tint, 0.2f, 0.005f, 0.0005f);
        nk_slider_float(ctx, -0.2f, &lk->manual.tint, 0.2f, 0.002f);
        break;
    }
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
        int pick;

        nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
        /* The standard list is already a third of a stop per entry, so a
         * step of one entry is a step of a third of a stop. */
        pick = sel;
        view_adjust_cycle(st, view_row(st, ctx), &pick, nb_items);
        nk_label(ctx, "exposure index", NK_TEXT_LEFT);
        {
            int chosen = nk_combo(ctx, items, nb_items, pick, 20, nk_vec2(140, 260));
            if (chosen != sel)
                lk->ei = view_iso_list[chosen];
        }
    }

    nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
    view_adjust_float(st, view_row(st, ctx), &lk->exposure,
                      1.0f / 6.0f, -3.0f, 3.0f);
    nk_property_float(ctx, "#bias (stops)", -3.0f, &lk->exposure, 3.0f, 0.1f, 0.01f);
    nk_slider_float(ctx, -3.0f, &lk->exposure, 3.0f, 0.05f);
    nk_labelf(ctx, NK_TEXT_LEFT, "gain %.2f, ceiling %.0f nits",
              view_total_gain(st) * exp2f(lk->exposure), view_peak_ceiling(st));
}

/* The "Denoise" section: the raw domain stage of DESIGN.md 14, which runs
 * on the CFA planes before the debayer. The pipeline is always built with
 * the pass (gpu.c), so everything here is live: the strength reaches the
 * next frame and a change while paused puts the picture through again.
 *
 * The model is the camera's, not the panel's. All this shows is which one
 * was found for the frame's own ISO, because a clip the camera has not been
 * characterized at cannot be denoised at all and the reason has to be
 * visible rather than inferred from nothing happening. */
static void view_panel_denoise(struct nk_context *ctx, struct view_state *st)
{
    struct view_denoise *d = &st->denoise;
    int on;

    nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
    on = d->on_flag;
    view_adjust_cycle(st, view_row(st, ctx), &on, 2);
    d->on_flag = on;
    nk_checkbox_label(ctx, "denoise", &d->on_flag);
    d->on = d->on_flag != 0;
    nk_labelf(ctx, NK_TEXT_LEFT, "%s",
              !d->have_model ? "no model: off"
                             : d->on ? "on, before the debayer" : "off");

    nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
    view_adjust_float(st, view_row(st, ctx), &d->strength, 0.25f, 0.25f, 3.0f);
    nk_property_float(ctx, "#strength", 0.25f, &d->strength, 3.0f, 0.25f, 0.01f);

    /* 1.0 is "remove noise at exactly the modelled level", so the numbers
     * either side of it are worth naming. */
    nk_labelf(ctx, NK_TEXT_LEFT, "%.2f: %s", (double)d->strength,
              d->strength < 0.95f  ? "under the modelled level"
            : d->strength < 1.05f  ? "the modelled level"
            : d->strength < 2.05f  ? "smooths across real detail"
                                   : "heavy");

    if (!d->have_model)
        nk_labelf(ctx, NK_TEXT_LEFT, "no noise model for ISO %.0f: off", (double)d->iso);
    else
        nk_labelf(ctx, NK_TEXT_LEFT, "noise a %.3g, b %.2g at ISO %.0f",
                  (double)d->model.a, (double)d->model.b, (double)d->iso);

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

/* The crop row: the mode a left drag draws in, what it has drawn, and a
 * way to take it back. The rectangle is in source pixels, so the readout
 * says which region of the frame the export will write whatever the window
 * is doing with it. */
/* The "Adjustments" section: soft light and vibrance, the two RawTherapee
 * tools adjust.c reimplements. It sits between Look and Film grain because
 * that is where the hooks sit in the render: after the film simulation the
 * Look section chooses, before the grain. */
static void view_panel_adjust(struct nk_context *ctx, struct view_state *st)
{
    struct view_adjust *a = &st->adjustments;
    char buf[128];

    /* Open when something is already on, which after --softlight or
     * --vibrance it is. nuklear remembers the state per title from the
     * first frame on, so this only decides where the section starts. */
    if (!nk_tree_push(ctx, NK_TREE_TAB, "Adjustments",
                      (a->softlight > 0.0f || a->pastels != 0.0f ||
                       a->saturated != 0.0f) ? NK_MAXIMIZED : NK_MINIMIZED))
        return;

    nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
    nk_labelf(ctx, NK_TEXT_LEFT, "%s",
              view_adjust_summary(a, buf, sizeof(buf)));

    /* Soft light first, even though it runs last: it is the one control
     * most people came for, and the panel is not the pipeline. */
    view_adjust_float(st, view_row(st, ctx), &a->softlight, 5.0f, 0.0f, 100.0f);
    nk_property_float(ctx, "#soft light", 0.0f, &a->softlight, 100.0f,
                      5.0f, 0.25f);

    view_adjust_float(st, view_row(st, ctx), &a->pastels, 5.0f, -100.0f, 100.0f);
    nk_property_float(ctx, "#pastels", -100.0f, &a->pastels, 100.0f,
                      5.0f, 0.25f);

    view_adjust_float(st, view_row(st, ctx), &a->saturated, 5.0f, -100.0f, 100.0f);
    nk_property_float(ctx, "#saturated", -100.0f, &a->saturated, 100.0f,
                      5.0f, 0.25f);

    view_adjust_float(st, view_row(st, ctx), &a->threshold, 5.0f, 0.0f, 100.0f);
    nk_property_float(ctx, "#threshold", 0.0f, &a->threshold, 100.0f,
                      5.0f, 0.25f);

    /* One focus row per nuklear row, each filling both columns: view_row()
     * draws its ring across the whole content width, so two focusable
     * widgets sharing one row would have the second one's ring painted over
     * the first one's widget. */
    nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
    {
        bool f = view_row(st, ctx);

        if (view_step(st, f))
            a->protect_skin_flag = !a->protect_skin_flag;
    }
    nk_checkbox_label(ctx, "protect skin", &a->protect_skin_flag);
    a->protect_skin = a->protect_skin_flag != 0;
    nk_label(ctx, a->protect_skin ? "warm hues held back" : "every hue alike",
             NK_TEXT_LEFT);

    {
        bool f = view_row(st, ctx);

        if (view_step(st, f))
            a->avoid_shift_flag = !a->avoid_shift_flag;
    }
    nk_checkbox_label(ctx, "avoid colour shift", &a->avoid_shift_flag);
    a->avoid_shift = a->avoid_shift_flag != 0;
    nk_label(ctx, a->avoid_shift ? "chroma kept in gamut" : "chroma clipped",
             NK_TEXT_LEFT);

    nk_tree_pop(ctx);
}

static void view_panel_crop(struct nk_context *ctx, struct view_state *st)
{
    int x, y, w, h;

    nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
    {
        bool f = view_row(st, ctx);

        if (view_step(st, f))
            st->crop.on_flag = !st->crop.on_flag;
    }
    nk_checkbox_label(ctx, "crop mode (c)", &st->crop.on_flag);
    st->crop.on = st->crop.on_flag != 0;

    if (nk_button_label(ctx, "clear"))
        view_crop_clear(st);

    nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
    /* "(clip)" while the rectangle is the one the clip's own frame header
     * marked rather than one that was drawn or asked for: it says why a
     * picture that was never cropped by hand is shaded all the same. */
    if (view_crop_rect(st, &x, &y, &w, &h))
        nk_labelf(ctx, NK_TEXT_LEFT, "crop: %dx%d at %d,%d%s", w, h, x, y,
                  st->crop.from_clip ? " (clip)" : "");
    else
        nk_label(ctx, "crop: none, drag one over the picture", NK_TEXT_LEFT);
}

static void view_panel(struct nk_context *ctx, struct view_state *st)
{
    struct view_look *lk = &st->look;
    const char *file = strrchr(st->cfg.input, '/');
    int sel;

    file = file ? file + 1 : st->cfg.input;

    nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
    {
        float z = view_zoom_scale(st);

        if (st->view.fit)
            nk_labelf(ctx, NK_TEXT_LEFT, "zoom: fit (%.2fx)", z);
        else if (z == 1.0f)
            nk_label(ctx, "zoom: 1:1", NK_TEXT_LEFT);
        else
            nk_labelf(ctx, NK_TEXT_LEFT, "zoom: %.2fx", z);
    }
    nk_labelf(ctx, NK_TEXT_LEFT, "%s  [%s %s]", file,
              st->dec.codec_name ? st->dec.codec_name : "?",
              st->dec.raw ? "raw"
                          : st->dec.hwaccel ? "video, gpu" : "video, cpu");
    /* The clip's own words, when they are what the picture is being read
     * as: a V-Log clip's stream tags say BT.709 and the panel would
     * otherwise give no sign that they are being ignored. */
    if (st->dec.vlog)
        nk_labelf(ctx, NK_TEXT_LEFT, "%s / %s", st->dec.capture_gamma,
                  st->dec.capture_gamut);
    nk_labelf(ctx, NK_TEXT_LEFT, "frame %" PRId64 " / %" PRId64 "  %.3f s",
              st->index, st->dec.nb_frames ? st->dec.nb_frames - 1 : 0, st->pts);
    nk_labelf(ctx, NK_TEXT_LEFT, "%.0f/%.0f fps  decode %.1f ms  craw %.2f ms",
              st->proc_fps, st->fps, st->decode_ms, st->gpu.last_gpu_ms);

    view_panel_transport(ctx, st);

    if (st->audio.available) {
        nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
        {
            bool f = view_row(st, ctx);
            int adj = view_step(st, f);

            if (adj)
                st->mute_flag = !st->mute_flag;
        }
        nk_checkbox_label(ctx, "mute", &st->mute_flag);
        st->mute = st->mute_flag != 0;
        view_adjust_float(st, view_row(st, ctx), &st->volume,
                          0.05f, 0.0f, 1.0f);
        nk_slider_float(ctx, 0.0f, &st->volume, 1.0f, 0.01f);

        nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
        nk_labelf(ctx, NK_TEXT_LEFT, "audio: %s", st->audio.mapping);
        nk_labelf(ctx, NK_TEXT_LEFT, "%.0f%%  buffered %3.0f ms  a/v %+.1f ms  "
                                     "drops %" PRId64 "  under %" PRIu64,
                  st->volume * 100.0f, view_audio_buffered(&st->audio) * 1000.0,
                  st->sync.diff * 1000.0, st->sync.dropped, st->audio.underruns);
    }

    /* An already developed clip has no mosaic: there is nothing to balance
     * and nothing to debayer, so those sections are not offered at all. */
    if (st->dec.raw && nk_tree_push(ctx, NK_TREE_TAB, "White balance", NK_MINIMIZED)) {
        view_panel_wb(ctx, st);
        nk_tree_pop(ctx);
    }

    if (nk_tree_push(ctx, NK_TREE_TAB, "Exposure", NK_MINIMIZED)) {
        view_panel_exposure(ctx, st);
        nk_tree_pop(ctx);
    }

    if (st->dec.raw && nk_tree_push(ctx, NK_TREE_TAB, "Debayer", NK_MINIMIZED)) {
        int dbg = (int)lk->debayer;

        nk_layout_row_dynamic(ctx, VIEW_ROW, 3);
        view_adjust_cycle(st, view_row(st, ctx), &dbg, CRAW_DEBAYER_NB);
        lk->debayer = (enum CRAWDebayer)dbg;
        for (int i = 0; i < CRAW_DEBAYER_NB; i++)
            if (nk_option_label(ctx, view_debayer_names[i], lk->debayer == i))
                lk->debayer = (enum CRAWDebayer)i;
        nk_tree_pop(ctx);
    }

    if (st->dec.raw && nk_tree_push(ctx, NK_TREE_TAB, "Denoise", NK_MINIMIZED)) {
        view_panel_denoise(ctx, st);
        nk_tree_pop(ctx);
    }

    if (nk_tree_push(ctx, NK_TREE_TAB, "Look", NK_MINIMIZED)) {
        nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
        view_adjust_cycle(st, view_row(st, ctx), &lk->tone_map, view_nb_tone_maps);
        nk_label(ctx, "tone map", NK_TEXT_LEFT);
        lk->tone_map = nk_combo(ctx, view_tone_map_names, view_nb_tone_maps,
                                lk->tone_map, 20, nk_vec2(160, 160));
        view_adjust_cycle(st, view_row(st, ctx), &lk->gamut_map, view_nb_gamut_maps);
        nk_label(ctx, "gamut map", NK_TEXT_LEFT);
        lk->gamut_map = nk_combo(ctx, view_gamut_map_names, view_nb_gamut_maps,
                                 lk->gamut_map, 20, nk_vec2(160, 180));

        nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
        {
            bool f = view_row(st, ctx);

            if (view_step(st, f))
                lk->peak_detect_flag = !lk->peak_detect_flag;
        }
        nk_checkbox_label(ctx, "peak detection", &lk->peak_detect_flag);
        lk->peak_detect = lk->peak_detect_flag != 0;
        if (lk->peak_detect && st->gpu.measured_peak > 0.0)
            nk_labelf(ctx, NK_TEXT_LEFT, "%.0f / %.0f nits",
                      st->gpu.measured_peak, view_peak_ceiling(st));
        else
            nk_labelf(ctx, NK_TEXT_LEFT, "ceiling %.0f nits", view_peak_ceiling(st));

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
            view_adjust_cycle(st, view_row(st, ctx), &sel, 2);
            nk_label(ctx, "LUT type", NK_TEXT_LEFT);
            sel = nk_combo(ctx, lut_types, 2, sel, 20, nk_vec2(140, 60));
            lk->lut_type = sel ? PL_LUT_NORMALIZED : PL_LUT_NATIVE;

            view_adjust_cycle(st, view_row(st, ctx), &lk->lut_space, view_nb_lut_spaces);
            nk_label(ctx, "LUT input", NK_TEXT_LEFT);
            sel = nk_combo(ctx, view_lut_space_names, view_nb_lut_spaces,
                           lk->lut_space, 20, nk_vec2(160, 120));
            if (sel != lk->lut_space) {
                lk->lut_space = sel;
                view_lut_apply_space(st->lut, sel);
            }
        }

        if (st->lut_error[0]) {
            nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
            nk_labelf(ctx, NK_TEXT_LEFT, "%s", st->lut_error);
        }
        nk_tree_pop(ctx);
    }

    view_panel_adjust(ctx, st);

    view_panel_grain(ctx, st);

    view_panel_crop(ctx, st);

    if (nk_tree_push(ctx, NK_TREE_TAB, "Export", NK_MINIMIZED)) {
        /* The field follows the frame on screen until it is typed into:
         * once it differs from the default that was last put there, it is
         * the user's and is left alone. */
        static char path[VIEW_MAX_PATH], def[VIEW_MAX_PATH];
        static char codec[64];
        char want[VIEW_MAX_PATH];

        view_export_default_path(st, st->index, want, sizeof(want));
        if (!path[0] || !strcmp(path, def)) {
            snprintf(path, sizeof(path), "%s", want);
            snprintf(def, sizeof(def), "%s", want);
        }

        nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
        nk_label(ctx, "file (the extension picks the format)", NK_TEXT_LEFT);
        if (nk_edit_string_zero_terminated(ctx, NK_EDIT_FIELD, path, sizeof(path),
                                           nk_filter_default) & NK_EDIT_ACTIVE)
            st->ui_edit_active = true;

        nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
        nk_label(ctx, "codec (empty: by name)", NK_TEXT_LEFT);
        if (nk_edit_string_zero_terminated(ctx, NK_EDIT_FIELD, codec, sizeof(codec),
                                           nk_filter_default) & NK_EDIT_ACTIVE)
            st->ui_edit_active = true;

        nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
        if (nk_button_label(ctx, "Export") && !st->exp.active) {
            /* The current frame, and nothing but it: a range of stills is
             * a video encode, which this is not. */
            st->playing = false;
            view_export_begin(st, st->index, path, codec);
        }
        {
            int x, y, w, h;

            if (view_crop_rect(st, &x, &y, &w, &h))
                nk_labelf(ctx, NK_TEXT_LEFT, "%dx%d cropped", w, h);
            else
                nk_labelf(ctx, NK_TEXT_LEFT, "%dx%d full", w, h);
        }

        if (st->exp.status[0]) {
            nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
            nk_labelf(ctx, NK_TEXT_LEFT, "%s", st->exp.status);
        }
        nk_tree_pop(ctx);
    }
}

/* ------------------------------------------------------- the crop shade */

static void view_crop_band(struct nk_command_buffer *out, float x0, float y0,
                           float x1, float y1, struct nk_color c)
{
    if (x1 > x0 && y1 > y0)
        nk_fill_rect(out, nk_rect(x0, y0, x1 - x0, y1 - y0), 0.0f, c);
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
    /* 60 % black. The overlay is composited by the renderer in linear
     * light, so what is left outside the rectangle is 40 % of the
     * picture's light, which comes out at about 0.64 encoded: dark enough
     * to read as excluded without hiding what is in it. */
    struct nk_color shade = nk_rgba(0, 0, 0, 153);
    pl_rect2df src, dst, crop;

    if (!st->crop.have || !view_crop_screen(st, &crop))
        return;

    view_zoom_rects(st, &src, &dst);
    if (pl_rect_w(dst) < 4.0f || pl_rect_h(dst) < 4.0f)
        return;

    nk_style_push_vec2(ctx, &ctx->style.window.padding, nk_vec2(0.0f, 0.0f));
    nk_style_push_style_item(ctx, &ctx->style.window.fixed_background,
                             nk_style_item_color(nk_rgba(0, 0, 0, 0)));

    if (nk_begin(ctx, "cineview-crop",
                 nk_rect(dst.x0, dst.y0, pl_rect_w(dst), pl_rect_h(dst)),
                 NK_WINDOW_NO_SCROLLBAR | NK_WINDOW_NO_INPUT |
                 NK_WINDOW_BACKGROUND)) {
        out = nk_window_get_canvas(ctx);

        /* Four bands that tile the picture minus the rectangle. Any of them
         * can come out empty, and any can hang over an edge when the crop
         * is partly scrolled off; the helper drops the first case and the
         * window's clip takes care of the second. */
        view_crop_band(out, dst.x0, dst.y0,  dst.x1,  crop.y0, shade);
        view_crop_band(out, dst.x0, crop.y1, dst.x1,  dst.y1,  shade);
        view_crop_band(out, dst.x0, crop.y0, crop.x0, crop.y1, shade);
        view_crop_band(out, crop.x1, crop.y0, dst.x1, crop.y1, shade);

        nk_stroke_rect(out, nk_rect(crop.x0, crop.y0,
                                    pl_rect_w(crop), pl_rect_h(crop)),
                       0.0f, 1.0f, nk_rgba(235, 235, 240, 220));
    }
    nk_end(ctx);

    nk_style_pop_style_item(ctx);
    nk_style_pop_vec2(ctx);
}

void view_ui_build(struct view_state *st)
{
    float h = (float)st->gpu.ui_h - 2 * VIEW_UI_MARGIN;

    /* The focusable rows are counted as they are emitted; the list changes
     * with the mode, so the focus is clamped to what was there last time. */
    st->ui_row = 0;
    st->ui_edit_active = false;
    st->menu.anchored = false;
    if (st->ui_rows > 0)
        st->focus = (st->focus % st->ui_rows + st->ui_rows) % st->ui_rows;

    /* Before the panel, which is what puts the panel on top of it. */
    view_crop_draw(view_nk, st);

    if (nk_begin(view_nk, "cineview",
                 nk_rect((float)st->gpu.ui_w - VIEW_UI_WIDTH - VIEW_UI_MARGIN,
                         VIEW_UI_MARGIN, VIEW_UI_WIDTH, h),
                 NK_WINDOW_BORDER | NK_WINDOW_TITLE | NK_WINDOW_SCROLL_AUTO_HIDE))
        view_panel(view_nk, st);

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

    st->ui_rows = st->ui_row;
    st->adjust = 0;
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

    view_gpu_lock(g);
    done = nk_sdl_render(g->queue, (uint32_t)slot, view_ui_hold_sem[slot],
                         NK_ANTI_ALIASING_ON);
    view_gpu_unlock(g);

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
