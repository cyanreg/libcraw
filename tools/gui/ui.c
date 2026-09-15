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

#define VIEW_UI_WIDTH  VIEW_PANEL_WIDTH
#define VIEW_UI_MARGIN VIEW_PANEL_MARGIN
#define VIEW_ROW       24.0f

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
                              float fine, float coarse, float lo, float hi)
{
    int adj = view_step(st, focused);

    if (adj) {
        *v = CRAW_CLAMP(*v + (float)adj * (st->adjust_coarse ? coarse : fine), lo, hi);
        /* Three sixths of a stop back from a half is not -0.00. */
        if (fabsf(*v) < fine * 1e-3f)
            *v = 0.0f;
    }
}

static void view_adjust_cycle(struct view_state *st, bool focused, int *v, int n)
{
    int adj = view_step(st, focused);

    if (adj)
        *v = (*v + adj + n) % n;
}

static void view_panel_transport(struct nk_context *ctx, struct view_state *st)
{
    int64_t total = st->dec.nb_frames;
    int pos;

    nk_layout_row_dynamic(ctx, VIEW_ROW, 4);
    if (nk_button_label(ctx, st->playing ? "Pause" : "Play"))
        st->playing = !st->playing;
    if (nk_button_label(ctx, "|<")) {
        st->playing = false;
        st->seek_to = st->index > 0 ? st->index - 1 : 0;
        st->need_frame = true;
    }
    if (nk_button_label(ctx, ">|")) {
        st->playing = false;
        st->need_frame = true;
    }
    if (nk_button_label(ctx, "<<")) {
        st->playing = false;
        st->seek_to = 0;
        st->need_frame = true;
    }

    if (total > 1) {
        bool f;
        int adj;

        pos = (int)CRAW_CLAMP(st->index, 0, total - 1);
        nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
        f = view_row(st, ctx);
        adj = view_step(st, f);
        if (adj) {
            pos = (int)CRAW_CLAMP((int64_t)pos + adj * (st->adjust_coarse ? 10 : 1),
                                  0, total - 1);
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
                          0.05f, 0.2f, 0.0f, 1.0f);
        nk_property_float(ctx, "#ambience", 0.0f, &st->awb.params.ambience, 1.0f,
                          0.05f, 0.005f);
        break;
    }

    case VIEW_WB_MANUAL:
        view_adjust_float(st, view_row(st, ctx), &lk->manual.cct,
                          100.0f, 500.0f, 2000.0f, 11500.0f);
        nk_property_float(ctx, "#CCT (K)", 2000.0f, &lk->manual.cct, 11500.0f,
                          50.0f, 5.0f);
        nk_slider_float(ctx, 2000.0f, &lk->manual.cct, 11500.0f, 10.0f);
        view_adjust_float(st, view_row(st, ctx), &lk->manual.tint,
                          0.01f, 0.05f, -0.2f, 0.2f);
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
         * step of one entry is a step of a third; coarse moves a stop. */
        pick = sel;
        view_adjust_cycle(st, view_row(st, ctx), &pick,
                          nb_items);
        if (st->adjust_coarse && pick != sel)
            pick = (int)CRAW_CLAMP(sel + (pick > sel ? 3 : -3), 0, nb_items - 1);
        nk_label(ctx, "exposure index", NK_TEXT_LEFT);
        {
            int chosen = nk_combo(ctx, items, nb_items, pick, 20, nk_vec2(140, 260));
            if (chosen != sel)
                lk->ei = view_iso_list[chosen];
        }
    }

    nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
    view_adjust_float(st, view_row(st, ctx), &lk->exposure,
                      1.0f / 6.0f, 1.0f, -3.0f, 3.0f);
    nk_property_float(ctx, "#bias (stops)", -3.0f, &lk->exposure, 3.0f, 0.1f, 0.01f);
    nk_slider_float(ctx, -3.0f, &lk->exposure, 3.0f, 0.05f);
    nk_labelf(ctx, NK_TEXT_LEFT, "gain %.2f, ceiling %.0f nits",
              view_total_gain(st) * exp2f(lk->exposure), view_peak_ceiling(st));
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
                          0.05f, 0.2f, 0.0f, 1.0f);
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
    if (st->dec.raw && nk_tree_push(ctx, NK_TREE_TAB, "White balance", NK_MAXIMIZED)) {
        view_panel_wb(ctx, st);
        nk_tree_pop(ctx);
    }

    if (nk_tree_push(ctx, NK_TREE_TAB, "Exposure", NK_MAXIMIZED)) {
        view_panel_exposure(ctx, st);
        nk_tree_pop(ctx);
    }

    if (st->dec.raw && nk_tree_push(ctx, NK_TREE_TAB, "Debayer", NK_MAXIMIZED)) {
        int dbg = (int)lk->debayer;

        nk_layout_row_dynamic(ctx, VIEW_ROW, 3);
        view_adjust_cycle(st, view_row(st, ctx), &dbg, CRAW_DEBAYER_NB);
        lk->debayer = (enum CRAWDebayer)dbg;
        for (int i = 0; i < CRAW_DEBAYER_NB; i++)
            if (nk_option_label(ctx, view_debayer_names[i], lk->debayer == i))
                lk->debayer = (enum CRAWDebayer)i;
        nk_tree_pop(ctx);
    }

    if (nk_tree_push(ctx, NK_TREE_TAB, "Look", NK_MAXIMIZED)) {
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

        nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
        nk_label(ctx, "CUBE LUT", NK_TEXT_LEFT);
        if (nk_edit_string_zero_terminated(ctx, NK_EDIT_FIELD, lk->lut_path,
                                           sizeof(lk->lut_path),
                                           nk_filter_default) & NK_EDIT_ACTIVE)
            st->ui_edit_active = true;

        nk_layout_row_dynamic(ctx, VIEW_ROW, 3);
        if (nk_button_label(ctx, "Load"))
            view_lut_load(st);
        if (nk_button_label(ctx, "Clear"))
            view_lut_clear(st);
        sel = lk->lut_type == PL_LUT_NORMALIZED ? 1 : 0;
        view_adjust_cycle(st, view_row(st, ctx), &sel, 2);
        {
            static const char *lut_types[] = { "native", "normalized" };
            sel = nk_combo(ctx, lut_types, 2, sel, 20, nk_vec2(140, 60));
            lk->lut_type = sel ? PL_LUT_NORMALIZED : PL_LUT_NATIVE;
        }

        nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
        view_adjust_cycle(st, view_row(st, ctx), &lk->lut_space, view_nb_lut_spaces);
        nk_label(ctx, "LUT input", NK_TEXT_LEFT);
        sel = nk_combo(ctx, view_lut_space_names, view_nb_lut_spaces,
                       lk->lut_space, 20, nk_vec2(160, 120));
        if (sel != lk->lut_space) {
            lk->lut_space = sel;
            view_lut_apply_space(st->lut, sel);
        }

        if (st->lut_error[0]) {
            nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
            nk_labelf(ctx, NK_TEXT_LEFT, "%s", st->lut_error);
        }
        nk_tree_pop(ctx);
    }

    if (nk_tree_push(ctx, NK_TREE_TAB, "Export", NK_MINIMIZED)) {
        static int first = 0, last = 0;
        static char dir[VIEW_MAX_PATH];

        nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
        nk_property_int(ctx, "#first", 0,  &first,
                        (int)CRAW_MAX(st->dec.nb_frames - 1, 0), 1, 1);
        nk_property_int(ctx, "#last", 0, &last,
                        (int)CRAW_MAX(st->dec.nb_frames - 1, 0), 1, 1);

        nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
        nk_label(ctx, "directory (empty: next to the clip)", NK_TEXT_LEFT);
        if (nk_edit_string_zero_terminated(ctx, NK_EDIT_FIELD, dir, sizeof(dir),
                                           nk_filter_default) & NK_EDIT_ACTIVE)
            st->ui_edit_active = true;

        nk_layout_row_dynamic(ctx, VIEW_ROW, 2);
        if (nk_button_label(ctx, st->exp.active ? "Stop" : "Export")) {
            if (st->exp.active) {
                st->exp.active = false;
            } else {
                st->playing = false;
                view_export_begin(st, first, last, dir[0] ? dir : view_clip_dir(st));
            }
        }
        if (st->exp.active && st->exp.last >= st->exp.first) {
            nk_size p = (nk_size)(st->exp.done * 100 /
                                  (st->exp.last - st->exp.first + 1));
            nk_progress(ctx, &p, 100, nk_false);
        } else {
            nk_label(ctx, "", NK_TEXT_LEFT);
        }

        if (st->exp.status[0]) {
            nk_layout_row_dynamic(ctx, VIEW_ROW, 1);
            nk_labelf(ctx, NK_TEXT_LEFT, "%s", st->exp.status);
        }
        nk_tree_pop(ctx);
    }
}

void view_ui_build(struct view_state *st)
{
    float h = (float)st->gpu.ui_h - 2 * VIEW_UI_MARGIN;

    /* The focusable rows are counted as they are emitted; the list changes
     * with the mode, so the focus is clamped to what was there last time. */
    st->ui_row = 0;
    st->ui_edit_active = false;
    if (st->ui_rows > 0)
        st->focus = (st->focus % st->ui_rows + st->ui_rows) % st->ui_rows;

    if (nk_begin(view_nk, "craw-view",
                 nk_rect((float)st->gpu.ui_w - VIEW_UI_WIDTH - VIEW_UI_MARGIN,
                         VIEW_UI_MARGIN, VIEW_UI_WIDTH, h),
                 NK_WINDOW_BORDER | NK_WINDOW_TITLE | NK_WINDOW_SCROLL_AUTO_HIDE))
        view_panel(view_nk, st);

    nk_end(view_nk);

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
