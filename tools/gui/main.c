/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* craw-view: open a camera raw clip, decode it on the GPU, run every frame
 * through libcraw on the same device and show the result through
 * libplacebo, with a nuklear panel for the grade and a still exporter.
 *
 * craw-process is the batch tool and the reference for the profile,
 * exposure and white balance semantics; this is the interactive one. The
 * options it shares with craw-process mean exactly the same thing. */

#include <getopt.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <libavutil/time.h>

#include "view.h"
#include "grain_ui.h"
#include "../profiles.h"

/* The embedded film simulation collection is optional; without it --lut
 * only takes a file and --list-luts says there is nothing to list. */
#ifndef CRAW_VIEW_HAVE_LUTS
#define CRAW_VIEW_HAVE_LUTS 0
#endif

#if CRAW_VIEW_HAVE_LUTS
#include "luts.h"
#endif

#define VIEW_WINDOW_W 1600
#define VIEW_WINDOW_H 900

/* BT.2408 diffuse white, which is what a pipeline value of 1.0 means. */
#define VIEW_REF_WHITE_NITS 203.0f

/* ---------------------------------------------------------------- tables */

/* The standard third-stop ISO list, which is what Apple's plug-in restricts
 * the working exposure index to (DESIGN.md 9b). */
const float view_iso_list[] = {
      50.0f,    64.0f,    80.0f,   100.0f,   125.0f,   160.0f,   200.0f,
     250.0f,   320.0f,   400.0f,   500.0f,   640.0f,   800.0f,  1000.0f,
    1250.0f,  1600.0f,  2000.0f,  2500.0f,  3200.0f,  4000.0f,  5000.0f,
    6400.0f,  8000.0f, 10000.0f, 12800.0f, 16000.0f, 20000.0f, 25600.0f,
};

const int view_nb_iso = (int)CRAW_ARRAY_ELEMS(view_iso_list);

const char *const view_debayer_names[] = { "bilinear", "mhc", "rcd" };

/* ---------------------------------------------------------------- helpers */

void view_log(const struct view_state *st, const char *fmt, ...)
{
    va_list args;

    if (!st->cfg.verbose)
        return;

    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

float view_total_gain(const struct view_state *st)
{
    CRAWColorParams cp;
    CRAWFrameParams fp;

    if (!st->have_prof)
        return 1.0f;

    view_frame_params(st, &fp);

    if (craw_profile_eval(&st->prof, fp.wb_mode, &fp.illuminant, &cp) < 0)
        return st->prof.gain > 0.0f ? st->prof.gain : 1.0f;

    return cp.gain;
}

/* A pixel clipped in only some channels comes out of the camera matrix
 * higher than a neutral clip does, because the rows of the combined matrix
 * do not all sum to one on their own. The ceiling the tone mapper is told
 * about has to cover that, or a saturated highlight would sit above what
 * libplacebo believes is the maximum. */
static float view_matrix_headroom(const CRAWColorParams *cp)
{
    float out[3][3];
    float headroom = 1.0f;

    if (craw_colorspace_from_xyz_d65(CRAW_CS_BT2020_LINEAR, out) < 0)
        return headroom;

    for (int i = 0; i < 3; i++) {
        float sum = 0.0f;

        for (int j = 0; j < 3; j++) {
            float c = 0.0f;

            for (int k = 0; k < 3; k++)
                c += out[i][k] * cp->cam_to_xyz[k][j];

            /* The extreme of the unit cube for this output channel. */
            if (c > 0.0f)
                sum += c;
        }

        headroom = CRAW_MAX(headroom, sum);
    }

    return headroom;
}

/* The exposure model the raw path hands to libcraw, as a plain linear gain
 * for the non-raw path's shader: the bias in stops times the working index
 * over the as-shot one. */
float view_video_exposure(const struct view_state *st)
{
    float gain = exp2f(st->look.exposure);

    if (st->look.ei > 0.0f && st->dec.meta_as_shot_ei > 0.0f)
        gain *= st->look.ei / st->dec.meta_as_shot_ei;

    return gain;
}

float view_peak_ceiling(const struct view_state *st)
{
    CRAWColorParams cp;
    CRAWFrameParams fp;

    /* Nothing here computes a ceiling for an already developed clip; what
     * libplacebo infers from the stream's own tags is the answer. */
    if (!st->dec.raw)
        return view_gpu_infer_ceiling(&st->gpu);

    if (!st->have_prof)
        return VIEW_REF_WHITE_NITS;

    view_frame_params(st, &fp);

    if (craw_profile_eval(&st->prof, fp.wb_mode, &fp.illuminant, &cp) < 0)
        return VIEW_REF_WHITE_NITS;

    return VIEW_REF_WHITE_NITS * cp.gain * exp2f(st->look.exposure)
         * view_matrix_headroom(&cp);
}

/* The noise model the denoise stage would use for the frame on screen, and
 * the exposure index it was looked up at.
 *
 * That index is the sensor's own, `as_shot_ei`, not the working one the
 * Exposure section sets: rating a clip at a different index scales the
 * picture afterwards, it does not change how the sensels were read
 * (DESIGN.md 14.2). Returns false when the camera has no model, which is
 * the one case the stage has to stay off in - without one it cannot know
 * what to remove, and craw_pipeline_record() refuses rather than guess. */
bool view_denoise_model(const struct view_state *st, CRAWNoiseModel *nm, float *iso)
{
    CRAWNoiseModel local;

    if (!st->dec.raw || !st->have_prof)
        return false;

    if (iso)
        *iso = st->prof.as_shot_ei;

    return craw_profile_noise_eval(&st->prof, st->prof.as_shot_ei,
                                   nm ? nm : &local) >= 0;
}

/* The strength the pass will actually run at: zero whenever the switch is
 * off, the strength is zero, or the camera has no model. Everything that
 * has to agree on whether the stage ran - the frame parameters, the look
 * the picture was processed with, the panel - asks this. */
float view_denoise_strength(const struct view_state *st)
{
    if (!st->denoise.on || !(st->denoise.strength > 0.0f))
        return 0.0f;
    if (!view_denoise_model(st, NULL, NULL))
        return 0.0f;

    return st->denoise.strength;
}

void view_frame_params(const struct view_state *st, CRAWFrameParams *fp)
{
    craw_frame_params_init(fp);

    fp->profile = &st->prof;
    fp->exposure_stops = st->look.exposure;
    /* The model itself is left to the profile: noise_override stays zero,
     * so craw_pipeline_record() evaluates the table at the frame's own
     * as-shot index. */
    fp->denoise_strength = view_denoise_strength(st);

    if (st->look.wb == VIEW_WB_MANUAL && st->prof.nb_wb_knots) {
        fp->wb_mode = CRAW_WB_REBALANCE;
        fp->illuminant = st->look.manual;
    } else if (st->look.wb == VIEW_WB_AUTO && st->awb.have_illum) {
        fp->wb_mode = CRAW_WB_REBALANCE;
        fp->illuminant = st->awb.illum;
    }
}

struct view_pipeline_look view_pipeline_look(const struct view_state *st)
{
    return (struct view_pipeline_look) {
        .wb         = st->look.wb,
        .manual     = st->look.manual,
        .illum      = st->awb.have_illum ? st->awb.illum : (CRAWIlluminant) { 0 },
        .have_illum = st->awb.have_illum,
        .ei         = st->look.ei,
        .exposure   = st->look.exposure,
        .ambience   = st->awb.params.ambience,
        .debayer    = st->look.debayer,
        .denoise    = view_denoise_strength(st),
    };
}

/* Field by field: the padding of a compound literal is nobody's promise,
 * so memcmp() would see changes that are not there. */
bool view_pipeline_look_equal(const struct view_pipeline_look *a,
                              const struct view_pipeline_look *b)
{
    return a->wb == b->wb &&
           a->manual.cct == b->manual.cct && a->manual.tint == b->manual.tint &&
           a->have_illum == b->have_illum &&
           a->illum.cct == b->illum.cct && a->illum.tint == b->illum.tint &&
           a->ei == b->ei && a->exposure == b->exposure &&
           a->ambience == b->ambience && a->debayer == b->debayer &&
           a->denoise == b->denoise;
}

const char *view_clip_dir(const struct view_state *st)
{
    static char dir[VIEW_MAX_PATH];
    const char *slash = strrchr(st->cfg.input, '/');
    size_t n;

    if (!slash)
        return ".";

    n = CRAW_MIN((size_t)(slash - st->cfg.input), sizeof(dir) - 1);
    memcpy(dir, st->cfg.input, n);
    dir[n] = '\0';

    return dir;
}

void view_lut_clear(struct view_state *st)
{
    pl_lut_free(&st->lut);
    st->lut_error[0] = '\0';
    st->look.lut_path[0] = '\0';
    st->look.film_sim = -1;
}

const char *view_lut_film_name(const struct view_state *st)
{
#if CRAW_VIEW_HAVE_LUTS
    const ViewLut *l = view_lut_get(st->look.film_sim);

    if (l)
        return l->name;
#else
    (void)st;
#endif
    return "none";
}

#if CRAW_VIEW_HAVE_LUTS

int view_lut_select_film(struct view_state *st, int index)
{
    const ViewLut *l = view_lut_get(index);
    char *text = NULL;
    size_t len = 0;
    int ret;

    st->lut_error[0] = '\0';

    if (!l) {
        snprintf(st->lut_error, sizeof(st->lut_error), "no such film simulation");
        return CRAW_ERROR(EINVAL);
    }

    /* The pack stores the LUT quantized; this is the one place it is
     * expanded, and only ever for the rendering that was picked. */
    ret = view_luts_cube_text(index, &text, &len);
    if (ret < 0) {
        snprintf(st->lut_error, sizeof(st->lut_error), "cannot unpack the LUT");
        return ret;
    }

    pl_lut_free(&st->lut);
    st->lut = pl_lut_parse_cube(st->gpu.log, text, len);
    view_luts_cube_free(text);
    if (!st->lut) {
        snprintf(st->lut_error, sizeof(st->lut_error), "the packed LUT did not parse");
        return CRAW_ERROR(EINVAL);
    }

    /* One LUT at a time: a film simulation and a user .cube would
     * otherwise both claim the renderer's single LUT slot. */
    st->look.lut_path[0] = '\0';
    st->look.film_sim = index;
    view_lut_apply_space(st->lut, st->look.lut_space);

    view_log(st, "film simulation: %s (%s, %s) %dx%dx%d\n", l->name,
             view_lut_film(l->film)->name, l->label,
             st->lut->size[0], st->lut->size[1], st->lut->size[2]);

    return 0;
}

/* Prints both category trees, indented, the way the menu walks them. */
static void view_list_luts(void)
{
    printf("Films\n");
    for (int ci = 0; ci < view_luts_film_category_count(); ci++) {
        const ViewLutFilmCategory *c = view_lut_film_category(ci);

        printf("  %s\n", c->name);
        for (int fi = c->first_film; fi < c->first_film + c->n_films; fi++) {
            const ViewLutFilm *f = view_lut_film(fi);

            printf("    %s\n", f->name);
            for (int i = f->first_lut; i < f->first_lut + f->n_luts; i++) {
                const ViewLut *l = view_lut_get(i);

                printf("      %-24s %-18s %s\n", l->name, l->label, l->slug);
            }
        }
    }

    printf("Looks\n");
    for (int ci = 0; ci < view_luts_look_category_count(); ci++) {
        const ViewLutLookCategory *c = view_lut_look_category(ci);

        printf("  %s\n", c->name);
        for (int i = 0; i < c->n_luts; i++) {
            const ViewLut *l = view_lut_get((int)c->luts[i]);

            printf("    %-24s %-26s %s\n", l->name,
                   view_lut_film(l->film)->name, l->slug);
        }
    }
}

#else /* !CRAW_VIEW_HAVE_LUTS */

int view_lut_select_film(struct view_state *st, int index)
{
    (void)index;
    snprintf(st->lut_error, sizeof(st->lut_error),
             "built without the film simulation collection");
    return CRAW_ERROR(ENOTSUP);
}

static void view_list_luts(void)
{
    printf("This build carries no film simulation collection.\n");
}

#endif /* CRAW_VIEW_HAVE_LUTS */

/* Resolves what --lut was given. A real file always wins, so a rendering
 * whose name happens to collide with something on disk can never shadow
 * it; failing that the name is looked up in the embedded collection by
 * slug and then by display name. */
static int view_lut_startup(struct view_state *st)
{
    FILE *f;

    if (!st->look.lut_path[0])
        return 0;

    f = fopen(st->look.lut_path, "rb");
    if (f) {
        fclose(f);
        return view_lut_load(st);
    }

#if CRAW_VIEW_HAVE_LUTS
    {
        int ix = view_luts_find(st->look.lut_path);

        if (ix >= 0)
            return view_lut_select_film(st, ix);
    }
    snprintf(st->lut_error, sizeof(st->lut_error),
             "no such file and no such film simulation");
#else
    snprintf(st->lut_error, sizeof(st->lut_error), "cannot open the LUT");
#endif

    return CRAW_ERROR(ENOENT);
}

int view_lut_load(struct view_state *st)
{
    char *data = NULL;
    size_t size = 0;
    FILE *f;
    int ret = 0;

    st->lut_error[0] = '\0';

    if (!st->look.lut_path[0]) {
        pl_lut_free(&st->lut);
        return 0;
    }

    f = fopen(st->look.lut_path, "rb");
    if (!f) {
        snprintf(st->lut_error, sizeof(st->lut_error), "cannot open the LUT");
        return CRAW_ERROR(ENOENT);
    }

    if (fseek(f, 0, SEEK_END) == 0) {
        long end = ftell(f);
        if (end > 0)
            size = (size_t)end;
        rewind(f);
    }

    if (!size || size > (64u << 20)) {
        snprintf(st->lut_error, sizeof(st->lut_error), "the LUT is empty or absurd");
        ret = CRAW_ERROR(EINVAL);
        goto fail;
    }

    data = malloc(size);
    if (!data) {
        ret = CRAW_ERROR(ENOMEM);
        goto fail;
    }
    if (fread(data, 1, size, f) != size) {
        snprintf(st->lut_error, sizeof(st->lut_error), "short read");
        ret = CRAW_ERROR(EIO);
        goto fail;
    }

    pl_lut_free(&st->lut);
    st->lut = pl_lut_parse_cube(st->gpu.log, data, size);
    if (!st->lut) {
        snprintf(st->lut_error, sizeof(st->lut_error), "not a .cube LUT");
        ret = CRAW_ERROR(EINVAL);
        goto fail;
    }

    /* One LUT at a time: loading a file drops any film simulation. */
    st->look.film_sim = -1;
    view_lut_apply_space(st->lut, st->look.lut_space);

    view_log(st, "loaded the LUT %s (%dx%dx%d)\n", st->look.lut_path,
             st->lut->size[0], st->lut->size[1], st->lut->size[2]);

fail:
    free(data);
    fclose(f);
    return ret;
}

/* ------------------------------------------------------------- the clock */

double view_clock(struct view_state *st)
{
    if (st->audio.available)
        return view_audio_clock(&st->audio);

    /* A silent clip gets the same pacing from wall time, anchored at the
     * last start or seek. */
    if (!st->playing)
        return st->wall_pts;

    return st->wall_pts + (double)(av_gettime_relative() - st->wall_t0) / 1.0e6;
}

static void view_wall_restart(struct view_state *st, double pts)
{
    st->wall_pts = pts;
    st->wall_t0 = av_gettime_relative();
}

static double view_frame_duration(const struct view_state *st)
{
    return st->dec.frame_rate.num ? 1.0 / av_q2d(st->dec.frame_rate) : 1.0 / 29.97;
}

/* Audio is the master clock and is never made to wait for the picture: a
 * frame is stepped only once the one on screen has had its turn, and the
 * lead accounts for what it costs to get the next one there - the decode
 * and pipeline time, plus half a display interval of presentation jitter.
 * Without that the picture would sit a consistent half-refresh late. */
static bool view_frame_due(struct view_state *st)
{
    double dur, lead;

    if (st->cfg.bench || !st->have_frame)
        return true;

    dur = view_frame_duration(st);
    lead = st->step_cost + st->loop_period * 0.5;

    /* Never launch more than half a frame early. Without the clamp the
     * first step's one-time cost - mapping textures, starting a hwaccel,
     * compiling shaders - seeds the estimate at a few hundred milliseconds
     * and the picture sprints that far ahead of the audio before the
     * average decays. */
    lead = CRAW_MIN(lead, dur * 0.5);

    return view_clock(st) + lead >= st->pts + dur;
}

/* Fills the ring before the device is unpaused, so that a run never starts
 * on an underrun. */
static void view_audio_start(struct view_state *st)
{
    if (!st->audio.available || !st->playing)
        return;

    view_audio_pump(&st->audio);
    view_audio_play(&st->audio);
}

/* ------------------------------------------------------------ the view */

void view_picture_area(const struct view_state *st, pl_rect2df *out)
{
    float w = (float)st->gpu.ui_w - VIEW_PANEL_WIDTH - 2 * VIEW_PANEL_MARGIN;

    *out = (pl_rect2df) {
        0.0f, 0.0f, CRAW_MAX(w, 16.0f), CRAW_MAX((float)st->gpu.ui_h, 16.0f),
    };
}

/* The scale at which the whole picture just fits the area. */
static float view_fit_scale(const struct view_state *st)
{
    pl_rect2df area;

    if (!st->gpu.width || !st->gpu.height)
        return 1.0f;

    view_picture_area(st, &area);

    return CRAW_MIN(pl_rect_w(area) / (float)st->gpu.width,
                    pl_rect_h(area) / (float)st->gpu.height);
}

float view_zoom_scale(const struct view_state *st)
{
    return st->view.fit ? view_fit_scale(st) : st->view.zoom;
}

void view_zoom_rects(const struct view_state *st, pl_rect2df *src, pl_rect2df *dst)
{
    float z = view_zoom_scale(st);
    float sw = (float)st->gpu.width, sh = (float)st->gpu.height;
    pl_rect2df area;
    float dw, dh, vw, vh, cx, cy;

    view_picture_area(st, &area);

    /* What of the picture is on screen, and how big it is there. */
    dw = CRAW_MIN(sw * z, pl_rect_w(area));
    dh = CRAW_MIN(sh * z, pl_rect_h(area));
    vw = dw / z;
    vh = dh / z;

    /* The centre is clamped so that no edge of the area is ever left empty
     * while the picture is big enough to fill it. */
    cx = vw >= sw ? sw * 0.5f : CRAW_CLAMP(st->view.cx, vw * 0.5f, sw - vw * 0.5f);
    cy = vh >= sh ? sh * 0.5f : CRAW_CLAMP(st->view.cy, vh * 0.5f, sh - vh * 0.5f);

    *src = (pl_rect2df) { cx - vw * 0.5f, cy - vh * 0.5f,
                          cx + vw * 0.5f, cy + vh * 0.5f };

    /* Centred in the area, which is what letterboxes a fitted picture. */
    *dst = (pl_rect2df) {
        area.x0 + (pl_rect_w(area) - dw) * 0.5f,
        area.y0 + (pl_rect_h(area) - dh) * 0.5f,
        0.0f, 0.0f,
    };
    dst->x1 = dst->x0 + dw;
    dst->y1 = dst->y0 + dh;
}

void view_zoom_by(struct view_state *st, float notches, int sx, int sy,
                  bool at_cursor)
{
    float fit = view_fit_scale(st);
    float z = view_zoom_scale(st);
    float want = z * powf(VIEW_ZOOM_STEP, notches);
    pl_rect2df src, dst;
    float px, py;

    if (!st->gpu.width)
        return;

    /* One-to-one is the stop a raw tool is judged on, so a step that would
     * pass it lands on it exactly. */
    if ((z < 1.0f && want > 1.0f) || (z > 1.0f && want < 1.0f))
        want = 1.0f;

    want = CRAW_CLAMP(want, fit, VIEW_ZOOM_MAX);

    /* Where the pointer is in the picture now, so it can be put back. */
    view_zoom_rects(st, &src, &dst);
    if (at_cursor && (float)sx >= dst.x0 && (float)sx <= dst.x1 &&
        (float)sy >= dst.y0 && (float)sy <= dst.y1) {
        px = src.x0 + ((float)sx - dst.x0) / z;
        py = src.y0 + ((float)sy - dst.y0) / z;
    } else {
        at_cursor = false;
        px = py = 0.0f;
    }

    if (want <= fit * 1.0001f) {
        st->view.fit = true;
        st->view.zoom = fit;
        st->view.cx = (float)st->gpu.width * 0.5f;
        st->view.cy = (float)st->gpu.height * 0.5f;
        return;
    }

    st->view.fit = false;
    st->view.zoom = want;

    view_zoom_rects(st, &src, &dst);
    if (at_cursor) {
        st->view.cx = px - ((float)sx - dst.x0) / want + pl_rect_w(src) * 0.5f;
        st->view.cy = py - ((float)sy - dst.y0) / want + pl_rect_h(src) * 0.5f;
    }
}

void view_zoom_reset(struct view_state *st, bool one_to_one)
{
    st->view.fit = !one_to_one;
    st->view.zoom = one_to_one ? 1.0f : view_fit_scale(st);
    /* Fitting forgets the pan as well; one to one keeps looking where the
     * picture was, unless nowhere has been looked at yet. */
    if (!one_to_one || (!st->view.cx && !st->view.cy)) {
        st->view.cx = (float)st->gpu.width * 0.5f;
        st->view.cy = (float)st->gpu.height * 0.5f;
    }
}

void view_pan_fraction(struct view_state *st, float fx, float fy)
{
    pl_rect2df src, dst;

    if (st->view.fit)
        return;

    view_zoom_rects(st, &src, &dst);
    st->view.cx = src.x0 + pl_rect_w(src) * (0.5f + fx);
    st->view.cy = src.y0 + pl_rect_h(src) * (0.5f + fy);
}

void view_pan_pixels(struct view_state *st, int dx, int dy)
{
    float z = view_zoom_scale(st);
    pl_rect2df src, dst;

    if (st->view.fit)
        return;

    view_zoom_rects(st, &src, &dst);
    st->view.cx = src.x0 + pl_rect_w(src) * 0.5f - (float)dx / z;
    st->view.cy = src.y0 + pl_rect_h(src) * 0.5f - (float)dy / z;
}

/* ------------------------------------------------------------- the crop */

/* Window pixels to source pixels, through the same two rectangles the
 * renderer is handed, so that the rubber band lands on the pixels it is
 * drawn over at any zoom and pan. */
static void view_source_pos(const struct view_state *st, int px, int py,
                            float *sx, float *sy)
{
    pl_rect2df src, dst;
    float dw, dh;

    view_zoom_rects(st, &src, &dst);
    dw = pl_rect_w(dst);
    dh = pl_rect_h(dst);

    *sx = src.x0 + ((float)px - dst.x0) * (dw > 0.0f ? pl_rect_w(src) / dw : 0.0f);
    *sy = src.y0 + ((float)py - dst.y0) * (dh > 0.0f ? pl_rect_h(src) / dh : 0.0f);
}

/* A source coordinate snapped to an even pixel inside the frame: half a
 * Bayer quad is not a thing to crop to, and every encoder that subsamples
 * chroma wants an even offset and an even size. */
static int view_crop_snap(float v, int max)
{
    int i = (int)lrintf(v * 0.5f) * 2;

    return CRAW_CLAMP(i, 0, max & ~1);
}

void view_crop_begin(struct view_state *st, int px, int py)
{
    st->crop.dragging = true;
    view_source_pos(st, px, py, &st->crop.ax, &st->crop.ay);
}

void view_crop_drag(struct view_state *st, int px, int py)
{
    struct view_crop *c = &st->crop;
    float bx, by;
    int x0, y0, x1, y1;

    if (!c->dragging || !st->gpu.width || !st->gpu.height)
        return;

    view_source_pos(st, px, py, &bx, &by);

    x0 = view_crop_snap(CRAW_MIN(c->ax, bx), (int)st->gpu.width);
    y0 = view_crop_snap(CRAW_MIN(c->ay, by), (int)st->gpu.height);
    x1 = view_crop_snap(CRAW_MAX(c->ax, bx), (int)st->gpu.width);
    y1 = view_crop_snap(CRAW_MAX(c->ay, by), (int)st->gpu.height);

    /* The rectangle is replaced only once the pointer has really moved, so
     * that a stray click keeps the one that is already there. */
    if (x1 == x0 && y1 == y0)
        return;

    c->x0 = x0;
    c->y0 = y0;
    c->x1 = x1;
    c->y1 = y1;
    c->have = true;
}

void view_crop_end(struct view_state *st)
{
    struct view_crop *c = &st->crop;

    c->dragging = false;
    /* A rectangle too small to hold a pixel pair is a mis-click, not a
     * crop. */
    if (c->have && (c->x1 - c->x0 < 2 || c->y1 - c->y0 < 2))
        c->have = false;
}

void view_crop_clear(struct view_state *st)
{
    st->crop.have = false;
    st->crop.dragging = false;
}

bool view_crop_rect(const struct view_state *st, int *x, int *y, int *w, int *h)
{
    const struct view_crop *c = &st->crop;
    int sw = (int)st->gpu.width, sh = (int)st->gpu.height;
    bool have = c->have;
    int x0 = 0, y0 = 0, x1 = sw, y1 = sh;

    if (have) {
        /* The command line sets a rectangle before the clip's size is
         * known, so this is where it meets it. */
        x0 = CRAW_CLAMP(c->x0, 0, sw);
        y0 = CRAW_CLAMP(c->y0, 0, sh);
        x1 = CRAW_CLAMP(c->x1, x0, sw);
        y1 = CRAW_CLAMP(c->y1, y0, sh);

        if (x1 - x0 < 2 || y1 - y0 < 2) {
            x0 = y0 = 0;
            x1 = sw;
            y1 = sh;
            have = false;
        }
    }

    *x = x0;
    *y = y0;
    *w = x1 - x0;
    *h = y1 - y0;

    return have;
}

bool view_crop_screen(const struct view_state *st, pl_rect2df *out)
{
    pl_rect2df src, dst;
    float zx, zy;
    int x, y, w, h;

    if (!st->crop.have || !view_crop_rect(st, &x, &y, &w, &h))
        return false;

    view_zoom_rects(st, &src, &dst);
    if (!(pl_rect_w(src) > 0.0f) || !(pl_rect_h(src) > 0.0f))
        return false;

    zx = pl_rect_w(dst) / pl_rect_w(src);
    zy = pl_rect_h(dst) / pl_rect_h(src);

    *out = (pl_rect2df) {
        dst.x0 + ((float)x - src.x0) * zx,
        dst.y0 + ((float)y - src.y0) * zy,
        dst.x0 + ((float)(x + w) - src.x0) * zx,
        dst.y0 + ((float)(y + h) - src.y0) * zy,
    };

    return true;
}

/* ------------------------------------------------------------- one frame */

static int view_awb_step(struct view_state *st)
{
    CRAWAwbHistogram hist;
    CRAWIlluminant out;
    int ret;

    /* The histogram read is of the frame before the one just submitted, so
     * the only fence involved is that frame's, which has had a whole frame
     * period to retire. The estimate therefore lands on the frame after
     * next rather than the next one; at thirty frames a second that is
     * another thirty milliseconds of lag on a track that is smoothed over a
     * second anyway. */
    ret = view_gpu_wait_prev(&st->gpu, &st->timing);
    if (ret < 0)
        return ret;

    if (st->gpu.submitted < 2)
        return 0;

    /* EINVAL here means the pipeline has not recorded that many frames yet,
     * which happens right after a rebuild for another debayer method: there
     * is simply no histogram to read this time round. */
    ret = craw_pipeline_get_awb_histogram_n(st->gpu.pipe, 1, &hist);
    if (ret == CRAW_ERROR(EINVAL))
        return 0;
    if (ret < 0)
        return ret;

    ret = craw_awb_estimate(&st->prof, &hist, &st->awb.params, &st->awb.last);
    if (ret < 0)
        return ret;

    ret = craw_awb_smoother_push(st->awb.smoother, st->pts, &st->awb.last, false, &out);
    if (ret < 0 && ret != CRAW_ERROR(EAGAIN))
        return ret;

    /* EAGAIN means there is no track yet and this estimate cannot start one,
     * so the next frame stays on whatever the last good one left. */
    if (ret >= 0) {
        st->awb.illum = out;
        st->awb.have_illum = true;
    }

    return 0;
}

/* The raw path from the held picture to a submitted pipeline pass: the
 * profile from the frame's side data, the exposure model, the parameters
 * from the look, the pass itself and the auto white balance step that reads
 * the previous pass's histogram. */
static int view_process_frame(struct view_state *st)
{
    CRAWFrameParams fp;
    int ret;

    st->prof = st->base;
    ret = view_profile_from_frame(&st->prof, st->frame);
    if (ret < 0)
        return ret;

    /* The same exposure model craw-process uses: the frame gain times the
     * working index over the as-shot ISO, times the bias in stops. */
    if (!(st->prof.as_shot_ei > 0.0f))
        st->prof.as_shot_ei = st->dec.meta_as_shot_ei;
    if (st->look.ei > 0.0f)
        st->prof.exposure_index = st->look.ei;
    else if (!(st->prof.exposure_index > 0.0f))
        st->prof.exposure_index = st->prof.as_shot_ei;
    st->have_prof = true;

    ret = view_gpu_ensure(&st->gpu, (uint32_t)st->frame->width,
                          (uint32_t)st->frame->height, st->look.debayer);
    if (ret < 0)
        return ret;

    view_frame_params(st, &fp);
    /* Taken before the white balance step below moves the track, so that a
     * new estimate counts as a change and a paused picture follows it. */
    st->processed = view_pipeline_look(st);

    ret = view_gpu_process(&st->gpu, st->frame, &fp, &st->timing);
    if (ret < 0)
        return ret;

    st->have_frame = true;

    /* What the panel reports, resolved once per frame rather than once per
     * panel redraw. */
    st->denoise.have_model = view_denoise_model(st, &st->denoise.model,
                                                &st->denoise.iso);

    /* libcraw's timestamps come back from a slot VIEW_SLOTS submissions
     * later, so the pass flag recorded with that slot says which of the two
     * averages this measurement belongs to. Held apart rather than
     * subtracted per frame, because the difference of two noisy numbers is
     * noise and the difference of two smoothed ones is not. */
    if (st->gpu.last_gpu_ms > 0.0) {
        double *acc = st->gpu.last_gpu_denoise ? &st->denoise.ms_on
                                               : &st->denoise.ms_off;

        *acc = *acc > 0.0 ? *acc * 0.9 + st->gpu.last_gpu_ms * 0.1
                          : st->gpu.last_gpu_ms;
    }

    /* The histogram describes the frame that was just processed; what comes
     * out of the smoother is applied to the next one, exactly as in
     * craw-process. */
    if (st->look.wb == VIEW_WB_AUTO && st->prof.nb_wb_knots) {
        ret = view_awb_step(st);
        if (ret < 0)
            return ret;
    }

    return 0;
}

int view_replay_frame(struct view_state *st)
{
    if (!st->dec.raw || !st->have_frame || !st->frame->buf[0] ||
        st->frame->format != AV_PIX_FMT_VULKAN) {
        st->processed = view_pipeline_look(st);
        return 0;
    }

    view_log(st, "replay frame %" PRId64 "\n", st->index);

    return view_process_frame(st);
}

int view_step_frame(struct view_state *st)
{
    AVStream *vs = st->dec.fmt->streams[st->dec.stream_index];
    int64_t start, pts, t0;
    int ret;

    if (st->seek_to >= 0) {
        /* Replaying the picture already on screen, which is how a grade
         * change repaints while paused, is not a move in time and must not
         * throw the illuminant track away. */
        bool jump = !st->have_frame || st->seek_to != st->index;
        int64_t want = st->seek_to;

        double target = st->dec.frame_rate.num
                      ? (double)st->seek_to / av_q2d(st->dec.frame_rate) : 0.0;

        ret = view_decoder_seek(&st->dec, st->seek_to);
        st->seek_to = -1;
        if (ret < 0)
            return ret;

        /* Both readers go with the seek: the audio fades out, drops
         * everything in front of the target and arms the new run. */
        view_audio_seek(&st->audio, target);
        view_wall_restart(st, target);
        view_gpu_reset_peak(&st->gpu, false);
        st->sync.run_frames = 0;
        st->seek_target = jump ? want : -1;
        st->seek_tries = 0;

        /* The clock now runs from the target, so the frame that lands there
         * is due at once. Left at the old picture's time, a backward seek
         * or a loop point would wait for the clock to climb all the way
         * back to it. */
        st->pts = target;

        /* A rebuilt decoder brings a whole new frame pool with it. */
        if (st->dec.frames_reset) {
            view_gpu_drop_src_views(&st->gpu);
            st->dec.frames_reset = false;
        }
        st->seek_tries = 0;

        if (jump) {
            craw_awb_smoother_reset(st->awb.smoother);
            st->awb.have_illum = false;
        }
    }

    t0 = av_gettime_relative();
    st->dec.read_ms = 0.0;
    ret = view_decoder_next(&st->dec, st->frame);
    st->decode_ms = (double)(av_gettime_relative() - t0) / 1000.0;
    st->timing.read += st->dec.read_ms;
    st->timing.decode += st->decode_ms - st->dec.read_ms;
    if (ret == 2) {
        /* The medium has not kept up. Leave the picture where it is: the
         * render loop must never block on a read. */
        st->starved++;
        st->timing.starved++;
        return 3;
    }
    st->timing.frames++;
    if (ret)
        return ret;

    start = vs->start_time != AV_NOPTS_VALUE ? vs->start_time : 0;
    pts = st->frame->pts != AV_NOPTS_VALUE ? st->frame->pts : 0;
    st->pts = (double)(pts - start) * av_q2d(st->dec.time_base);
    st->index = st->dec.frame_rate.num
              ? (int64_t)llrint(st->pts * av_q2d(st->dec.frame_rate))
              : st->index + 1;

    if (!st->dec.raw) {
        /* Already developed: straight from the decoder to libplacebo, with
         * the colour the stream itself declares - unless the clip metadata
         * says V-Log, which the camera's own VUI does not admit to. */
        ret = view_gpu_map(&st->gpu, st->frame, vs, st->dec.vlog, &st->timing);
        if (ret < 0)
            return ret;

        st->have_frame = true;
        av_frame_unref(st->frame);
        st->sync.run_frames++;

        return 0;
    }

    /* A decoder flushed after end of stream can still hand back a picture
     * from before the seek, so the landing is checked rather than trusted.
     * Every ProRes RAW frame is a keyframe, so one or two tries is all this
     * ever needs. */
    if (st->seek_target >= 0) {
        if ((st->index < st->seek_target || st->index > st->seek_target + 8) &&
            st->seek_tries < 8) {
            st->seek_tries++;
                av_frame_unref(st->frame);
            return 2;
        }
        st->seek_target = -1;
    }

    if (st->frame->format != AV_PIX_FMT_VULKAN) {
        fprintf(stderr, "the decoder produced software frames; craw-view needs the "
                        "Vulkan hwaccel for a raw clip\n");
        return CRAW_ERROR(ENOTSUP);
    }

    /* More than a frame behind the master clock: throw it away before it
     * costs a pipeline pass, which is the only way to catch up. Audio is
     * never stalled to wait for the picture. */
    if (st->playing && !st->exp.active && !st->cfg.bench && st->have_frame &&
        st->pts < view_clock(st) - view_frame_duration(st)) {
        st->sync.dropped++;
        av_frame_unref(st->frame);
        return 2;
    }

    ret = view_process_frame(st);
    if (ret < 0)
        return ret;

    /* The picture stays referenced for a replay; the next decode takes it
     * back. */
    st->sync.run_frames++;

    return 0;
}

/* --------------------------------------------------------------- options */

static int view_lookup(const char *const *names, int nb, const char *opt,
                       const char *arg, int *out)
{
    for (int i = 0; i < nb; i++) {
        if (!strcmp(names[i], arg)) {
            *out = i;
            return 0;
        }
    }

    fprintf(stderr, "%s: unknown value `%s`, expected one of:", opt, arg);
    for (int i = 0; i < nb; i++)
        fprintf(stderr, " %s", names[i]);
    fprintf(stderr, "\n");

    return CRAW_ERROR(EINVAL);
}

static int view_parse_wb(struct view_look *lk, const char *arg)
{
    char *end;
    double cct, tint = 0.0;

    if (!strcmp(arg, "as-shot")) {
        lk->wb = VIEW_WB_AS_SHOT;
        return 0;
    }
    if (!strcmp(arg, "auto")) {
        lk->wb = VIEW_WB_AUTO;
        return 0;
    }

    cct = strtod(arg, &end);
    if (end == arg || (*end && *end != ':')) {
        fprintf(stderr, "--wb: expected as-shot, auto or CCT[:TINT], got `%s`\n", arg);
        return CRAW_ERROR(EINVAL);
    }
    if (*end == ':') {
        const char *tp = end + 1;
        tint = strtod(tp, &end);
        if (end == tp || *end) {
            fprintf(stderr, "--wb: cannot parse the tint in `%s`\n", arg);
            return CRAW_ERROR(EINVAL);
        }
    }
    if (cct <= 0.0) {
        fprintf(stderr, "--wb: a colour temperature of %g K makes no sense\n", cct);
        return CRAW_ERROR(EINVAL);
    }

    lk->wb = VIEW_WB_MANUAL;
    lk->manual = (CRAWIlluminant) { .cct = (float)cct, .tint = (float)tint };

    return 0;
}

/* WxH+X+Y, the way every tool that takes a geometry spells it. The size
 * against the clip is checked later, when the clip's own size is known. */
static int view_parse_crop(struct view_crop *c, const char *arg)
{
    int w, h, x, y, n = 0;

    if (sscanf(arg, "%dx%d+%d+%d%n", &w, &h, &x, &y, &n) != 4 || arg[n] ||
        w <= 0 || h <= 0 || x < 0 || y < 0) {
        fprintf(stderr, "--crop: expected WxH+X+Y, got `%s`\n", arg);
        return CRAW_ERROR(EINVAL);
    }

    /* The same even edges the pointer draws, so that a rectangle read off
     * the panel and handed back on the command line is the same one. */
    c->x0 = view_crop_snap((float)x, INT_MAX);
    c->y0 = view_crop_snap((float)y, INT_MAX);
    c->x1 = view_crop_snap((float)(x + w), INT_MAX);
    c->y1 = view_crop_snap((float)(y + h), INT_MAX);
    c->have = true;

    return 0;
}

static void view_usage(FILE *f)
{
    fprintf(f,
"usage: craw-view [options] INPUT\n"
"\n"
"Opens a camera raw clip, decodes it with FFmpeg's Vulkan hwaccel, runs\n"
"every frame through libcraw on the same device and displays the result\n"
"through libplacebo, with a nuklear panel for the grade.\n"
"\n"
"  --device SUBSTR     use the Vulkan device whose name contains SUBSTR\n"
"  --debug             turn the Vulkan validation layer on\n"
"  --bench             play as fast as the pipeline allows instead of at the\n"
"                      clip's own frame rate\n"
"  --no-loop           stop at the end of the clip instead of starting over\n"
"  --loops N           play the clip N times and quit, printing the A/V\n"
"                      statistics (default: loop for ever)\n"
"  --audio-dump FILE   write exactly the float samples handed to the audio\n"
"                      device, raw interleaved float32 48 kHz stereo\n"
"  --audio-map A[,B]   audio stream indices, in audio stream order, to use\n"
"                      as left and right; one index means a stereo stream\n"
"                      or a mono one sent to both ears\n"
"  -v                  verbose: timings, metadata and libcraw's own log\n"
"  -h                  this text\n"
"\n"
"  --wb MODE           as-shot, auto, or CCT[:TINT] in Kelvin\n"
"  --ei INDEX          working exposure index; the frame is scaled by\n"
"                      INDEX / as-shot ISO\n"
"  --exposure STOPS    exposure bias on top of --ei (default: 0)\n"
"  --debayer METHOD    bilinear, mhc or rcd (default: rcd)\n"
"  --denoise STRENGTH  raw domain denoising, in units of the camera's\n"
"                      modelled noise level: 1.0 removes exactly that much,\n"
"                      0.25 to 3.0, 0 (the default) leaves it off. Needs a\n"
"                      noise model for the clip's ISO; the Denoise section\n"
"                      says which one it found\n"
"  --tone-map NAME     bt.2390, spline, reinhard or clip (default: bt.2390)\n"
"  --gamut-map NAME    perceptual, softclip, relative, clip or desaturate\n"
"  --lut NAME          a .cube LUT applied by the renderer: a file, or the\n"
"                      name or slug of one of the embedded film simulations\n"
"                      (a file of that name wins). --list-luts prints them\n"
"  --list-luts         print the embedded film simulation trees and exit\n"
"  --lut-type TYPE     native (gamma light) or normalized (linear light)\n"
"  --lut-space SPACE   the colour space the LUT was authored for: native\n"
"                      (the toggle above), bt709, hlg2020, linear2020 or\n"
"                      vlog (V-Log in V-Gamut, what Panasonic's own\n"
"                      V-Log to V-709 cubes take). Anything but native\n"
"                      makes it a conversion LUT to a BT.709 display.\n"
"  --no-peak-detect    tone map against the profile's sensor clip instead of\n"
"                      the peak libplacebo measures per frame\n"
"  --softlight PERCENT blend the picture with itself through the soft light\n"
"                      blend mode, 0 to 100 (default: 0, off)\n"
"  --vibrance PASTELS[,SATURATED[,THRESHOLD]]\n"
"                      raise chroma, more on pastel colours than on already\n"
"                      saturated ones. The two amounts run -100 to 100 and\n"
"                      one number sets both; THRESHOLD, 0 to 100, is the\n"
"                      chroma they meet at (default: 45). The panel also\n"
"                      has the skin protection and gamut toggles\n"
"  --grain GAUGE[,STOCK][:AMOUNT]\n"
"                      film grain, e.g. super8, 16mm,500t or super35,50d:0.6.\n"
"                      Gauges: regular8, super8, 16mm, super16, 2perf, 3perf,\n"
"                      academy, super35, 65mm, imax. Stocks: fine, medium,\n"
"                      coarse, 50d, 200t, 250d, 500t, doublex, trix,\n"
"                      ektachrome, portra400. The panel has the rest\n"
"  --zoom SCALE        start at fit, or at SCALE screen pixels per source\n"
"                      pixel (1 is one to one)\n"
"  --profile NAME      use a built-in per-CCT table set (%s)\n"
"  --tables FILE       load the per-CCT tables from a text file\n"
"\n"
"  --export FRAME OUT  write frame FRAME to the file OUT and exit, without\n"
"                      opening a window. The name picks the format through\n"
"                      libavformat: .png and .tif hold sixteen bits, .jpg,\n"
"                      .webp and .avif whatever their encoder takes. The\n"
"                      file holds the displayed look, sRGB encoded.\n"
"  --codec NAME        encode the still with this encoder rather than the\n"
"                      one the file name implies\n"
"  --enc KEY=VALUE[,...]\n"
"                      options for that encoder, e.g. cpu-used=8,threads=2\n"
"                      for an AVIF or qmin=1 for a JPEG\n"
"  --crop WxH+X+Y      export only that region of the source, in source\n"
"                      pixels, with the edges snapped to even ones. The\n"
"                      viewer shades everything outside it.\n"
"\n"
"Audio is the master clock. The clip's PCM track plays through SDL3 with\n"
"5 ms fades at every boundary; a silent clip falls back to wall time.\n"
"\n"
"Keys: space play/pause; up/down walk the panel rows; left/right adjust the\n"
"focused row; shift with left/right seeks one second; page up/down zoom in\n"
"and out; shift with page up/down goes to the first and last frame; control\n"
"with the arrows pans; tab or 0 fits the picture again, 1 is one to one;\n"
"enter opens the film simulation menu on its row and escape closes it;\n"
"the wheel zooms about the pointer and dragging pans; c turns crop mode on\n"
"and off, where a left drag draws the rectangle the export is cut to;\n"
"m mutes; q, or escape with no menu open, quits.\n",
    craw_tool_profile_builtin_list());
}

enum {
    VIEW_OPT_DEVICE = 0x100,
    VIEW_OPT_DEBUG,
    VIEW_OPT_BENCH,
    VIEW_OPT_WB,
    VIEW_OPT_EI,
    VIEW_OPT_EXPOSURE,
    VIEW_OPT_DEBAYER,
    VIEW_OPT_TONE_MAP,
    VIEW_OPT_GAMUT_MAP,
    VIEW_OPT_LUT,
    VIEW_OPT_LUT_TYPE,
    VIEW_OPT_LUT_SPACE,
    VIEW_OPT_PROFILE,
    VIEW_OPT_TABLES,
    VIEW_OPT_EXPORT,
    VIEW_OPT_CODEC,
    VIEW_OPT_ENC,
    VIEW_OPT_CROP,
    VIEW_OPT_LOOPS,
    VIEW_OPT_NO_LOOP,
    VIEW_OPT_AUDIO_DUMP,
    VIEW_OPT_AUDIO_MAP,
    VIEW_OPT_NO_PEAK,
    VIEW_OPT_ZOOM,
    VIEW_OPT_LIST_LUTS,
    VIEW_OPT_GRAIN,
    VIEW_OPT_DENOISE,
    VIEW_OPT_SOFTLIGHT,
    VIEW_OPT_VIBRANCE,
};

static int view_parse_args(struct view_state *st, int argc, char **argv,
                           const char **profile, const char **tables)
{
    static const struct option opts[] = {
        { "device",   required_argument, NULL, VIEW_OPT_DEVICE    },
        { "debug",    no_argument,       NULL, VIEW_OPT_DEBUG     },
        { "bench",    no_argument,       NULL, VIEW_OPT_BENCH     },
        { "wb",       required_argument, NULL, VIEW_OPT_WB        },
        { "ei",       required_argument, NULL, VIEW_OPT_EI        },
        { "exposure", required_argument, NULL, VIEW_OPT_EXPOSURE  },
        { "debayer",  required_argument, NULL, VIEW_OPT_DEBAYER   },
        { "tone-map", required_argument, NULL, VIEW_OPT_TONE_MAP  },
        { "gamut-map", required_argument, NULL, VIEW_OPT_GAMUT_MAP },
        { "lut",      required_argument, NULL, VIEW_OPT_LUT       },
        { "lut-type", required_argument, NULL, VIEW_OPT_LUT_TYPE  },
        { "lut-space", required_argument, NULL, VIEW_OPT_LUT_SPACE },
        { "list-luts", no_argument,       NULL, VIEW_OPT_LIST_LUTS },
        { "profile",  required_argument, NULL, VIEW_OPT_PROFILE   },
        { "tables",   required_argument, NULL, VIEW_OPT_TABLES    },
        { "export",   required_argument, NULL, VIEW_OPT_EXPORT    },
        { "codec",    required_argument, NULL, VIEW_OPT_CODEC     },
        { "enc",      required_argument, NULL, VIEW_OPT_ENC       },
        { "crop",     required_argument, NULL, VIEW_OPT_CROP      },
        { "loops",    required_argument, NULL, VIEW_OPT_LOOPS     },
        { "no-loop",  no_argument,       NULL, VIEW_OPT_NO_LOOP   },
        { "audio-dump", required_argument, NULL, VIEW_OPT_AUDIO_DUMP },
        { "audio-map", required_argument,  NULL, VIEW_OPT_AUDIO_MAP },
        { "no-peak-detect", no_argument,   NULL, VIEW_OPT_NO_PEAK   },
        { "zoom",     required_argument, NULL, VIEW_OPT_ZOOM      },
        { "grain",    required_argument, NULL, VIEW_OPT_GRAIN     },
        { "denoise",  required_argument, NULL, VIEW_OPT_DENOISE   },
        { "softlight", required_argument, NULL, VIEW_OPT_SOFTLIGHT },
        { "vibrance", required_argument, NULL, VIEW_OPT_VIBRANCE  },
        { "help",     no_argument,       NULL, 'h'                },
        { NULL,       0,                 NULL, 0                  },
    };
    static const char *const lut_types[] = { "native", "normalized" };
    char *end;
    int v, c, ret;

    while ((c = getopt_long(argc, argv, "vh", opts, NULL)) != -1) {
        switch (c) {
        case 'v':
            st->cfg.verbose = true;
            break;
        case 'h':
            view_usage(stdout);
            return 1;

        case VIEW_OPT_DEVICE:
            st->cfg.device = optarg;
            break;
        case VIEW_OPT_DEBUG:
            st->cfg.debug = true;
            break;
        case VIEW_OPT_BENCH:
            st->cfg.bench = true;
            break;
        case VIEW_OPT_NO_LOOP:
            st->cfg.loop = false;
            break;
        case VIEW_OPT_LOOPS:
            st->cfg.loops = strtoll(optarg, &end, 10);
            if (end == optarg || *end || st->cfg.loops < 0) {
                fprintf(stderr, "--loops: `%s` is not a pass count\n", optarg);
                return CRAW_ERROR(EINVAL);
            }
            break;
        case VIEW_OPT_AUDIO_DUMP:
            st->cfg.audio_dump = optarg;
            break;
        case VIEW_OPT_AUDIO_MAP:
            st->cfg.audio_map = optarg;
            break;
        case VIEW_OPT_ZOOM:
            if (!strcmp(optarg, "fit")) {
                st->view.fit = true;
            } else {
                st->view.zoom = strtof(optarg, &end);
                if (end == optarg || *end || !(st->view.zoom > 0.0f)) {
                    fprintf(stderr, "--zoom: expected fit or a scale, got `%s`\n",
                            optarg);
                    return CRAW_ERROR(EINVAL);
                }
                st->view.fit = false;
            }
            break;
        case VIEW_OPT_NO_PEAK:
            st->look.peak_detect = false;
            st->look.peak_detect_flag = 0;
            break;
        case VIEW_OPT_GRAIN:
            if ((ret = view_grain_parse(&st->grain, optarg)) < 0)
                return ret;
            break;
        case VIEW_OPT_DENOISE: {
            float v2 = strtof(optarg, &end);

            if (end == optarg || *end || v2 < 0.0f || v2 > 3.0f) {
                fprintf(stderr, "--denoise: expected a strength from 0 to 3, "
                                "got `%s`\n", optarg);
                return CRAW_ERROR(EINVAL);
            }
            /* Zero keeps it off and leaves the panel's own strength where
             * it was, so the section still opens on a sensible number. */
            st->denoise.on = v2 > 0.0f;
            st->denoise.on_flag = st->denoise.on ? 1 : 0;
            if (v2 > 0.0f)
                st->denoise.strength = v2;
            break;
        }
        case VIEW_OPT_SOFTLIGHT:
            if ((ret = view_adjust_parse_softlight(&st->adjustments, optarg)) < 0)
                return ret;
            break;
        case VIEW_OPT_VIBRANCE:
            if ((ret = view_adjust_parse_vibrance(&st->adjustments, optarg)) < 0)
                return ret;
            break;
        case VIEW_OPT_WB:
            if ((ret = view_parse_wb(&st->look, optarg)) < 0)
                return ret;
            break;
        case VIEW_OPT_EI:
            st->look.ei = strtof(optarg, &end);
            if (end == optarg || *end || !(st->look.ei > 0.0f)) {
                fprintf(stderr, "--ei must be a positive exposure index\n");
                return CRAW_ERROR(EINVAL);
            }
            break;
        case VIEW_OPT_EXPOSURE:
            st->look.exposure = strtof(optarg, &end);
            if (end == optarg || *end) {
                fprintf(stderr, "--exposure: `%s` is not a number\n", optarg);
                return CRAW_ERROR(EINVAL);
            }
            break;
        case VIEW_OPT_DEBAYER:
            if ((ret = view_lookup(view_debayer_names, CRAW_DEBAYER_NB,
                                   "--debayer", optarg, &v)) < 0)
                return ret;
            st->look.debayer = (enum CRAWDebayer)v;
            break;
        case VIEW_OPT_TONE_MAP:
            if ((ret = view_lookup(view_tone_map_names, view_nb_tone_maps,
                                   "--tone-map", optarg, &v)) < 0)
                return ret;
            st->look.tone_map = v;
            break;
        case VIEW_OPT_GAMUT_MAP:
            if ((ret = view_lookup(view_gamut_map_names, view_nb_gamut_maps,
                                   "--gamut-map", optarg, &v)) < 0)
                return ret;
            st->look.gamut_map = v;
            break;
        case VIEW_OPT_LUT:
            snprintf(st->look.lut_path, sizeof(st->look.lut_path), "%s", optarg);
            break;
        case VIEW_OPT_LUT_TYPE:
            if ((ret = view_lookup(lut_types, 2, "--lut-type", optarg, &v)) < 0)
                return ret;
            st->look.lut_type = v ? PL_LUT_NORMALIZED : PL_LUT_NATIVE;
            break;
        case VIEW_OPT_LUT_SPACE:
            if ((ret = view_lookup(view_lut_space_names, view_nb_lut_spaces,
                                   "--lut-space", optarg, &v)) < 0)
                return ret;
            st->look.lut_space = v;
            break;
        case VIEW_OPT_LIST_LUTS:
            view_list_luts();
            return 1;
        case VIEW_OPT_PROFILE:
            *profile = optarg;
            break;
        case VIEW_OPT_TABLES:
            *tables = optarg;
            break;

        case VIEW_OPT_EXPORT:
            /* Two arguments, which getopt cannot express: the first is the
             * option's own, the file name is taken from the tail. */
            if (optind >= argc) {
                fprintf(stderr, "--export takes FRAME OUT\n");
                return CRAW_ERROR(EINVAL);
            }
            st->cfg.export_frame = strtoll(optarg, &end, 10);
            if (end == optarg || *end || st->cfg.export_frame < 0) {
                fprintf(stderr, "--export: `%s` is not a frame number\n", optarg);
                return CRAW_ERROR(EINVAL);
            }
            snprintf(st->cfg.export_path, sizeof(st->cfg.export_path), "%s",
                     argv[optind]);
            optind += 1;
            st->cfg.headless = true;
            break;
        case VIEW_OPT_CODEC:
            st->cfg.export_codec = optarg;
            break;
        case VIEW_OPT_ENC:
            st->cfg.export_enc = optarg;
            break;
        case VIEW_OPT_CROP:
            if ((ret = view_parse_crop(&st->crop, optarg)) < 0)
                return ret;
            break;

        default:
            view_usage(stderr);
            return CRAW_ERROR(EINVAL);
        }
    }

    if (optind >= argc) {
        fprintf(stderr, "no input file\n");
        view_usage(stderr);
        return CRAW_ERROR(EINVAL);
    }
    st->cfg.input = argv[optind++];
    if (optind < argc) {
        fprintf(stderr, "unexpected argument `%s`\n", argv[optind]);
        return CRAW_ERROR(EINVAL);
    }

    return 0;
}

/* ------------------------------------------------------------ the profile */

static int view_profile_setup(struct view_state *st, const char *profile,
                              const char *tables)
{
    int ret;

    craw_profile_init(&st->base);

    /* An already developed clip has no raw to balance, so none of this
     * applies and its absence must not be reported as a problem. */
    if (!st->dec.raw)
        return 0;

    if (profile) {
        ret = craw_tool_profile_apply_builtin(&st->base, profile);
        if (ret < 0) {
            fprintf(stderr, "--profile: no built-in table set called `%s` (have: %s)\n",
                    profile, craw_tool_profile_builtin_list());
            return ret;
        }
    } else {
        const char *name = craw_tool_profile_detect(st->dec.model);

        if (name) {
            ret = craw_tool_profile_apply_builtin(&st->base, name);
            if (ret < 0)
                return ret;
            view_log(st, "recognized `%s` from the container metadata\n", name);
        } else if (!tables) {
            fprintf(stderr, "warning: no camera recognized in the container metadata, "
                            "carrying on without per-CCT tables\n");
        }
    }

    if (tables) {
        ret = craw_tool_profile_load_tables(&st->base, tables);
        if (ret < 0)
            return ret;
        view_log(st, "loaded %u wb and %u matrix knots from %s\n",
                 st->base.nb_wb_knots, st->base.nb_matrix_knots, tables);
    }

    if (st->look.wb != VIEW_WB_AS_SHOT && !st->base.nb_wb_knots) {
        fprintf(stderr, "warning: %s white balance needs the camera's per-CCT table "
                        "and none was found; staying as shot\n",
                st->look.wb == VIEW_WB_AUTO ? "auto" : "manual");
        st->look.wb = VIEW_WB_AS_SHOT;
    }

    return 0;
}

/* ----------------------------------------------------------------- main */

/* Window points to drawable pixels, which is what the picture area is in.
 * SDL3 reports pointer positions as floats, in window points as before. */
static void view_mouse_pos(const struct view_state *st, float x, float y,
                           int *px, int *py)
{
    float sx = st->win_w > 0 ? (float)st->gpu.ui_w / (float)st->win_w : 1.0f;
    float sy = st->win_h > 0 ? (float)st->gpu.ui_h / (float)st->win_h : 1.0f;

    *px = (int)(x * sx);
    *py = (int)(y * sy);
}

static bool view_over_picture(const struct view_state *st, float x, float y)
{
    pl_rect2df area;
    int px, py;

    /* The film simulation menu hangs over the picture, so while it is open
     * the wheel scrolls its columns and a click picks a rendering rather
     * than zooming and dragging what is behind it. */
    if (view_ui_menu_covers(st, x, y))
        return false;

    view_mouse_pos(st, x, y, &px, &py);
    view_picture_area(st, &area);

    return (float)px < area.x1;
}

/* One second of the clip, in frames: what Shift with Left or Right moves. */
static int64_t view_seek_second(const struct view_state *st)
{
    double fps = av_q2d(st->dec.frame_rate);

    return fps > 0.0 ? (int64_t)(fps + 0.5) : 25;
}

/* Seeks relative to the picture on screen without touching the transport:
 * a clip that was playing keeps playing from where it lands. */
static void view_seek_by(struct view_state *st, int64_t frames)
{
    int64_t last = CRAW_MAX(st->dec.nb_frames - 1, 0);

    st->seek_to = CRAW_CLAMP(st->index + frames, 0, last);
    st->need_frame = true;
}

/* Space plays; Up and Down walk the panel and Left and Right adjust the row
 * they land on; Shift with Left or Right seeks a second; Page Up and Page
 * Down zoom and Shift with them goes to the ends of the clip; Control with
 * the arrows pans, which only does anything while the picture is larger
 * than its area; c turns crop mode on and off. A key that toggles or seeks
 * acts once per press, the ones that step act on every repeat. */
static void view_handle_key(struct view_state *st, const SDL_Event *e, bool *running)
{
    bool shift = (e->key.mod & SDL_KMOD_SHIFT) != 0;
    bool ctrl = (e->key.mod & SDL_KMOD_CTRL) != 0;
    bool repeat = e->key.repeat;

    view_log(st, "key: %s (scancode %d, sym 0x%x, mod 0x%x%s)\n",
             SDL_GetKeyName(e->key.key), (int)e->key.scancode,
             (unsigned)e->key.key, (unsigned)e->key.mod,
             repeat ? ", repeat" : "");

    /* An open film simulation menu owns the arrows, Enter and Escape; it
     * says so itself, and says nothing while it is closed. */
    if (view_ui_menu_key(st, (int)e->key.key))
        return;

    switch (e->key.key) {
    case SDLK_SPACE:
        if (!repeat)
            st->playing = !st->playing;
        break;

    case SDLK_UP:
        if (ctrl)
            view_pan_fraction(st, 0.0f, -0.125f);
        else if (st->ui_rows > 0)
            st->focus = (st->focus - 1 + st->ui_rows) % st->ui_rows;
        break;
    case SDLK_DOWN:
        if (ctrl)
            view_pan_fraction(st, 0.0f, 0.125f);
        else if (st->ui_rows > 0)
            st->focus = (st->focus + 1) % st->ui_rows;
        break;

    case SDLK_LEFT:
        if (ctrl)
            view_pan_fraction(st, -0.125f, 0.0f);
        else if (shift)
            view_seek_by(st, -view_seek_second(st));
        else
            st->adjust = -1;
        break;
    case SDLK_RIGHT:
        if (ctrl)
            view_pan_fraction(st, 0.125f, 0.0f);
        else if (shift)
            view_seek_by(st, view_seek_second(st));
        else
            st->adjust = 1;
        break;

    case SDLK_PAGEUP:
        if (shift) {
            if (!repeat) {
                st->playing = false;
                st->seek_to = 0;
                st->need_frame = true;
            }
        } else {
            view_zoom_by(st, 1.0f, 0, 0, false);
        }
        break;
    case SDLK_PAGEDOWN:
        if (shift) {
            if (!repeat) {
                st->playing = false;
                st->seek_to = CRAW_MAX(st->dec.nb_frames - 1, 0);
                st->need_frame = true;
            }
        } else {
            view_zoom_by(st, -1.0f, 0, 0, false);
        }
        break;

    case SDLK_HOME:
        if (!repeat) {
            st->playing = false;
            st->seek_to = 0;
            st->need_frame = true;
        }
        break;
    case SDLK_END:
        if (!repeat) {
            st->playing = false;
            st->seek_to = CRAW_MAX(st->dec.nb_frames - 1, 0);
            st->need_frame = true;
        }
        break;

    case SDLK_RETURN:
    case SDLK_KP_ENTER:
        /* Enter only means anything on the row that owns the menu. */
        if (!repeat && st->focus == st->menu.row)
            view_ui_menu_open(st);
        break;

    case SDLK_TAB:
    case SDLK_0:
        if (!repeat)
            view_zoom_reset(st, false);
        break;
    case SDLK_1:
        if (!repeat)
            view_zoom_reset(st, true);
        break;

    case SDLK_C:
        /* Crop mode: a left drag draws the rectangle instead of panning.
         * The rectangle itself outlives the mode. */
        if (!repeat) {
            st->crop.on = !st->crop.on;
            st->crop.on_flag = st->crop.on;
        }
        break;

    case SDLK_M:
        if (!repeat) {
            st->mute = !st->mute;
            st->mute_flag = st->mute;
        }
        break;

    case SDLK_ESCAPE:
    case SDLK_Q:
        *running = false;
        break;
    default:
        break;
    }
}

static int view_run_window(struct view_state *st)
{
    SDL_Window *win = NULL;
    bool running = true;
    int64_t fps_t0, iter_t0;
    int fps_frames = 0, proc_frames = 0;
    int w = VIEW_WINDOW_W, h = VIEW_WINDOW_H;
    int ret;

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return CRAW_ERROR(ENOTSUP);
    }
    st->sdl_init = true;

    /* Nothing here asks for a backend: SDL3 picks Wayland on a Wayland
     * session by itself, and the window is a native surface rather than an
     * XWayland one. */
    view_log(st, "video driver: %s\n", SDL_GetCurrentVideoDriver());

    /* HIGH_PIXEL_DENSITY is what makes the drawable the display's own
     * resolution on a scaled output; the picture is then rendered at full
     * density and the panel is laid out in pixels, with pointer positions
     * scaled from window points. */
    win = SDL_CreateWindow("craw-view", w, h,
                           SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE |
                           SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (!win) {
        fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError());
        return CRAW_ERROR(ENOTSUP);
    }
    st->win = win;

    ret = view_gpu_open(&st->gpu, &st->cfg, win);
    if (ret < 0)
        return ret;

    SDL_GetWindowSizeInPixels(win, &w, &h);
    SDL_GetWindowSize(win, &st->win_w, &st->win_h);
    view_log(st, "window %dx%d points, %dx%d pixels\n", st->win_w, st->win_h, w, h);
    if (!pl_swapchain_resize(st->gpu.sw, &w, &h)) {
        fprintf(stderr, "cannot size the swapchain to %dx%d\n", w, h);
        return CRAW_ERROR(EIO);
    }

    ret = view_decoder_open(&st->dec, st->cfg.input, &st->gpu, st->cfg.verbose);
    if (ret < 0)
        return ret;

    /* --bench measures the picture pipeline; an audio track would pace the
     * loop at the clip's own rate and the draining at each loop point would
     * dominate a short clip's average. */
    if (!st->cfg.bench) {
        ret = view_audio_open(&st->audio, st->cfg.input, st->cfg.audio_map,
                              st->cfg.verbose);
        if (ret < 0)
            return ret;
    }

    if (st->cfg.audio_dump) {
        ret = view_audio_dump_open(&st->audio, st->cfg.audio_dump);
        if (ret < 0)
            return ret;
    }

    ret = view_profile_setup(st, st->cfg.profile, st->cfg.tables);
    if (ret < 0)
        return ret;

    ret = view_ui_open(st, win);
    if (ret < 0)
        return ret;

    if (view_lut_startup(st) < 0)
        fprintf(stderr, "--lut %s: %s\n", st->look.lut_path, st->lut_error);

    fps_t0 = av_gettime_relative();
    iter_t0 = fps_t0;
    ret = 0;

    while (running) {
        bool was_playing = st->playing;
        bool stepped = false;
        SDL_Event e;
        int64_t now;

        view_ui_input_begin();
        while (SDL_PollEvent(&e)) {
            bool to_nuklear = true;

            switch (e.type) {
            case SDL_EVENT_QUIT:
                running = false;
                break;
            case SDL_EVENT_KEY_DOWN:
                /* Let the text field have the keys while it is being typed
                 * into; nuklear tells us so through its own input state.
                 * Everything else reaches the picture whatever the pointer
                 * happens to be hovering. */
                if (!view_ui_wants_keys(st))
                    view_handle_key(st, &e, &running);
                break;

            /* SDL3 puts the pointer position in the wheel event itself, so
             * zooming about the cursor no longer has to ask for the mouse
             * state and risk a position from after the notch. */
            case SDL_EVENT_MOUSE_WHEEL:
                if (view_over_picture(st, e.wheel.mouse_x, e.wheel.mouse_y)) {
                    int px, py;

                    view_mouse_pos(st, e.wheel.mouse_x, e.wheel.mouse_y, &px, &py);
                    view_zoom_by(st, e.wheel.y, px, py, true);
                    to_nuklear = false;
                }
                break;

            case SDL_EVENT_MOUSE_BUTTON_DOWN:
                /* A click anywhere outside the open film simulation menu
                 * dismisses it and goes no further: that first click is
                 * the dismissal, not a pan, a crop or a panel widget. The
                 * columns themselves, and the row that toggles them, are
                 * left to nuklear. */
                if (e.button.button == SDL_BUTTON_LEFT && st->menu.open &&
                    !view_ui_menu_covers(st, e.button.x, e.button.y)) {
                    view_ui_menu_close(st);
                    to_nuklear = false;
                    break;
                }
                if (e.button.button == SDL_BUTTON_LEFT &&
                    view_over_picture(st, e.button.x, e.button.y)) {
                    /* In crop mode the drag draws a rectangle; otherwise it
                     * pans, which is what it has always done. */
                    if (st->crop.on) {
                        int px, py;

                        view_mouse_pos(st, e.button.x, e.button.y, &px, &py);
                        view_crop_begin(st, px, py);
                    } else {
                        st->view.dragging = true;
                        st->view.drag_x = e.button.x;
                        st->view.drag_y = e.button.y;
                    }
                    to_nuklear = false;
                }
                break;
            case SDL_EVENT_MOUSE_BUTTON_UP:
                if (e.button.button == SDL_BUTTON_LEFT &&
                    (st->view.dragging || st->crop.dragging)) {
                    st->view.dragging = false;
                    view_crop_end(st);
                    to_nuklear = false;
                }
                break;
            case SDL_EVENT_MOUSE_MOTION:
                if (st->crop.dragging) {
                    int px, py;

                    view_mouse_pos(st, e.motion.x, e.motion.y, &px, &py);
                    view_crop_drag(st, px, py);
                    to_nuklear = false;
                } else if (st->view.dragging) {
                    int ax, ay, bx, by;

                    view_mouse_pos(st, st->view.drag_x, st->view.drag_y, &ax, &ay);
                    view_mouse_pos(st, e.motion.x, e.motion.y, &bx, &by);
                    view_pan_pixels(st, bx - ax, by - ay);
                    st->view.drag_x = e.motion.x;
                    st->view.drag_y = e.motion.y;
                    to_nuklear = false;
                }
                break;

            /* Both arrive on a resize, and under a fractional scale the
             * pixel size can change on its own; taking either and reading
             * both sizes back covers the two cases with one path. */
            case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
            case SDL_EVENT_WINDOW_RESIZED:
                SDL_GetWindowSizeInPixels(win, &w, &h);
                SDL_GetWindowSize(win, &st->win_w, &st->win_h);
                if (!pl_swapchain_resize(st->gpu.sw, &w, &h)) {
                    running = false;
                } else {
                    view_log(st, "resized to %dx%d points, %dx%d pixels\n",
                             st->win_w, st->win_h, w, h);
                    view_ui_resize(st, w, h);
                }
                break;
            default:
                break;
            }

            if (to_nuklear)
                view_ui_event(&e);
        }
        view_ui_input_end();

        /* Keeps the audio ring topped up whatever the picture is doing. */
        if (!st->exp.active && view_audio_pump(&st->audio) < 0)
            break;

        if (st->exp.active) {
            if (view_export_step(st) < 0)
                st->exp.active = false;
        } else if (st->need_frame || !st->have_frame ||
                   (st->playing && view_frame_due(st))) {
            int64_t step_t0 = av_gettime_relative();

            do {
                ret = view_step_frame(st);
            } while (ret == 2);     /* late frames, dropped to catch up */

            /* A starved step keeps its request: right after a seek the
             * reader has just been flushed and its first packet lands a few
             * milliseconds later, and while paused nothing else would ask
             * again. */
            if (ret != 3)
                st->need_frame = false;

            if (ret < 0)
                break;

            if (ret == 3) {
                /* The medium has not caught up. Leave the picture where it
                 * is and try again next time round; this is not a drop. */
            } else if (ret > 0) {
                /* End of the clip. The last picture stays up while the
                 * audio tail plays out; only then does it loop or stop. */
                if (view_audio_drained(&st->audio)) {
                    st->loops_done++;
                    if (st->cfg.loops && st->loops_done >= st->cfg.loops) {
                        st->playing = false;
                        running = false;
                    } else if (st->cfg.loop) {
                        st->seek_to = 0;
                        st->need_frame = true;
                    } else {
                        st->playing = false;
                    }
                }
            } else {
                double cost = (double)(av_gettime_relative() - step_t0) / 1.0e6;

                stepped = true;
                proc_frames++;
                /* The first picture of a run pays for everything that is
                 * set up once, so it is no guide to what the next one will
                 * cost. */
                if (st->sync.run_frames > 1)
                    st->step_cost = st->step_cost > 0.0
                                  ? st->step_cost * 0.9 + cost * 0.1 : cost;

                /* One step per iteration is not enough to recover from a
                 * stall when the clip runs near the display's own rate, so
                 * a second one is taken while the picture is clearly
                 * behind - clearly, because a clip at exactly the refresh
                 * rate sits a fraction of a frame behind by nature and
                 * trading frames away for that would be a bad bargain. */
                while (st->playing && !st->cfg.bench &&
                       view_clock(st) - st->pts >= 1.5 * view_frame_duration(st)) {
                    st->sync.dropped++;
                    do {
                        ret = view_step_frame(st);
                    } while (ret == 2);
                    if (ret)
                        break;
                }
            }
        }

        view_ui_build(st);

        /* A grade change while paused replays the picture on screen; while
         * playing the next frame simply uses the current look, and so does
         * a decode that is already pending. */
        if (!st->playing && !st->exp.active && st->have_frame &&
            !st->need_frame && st->seek_to < 0) {
            struct view_pipeline_look look = view_pipeline_look(st);

            if (!view_pipeline_look_equal(&look, &st->processed)) {
                ret = view_replay_frame(st);
                if (ret < 0)
                    break;
            }
        }

        /* The transport drives the audio, never the other way round. */
        if (st->playing != was_playing) {
            if (st->playing)
                view_wall_restart(st, st->pts);
            else if (!st->audio.available)
                st->wall_pts += (double)(av_gettime_relative() - st->wall_t0) / 1.0e6;
        }

        view_audio_set_volume(&st->audio, st->volume, st->mute);

        if (!st->exp.active && st->audio.available) {
            if (st->playing && st->warm && !st->audio.playing)
                view_audio_start(st);
            else if ((!st->playing || !st->warm) && st->audio.playing)
                view_audio_pause(&st->audio);
        }

        if (stepped) {
            double clock = view_clock(st);
            double diff = clock - st->pts;
            double mag = fabs(diff);

            st->sync.diff = diff;
            st->sync.presented++;
            st->sync.max_abs = CRAW_MAX(st->sync.max_abs, mag);
            /* The first frames of a run are ahead of the audio by up to one
             * device buffer, because nothing has been heard yet, and the
             * very first of them also pay for compiling shaders. */
            if (st->sync.run_frames > 10)
                st->sync.max_abs_settled = CRAW_MAX(st->sync.max_abs_settled, mag);

            view_log(st, "frame %5" PRId64 " pts %7.3f s  clock %7.3f s  "
                         "a/v %+7.2f ms  peak %6.0f / %6.0f nits\n",
                     st->index, st->pts, clock, diff * 1000.0,
                     st->gpu.measured_peak, view_peak_ceiling(st));
        }

        ret = view_ui_render(st);
        if (ret < 0)
            break;

        ret = view_gpu_display(&st->gpu, st);
        view_ui_rotate(&st->gpu);
        if (ret == CRAW_ERROR(EAGAIN)) {
            SDL_Delay(10);
            continue;
        }
        if (ret < 0)
            break;

        if (!st->view.cx && !st->view.cy && st->gpu.width) {
            st->view.cx = (float)st->gpu.width * 0.5f;
            st->view.cy = (float)st->gpu.height * 0.5f;
        }

        if (!st->warm && st->have_frame) {
            /* A picture has made it all the way to the screen, so every
             * shader either side of it is compiled. Start the clock here. */
            st->warm = true;
            view_wall_restart(st, st->pts);
            st->sync.run_frames = 0;
        }

        fps_frames++;
        now = av_gettime_relative();
        {
            double period = (double)(now - iter_t0) / 1.0e6;
            st->loop_period = st->loop_period > 0.0
                            ? st->loop_period * 0.9 + period * 0.1 : period;
            iter_t0 = now;
        }
        if (now - fps_t0 >= 500000) {
            st->fps = (double)fps_frames * 1.0e6 / (double)(now - fps_t0);
            st->proc_fps = (double)proc_frames * 1.0e6 / (double)(now - fps_t0);
            fps_frames = proc_frames = 0;
            fps_t0 = now;
            view_log(st, "display %.1f fps, pipeline %.1f fps, decode %.1f ms, "
                         "libcraw %.2f ms\n",
                     st->fps, st->proc_fps, st->decode_ms, st->gpu.last_gpu_ms);

            if (st->timing.frames) {
                struct view_timing *tm = &st->timing;
                double n = (double)tm->frames;

                view_log(st, "  per frame: read %6.2f  decode %6.2f  submit %5.2f  "
                             "fence %6.2f  map %5.2f  finish %5.2f  ui %5.2f  "
                             "render %5.2f  present %6.2f ms; queue %.1f pkts, "
                             "%" PRId64 " starved\n",
                         tm->read / n, tm->decode / n, tm->submit / n,
                         tm->fence / n, tm->map / n, tm->finish / n,
                         tm->ui / n, tm->render / n, tm->present / n,
                         view_decoder_queued(&st->dec), tm->starved);
                *tm = (struct view_timing) { 0 };
            }
        }
    }

    if (st->cfg.verbose || st->cfg.loops)
        fprintf(stderr, "%" PRId64 " passes: %" PRId64 " frames presented, "
                        "%" PRId64 " dropped, max |a/v| %.2f ms (%.2f ms once "
                        "settled), %" PRIu64 " audio underruns (%" PRIu64
                        " sample frames of silence)\n",
                st->loops_done, st->sync.presented, st->sync.dropped,
                st->sync.max_abs * 1000.0, st->sync.max_abs_settled * 1000.0,
                st->audio.underruns, st->audio.underrun_frames);

    return ret > 0 ? 0 : ret;
}

/* Teardown in one place, and in the only order that works: the UI and
 * everything on the device first, then the VkSurfaceKHR (inside
 * view_gpu_close), and only then the window it belongs to. */
static void view_close(struct view_state *st)
{
    free(st->exp.rgba);
    pl_lut_free(&st->lut);
    av_frame_free(&st->frame);
    craw_awb_smoother_free(&st->awb.smoother);

    view_ui_close(st);
    view_audio_close(&st->audio);
    view_decoder_close(&st->dec);
    /* The grain hooks hold shader objects, so they go before the pl_gpu. */
    view_grain_close(&st->grain);
    view_adjust_close(&st->adjustments);
    view_gpu_close(&st->gpu);

    if (st->win)
        SDL_DestroyWindow(st->win);
    if (st->sdl_init)
        SDL_Quit();
}

/* Where a headless export's wall time goes, for -v: the device and the
 * decoder are paid for once whatever is exported, the rest is the frame. */
static int64_t view_phase(const struct view_state *st, int64_t t0, const char *what)
{
    int64_t now = av_gettime_relative();

    view_log(st, "%s: %.0f ms\n", what, (double)(now - t0) / 1000.0);

    return now;
}

static int view_run_headless(struct view_state *st)
{
    int64_t t0 = av_gettime_relative();
    int ret;

    ret = view_gpu_open(&st->gpu, &st->cfg, NULL);
    if (ret < 0)
        return ret;
    t0 = view_phase(st, t0, "device and pipeline");

    ret = view_decoder_open(&st->dec, st->cfg.input, &st->gpu, st->cfg.verbose);
    if (ret < 0)
        return ret;
    t0 = view_phase(st, t0, "decoder");

    ret = view_profile_setup(st, st->cfg.profile, st->cfg.tables);
    if (ret < 0)
        return ret;

    ret = view_lut_startup(st);
    if (ret < 0) {
        fprintf(stderr, "--lut %s: %s\n", st->look.lut_path, st->lut_error);
        return ret;
    }
    t0 = view_phase(st, t0, "profile and LUT");

    view_export_begin(st, st->cfg.export_frame, st->cfg.export_path,
                      st->cfg.export_codec);

    while (st->exp.active) {
        ret = view_export_step(st);
        if (ret < 0)
            return ret;
    }

    view_phase(st, t0, "seek, decode, render and encode");
    fprintf(stderr, "%s\n", st->exp.status);

    return 0;
}

int main(int argc, char **argv)
{
    static struct view_state st;
    const char *profile = NULL, *tables = NULL;
    int ret;

    st.look = (struct view_look) {
        .wb        = VIEW_WB_AS_SHOT,
        .manual    = { .cct = 5500.0f, .tint = 0.0f },
        .debayer   = CRAW_DEBAYER_RCD,
        .tone_map  = 0,
        .gamut_map = 0,
        .lut_type  = PL_LUT_NATIVE,
        .film_sim  = -1,
        .peak_detect = true,
        .peak_detect_flag = 1,
    };
    st.menu.kb = -1;
    for (int i = 0; i < VIEW_MENU_COLS; i++)
        st.menu.sel[i] = -1;
    st.seek_to = -1;
    st.seek_target = -1;
    st.playing = true;
    st.cfg.loop = true;
    st.view.fit = true;
    st.volume = 1.0f;
    /* Before the arguments, so that --grain and --denoise override the
     * defaults. The strength is set even though the stage is off, so that
     * the panel's switch turns it on at something sensible. */
    st.denoise.strength = 1.0f;
    view_grain_defaults(&st.grain);
    view_adjust_defaults(&st.adjustments);

#if CRAW_VIEW_HAVE_LUTS
    /* Reads the pack's tables, which is all pointer arithmetic over the
     * embedded array; nothing is decompressed until a rendering is picked.
     * It has to happen before the arguments, because --list-luts and
     * --lut NAME both walk those tables. */
    if (view_luts_init() < 0)
        fprintf(stderr, "the embedded film simulation pack did not load\n");
#endif

    ret = view_parse_args(&st, argc, argv, &profile, &tables);
    if (ret)
        return ret > 0 ? 0 : 1;

    st.cfg.profile = profile;
    st.cfg.tables = tables;

    craw_awb_params_init(&st.awb.params);

    ret = craw_awb_smoother_create(NULL, &st.awb.smoother);
    if (ret < 0) {
        fprintf(stderr, "craw_awb_smoother_create failed: %d\n", ret);
        return 1;
    }

    st.frame = av_frame_alloc();
    if (!st.frame)
        return 1;

    ret = st.cfg.headless ? view_run_headless(&st) : view_run_window(&st);

    view_close(&st);

    return ret < 0 ? 1 : 0;
}
