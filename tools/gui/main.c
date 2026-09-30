/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* cineview: open a camera raw clip, decode it on the GPU, run every frame
 * through libcraw on the same device and show the result through
 * libplacebo, with a nuklear panel for the grade and a still exporter.
 *
 * cineraw is the batch tool and the reference for the profile,
 * exposure and white balance semantics; this is the interactive one. The
 * options it shares with cineraw mean exactly the same thing. */

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
#include "colour.h"
#include "../profiles.h"

/* The embedded film simulation collection is optional; without it --lut
 * only takes a file and --list-luts says there is nothing to list. */
#ifndef CINEVIEW_HAVE_LUTS
#define CINEVIEW_HAVE_LUTS 0
#endif

#if CINEVIEW_HAVE_LUTS
#include "luts.h"
#endif

#define VIEW_WINDOW_W 1600
#define VIEW_WINDOW_H 900

/* How long the window may go without being drawn when nothing on it would
 * change, and how long the loop is ever allowed to sleep at a stretch. The
 * first keeps the panel's frame rate averages moving while paused; the
 * second keeps the audio ring topped up and the next picture's due time
 * from being overshot. */
#define VIEW_IDLE_PRESENT 200000        /* microseconds */
#define VIEW_IDLE_WAIT_MS 4

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

    /* The hand-off space, so the bound is over the values the display path
     * really sees. The gamut compression that follows it never raises the
     * largest channel - it only moves the others towards it - so the row
     * sums below still bound the output. */
    if (craw_colorspace_from_xyz_d65(CRAW_CS_ACES_AP1_LINEAR, out) < 0)
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

/* What the highlight reconstruction adds on top of the balanced clip,
 * DESIGN.md 18.2: an area blown in every channel is rebuilt to the plateau
 * where the channel with the largest gain clipped, which is that gain over
 * the smallest one. A strongly coloured blown area can be rebuilt higher
 * still; peak detection measures that, and this is only the ceiling it
 * starts from. */
static float view_highlight_headroom(const struct view_state *st, const CRAWColorParams *cp)
{
    float lo, hi;

    if (!st->look.highlights)
        return 1.0f;

    lo = CRAW_MIN(CRAW_MIN(cp->wb[0], cp->wb[1]), cp->wb[2]);
    hi = CRAW_MAX(CRAW_MAX(cp->wb[0], cp->wb[1]), cp->wb[2]);

    return lo > 0.0f ? hi / lo : 1.0f;
}

/* The Exposure section's switch, applied. Bypassed means a bias of zero at
 * the index the camera shot at - which is what rates a camera original at
 * 1.0 - while the panel keeps both numbers, so switching the section back
 * on returns to the exposure that was set rather than to a default. */
float view_exposure_bias(const struct view_state *st)
{
    return st->sections.exposure ? st->look.exposure : 0.0f;
}

float view_exposure_ei(const struct view_state *st)
{
    return st->sections.exposure ? st->look.ei : 0.0f;
}

/* The Look and Adjustments switches, with the one thing that overrides
 * them: a scene-referred export carries no display-referred operator at
 * all, so while one is selected all three sections are off whatever their
 * switches say (and the panel greys them out to explain it). */
bool view_look_active(const struct view_state *st)
{
    return st->sections.look && !view_export_is_scene(st);
}

bool view_adjust_active(const struct view_state *st)
{
    return st->sections.adjust && !view_export_is_scene(st);
}

/* The exposure model the raw path hands to libcraw, as a plain linear gain
 * for the non-raw path's shader: the bias in stops times the working index
 * over the as-shot one. */
float view_video_exposure(const struct view_state *st)
{
    float ei = view_exposure_ei(st);
    float gain = exp2f(view_exposure_bias(st));

    if (ei > 0.0f && st->dec.meta_as_shot_ei > 0.0f)
        gain *= ei / st->dec.meta_as_shot_ei;

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

    return VIEW_REF_WHITE_NITS * cp.gain * exp2f(view_exposure_bias(st))
         * view_matrix_headroom(&cp) * view_highlight_headroom(st, &cp);
}

bool view_export_is_scene(const struct view_state *st)
{
    int c = st->look.video_preset.colour;

    return c >= 0 && c < VIEW_NB_COLOURS && view_colours[c].scene;
}

enum CRAWColorSpace view_export_craw_space(const struct view_state *st)
{
    switch (st->look.video_preset.colour) {
    case VIEW_COLOUR_ACESCG: return CRAW_CS_ACES_AP1_LINEAR;
    case VIEW_COLOUR_ACES:   return CRAW_CS_ACES_AP0_LINEAR;
    default:                 return CRAW_CS_XYZ_D65;
    }
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

float view_chroma_strength(const struct view_state *st)
{
    if (!st->denoise.on || !(st->denoise.chroma > 0.0f))
        return 0.0f;

    return CRAW_MIN(st->denoise.chroma, 1.0f);
}

/* The temporal average needs the model as the spatial filter does: it is
 * the model that says which differences from the last frame are noise. */
float view_temporal_strength(const struct view_state *st)
{
    if (!st->denoise.on || !(st->denoise.temporal > 0.0f))
        return 0.0f;
    if (!view_denoise_model(st, NULL, NULL))
        return 0.0f;

    return CRAW_MIN(st->denoise.temporal, 0.95f);
}

/* The warp runs when the switch is on and the picture on screen carried a
 * record. A clip whose frames have none - every ProRes RAW clip that is not
 * from this camera - can never turn it on, which is why the panel's row
 * greys out rather than lying. */
bool view_lens_active(const struct view_state *st)
{
    return st->dec.raw && st->lens.on && st->lens.record.valid;
}

/* The same question plus the panel's fill / keep all row. Keeping
 * everything is a property of the warp, so it cannot be on while the warp
 * is off, and the row says so by reading "fill" whenever it cannot act. */
bool view_lens_keep_all(const struct view_state *st)
{
    return view_lens_active(st) && st->lens.keep;
}

/* The full count wherever the picture is looked at or kept: paused, and
 * on every frame an export develops, still or video, since an export never
 * plays. While playing, what fits the frame time. */
int view_sharpen_iterations(const struct view_state *st)
{
    const struct view_sharpen *sh = &st->sharpen;

    if (!st->dec.raw || !sh->on || !(sh->radius > 0.0f) || sh->iterations <= 0)
        return 0;
    if (st->playing && !st->exp.active && !st->video.active)
        return CRAW_MIN(sh->iterations, VIEW_SHARPEN_PLAY_ITERATIONS);
    return CRAW_MIN(sh->iterations, CRAW_SHARPEN_MAX_ITERATIONS);
}

/* Without a noise model the mask has nothing to measure contrast against,
 * and craw_pipeline_record() refuses one rather than guess, so on such a
 * camera the stage sharpens everything and the panel greys the row. */
float view_sharpen_contrast(const struct view_state *st)
{
    if (!view_sharpen_iterations(st) || !view_denoise_model(st, NULL, NULL))
        return 0.0f;

    return CRAW_MAX(st->sharpen.contrast, 0.0f);
}

void view_frame_params(const struct view_state *st, CRAWFrameParams *fp)
{
    craw_frame_params_init(fp);

    fp->profile = &st->prof;
    fp->exposure_stops = view_exposure_bias(st);
    /* The model itself is left to the profile: noise_override stays zero,
     * so craw_pipeline_record() evaluates the table at the frame's own
     * as-shot index. */
    fp->denoise_strength = view_denoise_strength(st);
    fp->denoise_chroma_strength = view_chroma_strength(st);
    /* Whether this is the next frame, a replay or a new start is the
     * caller's to say (view_process_frame()); the direct export's own
     * pipeline only ever gets next frames, from its first on. */
    fp->denoise_temporal_strength = view_temporal_strength(st);
    /* The pipeline is always built with the stage (gpu.c), so the count is
     * all it takes; zero is not dispatched. */
    fp->false_color_passes = (uint32_t)CRAW_CLAMP(st->look.false_color, 0,
                                                  CRAW_FALSE_COLOR_MAX_PASSES);
    /* Built into the pipeline the same way; unflagged, the pass is not
     * dispatched and the frame is clipped as it always was. */
    if (st->look.highlights)
        fp->flags |= CRAW_FRAME_FLAG_HIGHLIGHT_RECONSTRUCTION;
    fp->defringe_strength = CRAW_CLAMP(st->look.defringe, 0.0f, 1.0f);
    /* By value: the record is this frame's own and the next frame's will be
     * different, so the parameters must not point at anything the panel
     * owns. An invalid one is the documented "leave this frame alone". */
    if (view_lens_active(st))
        fp->lens = st->lens.record;
    if (view_lens_keep_all(st))
        fp->flags |= CRAW_FRAME_FLAG_LENS_KEEP_ALL;
    /* Built into the pipeline like the rest; no iterations is not
     * dispatched. */
    fp->sharpen_iterations = (uint32_t)view_sharpen_iterations(st);
    if (fp->sharpen_iterations) {
        fp->sharpen_radius = CRAW_CLAMP(st->sharpen.radius, 0.1f, CRAW_SHARPEN_MAX_RADIUS);
        fp->sharpen_contrast = view_sharpen_contrast(st);
    }

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
        .ei         = view_exposure_ei(st),
        .exposure   = view_exposure_bias(st),
        .ambience   = st->awb.params.ambience,
        .debayer    = st->look.debayer,
        .false_color = st->look.false_color,
        .highlights = st->look.highlights != 0,
        .defringe   = CRAW_CLAMP(st->look.defringe, 0.0f, 1.0f),
        .chroma     = view_chroma_strength(st),
        .denoise    = view_denoise_strength(st),
        .temporal   = view_temporal_strength(st),
        .lens       = view_lens_active(st),
        .lens_keep  = view_lens_keep_all(st),
        .sharpen_iterations = view_sharpen_iterations(st),
        .sharpen_radius = view_sharpen_iterations(st) ? st->sharpen.radius : 0.0f,
        .sharpen_contrast = view_sharpen_contrast(st),
    };
}

bool view_wb_applied(const struct view_state *st, CRAWIlluminant *out)
{
    const float *wb = st->prof.as_shot_wb;

    if (!st->dec.raw)
        return view_vwb_target(st, out) || view_vwb_as_shot(st, out);

    if (!st->have_prof || st->prof.nb_wb_knots < 2)
        return false;

    if (st->look.wb == VIEW_WB_MANUAL) {
        *out = st->look.manual;
        return true;
    }
    if (st->look.wb == VIEW_WB_AUTO && st->awb.have_illum) {
        *out = st->awb.illum;
        return true;
    }

    /* As shot: the camera's own gains, put back on the profile's locus. A
     * neutral's raw ratios are the reciprocals of the gains over green. */
    if (!(wb[0] > 0.0f) || !(wb[2] > 0.0f))
        return false;

    return craw_profile_neutral_to_illuminant(&st->prof, wb[1] / wb[0],
                                              wb[1] / wb[2], out) >= 0;
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
           a->false_color == b->false_color && a->highlights == b->highlights &&
           a->defringe == b->defringe &&
           a->chroma == b->chroma &&
           a->denoise == b->denoise && a->temporal == b->temporal &&
           a->lens == b->lens &&
           a->lens_keep == b->lens_keep &&
           a->sharpen_radius == b->sharpen_radius &&
           a->sharpen_iterations == b->sharpen_iterations &&
           a->sharpen_contrast == b->sharpen_contrast;
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
#if CINEVIEW_HAVE_LUTS
    const ViewLut *l = view_lut_get(st->look.film_sim);

    if (l)
        return l->name;
#else
    (void)st;
#endif
    return "none";
}

#if CINEVIEW_HAVE_LUTS

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

    /* A pick from the menu is a pick of what its highlight was on, which
     * the preview has already parsed and shown: that LUT is handed over
     * as it is. Otherwise - --lut on the command line - the pack's
     * quantized LUT is expanded here, for this rendering alone. */
    if (st->preview && st->preview_sim == index) {
        pl_lut_free(&st->lut);
        st->lut = st->preview;
        st->preview = NULL;
        st->preview_sim = -1;
    } else {
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
    }

    /* One LUT at a time: a film simulation and a user .cube would
     * otherwise both claim the renderer's single LUT slot. */
    st->look.lut_path[0] = '\0';
    st->look.film_sim = index;
    view_lut_apply_space(st->lut, st->look.lut_space);
    /* Picking one is the Look section's parameter being touched. */
    if (!st->sections.look_forced)
        st->sections.look = true;

    view_log(st, "film simulation: %s (%s, %s) %dx%dx%d\n", l->name,
             view_lut_film(l->film)->name, l->label,
             st->lut->size[0], st->lut->size[1], st->lut->size[2]);

    return 0;
}

void view_lut_preview(struct view_state *st, int index)
{
    char *text = NULL;
    size_t len = 0;

    /* The chosen rendering previews as itself: nothing to show that the
     * window is not showing already. */
    if (index == st->look.film_sim)
        index = -1;
    if (index == st->preview_sim)
        return;

    pl_lut_free(&st->preview);
    st->preview_sim = -1;
    if (index < 0)
        return;

    /* A rendering that cannot be unpacked or parsed previews as nothing,
     * and says nothing: the click on it is what reports the failure. */
    if (view_luts_cube_text(index, &text, &len) < 0)
        return;
    st->preview = pl_lut_parse_cube(st->gpu.log, text, len);
    view_luts_cube_free(text);
    if (!st->preview)
        return;

    view_lut_apply_space(st->preview, st->look.lut_space);
    st->preview_sim = index;
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

#else /* !CINEVIEW_HAVE_LUTS */

int view_lut_select_film(struct view_state *st, int index)
{
    (void)index;
    snprintf(st->lut_error, sizeof(st->lut_error),
             "built without the film simulation collection");
    return CRAW_ERROR(ENOTSUP);
}

void view_lut_preview(struct view_state *st, int index)
{
    (void)st;
    (void)index;
}

static void view_list_luts(void)
{
    printf("This build carries no film simulation collection.\n");
}

#endif /* CINEVIEW_HAVE_LUTS */

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

#if CINEVIEW_HAVE_LUTS
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
    if (!st->sections.look_forced)
        st->sections.look = true;

    view_log(st, "loaded the LUT %s (%dx%dx%d)\n", st->look.lut_path,
             st->lut->size[0], st->lut->size[1], st->lut->size[2]);

fail:
    free(data);
    fclose(f);
    return ret;
}

/* ------------------------------------------------------------- the clock */

/* How fast the smoother below closes on the audio device's own cursor, and
 * how far that cursor may move in one step before the move is taken for a
 * real one rather than for noise. */
#define VIEW_CLOCK_TAU  0.30            /* seconds */
#define VIEW_CLOCK_JUMP 0.10

double view_clock(struct view_state *st)
{
    double raw, dt, err;
    int64_t now;

    if (st->audio.available)
        raw = view_audio_clock(&st->audio);
    else if (!st->playing)
        raw = st->wall_pts;             /* a silent clip, stopped */
    else
        raw = st->wall_pts +
              (double)(av_gettime_relative() - st->wall_t0) / 1.0e6;

    now = av_gettime_relative();

    /* Stopped, or not yet running: the cursor is the answer and there is
     * nothing to smooth across. */
    if (!st->clk_have || !st->playing || !st->warm) {
        st->clk_t = raw;
        st->clk_wall = now;
        st->clk_have = true;
        return raw;
    }

    /* The audio cursor is read through two quantised numbers - what SDL
     * still has queued, and when we last handed it frames - and it wanders
     * about five milliseconds either side of wall time. Read once a picture
     * that did not matter; deciding when to present from it puts every one
     * of those milliseconds into the interval between one picture and the
     * next, and that interval is what the eye sees as judder. So the clock
     * runs on wall time and is pulled towards the cursor over a third of a
     * second: still locked to the sound, no longer shaking. A jump - a
     * seek, a loop point, the first frames of a run - is not noise and is
     * taken whole. */
    dt = (double)(now - st->clk_wall) / 1.0e6;
    st->clk_wall = now;
    st->clk_t += dt;

    err = raw - st->clk_t;
    if (fabs(err) > VIEW_CLOCK_JUMP)
        st->clk_t = raw;
    else
        st->clk_t += err * (1.0 - exp(-dt / VIEW_CLOCK_TAU));

    return st->clk_t;
}

static void view_wall_restart(struct view_state *st, double pts)
{
    st->wall_pts = pts;
    st->wall_t0 = av_gettime_relative();
    /* The clock has been moved on purpose; the smoother starts again from
     * where it was put rather than sliding there. */
    st->clk_have = false;
}

double view_frame_duration(const struct view_state *st)
{
    return st->dec.frame_rate.num ? 1.0 / av_q2d(st->dec.frame_rate) : 1.0 / 29.97;
}

/* Audio is the master clock and is never made to wait for the picture. The
 * next picture is started the moment the one on screen has been presented,
 * not a decode's length before it is due: the window holds what it is
 * showing in a texture of its own, so the slot, the frame and the renderer
 * are all free again, and the next picture then has a whole frame period to
 * be decoded, developed and rendered in. What it is actually shown at is
 * decided by the clock alone (the hold in the run loop), so nothing that
 * varies about a decode - six to twenty-five milliseconds at 5.8K - reaches
 * the interval between one picture and the next, which is what judder is.
 *
 * One picture in flight, never two: `present_due` is false only between a
 * present and the step that follows it. */
static bool view_frame_due(struct view_state *st)
{
    return st->cfg.bench || !st->have_frame || !st->present_due;
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

/* ------------------------------------------------------- the rotation */

/* The picture as it is looked at: the source turned by the clip's display
 * matrix. Everything below is in these pixels, so a portrait clip fits,
 * zooms, pans and crops as the tall picture it is on screen, and the
 * renderer is the only thing that ever hears about the turn.
 *
 * Nothing upstream of the renderer does: the frame libcraw develops, the
 * lens warp, the auto white balance histogram and the denoise all run on
 * the mosaic the sensor read out, in its own orientation. */
void view_disp_size(const struct view_state *st, int *w, int *h)
{
    bool quarter = st->dec.rotation % PL_ROTATION_180 == PL_ROTATION_90;

    *w = (int)(quarter ? st->gpu.height : st->gpu.width);
    *h = (int)(quarter ? st->gpu.width  : st->gpu.height);
}

/* One corner of the rotated picture, in source pixels.
 *
 * The turn is about the picture's centre, so undoing it is the turn the
 * other way: for a quarter turn clockwise the source's top-left corner is
 * the display's top-right, and so on round. Written out rather than built
 * from a matrix because four cases are shorter than one, and because the
 * arithmetic is exact - these are integers in float clothing. */
static void view_rot_src_point(pl_rotation rot, float sw, float sh,
                               float dx, float dy, float *x, float *y)
{
    switch (pl_rotation_normalize(rot)) {
    case PL_ROTATION_90:  *x = dy;      *y = sh - dx; break;
    case PL_ROTATION_180: *x = sw - dx; *y = sh - dy; break;
    case PL_ROTATION_270: *x = sw - dy; *y = dx;      break;
    default:              *x = dx;      *y = dy;      break;
    }
}

/* And the same corner the other way. */
static void view_rot_disp_point(pl_rotation rot, float sw, float sh,
                                float x, float y, float *dx, float *dy)
{
    switch (pl_rotation_normalize(rot)) {
    case PL_ROTATION_90:  *dx = sh - y; *dy = x;      break;
    case PL_ROTATION_180: *dx = sw - x; *dy = sh - y; break;
    case PL_ROTATION_270: *dx = y;      *dy = sw - x; break;
    default:              *dx = x;      *dy = y;      break;
    }
}

/* A rectangle through either of those: two opposite corners turned and the
 * result put back in order, since a quarter turn swaps which corner is the
 * lower one on an axis. */
void view_rot_to_src(const struct view_state *st, const pl_rect2df *in,
                     pl_rect2df *out)
{
    float sw = (float)st->gpu.width, sh = (float)st->gpu.height;

    view_rot_src_point(st->dec.rotation, sw, sh, in->x0, in->y0,
                       &out->x0, &out->y0);
    view_rot_src_point(st->dec.rotation, sw, sh, in->x1, in->y1,
                       &out->x1, &out->y1);
    pl_rect2df_normalize(out);
}

void view_rot_to_disp(const struct view_state *st, const pl_rect2df *in,
                      pl_rect2df *out)
{
    float sw = (float)st->gpu.width, sh = (float)st->gpu.height;

    view_rot_disp_point(st->dec.rotation, sw, sh, in->x0, in->y0,
                        &out->x0, &out->y0);
    view_rot_disp_point(st->dec.rotation, sw, sh, in->x1, in->y1,
                        &out->x1, &out->y1);
    pl_rect2df_normalize(out);
}

/* The scale at which the whole picture just fits the area. */
static float view_fit_scale(const struct view_state *st)
{
    pl_rect2df area;
    int dw, dh;

    if (!st->gpu.width || !st->gpu.height)
        return 1.0f;

    view_picture_area(st, &area);
    view_disp_size(st, &dw, &dh);

    return CRAW_MIN(pl_rect_w(area) / (float)dw,
                    pl_rect_h(area) / (float)dh);
}

float view_zoom_scale(const struct view_state *st)
{
    return st->view.fit ? view_fit_scale(st) : st->view.zoom;
}

void view_zoom_rects(const struct view_state *st, pl_rect2df *src, pl_rect2df *dst)
{
    float z = view_zoom_scale(st);
    pl_rect2df area;
    float sw, sh, dw, dh, vw, vh, cx, cy;
    int pw, ph;

    view_picture_area(st, &area);
    /* The rotated picture's own pixels: the centre and the zoom are about
     * what is on screen, not about the readout behind it. */
    view_disp_size(st, &pw, &ph);
    sw = (float)pw;
    sh = (float)ph;

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
    int dw, dh;

    if (!st->gpu.width)
        return;

    view_disp_size(st, &dw, &dh);

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
        st->view.cx = (float)dw * 0.5f;
        st->view.cy = (float)dh * 0.5f;
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
    int dw, dh;

    view_disp_size(st, &dw, &dh);

    st->view.fit = !one_to_one;
    st->view.zoom = one_to_one ? 1.0f : view_fit_scale(st);
    /* Fitting forgets the pan as well; one to one keeps looking where the
     * picture was, unless nowhere has been looked at yet. */
    if (!one_to_one || (!st->view.cx && !st->view.cy)) {
        st->view.cx = (float)dw * 0.5f;
        st->view.cy = (float)dh * 0.5f;
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

/* Window pixels to the rotated picture's pixels, through the same two
 * rectangles the display path is built from, so that the rubber band - and
 * the spot white balance's click - lands on the pixels it is drawn over at
 * any zoom and pan. */
void view_picture_pos(const struct view_state *st, int px, int py,
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
    view_picture_pos(st, px, py, &st->crop.ax, &st->crop.ay);
}

void view_crop_drag(struct view_state *st, int px, int py)
{
    struct view_crop *c = &st->crop;
    pl_rect2df drawn, r;
    float bx, by;
    int x0, y0, x1, y1;

    if (!c->dragging || !st->gpu.width || !st->gpu.height)
        return;

    view_picture_pos(st, px, py, &bx, &by);

    /* Drawn on the picture as it is shown and kept in source pixels: the
     * rectangle has to outlive a zoom, a pan and a resized window, and the
     * only frame of reference that does is the sensor's own. The turn back
     * is exact, so what is stored is the rectangle that was drawn. */
    drawn = (pl_rect2df) { c->ax, c->ay, bx, by };
    pl_rect2df_normalize(&drawn);
    view_rot_to_src(st, &drawn, &r);

    x0 = view_crop_snap(r.x0, (int)st->gpu.width);
    y0 = view_crop_snap(r.y0, (int)st->gpu.height);
    x1 = view_crop_snap(r.x1, (int)st->gpu.width);
    y1 = view_crop_snap(r.y1, (int)st->gpu.height);

    /* The rectangle is replaced only once the pointer has really moved, so
     * that a stray click keeps the one that is already there. */
    if (x1 == x0 && y1 == y0)
        return;

    c->x0 = x0;
    c->y0 = y0;
    c->x1 = x1;
    c->y1 = y1;
    c->have = true;
    /* Drawn by hand, so it is no longer the clip's own rectangle - and it
     * belongs to the frame it was drawn on, which is what makes it a
     * rectangle to drop when the next clip is another size. */
    c->from_clip = false;
    c->frame_w = (int)st->gpu.width;
    c->frame_h = (int)st->gpu.height;
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
    st->crop.from_clip = false;
}

/* The clip's own crop as the default rectangle, read once per clip.
 *
 * A ProRes RAW frame header can mark a picture inside the readout, and for
 * the cameras that do it the sensels around that picture are real, so the
 * viewer's decoder is told not to crop (decode.c): the frame stays the
 * whole readout, libcraw develops all of it, and this rectangle only
 * decides what is looked at. Clearing it brings the rest back.
 *
 * The CFA phase does not come into it, which is the point of doing it here
 * rather than in the pipeline: the frame still starts at its own (0, 0),
 * which is the sensel the profile's phase describes. The offsets happen to
 * be even as well, so the picture's own corner carries the same site.
 *
 * A rectangle from --crop wins: the command line is a choice, not a
 * default. */
static void view_crop_from_clip(struct view_state *st, const AVFrame *f)
{
    struct view_crop *c = &st->crop;
    int x0, y0, x1, y1;

    if (c->clip_read)
        return;
    c->clip_read = true;

    /* A rectangle is a region of a frame of a particular size, so a clip of
     * another size cannot hold it: it is dropped and the whole frame is
     * exported until another is drawn. A rectangle from --crop has met no
     * frame yet and is bound to this one. */
    if (c->have && c->frame_w && (c->frame_w != f->width || c->frame_h != f->height)) {
        view_log(st, "crop: %dx%d at %d,%d was drawn on a %dx%d frame and this "
                     "clip is %dx%d, so it is dropped\n",
                 c->x1 - c->x0, c->y1 - c->y0, c->x0, c->y0,
                 c->frame_w, c->frame_h, f->width, f->height);
        c->have = false;
        c->from_clip = false;
    }
    c->frame_w = f->width;
    c->frame_h = f->height;

    if (c->have)
        return;
    if (!f->crop_left && !f->crop_top && !f->crop_right && !f->crop_bottom)
        return;

    /* Snapped to even edges like every other rectangle: half a Bayer quad
     * is not a thing to crop to, whoever picked the numbers. */
    x0 = view_crop_snap((float)f->crop_left, (int)st->gpu.width);
    y0 = view_crop_snap((float)f->crop_top, (int)st->gpu.height);
    x1 = view_crop_snap((float)(f->width - (int)f->crop_right), (int)st->gpu.width);
    y1 = view_crop_snap((float)(f->height - (int)f->crop_bottom), (int)st->gpu.height);

    if (x1 - x0 < 2 || y1 - y0 < 2)
        return;

    c->x0 = x0;
    c->y0 = y0;
    c->x1 = x1;
    c->y1 = y1;
    c->have = true;
    c->from_clip = true;

    view_log(st, "clip crop: %dx%d at %d,%d of the %dx%d readout\n",
             x1 - x0, y1 - y0, x0, y0, f->width, f->height);
}

/* A clip brings its own crop, so whatever the last one defaulted to goes
 * with it; a rectangle the user drew or asked for on the command line
 * stays. */
static void view_crop_new_clip(struct view_state *st)
{
    if (st->crop.from_clip)
        view_crop_clear(st);
    st->crop.clip_read = false;
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

/* The same rectangle on the rotated picture: what the user drew, and the
 * region the export writes once it has been turned upright. */
bool view_crop_disp_rect(const struct view_state *st, int *x, int *y,
                         int *w, int *h)
{
    pl_rect2df r, d;
    int sx, sy, sw, sh;
    bool have = view_crop_rect(st, &sx, &sy, &sw, &sh);

    r = (pl_rect2df) { (float)sx, (float)sy, (float)(sx + sw), (float)(sy + sh) };
    view_rot_to_disp(st, &r, &d);

    *x = (int)lrintf(d.x0);
    *y = (int)lrintf(d.y0);
    *w = (int)lrintf(pl_rect_w(d));
    *h = (int)lrintf(pl_rect_h(d));

    return have;
}

bool view_crop_screen(const struct view_state *st, pl_rect2df *out)
{
    pl_rect2df src, dst, r, d;
    float zx, zy;
    int x, y, w, h;

    if (!st->crop.have || !view_crop_rect(st, &x, &y, &w, &h))
        return false;

    /* The zoom rectangles are in the rotated picture's pixels and the crop
     * is in the source's, so it turns on its way to the window - which is
     * what keeps the shade on the pixels the rubber band was drawn over. */
    r = (pl_rect2df) { (float)x, (float)y, (float)(x + w), (float)(y + h) };
    view_rot_to_disp(st, &r, &d);

    view_zoom_rects(st, &src, &dst);
    if (!(pl_rect_w(src) > 0.0f) || !(pl_rect_h(src) > 0.0f))
        return false;

    zx = pl_rect_w(dst) / pl_rect_w(src);
    zy = pl_rect_h(dst) / pl_rect_h(src);

    *out = (pl_rect2df) {
        dst.x0 + (d.x0 - src.x0) * zx,
        dst.y0 + (d.y0 - src.y0) * zy,
        dst.x0 + (d.x1 - src.x0) * zx,
        dst.y0 + (d.y1 - src.y0) * zy,
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

    /* The histogram this step would read belongs to the frame a seek left
     * behind; the next one is the first of the new position. */
    if (st->awb.skip > 0) {
        st->awb.skip--;
        return 0;
    }

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

    /* One frame period per estimate rather than the clip's pts: the filter
     * smooths what the eye sees, and a seek's jump in pts is not elapsed
     * time (see struct view_awb). */
    st->awb.clock += view_frame_duration(st);
    ret = craw_awb_smoother_push(st->awb.smoother, st->awb.clock, &st->awb.last, false, &out);
    if (ret < 0 && ret != CRAW_ERROR(EAGAIN))
        return ret;

    /* EAGAIN means there is no track yet and this estimate cannot start one,
     * so the next frame stays on whatever the last good one left. */
    if (ret >= 0) {
        st->awb.illum = out;
        st->awb.have_illum = true;
    }

    if (st->cfg.verbose)
        view_log(st, "awb: frame %" PRId64 " est %s cct %5.0f tint %+.4f conf %.2f"
                     " -> track %s cct %5.0f tint %+.4f\n",
                 st->index, st->awb.last.valid ? "ok" : "invalid",
                 st->awb.last.illuminant.cct, st->awb.last.illuminant.tint,
                 st->awb.last.confidence,
                 ret >= 0 ? "moved" : "held",
                 st->awb.have_illum ? st->awb.illum.cct : 0.0f,
                 st->awb.have_illum ? st->awb.illum.tint : 0.0f);

    return 0;
}

/* The profile a frame is developed with: the camera's tables and the
 * frame's own side data. */
static int view_frame_profile(struct view_state *st, const AVFrame *f)
{
    int ret;

    st->prof = st->base;
    ret = view_profile_from_frame(&st->prof, f);
    if (ret < 0)
        return ret;

    /* The same exposure model cineraw uses: the frame gain times the
     * working index over the as-shot ISO, times the bias in stops. */
    if (!(st->prof.as_shot_ei > 0.0f))
        st->prof.as_shot_ei = st->dec.meta_as_shot_ei;
    if (view_exposure_ei(st) > 0.0f)
        st->prof.exposure_index = view_exposure_ei(st);
    else if (!(st->prof.exposure_index > 0.0f))
        st->prof.exposure_index = st->prof.as_shot_ei;
    st->have_prof = true;

    return 0;
}

/* The raw path from the held picture to a submitted pipeline pass: the
 * profile from the frame's side data, the exposure model, the parameters
 * from the look, the pass itself and the auto white balance step that reads
 * the previous pass's histogram. `flags` are CRAW_FRAME_FLAG_* bits added
 * for this one pass on top of what the look asks for; zero for every
 * picture that is meant to be looked at. `replay` says the picture is the
 * one the last pass developed, again. */
static int view_process_frame(struct view_state *st, uint32_t flags, bool replay)
{
    CRAWFrameParams fp;
    bool reset;
    int ret;

    ret = view_frame_profile(st, st->frame);
    if (ret < 0)
        return ret;

    /* The whole decoded frame: the decoder was told not to crop, so this is
     * the camera's entire readout and every sensel of it is developed. What
     * the frame header marked as the picture becomes the crop rectangle
     * below, which is a view of the result and not a change to it. */
    ret = view_gpu_ensure(&st->gpu, (uint32_t)st->frame->width,
                          (uint32_t)st->frame->height, st->look.debayer);
    if (ret < 0)
        return ret;

    view_crop_from_clip(st, st->frame);

    view_frame_params(st, &fp);
    fp.flags |= flags;

    /* The temporal state, DESIGN.md 14.7. A frame that follows a break in
     * the run starts the average again; the frame the last pass developed,
     * developed again, is a replay whether it came from a grade change, a
     * pick or a seek to the picture already on screen, so that it comes
     * out as it did the first time instead of averaged with itself. A
     * frame the player dropped to keep up is neither: the next one is
     * still the next frame of the same scene, only further on, and the
     * pass takes what moved as motion. */
    reset = !replay && st->denoise.reset;
    if (!replay && !reset && st->denoise.have_last && st->denoise.last == st->index)
        replay = true;
    if (reset)
        fp.flags |= CRAW_FRAME_FLAG_TEMPORAL_RESET;
    fp.denoise_temporal_replay = replay;
    if (fp.denoise_temporal_strength > 0.0f && (fp.flags & CRAW_FRAME_FLAG_TEMPORAL_RESET))
        view_log(st, "temporal: frame %" PRId64 " starts the average again\n", st->index);

    /* Taken before the white balance step below moves the track, so that a
     * new estimate counts as a change and a paused picture follows it. */
    st->processed = view_pipeline_look(st);

    ret = view_gpu_process(&st->gpu, st->frame, &fp, &st->timing);
    if (ret < 0)
        return ret;

    st->have_frame = true;
    if (!replay) {
        st->denoise.reset = false;
        st->denoise.have_last = true;
        st->denoise.last = st->index;
    }

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

        acc = st->gpu.last_gpu_lens ? &st->lens.ms_on : &st->lens.ms_off;
        *acc = *acc > 0.0 ? *acc * 0.9 + st->gpu.last_gpu_ms * 0.1
                          : st->gpu.last_gpu_ms;
    }

    /* The histogram describes the frame that was just processed; what comes
     * out of the smoother is applied to the next one, exactly as in
     * cineraw. */
    if (st->look.wb == VIEW_WB_AUTO && st->prof.nb_wb_knots) {
        ret = view_awb_step(st);
        if (ret < 0)
            return ret;
    }

    return 0;
}

int view_replay_frame_flags(struct view_state *st, uint32_t flags)
{
    if (!st->dec.raw || !st->have_frame || !st->frame->buf[0] ||
        st->frame->format != AV_PIX_FMT_VULKAN) {
        st->processed = view_pipeline_look(st);
        return 0;
    }

    view_log(st, "replay frame %" PRId64 "%s\n", st->index,
             (flags & CRAW_FRAME_FLAG_NO_GAMUT_COMPRESSION) ? " (uncompressed, for a pick)"
           : (flags & CRAW_FRAME_FLAG_TEMPORAL_RESET) ? " (no temporal history)" : "");

    return view_process_frame(st, flags, true);
}

int view_replay_frame(struct view_state *st)
{
    return view_replay_frame_flags(st, 0);
}

/* ------------------------------------------------------------ the pre-roll */

/* A decoded picture's place on the clip's timeline, in seconds. */
static double view_frame_pts(const struct view_state *st, const AVFrame *f)
{
    const AVStream *vs = st->dec.fmt->streams[st->dec.stream_index];
    int64_t start = vs->start_time != AV_NOPTS_VALUE ? vs->start_time : 0;
    int64_t pts = f->pts != AV_NOPTS_VALUE ? f->pts : 0;

    return (double)(pts - start) * av_q2d(st->dec.time_base);
}

static int view_seek_decoder(struct view_state *st, int64_t frame)
{
    int ret = view_decoder_seek(&st->dec, frame);

    if (ret < 0)
        return ret;

    /* A rebuilt decoder brings a whole new frame pool with it. */
    if (st->dec.frames_reset) {
        view_gpu_drop_src_views(&st->gpu);
        st->dec.frames_reset = false;
    }

    return 0;
}

/* Whether a jump warms the temporal average before its target: the pass
 * is set to run, on a raw clip whose frames can be counted. Before a
 * clip's first picture there is no frame profile to ask for the noise
 * model, so the camera's table is asked at the clip's ISO, which is what
 * that picture will find. */
static bool view_preroll_wanted(const struct view_state *st)
{
    CRAWNoiseModel nm;

    if (!st->dec.raw || !st->dec.frame_rate.num ||
        !st->denoise.on || !(st->denoise.temporal > 0.0f))
        return false;
    if (st->have_prof)
        return view_temporal_strength(st) > 0.0f;

    return craw_profile_noise_eval(&st->base, st->dec.meta_as_shot_ei, &nm) >= 0;
}

/* One frame ahead of a seek's target, developed for the temporal state it
 * leaves and nothing else. The state is the raw stage's (DESIGN.md 14.7):
 * what it holds follows from the mosaic, the noise model, the spatial
 * strength and alpha_max, and from nothing after the stage, so the passes
 * there are left out - the sharpening at a paused picture's full count
 * above all, which costs more than the rest of the pipeline together.
 * `reset` starts the average at this frame. A scene-referred export on its
 * way to its first frame gets the frame through its own pipeline too. */
static int view_preroll_frame(struct view_state *st, const AVFrame *f,
                              int64_t index, bool reset)
{
    CRAWFrameParams fp;
    int ret;

    if (f->format != AV_PIX_FMT_VULKAN)
        return CRAW_ERROR(ENOTSUP);

    ret = view_frame_profile(st, f);
    if (ret < 0)
        return ret;

    ret = view_gpu_ensure(&st->gpu, (uint32_t)f->width, (uint32_t)f->height,
                          st->look.debayer);
    if (ret < 0)
        return ret;

    view_frame_params(st, &fp);
    fp.denoise_chroma_strength = 0.0f;
    fp.false_color_passes = 0;
    fp.defringe_strength = 0.0f;
    fp.lens = (CRAWLensCorrection) { 0 };
    fp.sharpen_iterations = 0;
    fp.flags &= ~(uint32_t)(CRAW_FRAME_FLAG_HIGHLIGHT_RECONSTRUCTION |
                            CRAW_FRAME_FLAG_LENS_KEEP_ALL);
    if (reset)
        fp.flags |= CRAW_FRAME_FLAG_TEMPORAL_RESET;

    ret = view_gpu_preroll(&st->gpu, f, &fp);
    if (ret < 0)
        return ret;

    ret = view_encode_preroll(st, f, &fp);
    if (ret < 0)
        return ret;

    st->denoise.reset = false;
    st->denoise.have_last = true;
    st->denoise.last = index;

    return 0;
}

/* Develops the frames from `first` up to the seek's target `want` through
 * the pipeline without showing any of them, the first starting the average
 * again, and then decodes the target itself into st->frame and sets
 * `held`, for view_step_frame() to develop in place of decoding one: a
 * target left to the next step could be starved by the reader, which the
 * pre-roll has just run dry, and the audio would set off without it. A
 * picture just past the target that turns up in its place is held the
 * same way, as the step's landing check would take it.
 *
 * Nothing else about the seek sees the pre-roll: no picture is presented
 * or counted, the audio is still parked at the target and the clock only
 * starts once the target is there, the auto white balance reads the
 * target's histogram a frame later as it would have, and the slots stay
 * as they were (view_gpu_preroll()). It waits for the medium, which is
 * what makes it slow when it is: the packets are read ahead on the reader
 * thread, and the seek has only just emptied its queue.
 *
 * One that cannot finish - a frame that will not decode, the end of the
 * clip, a medium that stops delivering, the device refusing a pass - is
 * abandoned: the decoder is sent to the target again and the target starts
 * the average itself, as every seek did before there was a pre-roll. Only
 * a medium that stops once the frames before the target are all in leaves
 * the average warm and the target to the next step. Negative only when the
 * seek back fails. */
static int view_preroll(struct view_state *st, int64_t first, int64_t want,
                        bool *held)
{
    AVFrame *f = av_frame_alloc();
    CRAWLensCorrection lens;
    int64_t t0 = av_gettime_relative(), idle = t0, index, at = first - 1;
    const char *why = "out of memory";
    int done = 0, stale = 0, ret;

    *held = false;
    if (!f)
        goto abandon;

    while (1) {
        ret = view_decoder_next(&st->dec, f);
        if (ret == 2) {
            if (av_gettime_relative() - idle > VIEW_PREROLL_PATIENCE_MS * 1000) {
                why = "the medium stopped delivering";
                if (at == want - 1)
                    break;
                goto abandon;
            }
            view_decoder_wait(&st->dec, 100);
            continue;
        }
        if (ret) {
            why = ret > 0 ? "the clip ended" : "a frame did not decode";
            goto abandon;
        }
        idle = av_gettime_relative();

        index = (int64_t)llrint(view_frame_pts(st, f) * av_q2d(st->dec.frame_rate));

        /* What a decoder flushed after the end of the stream can still hand
         * back from before the seek, the same thing view_step_frame()
         * checks a landing for. */
        if (index < first || index > want + 8) {
            av_frame_unref(f);
            if (++stale > 8) {
                why = "the decoder kept handing back frames from elsewhere";
                goto abandon;
            }
            continue;
        }

        if (index >= want) {
            av_frame_unref(st->frame);
            av_frame_move_ref(st->frame, f);
            *held = true;
            break;
        }

        /* The ring hands each record out once; this one is spoken for. */
        view_decoder_lens(&st->dec, f->pts, &lens);

        ret = view_preroll_frame(st, f, index, !done);
        av_frame_unref(f);
        if (ret < 0) {
            why = "the pipeline refused a frame";
            goto abandon;
        }
        done++;
        at = index;
    }

    av_frame_free(&f);
    view_log(st, "preroll: %d frames before %" PRId64 ", %.1f ms%s%s%s\n", done, want,
             (double)(av_gettime_relative() - t0) / 1000.0,
             *held ? "" : "; the target is not in yet (", *held ? "" : why,
             *held ? "" : ")");

    return 0;

abandon:
    av_frame_free(&f);
    view_log(st, "preroll: abandoned after %d frames before %" PRId64 " (%s); "
                 "the target starts the average\n", done, want, why);
    st->denoise.reset = true;
    view_encode_preroll(st, NULL, NULL);

    return view_seek_decoder(st, want);
}

int view_step_frame(struct view_state *st)
{
    AVStream *vs;
    CRAWLensCorrection prev_lens;
    bool held = false;
    int64_t t0;
    int ret;

    /* There is no clip open, which only happens when a switch could not
     * open the new one and could not get the old one back either: there is
     * nothing left to show and the run loop is told so. */
    if (!st->dec.fmt)
        return CRAW_ERROR(EIO);
    vs = st->dec.fmt->streams[st->dec.stream_index];

    if (st->seek_to >= 0) {
        /* Replaying the picture already on screen, which is how a grade
         * change repaints while paused, is not a move in time and must not
         * throw the illuminant track away. An export's first frame is a
         * jump even then (encode.c). */
        bool jump = st->seek_cold || !st->have_frame || st->seek_to != st->index;
        int64_t want = st->seek_to;
        int64_t first = want;

        double target = st->dec.frame_rate.num
                      ? (double)st->seek_to / av_q2d(st->dec.frame_rate) : 0.0;

        /* The temporal denoise's state does not survive a jump: it is last
         * frame's picture sample by sample, and after a jump no sample is
         * where it was. A loop back to the start is a jump like any other.
         * The average then starts again VIEW_PREROLL frames ahead of the
         * target rather than at it, and the frames between are developed
         * and not shown (view_preroll()), so that the target comes out as
         * the next frame of an average that has had time to settle instead
         * of the first of one that has none. The first frame of a clip has
         * nothing before it, and the seeks of a knob being dragged go
         * without (ui.c). */
        if (jump && !st->scrubbing && view_preroll_wanted(st))
            first = CRAW_MAX(want - VIEW_PREROLL, 0);
        st->seek_to = -1;
        st->seek_cold = false;

        ret = view_seek_decoder(st, first);
        if (ret < 0)
            return ret;

        /* Both readers go with the seek: the audio fades out, drops
         * everything in front of the target and arms the new run. */
        view_audio_seek(&st->audio, target);
        view_gpu_reset_peak(&st->gpu, false);
        st->sync.run_frames = 0;
        st->seek_target = jump ? want : -1;
        st->seek_tries = 0;

        /* The clock now runs from the target, so the frame that lands there
         * is due at once. Left at the old picture's time, a backward seek
         * or a loop point would wait for the clock to climb all the way
         * back to it. */
        st->pts = target;

        /* The auto track survives a seek: the clip is one scene, and the
         * smoother's own clamp on the time step already limits how far one
         * frame can pull it. Only the stale estimate is dropped, the one
         * whose histogram was read a frame behind and belongs to the
         * picture the seek left. */
        if (jump) {
            st->awb.skip = 1;
            st->denoise.reset = true;
        }

        if (first < want) {
            ret = view_preroll(st, first, want, &held);
            if (ret < 0)
                return ret;
        }

        /* Only now, so that the time the pre-roll took is not taken for
         * time the picture has fallen behind: the audio is parked at the
         * target until the run loop starts it again. */
        view_wall_restart(st, target);
    }

    t0 = av_gettime_relative();
    st->dec.read_ms = 0.0;
    ret = held ? 0 : view_decoder_next(&st->dec, st->frame);
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

    /* The lens record belongs to this picture and is taken exactly once,
     * here: view_process_frame() runs again whenever a paused frame is
     * replayed, and the ring hands each record out only the first time. */
    prev_lens = st->lens.record;
    view_decoder_lens(&st->dec, st->frame->pts, &st->lens.record);
    if (st->lens.record.valid) {
        craw_lens_corner_shift(&st->lens.record, (uint32_t)st->frame->width,
                               (uint32_t)st->frame->height, true, st->lens.corner);

        /* The keep-everything scale is a boundary walk of about a
         * millisecond, so it is worked out here rather than by the panel
         * sixty times a second, and only when the record has actually
         * changed - which on a clip that is not zooming is once. It is a
         * readout only: craw_pipeline_record() works out its own. The
         * record is the whole key because a clip does not change frame size
         * while it plays. */
        if (memcmp(&prev_lens, &st->lens.record, sizeof(prev_lens)) &&
            craw_lens_keep_all_scale(&st->lens.record, (uint32_t)st->frame->width,
                                     (uint32_t)st->frame->height, &st->lens.scale) < 0)
            st->lens.scale = 0.0f;
    }

    st->pts = view_frame_pts(st, st->frame);
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
        /* Nothing to check a landing against on this path: the picture
         * that comes out is where the seek went. */
        st->seek_target = -1;

        /* The auto white balance reads the previous picture and starts on
         * this one: a frame behind, like the raw path's histogram, and for
         * the same reason waited for rather than polled, so that an export
         * walks the same track every time. */
        view_vwb_step(st, true);

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
        fprintf(stderr, "the decoder produced software frames; cineview needs the "
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

    ret = view_process_frame(st, 0, false);
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

static int view_parse_wb(struct view_state *st, const char *arg)
{
    struct view_look *lk = &st->look;
    char *end;
    double cct, tint = 0.0;

    if (!strcmp(arg, "as-shot")) {
        lk->wb = VIEW_WB_AS_SHOT;
        return 0;
    }

    /* Anything but as-shot is a balance to apply, which on a developed
     * clip is the section's switch; a raw clip has none. */
    st->sections.wb = true;

    if (!strcmp(arg, "auto")) {
        lk->wb = VIEW_WB_AUTO;
        return 0;
    }

    /* pick:X,Y: the spot white balance on that source pixel of the first
     * frame the run develops - frame 0 in the window, the exported frame
     * headless - through exactly the path the panel's button takes. The
     * mode stays as it was until the pick lands, and becomes manual with
     * what it found. */
    if (!strncmp(arg, "pick:", 5)) {
        long x, y;
        const char *p = arg + 5;

        x = strtol(p, &end, 10);
        if (end == p || *end != ',') {
            fprintf(stderr, "--wb: expected pick:X,Y in source pixels, got `%s`\n",
                    arg);
            return CRAW_ERROR(EINVAL);
        }
        p = end + 1;
        y = strtol(p, &end, 10);
        if (end == p || *end || x < 0 || y < 0 || x > INT_MAX || y > INT_MAX) {
            fprintf(stderr, "--wb: expected pick:X,Y in source pixels, got `%s`\n",
                    arg);
            return CRAW_ERROR(EINVAL);
        }
        view_pick_at(st, (int)x, (int)y);
        return 0;
    }

    cct = strtod(arg, &end);
    if (end == arg || (*end && *end != ':')) {
        fprintf(stderr, "--wb: expected as-shot, auto, CCT[:TINT] or pick:X,Y, "
                        "got `%s`\n", arg);
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

/* `--bypass exposure,look,adjustments`: the panel's section switches, off.
 *
 * It exists for the three sections whose switch is a bypass and nothing
 * else, so that no other option can reach it: Denoise, Lens and Film grain
 * are switched by --denoise 0, --lens off and the absence of --grain, and
 * the crop mode is not a picture setting at all. A headless export can then
 * be handed exactly the state the panel is in. */
static int view_parse_bypass(struct view_sections *s, const char *arg)
{
    char buf[128];
    char *tok, *save = NULL;

    snprintf(buf, sizeof(buf), "%s", arg);

    for (tok = strtok_r(buf, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        if (!strcmp(tok, "exposure"))
            s->exposure = false, s->exposure_forced = true;
        else if (!strcmp(tok, "look"))
            s->look = false, s->look_forced = true;
        else if (!strcmp(tok, "adjustments") || !strcmp(tok, "adjust"))
            s->adjust = false, s->adjust_forced = true;
        else {
            fprintf(stderr, "--bypass: `%s` is not a section, expected "
                            "exposure, look or adjustments\n", tok);
            return CRAW_ERROR(EINVAL);
        }
    }

    return 0;
}

static void view_usage(FILE *f)
{
    fprintf(f,
"usage: cineview [options] FILE...\n"
"\n"
"Opens a camera raw clip, decodes it with FFmpeg's Vulkan hwaccel, runs\n"
"every frame through libcraw on the same device and displays the result\n"
"through libplacebo, with a nuklear panel for the grade.\n"
"\n"
"More than one file makes a playlist: `r` and `t` step to the previous and\n"
"the next clip, wrapping at both ends, and the panel's clip line says which\n"
"one of how many is open. The clips need not match - raw and already\n"
"developed, any size, any audio layout - and every setting below stays as\n"
"it was set across a switch; only what is read from the clip itself is\n"
"read again. A headless export (--export, --export-video) uses the first\n"
"file and ignores the rest.\n"
"\n"
"  --device SUBSTR     use the Vulkan device whose name contains SUBSTR\n"
"  --debug             turn the Vulkan validation layer on\n"
"  --bench             play as fast as the pipeline allows instead of at the\n"
"                      clip's own frame rate\n"
"  --trace FILE        write one line per present - when the packet arrived,\n"
"                      when it was decoded, developed and rendered, and the\n"
"                      interval to the last present - and a cadence summary\n"
"  --no-loop           stop at the end of the clip instead of starting over\n"
"  --loops N           play the clip N times and quit, printing the A/V\n"
"                      statistics (default: loop for ever)\n"
"  --audio-dump FILE   write exactly the float samples handed to the audio\n"
"                      device, raw interleaved float32 48 kHz stereo. The\n"
"                      file is the clip\'s: switching clips starts it again\n"
"  --audio-map A[,B]   audio stream indices, in audio stream order, to use\n"
"                      as left and right; one index means a stereo stream\n"
"                      or a mono one sent to both ears\n"
"  -v                  verbose: timings, metadata and libcraw's own log\n"
"  -h                  this text\n"
"\n"
"  --wb MODE           as-shot, auto, CCT[:TINT] in Kelvin, or pick:X,Y: the\n"
"                      spot white balance on the source pixel X,Y of the\n"
"                      first frame the run develops - frame 0 in the window,\n"
"                      the exported frame headless - which is the panel's\n"
"                      \"pick white\" button on that pixel. The 24x24 patch\n"
"                      around it is read back and the illuminant that makes\n"
"                      it neutral becomes the manual white balance; -v\n"
"                      prints what was found. A developed clip is\n"
"                      rebalanced from the temperature it records, and any\n"
"                      MODE but as-shot switches its section on\n"
"  --ei INDEX          working exposure index; the frame is scaled by\n"
"                      INDEX / as-shot ISO\n"
"  --exposure STOPS    exposure bias on top of --ei (default: 0)\n"
"  --debayer METHOD    bilinear, mhc or rcd (default: rcd)\n"
"  --false-colour PASSES\n"
"                      false colour suppression after the debayer: 1 to 5\n"
"                      passes of a median and a blur over the colour\n"
"                      differences, RawTherapee's steps, which turn the\n"
"                      coloured dots of glints smaller than a CFA quad\n"
"                      neutral and leave luminance alone; 0 (the default)\n"
"                      is off. The Debayer section has the same row\n"
"  --denoise STRENGTH[,CHROMA[,TEMPORAL]]\n"
"                      raw domain denoising, in units of the camera's\n"
"                      modelled noise level: 1.0 removes exactly that much,\n"
"                      up to 3.0, 0 (the default) leaves the section off.\n"
"                      Needs a noise model for the clip's ISO; the Denoise\n"
"                      section says which one it found. CHROMA, 0 to 1, is\n"
"                      the section's chroma denoise after the debayer, the\n"
"                      luma-guided smoothing of the quarter resolution\n"
"                      chroma band; it needs no model and defaults to 1\n"
"                      when the section is on. TEMPORAL, 0 to 0.95 and 0 by\n"
"                      default, averages every sample with the previous\n"
"                      frame's wherever the two agree within the model's\n"
"                      noise: the most of the previous frame a sample\n"
"                      keeps, 0.8 being about five frames of a static shot.\n"
"                      A seek or an export first develops the five frames\n"
"                      before the one it starts on, unseen, so that the\n"
"                      average has settled there.\n"
"                      A STRENGTH of 0 with a CHROMA or TEMPORAL above it\n"
"                      turns the section on with the spatial filter off\n"
"  --highlights on|off rebuild clipped highlights before the debayer (on by\n"
"                      default): every channel that clipped is estimated\n"
"                      from the channels that did not clip around it and\n"
"                      from the colour of the unclipped light around the\n"
"                      blown area, and may exceed the balanced clip, so a\n"
"                      sky where only green clipped keeps its clouds. The\n"
"                      Debayer section has the same switch\n"
"  --defringe STRENGTH pull the colour of pixels along strong edges back into\n"
"                      the range the two sides of the edge span, 0 to 1 (1 by\n"
"                      default, 0 is off): the purple, blue and pink rims\n"
"                      chromatic aberration leaves along lamps, windows and\n"
"                      branches against the sky. The Debayer section has the\n"
"                      same row\n"
"  --sharpen RADIUS[,ITERATIONS[,CONTRAST]]\n"
"                      capture sharpening after every other stage on the\n"
"                      camera RGB: Richardson-Lucy deconvolution of the luma\n"
"                      by a Gaussian of RADIUS pixels (0.8 is this camera's\n"
"                      in-focus blur; up to %g), ITERATIONS of it (1 to %d,\n"
"                      default 20) on a paused picture and on export and at\n"
"                      most %d while playing, and nothing sharpened where the\n"
"                      local contrast is below CONTRAST times the modelled\n"
"                      noise (default 1; with the Denoise section on, 0.5\n"
"                      sharpens more texture and still none of the noise it\n"
"                      leaves). 0 (the default) leaves the Sharpen section\n"
"                      off with its numbers kept\n"
"  --lens on|off       geometric lens correction from each frame's own\n"
"                      optical_correction record: a per-channel radial warp\n"
"                      after the debayer that takes the distortion out and\n"
"                      the lateral chromatic aberration with it. On by\n"
"                      default for a clip whose frames carry a record, and a\n"
"                      no-op for one that does not; the Lens section says\n"
"                      which it is and what the record does at the corner\n"
"  --lens-fit MODE     fill (the default) fills the frame and loses the\n"
"                      source's outer corners, which is the trade the camera\n"
"                      itself makes; keep shrinks the corrected picture until\n"
"                      every source pixel fits and paints the rest black. The\n"
"                      Lens section cycles the same row and prints the scale\n"
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
"  --tone-map NAME     the curve the window is tone mapped with, out of\n"
"                      libplacebo's own list: spline (the default), bt2390,\n"
"                      bt2446a, st2094-40, st2094-10, reinhard, mobius,\n"
"                      hable, gamma, linear, linearlight or clip. A\n"
"                      display-referred export is mapped with the same one,\n"
"                      so the file keeps matching the window\n"
"  --no-colour-management\n"
"                      leave the window untagged sRGB. On Wayland it\n"
"                      otherwise follows the compositor's preferred image\n"
"                      description: the picture is rendered into it and the\n"
"                      window tagged with it, so the compositor has nothing\n"
"                      left to convert, and again whenever the preference\n"
"                      changes - another output, HDR switched on. On X11\n"
"                      the window is untagged sRGB either way\n"
"  --softlight PERCENT blend the picture with itself through the soft light\n"
"                      blend mode, 0 to 100 (default: 0, off)\n"
"  --vibrance PASTELS[,SATURATED[,THRESHOLD]]\n"
"                      raise chroma, more on pastel colours than on already\n"
"                      saturated ones. The two amounts run -100 to 100 and\n"
"                      one number sets both; THRESHOLD, 0 to 100, is the\n"
"                      chroma they meet at (default: 45). The panel also\n"
"                      has the skin protection and gamut toggles\n"
"  --bypass SECTION[,...]\n"
"                      start with these panel sections switched off, their\n"
"                      settings kept: exposure (the bias and the working\n"
"                      index), look (the film simulation or --lut cube) and\n"
"                      adjustments (soft light and vibrance). The other\n"
"                      sections' switches are --denoise 0, --lens off and\n"
"                      the absence of --grain\n"
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
"                      opening a window: one image, the still export. The\n"
"                      name picks the format through libavformat: .png and\n"
"                      .tif hold sixteen bits, .jpg, .webp and .avif\n"
"                      whatever their encoder takes. The file holds the\n"
"                      displayed look, sRGB encoded, mapped as the window\n"
"                      is.\n"
"  --export-video START END OUT\n"
"                      encode frames START to END, inclusive, into the file\n"
"                      OUT and exit, without opening a window: the video\n"
"                      export, independent of the still one. The name picks\n"
"                      the container and --preset the encoder (--codec\n"
"                      overrides it); a Vulkan encoder (h264_vulkan,\n"
"                      hevc_vulkan, ffv1_vulkan, apv_vulkan,\n"
"                      prores_ks_vulkan) is fed on the GPU without a\n"
"                      read-back, anything else through libswscale. The\n"
"                      picture is what --export writes. A .mp4 or .mov is\n"
"                      always written with movflags +faststart, its moov\n"
"                      atom ahead of the media.\n"
"  --list-encoders     print every video encoder this FFmpeg has, marking\n"
"                      the ones that take Vulkan frames, and exit\n"
"  --size SIZE         how large the export is written: full (the default,\n"
"                      the crop rectangle's own size), 1/2, 1/4, 3840, 1920\n"
"                      or a typed WxH. A width larger than the source never\n"
"                      upscales. A software encoder at 5.8K wants three and\n"
"                      a half gigabytes, which is what this is for. Refused\n"
"                      for a scene-referred --colour, which is written at\n"
"                      native resolution or not at all. On a clip the camera\n"
"                      rotated this is the picture as it is written, upright:\n"
"                      1/2 of a 5760x3040 frame turned a quarter is 1520x2880\n"
"  --audio MODE        what the exported video does with the clip's sound:\n"
"                      none; copy (the default), the source track remuxed\n"
"                      and cut to the sample; opus[:KBPS] (libopus, 160,\n"
"                      120 ms frames); aac[:KBPS] (160); flac[:LEVEL] (0 to\n"
"                      12, 5); or wav, PCM at the source's depth: 24 bit\n"
"                      little endian, or 16 bit for a 16 bit clip. The four\n"
"                      encoders take the decoded samples, through\n"
"                      libswresample only where the encoder wants another\n"
"                      sample format or rate, cut to the same range as\n"
"                      copy. The stream is the one playback uses, and a four\n"
"                      channel stream is taken whole rather than cut down\n"
"                      to the pair the speakers get. A mode the container\n"
"                      cannot carry is refused with the muxer's reason:\n"
"                      .mov takes neither opus nor flac, .mp4 and .mkv take\n"
"                      all of them\n"
"  --threads N         threads for a software video encoder, 0 (the default)\n"
"                      for FFmpeg's own auto, 1 to the machine's core\n"
"                      count; the Vulkan encoders have none. -v prints what\n"
"                      the encoder settled on, since some clamp it\n"
"  --codec NAME        encode with this encoder rather than the one the file\n"
"                      name or the preset implies\n"
"  --enc KEY=VALUE[,...]\n"
"                      options for that encoder, e.g. cpu-used=8,threads=2\n"
"                      for an AVIF or qmin=1 for a JPEG. For a video these\n"
"                      are applied on top of --preset's, and win over them\n"
"  --preset FAMILY[/PRESET]\n"
"                      the video export's encoder and settings, out of the\n"
"                      same table the panel's menu shows, e.g.\n"
"                      \"FFv1 (Vulkan)/L4 (experimental)\" or \"x264\".\n"
"                      A family on its own takes its default preset\n"
"  --colour NAME       the colour space the video is written in: bt709 (the\n"
"                      display look, the default), bt2020-sdr, bt2020-pq or\n"
"                      bt2020-hlg, all four rendered and mapped exactly as\n"
"                      the window is, with libplacebo's own tone and gamut\n"
"                      mapping and per-frame peak detection; or the scene\n"
"                      referred acescg, aces and xyz, which are written\n"
"                      straight from the pipeline as float and so carry no\n"
"                      tone curve, no gamut mapping, no film simulation and\n"
"                      no grain at all. Those three need float samples,\n"
"                      which means FFv1 (Vulkan) with an L4 preset, and are\n"
"                      written at native resolution\n"
"  --pix-fmt NAME      the pixel format the video is written in, by FFmpeg's\n"
"                      own names: nv12, p010, yuv422p10, yuv444p12, gbrp16\n"
"                      and so on. The default is the one the preset's\n"
"                      profile implies, and only formats the encoder really\n"
"                      accepts can be asked for - --list-presets prints them\n"
"                      per family. A format the preset's profile forbids is\n"
"                      refused with the reason rather than written under the\n"
"                      wrong label: the ProRes profiles are 4:2:2 or 4:4:4,\n"
"                      AV1 Main is 4:2:0 and High is 4:4:4\n"
"  --list-presets      print every family, preset and colour space with the\n"
"                      options each one means and the pixel formats each\n"
"                      encoder accepts, and exit. The device is opened so\n"
"                      that a family whose engine is a Vulkan extension can\n"
"                      say whether this GPU has it\n"
"  --crop WxH+X+Y      export only that region of the source, in source\n"
"                      pixels, with the edges snapped to even ones. The\n"
"                      viewer shades everything outside it. A clip whose\n"
"                      frame headers mark a picture inside the readout\n"
"                      starts with that rectangle; the whole readout is\n"
"                      developed either way, and `clear` in the panel shows\n"
"                      all of it. These are the source's pixels before the\n"
"                      clip's display matrix is applied; the panel reports\n"
"                      the rectangle on the picture as it is shown, which is\n"
"                      also what a drag draws and what the file holds.\n"
"\n"
"A clip whose track header carries a display matrix is turned by it, the way\n"
"mpv and ffmpeg's autorotate turn one: the window, the crop rectangle and\n"
"every export are the picture upright, and the files written carry no matrix\n"
"of their own. Anything but a quarter turn is reported and left alone.\n"
"\n"
"Audio is the master clock. The clip's PCM track plays through SDL3 with\n"
"5 ms fades at every boundary; a silent clip falls back to wall time.\n"
"\n"
"The panel is worked with the pointer: the section headers' switches and\n"
"expanders, the sliders, the fields and the cascading menus are clicked,\n"
"and no key reaches a panel row. Keys: space play/pause; left/right seek\n"
"five seconds back and forward, shift with them one second; page up/down\n"
"zoom in and out; shift with page up/down goes to the first and last\n"
"frame; control with the arrows pans; tab or 0 fits the picture again, 1\n"
"is one to one; the wheel zooms about the pointer and dragging pans; c\n"
"turns crop mode on, clearing any rectangle, and off again keeping the\n"
"one that was drawn, where a left drag draws the rectangle the export is\n"
"cut to; ctrl+1 and ctrl+2 mark the video export's start and end at the\n"
"frame on screen; r and t go to the previous and the next file on the\n"
"command line; m mutes; escape disarms the White balance section's \"pick\n"
"white\" (whose next click on the picture samples instead of panning or\n"
"cropping), closes an open menu, and otherwise quits, as q does. While a\n"
"video export runs the panel and every key but q are locked; q stops the\n"
"export, finishes the file and quits.\n",
    (double)CRAW_SHARPEN_MAX_RADIUS, CRAW_SHARPEN_MAX_ITERATIONS, VIEW_SHARPEN_PLAY_ITERATIONS,
    craw_tool_profile_builtin_list());
}

enum {
    VIEW_OPT_DEVICE = 0x100,
    VIEW_OPT_DEBUG,
    VIEW_OPT_BENCH,
    VIEW_OPT_TRACE,
    VIEW_OPT_WB,
    VIEW_OPT_EI,
    VIEW_OPT_EXPOSURE,
    VIEW_OPT_DEBAYER,
    VIEW_OPT_LUT,
    VIEW_OPT_LUT_TYPE,
    VIEW_OPT_LUT_SPACE,
    VIEW_OPT_PROFILE,
    VIEW_OPT_TABLES,
    VIEW_OPT_EXPORT,
    VIEW_OPT_EXPORT_VIDEO,
    VIEW_OPT_LIST_ENCODERS,
    VIEW_OPT_SIZE,
    VIEW_OPT_AUDIO,
    VIEW_OPT_CODEC,
    VIEW_OPT_ENC,
    VIEW_OPT_PRESET,
    VIEW_OPT_COLOUR,
    VIEW_OPT_LIST_PRESETS,
    VIEW_OPT_CROP,
    VIEW_OPT_LOOPS,
    VIEW_OPT_NO_LOOP,
    VIEW_OPT_AUDIO_DUMP,
    VIEW_OPT_AUDIO_MAP,
    VIEW_OPT_ZOOM,
    VIEW_OPT_LIST_LUTS,
    VIEW_OPT_GRAIN,
    VIEW_OPT_DENOISE,
    VIEW_OPT_FALSE_COLOUR,
    VIEW_OPT_LENS,
    VIEW_OPT_HIGHLIGHTS,
    VIEW_OPT_DEFRINGE,
    VIEW_OPT_SHARPEN,
    VIEW_OPT_LENS_FIT,
    VIEW_OPT_SOFTLIGHT,
    VIEW_OPT_VIBRANCE,
    VIEW_OPT_PIX_FMT,
    VIEW_OPT_BYPASS,
    VIEW_OPT_THREADS,
    VIEW_OPT_TONE_MAP,
    VIEW_OPT_NO_COLOUR_MANAGEMENT,
};

static int view_parse_args(struct view_state *st, int argc, char **argv,
                           const char **profile, const char **tables)
{
    static const struct option opts[] = {
        { "device",   required_argument, NULL, VIEW_OPT_DEVICE    },
        { "debug",    no_argument,       NULL, VIEW_OPT_DEBUG     },
        { "bench",    no_argument,       NULL, VIEW_OPT_BENCH     },
        { "trace",    required_argument, NULL, VIEW_OPT_TRACE     },
        { "wb",       required_argument, NULL, VIEW_OPT_WB        },
        { "ei",       required_argument, NULL, VIEW_OPT_EI        },
        { "exposure", required_argument, NULL, VIEW_OPT_EXPOSURE  },
        { "debayer",  required_argument, NULL, VIEW_OPT_DEBAYER   },
        { "false-colour", required_argument, NULL, VIEW_OPT_FALSE_COLOUR },
        { "lut",      required_argument, NULL, VIEW_OPT_LUT       },
        { "lut-type", required_argument, NULL, VIEW_OPT_LUT_TYPE  },
        { "lut-space", required_argument, NULL, VIEW_OPT_LUT_SPACE },
        { "list-luts", no_argument,       NULL, VIEW_OPT_LIST_LUTS },
        { "profile",  required_argument, NULL, VIEW_OPT_PROFILE   },
        { "tables",   required_argument, NULL, VIEW_OPT_TABLES    },
        { "export",   required_argument, NULL, VIEW_OPT_EXPORT    },
        { "export-video", required_argument, NULL, VIEW_OPT_EXPORT_VIDEO },
        { "list-encoders", no_argument,  NULL, VIEW_OPT_LIST_ENCODERS },
        { "size",     required_argument, NULL, VIEW_OPT_SIZE      },
        { "audio",    required_argument, NULL, VIEW_OPT_AUDIO     },
        { "codec",    required_argument, NULL, VIEW_OPT_CODEC     },
        { "enc",      required_argument, NULL, VIEW_OPT_ENC       },
        { "preset",   required_argument, NULL, VIEW_OPT_PRESET    },
        { "colour",   required_argument, NULL, VIEW_OPT_COLOUR    },
        { "color",    required_argument, NULL, VIEW_OPT_COLOUR    },
        { "list-presets", no_argument,   NULL, VIEW_OPT_LIST_PRESETS },
        { "crop",     required_argument, NULL, VIEW_OPT_CROP      },
        { "loops",    required_argument, NULL, VIEW_OPT_LOOPS     },
        { "no-loop",  no_argument,       NULL, VIEW_OPT_NO_LOOP   },
        { "audio-dump", required_argument, NULL, VIEW_OPT_AUDIO_DUMP },
        { "audio-map", required_argument,  NULL, VIEW_OPT_AUDIO_MAP },
        { "zoom",     required_argument, NULL, VIEW_OPT_ZOOM      },
        { "grain",    required_argument, NULL, VIEW_OPT_GRAIN     },
        { "denoise",  required_argument, NULL, VIEW_OPT_DENOISE   },
        { "lens",     required_argument, NULL, VIEW_OPT_LENS      },
        { "highlights", required_argument, NULL, VIEW_OPT_HIGHLIGHTS },
        { "defringe", required_argument, NULL, VIEW_OPT_DEFRINGE },
        { "sharpen",  required_argument, NULL, VIEW_OPT_SHARPEN   },
        { "lens-fit", required_argument, NULL, VIEW_OPT_LENS_FIT  },
        { "softlight", required_argument, NULL, VIEW_OPT_SOFTLIGHT },
        { "vibrance", required_argument, NULL, VIEW_OPT_VIBRANCE  },
        { "pix-fmt",  required_argument, NULL, VIEW_OPT_PIX_FMT   },
        { "bypass",   required_argument, NULL, VIEW_OPT_BYPASS    },
        { "threads",  required_argument, NULL, VIEW_OPT_THREADS   },
        { "tone-map", required_argument, NULL, VIEW_OPT_TONE_MAP  },
        { "no-colour-management", no_argument, NULL, VIEW_OPT_NO_COLOUR_MANAGEMENT },
        { "no-color-management",  no_argument, NULL, VIEW_OPT_NO_COLOUR_MANAGEMENT },
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
        case VIEW_OPT_NO_COLOUR_MANAGEMENT:
            st->cfg.no_colour_management = true;
            break;
        case VIEW_OPT_TRACE:
            st->cfg.trace = optarg;
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
        case VIEW_OPT_GRAIN:
            if ((ret = view_grain_parse(&st->grain, optarg)) < 0)
                return ret;
            break;
        case VIEW_OPT_LENS:
            if (!strcmp(optarg, "on")) {
                st->lens.on = true;
            } else if (!strcmp(optarg, "off")) {
                st->lens.on = false;
            } else {
                fprintf(stderr, "--lens: expected on or off, got `%s`\n", optarg);
                return CRAW_ERROR(EINVAL);
            }
            break;
        case VIEW_OPT_DEFRINGE:
            st->look.defringe = strtof(optarg, &end);
            if (end == optarg || *end || !(st->look.defringe >= 0.0f && st->look.defringe <= 1.0f)) {
                fprintf(stderr, "--defringe: expected 0 to 1, got `%s`\n", optarg);
                return CRAW_ERROR(EINVAL);
            }
            break;
        case VIEW_OPT_SHARPEN: {
            /* RADIUS[,ITERATIONS[,CONTRAST]]; what is left out keeps the
             * panel's number, and a radius of zero leaves the section off
             * with its numbers where they were, as --denoise 0 does. */
            float r = strtof(optarg, &end);

            if (end == optarg || (*end && *end != ',') || !(r >= 0.0f && r <= CRAW_SHARPEN_MAX_RADIUS)) {
                fprintf(stderr, "--sharpen: expected RADIUS[,ITERATIONS[,CONTRAST]], the "
                                "radius 0 to %g, got `%s`\n", (double)CRAW_SHARPEN_MAX_RADIUS,
                        optarg);
                return CRAW_ERROR(EINVAL);
            }
            st->sharpen.on = r > 0.0f;
            if (r > 0.0f)
                st->sharpen.radius = r;
            if (*end == ',') {
                const char *arg = end + 1;
                long n = strtol(arg, &end, 10);

                if (end == arg || (*end && *end != ',') || n < 1 ||
                    n > CRAW_SHARPEN_MAX_ITERATIONS) {
                    fprintf(stderr, "--sharpen: ITERATIONS must be 1 to %d, got `%s`\n",
                            CRAW_SHARPEN_MAX_ITERATIONS, arg);
                    return CRAW_ERROR(EINVAL);
                }
                st->sharpen.iterations = (int)n;
            }
            if (*end == ',') {
                const char *arg = end + 1;
                float t = strtof(arg, &end);

                if (end == arg || *end || !(t >= 0.0f && t <= 10.0f)) {
                    fprintf(stderr, "--sharpen: CONTRAST must be 0 to 10, got `%s`\n", arg);
                    return CRAW_ERROR(EINVAL);
                }
                st->sharpen.contrast = t;
            }
            break;
        }
        case VIEW_OPT_HIGHLIGHTS:
            if (!strcmp(optarg, "on")) {
                st->look.highlights = 1;
            } else if (!strcmp(optarg, "off")) {
                st->look.highlights = 0;
            } else {
                fprintf(stderr, "--highlights: expected on or off, got `%s`\n", optarg);
                return CRAW_ERROR(EINVAL);
            }
            break;

        case VIEW_OPT_LENS_FIT:
            if (!strcmp(optarg, "fill")) {
                st->lens.keep = false;
            } else if (!strcmp(optarg, "keep")) {
                st->lens.keep = true;
            } else {
                fprintf(stderr, "--lens-fit: expected fill or keep, got `%s`\n",
                        optarg);
                return CRAW_ERROR(EINVAL);
            }
            st->lens.keep_sel = st->lens.keep ? 1 : 0;
            break;

        case VIEW_OPT_DENOISE: {
            float v2 = strtof(optarg, &end);

            if (end == optarg || (*end && *end != ',') || v2 < 0.0f || v2 > 3.0f) {
                fprintf(stderr, "--denoise: expected STRENGTH[,CHROMA[,TEMPORAL]], the "
                                "strength 0 to 3, got `%s`\n", optarg);
                return CRAW_ERROR(EINVAL);
            }
            /* Zero alone keeps it off and leaves the panel's own strength
             * where it was, so the section still opens on a sensible
             * number. */
            st->denoise.on = v2 > 0.0f;
            if (v2 > 0.0f)
                st->denoise.strength = v2;
            for (int k = 0; k < 2 && *end == ','; k++) {
                const char *arg = end + 1;
                float c2 = strtof(arg, &end);

                if (end == arg || (*end && (k || *end != ',')) || c2 < 0.0f ||
                    c2 > (k ? 0.95f : 1.0f)) {
                    fprintf(stderr, "--denoise: %s must be 0 to %s, got `%s`\n",
                            k ? "TEMPORAL" : "CHROMA", k ? "0.95" : "1", arg);
                    return CRAW_ERROR(EINVAL);
                }
                if (k)
                    st->denoise.temporal = c2;
                else
                    st->denoise.chroma = c2;
                /* Either alone is still a reason to switch the section on,
                 * and then a strength of zero means what it says: the
                 * spatial filter off. */
                if (c2 > 0.0f) {
                    st->denoise.on = true;
                    st->denoise.strength = v2;
                }
            }
            break;
        }
        case VIEW_OPT_SOFTLIGHT:
            if ((ret = view_adjust_parse_softlight(&st->adjustments, optarg)) < 0)
                return ret;
            if (!st->sections.adjust_forced)
                st->sections.adjust = true;
            break;
        case VIEW_OPT_VIBRANCE:
            if ((ret = view_adjust_parse_vibrance(&st->adjustments, optarg)) < 0)
                return ret;
            if (!st->sections.adjust_forced)
                st->sections.adjust = true;
            break;
        case VIEW_OPT_BYPASS:
            if ((ret = view_parse_bypass(&st->sections, optarg)) < 0)
                return ret;
            break;
        case VIEW_OPT_WB:
            if ((ret = view_parse_wb(st, optarg)) < 0)
                return ret;
            break;
        case VIEW_OPT_EI:
            st->look.ei = strtof(optarg, &end);
            if (end == optarg || *end || !(st->look.ei > 0.0f)) {
                fprintf(stderr, "--ei must be a positive exposure index\n");
                return CRAW_ERROR(EINVAL);
            }
            if (!st->sections.exposure_forced)
                st->sections.exposure = true;
            break;
        case VIEW_OPT_EXPOSURE:
            st->look.exposure = strtof(optarg, &end);
            if (end == optarg || *end) {
                fprintf(stderr, "--exposure: `%s` is not a number\n", optarg);
                return CRAW_ERROR(EINVAL);
            }
            if (!st->sections.exposure_forced)
                st->sections.exposure = true;
            break;
        case VIEW_OPT_DEBAYER:
            if ((ret = view_lookup(view_debayer_names, CRAW_DEBAYER_NB,
                                   "--debayer", optarg, &v)) < 0)
                return ret;
            st->look.debayer = (enum CRAWDebayer)v;
            break;
        case VIEW_OPT_FALSE_COLOUR: {
            long n = strtol(optarg, &end, 10);

            if (end == optarg || *end || n < 0 || n > CRAW_FALSE_COLOR_MAX_PASSES) {
                fprintf(stderr, "--false-colour: expected 0 to %d passes, got `%s`\n",
                        CRAW_FALSE_COLOR_MAX_PASSES, optarg);
                return CRAW_ERROR(EINVAL);
            }
            st->look.false_color = (int)n;
            break;
        }
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
        case VIEW_OPT_TONE_MAP:
            v = view_tone_map_find(optarg);
            if (v < 0) {
                fprintf(stderr, "--tone-map: `%s` is not one of libplacebo's "
                                "curves, expected one of", optarg);
                for (int i = 0; i < view_tone_map_count(); i++)
                    fprintf(stderr, "%s %s", i ? "," : "", view_tone_map_name(i));
                fprintf(stderr, "\n");
                return CRAW_ERROR(EINVAL);
            }
            st->look.tone_map = v;
            break;
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
        case VIEW_OPT_EXPORT_VIDEO:
            /* Three arguments, of which getopt can only express the first:
             * the other two come off the tail the way --export's do. */
            if (optind + 1 >= argc) {
                fprintf(stderr, "--export-video takes START END OUT\n");
                return CRAW_ERROR(EINVAL);
            }
            st->cfg.video_start = strtoll(optarg, &end, 10);
            if (end == optarg || *end || st->cfg.video_start < 0) {
                fprintf(stderr, "--export-video: `%s` is not a frame number\n",
                        optarg);
                return CRAW_ERROR(EINVAL);
            }
            st->cfg.video_end = strtoll(argv[optind], &end, 10);
            if (end == argv[optind] || *end ||
                st->cfg.video_end < st->cfg.video_start) {
                fprintf(stderr, "--export-video: `%s` is not a frame number at "
                                "or after %" PRId64 "\n",
                        argv[optind], st->cfg.video_start);
                return CRAW_ERROR(EINVAL);
            }
            snprintf(st->cfg.video_path, sizeof(st->cfg.video_path), "%s",
                     argv[optind + 1]);
            optind += 2;
            st->cfg.export_video = true;
            st->cfg.headless = true;
            break;
        case VIEW_OPT_LIST_ENCODERS:
            view_encode_list();
            return 1;
        case VIEW_OPT_SIZE:
            if ((ret = view_presets_parse_size(&st->look.video_preset, optarg)) < 0)
                return ret;
            break;
        case VIEW_OPT_AUDIO:
            if ((ret = view_presets_parse_audio(&st->look.video_preset,
                                                optarg)) < 0)
                return ret;
            break;
        case VIEW_OPT_THREADS:
            v = (int)strtol(optarg, &end, 10);
            if (end == optarg || *end || v < 0 || v > view_presets_max_threads()) {
                fprintf(stderr, "--threads: expected 0 (auto) or 1 to %d, "
                                "got `%s`\n", view_presets_max_threads(), optarg);
                return CRAW_ERROR(EINVAL);
            }
            st->look.video_preset.threads = v;
            break;
        case VIEW_OPT_CODEC:
            st->cfg.export_codec = optarg;
            break;
        case VIEW_OPT_ENC:
            st->cfg.export_enc = optarg;
            break;
        case VIEW_OPT_PRESET:
            if ((ret = view_presets_parse(&st->look.video_preset, optarg)) < 0)
                return ret;
            break;
        case VIEW_OPT_COLOUR:
            if ((ret = view_presets_parse_colour(&st->look.video_preset,
                                                 optarg)) < 0)
                return ret;
            break;
        case VIEW_OPT_PIX_FMT:
            if ((ret = view_presets_parse_fmt(&st->look.video_preset,
                                              optarg)) < 0)
                return ret;
            break;
        case VIEW_OPT_LIST_PRESETS:
            /* After the arguments, so that --device is honoured: the one
             * thing the table cannot say on its own is which Vulkan family
             * has no engine on this GPU. */
            st->cfg.list_presets = true;
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

    /* A listing wants no clip. */
    if (st->cfg.list_presets)
        return 0;

    /* Every remaining argument is a clip. getopt has moved them all to the
     * tail by now, including the ones --export and --export-video took off
     * it themselves, so what is left from optind on is the playlist in the
     * order it was typed. */
    if (optind >= argc) {
        fprintf(stderr, "no input file\n");
        view_usage(stderr);
        return CRAW_ERROR(EINVAL);
    }

    st->cfg.inputs = argv + optind;
    st->cfg.nb_inputs = argc - optind;
    st->cfg.input = st->cfg.inputs[0];

    return 0;
}

/* ------------------------------------------------------------ the profile */

/* `first` says this is the clip the command line opened, which is the only
 * one whose choices are worth arguing with: a later clip that brings no
 * per-CCT tables is no reason to throw away a white balance the user set
 * three clips ago, and view_frame_params() falls back to as shot by itself
 * on a profile that has none. */
static int view_profile_setup(struct view_state *st, bool first)
{
    const char *profile = st->cfg.profile, *tables = st->cfg.tables;
    int ret;

    craw_profile_init(&st->base);
    st->have_prof = false;

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

    if (first && st->look.wb != VIEW_WB_AS_SHOT && !st->base.nb_wb_knots) {
        fprintf(stderr, "warning: %s white balance needs the camera's per-CCT table "
                        "and none was found; staying as shot\n",
                st->look.wb == VIEW_WB_AUTO ? "auto" : "manual");
        st->look.wb = VIEW_WB_AS_SHOT;
    } else if (!first && st->look.wb != VIEW_WB_AS_SHOT && !st->base.nb_wb_knots) {
        view_log(st, "this clip has no per-CCT table, so it develops as shot; "
                     "the %s setting is kept for the next one that has\n",
                 st->look.wb == VIEW_WB_AUTO ? "auto" : "manual");
    }

    return 0;
}

/* ---------------------------------------------------------------- clips */

/* Everything the clip that is going away owns: both readers, the audio
 * device stream, the picture on screen and what it put on the device. None
 * of the user's settings is in here - not one of them is the clip's - so
 * whatever the panel and the command line have made of them is untouched.
 * What the clip itself said is read again by view_open_clip(). */
static void view_close_clip(struct view_state *st)
{
    /* The device stream is only destroyed after the ramp down has been
     * rendered, so a switch is a fade and never a cut. */
    if (st->audio.playing)
        view_audio_pause(&st->audio);

    /* The figure the end of a run prints is the session's, not the last
     * clip's. */
    st->underruns += st->audio.underruns;
    st->underrun_frames += st->audio.underrun_frames;

    view_audio_close(&st->audio);

    /* Before the decoder: the picture on screen is one of its frames, and
     * so is anything libplacebo still holds mapped. */
    av_frame_unref(st->frame);
    st->have_frame = false;
    view_gpu_release_frames(&st->gpu);

    view_decoder_close(&st->dec);
    /* view_decoder_open() fills what it needs and nothing else, so what the
     * last clip said - its model, its metadata, its V-Log flag, whether its
     * frames carried lens records - has to go here rather than linger into
     * the next one's panel. */
    st->dec = (struct view_decoder) { 0 };
}

/* One -v line for everything the user has set, printed as each clip opens.
 * A playlist's whole promise is that a switch changes the clip and nothing
 * else, and this is what makes that visible: the line is the same before
 * and after every switch, whatever the clips are. */
static void view_log_look(const struct view_state *st)
{
    const struct view_look *lk = &st->look;
    char adj[128], gr[64];

    if (!st->cfg.verbose)
        return;

    fprintf(stderr, "settings: wb ");
    switch (lk->wb) {
    case VIEW_WB_MANUAL:
        fprintf(stderr, "manual %.0f K", lk->manual.cct);
        if (lk->manual.tint != 0.0f)
            fprintf(stderr, " tint %+.3f", lk->manual.tint);
        break;
    case VIEW_WB_AUTO:   fprintf(stderr, "auto"); break;
    default:             fprintf(stderr, "as shot"); break;
    }
    /* Only a developed clip's balance has a switch. */
    if (!st->dec.raw && !st->sections.wb && lk->wb != VIEW_WB_AS_SHOT)
        fprintf(stderr, " (bypassed)");

    if (lk->ei > 0.0f)
        fprintf(stderr, ", ei %.0f", lk->ei);
    else
        fprintf(stderr, ", ei as shot");

    fprintf(stderr, ", exposure %+.2f", lk->exposure);
    /* The section switches say "kept but not applied", which is why the
     * numbers are printed either way: a bypassed section still carries the
     * settings it will come back with. */
    if (!st->sections.exposure)
        fprintf(stderr, " (bypassed)");

    fprintf(stderr, ", debayer %s",
            (unsigned)lk->debayer < CRAW_DEBAYER_NB
                ? view_debayer_names[lk->debayer] : "?");
    if (lk->false_color)
        fprintf(stderr, ", false colour x%d", lk->false_color);
    fprintf(stderr, ", highlights %s", lk->highlights ? "rebuilt" : "clipped");
    if (lk->defringe > 0.0f)
        fprintf(stderr, ", defringe %.2f", lk->defringe);
    else
        fprintf(stderr, ", defringe off");

    if (st->denoise.on && (st->denoise.strength > 0.0f || st->denoise.chroma > 0.0f ||
                           st->denoise.temporal > 0.0f))
        fprintf(stderr, ", denoise %.2f, chroma %.2f, temporal %.2f", st->denoise.strength,
                st->denoise.chroma, st->denoise.temporal);
    else
        fprintf(stderr, ", denoise off");

    fprintf(stderr, ", lens %s", st->lens.on ? st->lens.keep ? "on (keep all)"
                                                             : "on (fill)" : "off");
    if (st->sharpen.on)
        fprintf(stderr, ", sharpen %.2f x%d contrast %.2f", st->sharpen.radius,
                st->sharpen.iterations, st->sharpen.contrast);
    else
        fprintf(stderr, ", sharpen off");
    /* The export's sound and threads, which carry like every other export
     * setting. */
    fprintf(stderr, ", export audio %s, threads %d",
            view_presets_audio_label(&lk->video_preset, adj, sizeof(adj)),
            lk->video_preset.threads);
    fprintf(stderr, ", lut %s", lk->lut_path[0] ? lk->lut_path
                                                : view_lut_film_name(st));
    if (!st->sections.look)
        fprintf(stderr, " (bypassed)");
    /* The display's curve, which the Look section holds and every
     * display-referred export follows. */
    fprintf(stderr, ", tone map %s", view_tone_map_name(lk->tone_map));
    fprintf(stderr, ", adjustments %s",
            view_adjust_summary(&st->adjustments, adj, sizeof(adj)));
    if (!st->sections.adjust)
        fprintf(stderr, " (bypassed)");
    /* The gauge and the stock as the panel holds them, not the readout of
     * the last build: what carries across a clip is the setting. */
    fprintf(stderr, ", grain %s", view_grain_setting(&st->grain, gr, sizeof(gr)));
    fprintf(stderr, ", zoom %s", st->view.fit ? "fit" : "set");
    fprintf(stderr, ", volume %.0f%%%s\n", st->volume * 100.0f,
            st->mute ? ", muted" : "");
}

/* Opens one clip of the playlist and leaves it at frame 0. This is the only
 * path that opens a clip: startup is "open clip 0", and r and t are the
 * same call with another index. `first` is that startup one; see
 * view_profile_setup().
 *
 * The transport is not touched: a switch while paused stays paused, and one
 * while playing plays. */
static int view_open_clip(struct view_state *st, int index, bool first)
{
    int ret;

    st->clip = index;
    st->cfg.input = st->cfg.inputs[index];

    view_log(st, "clip %d/%d: %s\n", index + 1, st->cfg.nb_inputs, st->cfg.input);

    ret = view_decoder_open(&st->dec, st->cfg.input, &st->gpu, st->cfg.verbose);
    if (ret < 0)
        return ret;

    /* Now that the kind is known: a developed clip has no use for the raw
     * pipeline, and a raw one of the same size as the last keeps it. */
    view_gpu_new_clip(&st->gpu, st->dec.raw);

    view_crop_new_clip(st);

    /* A headless export has no transport to pace and no speakers to reach,
     * and --bench measures the picture pipeline: an audio track would pace
     * the loop at the clip's own rate and the draining at each loop point
     * would dominate a short clip's average. */
    if (!st->cfg.headless && !st->cfg.bench) {
        ret = view_audio_open(&st->audio, st->cfg.input, st->cfg.audio_map,
                              st->cfg.verbose);
        if (ret < 0)
            return ret;
    }

    /* Only where there is playback to dump. The file is the clip's, so a
     * switch starts it again rather than running two clips together. */
    if (!st->cfg.headless && st->cfg.audio_dump) {
        ret = view_audio_dump_open(&st->audio, st->cfg.audio_dump);
        if (ret < 0)
            return ret;
    }

    ret = view_profile_setup(st, first);
    if (ret < 0)
        return ret;

    /* The clip's own state, all of it read again from the clip that is now
     * open: the frame it is showing, the clock, the illuminant track, the
     * lens record and the two pass timings, the noise model, and whatever
     * an export on the last clip left in the panel. */
    st->index = 0;
    st->pts = 0.0;
    st->seek_to = -1;
    st->seek_target = -1;
    st->seek_tries = 0;
    st->seek_cold = false;
    st->scrubbing = false;
    st->need_frame = true;
    st->warm = false;
    st->loops_done = 0;
    st->starved = 0;
    st->decode_ms = 0.0;
    st->processed = (struct view_pipeline_look) { 0 };
    st->sync.run_frames = 0;
    st->timing = (struct view_timing) { 0 };

    craw_awb_smoother_reset(st->awb.smoother);
    st->awb.have_illum = false;
    st->vwb.gen++;
    st->vwb.converging = false;

    st->lens.record = (CRAWLensCorrection) { 0 };
    st->lens.scale = 0.0f;
    st->lens.ms_on = st->lens.ms_off = 0.0;
    st->denoise.have_model = false;
    st->denoise.ms_on = st->denoise.ms_off = 0.0;
    /* Another clip is a cut, whatever its first frame's number. */
    st->denoise.reset = true;
    st->denoise.have_last = false;

    st->clip_status[0] = '\0';
    st->exp.status[0] = '\0';
    st->video.status[0] = '\0';
    st->video.path[0] = '\0';
    st->video.progress = 0.0;
    /* The range follows the frame on screen again: the last clip's numbers
     * mean nothing on a clip of another length. */
    st->video.range_set = false;

    view_wall_restart(st, 0.0);

    /* A 1:1 zoom is about a point of the old frame, which the new one may
     * not have; fit needs nothing. */
    if (!st->view.fit) {
        st->view.cx = st->view.cy = 0.0f;
        view_log(st, "zoom: %.2fx kept, re-centred on the new clip\n", st->view.zoom);
    }

    view_log_look(st);

    return 0;
}

/* Can this path be opened and does it hold a video stream? Asked before the
 * clip on screen is touched, so that a name that is not there - a card that
 * was unplugged, a typo in the list - leaves the viewer exactly where it
 * was rather than on nothing at all. */
static bool view_clip_probe(const char *path, char *err, size_t n)
{
    AVFormatContext *fmt = NULL;
    const AVCodec *codec = NULL;
    int ret;

    ret = avformat_open_input(&fmt, path, NULL, NULL);
    if (ret < 0) {
        snprintf(err, n, "%s", av_err2str(ret));
        return false;
    }

    ret = avformat_find_stream_info(fmt, NULL);
    if (ret >= 0)
        ret = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);

    avformat_close_input(&fmt);

    if (ret < 0 || !codec) {
        snprintf(err, n, "no decodable video stream");
        return false;
    }

    return true;
}

/* r and t: the previous and the next clip, wrapping. */
static void view_switch_clip(struct view_state *st, int delta)
{
    const char *name;
    char err[128];
    int want, from;

    if (st->cfg.nb_inputs < 2)
        return;

    /* An export owns the decoder: it seeks it, it steps it, and its file is
     * half written. Refused rather than queued, because the one thing the
     * user would not want is the range they asked for silently coming from
     * two different clips. */
    if (st->exp.active || st->video.active) {
        snprintf(st->clip_status, sizeof(st->clip_status),
                 "export running: the clip changes when it ends");
        fprintf(stderr, "%s\n", st->clip_status);
        return;
    }

    want = (st->clip + delta + st->cfg.nb_inputs) % st->cfg.nb_inputs;
    name = st->cfg.inputs[want];

    if (!view_clip_probe(name, err, sizeof(err))) {
        snprintf(st->clip_status, sizeof(st->clip_status), "%s: %.80s",
                 strrchr(name, '/') ? strrchr(name, '/') + 1 : name, err);
        fprintf(stderr, "%s: %s\n", name, err);
        return;
    }

    from = st->clip;
    view_close_clip(st);

    if (view_open_clip(st, want, false) < 0) {
        /* It opened a moment ago as a probe and will not open now, so the
         * clip that was on screen comes back and the panel says why. The
         * message is written after that, because a clip that opens clears
         * it. If even the old one will not come back there is nothing left
         * to show and the run loop's next step says so. */
        fprintf(stderr, "%s: cannot be opened; staying on the previous clip\n", name);

        view_close_clip(st);
        if (view_open_clip(st, from, false) < 0) {
            view_close_clip(st);
            fprintf(stderr, "%s could not be re-opened either; there is no "
                            "clip left to show\n", st->cfg.inputs[from]);
        }

        snprintf(st->clip_status, sizeof(st->clip_status),
                 "%.60s could not be opened",
                 strrchr(name, '/') ? strrchr(name, '/') + 1 : name);
    }
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

/* A stretch of the clip in frames: five seconds for Left and Right, one
 * for Shift with them. Rounded, so that five seconds of 29.97 is 150
 * frames and not 149. */
static int64_t view_seek_frames(const struct view_state *st, double seconds)
{
    double fps = av_q2d(st->dec.frame_rate);

    return (int64_t)((fps > 0.0 ? fps : 25.0) * seconds + 0.5);
}

/* `frame` held to the clip: never past its last frame when the count is
 * known, never negative. What the marks, the seeks and the export's range
 * all go through, so that none of them can name a frame there is not. */
static int64_t view_last_frame(const struct view_state *st, int64_t frame)
{
    int64_t last = st->dec.nb_frames > 0 ? st->dec.nb_frames - 1 : INT64_MAX;

    return CRAW_CLAMP(frame, 0, last);
}

/* Seeks relative to the picture on screen without touching the transport:
 * a clip that was playing keeps playing from where it lands. */
static void view_seek_by(struct view_state *st, int64_t frames)
{
    st->seek_to = view_last_frame(st, st->index + frames);
    st->need_frame = true;
}

/* Space plays; Left and Right seek five seconds and Shift with them one;
 * Page Up and Page Down zoom and Shift with them goes to the ends of the
 * clip; Control with the arrows pans, which only does anything while the
 * picture is larger than its area; c turns crop mode on and off; r and t
 * walk the files the command line was given. The panel is the pointer's:
 * Up, Down and Enter do nothing at all, since there is no focus through
 * its rows for them to move. A key that toggles, seeks or changes clip
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

    /* A running video export owns the decoder and has copied the settings,
     * so there is nothing a key could change that it should: the panel is
     * locked, and the keyboard with it. What is left is q, which stops the
     * run and quits once the file has its trailer, and Escape closing a
     * menu, which is the one thing an open one has to be able to do. */
    if (view_ui_locked(st)) {
        if (e->key.key == SDLK_Q && !repeat) {
            view_encode_cancel(st);
            st->quit_pending = true;
        } else if (e->key.key == SDLK_ESCAPE) {
            view_ui_menu_close(st);
            view_pick_arm(st, false);
        }
        return;
    }

    switch (e->key.key) {
    case SDLK_SPACE:
        if (!repeat)
            st->playing = !st->playing;
        break;

    case SDLK_UP:
        if (ctrl)
            view_pan_fraction(st, 0.0f, -0.125f);
        break;
    case SDLK_DOWN:
        if (ctrl)
            view_pan_fraction(st, 0.0f, 0.125f);
        break;

    case SDLK_LEFT:
        if (ctrl)
            view_pan_fraction(st, -0.125f, 0.0f);
        else
            view_seek_by(st, -view_seek_frames(st, shift ? 1.0 : 5.0));
        break;
    case SDLK_RIGHT:
        if (ctrl)
            view_pan_fraction(st, 0.125f, 0.0f);
        else
            view_seek_by(st, view_seek_frames(st, shift ? 1.0 : 5.0));
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

    case SDLK_TAB:
    case SDLK_0:
        if (!repeat)
            view_zoom_reset(st, false);
        break;
    case SDLK_1:
        /* Ctrl+1 marks the export's start at the frame on screen; plain 1
         * is the 1:1 zoom. The end follows if it was before the new
         * start, so the range never turns inside out. */
        if (ctrl) {
            if (!repeat) {
                st->video.start_ui = (int)view_last_frame(st, st->index);
                if (st->video.end_ui < st->video.start_ui)
                    st->video.end_ui = st->video.start_ui;
                st->video.range_set = true;
            }
        } else if (!repeat) {
            view_zoom_reset(st, true);
        }
        break;
    case SDLK_2:
        /* Ctrl+2 marks the export's end at the frame on screen. */
        if (ctrl && !repeat) {
            st->video.end_ui = (int)view_last_frame(st, st->index);
            if (st->video.start_ui > st->video.end_ui)
                st->video.start_ui = st->video.end_ui;
            st->video.range_set = true;
        }
        break;

    case SDLK_C:
        /* Crop mode: a left drag draws the rectangle instead of panning.
         * Switching the mode on clears whatever rectangle there was, so the
         * whole frame is back and a new one can be drawn; switching it off
         * keeps the one that was drawn, shaded, for the export. The Crop
         * header's switch is this same switch and does the same. */
        if (!repeat) {
            st->crop.on = !st->crop.on;
            if (st->crop.on)
                view_crop_clear(st);
        }
        break;

    case SDLK_M:
        if (!repeat) {
            st->mute = !st->mute;
            st->mute_flag = st->mute;
        }
        break;

    /* The playlist, one clip per press: holding the key down would tear
     * through a card's worth of files, so the repeats are ignored the way
     * every other toggle's are. */
    case SDLK_R:
        if (!repeat)
            view_switch_clip(st, -1);
        break;
    case SDLK_T:
        if (!repeat)
            view_switch_clip(st, 1);
        break;

    case SDLK_ESCAPE:
        /* An open cascade is what Escape closes, and an armed pick is what
         * it disarms; with neither it quits, as q does. */
        if (st->menu.open) {
            view_ui_menu_close(st);
            break;
        }
        if (st->pick.armed) {
            view_pick_arm(st, false);
            break;
        }
        *running = false;
        break;
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
    /* A picture that has been developed but not yet drawn: the a/v figures
     * belong to the moment it reached the screen, which need not be the
     * iteration that decoded it. */
    bool stepped_pending = false;
    /* Whether the panel has consumed nuklear's input since it was last
     * opened. nk_input_begin() forgets the clicks and the deltas of the
     * cycle before it, so it may only run once the panel has been built
     * with them: an iteration that holds the picture back and builds
     * nothing must leave the cycle open, or a click that lands during the
     * hold - which while playing is nearly always - is wiped before any
     * button sees it. */
    bool ui_input_consumed = true;
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
    win = SDL_CreateWindow("cineview", w, h,
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

    /* The window's colour (DESIGN.md 22): on Wayland, the compositor's
     * preferred image description, rendered into and tagged. It needs the
     * swapchain to exist, because it switches it to pass-through, which
     * recreates it at the size it had; sized again here all the same, so
     * that the image is the window's own size whatever happened. Not
     * being able to is not an error: the window is then untagged sRGB, as
     * it is on X11 and with --no-colour-management. */
    if (!st->cfg.no_colour_management) {
        ret = view_colour_open(&st->gpu, win, st->cfg.verbose);
        if (ret < 0)
            return ret;
        SDL_GetWindowSizeInPixels(win, &w, &h);
        if (view_colour_active(&st->gpu) && !pl_swapchain_resize(st->gpu.sw, &w, &h)) {
            fprintf(stderr, "cannot size the swapchain to %dx%d\n", w, h);
            return CRAW_ERROR(EIO);
        }
    }

    /* Before the first clip: a LUT is a setting, and nothing about
     * resolving one - a file or a name out of the embedded pack - has
     * anything to do with which clip is open. */
    if (view_lut_startup(st) < 0)
        fprintf(stderr, "--lut %s: %s\n", st->look.lut_path, st->lut_error);

    /* Startup is nothing more than opening the first clip: r and t take
     * exactly this path with another index. */
    ret = view_open_clip(st, 0, true);
    if (ret < 0)
        return ret;

    ret = view_ui_open(st, win);
    if (ret < 0)
        return ret;

    if (st->cfg.trace) {
        const SDL_DisplayMode *mode =
            SDL_GetCurrentDisplayMode(SDL_GetDisplayForWindow(win));

        ret = view_trace_open(st, st->cfg.trace,
                              mode ? (double)mode->refresh_rate : 0.0);
        if (ret < 0)
            return ret;
    }

    fps_t0 = av_gettime_relative();
    iter_t0 = fps_t0;
    ret = 0;

    while (running) {
        bool was_playing = st->playing;
        bool stepped = false;
        bool events_seen = false;
        bool presented = false;
        double hold;
        SDL_Event e;
        int64_t now;

        view_trace_iter(st, av_gettime_relative());

        /* Whatever the compositor has said about the window's colour since
         * the last iteration - SDL's own reads of the socket queue it - is
         * acted on here, without waiting for anything. A new target colour
         * is a reason to draw: the picture pass's signature has the target
         * in it, so the picture is rendered again against it, and the frame
         * that carries it is the commit the new tag lands with. */
        if (view_colour_dispatch(&st->gpu))
            st->present_due = true;

        if (ui_input_consumed) {
            view_ui_input_begin();
            ui_input_consumed = false;
        }
        while (SDL_PollEvent(&e)) {
            bool to_nuklear = true;
            bool locked = view_ui_locked(st);

            /* Any event at all is a reason to draw: a key or a drag can
             * move the picture, the look or the panel, and chasing down
             * which ones do would only be a list to get wrong. */
            events_seen = true;

            switch (e.type) {
            case SDL_EVENT_QUIT:
                /* A window closed on a running export stops the run and
                 * leaves once the file has its trailer, the way q does. */
                if (view_ui_locked(st)) {
                    view_encode_cancel(st);
                    st->quit_pending = true;
                } else {
                    running = false;
                }
                break;
            case SDL_EVENT_KEY_DOWN:
                /* Let the text field have the keys while it is being typed
                 * into; nuklear tells us so through its own input state.
                 * Everything else reaches the picture whatever the pointer
                 * happens to be hovering. */
                if (!view_ui_wants_keys(st))
                    view_handle_key(st, &e, &running);
                break;

            /* The pointer over the picture zooms, pans and draws the crop -
             * except while a video export runs, when the picture is the
             * export's and the pointer's only business is the panel's stop
             * button. The events still reach nuklear, whose widgets are all
             * disabled but that one. */

            /* SDL3 puts the pointer position in the wheel event itself, so
             * zooming about the cursor no longer has to ask for the mouse
             * state and risk a position from after the notch. */
            case SDL_EVENT_MOUSE_WHEEL:
                if (!locked &&
                    view_over_picture(st, e.wheel.mouse_x, e.wheel.mouse_y)) {
                    int px, py;

                    view_mouse_pos(st, e.wheel.mouse_x, e.wheel.mouse_y, &px, &py);
                    view_zoom_by(st, e.wheel.y, px, py, true);
                    to_nuklear = false;
                }
                break;

            case SDL_EVENT_MOUSE_BUTTON_DOWN:
                if (locked)
                    break;
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
                    /* An armed pick takes the click and nothing else does:
                     * no pan, no rectangle, and crop mode stays whatever it
                     * was. Otherwise in crop mode the drag draws a
                     * rectangle, and otherwise it pans, which is what it
                     * has always done. */
                    if (st->pick.armed) {
                        int px, py;

                        view_mouse_pos(st, e.button.x, e.button.y, &px, &py);
                        view_pick_click(st, px, py);
                    } else if (st->crop.on) {
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
                if (!locked && e.button.button == SDL_BUTTON_LEFT &&
                    (st->view.dragging || st->crop.dragging)) {
                    st->view.dragging = false;
                    view_crop_end(st);
                    to_nuklear = false;
                }
                break;
            case SDL_EVENT_MOUSE_MOTION:
                if (locked) {
                    /* A drag that was in flight when the run began ends
                     * here rather than resuming a frame later on a picture
                     * that has moved. */
                    st->view.dragging = false;
                    view_crop_end(st);
                } else if (st->crop.dragging) {
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

        /* Keeps the audio ring topped up whatever the picture is doing.
         * An export owns the decoder while it runs, so the clock and the
         * ring are left alone until it is done. */
        if (!st->exp.active && !st->video.active &&
            view_audio_pump(&st->audio) < 0)
            break;

        if (st->video.active) {
            view_encode_step(st);
            /* q, or the window closing, during the run: the file has its
             * trailer now, so this is where the program leaves. */
            if (st->quit_pending && !st->video.active)
                running = false;
        } else if (st->exp.active) {
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
                if (st->trace.on) {
                    st->trace.cur.t_step = av_gettime_relative();
                    st->trace.cur.t_pkt = st->dec.frame_arrive;
                }
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

        /* A pick that is waiting - a click that landed, or --wb pick:X,Y
         * until the first picture - is taken here, once a picture is held
         * and nothing is about to replace it: it puts that picture through
         * the pipeline twice and reads the device back, which is not a
         * thing to do from inside the event handler or under an export.
         * The click has already paused the transport, so what it is taken
         * on is the picture on screen. */
        if (st->pick.pending && !st->exp.active && !st->video.active &&
            st->have_frame && !st->need_frame && st->seek_to < 0) {
            ret = view_pick_run(st);
            if (ret < 0)
                break;
            st->present_due = true;
        }

        /* While paused, a developed clip's auto white balance reads the
         * held picture again for as long as the track keeps moving; while
         * playing, each new picture's step does it. */
        if (!st->playing && !st->exp.active && !st->video.active &&
            st->have_frame)
            view_vwb_step(st, false);

        now = av_gettime_relative();
        /* What would make the window look different from what is on it.
         * The panel's own readouts move with the picture, so they need no
         * test of their own; the idle tick is what keeps the ones that move
         * with nothing - the frame rate averages - alive while paused. */
        if (stepped || events_seen || !st->warm || st->gpu.pic_redrew ||
            st->exp.active || st->video.active || st->need_frame ||
            st->seek_to >= 0 || now - st->last_present >= VIEW_IDLE_PRESENT)
            st->present_due = true;

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

        if (stepped)
            stepped_pending = true;

        /* The picture pass runs here rather than inside the present: it is
         * where the milliseconds are, and on the present path they would
         * land in the interval between one picture and the next. By the
         * time the picture is due, the texture already holds it and the
         * present is a copy, a panel blend and a swap. */
        if (st->present_due && !st->exp.active && !st->video.active)
            view_gpu_prepare(&st->gpu, st);

        /* A developed picture is held until its own time on the clip's
         * timeline. Presenting it the moment it finished developing puts
         * the decode's own variation - two to sixteen milliseconds at 5.8K
         * - straight into the interval between one picture and the next,
         * and that interval is what judder is. Everything else the window
         * draws is wanted at once. */
        hold = st->present_due && stepped_pending && st->playing && st->warm &&
               !st->cfg.bench && !st->exp.active && !st->video.active
             ? st->pts - view_clock(st) : 0.0;

        if (st->present_due && hold <= 0.0) {
            view_ui_build(st);
            ui_input_consumed = true;

            /* A grade change while paused replays the picture on screen;
             * while playing the next frame simply uses the current look,
             * and so does a decode that is already pending. */
            if (!st->playing && !st->exp.active && st->have_frame &&
                !st->need_frame && st->seek_to < 0) {
                struct view_pipeline_look look = view_pipeline_look(st);

                if (!view_pipeline_look_equal(&look, &st->processed)) {
                    ret = view_replay_frame(st);
                    if (ret < 0)
                        break;
                }
            }

            ret = view_ui_render(st);
            if (ret < 0)
                break;

            if (st->trace.on) {
                view_gpu_poll_fence(&st->gpu);
                st->trace.cur.index = st->index;
                st->trace.cur.pts = st->pts;
                st->trace.cur.clock = view_clock(st);
                st->trace.cur.stepped = stepped_pending;
            }

            ret = view_gpu_display(&st->gpu, st);
            view_ui_rotate(&st->gpu);
            if (ret == CRAW_ERROR(EAGAIN)) {
                SDL_Delay(10);
                continue;
            }
            if (ret < 0)
                break;

            /* A picture's place against the clock is where it was when it
             * reached the screen, not where it was when it was decoded: the
             * decode now runs a whole frame ahead of the present. */
            if (stepped_pending) {
                double clock = view_clock(st);
                double diff = clock - st->pts;
                double mag = fabs(diff);

                st->sync.diff = diff;
                st->sync.presented++;
                st->sync.max_abs = CRAW_MAX(st->sync.max_abs, mag);
                /* The first frames of a run are ahead of the audio by up to
                 * one device buffer, because nothing has been heard yet,
                 * and the very first of them also pay for compiling
                 * shaders. */
                if (st->sync.run_frames > 10)
                    st->sync.max_abs_settled = CRAW_MAX(st->sync.max_abs_settled,
                                                        mag);

                view_log(st, "frame %5" PRId64 " pts %7.3f s  clock %7.3f s  "
                             "a/v %+7.2f ms  peak %6.0f / %6.0f nits\n",
                         st->index, st->pts, clock, diff * 1000.0,
                         st->gpu.measured_peak, view_peak_ceiling(st));
            }

            view_trace_push(st);
            st->present_due = false;
            st->last_present = av_gettime_relative();
            presented = true;
            stepped_pending = false;
        } else {
            /* Nothing would be different yet, so the loop waits rather than
             * drawing the same picture again. The wait ends early on any
             * input and never runs past the moment something is wanted:
             * either the held picture's own time, or the point at which the
             * next one has to start being decoded. The cap keeps the audio
             * ring topped up whatever the clip's frame rate is. */
            int ms = hold > 0.0 ? (int)(hold * 1000.0) : VIEW_IDLE_WAIT_MS;

            ms = CRAW_MIN(ms, VIEW_IDLE_WAIT_MS);
            if (ms > 0)
                SDL_WaitEventTimeout(NULL, ms);
        }

        if (!st->view.cx && !st->view.cy && st->gpu.width) {
            int dw, dh;

            /* The centre of the picture as it is shown, which for a rotated
             * clip is not the centre of the readout. */
            view_disp_size(st, &dw, &dh);
            st->view.cx = (float)dw * 0.5f;
            st->view.cy = (float)dh * 0.5f;
        }

        if (!st->warm && st->have_frame) {
            /* A picture has made it all the way to the screen, so every
             * shader either side of it is compiled. Start the clock here. */
            st->warm = true;
            view_wall_restart(st, st->pts);
            st->sync.run_frames = 0;
        }

        if (presented)
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

    view_trace_close(st);

    if (st->cfg.verbose || st->cfg.loops)
        fprintf(stderr, "%" PRId64 " passes: %" PRId64 " frames presented, "
                        "%" PRId64 " dropped, max |a/v| %.2f ms (%.2f ms once "
                        "settled), %" PRIu64 " audio underruns (%" PRIu64
                        " sample frames of silence)\n",
                st->loops_done, st->sync.presented, st->sync.dropped,
                st->sync.max_abs * 1000.0, st->sync.max_abs_settled * 1000.0,
                st->underruns + st->audio.underruns,
                st->underrun_frames + st->audio.underrun_frames);

    return ret > 0 ? 0 : ret;
}

/* Teardown in one place, and in the only order that works: the UI and
 * everything on the device first, then the VkSurfaceKHR (inside
 * view_gpu_close), and only then the window it belongs to. */
static void view_close(struct view_state *st)
{
    /* Before the pl_gpu: the video exporter holds textures on it. */
    view_encode_close(st);
    free(st->exp.rgba);
    pl_lut_free(&st->lut);
    pl_lut_free(&st->preview);
    av_frame_free(&st->frame);
    craw_awb_smoother_free(&st->awb.smoother);
    free(st->vwb.hist);

    view_ui_close(st);
    view_audio_close(&st->audio);
    view_decoder_close(&st->dec);
    /* Once per program, whatever the playlist did. */
    view_decoder_cleanup();
    /* The grain hooks hold shader objects, so they go before the pl_gpu. */
    view_grain_close(&st->grain);
    view_adjust_close(&st->adjustments);
    view_gpu_close(&st->gpu);

    /* The crosshair, if a pick was still armed, before the window it was
     * shown on. */
    view_pick_arm(st, false);
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

/* --wb pick:X,Y without a window: the pick is taken on the first frame the
 * run exports, before the export starts, so that the still or every frame
 * of the video is developed with what it found. The walk is the export's
 * own - one seek, then decode until the frame is the one wanted - and the
 * export then finds the picture already held and does not walk again. */
static int view_headless_pick(struct view_state *st, int64_t frame)
{
    int ret;

    st->seek_to = frame;
    while (1) {
        ret = view_step_frame(st);
        if (ret == 2 || ret == 3)
            continue;           /* dropped as stale, or the reader is behind */
        if (ret < 0)
            return ret;
        if (ret > 0) {
            fprintf(stderr, "--wb pick: the clip ends before frame %" PRId64 "\n",
                    frame);
            return CRAW_ERROR(ENOENT);
        }
        if (st->index == frame)
            break;
        if (st->index > frame) {
            fprintf(stderr, "--wb pick: the seek to frame %" PRId64 " landed on "
                            "%" PRId64 "\n", frame, st->index);
            return CRAW_ERROR(EIO);
        }
    }

    return view_pick_run(st);
}

static int view_run_headless(struct view_state *st)
{
    int64_t t0 = av_gettime_relative();
    int ret;

    ret = view_gpu_open(&st->gpu, &st->cfg, NULL);
    if (ret < 0)
        return ret;
    t0 = view_phase(st, t0, "device and pipeline");

    ret = view_lut_startup(st);
    if (ret < 0) {
        fprintf(stderr, "--lut %s: %s\n", st->look.lut_path, st->lut_error);
        return ret;
    }
    /* Everything the command line asked for has been applied; from here on
     * a touched parameter is the user's, and switches its section on even
     * where --bypass had pinned it off. */
    st->sections.exposure_forced = st->sections.look_forced =
        st->sections.adjust_forced = false;

    /* The first file and only the first: an export writes one clip to one
     * file, and a playlist is a thing the window has. */
    ret = view_open_clip(st, 0, true);
    if (ret < 0)
        return ret;
    t0 = view_phase(st, t0, "decoder, profile and LUT");

    if (st->pick.pending) {
        ret = view_headless_pick(st, st->cfg.export_video ? st->cfg.video_start
                                                          : st->cfg.export_frame);
        if (ret < 0)
            return ret;
        t0 = view_phase(st, t0, "spot white balance");
    }

    if (st->cfg.export_video) {
        struct view_encode_opts o = {
            .start  = st->cfg.video_start,
            .end    = st->cfg.video_end,
            .path   = st->cfg.video_path,
            .sel    = &st->look.video_preset,
            .codec  = st->cfg.export_codec,
            .enc    = st->cfg.export_enc,
        };

        ret = view_encode_begin(st, &o);
        if (ret < 0)
            return ret;

        while (st->video.active) {
            ret = view_encode_step(st);
            if (ret < 0)
                return ret;
        }

        view_phase(st, t0, "seek, decode, render and encode");
        fprintf(stderr, "%s\n", st->video.status);

        return 0;
    }

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
        /* On: rebuilding what clipped is a correction, like the lens, not
         * a look. */
        .highlights = 1,
        /* On with it: the rims it takes out are the ones the rebuilt
         * highlights stop hiding. */
        .defringe  = 1.0f,
        .lut_type  = PL_LUT_NATIVE,
        .film_sim  = -1,
        /* libplacebo's own default curve, spline; --tone-map overrides. */
        .tone_map  = view_tone_map_default(),
    };
    /* Before the arguments, so that --preset, --colour, --size and
     * --pix-fmt all override it. */
    view_presets_default(&st.look.video_preset);
    for (int i = 0; i < VIEW_MENU_COLS; i++)
        st.menu.sel[i] = -1;
    st.preview_sim = -1;
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
    st.denoise.chroma = 1.0f;
    /* On by default: a frame that carries a record is a frame the camera
     * expects to be corrected, and one that does not is untouched either
     * way. --lens off turns it off. */
    st.lens.on = true;
    /* And filling the frame by default, because that is the framing the
     * camera chose and the one every other decoder shows. --lens-fit keep,
     * or the panel's row, keeps the corners instead. */
    st.lens.keep = false;
    st.lens.keep_sel = 0;
    /* Off, as in every raw converter, at the numbers the switch turns it
     * on with: this camera's in-focus blur, the full count, and a mask at
     * the modelled noise. DESIGN.md 20.4. */
    st.sharpen = (struct view_sharpen) {
        .radius     = 0.8f,
        .iterations = 20,
        .contrast   = 1.0f,
    };
    /* The three section switches that are nothing but a bypass start off,
     * as RawTherapee's tools do: their parameters are neutral, so off and
     * on develop the clip the same way, and a lit switch then only ever
     * means the section is doing something. Touching a parameter - from an
     * option, the panel or the keyboard - switches the section on. */
    st.sections.exposure = false;
    st.sections.look = false;
    st.sections.adjust = false;
    view_grain_defaults(&st.grain);
    view_adjust_defaults(&st.adjustments);

#if CINEVIEW_HAVE_LUTS
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

    /* --list-presets, with the device open so that the families whose
     * engine is a Vulkan extension can say whether this GPU has it. A
     * device that will not open leaves the table to say what the build
     * says. */
    if (st.cfg.list_presets) {
        if (view_gpu_open(&st.gpu, &st.cfg, NULL) < 0)
            fprintf(stderr, "no Vulkan device: the families are listed as "
                            "this build has them\n");
        view_presets_list();
        view_gpu_close(&st.gpu);
        return 0;
    }

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
