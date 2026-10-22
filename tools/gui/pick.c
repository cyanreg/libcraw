/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* The spot white balance: click on something white, get the illuminant.
 *
 * The click is a source pixel; the patch of VIEW_PICK_SIZE around it is
 * read back out of the pipeline's own output rather than out of the frame,
 * because that is the one image the viewer already has developed: the
 * mosaic is on the device inside the decoder's frame, and reading the
 * output back is a rectangle copy the display slot's texture allows.
 *
 * What is read is ACEScg from the colour stage of DESIGN.md 4.3, developed
 * with the illuminant the frame was on screen with and, for this one pass,
 * without the reference gamut compression of 8.1: a strongly cast patch
 * sits well outside AP1 and the compression would move it, so the pass is
 * repeated uncompressed and the picture is put back afterwards. The colour
 * stage is then undone in the order it was done:
 *
 *     xyz = craw_colorspace_to_xyz_d65(AP1) * out / (gain * 2^stops)
 *     c   = cam_to_xyz^-1 * xyz              balanced camera RGB, [0, 1]
 *     raw = c / wb                           the sensor's own ratios
 *
 * with cam_to_xyz, wb and gain what craw_profile_eval() gives for the
 * frame's mode and illuminant, which is exactly what craw_pipeline_record()
 * composed the pass from. The scale cancels in the ratios but is divided
 * out anyway so that `c` is in the units the prep's clip is in, which is
 * where the thresholds are taken. The ratios of the patch's mean raw R and
 * B to its mean raw G are then a neutral's response, and
 * craw_profile_neutral_to_illuminant() finds the point of the profile's
 * locus nearest them, which is the illuminant that makes the patch neutral.
 *
 * The lens warp, when it ran, resampled the output before it was read;
 * that is a linear operation on a flat patch and changes nothing about a
 * neutral. The denoise ran before the debayer and is in the values. */

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "view.h"

/* A balanced channel at or above this is within 2% of the clip at 1.0,
 * which is the prep's own clip: the sensor's, or the balanced one that
 * keeps blown highlights neutral. The auto white balance histogram uses the
 * same number (DESIGN.md 4.4). */
#define VIEW_PICK_CLIP 0.98

/* Too dark is decided by the camera's noise model where it has one: a raw
 * channel below the level at which the modelled signal is only this many
 * times its own standard deviation is noise wearing a ratio. Ten puts the
 * ratio of two such channels at about 14% per pixel and 1.2% over the
 * quarter of a patch that must survive, which is under the locus's own
 * resolution. Without a model the auto estimator's floor is used instead. */
#define VIEW_PICK_SNR   10.0
#define VIEW_PICK_FLOOR 0.002

/* Fewer than this fraction of the patch surviving is a refusal. */
#define VIEW_PICK_MIN_FRACTION 4

/* What one read of the patch amounts to. */
struct view_pick_sample {
    int x0, y0;                 /* the patch's corner, in source pixels */
    int total, kept, clipped, dark;
    double raw[3];              /* mean raw camera RGB of the kept pixels */
    double ap1[3];              /* mean ACEScg of the kept pixels, as read */
    double floor;               /* the dark threshold that was used */
};

/* IEEE half to float, the way cineraw reads the same output: the exponent
 * re-biased, the mantissa shifted, subnormals scaled exactly. */
static float view_half_to_float(uint16_t h)
{
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    uint32_t bits;
    float f;

    if (e == 0) {
        f = (float)m * 5.9604644775390625e-8f;
        return (h & 0x8000u) ? -f : f;
    }

    bits = (uint32_t)(h & 0x8000u) << 16;
    bits |= e == 31 ? (0x7F800000u | (m << 13)) : (((e + 112u) << 23) | (m << 13));
    memcpy(&f, &bits, sizeof(f));

    return f;
}

/* The adjugate over the determinant, in double: the camera matrices are
 * well conditioned but a 3x3 inverse in float is a needless place to lose
 * a digit. False when the matrix is singular. */
static bool view_mat3_invert(const float m[3][3], double out[3][3])
{
    double a[3][3], det;

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            a[i][j] = m[i][j];

    det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if (!(fabs(det) > 1e-12))
        return false;

    out[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / det;
    out[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / det;
    out[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / det;
    out[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / det;
    out[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / det;
    out[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / det;
    out[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / det;
    out[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / det;
    out[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / det;

    return true;
}

/* Where "too dark" starts, in the raw's normalized units: the signal `u`
 * at which u = SNR * sqrt(a u + b), the larger root of the quadratic. */
static double view_pick_floor(const struct view_state *st)
{
    CRAWNoiseModel nm;
    double s2 = VIEW_PICK_SNR * VIEW_PICK_SNR;

    if (!view_denoise_model(st, &nm, NULL))
        return VIEW_PICK_FLOOR;

    return 0.5 * (s2 * nm.a + sqrt(s2 * s2 * (double)nm.a * nm.a + 4.0 * s2 * nm.b));
}

/* Puts the held frame through uncompressed, reads the patch around (x, y)
 * and undoes the colour stage on it. Leaves the picture on screen
 * uncompressed; the caller puts it back. */
static int view_pick_sample(struct view_state *st, int x, int y,
                            struct view_pick_sample *s)
{
    CRAWFrameParams fp;
    CRAWColorParams cp;
    float to_xyz[3][3];
    double inv[3][3], scale, sum_raw[3] = { 0 }, sum_ap1[3] = { 0 };
    uint16_t half[VIEW_PICK_SIZE * VIEW_PICK_SIZE * 4];
    int ret;

    /* The parameters this pass is developed with, taken before it: the
     * auto white balance step at the end of a pass may move the track,
     * and what has to be undone is the illuminant the patch was read
     * under, not the one the next picture would get. */
    view_frame_params(st, &fp);

    /* Uncompressed, so that a patch outside AP1 reads as the colour stage
     * produced it. The viewer never sets CRAW_FRAME_FLAG_CLAMP_NEGATIVE,
     * so nothing below zero is lost either. */
    ret = view_replay_frame_flags(st, CRAW_FRAME_FLAG_NO_GAMUT_COMPRESSION);
    if (ret < 0)
        return ret;

    ret = craw_profile_eval(&st->prof, fp.wb_mode, &fp.illuminant, &cp);
    if (ret < 0)
        return ret;
    ret = craw_colorspace_to_xyz_d65(CRAW_CS_ACES_AP1_LINEAR, to_xyz);
    if (ret < 0)
        return ret;
    if (!view_mat3_invert(cp.cam_to_xyz, inv))
        return CRAW_ERROR(EINVAL);
    for (int c = 0; c < 3; c++)
        if (!(cp.wb[c] > 0.0f))
            return CRAW_ERROR(EINVAL);

    scale = (double)cp.gain * exp2((double)fp.exposure_stops);
    if (!(scale > 0.0))
        return CRAW_ERROR(EINVAL);

    *s = (struct view_pick_sample) {
        .x0    = CRAW_CLAMP(x - VIEW_PICK_SIZE / 2, 0,
                            (int)st->gpu.width - VIEW_PICK_SIZE),
        .y0    = CRAW_CLAMP(y - VIEW_PICK_SIZE / 2, 0,
                            (int)st->gpu.height - VIEW_PICK_SIZE),
        .total = VIEW_PICK_SIZE * VIEW_PICK_SIZE,
        .floor = view_pick_floor(st),
    };

    ret = view_gpu_read_region(&st->gpu, s->x0, s->y0, VIEW_PICK_SIZE,
                               VIEW_PICK_SIZE, half);
    if (ret < 0)
        return ret;

    for (int i = 0; i < s->total; i++) {
        double out[3], xyz[3], c[3], raw[3], peak = -INFINITY, low = INFINITY;

        for (int k = 0; k < 3; k++)
            out[k] = view_half_to_float(half[4 * i + k]);

        for (int r = 0; r < 3; r++) {
            xyz[r] = (to_xyz[r][0] * out[0] + to_xyz[r][1] * out[1] +
                      to_xyz[r][2] * out[2]) / scale;
        }
        for (int r = 0; r < 3; r++) {
            c[r] = inv[r][0] * xyz[0] + inv[r][1] * xyz[1] + inv[r][2] * xyz[2];
            peak = CRAW_MAX(peak, c[r]);
        }

        /* The clip test is on the balanced values, where the prep clipped;
         * the dark test on the raw ones, where the noise model lives. */
        if (!(peak < VIEW_PICK_CLIP)) {
            s->clipped++;
            continue;
        }
        for (int r = 0; r < 3; r++) {
            raw[r] = c[r] / cp.wb[r];
            low = CRAW_MIN(low, raw[r]);
        }
        if (!(low >= s->floor)) {
            s->dark++;
            continue;
        }

        for (int r = 0; r < 3; r++) {
            sum_raw[r] += raw[r];
            sum_ap1[r] += out[r];
        }
        s->kept++;
    }

    if (s->kept) {
        for (int r = 0; r < 3; r++) {
            s->raw[r] = sum_raw[r] / s->kept;
            s->ap1[r] = sum_ap1[r] / s->kept;
        }
    }

    return 0;
}

void view_pick_arm(struct view_state *st, bool on)
{
    if (on == st->pick.armed)
        return;

    st->pick.armed = on;
    view_log(st, "pick: %s\n", on ? "armed, the next click on the picture samples"
                                  : "disarmed");
    if (on) {
        /* The last verdict is about a pick that is now history. */
        st->pick.refused[0] = '\0';
        if (st->win) {
            st->pick.cursor = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_CROSSHAIR);
            if (!st->pick.cursor || !SDL_SetCursor(st->pick.cursor))
                view_log(st, "pick: no crosshair cursor: %s\n", SDL_GetError());
        }
        return;
    }

    if (st->pick.cursor) {
        SDL_SetCursor(SDL_GetDefaultCursor());
        SDL_DestroyCursor(st->pick.cursor);
        st->pick.cursor = NULL;
    }
}

void view_pick_at(struct view_state *st, int x, int y)
{
    st->pick.x = x;
    st->pick.y = y;
    st->pick.pending = true;
}

void view_pick_click(struct view_state *st, int px, int py)
{
    pl_rect2df disp, src;
    float sx, sy;
    int dx, dy, dw, dh;

    if (!st->gpu.width || !st->gpu.height)
        return;

    /* Through the same map as the crop's rubber band, then turned back to
     * the source the way the crop is: a point is a rectangle with no
     * extent, and the turn is exact. A click on the letterbox around the
     * picture is not on the picture, and the pick stays armed for one that
     * is. */
    view_picture_pos(st, px, py, &sx, &sy);
    view_disp_rect(st, &dx, &dy, &dw, &dh);
    if (sx < (float)dx || sy < (float)dy ||
        sx >= (float)(dx + dw) || sy >= (float)(dy + dh))
        return;

    disp = (pl_rect2df) { sx, sy, sx, sy };
    view_rot_to_src(st, &disp, &src);

    view_pick_at(st,
                 CRAW_CLAMP((int)floorf(src.x0), 0, (int)st->gpu.width - 1),
                 CRAW_CLAMP((int)floorf(src.y0), 0, (int)st->gpu.height - 1));
    /* One click, one pick, on this frame. */
    view_pick_arm(st, false);
    st->playing = false;
}

/* The spot white balance of a developed clip. There is no colour stage to
 * undo: the patch is read back as the linear light libplacebo's LINEAR
 * stage holds, in the clip's own primaries and before the correction, and
 * mapped onto the clip's own locus (videowb.c), which lives in exactly
 * those numbers. The clip test is against the source's peak, where its
 * channels clip; the dark test is the floor without a noise model, in the
 * same units, where 1.0 is diffuse white. */
static int view_pick_run_video(struct view_state *st)
{
    float rgba[VIEW_PICK_SIZE * VIEW_PICK_SIZE * 4], peak;
    const int total = VIEW_PICK_SIZE * VIEW_PICK_SIZE;
    int x0, y0, kept = 0, clipped = 0, dark = 0, ret;
    enum pl_color_primaries prim;
    double sum[3] = { 0 }, mean[3];
    CRAWIlluminant illum;
    CRAWProfile p;

    if (!st->have_frame || !view_gpu_source_space(&st->gpu, &prim, &peak) ||
        !view_vwb_profile(st, prim, &p)) {
        fprintf(stderr, "--wb pick: there is no picture to pick on\n");
        snprintf(st->pick.refused, sizeof(st->pick.refused), "no picture");
        return 0;
    }

    x0 = CRAW_CLAMP(st->pick.x - VIEW_PICK_SIZE / 2, 0,
                    (int)st->gpu.width - VIEW_PICK_SIZE);
    y0 = CRAW_CLAMP(st->pick.y - VIEW_PICK_SIZE / 2, 0,
                    (int)st->gpu.height - VIEW_PICK_SIZE);

    ret = view_gpu_read_linear(&st->gpu, x0, y0, VIEW_PICK_SIZE, VIEW_PICK_SIZE,
                               rgba);
    if (ret < 0)
        return ret;

    for (int i = 0; i < total; i++) {
        const float *px = &rgba[4 * i];
        float hi = CRAW_MAX(CRAW_MAX(px[0], px[1]), px[2]);
        float lo = CRAW_MIN(CRAW_MIN(px[0], px[1]), px[2]);

        if (!(hi < VIEW_PICK_CLIP * peak)) {
            clipped++;
            continue;
        }
        if (!(lo >= VIEW_PICK_FLOOR)) {
            dark++;
            continue;
        }
        for (int c = 0; c < 3; c++)
            sum[c] += px[c];
        kept++;
    }

    if (kept * VIEW_PICK_MIN_FRACTION < total) {
        snprintf(st->pick.refused, sizeof(st->pick.refused), "%s",
                 clipped >= dark ? "clipped" : "too dark");
        view_log(st, "pick: frame %" PRId64 " at %d,%d, patch %dx%d at %d,%d: "
                     "refused, %s (%d of %d kept: %d clipped at %.3f, %d below "
                     "%.5f)\n",
                 st->index, st->pick.x, st->pick.y, VIEW_PICK_SIZE,
                 VIEW_PICK_SIZE, x0, y0, st->pick.refused, kept, total, clipped,
                 VIEW_PICK_CLIP * peak, dark, VIEW_PICK_FLOOR);
        return 0;
    }

    for (int c = 0; c < 3; c++)
        mean[c] = sum[c] / kept;

    ret = craw_profile_neutral_to_illuminant(&p, (float)(mean[0] / mean[1]),
                                             (float)(mean[2] / mean[1]), &illum);
    if (ret < 0) {
        fprintf(stderr, "craw_profile_neutral_to_illuminant failed: %d\n", ret);
        snprintf(st->pick.refused, sizeof(st->pick.refused), "no locus");
        return 0;
    }

    view_log(st, "pick: frame %" PRId64 " at %d,%d, patch %dx%d at %d,%d: "
                 "%d of %d kept (%d clipped, %d dark), linear R/G %.4f B/G %.4f "
                 "-> %.0f K tint %+.4f (recorded %d K)\n",
             st->index, st->pick.x, st->pick.y, VIEW_PICK_SIZE, VIEW_PICK_SIZE,
             x0, y0, kept, total, clipped, dark, mean[0] / mean[1],
             mean[2] / mean[1], illum.cct, illum.tint, st->dec.wb_kelvin);

    st->look.manual = illum;
    st->look.wb = VIEW_WB_MANUAL;
    st->sections.wb = true;

    /* The check, with -v: the patch under the correction that was found
     * has to come out neutral. It is the matrix the LINEAR stage is given,
     * on the mean that was read, so what it checks is the locus mapping and
     * its inverse, not the render. */
    if (st->cfg.verbose) {
        pl_matrix3x3 m;
        float v[3] = { (float)mean[0], (float)mean[1], (float)mean[2] };

        if (view_vwb_correction(st, prim, &illum, &m)) {
            pl_matrix3x3_apply(&m, v);
            view_log(st, "pick: check: under %.0f K tint %+.4f the patch reads "
                         "R/G %.4f B/G %.4f (%+.2f%% / %+.2f%%)\n",
                     illum.cct, illum.tint, v[0] / v[1], v[2] / v[1],
                     (v[0] / v[1] - 1.0f) * 100.0f, (v[2] / v[1] - 1.0f) * 100.0f);
        }
    }

    return 0;
}

int view_pick_run(struct view_state *st)
{
    struct view_pick_sample s;
    CRAWIlluminant illum;
    int ret;

    st->pick.pending = false;
    st->pick.refused[0] = '\0';

    if (!st->dec.raw)
        return view_pick_run_video(st);

    /* Nothing to pick on yet. From the command line it is worth a line. */
    if (!st->have_frame || !st->have_prof ||
        !st->frame->buf[0] || st->frame->format != AV_PIX_FMT_VULKAN) {
        fprintf(stderr, "--wb pick: there is no picture to pick on\n");
        return 0;
    }
    if (!st->prof.nb_wb_knots) {
        fprintf(stderr, "a spot white balance needs the camera's per-CCT table "
                        "and this clip has none\n");
        snprintf(st->pick.refused, sizeof(st->pick.refused), "no table");
        return 0;
    }

    ret = view_pick_sample(st, st->pick.x, st->pick.y, &s);
    if (ret < 0)
        goto restore;

    if (s.kept * VIEW_PICK_MIN_FRACTION < s.total) {
        /* Whichever reason took more of the patch is the word. */
        snprintf(st->pick.refused, sizeof(st->pick.refused), "%s",
                 s.clipped >= s.dark ? "clipped" : "too dark");
        view_log(st, "pick: frame %" PRId64 " at %d,%d, patch %dx%d at %d,%d: "
                     "refused, %s (%d of %d kept: %d clipped, %d below %.5f)\n",
                 st->index, st->pick.x, st->pick.y, VIEW_PICK_SIZE, VIEW_PICK_SIZE,
                 s.x0, s.y0, st->pick.refused, s.kept, s.total, s.clipped,
                 s.dark, s.floor);
        ret = 0;
        goto restore;
    }

    ret = craw_profile_neutral_to_illuminant(&st->prof,
                                             (float)(s.raw[0] / s.raw[1]),
                                             (float)(s.raw[2] / s.raw[1]), &illum);
    if (ret < 0) {
        fprintf(stderr, "craw_profile_neutral_to_illuminant failed: %d\n", ret);
        snprintf(st->pick.refused, sizeof(st->pick.refused), "no table");
        ret = 0;
        goto restore;
    }

    view_log(st, "pick: frame %" PRId64 " at %d,%d, patch %dx%d at %d,%d: "
                 "%d of %d kept (%d clipped, %d below %.5f), AP1 R/G %.4f "
                 "B/G %.4f, raw R/G %.4f B/G %.4f -> %.0f K tint %+.4f\n",
             st->index, st->pick.x, st->pick.y, VIEW_PICK_SIZE, VIEW_PICK_SIZE,
             s.x0, s.y0, s.kept, s.total, s.clipped, s.dark, s.floor,
             s.ap1[0] / s.ap1[1], s.ap1[2] / s.ap1[1],
             s.raw[0] / s.raw[1], s.raw[2] / s.raw[1], illum.cct, illum.tint);

    st->look.manual = illum;
    st->look.wb = VIEW_WB_MANUAL;

    /* The check, with -v: the same patch under what was picked has to come
     * out neutral, and picking it again has to find the same illuminant.
     * Both go through the sampler above, so this is the whole inversion
     * checked against the pipeline, not against itself. */
    if (st->cfg.verbose) {
        struct view_pick_sample t;
        CRAWIlluminant again = { 0 };

        ret = view_pick_sample(st, st->pick.x, st->pick.y, &t);
        if (ret < 0)
            goto restore;
        if (t.kept)
            craw_profile_neutral_to_illuminant(&st->prof,
                                               (float)(t.raw[0] / t.raw[1]),
                                               (float)(t.raw[2] / t.raw[1]),
                                               &again);
        view_log(st, "pick: check: under %.0f K tint %+.4f the patch reads AP1 "
                     "R/G %.4f B/G %.4f (%+.2f%% / %+.2f%%), %d of %d kept; "
                     "picked again -> %.0f K tint %+.4f (%+.0f K)\n",
                 illum.cct, illum.tint,
                 t.kept ? t.ap1[0] / t.ap1[1] : 0.0,
                 t.kept ? t.ap1[2] / t.ap1[1] : 0.0,
                 t.kept ? (t.ap1[0] / t.ap1[1] - 1.0) * 100.0 : 0.0,
                 t.kept ? (t.ap1[2] / t.ap1[1] - 1.0) * 100.0 : 0.0,
                 t.kept, t.total, again.cct, again.tint, again.cct - illum.cct);
    }

restore:
    /* The frame as it is meant to be looked at: compressed, and with the
     * result when there is one. A refusal or a failure leaves the look
     * alone and still has to put the uncompressed picture back. */
    {
        int r = view_replay_frame(st);

        if (r < 0 && ret >= 0)
            ret = r;
    }

    return ret;
}
