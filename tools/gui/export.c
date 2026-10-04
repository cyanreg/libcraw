/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Still export: one frame, one file.
 *
 * The frame goes through exactly the same libplacebo path as the display,
 * only into an offscreen sRGB target the size of the crop, so what lands
 * on disk is the look on screen rather than the scene-linear numbers:
 * display referred, sixteen bit. still.c turns that into a file, with the
 * format taken from the name it is given.
 *
 * A range of frames is deliberately not a thing here: a sequence of stills
 * is a video encode wearing a hat, and this is the screenshot path. */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavformat/avformat.h>
#include <libavutil/time.h>

#include "view.h"
#include "still.h"

void view_export_default_path(const struct view_state *st, int64_t frame,
                              char *out, size_t size)
{
    const char *file = strrchr(st->cfg.input, '/');
    const char *dot;
    char base[128];
    size_t n;

    file = file ? file + 1 : st->cfg.input;
    dot = strrchr(file, '.');
    n = dot ? (size_t)(dot - file) : strlen(file);
    n = CRAW_MIN(n, sizeof(base) - 1);
    memcpy(base, file, n);
    base[n] = '\0';

    /* In ~/Processed (view_export_dir()), named after the clip and the
     * frame, so that two frames of one clip never land on the same file.
     * JPEG, the format a still is looked at and sent in (Lynne's choice;
     * it was PNG, which holds what was rendered without a decision and is
     * a hundred megabytes of it). The field's extension picks another. */
    snprintf(out, size, "%s/%s-%05d.jpg", view_export_dir(st), base, (int)frame);
}

void view_export_complete_path(char *path, size_t size)
{
    size_t len = strlen(path);

    /* A name that picks no muxer has no extension libavformat knows, and
     * the default's JPEG is what such a name means. A name that ends in a
     * slash is a directory, which no extension makes a file. */
    if (!len || av_guess_format(NULL, path, NULL) || path[len - 1] == '/')
        return;
    if (path[len - 1] == '.')
        len--;
    snprintf(path + len, size - len, ".jpg");
}

void view_export_begin(struct view_state *st, int64_t frame, const char *path,
                       const char *codec)
{
    struct view_export *e = &st->exp;

    /* With every other frame, the shown frame at or before it, as a seek
     * lands: the frames between are never developed. */
    e->frame = CRAW_MAX(frame, 0);
    e->frame -= e->frame % view_frame_step(st);
    e->active = true;
    e->sought = false;
    snprintf(e->codec, sizeof(e->codec), "%s", codec ? codec : "");
    /* Whatever --enc was given at launch, which is as true of an export
     * from the panel as of one from the command line. */
    snprintf(e->enc, sizeof(e->enc), "%s",
             st->cfg.export_enc ? st->cfg.export_enc : "");
    if (path && path[0])
        snprintf(e->path, sizeof(e->path), "%s", path);
    else
        view_export_default_path(st, e->frame, e->path, sizeof(e->path));
    view_export_complete_path(e->path, sizeof(e->path));

    /* A clean peak detection track, so that one frame exported twice gives
     * the same file twice. */
    view_gpu_reset_peak(&st->gpu, true);
    snprintf(e->status, sizeof(e->status), "exporting frame %" PRId64, e->frame);
}

/* Called once per iteration of whichever loop is running, so that the
 * window keeps redrawing while the decoder walks to the frame. */
int view_export_step(struct view_state *st)
{
    struct view_export *e = &st->exp;
    char info[64], region[48] = "";
    const char *name;
    int64_t t0, t1;
    size_t nval;
    int x, y, w, h;
    int ret;

    if (!e->active)
        return 0;

    /* Walk to the wanted picture. Every ProRes RAW frame is a keyframe, so
     * a seek normally lands on it; decoding on is the fallback for a source
     * whose timestamps do not map to frame numbers exactly. */
    if (!st->have_frame || st->index != e->frame) {
        /* One seek per approach: a starved step comes back before the
         * reader has queued the packet, and seeking again would throw that
         * packet away and start over, a livelock on any medium slower than
         * this loop (seen on the CFexpress card, 8.9 MB re-read every 2 ms
         * for minutes). The seek is re-armed only once a picture has
         * actually arrived and is the wrong one. */
        if (!e->sought &&
            (!st->have_frame || st->index >= e->frame || st->index < e->frame - 8)) {
            st->seek_to = e->frame;
            e->sought = true;
        }

        do {
            ret = view_step_frame(st);
            if (ret == 3)
                return 0;       /* the reader is behind; try again later */
        } while (ret == 2);

        if (ret < 0)
            goto fail;
        if (ret > 0) {
            snprintf(e->status, sizeof(e->status),
                     "the clip ends before frame %" PRId64, e->frame);
            e->active = false;
            return 0;
        }
        if (st->index != e->frame) {
            e->sought = false;
            return 0;
        }
    }

    /* The render target is the crop, or the whole frame when there is
     * none, always at source resolution - and upright, so its dimensions
     * are the crop's turned by the clip's display matrix. The region the
     * line reports is the source rectangle, which is the one `--crop`
     * takes; view_still_write() reports the size that was written. */
    if (view_crop_rect(st, &x, &y, &w, &h))
        snprintf(region, sizeof(region), " crop %dx%d+%d+%d", w, h, x, y);

    view_crop_disp_rect(st, &x, &y, &w, &h);

    nval = (size_t)w * h * 4;
    if (e->rgba_values < nval) {
        uint16_t *p = realloc(e->rgba, nval * sizeof(*p));
        if (!p) {
            ret = CRAW_ERROR(ENOMEM);
            goto fail;
        }
        e->rgba = p;
        e->rgba_values = nval;
    }

    t0 = av_gettime_relative();
    ret = view_gpu_render_export(&st->gpu, st, e->rgba);
    if (ret < 0)
        goto fail;

    t1 = av_gettime_relative();
    ret = view_still_write(e->path, e->codec[0] ? e->codec : NULL, e->enc,
                           e->rgba, (uint32_t)w, (uint32_t)h, info, sizeof(info));
    if (ret < 0)
        goto fail;

    /* The panel only has room for the file name; the full path goes to the
     * log, which is where a batch run reads it from anyway. Rendering and
     * encoding are reported apart because they are so far apart: the GPU
     * pass is milliseconds and the encoder is seconds. */
    name = strrchr(e->path, '/');
    snprintf(e->status, sizeof(e->status), "%.40s: %s",
             name ? name + 1 : e->path, info);
    view_log(st, "frame %" PRId64 "%s -> %s: %s (render %.0f ms, encode %.0f ms)\n",
             e->frame, region, e->path, info, (double)(t1 - t0) / 1000.0,
             (double)(av_gettime_relative() - t1) / 1000.0);

    e->active = false;

    return 0;

fail:
    snprintf(e->status, sizeof(e->status), "failed: %d", ret);
    e->active = false;
    return ret;
}
