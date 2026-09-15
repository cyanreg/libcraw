/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* PNG sequence export.
 *
 * The frames go through exactly the same libplacebo path as the display,
 * only into an offscreen sRGB target, so what lands on disk is the look on
 * screen rather than the scene-linear numbers: display referred, 16 bit. */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>

#include "view.h"

int view_export_png(const char *path, const uint16_t *rgba, uint32_t w, uint32_t h)
{
    const AVCodec *codec;
    AVCodecContext *avc = NULL;
    AVFrame *frame = NULL;
    AVPacket *pkt = NULL;
    FILE *f = NULL;
    int ret;

    codec = avcodec_find_encoder(AV_CODEC_ID_PNG);
    if (!codec) {
        fprintf(stderr, "libavcodec has no PNG encoder\n");
        return CRAW_ERROR(ENOSYS);
    }

    avc = avcodec_alloc_context3(codec);
    if (!avc) {
        ret = CRAW_ERROR(ENOMEM);
        goto fail;
    }

    avc->width = (int)w;
    avc->height = (int)h;
    avc->pix_fmt = AV_PIX_FMT_RGB48BE;
    avc->time_base = (AVRational) { 1, 1 };

    ret = avcodec_open2(avc, codec, NULL);
    if (ret < 0) {
        fprintf(stderr, "avcodec_open2(png): %s\n", av_err2str(ret));
        ret = CRAW_ERROR(EINVAL);
        goto fail;
    }

    frame = av_frame_alloc();
    pkt = av_packet_alloc();
    if (!frame || !pkt) {
        ret = CRAW_ERROR(ENOMEM);
        goto fail;
    }

    frame->format = avc->pix_fmt;
    frame->width = (int)w;
    frame->height = (int)h;

    ret = av_frame_get_buffer(frame, 0);
    if (ret < 0) {
        ret = CRAW_ERROR(ENOMEM);
        goto fail;
    }

    /* 16 bit PNG is big endian by definition, so the samples are written
     * byte by byte and the host's own order never comes into it. */
    for (uint32_t y = 0; y < h; y++) {
        const uint16_t *src = rgba + (size_t)y * w * 4;
        uint8_t *dst = frame->data[0] + (ptrdiff_t)y * frame->linesize[0];

        for (uint32_t x = 0; x < w; x++) {
            for (int c = 0; c < 3; c++) {
                unsigned v = src[x * 4 + c];
                dst[(x * 3 + c) * 2 + 0] = (uint8_t)(v >> 8);
                dst[(x * 3 + c) * 2 + 1] = (uint8_t)(v & 0xFF);
            }
        }
    }

    ret = avcodec_send_frame(avc, frame);
    if (ret >= 0)
        ret = avcodec_send_frame(avc, NULL);
    if (ret < 0) {
        fprintf(stderr, "%s: avcodec_send_frame: %s\n", path, av_err2str(ret));
        ret = CRAW_ERROR(EINVAL);
        goto fail;
    }

    ret = avcodec_receive_packet(avc, pkt);
    if (ret < 0) {
        fprintf(stderr, "%s: avcodec_receive_packet: %s\n", path, av_err2str(ret));
        ret = CRAW_ERROR(EINVAL);
        goto fail;
    }

    f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "%s: cannot open for writing\n", path);
        ret = CRAW_ERROR(EIO);
        goto fail;
    }

    if (fwrite(pkt->data, 1, (size_t)pkt->size, f) != (size_t)pkt->size) {
        fprintf(stderr, "%s: short write\n", path);
        ret = CRAW_ERROR(EIO);
        goto fail;
    }

    ret = 0;

fail:
    if (f)
        fclose(f);
    av_packet_free(&pkt);
    av_frame_free(&frame);
    avcodec_free_context(&avc);
    return ret;
}

void view_export_begin(struct view_state *st, int64_t first, int64_t last,
                       const char *dir)
{
    if (last < first)
        last = first;

    st->exp.first = CRAW_MAX(first, 0);
    st->exp.last = CRAW_MAX(last, 0);
    st->exp.next = st->exp.first;
    st->exp.done = 0;
    st->exp.active = true;

    /* A clean peak detection track, so that the same range exported twice
     * produces the same files. */
    view_gpu_reset_peak(&st->gpu, true);
    snprintf(st->exp.dir, sizeof(st->exp.dir), "%s", dir ? dir : "");
    snprintf(st->exp.status, sizeof(st->exp.status), "starting");
}

/* One frame per call so the window keeps redrawing while a long range is
 * being written. */
int view_export_step(struct view_state *st)
{
    struct view_export *e = &st->exp;
    size_t nval;
    char path[VIEW_MAX_PATH + 64];
    const char *name;
    int ret;

    if (!e->active)
        return 0;

    if (e->next > e->last) {
        snprintf(e->status, sizeof(e->status), "wrote %" PRId64 " frames", e->done);
        e->active = false;
        return 0;
    }

    /* Walk to the wanted picture. Every ProRes RAW frame is a keyframe, so
     * a seek normally lands on it; decoding on is the fallback for a source
     * whose timestamps do not map to frame numbers exactly. */
    if (!st->have_frame || st->index != e->next) {
        if (!st->have_frame || st->index >= e->next || st->index < e->next - 8)
            st->seek_to = e->next;

        do {
            ret = view_step_frame(st);
            if (ret == 3)
                return 0;       /* the reader is behind; try again later */
        } while (ret == 2);

        if (ret < 0)
            goto fail;
        if (ret > 0) {
            snprintf(e->status, sizeof(e->status),
                     "clip ended after %" PRId64 " frames", e->done);
            e->active = false;
            return 0;
        }
        if (st->index != e->next)
            return 0;
    }

    nval = (size_t)st->gpu.width * st->gpu.height * 4;
    if (e->rgba_values < nval) {
        uint16_t *p = realloc(e->rgba, nval * sizeof(*p));
        if (!p) {
            ret = CRAW_ERROR(ENOMEM);
            goto fail;
        }
        e->rgba = p;
        e->rgba_values = nval;
    }

    ret = view_gpu_render_export(&st->gpu, st, e->rgba);
    if (ret < 0)
        goto fail;

    snprintf(path, sizeof(path), "%s/craw-view-%05d.png",
             e->dir[0] ? e->dir : ".", (int)e->next);

    ret = view_export_png(path, e->rgba, st->gpu.width, st->gpu.height);
    if (ret < 0)
        goto fail;

    e->done++;
    /* The panel only has room for the file name; the full path goes to the
     * log, which is where a batch run reads it from anyway. */
    name = strrchr(path, '/');
    snprintf(e->status, sizeof(e->status), "frame %" PRId64 " -> %.64s",
             e->next, name ? name + 1 : path);
    view_log(st, "frame %" PRId64 " -> %s\n", e->next, path);

    e->next++;

    return 0;

fail:
    snprintf(e->status, sizeof(e->status), "failed: %d", ret);
    e->active = false;
    return ret;
}
