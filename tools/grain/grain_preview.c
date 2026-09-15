/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Headless proof for the film grain model in tools/gui/grain.c.
 *
 * Creates a libplacebo Vulkan context with no surface, pushes a test picture
 * through pl_render_image exactly the way craw-view does, applies grain by
 * each of the two paths and each preset, measures what came out and writes
 * contact sheets to look at. Nothing here is part of craw-view; it exists so
 * that the numbers in DESIGN.md's film grain section are measurements and
 * not assertions.
 *
 * Anything frame sized goes out as raw RGB48LE in a NUT container, because
 * deflating a megapixel of noise costs more than rendering it and the
 * measurements are taken from the pixels in memory anyway. Only the contact
 * sheets - crops assembled to be looked at, each well under a megapixel -
 * are written as PNG. To see a frame, crop it first:
 *
 *   ffmpeg -i frame_super8.nut -vf crop=640:480:560:300 -frames:v 1 crop.png
 */

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>

#include <libplacebo/renderer.h>
#include <libplacebo/utils/upload.h>
#include <libplacebo/vulkan.h>

#include "grain.h"

/* IEEE 754 binary32 to binary16, round to nearest even, with the
 * subnormal and overflow cases handled - the chart has values at 0.02 and
 * the frame has values well above 1. */
static uint16_t f32_to_f16(float f)
{
    union { float f; uint32_t u; } v = { .f = f };
    uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xFF) - 127 + 15;
    uint32_t man = v.u & 0x7FFFFFu;

    if (exp >= 0x1F)                    /* overflow, infinity or NaN */
        return (uint16_t)(sign | 0x7C00u | (man && exp == 0x8F + 15 ? 0x200u : 0));
    if (exp <= 0) {                     /* subnormal or zero */
        if (exp < -10)
            return (uint16_t)sign;
        man |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = man >> shift;
        if ((man >> (shift - 1)) & 1)
            half++;
        return (uint16_t)(sign | half);
    }
    uint32_t half = ((uint32_t)exp << 10) | (man >> 13);
    if ((man & 0x1FFFu) > 0x1000u || ((man & 0x1FFFu) == 0x1000u && (half & 1)))
        half++;
    return (uint16_t)(sign | half);
}

/* ------------------------------------------------------------- picture */

struct image {
    int w, h;
    float *rgb;                 /* scene linear, diffuse white at 1.0 */
};

static void image_free(struct image *im)
{
    free(im->rgb);
    im->rgb = NULL;
}

/* The chart: a horizontal scene-linear ramp across the top, then four flat
 * fields big enough to see grain structure in and to measure on. The flat
 * values are scene reflectances against diffuse white. */
#define RAMP_H 96
static const float patch_value[4] = { 0.02f, 0.18f, 0.50f, 0.90f };

static int image_chart(struct image *im, int w, int h)
{
    im->w = w;
    im->h = h;
    im->rgb = malloc((size_t)w * h * 3 * sizeof(float));
    if (!im->rgb)
        return -ENOMEM;

    const int ramp_h = h / 10 > RAMP_H ? RAMP_H : h / 10;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float v;
            if (y < ramp_h) {
                v = 1.2f * x / (float)(w - 1);
            } else {
                int col = x * 4 / w;
                v = patch_value[col > 3 ? 3 : col];
            }
            float *p = &im->rgb[((size_t)y * w + x) * 3];
            p[0] = p[1] = p[2] = v;
        }
    }
    return 0;
}

/* Where a patch's measurement window lands in the output, given that the
 * chart is rendered 1:1 into the target. */
static void patch_rect(int idx, int w, int h, int *x0, int *y0, int *x1, int *y1)
{
    const int ramp_h = h / 10 > RAMP_H ? RAMP_H : h / 10;
    const int inset = 24;
    *x0 = idx * w / 4 + inset;
    *x1 = (idx + 1) * w / 4 - inset;
    *y0 = ramp_h + inset;
    *y1 = h - inset;
}

/* PFM, as craw-process writes it with `--encode linear -o x.pfm`: linear
 * float RGB, bottom row first. Box-averaged down to the wanted width, which
 * is also what a real viewer does to a 5760 wide frame. */
static int image_pfm(struct image *im, const char *path, int want_w)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "%s: %s\n", path, strerror(errno));
        return -errno;
    }
    char magic[3] = {0};
    int w, h;
    float scale;
    if (fscanf(f, "%2s %d %d %f", magic, &w, &h, &scale) != 4 ||
        strcmp(magic, "PF")) {
        fprintf(stderr, "%s: not a colour PFM\n", path);
        fclose(f);
        return -EINVAL;
    }
    fgetc(f);                   /* the single whitespace after the scale */

    float *row = malloc((size_t)w * 3 * sizeof(float));
    float *full = malloc((size_t)w * h * 3 * sizeof(float));
    if (!row || !full) {
        free(row);
        free(full);
        fclose(f);
        return -ENOMEM;
    }
    for (int y = 0; y < h; y++) {
        if (fread(row, sizeof(float), (size_t)w * 3, f) != (size_t)w * 3) {
            fprintf(stderr, "%s: short read at row %d\n", path, y);
            free(row);
            free(full);
            fclose(f);
            return -EIO;
        }
        memcpy(&full[(size_t)(h - 1 - y) * w * 3], row, (size_t)w * 3 * sizeof(float));
    }
    free(row);
    fclose(f);

    int k = want_w > 0 && want_w < w ? w / want_w : 1;
    if (k < 1)
        k = 1;
    im->w = w / k;
    im->h = h / k;
    im->rgb = malloc((size_t)im->w * im->h * 3 * sizeof(float));
    if (!im->rgb) {
        free(full);
        return -ENOMEM;
    }
    for (int y = 0; y < im->h; y++) {
        for (int x = 0; x < im->w; x++) {
            double acc[3] = {0};
            for (int j = 0; j < k; j++)
                for (int i = 0; i < k; i++)
                    for (int c = 0; c < 3; c++)
                        acc[c] += full[((size_t)(y * k + j) * w + x * k + i) * 3 + c];
            for (int c = 0; c < 3; c++)
                im->rgb[((size_t)y * im->w + x) * 3 + c] = (float)(acc[c] / (k * k));
        }
    }
    free(full);
    return 0;
}

/* ---------------------------------------------------------- raw output */

/* One frame of raw RGB48LE in a NUT container. No encoder, no compression:
 * the point is to get the pixels onto disk without paying for deflate on
 * what is by construction incompressible noise. */
static int write_nut(const char *path, const uint16_t *rgba, int w, int h)
{
    AVFormatContext *fc = NULL;
    AVPacket *pkt = NULL;
    AVStream *st;
    int ret;

    ret = avformat_alloc_output_context2(&fc, NULL, "nut", path);
    if (ret < 0 || !fc) {
        fprintf(stderr, "%s: no NUT muxer\n", path);
        return -ENOSYS;
    }
    st = avformat_new_stream(fc, NULL);
    if (!st) {
        avformat_free_context(fc);
        return -ENOMEM;
    }
    st->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    st->codecpar->codec_id = AV_CODEC_ID_RAWVIDEO;
    st->codecpar->codec_tag = 0;
    st->codecpar->format = AV_PIX_FMT_RGB48LE;
    st->codecpar->width = w;
    st->codecpar->height = h;
    st->time_base = (AVRational) { 1, 1 };

    ret = avio_open(&fc->pb, path, AVIO_FLAG_WRITE);
    if (ret < 0) {
        fprintf(stderr, "%s: %s\n", path, av_err2str(ret));
        avformat_free_context(fc);
        return -EIO;
    }
    ret = avformat_write_header(fc, NULL);
    if (ret < 0)
        goto done;

    pkt = av_packet_alloc();
    if (!pkt) {
        ret = -ENOMEM;
        goto done;
    }
    ret = av_new_packet(pkt, w * h * 6);
    if (ret < 0)
        goto done;

    /* The renderer hands back RGBA; the alpha is always opaque here. */
    for (size_t i = 0, n = (size_t)w * h; i < n; i++) {
        uint16_t *dst = (uint16_t *)(pkt->data + i * 6);
        dst[0] = rgba[i * 4 + 0];
        dst[1] = rgba[i * 4 + 1];
        dst[2] = rgba[i * 4 + 2];
    }
    pkt->stream_index = 0;
    pkt->pts = pkt->dts = 0;
    pkt->duration = 1;
    pkt->flags |= AV_PKT_FLAG_KEY;

    ret = av_write_frame(fc, pkt);
    if (ret >= 0)
        ret = av_write_trailer(fc);
done:
    if (ret < 0)
        fprintf(stderr, "%s: %s\n", path, av_err2str(ret));
    av_packet_free(&pkt);
    if (fc->pb)
        avio_closep(&fc->pb);
    avformat_free_context(fc);
    return ret < 0 ? -EIO : 0;
}

/* --------------------------------------------------------------- PNG */

/* Only for the contact sheets. A full frame goes through write_nut(). */
static int write_png(const char *path, const uint16_t *rgba, int w, int h)
{
    const AVCodec *codec = avcodec_find_encoder(AV_CODEC_ID_PNG);
    AVCodecContext *avc = NULL;
    AVFrame *frame = NULL;
    AVPacket *pkt = NULL;
    FILE *f = NULL;
    int ret = -EINVAL;

    if (!codec)
        return -ENOSYS;
    if ((long)w * h > 1200000L)
        fprintf(stderr, "%s: %dx%d is over a megapixel for a PNG; frame "
                "sized output belongs in a NUT\n", path, w, h);
    avc = avcodec_alloc_context3(codec);
    if (!avc)
        return -ENOMEM;
    avc->width = w;
    avc->height = h;
    avc->pix_fmt = AV_PIX_FMT_RGB48BE;
    avc->time_base = (AVRational) { 1, 1 };
    if (avcodec_open2(avc, codec, NULL) < 0)
        goto done;

    frame = av_frame_alloc();
    pkt = av_packet_alloc();
    if (!frame || !pkt)
        goto done;
    frame->format = avc->pix_fmt;
    frame->width = w;
    frame->height = h;
    if (av_frame_get_buffer(frame, 0) < 0)
        goto done;

    for (int y = 0; y < h; y++) {
        const uint16_t *src = rgba + (size_t)y * w * 4;
        uint8_t *dst = frame->data[0] + (ptrdiff_t)y * frame->linesize[0];
        for (int x = 0; x < w; x++)
            for (int c = 0; c < 3; c++) {
                unsigned v = src[x * 4 + c];
                dst[(x * 3 + c) * 2 + 0] = (uint8_t)(v >> 8);
                dst[(x * 3 + c) * 2 + 1] = (uint8_t)(v & 0xFF);
            }
    }

    if (avcodec_send_frame(avc, frame) < 0 || avcodec_send_frame(avc, NULL) < 0)
        goto done;
    if (avcodec_receive_packet(avc, pkt) < 0)
        goto done;
    f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "%s: %s\n", path, strerror(errno));
        goto done;
    }
    ret = fwrite(pkt->data, 1, (size_t)pkt->size, f) == (size_t)pkt->size
        ? 0 : -EIO;
done:
    if (f)
        fclose(f);
    av_packet_free(&pkt);
    av_frame_free(&frame);
    avcodec_free_context(&avc);
    return ret;
}

/* -------------------------------------------------------------- state */

struct ctx {
    pl_log log;
    pl_vulkan vk;
    pl_gpu gpu;
    pl_renderer rr;
    craw_grain_hook *hook;

    pl_tex src;
    int src_w, src_h;
    pl_tex dst;
    int dst_w, dst_h;
    uint16_t *out;              /* dst_w * dst_h * 4 */
};

/* One probe hook, to confirm rather than assume what the image looks like
 * where the grain hook runs. */
static struct pl_color_space probe_color;
static struct pl_color_repr probe_repr;
static int probe_w, probe_h, probe_comps;
static bool probe_hit;

static struct pl_hook_res probe_fn(void *priv, const struct pl_hook_params *hp)
{
    (void) priv;
    probe_color = hp->color;
    probe_repr = hp->repr;
    probe_w = hp->tex ? hp->tex->params.w : 0;
    probe_h = hp->tex ? hp->tex->params.h : 0;
    probe_comps = hp->components;
    probe_hit = true;
    return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };
}

static const struct pl_hook probe_hook = {
    .stages = PL_HOOK_PRE_OUTPUT,
    .input = PL_HOOK_SIG_TEX,
    .hook = probe_fn,
    .signature = 0x70726f6265ULL,
};

/* A second hook that applies grain data handed to it directly, bypassing the
 * parameter model. Only the calibration mode uses it: the constants grain.c
 * needs for H.274 are properties of libplacebo's shader, so they have to be
 * measured through the shader rather than predicted. */
static struct pl_film_grain_data raw_data;
static pl_shader_obj raw_state;

static struct pl_hook_res raw_fn(void *priv, const struct pl_hook_params *hp)
{
    (void) priv;
    struct pl_color_repr repr = hp->repr;
    repr.sys = PL_COLOR_SYSTEM_RGB;
    repr.levels = PL_COLOR_LEVELS_FULL;

    struct pl_film_grain_params fg = {
        .data = raw_data,
        .tex = hp->tex,
        .repr = &repr,
        .components = 3,
        .component_mapping = { 0, 1, 2, -1 },
    };
    if (!pl_needs_film_grain(&fg))
        return (struct pl_hook_res) { .failed = true };

    pl_shader sh = pl_dispatch_begin(hp->dispatch);
    if (!pl_shader_film_grain(sh, &raw_state, &fg)) {
        pl_dispatch_abort(hp->dispatch, &sh);
        return (struct pl_hook_res) { .failed = true };
    }
    return (struct pl_hook_res) {
        .output = PL_HOOK_SIG_COLOR, .sh = sh, .repr = repr,
        .color = hp->color, .components = hp->components, .rect = hp->rect,
    };
}

static const struct pl_hook raw_hook = {
    .stages = PL_HOOK_PRE_OUTPUT,
    .input = PL_HOOK_SIG_TEX,
    .hook = raw_fn,
    .signature = 0x7261777467ULL,
};

static int ctx_init(struct ctx *c, const char *device)
{
    c->log = pl_log_create(PL_API_VER, pl_log_params(
        .log_cb = pl_log_color,
        .log_level = PL_LOG_WARN,
    ));
    c->vk = pl_vulkan_create(c->log, pl_vulkan_params(
        .device_name = device,
        .allow_software = true,
    ));
    if (!c->vk) {
        fprintf(stderr, "no Vulkan device libplacebo will take\n");
        return -ENODEV;
    }
    c->gpu = c->vk->gpu;
    c->rr = pl_renderer_create(c->log, c->gpu);
    if (!c->rr)
        return -ENOMEM;
    c->hook = craw_grain_hook_create(c->gpu);
    if (!c->hook)
        return -ENOMEM;
    printf("gpu: compute %s, subgroup size %d, GLSL %d.%d0\n",
           c->gpu->glsl.compute ? "yes" : "no", c->gpu->glsl.subgroup_size,
           c->gpu->glsl.version / 100, (c->gpu->glsl.version / 10) % 10);
    return 0;
}

static void ctx_free(struct ctx *c)
{
    craw_grain_hook_destroy(&c->hook);
    pl_tex_destroy(c->gpu, &c->src);
    pl_tex_destroy(c->gpu, &c->dst);
    pl_renderer_destroy(&c->rr);
    pl_vulkan_destroy(&c->vk);
    pl_log_destroy(&c->log);
    free(c->out);
}

static int ctx_upload(struct ctx *c, const struct image *im)
{
    pl_fmt fmt = pl_find_fmt(c->gpu, PL_FMT_FLOAT, 4, 16, 16, PL_FMT_CAP_SAMPLEABLE);
    if (!fmt) {
        fprintf(stderr, "no sampleable RGBA16F format\n");
        return -ENOTSUP;
    }
    if (!c->src || c->src_w != im->w || c->src_h != im->h) {
        pl_tex_destroy(c->gpu, &c->src);
        c->src = pl_tex_create(c->gpu, pl_tex_params(
            .w = im->w, .h = im->h, .format = fmt,
            .sampleable = true, .host_writable = true,
        ));
        if (!c->src)
            return -ENOMEM;
        c->src_w = im->w;
        c->src_h = im->h;
    }

    /* pl_tex_upload is a byte copy, not a conversion, so the host data has
     * to already be in the texture's format - fp16, the same as the texture
     * craw-view hands libplacebo. */
    uint16_t *rgba = malloc((size_t)im->w * im->h * 4 * sizeof(uint16_t));
    if (!rgba)
        return -ENOMEM;
    for (size_t i = 0; i < (size_t)im->w * im->h; i++) {
        for (int k = 0; k < 3; k++)
            rgba[i * 4 + k] = f32_to_f16(im->rgb[i * 3 + k]);
        rgba[i * 4 + 3] = f32_to_f16(1.0f);
    }
    bool ok = pl_tex_upload(c->gpu, pl_tex_transfer_params(
        .tex = c->src, .ptr = rgba,
    ));
    free(rgba);
    return ok ? 0 : -EIO;
}

static int ctx_target(struct ctx *c, int w, int h)
{
    if (c->dst && c->dst_w == w && c->dst_h == h)
        return 0;
    pl_tex_destroy(c->gpu, &c->dst);
    free(c->out);
    c->out = NULL;

    pl_fmt fmt = pl_find_named_fmt(c->gpu, "rgba16");
    if (!fmt || !(fmt->caps & PL_FMT_CAP_RENDERABLE) ||
        !(fmt->caps & PL_FMT_CAP_HOST_READABLE)) {
        fprintf(stderr, "the GPU cannot render to a readable RGBA16 texture\n");
        return -ENOTSUP;
    }
    c->dst = pl_tex_create(c->gpu, pl_tex_params(
        .w = w, .h = h, .format = fmt,
        .renderable = true,
        .storable = !!(fmt->caps & PL_FMT_CAP_STORABLE),
        .host_readable = true,
    ));
    if (!c->dst)
        return -ENOMEM;
    c->out = malloc((size_t)w * h * 4 * sizeof(uint16_t));
    if (!c->out)
        return -ENOMEM;
    c->dst_w = w;
    c->dst_h = h;
    return 0;
}

/* The frame craw-view hands libplacebo: one RGBA16F plane, scene linear
 * BT.2020 tagged scRGB, diffuse white at 1.0, the sensor clip as the static
 * HDR ceiling. See tools/gui/gpu.c, view_gpu_image(). */
static struct pl_frame make_image(const struct ctx *c, float ceiling)
{
    return (struct pl_frame) {
        .num_planes = 1,
        .planes[0] = {
            .texture = c->src,
            .components = 3,
            .component_mapping = { 0, 1, 2, -1 },
        },
        .repr = {
            .sys = PL_COLOR_SYSTEM_RGB,
            .levels = PL_COLOR_LEVELS_PC,
            .alpha = PL_ALPHA_NONE,
        },
        .color = {
            .primaries = PL_COLOR_PRIM_BT_2020,
            .transfer = PL_COLOR_TRC_SCRGB,
            .hdr = { .min_luma = PL_COLOR_HDR_BLACK, .max_luma = ceiling },
        },
        .crop = { 0, 0, (float)c->src_w, (float)c->src_h },
    };
}

static struct pl_frame make_target(const struct ctx *c)
{
    return (struct pl_frame) {
        .num_planes = 1,
        .planes[0] = {
            .texture = c->dst,
            .components = 4,
            .component_mapping = { 0, 1, 2, 3 },
        },
        .repr = {
            .sys = PL_COLOR_SYSTEM_RGB,
            .levels = PL_COLOR_LEVELS_PC,
            .alpha = PL_ALPHA_NONE,
            .bits = { .sample_depth = 16, .color_depth = 16 },
        },
        .color = pl_color_space_srgb,
        .crop = { 0, 0, (float)c->dst_w, (float)c->dst_h },
    };
}

/* Renders one case. `gp` may be NULL for the ungrained reference. */
static int render(struct ctx *c, const struct craw_grain_params *gp,
                  struct craw_grain_out *go, bool probe)
{
    struct pl_frame image = make_image(c, 1000.0f);
    struct pl_frame target = make_target(c);
    struct pl_render_params rp = pl_render_default_params;
    struct pl_color_adjustment adj = pl_color_adjustment_neutral;
    const struct pl_hook *hooks[4];
    int nb_hooks = 0;

    /* craw-view's scRGB scale correction, so the tone mapper sees the
     * pipeline's own numbers. */
    adj.contrast = PL_COLOR_SDR_WHITE / PL_COLOR_SCRGB_WHITE;
    rp.color_adjustment = &adj;
    /* Peak detection is stateful across frames and would make the same case
     * render differently depending on what ran before it. */
    rp.peak_detect_params = NULL;

    if (go)
        memset(go, 0, sizeof(*go));

    if (gp && gp->amount > 0.0f) {
        if (gp->path == CRAW_GRAIN_PATH_SOURCE) {
            struct craw_grain_out o;
            if (craw_grain_build(gp, &o) < 0) {
                fprintf(stderr, "craw_grain_build failed\n");
                return -EINVAL;
            }
            const char *bad = craw_grain_validate(&o.data);
            if (bad) {
                fprintf(stderr, "invalid grain data: %s\n", bad);
                return -ERANGE;
            }
            image.film_grain = o.data;
            if (go)
                *go = o;
        } else {
            const struct pl_hook *h = craw_grain_hook_update(c->hook, gp, go);
            if (h)
                hooks[nb_hooks++] = h;
        }
    }
    if (raw_data.type != PL_FILM_GRAIN_NONE)
        hooks[nb_hooks++] = &raw_hook;
    if (probe) {
        probe_hit = false;
        hooks[nb_hooks++] = &probe_hook;
    }
    rp.hooks = hooks;
    rp.num_hooks = nb_hooks;

    pl_renderer_flush_cache(c->rr);
    if (!pl_render_image(c->rr, &image, &target, &rp)) {
        fprintf(stderr, "pl_render_image failed\n");
        return -EIO;
    }
    if (!pl_tex_download(c->gpu, pl_tex_transfer_params(
            .tex = c->dst, .ptr = c->out))) {
        fprintf(stderr, "pl_tex_download failed\n");
        return -EIO;
    }
    return 0;
}

/* ---------------------------------------------------------- measuring */

struct meas {
    double mean[3];
    double sd[3];               /* per channel, normalised to full scale */
    double sd_luma;             /* of the mean of the three, i.e. what the
                                 * eye mostly sees */
    double fwhm;                /* blob size of the green channel */
};

static void measure(const uint16_t *buf, int stride, int x0, int y0,
                    int x1, int y1, struct meas *m)
{
    const int w = x1 - x0, h = y1 - y0;
    double *g = malloc((size_t)w * h * sizeof(double));
    memset(m, 0, sizeof(*m));

    for (int c = 0; c < 3; c++) {
        double s = 0, s2 = 0;
        for (int y = y0; y < y1; y++)
            for (int x = x0; x < x1; x++) {
                double v = buf[((size_t)y * stride + x) * 4 + c] / 65535.0;
                s += v;
                s2 += v * v;
            }
        double n = (double)w * h;
        m->mean[c] = s / n;
        double var = s2 / n - m->mean[c] * m->mean[c];
        m->sd[c] = var > 0 ? sqrt(var) : 0.0;
    }

    double s = 0, s2 = 0;
    for (int y = y0; y < y1; y++)
        for (int x = x0; x < x1; x++) {
            double v = 0;
            for (int c = 0; c < 3; c++)
                v += buf[((size_t)y * stride + x) * 4 + c] / 65535.0;
            v /= 3.0;
            s += v;
            s2 += v * v;
            if (g)
                g[(size_t)(y - y0) * w + (x - x0)] =
                    buf[((size_t)y * stride + x) * 4 + 1] / 65535.0;
        }
    double n = (double)w * h;
    double mean = s / n, var = s2 / n - mean * mean;
    m->sd_luma = var > 0 ? sqrt(var) : 0.0;

    /* Blob size, as the full width at half maximum of the autocorrelation,
     * the same measure grain.c solves against. */
    m->fwhm = 1.0;
    if (g) {
        double gm = 0, gv = 0;
        for (size_t i = 0; i < (size_t)w * h; i++)
            gm += g[i];
        gm /= (double)w * h;
        for (size_t i = 0; i < (size_t)w * h; i++)
            gv += (g[i] - gm) * (g[i] - gm);
        gv /= (double)w * h;
        if (gv > 0) {
            double half[2] = { 0.5, 0.5 };
            for (int axis = 0; axis < 2; axis++) {
                double prev = 1.0;
                for (int lag = 1; lag < 24; lag++) {
                    double acc = 0;
                    long cnt = 0;
                    for (int y = 0; y < h - (axis ? lag : 0); y++)
                        for (int x = 0; x < w - (axis ? 0 : lag); x++) {
                            double a = g[(size_t)y * w + x] - gm;
                            double b = axis ? g[(size_t)(y + lag) * w + x] - gm
                                            : g[(size_t)y * w + x + lag] - gm;
                            acc += a * b;
                            cnt++;
                        }
                    double rho = acc / cnt / gv;
                    if (rho < 0.5) {
                        half[axis] = (lag - 1) + (prev - 0.5) / (prev - rho);
                        break;
                    }
                    prev = rho;
                    half[axis] = lag;
                }
            }
            m->fwhm = half[0] + half[1];
        }
        free(g);
    }
}

/* ------------------------------------------------------------- sheets */

struct sheet {
    uint16_t *px;
    int w, h, cell, cols;
    int n;
};

static int sheet_init(struct sheet *s, int cell, int cols, int rows)
{
    const int gap = 4;
    s->cell = cell;
    s->cols = cols;
    s->w = cols * cell + (cols + 1) * gap;
    s->h = rows * cell + (rows + 1) * gap;
    s->n = 0;
    s->px = calloc((size_t)s->w * s->h * 4, sizeof(uint16_t));
    return s->px ? 0 : -ENOMEM;
}

static void sheet_add(struct sheet *s, const uint16_t *src, int stride,
                      int sx, int sy)
{
    const int gap = 4;
    int col = s->n % s->cols, row = s->n / s->cols;
    int ox = gap + col * (s->cell + gap);
    int oy = gap + row * (s->cell + gap);
    for (int y = 0; y < s->cell; y++)
        memcpy(&s->px[((size_t)(oy + y) * s->w + ox) * 4],
               &src[((size_t)(sy + y) * stride + sx) * 4],
               (size_t)s->cell * 4 * sizeof(uint16_t));
    s->n++;
}

/* Cell showing only what the grain added, around mid grey and multiplied
 * up, optionally magnified by an integer factor so that the blob size is
 * visible at the size a person actually looks at the sheet. Everything the
 * eye needs to judge - blob size, the 32x32 block grid, colour balance - is
 * two orders of magnitude below the picture and invisible without this. */
static void sheet_add_diff(struct sheet *s, const uint16_t *cur,
                           const uint16_t *ref, int stride, int sx, int sy,
                           float gain, int zoom)
{
    const int gap = 4;
    int col = s->n % s->cols, row = s->n / s->cols;
    int ox = gap + col * (s->cell + gap);
    int oy = gap + row * (s->cell + gap);
    if (zoom < 1)
        zoom = 1;
    for (int y = 0; y < s->cell; y++) {
        for (int x = 0; x < s->cell; x++) {
            size_t si = ((size_t)(sy + y / zoom) * stride + sx + x / zoom) * 4;
            size_t di = ((size_t)(oy + y) * s->w + ox + x) * 4;
            for (int c = 0; c < 3; c++) {
                float d = ((float)cur[si + c] - (float)ref[si + c]) * gain +
                          32768.0f;
                s->px[di + c] = (uint16_t)(d < 0 ? 0 : (d > 65535 ? 65535 : d));
            }
            s->px[di + 3] = 65535;
        }
    }
    s->n++;
}

static int sheet_write(struct sheet *s, const char *path)
{
    int ret = write_png(path, s->px, s->w, s->h);
    free(s->px);
    s->px = NULL;
    if (!ret)
        printf("  wrote %s (%dx%d)\n", path, s->w, s->h);
    return ret;
}


/* Measures what libplacebo's own H.274 shader produces for each cut-off
 * index, which is what grain.c's h274_cal[] table has to hold. Printed as C
 * so the result can be pasted straight in. */
static void calibrate_h274(struct ctx *c, int width, int height)
{
    static uint8_t lo[1], hi[1];
    static int16_t val[1][6];
    const int scale = 128, log2s = 4;
    const float unit = scale / (float)(1 << (log2s + 6));

    printf("\n=== H.274 calibration, measured through libplacebo ===\n");
    printf("%4s %6s %10s %10s %8s %8s\n", "idx", "cutoff", "measured sd",
           "sigma", "fwhm x", "fwhm y");
    lo[0] = 0;
    hi[0] = 255;
    for (int idx = 0; idx <= 12; idx++) {
        struct meas m;
        int x0, y0, x1, y1;
        val[0][0] = scale;
        val[0][1] = (int16_t)(idx + 2);
        val[0][2] = (int16_t)(idx + 2);
        val[0][3] = val[0][4] = val[0][5] = 0;
        raw_data = (struct pl_film_grain_data) {
            .type = PL_FILM_GRAIN_H274,
            .seed = 0x1234,
            .params.h274 = {
                .model_id = 0,
                .blending_mode_id = 0,
                .log2_scale_factor = log2s,
                /* All three components carry the same parameters, so that
                 * the number does not depend on how libplacebo maps model
                 * components onto picture channels for an RGB image - which
                 * it gets wrong in 7.371, see build_h274() in grain.c. */
                .component_model_present = { true, true, true },
                .num_intensity_intervals = { 1, 1, 1 },
                .num_model_values = { 3, 3, 3 },
                .intensity_interval_lower_bound = { lo, lo, lo },
                .intensity_interval_upper_bound = { hi, hi, hi },
                .comp_model_value = { (const int16_t (*)[6])val,
                                      (const int16_t (*)[6])val,
                                      (const int16_t (*)[6])val },
            },
        };
        if (render(c, NULL, NULL, false) < 0)
            break;
        patch_rect(1, width, height, &x0, &y0, &x1, &y1);
        measure(c->out, width, x0, y0, x1, y1, &m);
        printf("%4d %6d %10.6f %10.6f %8.2f %8.2f    "
               "{ %.6ff, %.2ff },\n",
               idx, ((idx + 3) << 2) - 1, m.sd[1], m.sd[1] / unit,
               m.fwhm, m.fwhm, m.sd[1] / unit, m.fwhm);
    }
    raw_data.type = PL_FILM_GRAIN_NONE;
    pl_shader_obj_destroy(&raw_state);
}

/* Same for AV1: what one unit of scaling point actually delivers, which is
 * the check on the 2040 constant in grain.h section 4. */
static void calibrate_av1(struct ctx *c, int width, int height)
{
    printf("\n=== AV1 calibration, measured through libplacebo ===\n");
    printf("%6s %6s %6s %12s %12s %8s %8s\n", "lag", "gss", "point",
           "measured sd", "predicted", "ratio", "fwhm");
    static const struct { int lag, gss, point; float a; } cases[] = {
        { 0, 0, 255, 0.0f }, { 0, 0, 128, 0.0f }, { 0, 0, 64, 0.0f },
        { 1, 0, 128, 0.5f }, { 1, 1, 128, 0.75f }, { 1, 2, 128, 0.85f },
    };
    for (size_t k = 0; k < sizeof(cases) / sizeof(*cases); k++) {
        struct pl_av1_grain_data d = {0};
        struct meas m;
        int16_t buf[73][82];
        int x0, y0, x1, y1;

        d.ar_coeff_lag = cases[k].lag;
        d.ar_coeff_shift = 7;
        d.grain_scale_shift = cases[k].gss;
        if (cases[k].lag == 1) {
            float a = cases[k].a;
            d.ar_coeffs_y[0] = (int8_t)lrintf(-a * a * 128.0f);
            d.ar_coeffs_y[1] = (int8_t)lrintf(a * 128.0f);
            d.ar_coeffs_y[3] = (int8_t)lrintf(a * 128.0f);
        }
        d.scaling_shift = 8;
        d.num_points_y = 2;
        d.points_y[0][0] = 0;
        d.points_y[0][1] = (uint8_t)cases[k].point;
        d.points_y[1][0] = 255;
        d.points_y[1][1] = (uint8_t)cases[k].point;
        d.overlap = true;

        float sd = craw_grain_template_y(buf, &d, 8, 0x1234);
        float predicted = cases[k].point * sd / (256.0f * 255.0f);

        raw_data = (struct pl_film_grain_data) {
            .type = PL_FILM_GRAIN_AV1, .seed = 0x1234, .params.av1 = d,
        };
        if (render(c, NULL, NULL, false) < 0)
            break;
        patch_rect(1, width, height, &x0, &y0, &x1, &y1);
        measure(c->out, width, x0, y0, x1, y1, &m);
        printf("%6d %6d %6d %12.6f %12.6f %8.3f %8.2f\n",
               cases[k].lag, cases[k].gss, cases[k].point, m.sd[1],
               predicted, m.sd[1] / predicted, m.fwhm);
    }
    raw_data.type = PL_FILM_GRAIN_NONE;
    pl_shader_obj_destroy(&raw_state);
}


/* Renders the exact data the model built, and the same data with its
 * scaling function flattened to a single known value, so that a mismatch
 * can be pinned on either the template's standard deviation or the scaling
 * function rather than on "somewhere in there". */
static void debug_av1(struct ctx *c, int width, int height, float level)
{
    printf("\n=== AV1 amplitude, built data against a flat scaling ===\n");
    printf("%-14s %4s %4s %4s | %8s %9s %9s %6s | %9s %9s %6s\n",
           "gauge", "lag", "gss", "ss", "tpl sd", "want", "got", "ratio",
           "flat want", "flat got", "ratio");
    for (int g = 0; g < CRAW_GRAIN_GAUGE_NB; g++) {
        struct craw_grain_params p = craw_grain_default_params();
        struct craw_grain_out o;
        struct meas m;
        int x0, y0, x1, y1;
        int16_t buf[73][82];

        p.gauge = g;
        p.stock = CRAW_GRAIN_STOCK_V3_500T;
        p.response = CRAW_GRAIN_RESPONSE_FLAT;
        p.model = CRAW_GRAIN_MODEL_AV1;
        p.width = width;
        p.height = height;
        p.seed = 0x1234;
        if (craw_grain_build(&p, &o) < 0)
            continue;

        raw_data = o.data;
        if (render(c, NULL, NULL, false) < 0)
            break;
        patch_rect(1, width, height, &x0, &y0, &x1, &y1);
        measure(c->out, width, x0, y0, x1, y1, &m);
        float want = craw_grain_sigma_at(&o, level);
        float got = (float)m.sd[1];

        /* Now the same template with a flat two point scaling at 128 and
         * scaling_shift 8, where the expected value is arithmetic. */
        struct pl_av1_grain_data d = o.data.params.av1;
        d.scaling_shift = 8;
        d.num_points_y = 2;
        d.points_y[0][0] = 0;   d.points_y[0][1] = 128;
        d.points_y[1][0] = 255; d.points_y[1][1] = 128;
        d.num_points_uv[0] = d.num_points_uv[1] = 0;
        float tpl = craw_grain_template_y(buf, &d, 8, (uint16_t)(p.seed & 0xFFFF));
        float fwant = 128.0f * tpl / (256.0f * 255.0f);
        raw_data.params.av1 = d;
        if (render(c, NULL, NULL, false) < 0)
            break;
        measure(c->out, width, x0, y0, x1, y1, &m);

        printf("%-14s %4d %4d %4d | %8.2f %9.5f %9.5f %6.2f | %9.5f %9.5f %6.2f\n",
               craw_grain_gauge_name(g), o.data.params.av1.ar_coeff_lag,
               o.data.params.av1.grain_scale_shift,
               o.data.params.av1.scaling_shift, tpl, want, got, got / want,
               fwant, m.sd[1], m.sd[1] / fwant);
    }
    raw_data.type = PL_FILM_GRAIN_NONE;
    pl_shader_obj_destroy(&raw_state);
}

/* ---------------------------------------------------------------- main */

static const char *outdir = ".";

static void path_for(char *buf, size_t n, const char *name)
{
    snprintf(buf, n, "%s/%s", outdir, name);
}

static void print_tables(void)
{
    static const int widths[] = { 5760, 3840, 1920 };
    printf("\n=== gauge table: 500T, flat response, output path ===\n");
    for (size_t k = 0; k < sizeof(widths) / sizeof(*widths); k++) {
        printf("\n-- applied at %d px wide --\n", widths[k]);
        printf("%-14s %8s %8s %8s %8s %9s %9s %6s\n", "gauge", "pitch um",
               "blob um", "blob px", "built px", "sigma_sig", "built", "model");
        for (int g = 0; g < CRAW_GRAIN_GAUGE_NB; g++) {
            struct craw_grain_params p = craw_grain_default_params();
            struct craw_grain_out o;
            p.gauge = g;
            p.stock = CRAW_GRAIN_STOCK_V3_500T;
            p.response = CRAW_GRAIN_RESPONSE_FLAT;
            p.width = widths[k];
            p.height = widths[k] * 9 / 16;
            p.seed = 1;
            if (craw_grain_build(&p, &o) < 0)
                continue;
            printf("%-14s %8.2f %8.2f %8.2f %8.2f %9.5f %9.5f %6s\n",
                   craw_grain_gauge_name(g), o.pitch_um, o.blob_um,
                   o.blob_px, o.blob_clamped, o.sigma_signal, o.sigma_built,
                   craw_grain_model_name(o.model_used));
        }
    }

    printf("\n=== stocks, Super 35 at 1920, flat response ===\n");
    printf("%-16s %6s %8s %10s %10s  %s\n", "stock", "rms", "blob um",
           "sigma_sig", "built", "source");
    for (int s = 0; s < CRAW_GRAIN_STOCK_NB; s++) {
        struct craw_grain_params p = craw_grain_default_params();
        struct craw_grain_out o;
        p.stock = s;
        p.response = CRAW_GRAIN_RESPONSE_FLAT;
        p.width = 1920;
        p.height = 1080;
        p.seed = 1;
        if (craw_grain_build(&p, &o) < 0)
            continue;
        printf("%-16s %6.1f %8.1f %10.5f %10.5f  %.60s\n",
               craw_grain_stock_name(s), craw_grain_stock_rms(s),
               craw_grain_stock_blob_um(s), o.sigma_signal, o.sigma_built,
               craw_grain_stock_source(s));
    }
}

static void usage(void)
{
    printf(
"usage: grain_preview [options]\n"
"  --out DIR      where the PNGs go (default: the working directory)\n"
"  --frame PFM    a developed frame from `craw-process --encode linear`\n"
"  --width N      output width (default 1920)\n"
"  --src-width N  chart width before scaling (default: the output width)\n"
"  --device SUB   Vulkan device whose name contains SUB\n"
"  --table        print the model's tables and exit, no GPU needed\n"
"  --calibrate    measure what each model delivers, for grain.c's tables\n"
"  --only SUB     run only the sheets whose name contains SUB\n");
}

static bool want(const char *only, const char *name)
{
    return !only || strstr(name, only);
}

int main(int argc, char **argv)
{
    const char *frame = NULL, *device = NULL, *only = NULL;
    int width = 1920, src_width = 0;
    bool table_only = false, calibrate = false, debug = false;
    char path[1024];

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--out") && i + 1 < argc)
            outdir = argv[++i];
        else if (!strcmp(argv[i], "--frame") && i + 1 < argc)
            frame = argv[++i];
        else if (!strcmp(argv[i], "--device") && i + 1 < argc)
            device = argv[++i];
        else if (!strcmp(argv[i], "--only") && i + 1 < argc)
            only = argv[++i];
        else if (!strcmp(argv[i], "--width") && i + 1 < argc)
            width = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--src-width") && i + 1 < argc)
            src_width = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--table"))
            table_only = true;
        else if (!strcmp(argv[i], "--calibrate"))
            calibrate = true;
        else if (!strcmp(argv[i], "--debug-av1"))
            debug = true;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage();
            return 0;
        } else {
            fprintf(stderr, "unknown option %s\n", argv[i]);
            usage();
            return 1;
        }
    }
    if (src_width <= 0)
        src_width = width;

    print_tables();
    if (table_only)
        return 0;

    struct ctx c = {0};
    if (ctx_init(&c, device) < 0)
        return 1;

    const int height = width * 9 / 16;
    struct image chart = {0};
    if (image_chart(&chart, src_width, src_width * 9 / 16) < 0)
        return 1;
    if (ctx_upload(&c, &chart) < 0 || ctx_target(&c, width, height) < 0)
        return 1;

    /* Confirm what the image actually is where the grain hook runs, rather
     * than trusting the reading of renderer.c. */
    if (render(&c, NULL, NULL, true) < 0)
        return 1;
    if (probe_hit) {
        printf("\n=== PL_HOOK_PRE_OUTPUT, measured ===\n"
               "  texture %dx%d (target is %dx%d, source is %dx%d)\n"
               "  transfer %d (srgb=%d, linear=%d), primaries %d, sys %d,"
               " levels %d, components %d\n",
               probe_w, probe_h, width, height, src_width,
               src_width * 9 / 16, probe_color.transfer,
               PL_COLOR_TRC_SRGB, PL_COLOR_TRC_LINEAR,
               probe_color.primaries, probe_repr.sys, probe_repr.levels,
               probe_comps);
    } else {
        printf("\n!! the PL_HOOK_PRE_OUTPUT probe never fired\n");
    }

    if (calibrate) {
        calibrate_av1(&c, width, height);
        calibrate_h274(&c, width, height);
    }

    /* The reference: no grain at all, and the patch levels it lands on. The
     * copy is what the grain-only sheets subtract. */
    if (render(&c, NULL, NULL, false) < 0)
        return 1;
    uint16_t *plain = malloc((size_t)width * height * 4 * sizeof(uint16_t));
    if (!plain)
        return 1;
    memcpy(plain, c.out, (size_t)width * height * 4 * sizeof(uint16_t));
    path_for(path, sizeof(path), "chart_none.nut");
    write_nut(path, c.out, width, height);
    printf("  wrote %s (%dx%d raw rgb48le)\n", path, width, height);

    struct meas ref[4];
    printf("\n=== reference patch levels, no grain ===\n");
    for (int i = 0; i < 4; i++) {
        int x0, y0, x1, y1;
        patch_rect(i, width, height, &x0, &y0, &x1, &y1);
        measure(c.out, width, x0, y0, x1, y1, &ref[i]);
        printf("  scene %.2f -> encoded %.4f, residual sd %.6f\n",
               patch_value[i], ref[i].mean[1], ref[i].sd_luma);
    }

    if (debug)
        debug_av1(&c, width, height, (float)ref[1].mean[1]);

    const int cell = 256;

    /* ---- every gauge, output path ---- */
    if (want(only, "gauges")) {
        printf("\n=== gauges, output path, 500T, print response ===\n");
        printf("%-14s %8s %8s | %9s %9s %8s | %8s %8s %9s\n", "gauge", "model",
               "want px", "want sd", "got G sd", "ratio", "got px",
               "chr/lum", "rgb mean");
        struct sheet s, sg;
        sheet_init(&s, cell, 5, 2);
        sheet_init(&sg, cell, 5, 2);
        for (int g = 0; g < CRAW_GRAIN_GAUGE_NB; g++) {
            struct craw_grain_params p = craw_grain_default_params();
            struct craw_grain_out o;
            struct meas m;
            int x0, y0, x1, y1;
            p.gauge = g;
            p.stock = CRAW_GRAIN_STOCK_V3_500T;
            p.response = CRAW_GRAIN_RESPONSE_PRINT;
            p.width = width;
            p.height = height;
            p.seed = 0x1234;
            if (render(&c, &p, &o, false) < 0)
                return 1;
            patch_rect(1, width, height, &x0, &y0, &x1, &y1);   /* the 18% */
            measure(c.out, width, x0, y0, x1, y1, &m);
            double chroma = m.sd[1] > 0
                ? 0.5 * (m.sd[0] + m.sd[2]) / m.sd[1] : 0.0;
            /* Compare at the level the patch actually landed on, not at mid
             * grey: the response curve is not flat and the 18 percent patch
             * tone maps to 0.40, not 0.5. */
            float want = craw_grain_sigma_at(&o, (float)ref[1].mean[1]);
            printf("%-14s %8s %8.2f | %9.5f %9.5f %8.2f | %8.2f %8.2f %9.5f\n",
                   craw_grain_gauge_name(g),
                   craw_grain_model_name(o.model_used), o.blob_px,
                   want, m.sd[1], m.sd[1] / want,
                   m.fwhm, chroma, m.sd_luma);
            sheet_add(&s, c.out, width, x0, y0);
            sheet_add_diff(&sg, c.out, plain, width, x0, y0, 12.0f, 2);
        }
        path_for(path, sizeof(path), "sheet_gauges_output.png");
        sheet_write(&s, path);
        path_for(path, sizeof(path), "sheet_gauges_grain.png");
        sheet_write(&sg, path);
        printf("  (grain sheet: the difference from the ungrained render,"
               " x12 around mid grey, magnified 2x)\n");
    }

    /* ---- the two paths, same gauges ---- */
    if (want(only, "paths") && src_width > width) {
        printf("\n=== source path (grain at %d) vs output path (at %d) ===\n",
               src_width, width);
        printf("%-14s %9s %9s %8s | %9s %9s %8s\n", "gauge",
               "src sd", "src px", "model", "out sd", "out px", "model");
        struct sheet s, sg;
        sheet_init(&s, cell, 4, 2);
        sheet_init(&sg, cell, 4, 2);
        static const int show[] = {
            CRAW_GRAIN_GAUGE_SUPER8, CRAW_GRAIN_GAUGE_16MM,
            CRAW_GRAIN_GAUGE_SUPER35, CRAW_GRAIN_GAUGE_IMAX,
        };
        struct meas got[2][4];
        struct craw_grain_out gout[2][4];
        for (int pass = 0; pass < 2; pass++) {
            for (int k = 0; k < 4; k++) {
                struct craw_grain_params p = craw_grain_default_params();
                int x0, y0, x1, y1;
                p.gauge = show[k];
                p.stock = CRAW_GRAIN_STOCK_V3_500T;
                p.response = CRAW_GRAIN_RESPONSE_PRINT;
                p.path = pass ? CRAW_GRAIN_PATH_SOURCE : CRAW_GRAIN_PATH_OUTPUT;
                p.width = pass ? src_width : width;
                p.height = pass ? src_width * 9 / 16 : height;
                p.seed = 0x1234;
                if (render(&c, &p, &gout[pass][k], false) < 0)
                    return 1;
                patch_rect(1, width, height, &x0, &y0, &x1, &y1);
                measure(c.out, width, x0, y0, x1, y1, &got[pass][k]);
                sheet_add(&s, c.out, width, x0, y0);
                sheet_add_diff(&sg, c.out, plain, width, x0, y0, 12.0f, 2);
            }
        }
        for (int k = 0; k < 4; k++)
            printf("%-14s %9.5f %9.2f %8s | %9.5f %9.2f %8s\n",
                   craw_grain_gauge_name(show[k]),
                   got[1][k].sd_luma, got[1][k].fwhm,
                   craw_grain_model_name(gout[1][k].model_used),
                   got[0][k].sd_luma, got[0][k].fwhm,
                   craw_grain_model_name(gout[0][k].model_used));
        path_for(path, sizeof(path), "sheet_paths.png");
        sheet_write(&s, path);
        path_for(path, sizeof(path), "sheet_paths_grain.png");
        sheet_write(&sg, path);
        printf("  (top row: output path; bottom row: source path)\n");
    }

    /* ---- AV1 against H.274 at the same asked-for blob size ---- */
    if (want(only, "models")) {
        printf("\n=== AV1 against H.274, same gauge, output path ===\n");
        printf("%-14s %6s | %9s %9s %8s | %9s %9s %8s\n", "gauge", "want px",
               "av1 sd", "av1 px", "av1 c/l", "h274 sd", "h274 px", "h274 c/l");
        struct sheet s, sg;
        sheet_init(&s, cell, 5, 2);
        sheet_init(&sg, cell, 5, 2);
        static const int show[] = {
            CRAW_GRAIN_GAUGE_REGULAR8, CRAW_GRAIN_GAUGE_SUPER8,
            CRAW_GRAIN_GAUGE_16MM, CRAW_GRAIN_GAUGE_SUPER35,
            CRAW_GRAIN_GAUGE_IMAX,
        };
        struct meas got[2][5];
        struct craw_grain_out gout[2][5];
        for (int mdl = 0; mdl < 2; mdl++) {
            for (int k = 0; k < 5; k++) {
                struct craw_grain_params p = craw_grain_default_params();
                int x0, y0, x1, y1;
                p.gauge = show[k];
                p.stock = CRAW_GRAIN_STOCK_V3_500T;
                p.response = CRAW_GRAIN_RESPONSE_PRINT;
                p.model = mdl ? CRAW_GRAIN_MODEL_H274 : CRAW_GRAIN_MODEL_AV1;
                p.width = width;
                p.height = height;
                p.seed = 0x1234;
                if (render(&c, &p, &gout[mdl][k], false) < 0)
                    return 1;
                patch_rect(1, width, height, &x0, &y0, &x1, &y1);
                measure(c.out, width, x0, y0, x1, y1, &got[mdl][k]);
                sheet_add(&s, c.out, width, x0, y0);
                sheet_add_diff(&sg, c.out, plain, width, x0, y0, 12.0f, 2);
            }
        }
        for (int k = 0; k < 5; k++) {
            double c0 = got[0][k].sd[1] > 0
                ? 0.5 * (got[0][k].sd[0] + got[0][k].sd[2]) / got[0][k].sd[1] : 0;
            double c1 = got[1][k].sd[1] > 0
                ? 0.5 * (got[1][k].sd[0] + got[1][k].sd[2]) / got[1][k].sd[1] : 0;
            printf("%-14s %6.2f | %9.5f %9.2f %8.2f | %9.5f %9.2f %8.2f\n",
                   craw_grain_gauge_name(show[k]), gout[0][k].blob_px,
                   got[0][k].sd_luma, got[0][k].fwhm, c0,
                   got[1][k].sd_luma, got[1][k].fwhm, c1);
        }
        path_for(path, sizeof(path), "sheet_models.png");
        sheet_write(&s, path);
        path_for(path, sizeof(path), "sheet_models_grain.png");
        sheet_write(&sg, path);
        printf("  (top row: AV1; bottom row: H.274)\n");
    }

    /* ---- the response curves, measured across all four patches ---- */
    if (want(only, "response")) {
        printf("\n=== responses, measured on each patch ===\n");
        printf("  (each cell is wanted/measured green channel sd)\n");
        printf("%-10s", "response");
        for (int i = 0; i < 4; i++)
            printf("  %4.0f%% enc %.3f ", patch_value[i] * 100, ref[i].mean[1]);
        printf("\n");
        struct sheet s;
        /* Four responses by four patches at the full cell would just clear
         * a megapixel, which is the line above which a PNG stops being
         * worth the wait. */
        sheet_init(&s, 224, 4, CRAW_GRAIN_RESPONSE_NB);
        for (int r = 0; r < CRAW_GRAIN_RESPONSE_NB; r++) {
            struct craw_grain_params p = craw_grain_default_params();
            struct craw_grain_out o;
            p.stock = CRAW_GRAIN_STOCK_COARSE;
            p.gauge = CRAW_GRAIN_GAUGE_16MM;
            p.response = r;
            p.width = width;
            p.height = height;
            p.seed = 0x1234;
            if (render(&c, &p, &o, false) < 0)
                return 1;
            printf("%-10s", craw_grain_response_name(r));
            for (int i = 0; i < 4; i++) {
                int x0, y0, x1, y1;
                struct meas m;
                patch_rect(i, width, height, &x0, &y0, &x1, &y1);
                measure(c.out, width, x0, y0, x1, y1, &m);
                float want = craw_grain_sigma_at(&o, (float)ref[i].mean[1]);
                printf("  %8.5f/%-8.5f", want, m.sd[1]);
                sheet_add(&s, c.out, width, x0, y0);
            }
            printf("\n");
        }
        path_for(path, sizeof(path), "sheet_response.png");
        sheet_write(&s, path);
        printf("  (one row per response; columns 2%%, 18%%, 50%%, 90%%)\n");
    }

    /* ---- two seeds, and the amount control ---- */
    if (want(only, "boil")) {
        printf("\n=== two seeds and amount 0 ===\n");
        struct sheet s, sg;
        sheet_init(&s, cell, 3, 1);
        sheet_init(&sg, cell, 3, 1);
        uint16_t *first = NULL;
        double diff = 0;
        for (int k = 0; k < 3; k++) {
            struct craw_grain_params p = craw_grain_default_params();
            struct craw_grain_out o;
            struct meas m;
            int x0, y0, x1, y1;
            p.gauge = CRAW_GRAIN_GAUGE_16MM;
            p.stock = CRAW_GRAIN_STOCK_COARSE;
            p.width = width;
            p.height = height;
            p.seed = k ? 0x9876 : 0x1234;
            if (k == 2)
                p.amount = 0.0f;
            if (render(&c, &p, &o, false) < 0)
                return 1;
            patch_rect(1, width, height, &x0, &y0, &x1, &y1);
            measure(c.out, width, x0, y0, x1, y1, &m);
            printf("  seed %#lx amount %.1f: sd %.5f, enabled %d\n",
                   (unsigned long)p.seed, p.amount, m.sd_luma, o.enabled);
            if (!k) {
                first = malloc((size_t)width * height * 4 * sizeof(uint16_t));
                memcpy(first, c.out, (size_t)width * height * 4 * sizeof(uint16_t));
            } else if (k == 1 && first) {
                double acc = 0;
                long n = 0;
                for (int y = y0; y < y1; y++)
                    for (int x = x0; x < x1; x++) {
                        double a = first[((size_t)y * width + x) * 4 + 1] / 65535.0;
                        double b = c.out[((size_t)y * width + x) * 4 + 1] / 65535.0;
                        acc += (a - b) * (a - b);
                        n++;
                    }
                diff = sqrt(acc / n);
            }
            sheet_add(&s, c.out, width, x0, y0);
            sheet_add_diff(&sg, c.out, plain, width, x0, y0, 12.0f, 2);
        }
        free(first);
        printf("  rms difference between the two seeds: %.5f\n", diff);
        path_for(path, sizeof(path), "sheet_boil.png");
        sheet_write(&s, path);
        path_for(path, sizeof(path), "sheet_boil_grain.png");
        sheet_write(&sg, path);
        printf("  (seed A, seed B, amount 0)\n");
    }


    /* ---- block structure, magnified enough to see a seam ---- */
    if (want(only, "blocks")) {
        printf("\n=== block structure, grain only, 4x ===\n");
        struct sheet s;
        sheet_init(&s, 256, 4, 1);
        static const struct { int gauge, model; } cases[] = {
            { CRAW_GRAIN_GAUGE_REGULAR8, CRAW_GRAIN_MODEL_AV1 },
            { CRAW_GRAIN_GAUGE_REGULAR8, CRAW_GRAIN_MODEL_H274 },
            { CRAW_GRAIN_GAUGE_SUPER35,  CRAW_GRAIN_MODEL_AV1 },
            { CRAW_GRAIN_GAUGE_SUPER35,  CRAW_GRAIN_MODEL_H274 },
        };
        for (size_t k = 0; k < sizeof(cases) / sizeof(*cases); k++) {
            struct craw_grain_params p = craw_grain_default_params();
            struct craw_grain_out o;
            int x0, y0, x1, y1;
            p.gauge = cases[k].gauge;
            p.stock = CRAW_GRAIN_STOCK_COARSE;
            p.response = CRAW_GRAIN_RESPONSE_FLAT;
            p.model = cases[k].model;
            p.width = width;
            p.height = height;
            p.seed = 0x1234;
            if (render(&c, &p, &o, false) < 0)
                return 1;
            patch_rect(1, width, height, &x0, &y0, &x1, &y1);
            /* Start on a multiple of 32 so a block boundary is in frame. */
            sheet_add_diff(&s, c.out, plain, width, (x0 + 31) & ~31,
                           (y0 + 31) & ~31, 8.0f, 4);
            printf("  %-14s %-6s blob %.2f px\n",
                   craw_grain_gauge_name(cases[k].gauge),
                   craw_grain_model_name(o.model_used), o.blob_clamped);
        }
        path_for(path, sizeof(path), "sheet_blocks.png");
        sheet_write(&s, path);
        printf("  (AV1 coarse, H.274 coarse, AV1 fine, H.274 fine;"
               " 64x64 pixels each, block grid starts at the left edge)\n");
    }

    /* ---- how coarse can each model go ---- */
    if (want(only, "ceiling")) {
        printf("\n=== largest blob each model will build ===\n");
        printf("%8s | %10s %10s | %10s %10s\n", "asked", "av1 px",
               "av1 sd/want", "h274 px", "h274 sd/want");
        for (float mult = 1.0f; mult <= 8.01f; mult *= 1.4142f) {
            float px[2], rat[2];
            for (int mdl = 0; mdl < 2; mdl++) {
                struct craw_grain_params p = craw_grain_default_params();
                struct craw_grain_out o;
                struct meas m;
                int x0, y0, x1, y1;
                p.gauge = CRAW_GRAIN_GAUGE_SUPER35;
                p.stock = CRAW_GRAIN_STOCK_V3_500T;
                p.response = CRAW_GRAIN_RESPONSE_FLAT;
                p.model = mdl ? CRAW_GRAIN_MODEL_H274 : CRAW_GRAIN_MODEL_AV1;
                p.size = mult;
                p.width = width;
                p.height = height;
                p.seed = 0x1234;
                if (render(&c, &p, &o, false) < 0)
                    return 1;
                patch_rect(1, width, height, &x0, &y0, &x1, &y1);
                measure(c.out, width, x0, y0, x1, y1, &m);
                px[mdl] = (float)m.fwhm;
                float want_sd = craw_grain_sigma_at(&o, (float)ref[1].mean[1]);
                rat[mdl] = want_sd > 0 ? (float)m.sd[1] / want_sd : 0.0f;
            }
            struct craw_grain_params q = craw_grain_default_params();
            struct craw_grain_out qo;
            q.gauge = CRAW_GRAIN_GAUGE_SUPER35;
            q.stock = CRAW_GRAIN_STOCK_V3_500T;
            q.size = mult;
            q.width = width;
            q.height = height;
            q.seed = 1;
            craw_grain_build(&q, &qo);
            printf("%8.2f | %10.2f %10.2f | %10.2f %10.2f\n",
                   qo.blob_px, px[0], rat[0], px[1], rat[1]);
        }
    }

    /* ---- the real frame ---- */
    if (frame && want(only, "frame")) {
        struct image real = {0};
        printf("\n=== real frame ===\n");
        if (image_pfm(&real, frame, width) < 0)
            return 1;
        printf("  %s -> %dx%d\n", frame, real.w, real.h);
        if (ctx_upload(&c, &real) < 0 || ctx_target(&c, real.w, real.h) < 0)
            return 1;
        static const struct { const char *name; int gauge; float amount; }
        cases[] = {
            { "frame_none.nut",     CRAW_GRAIN_GAUGE_SUPER35, 0.0f },
            { "frame_super35.nut",  CRAW_GRAIN_GAUGE_SUPER35, 1.0f },
            { "frame_16mm.nut",     CRAW_GRAIN_GAUGE_16MM,    1.0f },
            { "frame_super8.nut",   CRAW_GRAIN_GAUGE_SUPER8,  1.0f },
        };
        for (size_t k = 0; k < sizeof(cases) / sizeof(*cases); k++) {
            struct craw_grain_params p = craw_grain_default_params();
            struct craw_grain_out o;
            p.gauge = cases[k].gauge;
            p.stock = CRAW_GRAIN_STOCK_V3_500T;
            p.amount = cases[k].amount;
            p.width = real.w;
            p.height = real.h;
            p.seed = 0x1234;
            if (render(&c, cases[k].amount > 0 ? &p : NULL, &o, false) < 0)
                return 1;
            path_for(path, sizeof(path), cases[k].name);
            write_nut(path, c.out, real.w, real.h);
            printf("  wrote %s (%dx%d raw rgb48le)\n", path, real.w, real.h);
        }
        printf("  to look at one: ffmpeg -i <name>.nut"
               " -vf crop=640:480:560:300 -frames:v 1 crop.png\n");
        image_free(&real);
    }

    free(plain);
    image_free(&chart);
    ctx_free(&c);
    return 0;
}
