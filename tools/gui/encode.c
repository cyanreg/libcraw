/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* The video exporter: a range of frames, the look, one file.
 *
 * The picture is the still export's picture. Same renderer, same hooks,
 * same crop rectangle, same display-referred sRGB output - a frame of the
 * movie and the same frame written on its own with --export are the same
 * numbers, which is the property the tests lean on. What differs is where
 * it is rendered to.
 *
 * On a Vulkan encoder the target is the encoder's own image. An
 * AVHWFramesContext is built on the AVVulkanDeviceContext decode.c already
 * made out of libplacebo's device, so the encoder allocates its pictures on
 * the one VkDevice everything else lives on; libplacebo wraps each one,
 * renders straight into it - doing the RGB to YCbCr conversion and the
 * chroma subsampling on the way - and hands it back through the AVVkFrame
 * timeline semaphore. Nothing is copied and nothing waits for the device:
 * the frame goes to the encoder carrying the semaphore value it has to wait
 * for, which is exactly what libavutil/hwcontext_vulkan.h documents and
 * exactly what the decoder does to us in the other direction.
 *
 * On a software encoder the target is the RGBA16 texture the still export
 * reads back, and libswscale takes it to the deepest format the encoder
 * accepts. Two textures and two buffers are kept, so the read-back of the
 * next frame is already in flight while the scaler and the encoder work on
 * this one.
 *
 * Audio is remuxed rather than re-encoded: the clip's own packets, cut to
 * the exported range on a sample boundary where the codec is PCM, which is
 * what every camera here records. */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libavutil/time.h>
#include <libswscale/swscale.h>

/* After <vulkan/vulkan.h>, which view.h has already pulled in: that is what
 * turns on libplacebo's AVVkFrame support. */
#include <libplacebo/utils/libav.h>

#include "view.h"
#include "encode.h"

/* Frames in flight on the software path. Two is enough to keep the GPU
 * busy: one is being read back while the other is scaled and encoded. */
#define VIEW_ENC_SLOTS 2

struct view_enc_slot {
    pl_tex tex;
    uint16_t *rgba;
    size_t values;
    bool busy;                  /* a read-back is in flight or ready */
    int64_t index;              /* the frame it holds */
};

struct view_encode {
    struct view_gpu *g;
    bool verbose, report;

    AVFormatContext *oc;
    AVStream *vs;
    AVCodecContext *avc;
    const AVCodec *codec;
    AVPacket *pkt;
    bool header;

    /* The Vulkan path: frames the encoder owns, allocated on our device. */
    bool hw;
    AVBufferRef *frames;
    enum AVPixelFormat sw_fmt;

    /* The software path. */
    struct view_enc_slot slot[VIEW_ENC_SLOTS];
    int cur;
    AVFrame *sf;                /* what the encoder is handed */
    AVFrame *ss;                /* the RGBA64 read-back, as libswscale input */
    SwsContext *sws;

    int w, h;
    int req_w, req_h;           /* --size, zero for the crop's own */
    bool req_audio;
    AVRational fr;
    int64_t start, end, next;
    int64_t encoded;
    /* The output cannot be opened until a frame has been decoded: the
     * clip's size, and with it the crop rectangle's, is only known then. */
    bool opened;
    bool sought, draining, done;

    enum AVColorRange range;
    enum AVColorSpace csp;
    enum AVChromaLocation chroma;

    /* Audio, remuxed. `pcm_bytes` is the size of one sample frame, which is
     * what lets the range be cut on a sample; zero for a codec with no such
     * thing, where the cut is a packet boundary instead. */
    AVFormatContext *aic;
    AVPacket *apkt;
    int nb_audio;
    int a_in[2], a_out[2];
    int pcm_bytes[2];
    int rate[2];
    int64_t a_first[2], a_want[2], a_written[2];
    bool audio_eof;

    int64_t t0, last_report;
    double fps;
};

/* ------------------------------------------------------- the encoder list */

static bool view_enc_is_vulkan(const AVCodec *c)
{
    const AVCodecHWConfig *hw;

    for (int i = 0; (hw = avcodec_get_hw_config(c, i)); i++)
        if (hw->device_type == AV_HWDEVICE_TYPE_VULKAN ||
            hw->pix_fmt == AV_PIX_FMT_VULKAN)
            return true;

    /* A name is not evidence on its own, but an encoder that calls itself
     * Vulkan and reports nothing is not one we would drive by uploading
     * system memory into it either. */
    return strstr(c->name, "_vulkan") != NULL;
}

void view_encode_list(void)
{
    const AVCodec *c;
    void *it = NULL;

    printf("video encoders in this build. V: takes Vulkan frames, "
           "H: other hardware\n\n");

    while ((c = av_codec_iterate(&it))) {
        const enum AVPixelFormat *fmts = NULL;
        int nb = 0;
        char kind;

        if (!av_codec_is_encoder(c) || c->type != AVMEDIA_TYPE_VIDEO)
            continue;

        if (view_enc_is_vulkan(c))
            kind = 'V';
        else if (c->capabilities & AV_CODEC_CAP_HARDWARE)
            kind = 'H';
        else
            kind = ' ';

        printf("%c %-22s %s\n", kind, c->name, c->long_name ? c->long_name : "");

        /* What it takes. A Vulkan encoder says nothing but "vulkan" here:
         * the software format behind it is a property of the frames
         * context, not of the codec, and cineview finds it by asking the
         * encoder to open with each one in turn. */
        if (avcodec_get_supported_config(NULL, c, AV_CODEC_CONFIG_PIX_FORMAT,
                                         0, (const void **)&fmts, &nb) >= 0 &&
            fmts && nb > 0) {
            printf("  %22s", "");
            for (int i = 0; i < nb; i++)
                printf("%s%s", i ? " " : "", av_get_pix_fmt_name(fmts[i]));
            printf("\n");
        }
    }
}

/* ----------------------------------------------------------- pixel formats */

/* What we would rather hand a software encoder, best first, by what the
 * codec is for. A lossless or an intermediate codec gets deep RGB, because
 * that is what the export renders and anything else is a conversion nobody
 * asked for; ProRes gets the ten bit 4:2:2 it was designed around; a
 * delivery codec gets the deepest 4:2:0 it will take. */
static const enum AVPixelFormat view_enc_want_rgb[] = {
    AV_PIX_FMT_GBRP16LE, AV_PIX_FMT_GBRP16BE,
    AV_PIX_FMT_GBRP14LE, AV_PIX_FMT_GBRP12LE, AV_PIX_FMT_GBRP10LE,
    AV_PIX_FMT_RGB48LE,  AV_PIX_FMT_RGB48BE,  AV_PIX_FMT_GBRP,
    AV_PIX_FMT_YUV444P16LE, AV_PIX_FMT_YUV444P12LE, AV_PIX_FMT_YUV444P10LE,
    AV_PIX_FMT_YUV444P,
    AV_PIX_FMT_YUV422P10LE, AV_PIX_FMT_YUV422P,
    AV_PIX_FMT_YUV420P10LE, AV_PIX_FMT_YUV420P,
};

static const enum AVPixelFormat view_enc_want_422[] = {
    AV_PIX_FMT_YUV422P10LE, AV_PIX_FMT_YUV444P10LE,
    AV_PIX_FMT_YUV422P12LE, AV_PIX_FMT_YUV444P12LE,
    AV_PIX_FMT_YUVA444P10LE,
    AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV444P,
    AV_PIX_FMT_GBRP10LE, AV_PIX_FMT_GBRP,
};

static const enum AVPixelFormat view_enc_want_420[] = {
    AV_PIX_FMT_YUV420P10LE, AV_PIX_FMT_YUV420P12LE, AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_P010LE,      AV_PIX_FMT_NV12,
    AV_PIX_FMT_YUV422P10LE, AV_PIX_FMT_YUV422P,
    AV_PIX_FMT_YUV444P10LE, AV_PIX_FMT_YUV444P,
    AV_PIX_FMT_GBRP10LE,    AV_PIX_FMT_GBRP,
};

/* The same three for the Vulkan path, where the choice is the frames
 * context's software format rather than the codec's pixel format. */
static const enum AVPixelFormat view_enc_vk_rgb[] = {
    AV_PIX_FMT_GBRP16, AV_PIX_FMT_GBRP14, AV_PIX_FMT_GBRP12, AV_PIX_FMT_GBRP10,
    AV_PIX_FMT_GBRP,
    AV_PIX_FMT_YUV444P16, AV_PIX_FMT_YUV444P12, AV_PIX_FMT_YUV444P10,
    AV_PIX_FMT_YUV444P,
    AV_PIX_FMT_YUV422P12, AV_PIX_FMT_YUV422P10, AV_PIX_FMT_YUV422P,
    AV_PIX_FMT_RGBA64,
};

static const enum AVPixelFormat view_enc_vk_422[] = {
    AV_PIX_FMT_YUV422P10, AV_PIX_FMT_YUV444P10,
    AV_PIX_FMT_YUV422P12, AV_PIX_FMT_YUV444P12,
    AV_PIX_FMT_YUV422P,   AV_PIX_FMT_YUV444P,
};

static const enum AVPixelFormat view_enc_vk_420[] = {
    AV_PIX_FMT_P010, AV_PIX_FMT_NV12, AV_PIX_FMT_P012, AV_PIX_FMT_P016,
    AV_PIX_FMT_YUV420P10, AV_PIX_FMT_YUV420P,
};

/* Which of the three a codec belongs to. Nothing here is a guess about a
 * particular encoder: it is what the format is for. */
static void view_enc_wanted(enum AVCodecID id, bool hw,
                            const enum AVPixelFormat **list, size_t *n)
{
    switch (id) {
    case AV_CODEC_ID_FFV1:
    case AV_CODEC_ID_HUFFYUV:
    case AV_CODEC_ID_FFVHUFF:
    case AV_CODEC_ID_UTVIDEO:
    case AV_CODEC_ID_MAGICYUV:
    case AV_CODEC_ID_RAWVIDEO:
        *list = hw ? view_enc_vk_rgb : view_enc_want_rgb;
        *n    = hw ? CRAW_ARRAY_ELEMS(view_enc_vk_rgb)
                   : CRAW_ARRAY_ELEMS(view_enc_want_rgb);
        return;
    case AV_CODEC_ID_PRORES:
    case AV_CODEC_ID_APV:
    case AV_CODEC_ID_DNXHD:
    case AV_CODEC_ID_CFHD:
    case AV_CODEC_ID_JPEG2000:
        *list = hw ? view_enc_vk_422 : view_enc_want_422;
        *n    = hw ? CRAW_ARRAY_ELEMS(view_enc_vk_422)
                   : CRAW_ARRAY_ELEMS(view_enc_want_422);
        return;
    default:
        *list = hw ? view_enc_vk_420 : view_enc_want_420;
        *n    = hw ? CRAW_ARRAY_ELEMS(view_enc_vk_420)
                   : CRAW_ARRAY_ELEMS(view_enc_want_420);
        return;
    }
}

static enum AVPixelFormat view_enc_pick_sw(const AVCodec *codec)
{
    const enum AVPixelFormat *list = NULL, *want;
    size_t nwant;
    int n = 0;

    view_enc_wanted(codec->id, false, &want, &nwant);

    /* An encoder that reports no list takes anything, so the first choice
     * is ours to make. */
    if (avcodec_get_supported_config(NULL, codec, AV_CODEC_CONFIG_PIX_FORMAT,
                                     0, (const void **)&list, &n) < 0 || !list)
        return want[0];

    for (size_t i = 0; i < nwant; i++)
        for (int j = 0; j < n; j++)
            if (list[j] == want[i])
                return want[i];

    return n > 0 ? list[0] : AV_PIX_FMT_NONE;
}

static bool view_enc_is_rgb(enum AVPixelFormat fmt)
{
    const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(fmt);

    return d && (d->flags & AV_PIX_FMT_FLAG_RGB);
}

/* ---------------------------------------------------------- the hw frames */

/* A pool of encoder pictures on the viewer's device. Nothing is
 * preallocated: the pool grows to whatever the encoder holds on to, so
 * frames stay in flight without a device wait between them. */
static int view_enc_frames(struct view_encode *e, struct view_state *st,
                           enum AVPixelFormat sw, AVBufferRef **out)
{
    AVBufferRef *ref;
    AVHWFramesContext *fc;
    AVVulkanFramesContext *vk;
    int ret;

    if (!st->dec.hw_device)
        return CRAW_ERROR(ENODEV);

    ref = av_hwframe_ctx_alloc(st->dec.hw_device);
    if (!ref)
        return CRAW_ERROR(ENOMEM);

    fc = (AVHWFramesContext *)ref->data;
    vk = fc->hwctx;

    fc->format            = AV_PIX_FMT_VULKAN;
    fc->sw_format         = sw;
    fc->width             = e->w;
    fc->height            = e->h;
    fc->initial_pool_size = 0;

    /* libplacebo has to be able to render into these, which means a colour
     * attachment; the rest is what libavutil would have chosen anyway and
     * is spelled out so the intent is in one place. The encode usage the
     * encoder itself needs is added by av_hwframe_ctx_init(). */
    vk->usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    ret = av_hwframe_ctx_init(ref);
    if (ret < 0) {
        av_buffer_unref(&ref);
        return ret;
    }

    *out = ref;

    return 0;
}

/* --------------------------------------------------------- opening it all */

static void view_enc_colours(struct view_encode *e, enum AVPixelFormat fmt)
{
    /* What the export really is: libplacebo renders into a display referred
     * sRGB target, so the samples are BT.709 primaries with the IEC
     * 61966-2-1 transfer. An RGB encoder keeps them at full range, which is
     * what the still writer does too and what makes a lossless video of one
     * frame the same numbers as the still of it; a YCbCr one gets the
     * BT.709 matrix and the studio swing every muxer and player assumes of
     * a movie. */
    if (view_enc_is_rgb(fmt)) {
        e->csp    = AVCOL_SPC_RGB;
        e->range  = AVCOL_RANGE_JPEG;
        e->chroma = AVCHROMA_LOC_UNSPECIFIED;
    } else {
        e->csp    = AVCOL_SPC_BT709;
        e->range  = AVCOL_RANGE_MPEG;
        e->chroma = AVCHROMA_LOC_LEFT;
    }
}

static void view_enc_tag_frame(const struct view_encode *e, AVFrame *f)
{
    f->color_range     = e->range;
    f->colorspace      = e->csp;
    f->color_primaries = AVCOL_PRI_BT709;
    f->color_trc       = AVCOL_TRC_IEC61966_2_1;
    /* Where the chroma samples sit. Left siting is what H.264, HEVC and
     * every muxer's default mean by 4:2:0 and 4:2:2, and saying so is not
     * cosmetic: libplacebo subsamples where it is told to, and a frame that
     * left it unspecified came back from the decoder half a chroma sample
     * across, which measured four dB against the still. */
    f->chroma_location = e->chroma;
}

/* Builds and opens the encoder for one candidate format. The whole context
 * is thrown away and rebuilt per candidate, because an AVCodecContext that
 * failed to open cannot be opened again. */
static int view_enc_try(struct view_encode *e, struct view_state *st,
                        enum AVPixelFormat fmt, const char *enc_opts)
{
    AVDictionary *opts = NULL;
    AVCodecContext *avc = NULL;
    AVBufferRef *frames = NULL;
    int ret;

    if (e->hw) {
        ret = view_enc_frames(e, st, fmt, &frames);
        if (ret < 0) {
            if (e->verbose)
                fprintf(stderr, "no Vulkan frames for %s: %s\n",
                        av_get_pix_fmt_name(fmt), av_err2str(ret));
            return ret;
        }
    }

    avc = avcodec_alloc_context3(e->codec);
    if (!avc) {
        ret = CRAW_ERROR(ENOMEM);
        goto fail;
    }

    avc->width     = e->w;
    avc->height    = e->h;
    avc->time_base = av_inv_q(e->fr);
    avc->framerate = e->fr;
    avc->sample_aspect_ratio = (AVRational) { 1, 1 };

    view_enc_colours(e, fmt);
    avc->color_range     = e->range;
    avc->colorspace      = e->csp;
    avc->color_primaries = AVCOL_PRI_BT709;
    avc->color_trc       = AVCOL_TRC_IEC61966_2_1;
    avc->chroma_sample_location = e->chroma;

    if (e->hw) {
        avc->pix_fmt    = AV_PIX_FMT_VULKAN;
        avc->sw_pix_fmt = fmt;
        avc->hw_frames_ctx = av_buffer_ref(frames);
        if (!avc->hw_frames_ctx) {
            ret = CRAW_ERROR(ENOMEM);
            goto fail;
        }
    } else {
        avc->pix_fmt = fmt;
    }

    if (e->oc->oformat->flags & AVFMT_GLOBALHEADER)
        avc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    /* Last, so that what was asked for wins over what we would have
     * chosen. */
    if (enc_opts && enc_opts[0]) {
        ret = av_dict_parse_string(&opts, enc_opts, "=", ",", 0);
        if (ret < 0) {
            fprintf(stderr, "--enc: cannot parse `%s`\n", enc_opts);
            ret = CRAW_ERROR(EINVAL);
            goto fail;
        }
    }

    ret = avcodec_open2(avc, e->codec, &opts);
    if (ret < 0) {
        if (e->verbose)
            fprintf(stderr, "%s with %s: %s\n", e->codec->name,
                    av_get_pix_fmt_name(fmt), av_err2str(ret));
        goto fail;
    }
    if (av_dict_count(opts))
        fprintf(stderr, "the %s encoder ignored an option\n", e->codec->name);

    av_dict_free(&opts);
    e->avc    = avc;
    e->frames = frames;
    e->sw_fmt = fmt;

    return 0;

fail:
    av_dict_free(&opts);
    avcodec_free_context(&avc);
    av_buffer_unref(&frames);

    return ret;
}

/* --------------------------------------------------------------- audio in */

/* Which audio streams go into the file. The rule is the viewer's own: one
 * stereo stream is the whole of it, and four mono tracks means the last two
 * are the stereo pair. A single four channel stream is copied whole -
 * playback picks channels 3 and 4 out of it, but an export that dropped the
 * other two would be throwing away a recording. */
static void view_enc_audio_pick(struct view_encode *e)
{
    int audio[64], nb_audio = 0;
    int mono = 0, quad = -1;

    for (unsigned i = 0; i < e->aic->nb_streams && nb_audio < 64; i++) {
        AVCodecParameters *cp = e->aic->streams[i]->codecpar;

        if (cp->codec_type != AVMEDIA_TYPE_AUDIO)
            continue;
        if (cp->ch_layout.nb_channels == 1)
            mono++;
        else if (cp->ch_layout.nb_channels >= 4 && quad < 0)
            quad = nb_audio;
        audio[nb_audio++] = (int)i;
    }

    if (!nb_audio)
        return;

    if (nb_audio >= 4 && mono >= 4) {
        e->a_in[0] = audio[nb_audio - 2];
        e->a_in[1] = audio[nb_audio - 1];
        e->nb_audio = 2;
    } else if (quad >= 0) {
        e->a_in[0] = audio[quad];
        e->nb_audio = 1;
    } else {
        e->a_in[0] = audio[0];
        e->nb_audio = 1;
    }
}

/* Opens the clip a second time for its audio and adds the streams to the
 * output. A container that will not carry the codec is not an error: the
 * video is still worth writing, so the audio is dropped with a word. */
static int view_enc_audio_open(struct view_encode *e, struct view_state *st)
{
    int ret;

    ret = avformat_open_input(&e->aic, st->cfg.input, NULL, NULL);
    if (ret < 0)
        return 0;
    if (avformat_find_stream_info(e->aic, NULL) < 0)
        goto drop;

    view_enc_audio_pick(e);
    if (!e->nb_audio)
        goto drop;

    for (int i = 0; i < e->nb_audio; i++) {
        AVStream *is = e->aic->streams[e->a_in[i]];
        AVCodecParameters *cp = is->codecpar;
        AVStream *os;
        int bits;

        if (cp->sample_rate <= 0)
            goto drop;
        if (avformat_query_codec(e->oc->oformat, cp->codec_id,
                                 FF_COMPLIANCE_NORMAL) <= 0) {
            fprintf(stderr, "%s cannot carry %s audio; writing video only\n",
                    e->oc->oformat->name, avcodec_get_name(cp->codec_id));
            goto drop;
        }

        os = avformat_new_stream(e->oc, NULL);
        if (!os)
            goto drop;
        ret = avcodec_parameters_copy(os->codecpar, cp);
        if (ret < 0)
            goto drop;
        os->codecpar->codec_tag = 0;
        os->time_base = (AVRational) { 1, cp->sample_rate };

        /* A camera writes LPCM with no channel layout at all, which the
         * QuickTime muxer merely grumbles about and the MP4 one refuses to
         * write a 'chan' box for at trailer time. The default layout for
         * that many channels is what the file means anyway. */
        if (os->codecpar->ch_layout.order == AV_CHANNEL_ORDER_UNSPEC)
            av_channel_layout_default(&os->codecpar->ch_layout,
                                      cp->ch_layout.nb_channels);

        e->a_out[i] = os->index;
        e->rate[i]  = cp->sample_rate;

        /* PCM, which is what every camera here records, has a sample frame
         * of a known size, so the range can be cut on a sample rather than
         * on a packet and the audio lines up with the picture exactly. */
        bits = av_get_bits_per_sample(cp->codec_id);
        e->pcm_bytes[i] = bits && !(bits % 8)
                        ? bits / 8 * cp->ch_layout.nb_channels : 0;

        /* The first sample of the range and how many of them it holds, in
         * the stream's own sample rate. Integer arithmetic throughout: at
         * 48 kHz these come out exact for every frame rate a camera
         * writes, 29.97 included. */
        e->a_first[i] = av_rescale(e->start * e->fr.den, cp->sample_rate, e->fr.num);
        e->a_want[i]  = av_rescale((e->end - e->start + 1) * e->fr.den,
                                   cp->sample_rate, e->fr.num);
    }

    /* Start reading near the range rather than from the top of the file. */
    if (e->start > 0) {
        AVStream *is = e->aic->streams[e->a_in[0]];
        int64_t ts = av_rescale_q(e->a_first[0], (AVRational) { 1, e->rate[0] },
                                  is->time_base);

        av_seek_frame(e->aic, e->a_in[0], ts, AVSEEK_FLAG_BACKWARD);
    }

    return 0;

drop:
    /* A stream added before the refusal stays: an AVFormatContext has no
     * way to drop one. It is simply never written to, and a muxer with an
     * empty stream still writes a valid file. */
    e->nb_audio = 0;
    avformat_close_input(&e->aic);

    return 0;
}

/* Writes the audio of the range up to `until` seconds into it, so the
 * interleaver never has to hold more than a frame's worth. */
static int view_enc_audio_pump(struct view_encode *e, double until)
{
    int ret;

    while (e->nb_audio && !e->audio_eof) {
        AVStream *is;
        AVRational sr;
        int slot = -1;
        int64_t first, n, s0, s1, keep0, keep1;
        bool all_done = true;

        for (int i = 0; i < e->nb_audio; i++)
            if (e->a_written[i] < e->a_want[i])
                all_done = false;
        if (all_done) {
            e->audio_eof = true;
            break;
        }

        ret = av_read_frame(e->aic, e->apkt);
        if (ret < 0) {
            e->audio_eof = true;
            break;
        }

        for (int i = 0; i < e->nb_audio; i++)
            if (e->apkt->stream_index == e->a_in[i])
                slot = i;
        if (slot < 0 || e->apkt->pts == AV_NOPTS_VALUE) {
            av_packet_unref(e->apkt);
            continue;
        }

        is = e->aic->streams[e->a_in[slot]];
        sr = (AVRational) { 1, e->rate[slot] };
        first = e->a_first[slot];

        /* Everything in sample units of this stream, which is also the
         * output stream's time base. */
        s0 = av_rescale_q(e->apkt->pts, is->time_base, sr);
        n  = e->pcm_bytes[slot] ? e->apkt->size / e->pcm_bytes[slot]
                                : av_rescale_q(e->apkt->duration, is->time_base, sr);
        s1 = s0 + (n > 0 ? n : 1);

        if (s1 <= first || s0 >= first + e->a_want[slot]) {
            av_packet_unref(e->apkt);
            continue;
        }

        keep0 = CRAW_MAX(s0, first);
        keep1 = CRAW_MIN(s1, first + e->a_want[slot]);

        if (e->pcm_bytes[slot] && (keep0 != s0 || keep1 != s1)) {
            /* Cut to the sample. The packet is not made writable: moving
             * the data pointer and trimming the size is all a copy of it
             * into the muxer needs. */
            e->apkt->data += (keep0 - s0) * e->pcm_bytes[slot];
            e->apkt->size  = (int)((keep1 - keep0) * e->pcm_bytes[slot]);
        }

        e->apkt->stream_index = e->a_out[slot];
        e->apkt->pts = e->apkt->dts = keep0 - first;
        e->apkt->duration = keep1 - keep0;
        e->apkt->pos = -1;
        av_packet_rescale_ts(e->apkt, sr,
                             e->oc->streams[e->a_out[slot]]->time_base);

        e->a_written[slot] += keep1 - keep0;

        ret = av_interleaved_write_frame(e->oc, e->apkt);
        av_packet_unref(e->apkt);
        if (ret < 0)
            return ret;

        if ((double)(keep1 - first) / e->rate[slot] >= until)
            break;
    }

    return 0;
}

/* ------------------------------------------------------------- the target */

/* The software path's render target: the RGBA16 texture the still export
 * uses, one per slot. */
static int view_enc_slot_tex(struct view_encode *e, struct view_enc_slot *s)
{
    pl_fmt fmt;

    if (s->tex)
        return 0;

    fmt = pl_find_named_fmt(e->g->gpu, "rgba16");
    if (!fmt || !(fmt->caps & PL_FMT_CAP_RENDERABLE) ||
        !(fmt->caps & PL_FMT_CAP_HOST_READABLE)) {
        fprintf(stderr, "the GPU cannot render to a readable RGBA16 texture\n");
        return CRAW_ERROR(ENOTSUP);
    }

    s->tex = pl_tex_create(e->g->gpu, pl_tex_params(
        .w             = e->w,
        .h             = e->h,
        .format        = fmt,
        .renderable    = true,
        .storable      = !!(fmt->caps & PL_FMT_CAP_STORABLE),
        .blit_dst      = !!(fmt->caps & PL_FMT_CAP_BLITTABLE),
        .host_readable = true,
    ));
    if (!s->tex) {
        fprintf(stderr, "cannot create the %dx%d export texture\n", e->w, e->h);
        return CRAW_ERROR(ENOMEM);
    }

    s->values = (size_t)e->w * e->h * 4;
    s->rgba = malloc(s->values * sizeof(*s->rgba));
    if (!s->rgba)
        return CRAW_ERROR(ENOMEM);

    return 0;
}

/* Renders the frame on screen into `slot` and starts reading it back. */
static int view_enc_render_sw(struct view_encode *e, struct view_state *st,
                              struct view_enc_slot *s)
{
    struct pl_frame target;
    int ret;

    ret = view_enc_slot_tex(e, s);
    if (ret < 0)
        return ret;

    /* Display referred on purpose, exactly as the still export's target
     * is: what the file holds is the look on screen, sRGB encoded, not the
     * scene-linear numbers. */
    target = (struct pl_frame) {
        .num_planes = 1,
        .planes[0] = {
            .texture           = s->tex,
            .components        = 4,
            .component_mapping = { 0, 1, 2, 3 },
        },
        .repr = {
            .sys    = PL_COLOR_SYSTEM_RGB,
            .levels = PL_COLOR_LEVELS_PC,
            .alpha  = PL_ALPHA_NONE,
            .bits   = { .sample_depth = 16, .color_depth = 16 },
        },
        .color = pl_color_space_srgb,
        .crop  = { 0, 0, (float)e->w, (float)e->h },
    };

    ret = view_gpu_render_target(e->g, st, &target);
    if (ret < 0)
        return ret;

    if (!pl_tex_download(e->g->gpu, pl_tex_transfer_params(
            .tex = s->tex,
            .ptr = s->rgba,
        ))) {
        fprintf(stderr, "cannot read the export texture back\n");
        return CRAW_ERROR(EIO);
    }

    s->busy = true;
    s->index = st->index;

    /* The read-back is asynchronous: pl_tex_download() only queues it, and
     * the slot is not looked at until pl_tex_poll() says the texture is
     * free. Flushing here is what puts the work on the queue, so the GPU is
     * busy with this frame while the encoder chews on the last one. */
    pl_gpu_flush(e->g->gpu);

    return 0;
}

static void view_enc_wait_slot(struct view_encode *e, struct view_enc_slot *s)
{
    while (pl_tex_poll(e->g->gpu, s->tex, UINT64_MAX))
        ;
    s->busy = false;
}

/* The packed RGB targets are filled by hand, the way the still writer does
 * it: dropping the alpha and putting the bytes in the encoder's order is
 * all they need, and it keeps the sixteen bit path exact. */
static bool view_enc_pack(AVFrame *f, const uint16_t *rgba)
{
    uint32_t w = (uint32_t)f->width, h = (uint32_t)f->height;
    int hi, lo;

    switch (f->format) {
    case AV_PIX_FMT_RGB48BE:
    case AV_PIX_FMT_RGB48LE:
        hi = f->format == AV_PIX_FMT_RGB48BE ? 0 : 1;
        lo = 1 - hi;

        for (uint32_t y = 0; y < h; y++) {
            const uint16_t *s = rgba + (size_t)y * w * 4;
            uint8_t *d = f->data[0] + (ptrdiff_t)y * f->linesize[0];

            for (uint32_t x = 0; x < w; x++)
                for (int c = 0; c < 3; c++) {
                    unsigned v = s[x * 4 + c];

                    d[(x * 3 + c) * 2 + hi] = (uint8_t)(v >> 8);
                    d[(x * 3 + c) * 2 + lo] = (uint8_t)v;
                }
        }
        return true;

    default:
        return false;
    }
}

/* Everything else through libswscale, from the RGBA64 the slot already is.
 * The frames carry the colour properties, so the matrix and the range come
 * out of the tags rather than out of a guess. */
static int view_enc_scale(struct view_encode *e, AVFrame *f,
                          const uint16_t *rgba)
{
    AVFrame *src = e->ss;
    int ret;

    if (!e->sws) {
        e->sws = sws_alloc_context();
        if (!e->sws)
            return CRAW_ERROR(ENOMEM);
    }

    src->format          = AV_PIX_FMT_RGBA64;
    src->width           = f->width;
    src->height          = f->height;
    src->colorspace      = AVCOL_SPC_RGB;
    src->color_range     = AVCOL_RANGE_JPEG;
    src->color_primaries = AVCOL_PRI_BT709;
    src->color_trc       = AVCOL_TRC_IEC61966_2_1;

    /* The slot is an RGBA64 plane in host order already, so the scaler
     * reads it where it lies. */
    ret = av_image_fill_arrays(src->data, src->linesize, (const uint8_t *)rgba,
                               src->format, src->width, src->height, 1);
    if (ret < 0)
        return ret;

    ret = sws_scale_frame(e->sws, f, src);
    if (ret < 0) {
        fprintf(stderr, "sws_scale_frame: %s\n", av_err2str(ret));
        return ret;
    }

    return 0;
}

/* ------------------------------------------------------------ the packets */

static int view_enc_drain(struct view_encode *e)
{
    int ret;

    while (1) {
        ret = avcodec_receive_packet(e->avc, e->pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return 0;
        if (ret < 0) {
            fprintf(stderr, "avcodec_receive_packet: %s\n", av_err2str(ret));
            return CRAW_ERROR(EIO);
        }

        e->pkt->stream_index = e->vs->index;
        av_packet_rescale_ts(e->pkt, e->avc->time_base, e->vs->time_base);

        ret = av_interleaved_write_frame(e->oc, e->pkt);
        av_packet_unref(e->pkt);
        if (ret < 0) {
            fprintf(stderr, "cannot write the packet: %s\n", av_err2str(ret));
            return CRAW_ERROR(EIO);
        }
    }
}

/* One frame into the encoder, the Vulkan way: libplacebo renders into the
 * encoder's own image and hands it back on the frame's timeline semaphore,
 * which is the value the encoder then waits for. */
static int view_enc_send_hw(struct view_encode *e, struct view_state *st,
                            int64_t pts)
{
    struct pl_frame target;
    AVFrame *f = av_frame_alloc();
    int ret;

    if (!f)
        return CRAW_ERROR(ENOMEM);

    ret = av_hwframe_get_buffer(e->frames, f, 0);
    if (ret < 0) {
        fprintf(stderr, "av_hwframe_get_buffer: %s\n", av_err2str(ret));
        goto out;
    }

    f->pts      = pts;
    f->duration = 1;
    view_enc_tag_frame(e, f);

    /* pl_map_avframe_ex() wraps the frame's images and installs the
     * acquire/release pair that does the AVVkFrame dance: the renderer
     * gives the image to libplacebo at the frame's current semaphore value
     * and takes it back signalling that value plus one, writing the layout
     * and the new value back into the AVVkFrame. That is the contract in
     * libavutil/hwcontext_vulkan.h, read from the producer's side. The
     * colour tags set above are what the mapping turns into the target's
     * pl_color_repr, so libplacebo does the RGB to YCbCr conversion, the
     * range scaling and the chroma subsampling itself. */
    if (!pl_map_avframe_ex(e->g->gpu, &target, pl_avframe_params(.frame = f))) {
        fprintf(stderr, "cannot map the encoder's frame for rendering\n");
        ret = CRAW_ERROR(EIO);
        goto out;
    }

    target.crop = (pl_rect2df) { 0, 0, (float)e->w, (float)e->h };

    ret = view_gpu_render_target(e->g, st, &target);
    pl_unmap_avframe(e->g->gpu, &target);
    if (ret < 0)
        goto out;

    /* The encoder's submission waits on a semaphore value libplacebo has
     * promised but not necessarily put on the queue yet, so the work has to
     * be submitted before the frame is handed over. */
    pl_gpu_flush(e->g->gpu);

    ret = avcodec_send_frame(e->avc, f);
    if (ret < 0)
        fprintf(stderr, "avcodec_send_frame: %s\n", av_err2str(ret));

out:
    av_frame_free(&f);

    return ret < 0 ? ret : 0;
}

/* One frame into the encoder out of a read-back slot. */
static int view_enc_send_sw(struct view_encode *e, struct view_enc_slot *s,
                            int64_t pts)
{
    AVFrame *f = e->sf;
    int ret;

    /* The encoder may still hold the last one, in which case a new buffer
     * is taken rather than the old contents copied into one. */
    if (!av_frame_is_writable(f)) {
        av_frame_unref(f);
        f->format = e->sw_fmt;
        f->width  = e->w;
        f->height = e->h;
        ret = av_frame_get_buffer(f, 0);
        if (ret < 0)
            return CRAW_ERROR(ENOMEM);
    }

    /* Before the conversion, not after: libswscale reads the destination's
     * matrix, range and chroma siting off the frame, and a frame that still
     * said "unspecified" was converted through BT.601 into limited range
     * and then labelled BT.709, which cost five dB against the still. */
    view_enc_tag_frame(e, f);

    if (!view_enc_pack(f, s->rgba)) {
        ret = view_enc_scale(e, f, s->rgba);
        if (ret < 0)
            return ret;
        /* sws_scale_frame() copies the source's properties over the
         * destination's, so they go back on afterwards. */
        view_enc_tag_frame(e, f);
    }

    f->pts      = pts;
    f->duration = 1;

    ret = avcodec_send_frame(e->avc, f);
    if (ret < 0) {
        fprintf(stderr, "avcodec_send_frame: %s\n", av_err2str(ret));
        return CRAW_ERROR(EIO);
    }

    return 0;
}

/* ---------------------------------------------------------------- the run */

void view_encode_default_path(const struct view_state *st, int64_t start,
                              int64_t end, const char *codec,
                              char *out, size_t size)
{
    const char *file = strrchr(st->cfg.input, '/');
    const char *dot, *ext = "mov";
    char base[128];
    size_t n;

    file = file ? file + 1 : st->cfg.input;
    dot = strrchr(file, '.');
    n = dot ? (size_t)(dot - file) : strlen(file);
    n = CRAW_MIN(n, sizeof(base) - 1);
    memcpy(base, file, n);
    base[n] = '\0';

    /* QuickTime for the codecs a camera writes, Matroska for the archival
     * ones it does not: a .mov with FFV1 in it is not a file anything else
     * will open. */
    if (codec && (strstr(codec, "ffv1") || strstr(codec, "x265") ||
                  strstr(codec, "hevc") || strstr(codec, "vp9") ||
                  strstr(codec, "huff")))
        ext = "mkv";
    else if (codec && (strstr(codec, "av1") || strstr(codec, "apv")))
        ext = "mp4";

    snprintf(out, size, "%s/%s-%05d-%05d.%s", view_clip_dir(st), base,
             (int)start, (int)end, ext);
}

int view_encode_begin(struct view_state *st, const struct view_encode_opts *o)
{
    struct view_video *v = &st->video;
    struct view_encode *e;

    view_encode_close(st);

    e = calloc(1, sizeof(*e));
    if (!e)
        return CRAW_ERROR(ENOMEM);

    v->e = e;
    e->g = &st->gpu;
    e->verbose = st->cfg.verbose;
    e->report = !st->win;
    e->start = CRAW_MAX(o->start, 0);
    e->end   = CRAW_MAX(o->end, e->start);
    e->next  = e->start;
    e->req_w = o->width;
    e->req_h = o->height;
    e->req_audio = o->audio;
    e->t0 = e->last_report = av_gettime_relative();

    v->start = e->start;
    v->end   = e->end;
    v->audio = o->audio;
    v->active = true;
    v->stop = false;
    v->progress = 0.0;
    snprintf(v->codec, sizeof(v->codec), "%s", o->codec ? o->codec : "");
    snprintf(v->enc, sizeof(v->enc), "%s", o->enc ? o->enc : "");

    if (o->path && o->path[0])
        snprintf(v->path, sizeof(v->path), "%s", o->path);
    else
        view_encode_default_path(st, e->start, e->end,
                                 v->codec[0] ? v->codec : NULL,
                                 v->path, sizeof(v->path));

    /* A clean peak detection track, so that a range exported twice gives
     * the same file twice. The rest waits for the first frame: the clip's
     * size, and with it the crop rectangle's, is not known until one has
     * been through the pipeline. */
    view_gpu_reset_peak(&st->gpu, true);
    st->playing = false;

    snprintf(v->status, sizeof(v->status), "starting at frame %" PRId64,
             e->start);

    return 0;
}

/* Everything that needs the first frame: the size, and with it the muxer,
 * the encoder and the audio streams. */
static int view_enc_setup(struct view_encode *e, struct view_state *st)
{
    struct view_video *v = &st->video;
    const enum AVPixelFormat *want = NULL;
    enum AVCodecID id = AV_CODEC_ID_NONE;
    size_t nwant = 0;
    int x, y, cw, ch;
    int ret;

    /* The size: what was asked for, or the crop rectangle's own. Always
     * even, which is what a subsampled encoder wants and what the crop
     * already is. */
    view_crop_rect(st, &x, &y, &cw, &ch);
    e->w = (e->req_w > 0 ? e->req_w : cw) & ~1;
    e->h = (e->req_h > 0 ? e->req_h : ch) & ~1;
    if (e->w <= 0 || e->h <= 0) {
        fprintf(stderr, "the output size is empty\n");
        ret = CRAW_ERROR(EINVAL);
        goto fail;
    }
    v->out_w = e->w;
    v->out_h = e->h;

    e->fr = st->dec.frame_rate;
    if (e->fr.num <= 0 || e->fr.den <= 0)
        e->fr = (AVRational) { 25, 1 };

    ret = avformat_alloc_output_context2(&e->oc, NULL, NULL, v->path);
    if (ret < 0 || !e->oc) {
        fprintf(stderr, "%s: no muxer for this file name: %s\n", v->path,
                av_err2str(ret));
        ret = CRAW_ERROR(EINVAL);
        goto fail;
    }

    /* The name picks the codec as well as the muxer, the way it does for a
     * still: only av_guess_codec() looks at the extension. */
    if (v->codec[0]) {
        e->codec = avcodec_find_encoder_by_name(v->codec);
        if (e->codec)
            id = e->codec->id;
    } else {
        id = av_guess_codec(e->oc->oformat, NULL, v->path, NULL, AVMEDIA_TYPE_VIDEO);
        e->codec = avcodec_find_encoder(id);
    }
    if (!e->codec || e->codec->type != AVMEDIA_TYPE_VIDEO) {
        fprintf(stderr, "%s: this FFmpeg has no video encoder called %s\n",
                v->path, v->codec[0] ? v->codec : avcodec_get_name(id));
        ret = CRAW_ERROR(ENOSYS);
        goto fail;
    }

    e->hw = view_enc_is_vulkan(e->codec);

    e->pkt  = av_packet_alloc();
    e->apkt = av_packet_alloc();
    e->vs   = avformat_new_stream(e->oc, NULL);
    if (!e->pkt || !e->apkt || !e->vs) {
        ret = CRAW_ERROR(ENOMEM);
        goto fail;
    }

    /* The audio streams are added after the video one, so the picture stays
     * stream 0, and before the header, which is the only time a stream can
     * be added at all. */
    if (e->req_audio) {
        ret = view_enc_audio_open(e, st);
        if (ret < 0)
            goto fail;
    }

    if (e->hw) {
        /* A Vulkan encoder reports nothing but AV_PIX_FMT_VULKAN, so what
         * it really takes is found by handing it a frames context of each
         * candidate in turn and seeing which one opens. */
        view_enc_wanted(e->codec->id, true, &want, &nwant);
        ret = CRAW_ERROR(ENOTSUP);
        for (size_t i = 0; i < nwant; i++) {
            ret = view_enc_try(e, st, want[i], v->enc);
            if (ret >= 0)
                break;
        }
        if (ret < 0) {
            fprintf(stderr, "%s: the %s encoder took none of the formats this "
                            "device can render into; the last said: %s. "
                            "Run with -v to see every attempt.\n",
                    v->path, e->codec->name, av_err2str(ret));
            ret = CRAW_ERROR(ENOTSUP);
            goto fail;
        }
    } else {
        enum AVPixelFormat fmt = view_enc_pick_sw(e->codec);

        if (fmt == AV_PIX_FMT_NONE) {
            fprintf(stderr, "%s: %s takes no pixel format we can write\n",
                    v->path, e->codec->name);
            ret = CRAW_ERROR(ENOTSUP);
            goto fail;
        }

        ret = view_enc_try(e, st, fmt, v->enc);
        if (ret < 0) {
            fprintf(stderr, "%s: cannot open the %s encoder with %s: %s\n",
                    v->path, e->codec->name, av_get_pix_fmt_name(fmt),
                    av_err2str(ret));
            ret = CRAW_ERROR(EINVAL);
            goto fail;
        }
    }

    ret = avcodec_parameters_from_context(e->vs->codecpar, e->avc);
    if (ret < 0)
        goto fail;
    e->vs->time_base      = e->avc->time_base;
    e->vs->avg_frame_rate = e->fr;

    if (!(e->oc->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&e->oc->pb, v->path, AVIO_FLAG_WRITE);
        if (ret < 0) {
            fprintf(stderr, "%s: cannot open for writing: %s\n", v->path,
                    av_err2str(ret));
            ret = CRAW_ERROR(EIO);
            goto fail;
        }
    }

    ret = avformat_write_header(e->oc, NULL);
    if (ret < 0) {
        fprintf(stderr, "%s: avformat_write_header: %s\n", v->path,
                av_err2str(ret));
        ret = CRAW_ERROR(EIO);
        goto fail;
    }
    e->header = true;

    if (!e->hw) {
        e->sf = av_frame_alloc();
        e->ss = av_frame_alloc();
        if (!e->sf || !e->ss) {
            ret = CRAW_ERROR(ENOMEM);
            goto fail;
        }
        e->sf->format = e->sw_fmt;
        e->sf->width  = e->w;
        e->sf->height = e->h;
        ret = av_frame_get_buffer(e->sf, 0);
        if (ret < 0) {
            ret = CRAW_ERROR(ENOMEM);
            goto fail;
        }
    }

    e->opened = true;

    snprintf(v->status, sizeof(v->status), "%s %dx%d %s, %" PRId64 " frames",
             e->codec->name, e->w, e->h, av_get_pix_fmt_name(e->sw_fmt),
             e->end - e->start + 1);
    view_log(st, "encoding frames %" PRId64 " to %" PRId64 " into %s: %s %s "
                 "%dx%d %s, %d/%d fps%s%s\n",
             e->start, e->end, v->path, e->oc->oformat->name, e->codec->name,
             e->w, e->h, av_get_pix_fmt_name(e->sw_fmt), e->fr.num, e->fr.den,
             e->hw ? ", Vulkan frames" : "",
             e->nb_audio ? ", audio copied" : "");

    return 0;

fail:
    view_encode_close(st);
    v->active = false;
    snprintf(v->status, sizeof(v->status), "failed: %s", av_err2str(ret));

    return ret;
}

/* Walks the decoder to `want`. 2: the frame is on screen, 1: the clip ended,
 * 0: not there yet, negative: an error. */
static int view_enc_walk(struct view_encode *e, struct view_state *st,
                         int64_t want)
{
    int ret;

    if (st->have_frame && st->index == want)
        return 2;

    /* One seek, at the start: every frame after it is the next one. */
    if (!e->sought) {
        st->seek_to = want;
        e->sought = true;
    }

    do {
        ret = view_step_frame(st);
        if (ret == 3)
            return 0;           /* the reader is behind; try again later */
    } while (ret == 2);

    if (ret < 0)
        return ret;
    if (ret > 0)
        return 1;

    return st->index == want ? 2 : 0;
}

static void view_enc_progress(struct view_encode *e, struct view_state *st)
{
    struct view_video *v = &st->video;
    int64_t total = e->end - e->start + 1;
    int64_t now = av_gettime_relative();
    double el = (double)(now - e->t0) / 1.0e6;
    double left;

    e->fps = el > 0.0 ? (double)e->encoded / el : 0.0;
    v->progress = total > 0 ? (double)e->encoded / (double)total : 0.0;
    left = e->fps > 0.0 ? (double)(total - e->encoded) / e->fps : 0.0;

    snprintf(v->status, sizeof(v->status),
             "%" PRId64 "/%" PRId64 " frames, %.1f fps, %.0f s left",
             e->encoded, total, e->fps, left);

    if (e->report && now - e->last_report >= 1000000) {
        e->last_report = now;
        fprintf(stderr, "%s\n", v->status);
    }
}

/* Flush the encoder, finish the audio, write the trailer. */
static int view_enc_finish(struct view_encode *e, struct view_state *st)
{
    struct view_video *v = &st->video;
    int ret;

    if (!e->draining) {
        e->draining = true;
        avcodec_send_frame(e->avc, NULL);
    }

    ret = view_enc_drain(e);
    if (ret < 0)
        return ret;

    /* Whatever audio the range still owes, now that the video is done. */
    ret = view_enc_audio_pump(e, 1.0e9);
    if (ret < 0)
        return ret;

    ret = av_write_trailer(e->oc);
    e->header = false;
    if (ret < 0) {
        fprintf(stderr, "%s: av_write_trailer: %s\n", v->path, av_err2str(ret));
        return CRAW_ERROR(EIO);
    }

    e->done = true;

    return 0;
}

int view_encode_step(struct view_state *st)
{
    struct view_video *v = &st->video;
    struct view_encode *e = v->e;
    struct view_enc_slot *s, *prev;
    double el, mb = 0.0;
    int64_t total;
    int ret, got;

    if (!e || !v->active)
        return 0;

    if (v->stop || e->done)
        goto finish;

    if (!e->opened) {
        /* Nothing can be opened before a frame has been through the
         * pipeline, because nothing knows how big it is until then. */
        got = view_enc_walk(e, st, e->next);
        if (got < 0) {
            ret = got;
            goto fail;
        }
        if (got == 0)
            return 0;
        if (got == 1) {
            fprintf(stderr, "the clip has no frame %" PRId64 "\n", e->next);
            ret = CRAW_ERROR(ERANGE);
            goto fail;
        }

        ret = view_enc_setup(e, st);
        if (ret < 0)
            return ret;         /* view_enc_setup() has already cleaned up */
    }

    if (e->next > e->end) {
        /* The software path runs a frame behind the renderer, so the last
         * slot is still waiting to be encoded when the range runs out. */
        prev = &e->slot[e->cur ^ 1];
        if (e->hw || !prev->busy)
            goto finish;

        view_enc_wait_slot(e, prev);
        ret = view_enc_send_sw(e, prev, prev->index - e->start);
        if (ret < 0)
            goto fail;
        e->encoded++;
        goto wrote;
    }

    got = view_enc_walk(e, st, e->next);
    if (got < 0) {
        ret = got;
        goto fail;
    }
    if (got == 0)
        return 0;
    if (got == 1) {
        fprintf(stderr, "the clip ends before frame %" PRId64 "\n", e->next);
        e->end = e->next - 1;
        goto finish;
    }

    if (e->hw) {
        ret = view_enc_send_hw(e, st, e->next - e->start);
        if (ret < 0)
            goto fail;
        e->encoded++;
    } else {
        s = &e->slot[e->cur];
        prev = &e->slot[e->cur ^ 1];

        ret = view_enc_render_sw(e, st, s);
        if (ret < 0)
            goto fail;

        if (prev->busy) {
            view_enc_wait_slot(e, prev);
            ret = view_enc_send_sw(e, prev, prev->index - e->start);
            if (ret < 0)
                goto fail;
            e->encoded++;
        }
        e->cur ^= 1;
    }

    e->next++;

wrote:
    ret = view_enc_drain(e);
    if (ret < 0)
        goto fail;

    /* Audio as far as the picture has got, so the muxer never has to hold
     * more than a frame of interleaving. */
    ret = view_enc_audio_pump(e, (double)e->encoded * e->fr.den / e->fr.num);
    if (ret < 0)
        goto fail;

    view_enc_progress(e, st);

    return 0;

finish:
    /* A run stopped before the first frame has nothing open to close. */
    if (!e->opened) {
        v->active = false;
        view_encode_close(st);
        return 1;
    }

    ret = view_enc_finish(e, st);
    if (ret < 0)
        goto fail;

    total = e->encoded;
    el = (double)(av_gettime_relative() - e->t0) / 1.0e6;
    if (e->oc->pb)
        mb = (double)avio_size(e->oc->pb) / 1e6;

    snprintf(v->status, sizeof(v->status),
             "%" PRId64 " frames, %.1f fps, %.1f MB", total,
             el > 0.0 ? (double)total / el : 0.0, mb);
    view_log(st, "%s: %s in %.1f s\n", v->path, v->status, el);

    v->progress = 1.0;
    v->active = false;
    view_encode_close(st);

    return 1;

fail:
    snprintf(v->status, sizeof(v->status), "failed: %s", av_err2str(ret));
    fprintf(stderr, "%s: %s\n", v->path, v->status);
    v->active = false;
    view_encode_close(st);

    return ret;
}

void view_encode_cancel(struct view_state *st)
{
    if (st->video.e)
        st->video.stop = true;
}

void view_encode_close(struct view_state *st)
{
    struct view_encode *e = st->video.e;

    if (!e)
        return;

    /* The renderer may still be writing into a slot texture, and the
     * encoder may still hold frames of the pool. */
    if (e->g && e->g->gpu)
        pl_gpu_finish(e->g->gpu);

    if (e->header && e->oc)
        av_write_trailer(e->oc);

    for (int i = 0; i < VIEW_ENC_SLOTS; i++) {
        if (e->g && e->g->gpu)
            pl_tex_destroy(e->g->gpu, &e->slot[i].tex);
        free(e->slot[i].rgba);
    }

    sws_freeContext(e->sws);
    av_frame_free(&e->sf);
    av_frame_free(&e->ss);
    av_packet_free(&e->pkt);
    av_packet_free(&e->apkt);
    avcodec_free_context(&e->avc);
    av_buffer_unref(&e->frames);
    avformat_close_input(&e->aic);

    if (e->oc) {
        if (e->oc->pb && !(e->oc->oformat->flags & AVFMT_NOFILE))
            avio_closep(&e->oc->pb);
        avformat_free_context(e->oc);
    }

    free(e);
    st->video.e = NULL;
}
