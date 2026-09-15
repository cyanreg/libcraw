/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* FFmpeg's Vulkan hwaccel on libplacebo's device.
 *
 * The AVHWDeviceContext is filled from the pl_vulkan struct rather than
 * created by libavutil, so the decoder writes its output into images that
 * live on the same VkDevice as the raw pipeline and the renderer. The queue
 * locks forward to libplacebo's, which is what keeps three independent
 * libraries submitting to one queue safely. */

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavutil/dict.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
#include <libavutil/pixdesc.h>
#include <libavutil/raw_color_params.h>
#include <libavutil/time.h>

#include "view.h"
#include "../profiles.h"

static enum AVPixelFormat view_get_format(AVCodecContext *avc,
                                          const enum AVPixelFormat *fmts);

/* Builds the codec context. Also used to rebuild it on a seek: a decoder
 * that has reported end of stream does not reliably come back from
 * avcodec_flush_buffers() with this hwaccel, and a loop point is exactly
 * that case. Re-creating it is a few milliseconds once per seek. */
static int view_decoder_codec_open(struct view_decoder *d)
{
    AVStream *st = d->fmt->streams[d->stream_index];
    const AVCodec *codec = avcodec_find_decoder(st->codecpar->codec_id);
    int ret;

    if (!codec)
        return CRAW_ERROR(ENOSYS);

    avcodec_free_context(&d->dec);

    d->dec = avcodec_alloc_context3(codec);
    if (!d->dec)
        return CRAW_ERROR(ENOMEM);

    ret = avcodec_parameters_to_context(d->dec, st->codecpar);
    if (ret < 0)
        return CRAW_ERROR(EINVAL);

    d->dec->hw_device_ctx = av_buffer_ref(d->hw_device);
    if (!d->dec->hw_device_ctx)
        return CRAW_ERROR(ENOMEM);
    d->dec->opaque = d;
    d->dec->get_format = view_get_format;
    /* On the raw path the hwaccel does the work on the GPU and extra CPU
     * threads would only race the frame properties read under the frames
     * context lock. A non-raw clip may end up decoding in software, where
     * threads are the difference between real time and not. */
    d->dec->thread_count = d->raw ? 1 : 0;

    ret = avcodec_open2(d->dec, codec, NULL);
    if (ret < 0) {
        fprintf(stderr, "cannot open the %s decoder: %s\n", codec->name,
                av_err2str(ret));
        return CRAW_ERROR(EINVAL);
    }

    return 0;
}

static int  view_reader_start(struct view_decoder *d);
static void view_reader_stop(struct view_decoder *d);

/* ---------------------------------------------------- the device context */

static void view_lock_queue(AVHWDeviceContext *ctx, uint32_t qf, uint32_t idx)
{
    pl_vulkan vk = ctx->user_opaque;
    vk->lock_queue(vk, qf, idx);
}

static void view_unlock_queue(AVHWDeviceContext *ctx, uint32_t qf, uint32_t idx)
{
    pl_vulkan vk = ctx->user_opaque;
    vk->unlock_queue(vk, qf, idx);
}

static int view_hw_device_create(struct view_decoder *d, struct view_gpu *g)
{
    AVHWDeviceContext *hwctx;
    AVVulkanDeviceContext *vkctx;
    VkQueueFamilyProperties *qprops = NULL;
    uint32_t nb_q = 0;
    int ret;

    d->hw_device = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_VULKAN);
    if (!d->hw_device)
        return CRAW_ERROR(ENOMEM);

    hwctx = (AVHWDeviceContext *)d->hw_device->data;
    hwctx->user_opaque = (void *)g->vk;
    vkctx = hwctx->hwctx;

    vkctx->get_proc_addr = g->vk->get_proc_addr;
    vkctx->inst          = g->vk->instance;
    vkctx->phys_dev      = g->vk->phys_device;
    vkctx->act_dev       = g->vk->device;
    vkctx->device_features = *g->vk->features;

    vkctx->enabled_inst_extensions    = g->inst->extensions;
    vkctx->nb_enabled_inst_extensions = g->inst->num_extensions;
    vkctx->enabled_dev_extensions     = g->vk->extensions;
    vkctx->nb_enabled_dev_extensions  = g->vk->num_extensions;

    /* Deprecated in favour of VK_KHR_internally_synchronized_queues, but the
     * installed libavutil still uses them, and without them FFmpeg would
     * take its own mutex while libplacebo takes a different one. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    vkctx->lock_queue   = view_lock_queue;
    vkctx->unlock_queue = view_unlock_queue;
#pragma GCC diagnostic pop

    /* The family this tool works on comes first, and every other family
     * libplacebo enabled queues for follows. More than one entry is what
     * makes FFmpeg allocate concurrent images, which in turn makes every
     * barrier on them queue-family agnostic; see VIEW_EXTRA_QUEUES. */
    vkGetPhysicalDeviceQueueFamilyProperties(g->vk->phys_device, &nb_q, NULL);
    if (g->qf >= nb_q) {
        ret = CRAW_ERROR(EINVAL);
        goto fail;
    }
    qprops = calloc(nb_q, sizeof(*qprops));
    if (!qprops) {
        ret = CRAW_ERROR(ENOMEM);
        goto fail;
    }
    vkGetPhysicalDeviceQueueFamilyProperties(g->vk->phys_device, &nb_q, qprops);

    vkctx->qf[0] = (AVVulkanDeviceQueueFamily) {
        .idx   = (int)g->qf,
        .num   = 1,
        .flags = (VkQueueFlagBits)qprops[g->qf].queueFlags,
    };
    vkctx->nb_qf = 1;

    for (uint32_t i = 0; i < nb_q && vkctx->nb_qf < 8; i++) {
        if (i == g->qf || !(qprops[i].queueFlags & VIEW_EXTRA_QUEUES))
            continue;
        vkctx->qf[vkctx->nb_qf++] = (AVVulkanDeviceQueueFamily) {
            .idx   = (int)i,
            .num   = 1,
            .flags = (VkQueueFlagBits)qprops[i].queueFlags,
        };
    }

    vkctx->queue_flags = g->queue_flags;

    ret = av_hwdevice_ctx_init(d->hw_device);
    if (ret < 0) {
        fprintf(stderr, "av_hwdevice_ctx_init: %s\n", av_err2str(ret));
        ret = CRAW_ERROR(EINVAL);
        goto fail;
    }

    ret = 0;

fail:
    free(qprops);
    return ret;
}

/* --------------------------------------------------------- the decoder */

/* Vulkan frames if the decoder offers them, and whatever it would have
 * produced otherwise. libavcodec calls this again without the hardware
 * entry when the hwaccel fails to initialize, which is exactly the fallback
 * a non-raw clip wants: a software HEVC decode with threads still plays,
 * and the pacing drops frames if it cannot keep up. */
static enum AVPixelFormat view_get_format(AVCodecContext *avc,
                                          const enum AVPixelFormat *fmts)
{
    const struct view_decoder *d = avc->opaque;

    for (const enum AVPixelFormat *f = fmts; *f != AV_PIX_FMT_NONE; f++)
        if (*f == AV_PIX_FMT_VULKAN)
            return *f;

    if (d && d->raw) {
        fprintf(stderr, "the decoder does not offer Vulkan frames\n");
        return AV_PIX_FMT_NONE;
    }

    /* Skip the other hardware formats: this tool has one device context and
     * it is a Vulkan one, so offering VAAPI here only produces a complaint
     * and another round trip. */
    for (const enum AVPixelFormat *f = fmts; *f != AV_PIX_FMT_NONE; f++) {
        const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(*f);

        if (desc && !(desc->flags & AV_PIX_FMT_FLAG_HWACCEL))
            return *f;
    }

    return fmts[0];
}

/* The text of one XML element, with the surrounding whitespace dropped.
 *
 * A parser would be the honest answer to a document, but this one is a
 * fixed two-kilobyte record a camera writes into a metadata dictionary
 * entry, and the three values wanted from it are leaf elements with no
 * attributes and no namespace prefix. Matching `<Tag>` and the `</Tag>`
 * after it is the whole grammar that needs supporting, and it costs no
 * dependency. Anything unexpected - a missing end tag, a value longer than
 * the buffer - reads as absent rather than as something made up. */
static bool view_xml_text(const char *xml, const char *tag, char *out, size_t size)
{
    char open[32], close[32];
    const char *b, *e;
    size_t n;

    if (!xml)
        return false;

    snprintf(open, sizeof(open), "<%s>", tag);
    snprintf(close, sizeof(close), "</%s>", tag);

    b = strstr(xml, open);
    if (!b)
        return false;
    b += strlen(open);

    e = strstr(b, close);
    if (!e)
        return false;

    while (b < e && (unsigned char)*b <= ' ')
        b++;
    while (e > b && (unsigned char)e[-1] <= ' ')
        e--;

    n = (size_t)(e - b);
    if (!n || n >= size)
        return false;

    memcpy(out, b, n);
    out[n] = '\0';

    return true;
}

/* The camera model and the as-shot ISO hide in whichever metadata
 * dictionary the muxer felt like using, so look at both. */
static void view_read_metadata(struct view_decoder *d)
{
    AVDictionary *const dicts[2] = {
        d->fmt->metadata, d->fmt->streams[d->stream_index]->metadata,
    };

    for (int i = 0; i < 2; i++) {
        const AVDictionaryEntry *e;

        e = av_dict_get(dicts[i], "org.smpte.rdd18.camera.isosensitivity", NULL, 0);
        if (e && !(d->meta_as_shot_ei > 0.0f))
            d->meta_as_shot_ei = (float)atof(e->value);

        e = av_dict_get(dicts[i], "com.apple.proapps.modelname", NULL, 0);
        if (e && !d->model[0])
            snprintf(d->model, sizeof(d->model), "%s", e->value);
    }

    /* The Panasonic clip XML, `com.panasonic.Semi-Pro.metadata.xml`. It is
     * the only record of what the camera was actually recording: an HEVC
     * V-Log clip carries a BT.709 VUI - primaries, transfer and matrix all
     * 709 - which is true of the matrix and false of the other two. Every
     * entry in both dictionaries is scanned rather than the one key, so a
     * muxer that files the blob somewhere else still works. */
    for (int i = 0; i < 2; i++) {
        const AVDictionaryEntry *e = NULL;

        while ((e = av_dict_iterate(dicts[i], e))) {
            char iso[32];

            if (!d->capture_gamma[0])
                view_xml_text(e->value, "CaptureGamma", d->capture_gamma,
                              sizeof(d->capture_gamma));
            if (!d->capture_gamut[0])
                view_xml_text(e->value, "CaptureGamut", d->capture_gamut,
                              sizeof(d->capture_gamut));

            /* The as-shot exposure index, for a clip that has no RDD 18
             * keys at all - which is every clip this camera writes. What
             * the exposure section re-rates against. */
            if (!(d->meta_as_shot_ei > 0.0f) &&
                view_xml_text(e->value, "ISOSensitivity", iso, sizeof(iso)))
                d->meta_as_shot_ei = (float)atof(iso);
        }
    }

    /* The one clip the tags have to be overridden for. The codec comes
     * first because the same two elements appear in a ProRes RAW clip's
     * XML, where they describe what the sensor data would become and not
     * how it is stored: a raw clip is developed by libcraw and never goes
     * near this. */
    d->vlog = !d->raw && !strcmp(d->capture_gamma, "V-Log") &&
              !strcmp(d->capture_gamut, "V-Gamut");

    /* Some muxers only carry the model inside a larger blob - a whole XML
     * document, in this camera's case - so what is kept is the table set it
     * identifies, never the value itself. */
    if (!d->model[0]) {
        for (int i = 0; i < 2; i++) {
            const AVDictionaryEntry *e = NULL;
            while ((e = av_dict_iterate(dicts[i], e))) {
                const char *name = craw_tool_profile_detect(e->value);

                if (name) {
                    snprintf(d->model, sizeof(d->model), "%s", name);
                    return;
                }
            }
        }
    }
}

int view_decoder_open(struct view_decoder *d, const char *path,
                      struct view_gpu *g, bool verbose)
{
    AVStream *st;
    const AVCodec *codec;
    int ret;

    ret = avformat_open_input(&d->fmt, path, NULL, NULL);
    if (ret < 0) {
        fprintf(stderr, "%s: %s\n", path, av_err2str(ret));
        return CRAW_ERROR(EIO);
    }

    ret = avformat_find_stream_info(d->fmt, NULL);
    if (ret < 0) {
        fprintf(stderr, "%s: cannot read the stream info: %s\n", path, av_err2str(ret));
        return CRAW_ERROR(EIO);
    }

    d->stream_index = av_find_best_stream(d->fmt, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (d->stream_index < 0 || !codec) {
        fprintf(stderr, "%s: no decodable video stream\n", path);
        return CRAW_ERROR(EINVAL);
    }

    /* The audio has its own reader, so this one only wants the pictures. */
    for (unsigned i = 0; i < d->fmt->nb_streams; i++)
        if ((int)i != d->stream_index)
            d->fmt->streams[i]->discard = AVDISCARD_ALL;

    st = d->fmt->streams[d->stream_index];

    /* A Bayer source is one the pipeline has to debayer, and the only thing
     * that says so reliably is the codec: the Panasonic XML in a ProRes RAW
     * clip advertises the same capture gamma and gamut a log recording
     * would, so it cannot be used to tell them apart. */
    d->raw = st->codecpar->codec_id == AV_CODEC_ID_PRORES_RAW;
    d->codec_name = codec->name;
    d->verbose = verbose;

    d->time_base  = st->time_base;
    d->frame_rate = st->avg_frame_rate.num ? st->avg_frame_rate : st->r_frame_rate;
    d->nb_frames  = st->nb_frames;

    if (!d->nb_frames && st->duration > 0 && d->frame_rate.num)
        d->nb_frames = av_rescale_q(st->duration, d->time_base,
                                    av_inv_q(d->frame_rate));

    view_read_metadata(d);

    ret = view_hw_device_create(d, g);
    if (ret < 0)
        return ret;

    ret = view_decoder_codec_open(d);
    if (ret < 0)
        return ret;

    d->pkt = av_packet_alloc();
    if (!d->pkt)
        return CRAW_ERROR(ENOMEM);

    ret = view_reader_start(d);
    if (ret < 0)
        return ret;

    if (verbose)
        fprintf(stderr, "input %s: stream %d, %s (%s), %dx%d, %.3f fps, "
                        "%" PRId64 " frames, model `%s`, as-shot ISO %.0f\n",
                path, d->stream_index, codec->name, d->raw ? "raw" : "video",
                d->dec->width, d->dec->height,
                d->frame_rate.den ? av_q2d(d->frame_rate) : 0.0, d->nb_frames,
                d->model[0] ? d->model : "(unknown)", d->meta_as_shot_ei);

    if (verbose && (d->capture_gamma[0] || d->capture_gamut[0]))
        fprintf(stderr, "clip metadata: capture gamma `%s`, gamut `%s`%s\n",
                d->capture_gamma[0] ? d->capture_gamma : "(none)",
                d->capture_gamut[0] ? d->capture_gamut : "(none)",
                d->vlog ? ", overriding the stream's colour tags"
                        : d->raw ? ", raw: the pipeline develops it" : "");

    return 0;
}

void view_decoder_close(struct view_decoder *d)
{
    view_reader_stop(d);
    av_packet_free(&d->pkt);
    avcodec_free_context(&d->dec);
    av_buffer_unref(&d->hw_device);
    avformat_close_input(&d->fmt);
}

/* The audio has its own reader (see audio.h), so everything that is not a
 * video packet is simply skipped here. */
/* ------------------------------------------------------- the reader */

/* Demuxing runs on its own thread so the render loop never blocks on the
 * medium. At two gigabits a second a single frame's packet is fifteen
 * milliseconds of read on a fast NVMe and more on a card reader, which is
 * half a frame period spent doing nothing else. */

/* Caller holds the lock. */
static void view_reader_drop(struct view_reader *r)
{
    while (r->count) {
        AVPacket *p = r->q[r->head];

        r->bytes -= (size_t)p->size;
        av_packet_unref(p);
        r->head = (r->head + 1) % VIEW_PKTQ_MAX;
        r->count--;
    }
    r->bytes = 0;
}

static int view_reader_thread(void *opaque)
{
    struct view_decoder *d = opaque;
    struct view_reader *r = &d->reader;
    AVPacket *pkt = av_packet_alloc();

    if (!pkt)
        return -1;

    SDL_LockMutex(r->lock);

    while (!r->quit) {
        int64_t ts;
        int ret;

        if (r->seek_pending) {
            ts = r->seek_ts;
            r->seek_pending = false;
            view_reader_drop(r);
            r->eof = false;

            SDL_UnlockMutex(r->lock);
            ret = av_seek_frame(d->fmt, d->stream_index, ts, AVSEEK_FLAG_BACKWARD);
            SDL_LockMutex(r->lock);

            if (ret < 0)
                fprintf(stderr, "cannot seek to %" PRId64 ": %s\n", ts, av_err2str(ret));


            /* Whoever asked for the seek is waiting to flush the decoder. */
            SDL_BroadcastCondition(r->ready);
            continue;
        }

        if (r->eof || r->count == VIEW_PKTQ_MAX || r->bytes >= VIEW_PKTQ_BYTES) {
            SDL_WaitCondition(r->wake, r->lock);
            continue;
        }

        SDL_UnlockMutex(r->lock);
        ret = av_read_frame(d->fmt, pkt);
        SDL_LockMutex(r->lock);

        if (ret == AVERROR_EOF) {
            r->eof = true;
            SDL_BroadcastCondition(r->ready);
            continue;
        }
        if (ret < 0) {
            fprintf(stderr, "reading failed: %s\n", av_err2str(ret));
            r->eof = true;
            SDL_BroadcastCondition(r->ready);
            continue;
        }

        /* A seek raced this read: the packet belongs to the old position. */
        if (r->seek_pending || pkt->stream_index != d->stream_index) {
            av_packet_unref(pkt);
            continue;
        }

        av_packet_move_ref(r->q[(r->head + r->count) % VIEW_PKTQ_MAX], pkt);
        r->bytes += (size_t)r->q[(r->head + r->count) % VIEW_PKTQ_MAX]->size;
        r->count++;
        r->depth_sum += r->count;
        r->depth_n++;
        SDL_BroadcastCondition(r->ready);
    }

    SDL_UnlockMutex(r->lock);
    av_packet_free(&pkt);

    return 0;
}

static int view_reader_start(struct view_decoder *d)
{
    struct view_reader *r = &d->reader;

    r->lock = SDL_CreateMutex();
    r->wake = SDL_CreateCondition();
    r->ready = SDL_CreateCondition();
    if (!r->lock || !r->wake || !r->ready)
        return CRAW_ERROR(ENOMEM);

    for (int i = 0; i < VIEW_PKTQ_MAX; i++) {
        r->q[i] = av_packet_alloc();
        if (!r->q[i])
            return CRAW_ERROR(ENOMEM);
    }

    r->thread = SDL_CreateThread(view_reader_thread, "craw-view reader", d);
    if (!r->thread)
        return CRAW_ERROR(EAGAIN);

    return 0;
}

static void view_reader_stop(struct view_decoder *d)
{
    struct view_reader *r = &d->reader;

    if (r->thread) {
        SDL_LockMutex(r->lock);
        r->quit = true;
        SDL_BroadcastCondition(r->wake);
        SDL_UnlockMutex(r->lock);
        SDL_WaitThread(r->thread, NULL);
        r->thread = NULL;
    }

    if (r->lock) {
        SDL_LockMutex(r->lock);
        view_reader_drop(r);
        SDL_UnlockMutex(r->lock);
    }

    for (int i = 0; i < VIEW_PKTQ_MAX; i++)
        av_packet_free(&r->q[i]);

    if (r->wake)
        SDL_DestroyCondition(r->wake);
    if (r->ready)
        SDL_DestroyCondition(r->ready);
    if (r->lock)
        SDL_DestroyMutex(r->lock);

    *r = (struct view_reader) { 0 };
}

double view_decoder_queued(struct view_decoder *d)
{
    struct view_reader *r = &d->reader;
    double avg;

    SDL_LockMutex(r->lock);
    avg = r->depth_n ? (double)r->depth_sum / (double)r->depth_n : (double)r->count;
    r->depth_sum = r->depth_n = 0;
    SDL_UnlockMutex(r->lock);

    return avg;
}

/* 0: `pkt` now holds a packet, 1: end of file, EAGAIN: the medium has not
 * kept up and the caller must not block on it. */
static int view_reader_get(struct view_decoder *d, AVPacket *pkt)
{
    struct view_reader *r = &d->reader;
    int ret;

    SDL_LockMutex(r->lock);

    if (r->count) {
        av_packet_move_ref(pkt, r->q[r->head]);
        r->bytes -= (size_t)pkt->size;
        r->head = (r->head + 1) % VIEW_PKTQ_MAX;
        r->count--;
        SDL_BroadcastCondition(r->wake);
        ret = 0;
    } else {
        ret = r->eof ? 1 : CRAW_ERROR(EAGAIN);
    }

    SDL_UnlockMutex(r->lock);

    return ret;
}

/* Moves up to `budget` packets from the reader into the decoder. Returns
 * the number moved, 0 when the queue is dry, or a negative error. */
static int view_decoder_feed(struct view_decoder *d, int budget)
{
    int moved = 0;

    while (moved < budget && !d->draining) {
        int ret;

        /* A packet the decoder would not take is kept, not thrown away:
         * dropping it here would silently lose a picture. */
        if (!d->pkt_held) {
            ret = view_reader_get(d, d->pkt);

            if (ret == CRAW_ERROR(EAGAIN))
                break;
            if (ret > 0) {
                /* Only a decoder that buffers pictures needs to be drained,
                 * and draining one is a state a flush does not always undo:
                 * after a loop point the prores_raw hwaccel keeps reporting
                 * end of stream. A codec without delay has nothing held
                 * back, so it is simply left alone. */
                if (d->dec->codec->capabilities & AV_CODEC_CAP_DELAY)
                    avcodec_send_packet(d->dec, NULL);
                d->draining = true;
                return moved + 1;
            }
            d->pkt_held = true;
        }

        ret = avcodec_send_packet(d->dec, d->pkt);
        if (ret == AVERROR(EAGAIN))
            break;              /* the decoder is full, which is the point */

        av_packet_unref(d->pkt);
        d->pkt_held = false;

        if (ret < 0) {
            fprintf(stderr, "feeding the decoder failed: %s\n", av_err2str(ret));
            return CRAW_ERROR(EINVAL);
        }
        moved++;
    }

    return moved;
}

int view_decoder_next(struct view_decoder *d, AVFrame *frame)
{
    int64_t t0;
    int ret;

    while (1) {
        ret = avcodec_receive_frame(d->dec, frame);
        if (ret >= 0) {
            if (!d->know_hwaccel) {
                d->hwaccel = frame->format == AV_PIX_FMT_VULKAN;
                d->know_hwaccel = true;
                if (d->verbose)
                    fprintf(stderr, "decoding %s: %s\n", d->codec_name,
                            d->hwaccel ? "Vulkan hwaccel, frames stay on the GPU"
                                       : "software, frames uploaded per picture");
            }

            /* Keep the decoder fed while it will take more, so the next
             * picture's work is already under way on the GPU. */
            view_decoder_feed(d, 2);
            return 0;
        }
        if (ret == AVERROR_EOF) {
            d->eof = true;
            return 1;
        }
        if (ret != AVERROR(EAGAIN)) {
            fprintf(stderr, "decoding failed: %s\n", av_err2str(ret));
            return CRAW_ERROR(EINVAL);
        }
        if (d->draining) {
            d->eof = true;
            return 1;
        }

        t0 = av_gettime_relative();
        ret = view_decoder_feed(d, 1);
        d->read_ms += (double)(av_gettime_relative() - t0) / 1000.0;

        if (ret < 0)
            return ret;
        if (!ret)
            return 2;           /* the medium has not kept up */
    }
}

int view_decoder_seek(struct view_decoder *d, int64_t frame_index)
{
    struct view_reader *r = &d->reader;
    int64_t ts;

    if (frame_index < 0)
        frame_index = 0;

    /* Every ProRes RAW frame is a keyframe, so a backward seek lands exactly
     * on the requested picture. */
    ts = d->frame_rate.num ? av_rescale_q(frame_index, av_inv_q(d->frame_rate),
                                          d->time_base)
                           : 0;
    ts += d->fmt->streams[d->stream_index]->start_time != AV_NOPTS_VALUE
          ? d->fmt->streams[d->stream_index]->start_time : 0;

    /* The seek happens on the reader thread, which owns the file. Waiting
     * for it to acknowledge is the one place the main loop blocks on the
     * medium, and only when the user asks for it. */
    SDL_LockMutex(r->lock);
    r->seek_ts = ts;
    r->seek_pending = true;
    r->serial++;
    SDL_BroadcastCondition(r->wake);
    while (r->seek_pending && !r->quit)
        SDL_WaitCondition(r->ready, r->lock);
    SDL_UnlockMutex(r->lock);

    if (d->pkt_held) {
        av_packet_unref(d->pkt);
        d->pkt_held = false;
    }

    if (d->eof || d->draining) {
        int ret = view_decoder_codec_open(d);

        if (ret < 0)
            return ret;
        d->know_hwaccel = false;
        d->frames_reset = true;
    } else {
        avcodec_flush_buffers(d->dec);
    }

    d->draining = false;
    d->eof = false;

    return 0;
}

/* --------------------------------------------------------- the profile */

static float view_q2f(AVRational r)
{
    return r.den ? (float)av_q2d(r) : 0.0f;
}

int view_profile_from_frame(CRAWProfile *p, const AVFrame *frame)
{
    const AVFrameSideData *sd;
    const AVRawColorParams *rc;
    const AVProResRawColorParams *pr;
    int ret;

    sd = av_frame_get_side_data(frame, AV_FRAME_DATA_RAW_COLOR_PARAMS);
    if (!sd) {
        fprintf(stderr, "the decoder produced no AV_FRAME_DATA_RAW_COLOR_PARAMS "
                        "side data; craw-view only handles raw camera clips\n");
        return CRAW_ERROR(EINVAL);
    }

    rc = (const AVRawColorParams *)sd->data;
    if (rc->type != AV_RAW_COLOR_PARAMS_PRORES_RAW) {
        fprintf(stderr, "raw colour parameters of type %d are not supported\n",
                (int)rc->type);
        return CRAW_ERROR(ENOTSUP);
    }
    pr = &rc->codec.prores_raw;

    for (int i = 0; i < 4; i++) {
        p->black_level[i] = view_q2f(rc->black_level);
        p->white_level[i] = view_q2f(rc->white_level);
    }

    p->cfa = CRAW_CFA_RGGB;
    p->as_shot_wb[0] = view_q2f(pr->wb_red);
    p->as_shot_wb[1] = 1.0f;
    p->as_shot_wb[2] = view_q2f(pr->wb_blue);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            p->as_shot_matrix[i][j] = view_q2f(pr->color_matrix[i][j]);

    p->gain = view_q2f(pr->gain);
    p->as_shot_cct = (float)rc->wb_cct;

#if CRAW_TOOL_HAVE_EXPOSURE_INDEX
    p->as_shot_ei = view_q2f(rc->as_shot_ei);
    p->exposure_index = view_q2f(rc->exposure_index);
#endif

    ret = craw_profile_validate(p);
    if (ret < 0)
        fprintf(stderr, "the frame's colour parameters do not make a valid profile: %d\n",
                ret);

    return ret;
}
