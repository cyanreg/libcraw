/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* The still image writer.
 *
 * One frame, one file, and the format taken from the file name: PNG, TIFF,
 * JPEG, WebP, AVIF and everything else FFmpeg can encode go through the
 * same code, because libavformat already knows which muxer an extension
 * belongs to and which codec that muxer writes. Nothing here is a
 * sequence: exporting a range of frames is a video encode, and this is
 * the screenshot path.
 *
 * What comes in is the sixteen bit RGBA the export target is read back as.
 * What goes out is the deepest format the chosen encoder will take, which
 * for a still means RGB where the encoder has one and 4:4:4 where it does
 * not; the conversion to anything but plain RGB goes through libswscale.
 *
 * The samples are display referred: the export renders through libplacebo
 * into an sRGB target, so what lands in the file is BT.709 primaries with
 * the sRGB transfer at full range, and the stream is tagged to say so. */

#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>

#include <craw/utils.h>

#include "still.h"

/* What we would rather hand an encoder, best first: sixteen bit RGB, then
 * the deep planar RGB the AV1 encoders take, then eight bit RGB, and only
 * then YUV, deepest and least subsampled first. Nothing carrying alpha is
 * offered - the frame is opaque, and a fourth channel of ones is a third
 * of a PNG wasted and a different code path in half the readers. */
static const enum AVPixelFormat view_still_wanted[] = {
    AV_PIX_FMT_RGB48BE,     AV_PIX_FMT_RGB48LE,
    AV_PIX_FMT_GBRP16BE,    AV_PIX_FMT_GBRP16LE,
    AV_PIX_FMT_GBRP12BE,    AV_PIX_FMT_GBRP12LE,
    AV_PIX_FMT_GBRP10BE,    AV_PIX_FMT_GBRP10LE,
    AV_PIX_FMT_RGB24,       AV_PIX_FMT_BGR24,       AV_PIX_FMT_GBRP,
    AV_PIX_FMT_YUV444P12LE, AV_PIX_FMT_YUV444P10LE,
    AV_PIX_FMT_YUV444P,     AV_PIX_FMT_YUVJ444P,
    AV_PIX_FMT_YUV422P10LE, AV_PIX_FMT_YUV422P,     AV_PIX_FMT_YUVJ422P,
    AV_PIX_FMT_YUV420P10LE, AV_PIX_FMT_YUV420P,     AV_PIX_FMT_YUVJ420P,
};

static enum AVPixelFormat view_still_pick(const AVCodec *codec)
{
    const enum AVPixelFormat *list = NULL;
    int n = 0;

    /* An encoder that reports no list takes anything, so the first choice
     * is ours to make. */
    if (avcodec_get_supported_config(NULL, codec, AV_CODEC_CONFIG_PIX_FORMAT,
                                     0, (const void **)&list, &n) < 0 || !list)
        return view_still_wanted[0];

    for (size_t i = 0; i < CRAW_ARRAY_ELEMS(view_still_wanted); i++)
        for (int j = 0; j < n; j++)
            if (list[j] == view_still_wanted[i])
                return view_still_wanted[i];

    /* Nothing we would have chosen: take the encoder's own first format
     * and let the scaler reach it. */
    return n > 0 ? list[0] : AV_PIX_FMT_NONE;
}

static bool view_still_is_rgb(enum AVPixelFormat fmt)
{
    const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(fmt);

    return d && (d->flags & AV_PIX_FMT_FLAG_RGB);
}

/* The RGB targets are packed here rather than through libswscale: dropping
 * the alpha and putting the bytes in the file's order is all they need,
 * and doing it by hand keeps the sixteen bit path exact. */
static bool view_still_pack(AVFrame *f, const uint16_t *rgba)
{
    uint32_t w = (uint32_t)f->width, h = (uint32_t)f->height;
    int hi, lo;

    switch (f->format) {
    case AV_PIX_FMT_RGB48BE:
    case AV_PIX_FMT_RGB48LE:
        /* Sixteen bit PNG is big endian by definition and TIFF is written
         * little endian, so the samples go out byte by byte and the host's
         * own order never comes into it. */
        hi = f->format == AV_PIX_FMT_RGB48BE ? 0 : 1;
        lo = 1 - hi;

        for (uint32_t y = 0; y < h; y++) {
            const uint16_t *s = rgba + (size_t)y * w * 4;
            uint8_t *d = f->data[0] + (ptrdiff_t)y * f->linesize[0];

            for (uint32_t x = 0; x < w; x++) {
                for (int c = 0; c < 3; c++) {
                    unsigned v = s[x * 4 + c];

                    d[(x * 3 + c) * 2 + hi] = (uint8_t)(v >> 8);
                    d[(x * 3 + c) * 2 + lo] = (uint8_t)v;
                }
            }
        }
        return true;

    case AV_PIX_FMT_RGB24:
        for (uint32_t y = 0; y < h; y++) {
            const uint16_t *s = rgba + (size_t)y * w * 4;
            uint8_t *d = f->data[0] + (ptrdiff_t)y * f->linesize[0];

            for (uint32_t x = 0; x < w; x++)
                for (int c = 0; c < 3; c++) {
                    /* 65535 to 255 the way the round trip is exact, rather
                     * than a shift, which darkens every value. */
                    unsigned v = s[x * 4 + c];

                    d[x * 3 + c] = (uint8_t)((v * 255u + 32895u) >> 16);
                }
        }
        return true;

    default:
        return false;
    }
}

/* Everything else - planar RGB, and the 4:4:4 and 4:2:0 an AVIF or a JPEG
 * wants - through libswscale, from the RGBA64 the caller's buffer already
 * is. The frames carry the colour properties, so the matrix and the range
 * come out of the tags rather than out of a guess. */
static int view_still_scale(AVFrame *f, const uint16_t *rgba)
{
    AVFrame *src = av_frame_alloc();
    SwsContext *sws = sws_alloc_context();
    int ret;

    if (!src || !sws) {
        ret = CRAW_ERROR(ENOMEM);
        goto out;
    }

    src->format          = AV_PIX_FMT_RGBA64;
    src->width           = f->width;
    src->height          = f->height;
    src->colorspace      = AVCOL_SPC_RGB;
    src->color_range     = f->color_range;
    src->color_primaries = f->color_primaries;
    src->color_trc       = f->color_trc;

    /* The caller's buffer is an RGBA64 plane in host order already, so the
     * scaler reads it where it lies. */
    ret = av_image_fill_arrays(src->data, src->linesize, (const uint8_t *)rgba,
                               src->format, src->width, src->height, 1);
    if (ret < 0)
        goto out;

    ret = sws_scale_frame(sws, f, src);
    if (ret < 0)
        fprintf(stderr, "sws_scale_frame: %s\n", av_err2str(ret));

out:
    sws_freeContext(sws);
    av_frame_free(&src);
    return ret < 0 ? ret : 0;
}

int view_still_write(const char *path, const char *codec_name,
                     const char *enc_opts, const uint16_t *rgba,
                     uint32_t w, uint32_t h, char *info, size_t info_size)
{
    AVFormatContext *oc = NULL;
    AVCodecContext *avc = NULL;
    const AVCodec *codec = NULL;
    AVStream *stream;
    AVFrame *frame = NULL;
    AVPacket *pkt = NULL;
    AVDictionary *copts = NULL, *mopts = NULL;
    enum AVCodecID id = AV_CODEC_ID_NONE;
    enum AVPixelFormat pix_fmt;
    bool header = false;
    struct stat sb;
    int ret;

    ret = avformat_alloc_output_context2(&oc, NULL, NULL, path);
    if (ret < 0 || !oc) {
        fprintf(stderr, "%s: no muxer for this file name: %s\n",
                path, av_err2str(ret));
        return CRAW_ERROR(EINVAL);
    }

    /* The name picks the codec as well as the muxer: the image2 muxer's
     * own default is MJPEG whatever the extension says, and only
     * av_guess_codec() looks at the name. */
    if (codec_name && codec_name[0]) {
        codec = avcodec_find_encoder_by_name(codec_name);
        if (codec)
            id = codec->id;
    } else {
        id = av_guess_codec(oc->oformat, NULL, path, NULL, AVMEDIA_TYPE_VIDEO);
        codec = avcodec_find_encoder(id);
    }
    if (!codec || codec->type != AVMEDIA_TYPE_VIDEO) {
        fprintf(stderr, "%s: this FFmpeg has no video encoder called %s\n", path,
                codec_name && codec_name[0] ? codec_name : avcodec_get_name(id));
        ret = CRAW_ERROR(ENOSYS);
        goto fail;
    }

    pix_fmt = view_still_pick(codec);
    if (pix_fmt == AV_PIX_FMT_NONE) {
        fprintf(stderr, "%s: %s takes no pixel format we can write\n",
                path, codec->name);
        ret = CRAW_ERROR(ENOTSUP);
        goto fail;
    }

    stream = avformat_new_stream(oc, NULL);
    avc = avcodec_alloc_context3(codec);
    pkt = av_packet_alloc();
    frame = av_frame_alloc();
    if (!stream || !avc || !pkt || !frame) {
        ret = CRAW_ERROR(ENOMEM);
        goto fail;
    }

    avc->width   = (int)w;
    avc->height  = (int)h;
    avc->pix_fmt = pix_fmt;
    /* A still has no frame rate; one second is as good a time base as any
     * and keeps the muxers that want a duration happy. */
    avc->time_base = (AVRational) { 1, 1 };
    avc->framerate = (AVRational) { 1, 1 };

    /* What the export actually is. libplacebo renders it into an sRGB
     * target, so the samples are display referred BT.709 primaries with
     * the IEC 61966-2-1 transfer at full range; a YUV target additionally
     * gets the BT.709 matrix the scaler used to reach it. */
    avc->color_range     = AVCOL_RANGE_JPEG;
    avc->color_primaries = AVCOL_PRI_BT709;
    avc->color_trc       = AVCOL_TRC_IEC61966_2_1;
    avc->colorspace      = view_still_is_rgb(pix_fmt) ? AVCOL_SPC_RGB
                                                      : AVCOL_SPC_BT709;

    /* JFIF and WebP carry no matrix of their own and their readers do not
     * assume the same one: a JPEG is full range BT.601 by definition, and
     * libwebp's tables are the studio swing BT.601 its own decoder
     * inverts. Handing either BT.709 luma, or handing WebP full range
     * luma, comes back shifted from every reader; measured against the
     * PNG of the same frame it is worth four dB for the JPEG and three
     * for the WebP. Anything that can signal what it holds keeps full
     * range BT.709. */
    if (avc->colorspace == AVCOL_SPC_BT709) {
        if (avc->codec_id == AV_CODEC_ID_MJPEG) {
            avc->colorspace = AVCOL_SPC_BT470BG;
        } else if (avc->codec_id == AV_CODEC_ID_WEBP) {
            avc->colorspace = AVCOL_SPC_BT470BG;
            avc->color_range = AVCOL_RANGE_MPEG;
        }
    }
    if (oc->oformat->flags & AVFMT_GLOBALHEADER)
        avc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    switch (avc->codec_id) {
    case AV_CODEC_ID_PNG:
        /* zlib level 1. A 5760x3040 RGB48 frame takes a fifth of the time
         * of the default level 6 and comes out within a per cent of the
         * same size: a screenshot is not an archive. */
        avc->compression_level = 1;
        break;
    case AV_CODEC_ID_TIFF:
        /* A TIFF here is the uncompressed one, for a tool downstream that
         * wants the samples without a decoder. */
        av_dict_set(&copts, "compression_algo", "raw", 0);
        break;
    default:
        break;
    }

    /* Last, so that what was asked for wins over what we would have
     * chosen: an encoder with no way to set its speed or its quality is
     * not much use to anyone. */
    if (enc_opts && enc_opts[0]) {
        ret = av_dict_parse_string(&copts, enc_opts, "=", ",", 0);
        if (ret < 0) {
            fprintf(stderr, "--enc: cannot parse `%s`\n", enc_opts);
            ret = CRAW_ERROR(EINVAL);
            goto fail;
        }
    }

    ret = avcodec_open2(avc, codec, &copts);
    if (ret < 0) {
        fprintf(stderr, "%s: cannot open the %s encoder: %s\n",
                path, codec->name, av_err2str(ret));
        ret = CRAW_ERROR(EINVAL);
        goto fail;
    }
    if (av_dict_count(copts))
        fprintf(stderr, "%s: the %s encoder ignored an option\n", path, codec->name);

    ret = avcodec_parameters_from_context(stream->codecpar, avc);
    if (ret < 0)
        goto fail;
    stream->time_base = avc->time_base;
    stream->avg_frame_rate = avc->framerate;

    frame->format = pix_fmt;
    frame->width  = (int)w;
    frame->height = (int)h;
    frame->color_range     = avc->color_range;
    frame->color_primaries = avc->color_primaries;
    frame->color_trc       = avc->color_trc;
    frame->colorspace      = avc->colorspace;
    frame->pts    = 0;

    ret = av_frame_get_buffer(frame, 0);
    if (ret < 0) {
        ret = CRAW_ERROR(ENOMEM);
        goto fail;
    }

    if (!view_still_pack(frame, rgba)) {
        ret = view_still_scale(frame, rgba);
        if (ret < 0)
            goto fail;
    }

    if (!(oc->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&oc->pb, path, AVIO_FLAG_WRITE);
        if (ret < 0) {
            fprintf(stderr, "%s: cannot open for writing: %s\n",
                    path, av_err2str(ret));
            ret = CRAW_ERROR(EIO);
            goto fail;
        }
    }

    /* image2 writes a numbered sequence unless it is told that the name it
     * was given is the whole of it. */
    if (!strcmp(oc->oformat->name, "image2"))
        av_dict_set(&mopts, "update", "1", 0);

    ret = avformat_write_header(oc, &mopts);
    if (ret < 0) {
        fprintf(stderr, "%s: avformat_write_header: %s\n", path, av_err2str(ret));
        ret = CRAW_ERROR(EIO);
        goto fail;
    }
    header = true;

    ret = avcodec_send_frame(avc, frame);
    if (ret >= 0)
        ret = avcodec_send_frame(avc, NULL);
    if (ret < 0) {
        fprintf(stderr, "%s: avcodec_send_frame: %s\n", path, av_err2str(ret));
        ret = CRAW_ERROR(EINVAL);
        goto fail;
    }

    /* The AV1 encoders hold a frame or two before they say anything, so
     * the packets are taken until the drain runs out rather than once. */
    while (1) {
        ret = avcodec_receive_packet(avc, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            break;
        if (ret < 0) {
            fprintf(stderr, "%s: avcodec_receive_packet: %s\n", path, av_err2str(ret));
            ret = CRAW_ERROR(EINVAL);
            goto fail;
        }

        pkt->stream_index = stream->index;
        av_packet_rescale_ts(pkt, avc->time_base, stream->time_base);

        ret = av_interleaved_write_frame(oc, pkt);
        if (ret < 0) {
            fprintf(stderr, "%s: cannot write the packet: %s\n", path, av_err2str(ret));
            ret = CRAW_ERROR(EIO);
            goto fail;
        }
    }

    ret = av_write_trailer(oc);
    header = false;
    if (ret < 0) {
        fprintf(stderr, "%s: av_write_trailer: %s\n", path, av_err2str(ret));
        ret = CRAW_ERROR(EIO);
        goto fail;
    }

    if (info && info_size) {
        double mb = stat(path, &sb) == 0 ? (double)sb.st_size / 1e6 : 0.0;

        snprintf(info, info_size, "%ux%u %s %s %s, %.1f MB", w, h,
                 oc->oformat->name, codec->name,
                 av_get_pix_fmt_name(pix_fmt), mb);
    }

    ret = 0;

fail:
    if (header)
        av_write_trailer(oc);
    av_dict_free(&copts);
    av_dict_free(&mopts);
    av_packet_free(&pkt);
    av_frame_free(&frame);
    avcodec_free_context(&avc);
    if (oc && oc->pb && !(oc->oformat->flags & AVFMT_NOFILE))
        avio_closep(&oc->pb);
    avformat_free_context(oc);

    return ret;
}
