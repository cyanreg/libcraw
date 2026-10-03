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
#include <time.h>

#include <libavcodec/defs.h>
#include <libavutil/dict.h>
#include <libavutil/display.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
#include <libavutil/parseutils.h>
#include <libavutil/pixdesc.h>
#include <libavutil/time.h>

#include <libxml/parser.h>
#include <libxml/tree.h>
#include <libxml/xpath.h>

#include "view.h"
#include "../profiles.h"

static enum AVPixelFormat view_get_format(AVCodecContext *avc,
                                          const enum AVPixelFormat *fmts);

/* Whether the decoder can hand its pictures over on the GPU through a
 * Vulkan hwaccel - ProRes RAW's can - rather than only in system memory,
 * which is all the RW2 decoder has. */
static bool view_codec_has_vulkan(const AVCodec *codec)
{
    const AVCodecHWConfig *cfg;

    for (int i = 0; (cfg = avcodec_get_hw_config(codec, i)); i++)
        if (cfg->device_type == AV_HWDEVICE_TYPE_VULKAN &&
            (cfg->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX))
            return true;

    return false;
}

/* Whether a decoder in the pool can take this stream through a flush and
 * the stream's extradata on its first packet, which is how every codec
 * this viewer has met so far takes the next clip. The RW2 decoder cannot:
 * it reads the file's whole header - the stripes, the levels, the white
 * balance - from the context's extradata at every picture and looks at
 * nothing on the packet, so a decoder opened on one still would decode the
 * next still's stripes with the first one's header. It is kept only for a
 * stream with the same header, which is the same file again as a playlist
 * comes back round to it; anything else gets a decoder of its own, which
 * for a software decoder with nothing to set up costs nothing. */
static bool view_decoder_takes(const AVCodecContext *avc, const AVCodecParameters *par)
{
#if CRAW_TOOL_HAVE_RW2
    if (par->codec_id == AV_CODEC_ID_RW2)
        return avc->extradata_size == par->extradata_size &&
               (!par->extradata_size ||
                !memcmp(avc->extradata, par->extradata, (size_t)par->extradata_size));
#else
    (void)avc;
    (void)par;
#endif

    return true;
}

/* Opens the decoder for the clip's stream, or takes the one the pool holds
 * for its codec. libavcodec takes one stream after another through a
 * context: flushed between them, handed the new stream's extradata on its
 * first packet (view_decoder_feed()), and reading a change of size or
 * format off the bitstream itself, which is the one case where it sets the
 * hwaccel up again. Building a context per clip instead cost the hwaccel's
 * two shaders and its frame pool every time: 25 to 80 ms and the first two
 * or three pictures, measured at every switch of a playlist and at every
 * loop point - the seek used to rebuild the context after end of stream,
 * on a report that this hwaccel did not come back from a flush, which it
 * does (see view_decoder_seek()). */
static int view_decoder_codec_open(struct view_decoder *d)
{
    AVStream *st = d->fmt->streams[d->stream_index];
    const AVCodec *codec = avcodec_find_decoder(st->codecpar->codec_id);
    AVCodecContext *avc, **pool;
    int ret;

    if (!codec)
        return CRAW_ERROR(ENOSYS);

    for (int i = 0; i < d->codec.nb_pool; i++) {
        if (d->codec.pool[i]->codec_id != st->codecpar->codec_id)
            continue;
        /* One that cannot take this stream goes, and a new one opens in its
         * place below; the pool keeps one decoder per codec either way. */
        if (!view_decoder_takes(d->codec.pool[i], st->codecpar)) {
            avcodec_free_context(&d->codec.pool[i]);
            d->codec.pool[i] = d->codec.pool[--d->codec.nb_pool];
            if (d->verbose)
                fprintf(stderr, "decoder: %s from the pool cannot take another "
                                "file's header; a new one opens\n", codec->name);
            break;
        }
        /* Clean since its last clip closed (view_decoder_close()). */
        d->codec.avc = d->codec.pool[i];
        d->new_extradata = st->codecpar->extradata_size > 0;
        if (d->verbose)
            fprintf(stderr, "decoder: %s from the pool%s\n", codec->name,
                    d->new_extradata ? ", the stream's extradata goes on its first packet"
                                     : "");
        return 0;
    }

    avc = avcodec_alloc_context3(codec);
    if (!avc)
        return CRAW_ERROR(ENOMEM);

    ret = avcodec_parameters_to_context(avc, st->codecpar);
    if (ret < 0) {
        ret = CRAW_ERROR(EINVAL);
        goto fail;
    }

    avc->hw_device_ctx = av_buffer_ref(d->codec.hw_device);
    if (!avc->hw_device_ctx) {
        ret = CRAW_ERROR(ENOMEM);
        goto fail;
    }
    avc->opaque = d;
    avc->get_format = view_get_format;
    /* On the raw path the hwaccel does the work on the GPU and extra CPU
     * threads would only race the frame properties read under the frames
     * context lock: the AVVkFrame's layout, access and semaphore value,
     * which the decode threads' submissions move while view_gpu_develop()
     * reads them. A raw decoder without a Vulkan hwaccel - RW2's - decodes
     * into system memory, where no frames context and no lock are involved
     * until the picture is uploaded on this thread, and its threads are the
     * decode time. Slice threads, because a still is one packet: they split
     * that one picture's stripes across the cores, where frame threads
     * would give it one core and hold it back until the stream ends. A
     * non-raw clip may end up decoding in software, where threads are the
     * difference between real time and not. */
    if (d->raw && view_codec_has_vulkan(codec)) {
        avc->thread_count = 1;
    } else if (d->raw) {
        avc->thread_type = FF_THREAD_SLICE;
        avc->thread_count = 0;
    } else {
        avc->thread_count = 0;
    }

    /* A ProRes RAW frame header can carry a crop (the Sony FX6 clips mark
     * 3840x2160 inside a 3872x2192 readout), and for this camera the
     * sensels outside it are real picture, not padding. So the decoder is
     * told not to crop: the frame stays the full coded size with all four
     * crop fields intact, libcraw develops every sensel of it, and the
     * crop becomes the viewer's default crop rectangle, which the user can
     * clear to see the whole readout. A non-raw clip keeps FFmpeg's own
     * behaviour, where a crop is a container's business and not a look. */
    if (d->raw)
        avc->apply_cropping = 0;

    ret = avcodec_open2(avc, codec, NULL);
    if (ret < 0) {
        fprintf(stderr, "cannot open the %s decoder: %s\n", codec->name,
                av_err2str(ret));
        ret = CRAW_ERROR(EINVAL);
        goto fail;
    }

    pool = realloc(d->codec.pool, (size_t)(d->codec.nb_pool + 1) * sizeof(*pool));
    if (!pool) {
        ret = CRAW_ERROR(ENOMEM);
        goto fail;
    }
    d->codec.pool = pool;
    d->codec.pool[d->codec.nb_pool++] = avc;
    d->codec.avc = avc;

    if (d->verbose)
        fprintf(stderr, "decoder: %s opened, %d in the pool\n", codec->name,
                d->codec.nb_pool);

    return 0;

fail:
    avcodec_free_context(&avc);
    return ret;
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

/* Once per program: the device is libplacebo's and lives as long as the
 * window does. */
static int view_hw_device_create(struct view_decoder *d, struct view_gpu *g)
{
    AVHWDeviceContext *hwctx;
    AVVulkanDeviceContext *vkctx;
    VkQueueFamilyProperties *qprops = NULL;
    uint32_t nb_q = 0;
    int ret;

    if (d->codec.hw_device)
        return 0;

    d->codec.hw_device = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_VULKAN);
    if (!d->codec.hw_device)
        return CRAW_ERROR(ENOMEM);

    hwctx = (AVHWDeviceContext *)d->codec.hw_device->data;
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

    ret = av_hwdevice_ctx_init(d->codec.hw_device);
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
 * and the pacing drops frames if it cannot keep up. A raw clip takes the
 * same fallback, and so does every RW2, whose decoder has no hwaccel: the
 * mosaic comes in system memory and view_decoder_next() uploads it. */
static enum AVPixelFormat view_get_format(AVCodecContext *avc,
                                          const enum AVPixelFormat *fmts)
{
    for (const enum AVPixelFormat *f = fmts; *f != AV_PIX_FMT_NONE; f++)
        if (*f == AV_PIX_FMT_VULKAN)
            return *f;

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

/* ------------------------------------------------- the clip metadata XML */

/* Cameras write their acquisition record into the container as an XML
 * document - Panasonic's `com.panasonic.Semi-Pro.metadata.xml` here, and
 * ARRI's is another - and what it says decides how the picture is read, so
 * it is parsed rather than pattern-matched.
 *
 * Every expression below matches on local names instead of registering the
 * document's prefixes. That is not laziness about namespaces: the two
 * subtrees are in different ones to begin with - ClipMain in
 * `urn:schemas-Professional-Plug-in:Semi-Pro:ClipMetadata` and
 * AcquisitionMetadata in `urn:schemas-Professional-Plug-in:P2:CameraMetadata`
 * - and the first of those carries a schema version in the URI that the
 * camera really does change: the HEVC clips say v1.0 and the ProRes RAW
 * clip out of the same body says v1.2. Pinning the URI would make the next
 * firmware release read as a clip with no metadata at all. The path is kept
 * full so that the element is still matched in its own context. */
#define VIEW_XP_CUM "//*[local-name()='CameraUnitMetadata']"
#define VIEW_XP_ESS "//*[local-name()='EssenceList']/*[local-name()='Video']"
#define VIEW_XP_DEV "//*[local-name()='ClipMetadata']/*[local-name()='Device']"
#define VIEW_XP_ACC "//*[local-name()='ClipMetadata']/*[local-name()='Access']"

/* The text of the first node an expression selects, whitespace trimmed.
 * Anything unexpected - no match, an empty element, a value longer than the
 * buffer - leaves `out` alone, so a field reads as absent rather than as
 * something made up. */
static bool view_xpath_text(xmlXPathContextPtr xp, const char *expr,
                            char *out, size_t size)
{
    xmlXPathObjectPtr obj = xmlXPathEvalExpression((const xmlChar *)expr, xp);
    xmlChar *text;
    bool ok = false;

    if (!obj)
        return false;

    if (obj->type == XPATH_NODESET && obj->nodesetval &&
        obj->nodesetval->nodeNr > 0 &&
        (text = xmlNodeGetContent(obj->nodesetval->nodeTab[0]))) {
        const char *b = (const char *)text;
        const char *e = b + strlen(b);
        size_t n;

        while (b < e && (unsigned char)*b <= ' ')
            b++;
        while (e > b && (unsigned char)e[-1] <= ' ')
            e--;

        n = (size_t)(e - b);
        if (n && n < size) {
            memcpy(out, b, n);
            out[n] = '\0';
            ok = true;
        }

        xmlFree(text);
    }

    xmlXPathFreeObject(obj);

    return ok;
}

/* The XML's dates are ISO 8601 with the camera's own zone -
 * "2025-07-05T01:42:48+09:00" - which is kept as it is, the T made a space
 * and the zone set off by one, so that the row reads as a date and a time
 * rather than as a token. Anything shaped differently is passed through
 * with only the T changed: it is still the camera's word for when. */
static void view_date_from_xml(const char *iso, char *out, size_t n)
{
    size_t len = strlen(iso);
    size_t o = 0;

    for (size_t i = 0; i < len && o + 2 < n; i++) {
        if (i == 10 && iso[i] == 'T') {
            out[o++] = ' ';
            continue;
        }
        if (i == 19 && (iso[i] == '+' || iso[i] == '-'))
            out[o++] = ' ';
        out[o++] = iso[i];
    }
    out[o] = '\0';
}

/* Reads one such document. Networked entity loading is off: this is a blob
 * out of a file the user opened, and it must not become a fetch. */
static void view_read_clip_xml(struct view_decoder *d, const char *xml)
{
    xmlDocPtr doc;
    xmlXPathContextPtr xp;
    char iso[32], num[32], date[40];

    /* Idempotent, and matched by the xmlCleanupParser() in
     * view_decoder_free(): libxml2's global state is not a per-clip thing,
     * so it is set up whenever a document is read and released once, when
     * the program ends. */
    xmlInitParser();

    doc = xmlReadMemory(xml, (int)strlen(xml), NULL, NULL,
                        XML_PARSE_NONET | XML_PARSE_NOERROR |
                        XML_PARSE_NOWARNING);
    if (!doc) {
        if (d->verbose)
            fprintf(stderr, "the clip metadata XML did not parse; only the "
                            "stream's own colour tags are left\n");
        return;
    }

    xp = xmlXPathNewContext(doc);
    if (xp) {
        view_xpath_text(xp, VIEW_XP_CUM "/*[local-name()='Gamma']"
                            "/*[local-name()='CaptureGamma']",
                        d->capture_gamma, sizeof(d->capture_gamma));
        view_xpath_text(xp, VIEW_XP_CUM "/*[local-name()='Gamut']"
                            "/*[local-name()='CaptureGamut']",
                        d->capture_gamut, sizeof(d->capture_gamut));

        /* What was recorded, in the camera's words: `H265_420_LongGOP` or
         * `ProRes_RAW`. Nothing is decided by it - the codec id is what
         * says whether a clip is Bayer - but it is the clearest
         * confirmation of that rule there is, so -v prints it. */
        view_xpath_text(xp, VIEW_XP_ESS "/*[local-name()='Codec']",
                        d->essence_codec, sizeof(d->essence_codec));

        /* The as-shot exposure index, for a clip that has no RDD 18 keys at
         * all - which is every HEVC clip this camera writes. What the
         * exposure section re-rates against. */
        if (!(d->meta_as_shot_ei > 0.0f) &&
            view_xpath_text(xp, VIEW_XP_CUM "/*[local-name()='ISOSensitivity']",
                            iso, sizeof(iso)))
            d->meta_as_shot_ei = (float)atof(iso);

        /* The rest of the record, for the Clip block. The camera and the
         * white balance fill in behind the Apple and RDD 18 tags, which a
         * developed clip does not carry; the frame rate and the depth are
         * the XML's own words wherever it has them, since "29.97p" says
         * more than 30000/1001 does. */
        if (!d->cam_make[0])
            view_xpath_text(xp, VIEW_XP_DEV "/*[local-name()='Manufacturer']",
                            d->cam_make, sizeof(d->cam_make));
        if (!d->cam_model[0])
            view_xpath_text(xp, VIEW_XP_DEV "/*[local-name()='ModelName']",
                            d->cam_model, sizeof(d->cam_model));
        view_xpath_text(xp, VIEW_XP_ESS "/*[local-name()='FrameRate']",
                        d->frame_rate_str, sizeof(d->frame_rate_str));
        if (view_xpath_text(xp, VIEW_XP_ESS "/*[local-name()='BitDepth']",
                            num, sizeof(num)) && atoi(num) > 0)
            d->bit_depth = atoi(num);
        /* Only a clip recorded in the camera's Kelvin mode carries this,
         * and then as "5500K". */
        if (!d->wb_kelvin &&
            view_xpath_text(xp, VIEW_XP_CUM
                            "/*[local-name()='WhiteBalanceColorTemperature']",
                            num, sizeof(num)) && atoi(num) > 0)
            d->wb_kelvin = atoi(num);
        if (!d->timecode[0])
            view_xpath_text(xp, VIEW_XP_ESS "/*[local-name()='StartTimecode']",
                            d->timecode, sizeof(d->timecode));
        /* Preferred over the container's creation_time, which is UTC: this
         * one carries the zone the camera was set to, which is where the
         * clip was shot. */
        if (view_xpath_text(xp, VIEW_XP_ACC "/*[local-name()='CreationDate']",
                            date, sizeof(date)))
            view_date_from_xml(date, d->recorded, sizeof(d->recorded));

        xmlXPathFreeContext(xp);
    }

    xmlFreeDoc(doc);
}

/* The container's creation_time, an ISO 8601 UTC instant, as a date and
 * time in this machine's zone: what the clock on the wall said here when
 * the clip was shot, which is the best a file without the XML can do. */
static void view_date_from_utc(const char *iso, char *out, size_t n)
{
    int64_t us;
    time_t t;
    struct tm tm;
    char zone[8];
    size_t len;

    if (av_parse_time(&us, iso, 0) < 0)
        return;

    t = (time_t)(us / 1000000);
    if (!localtime_r(&t, &tm))
        return;

    len = strftime(out, n, "%Y-%m-%d %H:%M:%S ", &tm);
    if (!len)
        return;
    /* strftime's zone is "+0900"; the XML writes "+09:00", and one shape
     * on the row is better than two. */
    if (strftime(zone, sizeof(zone), "%z", &tm) == 5)
        snprintf(out + len, n - len, "%.3s:%s", zone, zone + 3);
}

/* The camera model and the as-shot ISO hide in whichever metadata
 * dictionary the muxer felt like using, so look at both. `xml` is the clip
 * XML pana.c found in an MP4's psmd box, NULL for any other file. */
static void view_read_metadata(struct view_decoder *d, const char *xml)
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
        if (e && !d->cam_model[0])
            snprintf(d->cam_model, sizeof(d->cam_model), "%s", e->value);

        /* The rest of what a ProRes RAW clip's Apple and RDD 18 tags say
         * about the take, for the Clip block. The two aperture keys carry
         * the same number in two spellings - "2.800000" and "F2.8" - so
         * either serves. */
        e = av_dict_get(dicts[i], "com.apple.proapps.manufacturer", NULL, 0);
        if (e && !d->cam_make[0])
            snprintf(d->cam_make, sizeof(d->cam_make), "%s", e->value);

        e = av_dict_get(dicts[i], "com.apple.proapps.exif.{Exif}.FNumber", NULL, 0);
        if (e && !(d->aperture > 0.0f))
            d->aperture = (float)atof(e->value);
        e = av_dict_get(dicts[i], "org.smpte.rdd18.lens.irisfnumber", NULL, 0);
        if (e && !(d->aperture > 0.0f))
            d->aperture = (float)atof(e->value + (e->value[0] == 'F' ||
                                                  e->value[0] == 'f'));

        e = av_dict_get(dicts[i], "com.apple.proapps.exif.{Exif}.ExposureIndex", NULL, 0);
        if (e && !(d->exif_ei > 0.0f))
            d->exif_ei = (float)atof(e->value);

        e = av_dict_get(dicts[i], "org.smpte.rdd18.camera.whitebalance", NULL, 0);
        if (e && !d->wb_kelvin && atoi(e->value) > 0)
            d->wb_kelvin = atoi(e->value);

        /* The tmcd track's start, which the demuxer copies onto the video
         * stream; the format dictionary has none and the loop is harmless
         * there. */
        e = av_dict_get(dicts[i], "timecode", NULL, 0);
        if (e && !d->timecode[0])
            snprintf(d->timecode, sizeof(d->timecode), "%s", e->value);

        /* Overwritten below by the XML's own date, which knows the zone. */
        e = av_dict_get(dicts[i], "creation_time", NULL, 0);
        if (e && !d->recorded[0])
            view_date_from_utc(e->value, d->recorded, sizeof(d->recorded));
    }

    /* The Panasonic clip XML. It is the only record of what the camera was
     * actually recording: an HEVC V-Log clip carries a BT.709 VUI -
     * primaries, transfer and matrix all 709 - which is true of the matrix
     * and false of the other two. The key is looked up by name first and
     * then any entry that opens like a document is tried, so a muxer that
     * files the blob somewhere else still works. */
    for (int i = 0; i < 2 && !d->capture_gamma[0]; i++) {
        const AVDictionaryEntry *e;

        e = av_dict_get(dicts[i], "com.panasonic.Semi-Pro.metadata.xml", NULL, 0);
        if (e)
            view_read_clip_xml(d, e->value);
    }

    for (int i = 0; i < 2 && !d->capture_gamma[0]; i++) {
        const AVDictionaryEntry *e = NULL;

        while ((e = av_dict_iterate(dicts[i], e)))
            if (e->value && !strncmp(e->value, "<?xml", 5))
                view_read_clip_xml(d, e->value);
    }

    /* An MP4 clip's copy of the same document, which the demuxer skips. */
    if (!d->capture_gamma[0] && xml)
        view_read_clip_xml(d, xml);

    /* The maker note fills in behind everything above, and is all a
     * developed clip has for the aperture and for a white balance the
     * camera chose itself: the XML names a colour temperature only when it
     * was set in Kelvin. */
    if (!(d->meta_as_shot_ei > 0.0f))
        d->meta_as_shot_ei = d->pana.iso;
    if (!(d->aperture > 0.0f))
        d->aperture = d->pana.fnumber;
    if (!d->wb_kelvin)
        d->wb_kelvin = d->pana.wb_kelvin;

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

/* The codec the way a person names it, for the Clip block's format row.
 * FFmpeg's short names are identifiers - `prores_raw`, `hevc` - and its
 * long ones are catalogue entries, so the few this viewer is likely to meet
 * are named here and anything else keeps the short name. A ProRes profile
 * is part of what the format is, and the stream carries it. */
static void view_format_name(const AVCodecParameters *par, const AVCodec *codec,
                             char *out, size_t n)
{
    const char *profile = NULL;

    switch (par->codec_id) {
    case AV_CODEC_ID_PRORES_RAW:
        snprintf(out, n, "ProRes RAW");
        return;
#if CRAW_TOOL_HAVE_RW2
    case AV_CODEC_ID_RW2:
        snprintf(out, n, "RW2");
        return;
#endif
    case AV_CODEC_ID_PRORES:
        switch (par->profile) {
        case AV_PROFILE_PRORES_PROXY:    profile = "422 Proxy"; break;
        case AV_PROFILE_PRORES_LT:       profile = "422 LT";    break;
        case AV_PROFILE_PRORES_STANDARD: profile = "422";       break;
        case AV_PROFILE_PRORES_HQ:       profile = "422 HQ";    break;
        case AV_PROFILE_PRORES_4444:     profile = "4444";      break;
        case AV_PROFILE_PRORES_XQ:       profile = "4444 XQ";   break;
        default: break;
        }
        snprintf(out, n, "ProRes%s%s", profile ? " " : "", profile ? profile : "");
        return;
    case AV_CODEC_ID_HEVC: snprintf(out, n, "HEVC");  return;
    case AV_CODEC_ID_H264: snprintf(out, n, "H.264"); return;
    case AV_CODEC_ID_AV1:  snprintf(out, n, "AV1");   return;
    default:
        snprintf(out, n, "%s", codec->name);
        return;
    }
}

/* Panasonic's white balance modes, as exiftool names them. */
static const char *view_wb_mode_name(int mode)
{
    switch (mode) {
    case 1:  return "auto";
    case 2:  return "daylight";
    case 3:  return "cloudy";
    case 4:  return "incandescent";
    case 5:  return "manual";
    case 8:  return "flash";
    case 11: return "manual 2";
    case 12: return "shade";
    case 14: return "manual 3";
    case 15: return "manual 4";
    case 19: return "auto (cool)";
    default: return NULL;
    }
}

int view_decoder_info(const struct view_decoder *d, struct view_info_row *rows,
                      int max)
{
    int n = 0;

#define VIEW_INFO(lbl, ...) do {                                            \
        if (n < max) {                                                      \
            rows[n].label = lbl;                                            \
            snprintf(rows[n].value, sizeof(rows[n].value), __VA_ARGS__);    \
            n++;                                                            \
        }                                                                   \
    } while (0)

    if (d->cam_make[0] || d->cam_model[0])
        VIEW_INFO("camera", "%s%s%s", d->cam_make,
                  d->cam_make[0] && d->cam_model[0] ? " " : "", d->cam_model);

    /* The focal length is the first frame's: a zoom during the take is
     * not in the file. */
    if (d->pana.lens[0] && d->pana.focal_mm > 0.0f)
        VIEW_INFO("lens", "%s, %.0f mm", d->pana.lens, (double)d->pana.focal_mm);
    else if (d->pana.lens[0])
        VIEW_INFO("lens", "%s", d->pana.lens);
    else if (d->pana.focal_mm > 0.0f)
        VIEW_INFO("lens", "%.0f mm", (double)d->pana.focal_mm);

    if (d->format_name[0]) {
        char depth[16] = "";

        if (d->bit_depth > 0)
            snprintf(depth, sizeof(depth), " %d-bit", d->bit_depth);
        VIEW_INFO("format", "%s%s %dx%d", d->format_name, depth,
                  d->coded_w, d->coded_h);
    }

    if (d->still) {
        /* A still's rate is the demuxer's placeholder for one picture, and
         * says nothing about it. */
    } else if (d->frame_rate_str[0]) {
        VIEW_INFO("frame rate", "%s", d->frame_rate_str);
    } else if (d->frame_rate.num > 0 && d->frame_rate.den > 0) {
        VIEW_INFO("frame rate", d->frame_rate.num % d->frame_rate.den
                  ? "%.2f fps" : "%.0f fps", av_q2d(d->frame_rate));
    }

    /* The exposure on one row, "ISO 100  f/2.8  1/15", with what the clip
     * has of the three. */
    {
        char exposure[64] = "";
        size_t len = 0;

        if (d->meta_as_shot_ei > 0.0f)
            len += (size_t)snprintf(exposure + len, sizeof(exposure) - len,
                                    "ISO %.0f", (double)d->meta_as_shot_ei);
        if (d->aperture > 0.0f)
            len += (size_t)snprintf(exposure + len, sizeof(exposure) - len,
                                    "%sf/%.1f", len ? "  " : "", (double)d->aperture);
        if (d->pana.exposure_s >= 1.0f)
            snprintf(exposure + len, sizeof(exposure) - len, "%s%.1f s",
                     len ? "  " : "", (double)d->pana.exposure_s);
        else if (d->pana.exposure_s > 0.0f)
            snprintf(exposure + len, sizeof(exposure) - len, "%s1/%.0f",
                     len ? "  " : "", 1.0 / d->pana.exposure_s);
        if (exposure[0])
            VIEW_INFO("exposure", "%s", exposure);
    }
    /* The mode is said only when it is not Kelvin, which the number
     * already says. */
    if (d->wb_kelvin > 0 && view_wb_mode_name(d->pana.wb_mode))
        VIEW_INFO("white balance", "%s, %d K",
                  view_wb_mode_name(d->pana.wb_mode), d->wb_kelvin);
    else if (d->wb_kelvin > 0)
        VIEW_INFO("white balance", "%d K", d->wb_kelvin);
    /* The Exif index is the ISO again on every clip seen so far; the row
     * exists for the day a clip rates itself differently. */
    if (d->exif_ei > 0.0f && fabsf(d->exif_ei - d->meta_as_shot_ei) >= 0.5f)
        VIEW_INFO("exposure index", "%.0f", (double)d->exif_ei);

    /* A raw clip's XML says V-Log too, describing what the sensor data
     * would have become; the pipeline develops it and the row would
     * mislead. */
    if (!d->raw && (d->capture_gamma[0] || d->capture_gamut[0]))
        VIEW_INFO("gamma", "%s%s%s", d->capture_gamma,
                  d->capture_gamma[0] && d->capture_gamut[0] ? " / " : "",
                  d->capture_gamut);

    if (d->timecode[0])
        VIEW_INFO("timecode", "%s", d->timecode);
    if (d->recorded[0])
        VIEW_INFO("recorded", "%s", d->recorded);

#undef VIEW_INFO

    return n;
}

/* The track header's display matrix, which is how a camera says the body
 * was on its side - or, for a still, the frame's, which the RW2 decoder
 * makes from the file's Exif orientation (view_read_frame_info()).
 *
 * FFmpeg carries a clip's as AV_PKT_DATA_DISPLAYMATRIX in the stream's
 * coded side data, and av_display_rotation_get() reduces it to the angle
 * the matrix turns the frame counter-clockwise - the turn that displays it
 * upright, not the one that was applied when shooting. pl_rotation counts
 * quarter turns clockwise, so the two are each other's negation and
 * nothing else.
 * The Lumix writes 90 on every clip shot on its side, and ffmpeg's
 * autorotate answers it with `transpose=cclock`.
 *
 * Anything that is not a quarter turn is refused rather than approximated:
 * an arbitrary angle needs a resample and a mirror is not a rotation at
 * all, so a matrix carrying either leaves the picture alone and says so.
 * The determinant of the linear part is what a mirror shows up in - a
 * reflection reverses orientation and so makes it negative - and it is
 * taken in 64 bits because the entries are 16.16 fixed point and their
 * products do not fit in 32. */
static void view_apply_display_matrix(struct view_decoder *d, const int32_t *m)
{
    double angle;
    int64_t det;
    int deg;

    angle = av_display_rotation_get(m);
    if (isnan(angle)) {
        fprintf(stderr, "the display matrix is singular; the picture is left "
                        "as it was decoded\n");
        return;
    }

    deg = ((int)lrint(angle) % 360 + 360) % 360;
    det = (int64_t)m[0] * m[4] - (int64_t)m[1] * m[3];

    if (fabs(angle - lrint(angle)) > 0.5 || deg % 90 || det <= 0) {
        fprintf(stderr, "the display matrix is %.2f degrees%s, which this "
                        "viewer does not apply; the picture is left as it was "
                        "decoded\n",
                angle, det < 0 ? " with a flip" : "");
        return;
    }

    d->rotation_deg = deg;
    /* Counter-clockwise degrees to clockwise quarter turns. */
    d->rotation = pl_rotation_normalize(-(deg / 90));
}

static void view_read_display_matrix(struct view_decoder *d, const AVStream *st)
{
    const AVPacketSideData *sd;

    sd = av_packet_side_data_get(st->codecpar->coded_side_data,
                                 st->codecpar->nb_coded_side_data,
                                 AV_PKT_DATA_DISPLAYMATRIX);
    if (!sd || sd->size < 9 * sizeof(int32_t))
        return;

    d->stream_matrix = true;
    view_apply_display_matrix(d, (const int32_t *)sd->data);
}

/* An Exif value as the decoder prints it, without the spaces it is padded
 * with ("LUMIX S 24-60/F2.8" and fifty spaces). Anything that does not fit
 * leaves `out` alone, as view_xpath_text() does. */
static bool view_meta_text(const AVDictionary *m, const char *key, char *out,
                           size_t size)
{
    const AVDictionaryEntry *e = av_dict_get(m, key, NULL, 0);
    const char *b, *end;
    size_t n;

    if (!e)
        return false;

    b = e->value;
    end = b + strlen(b);
    while (b < end && (unsigned char)*b <= ' ')
        b++;
    while (end > b && (unsigned char)end[-1] <= ' ')
        end--;

    n = (size_t)(end - b);
    if (!n || n >= size)
        return false;

    memcpy(out, b, n);
    out[n] = '\0';

    return true;
}

/* A number the way av_exif prints one: "    100", or a rational as
 * "    28:10". 0 when there is none. */
static float view_meta_number(const AVDictionary *m, const char *key)
{
    const AVDictionaryEntry *e = av_dict_get(m, key, NULL, 0);
    double num, den;
    char *end;

    if (!e)
        return 0.0f;

    num = strtod(e->value, &end);
    if (end == e->value)
        return 0.0f;
    while (*end == ' ')
        end++;
    if (*end == ':' || *end == '/') {
        den = strtod(end + 1, NULL);
        return den > 0.0 ? (float)(num / den) : 0.0f;
    }

    return (float)num;
}

static void view_decoder_log_info(const struct view_decoder *d)
{
    struct view_info_row rows[VIEW_INFO_MAX];
    int n = view_decoder_info(d, rows, VIEW_INFO_MAX);

    fprintf(stderr, "clip:");
    for (int i = 0; i < n; i++)
        fprintf(stderr, "%s %s %s", i ? "," : "", rows[i].label, rows[i].value);
    fprintf(stderr, "%s\n", n ? "" : " nothing the demuxer exposes");
}

/* What a still says about itself. An RW2 is its own container and its
 * demuxer exports no tags; the decoder attaches the file's Exif IFD to the
 * picture instead, keyed "Make", "Model", "ExifIFD/FNumber" and so on, with
 * the values padded and the rationals written "28:10". Read once, from the
 * first picture, into what the container left empty, which for a clip that
 * has its tags is nothing: the frame only fills in. The display matrix
 * likewise, when the track header had none. */
static void view_read_frame_info(struct view_decoder *d, const AVFrame *frame)
{
    const AVDictionary *m = frame->metadata;
    const AVFrameSideData *sd;
    char date[40], zone[16];
    float v;

    if (!d->cam_make[0])
        view_meta_text(m, "Make", d->cam_make, sizeof(d->cam_make));
    if (!d->cam_model[0])
        view_meta_text(m, "Model", d->cam_model, sizeof(d->cam_model));
    /* The camera's own words are what view_profile_setup() matches the
     * built-in tables against, as it does an Apple model name. */
    if (!d->model[0] && d->cam_model[0] && m && av_dict_get(m, "Model", NULL, 0))
        snprintf(d->model, sizeof(d->model), "%s", d->cam_model);

    if (!(d->meta_as_shot_ei > 0.0f) &&
        (v = view_meta_number(m, "ExifIFD/ISOSpeedRatings")) > 0.0f)
        d->meta_as_shot_ei = v;
    if (!(d->aperture > 0.0f) && (v = view_meta_number(m, "ExifIFD/FNumber")) > 0.0f)
        d->aperture = v;

    /* The shutter, the focal length and the lens go where a clip's maker
     * note puts them, which the Clip block reads them from. */
    if (!(d->pana.exposure_s > 0.0f) &&
        (v = view_meta_number(m, "ExifIFD/ExposureTime")) > 0.0f)
        d->pana.exposure_s = v;
    if (!(d->pana.focal_mm > 0.0f) &&
        (v = view_meta_number(m, "ExifIFD/FocalLength")) > 0.0f)
        d->pana.focal_mm = v;
    /* LensModel, which this libavcodec's table does not name and prints
     * by its tag number. */
    if (!d->pana.lens[0] &&
        !view_meta_text(m, "ExifIFD/LensModel", d->pana.lens, sizeof(d->pana.lens)))
        view_meta_text(m, "ExifIFD/0xA434", d->pana.lens, sizeof(d->pana.lens));

    /* "2025:07:06 19:56:43" and the zone the camera was set to, "+09:00"
     * (OffsetTimeOriginal, by number too), as the XML's date reads. */
    if (!d->recorded[0] &&
        view_meta_text(m, "ExifIFD/DateTimeOriginal", date, sizeof(date)) &&
        strlen(date) == 19) {
        date[4] = date[7] = '-';
        if (view_meta_text(m, "ExifIFD/OffsetTimeOriginal", zone, sizeof(zone)) ||
            view_meta_text(m, "ExifIFD/0x9011", zone, sizeof(zone)))
            snprintf(d->recorded, sizeof(d->recorded), "%s %s", date, zone);
        else
            snprintf(d->recorded, sizeof(d->recorded), "%s", date);
    }

    if (!d->stream_matrix &&
        (sd = av_frame_get_side_data(frame, AV_FRAME_DATA_DISPLAYMATRIX)) &&
        sd->size >= 9 * sizeof(int32_t))
        view_apply_display_matrix(d, (const int32_t *)sd->data);

    /* Said again once the picture has filled it in, which for a clip with
     * its tags changes nothing and is not worth a line. */
    if (d->verbose && (m || d->rotation)) {
        char rot[48] = "";

        if (d->rotation)
            snprintf(rot, sizeof(rot), ", rotated %d deg -> %dx%d", d->rotation_deg,
                     d->rotation % PL_ROTATION_180 ? d->coded_h : d->coded_w,
                     d->rotation % PL_ROTATION_180 ? d->coded_w : d->coded_h);
        fprintf(stderr, "first picture: model `%s`, as-shot ISO %.0f%s\n",
                d->model[0] ? d->model : "(unknown)", d->meta_as_shot_ei, rot);
        view_decoder_log_info(d);
    }
}

/* A raw picture that came in system memory - every RW2, and ProRes RAW
 * whenever its hwaccel would not start - goes to the device here, so that
 * everything after the decoder sees what the hwaccel would have handed
 * over: an AVVkFrame of one R16_UNORM plane. av_hwframe_transfer_data()
 * keeps the hwcontext's contract on it - the copy waits for the frame's
 * semaphore at its value and signals the next one, and the layout, access
 * and queue family it leaves are written back into the AVVkFrame, or the
 * copy is done on the host when the device can - which is exactly what
 * view_gpu_develop() waits for and transitions from, under the same frames
 * lock, as it does for a decoded picture. The side data, the metadata, the
 * crop and the timestamps come across with av_frame_copy_props().
 *
 * The frames context is made on the clip's first such picture, on the
 * device the decoders share, and goes with the clip
 * (view_decoder_close()); a picture of another size or format makes
 * another, which view_decoder_next() sees as a new frame pool and the view
 * cache follows. Only RGGB has a Vulkan format in libavutil, which is all
 * the pipeline develops anyway. */
static int view_decoder_upload(struct view_decoder *d, AVFrame *frame)
{
    AVHWFramesContext *fc = d->upload ? (AVHWFramesContext *)d->upload->data : NULL;
    int64_t t0 = av_gettime_relative();
    AVFrame *hw = NULL;
    int ret;

    if (fc && (fc->sw_format != frame->format || fc->width != frame->width ||
               fc->height != frame->height))
        av_buffer_unref(&d->upload);

    if (!d->upload) {
        AVVulkanFramesContext *vk;

        d->upload = av_hwframe_ctx_alloc(d->codec.hw_device);
        if (!d->upload)
            return CRAW_ERROR(ENOMEM);

        fc = (AVHWFramesContext *)d->upload->data;
        vk = fc->hwctx;

        fc->format            = AV_PIX_FMT_VULKAN;
        fc->sw_format         = (enum AVPixelFormat)frame->format;
        fc->width             = frame->width;
        fc->height            = frame->height;
        fc->initial_pool_size = 0;

        /* libcraw reads the mosaic as a storage image and the upload
         * writes it; libavutil adds whatever else the format supports. */
        vk->usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

        ret = av_hwframe_ctx_init(d->upload);
        if (ret < 0) {
            fprintf(stderr, "cannot put the decoder's %dx%d %s mosaic on the device: "
                            "%s%s\n", frame->width, frame->height,
                    av_get_pix_fmt_name((enum AVPixelFormat)frame->format),
                    av_err2str(ret),
                    frame->format != AV_PIX_FMT_BAYER_RGGB16
                        ? " (libavutil has a Vulkan format for RGGB only)" : "");
            av_buffer_unref(&d->upload);
            return CRAW_ERROR(ENOTSUP);
        }
    }

    hw = av_frame_alloc();
    if (!hw)
        return CRAW_ERROR(ENOMEM);

    ret = av_hwframe_get_buffer(d->upload, hw, 0);
    if (ret >= 0)
        ret = av_hwframe_transfer_data(hw, frame, 0);
    if (ret >= 0)
        ret = av_frame_copy_props(hw, frame);
    if (ret < 0) {
        fprintf(stderr, "uploading the %dx%d mosaic failed: %s\n", frame->width,
                frame->height, av_err2str(ret));
        av_frame_free(&hw);
        return CRAW_ERROR(EIO);
    }

    av_frame_unref(frame);
    av_frame_move_ref(frame, hw);
    av_frame_free(&hw);

    if (d->verbose && !d->uploads)
        fprintf(stderr, "upload: %dx%d %s to the device in %.1f ms\n",
                frame->width, frame->height, av_get_pix_fmt_name(fc->sw_format),
                (double)(av_gettime_relative() - t0) / 1000.0);
    d->uploads++;

    return 0;
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
     * would, so it cannot be used to tell them apart. An RW2 still is the
     * same camera's mosaic, one picture of it. */
    d->raw = st->codecpar->codec_id == AV_CODEC_ID_PRORES_RAW;
#if CRAW_TOOL_HAVE_RW2
    d->raw = d->raw || st->codecpar->codec_id == AV_CODEC_ID_RW2;
#endif
    d->codec_name = codec->name;
    d->verbose = verbose;

    d->time_base  = st->time_base;
    d->frame_rate = st->avg_frame_rate.num ? st->avg_frame_rate : st->r_frame_rate;
    d->nb_frames  = st->nb_frames;

    if (!d->nb_frames && st->duration > 0 && d->frame_rate.num)
        d->nb_frames = av_rescale_q(st->duration, d->time_base,
                                    av_inv_q(d->frame_rate));

    /* One picture, whatever the container says its rate is: an RW2's is a
     * placeholder of 1 fps, and a one-frame clip's is that of a clip it
     * was cut from. main.c shows it and does not loop it. */
    d->still = d->nb_frames == 1;

    d->recv = av_frame_alloc();
    if (!d->recv)
        return CRAW_ERROR(ENOMEM);

    view_read_display_matrix(d, st);

    /* Only a QuickTime or ISO file has boxes to walk. */
    {
        char *xml = NULL;

        if (strstr(d->fmt->iformat->name, "mov"))
            view_pana_read(path, &d->pana, &xml);
        view_read_metadata(d, xml);
        av_free(xml);
    }

    /* The format row's own facts. The XML's bit depth, when there was one,
     * is already in; the stream is asked otherwise - ProRes RAW says 16 in
     * its own field, a developed clip says nothing there and its pixel
     * format says 10. */
    view_format_name(st->codecpar, codec, d->format_name, sizeof(d->format_name));
    d->coded_w = st->codecpar->width;
    d->coded_h = st->codecpar->height;
    if (!d->bit_depth) {
        const AVPixFmtDescriptor *desc =
            av_pix_fmt_desc_get((enum AVPixelFormat)st->codecpar->format);

        if (st->codecpar->bits_per_raw_sample > 0)
            d->bit_depth = st->codecpar->bits_per_raw_sample;
        else if (desc && desc->nb_components > 0)
            d->bit_depth = desc->comp[0].depth;
    }

    ret = view_hw_device_create(d, g);
    if (ret < 0)
        return ret;

    ret = view_decoder_codec_open(d);
    if (ret < 0)
        return ret;

    d->pkt = av_packet_alloc();
    if (!d->pkt)
        return CRAW_ERROR(ENOMEM);

    /* Only a Bayer clip has a lens record to find, and only the pipeline
     * that develops one can use it. Set before the reader starts, which is
     * what makes the flag safe to read from that thread without the lock. */
    d->reader.scan = d->raw;

    ret = view_reader_start(d);
    if (ret < 0)
        return ret;

    if (verbose) {
        char rot[32] = "";

        /* The size printed is the decoder's; the rotation says what is put
         * on screen and into a file, which for a quarter turn is the other
         * way round. */
        if (d->rotation)
            snprintf(rot, sizeof(rot), ", rotated %d deg -> %dx%d",
                     d->rotation_deg,
                     d->rotation % PL_ROTATION_180 ? d->coded_h : d->coded_w,
                     d->rotation % PL_ROTATION_180 ? d->coded_w : d->coded_h);

        fprintf(stderr, "input %s: stream %d, %s (%s), %dx%d%s, %.3f fps, "
                        "%" PRId64 " frames, model `%s`, as-shot ISO %.0f\n",
                path, d->stream_index, codec->name, d->raw ? "raw" : "video",
                d->coded_w, d->coded_h, rot,
                d->frame_rate.den ? av_q2d(d->frame_rate) : 0.0, d->nb_frames,
                d->model[0] ? d->model : "(unknown)", d->meta_as_shot_ei);
    }

    if (verbose && (d->capture_gamma[0] || d->capture_gamut[0] ||
                    d->essence_codec[0]))
        fprintf(stderr, "clip metadata: capture gamma `%s`, gamut `%s`, "
                        "essence `%s`%s\n",
                d->capture_gamma[0] ? d->capture_gamma : "(none)",
                d->capture_gamut[0] ? d->capture_gamut : "(none)",
                d->essence_codec[0] ? d->essence_codec : "(none)",
                d->vlog ? ", overriding the stream's colour tags"
                        : d->raw ? ", raw: the pipeline develops it" : "");

    /* The multipliers are for the white balance of a developed clip and
     * have no row of their own. */
    if (verbose && d->pana.wb_levels[1] > 0.0f)
        fprintf(stderr, "maker note: white balance mode %d, %d K, "
                        "multipliers %.4f %.4f %.4f\n",
                d->pana.wb_mode, d->pana.wb_kelvin, d->pana.wb_levels[0],
                d->pana.wb_levels[1], d->pana.wb_levels[2]);

    /* The Clip block's rows, exactly as the panel shows them. */
    if (verbose)
        view_decoder_log_info(d);

    return 0;
}

void view_decoder_close(struct view_decoder *d)
{
    struct view_codec codec = d->codec;

    view_reader_stop(d);
    av_packet_free(&d->pkt);
    av_frame_free(&d->recv);
    av_buffer_unref(&d->frames);
    /* The uploads' pool is the clip's. A picture still referencing it - the
     * one on screen, which view_close_clip() lets go of first anyway -
     * keeps it alive until it goes. */
    av_buffer_unref(&d->upload);
    avformat_close_input(&d->fmt);

    /* The decoder lets go of the pictures it still held now, at the
     * switch, rather than when a clip of its codec next comes round -
     * which may be never: they would sit in its frame pool's memory until
     * then. It goes back to the pool clean, and the next clip of its codec
     * takes it as it is. */
    if (d->codec.avc)
        avcodec_flush_buffers(d->codec.avc);

    /* view_decoder_open() fills what it needs and nothing else, so what the
     * clip said - its model, its metadata, its V-Log flag, whether its
     * frames carried lens records - goes here rather than linger into the
     * next one's panel. */
    *d = (struct view_decoder) { 0 };
    d->codec = codec;
    d->codec.avc = NULL;
}

void view_decoder_free(struct view_decoder *d)
{
    view_decoder_close(d);
    for (int i = 0; i < d->codec.nb_pool; i++)
        avcodec_free_context(&d->codec.pool[i]);
    free(d->codec.pool);
    d->codec.pool = NULL;
    d->codec.nb_pool = 0;
    d->codec.avc = NULL;
    av_buffer_unref(&d->codec.hw_device);

    /* Pairs with the xmlInitParser() in view_read_clip_xml(), once per
     * program rather than once per clip: libxml2's globals are not a
     * clip's to release while the next one may still parse a document.
     * Safe when none ever did. */
    xmlCleanupParser();
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
    CRAWLensCorrection lens = { 0 };

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

        if (r->eof || r->count >= r->limit || r->bytes >= VIEW_PKTQ_BYTES) {
            SDL_WaitCondition(r->wake, r->lock);
            continue;
        }

        SDL_UnlockMutex(r->lock);
        ret = av_read_frame(d->fmt, pkt);
        /* Off the lock, because it is a walk of a five megabyte packet -
         * microseconds, but the render loop can be waiting on this mutex.
         * `scan` is set before the thread starts and never written again. */
        lens = (CRAWLensCorrection) { 0 };
        if (ret >= 0 && r->scan && pkt->stream_index == d->stream_index)
            craw_lens_from_prores_raw(pkt->data, (size_t)pkt->size, &lens);
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
        r->lens[(r->head + r->count) % VIEW_PKTQ_MAX] = lens;
        r->arrive[(r->head + r->count) % VIEW_PKTQ_MAX] = av_gettime_relative();
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

    r->limit = VIEW_PKTQ_MAX;
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

    r->thread = SDL_CreateThread(view_reader_thread, "cineview reader", d);
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
    r->scan = d->raw;
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
static int view_reader_get(struct view_decoder *d, AVPacket *pkt,
                           CRAWLensCorrection *lens)
{
    struct view_reader *r = &d->reader;
    int ret;

    *lens = (CRAWLensCorrection) { 0 };
    d->pkt_arrive = 0;

    SDL_LockMutex(r->lock);

    if (r->count) {
        av_packet_move_ref(pkt, r->q[r->head]);
        *lens = r->lens[r->head];
        d->pkt_arrive = r->arrive[r->head];
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

bool view_decoder_wait(struct view_decoder *d, int ms)
{
    struct view_reader *r = &d->reader;
    bool ready;

    SDL_LockMutex(r->lock);
    if (!r->count && !r->eof && !r->quit)
        SDL_WaitConditionTimeout(r->ready, r->lock, ms);
    ready = r->count || r->eof;
    SDL_UnlockMutex(r->lock);

    return ready;
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
            ret = view_reader_get(d, d->pkt, &d->pkt_lens);

            if (ret == CRAW_ERROR(EAGAIN))
                break;
            if (ret > 0) {
                /* Only a decoder that buffers pictures needs to be drained,
                 * and draining one is a state a flush does not always undo:
                 * after a loop point the prores_raw hwaccel keeps reporting
                 * end of stream. A codec without delay has nothing held
                 * back, so it is simply left alone - unless it runs frame
                 * threads, which hold every picture back until each thread
                 * has a packet or the stream is drained, whatever the
                 * codec's own delay. Not draining one is what made a clip
                 * of one frame never show it: the picture stayed in the
                 * thread, the end of the stream came, the loop sought back
                 * to the start and the flush threw it away, over and over
                 * (a one-frame ProRes 422 clip: 1041 passes and no picture
                 * in 35 s). A clip of more frames lost its last few to it
                 * at every loop point. */
                if ((d->codec.avc->codec->capabilities & AV_CODEC_CAP_DELAY) ||
                    (d->codec.avc->active_thread_type & FF_THREAD_FRAME))
                    avcodec_send_packet(d->codec.avc, NULL);
                d->draining = true;
                return moved + 1;
            }
            d->pkt_held = true;
        }

        /* The stream's own parameter sets, on the first packet a kept
         * decoder sees of it. A decoder that keeps them (H.264, HEVC)
         * compares and keeps what is unchanged, so an identical set costs a
         * parse and nothing else, and a packet the decoder refuses below
         * is resent with the side data still on it. */
        if (d->new_extradata) {
            const AVCodecParameters *par =
                d->fmt->streams[d->stream_index]->codecpar;
            uint8_t *sd = av_packet_new_side_data(d->pkt, AV_PKT_DATA_NEW_EXTRADATA,
                                                  (size_t)par->extradata_size);

            if (!sd)
                return CRAW_ERROR(ENOMEM);
            memcpy(sd, par->extradata, (size_t)par->extradata_size);
            d->new_extradata = false;
        }

        ret = avcodec_send_packet(d->codec.avc, d->pkt);
        if (ret == AVERROR(EAGAIN))
            break;              /* the decoder is full, which is the point */

        {
            size_t n = CRAW_ARRAY_ELEMS(d->arrive_ring);

            d->arrive_ring[d->arrive_pushed % n].pts = d->pkt->pts;
            d->arrive_ring[d->arrive_pushed % n].t = d->pkt_arrive;
            d->arrive_pushed++;
        }

        if (d->reader.scan) {
            size_t n = CRAW_ARRAY_ELEMS(d->lens_ring);

            d->lens_ring[d->lens_pushed % n].pts = d->pkt->pts;
            d->lens_ring[d->lens_pushed % n].lc = d->pkt_lens;
            d->lens_ring[d->lens_pushed % n].used = d->pkt_lens.valid;
            d->lens_pushed++;
        }

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
        /* Into a frame of the decoder's own, and handed over only when a
         * picture came: avcodec_receive_frame() unreferences what it is
         * given before it looks, and at the end of the stream the caller's
         * frame is the picture on screen, which a grade change still has
         * to be able to develop again - a still's only picture, a clip's
         * last one with --no-loop. */
        ret = avcodec_receive_frame(d->codec.avc, d->recv);
        if (ret >= 0) {
            av_frame_unref(frame);
            av_frame_move_ref(frame, d->recv);

            if (!d->know_hwaccel) {
                d->hwaccel = frame->format == AV_PIX_FMT_VULKAN;
                d->know_hwaccel = true;
                if (d->verbose)
                    fprintf(stderr, "decoding %s: %s, %d threads\n", d->codec_name,
                            d->hwaccel ? "Vulkan hwaccel, frames stay on the GPU"
                                       : "software, frames uploaded per picture",
                            d->codec.avc->thread_count);
            }

            if (!d->frame_info) {
                d->frame_info = true;
                view_read_frame_info(d, frame);
            }

            if (d->raw && frame->format != AV_PIX_FMT_VULKAN) {
                ret = view_decoder_upload(d, frame);
                if (ret < 0)
                    return ret;
            }

            /* A picture out of another frame pool than the last one's; see
             * `frames` in view.h. The first of a clip always is. */
            if (frame->hw_frames_ctx &&
                (!d->frames || d->frames->data != frame->hw_frames_ctx->data)) {
                av_buffer_unref(&d->frames);
                d->frames = av_buffer_ref(frame->hw_frames_ctx);
                d->frames_reset = true;
                if (d->verbose && d->frames_pools++)
                    fprintf(stderr, "decoding %s: the hwaccel was set up again, "
                                    "new frame pool\n", d->codec_name);
            }

            d->frame_arrive = 0;
            for (size_t i = 0; i < CRAW_ARRAY_ELEMS(d->arrive_ring); i++) {
                if (d->arrive_ring[i].t && d->arrive_ring[i].pts == frame->pts) {
                    d->frame_arrive = d->arrive_ring[i].t;
                    break;
                }
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

void view_decoder_lens(struct view_decoder *d, int64_t pts,
                       CRAWLensCorrection *out)
{
    *out = (CRAWLensCorrection) { 0 };

    if (!d->reader.scan)
        return;

    for (size_t i = 0; i < CRAW_ARRAY_ELEMS(d->lens_ring); i++) {
        if (d->lens_ring[i].used && d->lens_ring[i].pts == pts) {
            *out = d->lens_ring[i].lc;
            d->lens_ring[i].used = false;
            break;
        }
    }

    /* Decided as pictures come out rather than as packets go in, so that
     * what the panel says matches what the pipeline was handed. */
    if (out->valid)
        d->lens_any = true;
    else
        d->lens_none = true;
}

void view_decoder_readahead(struct view_decoder *d, int packets)
{
    struct view_reader *r = &d->reader;

    if (!r->lock)
        return;

    SDL_LockMutex(r->lock);
    r->limit = CRAW_CLAMP(packets, 1, VIEW_PKTQ_MAX);
    SDL_BroadcastCondition(r->wake);
    SDL_UnlockMutex(r->lock);
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
    d->pkt_lens = (CRAWLensCorrection) { 0 };
    memset(d->lens_ring, 0, sizeof(d->lens_ring));
    /* Presentation timestamps repeat after a loop point, so a stale entry
     * here would put an old packet's arrival on a new picture. */
    memset(d->arrive_ring, 0, sizeof(d->arrive_ring));

    /* The flush is what undoes a drain as well: libavcodec leaves the
     * draining state with it, whatever the codec and whether or not an
     * hwaccel is in. The context used to be rebuilt here after end of
     * stream instead, on a report that this hwaccel did not come back from
     * a flush; it does, through every loop point of a ProRes RAW clip
     * (which never drains - the codec has no delay) and of an HEVC one
     * (which does). */
    avcodec_flush_buffers(d->codec.avc);

    d->draining = false;
    d->eof = false;

    return 0;
}
