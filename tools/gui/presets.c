/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* The video export's preset table.
 *
 * Every option name below is one an encoder in this build really has. The
 * ones that are not obvious, and where they were read:
 *
 *  - `profile` on the two ProRes encoders is their own int option with the
 *    named constants auto/proxy/lt/standard/hq/4444/4444xq
 *    (libavcodec/proresenc_kostya.c:975, proresenc_kostya_vulkan.c:1084).
 *    It picks the bitrate table and the quantisation matrices; it does NOT
 *    pick the chroma format. ff_prores_kostya_encode_init() takes that
 *    from the input alone -
 *
 *        ctx->chroma_factor = pix_fmt == AV_PIX_FMT_YUV422P10
 *                             ? CFACTOR_Y422 : CFACTOR_Y444;
 *
 *    (proresenc_kostya_common.c:211) - so a 4:4:4 frame handed to
 *    `-profile hq` is written as 4:4:4 and labelled HQ, which no other
 *    ProRes reader expects. Hence the pinned formats.
 *
 *  - FFV1's version is the generic AVCodecContext `level` option and the
 *    experimental gate is the generic `strict`: ffv1enc.c:626 assigns
 *    s->version = avctx->level, and ffv1enc.c:652 (and
 *    ffv1enc_vulkan.c:1350 for the Vulkan one) refuse version 2 or 4
 *    unless strict_std_compliance <= FF_COMPLIANCE_EXPERIMENTAL, which is
 *    -2. `context` is each encoder's own option.
 *
 *  - "1024 tiles" is FFV1's slice grid, and 1024 is exactly MAX_SLICES
 *    (ffv1.h:71). The Vulkan encoder takes it as two numbers, slices_h and
 *    slices_v, 32 by 32 (ffv1enc_vulkan.c:1644); the software one takes
 *    the generic `slices` and searches for a grid that multiplies out to
 *    it, ff_ffv1_encode_determine_slices() at ffv1enc.c:570, which finds
 *    32 by 32 for 1024. Both were run, at 1920x1013 and at full size.
 *
 *  - APV's tiles are `tile_width` and `tile_height`, in macroblocks
 *    (apv_encode_vulkan.c:1094). The quantiser is its own `qp`, valid to
 *    3 + 6 * bit_depth, so 63 at ten bits (apv_encode_vulkan.c:983).
 *
 *  - SVT-AV1 takes the parameter string through `svtav1-params` and the
 *    speed through `preset` (libsvtav1.c:793). Its rate control is not an
 *    option of its own: libsvtav1.c:240 reads avctx->bit_rate and makes it
 *    VBR when rc_max_rate differs from it and CBR when it does not, and
 *    libsvtav1.c:256 makes `crf` or `qp` constant quality instead. CBR
 *    additionally needs the low delay prediction structure - SVT refuses
 *    it otherwise with "CBR Rate control is currently not supported for
 *    RANDOM_ACCESS/ALL_INTRA" - so that mode adds pred-struct=1, after
 *    which SVT itself forces hierarchical-levels to 2 and says so.
 *
 *  - av1_vulkan's rate control is its own `rc_mode` with the constants
 *    auto/driver/cqp/cbr/vbr, and its profiles are main/high/professional
 *    (libavcodec/vulkan_encode_av1.c's option table).
 *
 *  - `b`, `maxrate`, `bufsize`, `level`, `strict`, `slices` and `profile`
 *    are AVCodecContext options (libavcodec/options_table.h) and reach the
 *    encoder through the same dictionary, because avcodec_open2() applies
 *    it with AV_OPT_SEARCH_CHILDREN. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include <libavutil/buffer.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/time.h>
#include <libavutil/pixdesc.h>

#include "craw/craw.h"
#include "presets.h"

/* ------------------------------------------------------------- ProRes */

/* auto is left unpinned on purpose: with no format of its own it gets the
 * engine's first choice for a ProRes-shaped codec, which is 4:2:2 ten bit,
 * and the encoder then autoselects HQ from it. */
static const struct view_preset view_presets_prores[] = {
    { "auto",     "profile=auto",     AV_PIX_FMT_NONE },
    { "proxy",    "profile=proxy",    AV_PIX_FMT_YUV422P10 },
    { "lt",       "profile=lt",       AV_PIX_FMT_YUV422P10 },
    { "standard", "profile=standard", AV_PIX_FMT_YUV422P10 },
    { "hq",       "profile=hq",       AV_PIX_FMT_YUV422P10 },
    { "4444",     "profile=4444",     AV_PIX_FMT_YUV444P10 },
    { "4444xq",   "profile=4444xq",   AV_PIX_FMT_YUV444P10 },
};
#define VIEW_PRORES_DEF 5       /* 4444 */

/* --------------------------------------------------------------- FFV1 */

/* The deepest RGB either FFV1 encoder takes. Planar rather than packed:
 * the software encoder lists both gbrp16le and rgb48le and the Vulkan one
 * only planar, so one format covers both paths. */
#define VIEW_FFV1_FMT AV_PIX_FMT_GBRP16

/* Only the version 4 presets can carry linear light: float samples are a
 * version 4 feature of FFV1, and both encoders refuse gbrpf32 below it.
 * Both were run. */
static const struct view_preset view_presets_ffv1_vk[] = {
    { "L3",                   "level=3",           VIEW_FFV1_FMT },
    { "L3 (large context)",   "level=3,context=1", VIEW_FFV1_FMT },
    { "L4 (experimental)",    "strict=-2,level=4", VIEW_FFV1_FMT, true },
    { "L4 (experimental, large context)",
      "strict=-2,level=4,context=1", VIEW_FFV1_FMT, true },
};

/* Same four, and the same option names: `context`, `level` and `strict`
 * all exist on the software encoder too. Only the slice grid is spelled
 * differently, which is the family's always-options below. */
#define view_presets_ffv1_sw view_presets_ffv1_vk

/* ---------------------------------------------------------------- APV */

/* 4:2:2 ten bit is APV's own working format and the one profile_idc comes
 * out of (apv_encode_vulkan.c:219). */
static const struct view_preset view_presets_apv[] = {
    { "standard", "", AV_PIX_FMT_YUV422P10 },
    /* The option ranges stop at 32 by 16, but the entropy shader's shared
     * buffers are sized for 1024 transform blocks per tile-component and
     * this build's apv_vulkan refuses anything larger
     * (apv_encode_vulkan.c:970). 32 by 16 macroblocks is 2048, so it is
     * refused; 32 by 8 and 16 by 16 are the largest that open. Left in the
     * table, greyed, rather than quietly encoding something else. */
    { "32x16 tiles (experimental)", "tile_width=32,tile_height=16",
      AV_PIX_FMT_YUV422P10, .why_not =
      "apv_vulkan: 32x16 macroblock tiles are 2048 transform blocks per "
      "tile-component, over this encoder's 1024 ceiling (32x8 and 16x16 "
      "are the largest it opens)" },
};

/* ------------------------------------------------------------- the AV1s */

/* AV1 High profile is 4:4:4, which libsvtav1 has no pixel format for: it
 * lists yuv420p and yuv420p10le and nothing else, and SVT itself answers
 * "Profile 1 requires 4:4:4 color format". Both were run. */
static const struct view_preset view_presets_svtav1[] = {
    { "Main", "profile=main", AV_PIX_FMT_YUV420P10 },
    { "High", "profile=high", AV_PIX_FMT_YUV444P10, .why_not =
      "libsvtav1 takes only 4:2:0 in this build, and AV1 High profile is "
      "4:4:4" },
};

static const struct view_preset view_presets_av1_vk[] = {
    { "Main", "profile=main", AV_PIX_FMT_YUV420P10 },
    { "High", "profile=high", AV_PIX_FMT_YUV444P10 },
};

/* ---------------------------------------------------------------- x264 */

/* x264's own preset names, which are what the family's presets are. The
 * pixel format and the profile go together: ten bit 4:2:0 is High 10. */
static const struct view_preset view_presets_x264[] = {
    { "veryfast", "preset=veryfast", AV_PIX_FMT_YUV420P10 },
    { "fast",     "preset=fast",     AV_PIX_FMT_YUV420P10 },
    { "medium",   "preset=medium",   AV_PIX_FMT_YUV420P10 },
    { "slow",     "preset=slow",     AV_PIX_FMT_YUV420P10 },
    { "veryslow", "preset=veryslow", AV_PIX_FMT_YUV420P10 },
};
#define VIEW_X264_DEF 4         /* veryslow */

/* --------------------------------------------------------- the families */

/* The parameters every SVT-AV1 encode carries, whatever the mode. */
#define VIEW_SVT_PARAMS \
    "hierarchical-levels=5:aq-mode=2:variance-boost-strength=2:" \
    "variance-octile=6:enable-qm=1:qm-min=0:qm-max=15:tune=0"

const struct view_preset_family view_preset_families[] = {
    {
        .name = "ProRes (Vulkan)", .encoder = "prores_ks_vulkan",
        .presets = view_presets_prores,
        .nb_presets = (int)CRAW_ARRAY_ELEMS(view_presets_prores),
        .def = VIEW_PRORES_DEF,
        .ctrl = VIEW_CTRL_ASYNC, .container = "mov",
        .scene_fmt = AV_PIX_FMT_NONE,
        .async_min = 1, .async_max = 4, .async_def = 2,
    },
    {
        .name = "FFv1 (Vulkan)", .encoder = "ffv1_vulkan",
        .presets = view_presets_ffv1_vk,
        .nb_presets = (int)CRAW_ARRAY_ELEMS(view_presets_ffv1_vk),
        .def = 0,
        .ctrl = VIEW_CTRL_ASYNC, .container = "mkv",
        .always = "slices_h=32,slices_v=32",
        /* The one path that can write linear light: libplacebo renders
         * straight into the encoder's own float planes, so nothing is
         * quantised and nothing above diffuse white is clipped. */
        .scene_fmt = AV_PIX_FMT_GBRPF32,
        .async_min = 1, .async_max = 4, .async_def = 2,
    },
    {
        .name = "APV (Vulkan)", .encoder = "apv_vulkan",
        .presets = view_presets_apv,
        .nb_presets = (int)CRAW_ARRAY_ELEMS(view_presets_apv),
        .def = 0,
        .ctrl = VIEW_CTRL_ASYNC | VIEW_CTRL_QP, .container = "mov",
        .scene_fmt = AV_PIX_FMT_NONE,
        .async_min = 1, .async_max = 4, .async_def = 2,
        .qp_min = 0, .qp_max = 63, .qp_def = 22,
    },
    {
        .name = "FFv1", .encoder = "ffv1",
        .presets = view_presets_ffv1_sw,
        .nb_presets = (int)CRAW_ARRAY_ELEMS(view_presets_ffv1_sw),
        .def = 0,
        .container = "mkv",
        .always = "slices=1024",
        /* The software encoder carries float too, and nothing about the
         * direct path stops it: the read-back is RGBA32F either way and the
         * frame it is handed is the same planar float the Vulkan one gets.
         * What used to rule this family out was the rendered path's 16 bit
         * unorm texture, and linear light does not go through that any
         * more. */
        .scene_fmt = AV_PIX_FMT_GBRPF32,
    },
    {
        .name = "ProRes", .encoder = "prores_ks",
        .presets = view_presets_prores,
        .nb_presets = (int)CRAW_ARRAY_ELEMS(view_presets_prores),
        .def = VIEW_PRORES_DEF,
        .container = "mov",
        .scene_fmt = AV_PIX_FMT_NONE,
    },
    {
        .name = "SVT-AV1", .encoder = "libsvtav1",
        .presets = view_presets_svtav1,
        .nb_presets = (int)CRAW_ARRAY_ELEMS(view_presets_svtav1),
        .def = 0,
        .ctrl = VIEW_CTRL_CPU | VIEW_CTRL_RC, .container = "mkv",
        .always = "svtav1-params=" VIEW_SVT_PARAMS,
        .scene_fmt = AV_PIX_FMT_NONE,
        /* SVT's own default speed, so an untouched slider is the encoder's
         * own behaviour. */
        .cpu_min = 0, .cpu_max = 13, .cpu_def = 10,
        .rc = {
            { "VBR", true },
            { "CBR", true },
            /* Constant quality, not constant quantiser: SVT's rate control
             * mode 0 is CRF and it is what aq-mode=2 is tuned for. */
            { "CRF", false, 1, 63, 35 },
        },
        .nb_rc = 3, .rc_def = 0, .kbit_def = 40000,
    },
    {
        .name = "x264", .encoder = "libx264",
        .presets = view_presets_x264,
        .nb_presets = (int)CRAW_ARRAY_ELEMS(view_presets_x264),
        .def = VIEW_X264_DEF,
        .ctrl = VIEW_CTRL_RC, .container = "mp4",
        .always = "profile=high10",
        .scene_fmt = AV_PIX_FMT_NONE,
        .rc = {
            { "CRF", false, 0, 51, 18 },
            { "VBR", true },
            { "CBR", true },
        },
        .nb_rc = 3, .rc_def = 0, .kbit_def = 40000,
    },
    {
        .name = "AV1 (Vulkan)", .encoder = "av1_vulkan",
        .presets = view_presets_av1_vk,
        .nb_presets = (int)CRAW_ARRAY_ELEMS(view_presets_av1_vk),
        .def = 0,
        .ctrl = VIEW_CTRL_ASYNC | VIEW_CTRL_RC, .container = "mkv",
        .scene_fmt = AV_PIX_FMT_NONE,
        .async_min = 1, .async_max = 4, .async_def = 2,
        .rc = {
            { "VBR", true },
            { "CBR", true },
            /* The hardware's own mode: a constant quantiser, 0 to 255. */
            { "CQP", false, 0, 255, 100 },
        },
        .nb_rc = 3, .rc_def = 0, .kbit_def = 40000,
        .probe = true,
    },
};

const int view_nb_preset_families =
    (int)CRAW_ARRAY_ELEMS(view_preset_families);

/* --------------------------------------------------------- the colours */

/* What the export is written in.
 *
 * The display entries are the picture as the viewer shows it: tone mapped,
 * graded, grained, and encoded with a transfer a player will undo. BT.709
 * uses the *video* curve rather than sRGB - the still export keeps sRGB
 * because a still is looked at in an image viewer, a movie is played in a
 * player, and the two decode a file differently.
 *
 * The scene entries are the pipeline's own linear light with nothing
 * display-referred on it, written by the direct path: a second pipeline run
 * per frame into an RGBA32F texture, read back and handed to the encoder.
 * H.273 has no code points for the ACES spaces, so those go out as
 * unspecified primaries with a linear transfer and are named in container
 * metadata instead; CIE XYZ does have one, SMPTE ST 428-1's - though note
 * that libplacebo reads that tag back as PL_COLOR_SYSTEM_XYZ and means DCDM
 * X'Y'Z' by it, which is a DCI white point and an ST 428 transfer, not
 * this. That is a reason never to make XYZ a libplacebo target and no
 * reason not to write the tag, which is the only code point the standard
 * has for these values. */
const struct view_colour view_colours[VIEW_NB_COLOURS] = {
    [VIEW_COLOUR_709] = {
        .name = "BT.709", .key = "bt709", .group = "BT.709",
        .pri = AVCOL_PRI_BT709, .trc = AVCOL_TRC_BT709,
        .spc = AVCOL_SPC_BT709,
    },
    [VIEW_COLOUR_2020_SDR] = {
        .name = "BT.2020 SDR", .key = "bt2020-sdr", .group = "BT.2020",
        .pri = AVCOL_PRI_BT2020, .trc = AVCOL_TRC_BT709,
        .spc = AVCOL_SPC_BT2020_NCL,
    },
    [VIEW_COLOUR_2020_PQ] = {
        .name = "BT.2020 PQ", .key = "bt2020-pq", .group = "BT.2020",
        .hdr = true,
        .pri = AVCOL_PRI_BT2020, .trc = AVCOL_TRC_SMPTE2084,
        .spc = AVCOL_SPC_BT2020_NCL,
    },
    [VIEW_COLOUR_2020_HLG] = {
        .name = "BT.2020 HLG", .key = "bt2020-hlg", .group = "BT.2020",
        .hdr = true,
        .pri = AVCOL_PRI_BT2020, .trc = AVCOL_TRC_ARIB_STD_B67,
        .spc = AVCOL_SPC_BT2020_NCL,
    },
    [VIEW_COLOUR_ACES] = {
        .name = "ACES", .key = "aces", .group = "ACES",
        .scene = true, .needs_deep_rgb = true,
        .pri = AVCOL_PRI_UNSPECIFIED, .trc = AVCOL_TRC_LINEAR,
        .spc = AVCOL_SPC_RGB, .tag = "ACES2065-1",
    },
    [VIEW_COLOUR_ACESCG] = {
        .name = "ACEScg", .key = "acescg", .group = "ACEScg",
        .scene = true, .needs_deep_rgb = true,
        .pri = AVCOL_PRI_UNSPECIFIED, .trc = AVCOL_TRC_LINEAR,
        .spc = AVCOL_SPC_RGB, .tag = "ACEScg",
    },
    [VIEW_COLOUR_XYZ] = {
        .name = "XYZ linear", .key = "xyz", .group = "XYZ linear",
        .scene = true, .needs_deep_rgb = true,
        .pri = AVCOL_PRI_SMPTE428, .trc = AVCOL_TRC_LINEAR,
        .spc = AVCOL_SPC_RGB, .tag = "CIE-XYZ-D65",
    },
};

/* ---------------------------------------------------------- selections */

static const struct view_preset_family *view_family(int i)
{
    if (i < 0 || i >= view_nb_preset_families)
        return NULL;

    return &view_preset_families[i];
}

static const struct view_preset *view_preset(int f, int p)
{
    const struct view_preset_family *fam = view_family(f);

    if (!fam || p < 0 || p >= fam->nb_presets)
        return NULL;

    return &fam->presets[p];
}

void view_presets_family_defaults(struct view_preset_sel *s)
{
    const struct view_preset_family *f = view_family(s->family);

    if (!f)
        return;

    s->async = f->async_def;
    s->qp    = f->qp_def;
    s->cpu   = f->cpu_def;
    s->rc    = f->rc_def;
    s->kbit  = f->kbit_def ? f->kbit_def : 40000;
    s->quant = f->nb_rc ? f->rc[f->rc_def].qdef : 0;
    /* A mode with no quantiser of its own still has to leave the field on
     * something sensible for when the switch moves to one that does. */
    if (!s->quant)
        for (int i = 0; i < f->nb_rc; i++)
            if (!f->rc[i].bitrate)
                s->quant = f->rc[i].qdef;
}

void view_presets_default(struct view_preset_sel *s)
{
    memset(s, 0, sizeof(*s));

    /* FFv1 on the GPU: lossless, fast here, and the one family whose
     * output is the same numbers as a still of the same frame. */
    s->family = 1;
    s->preset = view_preset_families[1].def;
    s->colour = VIEW_COLOUR_709;
    s->hdr_peak = 1000;
    s->tone_map = VIEW_TONE_MAP_DEF;
    s->gamut_map = VIEW_GAMUT_MAP_DEF;
    s->peak = 1;
    s->size = VIEW_SIZE_FULL;
    view_presets_family_defaults(s);
}

/* --------------------------------------------------------------- the size */

const char *const view_size_names[VIEW_NB_SIZES] = {
    "full", "1/2", "1/4", "3840 wide", "1920 wide", "typed",
};

void view_presets_size(const struct view_preset_sel *s, int cw, int ch,
                       int *w, int *h)
{
    int ow = cw, oh = ch;

    switch (s->size) {
    case VIEW_SIZE_HALF:    ow = cw / 2; oh = ch / 2; break;
    case VIEW_SIZE_QUARTER: ow = cw / 4; oh = ch / 4; break;
    case VIEW_SIZE_3840:
    case VIEW_SIZE_1920: {
        int want = s->size == VIEW_SIZE_3840 ? 3840 : 1920;

        /* Never upscale: a clip narrower than the target keeps its own
         * size rather than being blown up to fill a number. */
        if (cw > want && cw > 0) {
            ow = want;
            /* Rounded rather than truncated, so a 5760x3040 frame at 3840
             * comes out 2027 and not 2026: the aspect ratio is the thing
             * being preserved. */
            oh = (int)(((int64_t)ch * want + cw / 2) / cw);
        }
        break;
    }
    case VIEW_SIZE_TYPED:
        if (s->size_w > 0 && s->size_h > 0) {
            ow = s->size_w;
            oh = s->size_h;
        }
        break;
    default:
        break;
    }

    /* Even, which is what a subsampled encoder wants and what the crop
     * already is, and never empty. */
    *w = CRAW_MAX(ow & ~1, 2);
    *h = CRAW_MAX(oh & ~1, 2);
}

const char *view_presets_size_label(const struct view_preset_sel *s,
                                    int cw, int ch, char *buf, size_t n)
{
    int w, h;

    view_presets_size(s, cw, ch, &w, &h);
    snprintf(buf, n, "%s (%dx%d)",
             view_size_names[CRAW_CLAMP(s->size, 0, VIEW_NB_SIZES - 1)], w, h);

    return buf;
}

int view_presets_parse_size(struct view_preset_sel *s, const char *arg)
{
    static const char *const alias[VIEW_NB_SIZES] = {
        "full", "1/2", "1/4", "3840", "1920", NULL,
    };
    int w, h;
    char tail;

    if (!arg || !arg[0]) {
        fprintf(stderr, "--size: expected full, 1/2, 1/4, 3840, 1920 or WxH\n");
        return CRAW_ERROR(EINVAL);
    }

    for (int i = 0; i < VIEW_NB_SIZES; i++) {
        if (alias[i] && (!strcasecmp(alias[i], arg) ||
                         !strcasecmp(view_size_names[i], arg))) {
            s->size = i;
            return 0;
        }
    }

    /* WxH, which is what this option took before the row existed. */
    if (sscanf(arg, "%dx%d%c", &w, &h, &tail) == 2 && w > 0 && h > 0) {
        s->size = VIEW_SIZE_TYPED;
        s->size_w = w;
        s->size_h = h;
        return 0;
    }

    fprintf(stderr, "--size: expected full, 1/2, 1/4, 3840, 1920 or WxH, "
                    "got `%s`\n", arg);

    return CRAW_ERROR(EINVAL);
}

const char *view_presets_label(const struct view_preset_sel *s, char *buf, size_t n)
{
    const struct view_preset_family *f = view_family(s->family);
    const struct view_preset *p = view_preset(s->family, s->preset);

    if (!f || !p)
        snprintf(buf, n, "none");
    else
        snprintf(buf, n, "%s / %s", f->name, p->name);

    return buf;
}

const char *view_presets_encoder(const struct view_preset_sel *s)
{
    const struct view_preset_family *f = view_family(s->family);

    return f ? f->encoder : NULL;
}

enum AVPixelFormat view_presets_pix_fmt(const struct view_preset_sel *s)
{
    const struct view_preset *p = view_preset(s->family, s->preset);

    return p ? p->fmt : AV_PIX_FMT_NONE;
}

const char *view_presets_container(const struct view_preset_sel *s)
{
    const struct view_preset_family *f = view_family(s->family);

    return f && f->container ? f->container : "mkv";
}

/* ------------------------------------------------------- the option string */

/* Appends "key=value" with the comma av_dict_parse_string() splits on. */
static void view_opt_add(char *buf, size_t n, const char *fmt, ...)
{
    size_t len = strlen(buf);
    va_list ap;

    if (len + 1 >= n)
        return;
    if (len) {
        buf[len++] = ',';
        buf[len] = '\0';
    }

    va_start(ap, fmt);
    vsnprintf(buf + len, n - len, fmt, ap);
    va_end(ap);
}

static void view_opt_cat(char *buf, size_t n, const char *s)
{
    if (s && s[0])
        view_opt_add(buf, n, "%s", s);
}

const char *view_presets_opts(const struct view_preset_sel *s, char *buf, size_t n)
{
    const struct view_preset_family *f = view_family(s->family);
    const struct view_preset *p = view_preset(s->family, s->preset);
    const struct view_rc_spec *rc;
    int64_t bps;

    buf[0] = '\0';
    if (!f || !p)
        return buf;

    /* SVT-AV1's CBR needs the low delay prediction structure, so its
     * parameter string is not the family's constant one. Everything else
     * takes the family's as it stands. */
    if (f->ctrl & VIEW_CTRL_RC && !strcmp(f->encoder, "libsvtav1") &&
        s->rc >= 0 && s->rc < f->nb_rc && !strcmp(f->rc[s->rc].label, "CBR"))
        view_opt_add(buf, n, "svtav1-params=pred-struct=1:%s", VIEW_SVT_PARAMS);
    else
        view_opt_cat(buf, n, f->always);

    view_opt_cat(buf, n, p->opts);

    if (f->ctrl & VIEW_CTRL_ASYNC)
        view_opt_add(buf, n, "async_depth=%d",
                     CRAW_CLAMP(s->async, f->async_min, f->async_max));
    if (f->ctrl & VIEW_CTRL_QP)
        view_opt_add(buf, n, "qp=%d", CRAW_CLAMP(s->qp, f->qp_min, f->qp_max));
    if (f->ctrl & VIEW_CTRL_CPU)
        view_opt_add(buf, n, "preset=%d",
                     CRAW_CLAMP(s->cpu, f->cpu_min, f->cpu_max));

    if (!(f->ctrl & VIEW_CTRL_RC) || s->rc < 0 || s->rc >= f->nb_rc)
        return buf;

    rc = &f->rc[s->rc];
    bps = (int64_t)CRAW_MAX(s->kbit, 1) * 1000;

    /* av1_vulkan says which mode it is in; the two software encoders are
     * put into one by what is and is not set, which is what their code
     * reads. */
    if (!strcmp(f->encoder, "av1_vulkan")) {
        if (rc->bitrate)
            view_opt_add(buf, n, "rc_mode=%s,b=%" PRId64,
                         strcmp(rc->label, "CBR") ? "vbr" : "cbr", bps);
        else
            view_opt_add(buf, n, "rc_mode=cqp,qp=%d",
                         CRAW_CLAMP(s->quant, rc->qmin, rc->qmax));
    } else if (!strcmp(f->encoder, "libsvtav1")) {
        if (!rc->bitrate)
            view_opt_add(buf, n, "crf=%d",
                         CRAW_CLAMP(s->quant, rc->qmin, rc->qmax));
        else if (!strcmp(rc->label, "CBR"))
            view_opt_add(buf, n, "b=%" PRId64 ",maxrate=%" PRId64
                                 ",bufsize=%" PRId64, bps, bps, bps);
        else
            /* VBR is bit_rate set and rc_max_rate different from it
             * (libsvtav1.c:242), and rc_max_rate defaults to zero, so the
             * target alone says it. A ceiling must NOT be set: SVT answers
             * "Max Bitrate only supported with CRF mode" and refuses to
             * open. Run, and it does. */
            view_opt_add(buf, n, "b=%" PRId64, bps);
    } else {
        if (!rc->bitrate)
            view_opt_add(buf, n, "crf=%d",
                         CRAW_CLAMP(s->quant, rc->qmin, rc->qmax));
        else if (!strcmp(rc->label, "CBR"))
            view_opt_add(buf, n, "b=%" PRId64 ",maxrate=%" PRId64
                                 ",bufsize=%" PRId64 ",nal-hrd=cbr",
                         bps, bps, bps);
        else
            /* x264's average bitrate mode: a target and no VBV, which is
             * what "variable bitrate" means without a delivery channel to
             * constrain it to. */
            view_opt_add(buf, n, "b=%" PRId64, bps);
    }

    return buf;
}

int view_presets_dict(const struct view_preset_sel *s, const char *extra,
                      AVDictionary **out)
{
    char buf[512];
    int ret;

    view_presets_opts(s, buf, sizeof(buf));

    if (buf[0]) {
        ret = av_dict_parse_string(out, buf, "=", ",", 0);
        if (ret < 0) {
            fprintf(stderr, "the preset table produced `%s`, which is not "
                            "an option list\n", buf);
            return CRAW_ERROR(EINVAL);
        }
    }

    /* Last, so that --enc wins over the table. */
    if (extra && extra[0]) {
        ret = av_dict_parse_string(out, extra, "=", ",", 0);
        if (ret < 0) {
            fprintf(stderr, "--enc: cannot parse `%s`\n", extra);
            return CRAW_ERROR(EINVAL);
        }
    }

    return 0;
}

/* --------------------------------------------------------- availability */

/* One entry per family: what the probe found, or nothing yet. */
static struct {
    bool done;
    char why[192];
} view_probe[16];

static const char *view_presets_build_why_not(const struct view_preset_family *f)
{
    static char buf[160];

    if (!avcodec_find_encoder_by_name(f->encoder)) {
        snprintf(buf, sizeof(buf), "this FFmpeg has no %s encoder", f->encoder);
        return buf;
    }

    return NULL;
}

const char *view_presets_family_why_not(int family)
{
    const struct view_preset_family *f = view_family(family);
    const char *why;

    if (!f)
        return "no such family";

    why = view_presets_build_why_not(f);
    if (why)
        return why;

    if (family < (int)CRAW_ARRAY_ELEMS(view_probe) &&
        view_probe[family].done && view_probe[family].why[0])
        return view_probe[family].why;

    return NULL;
}

const char *view_presets_preset_why_not(int family, int preset)
{
    const struct view_preset *p = view_preset(family, preset);
    const char *why = view_presets_family_why_not(family);

    if (why)
        return why;

    return p ? p->why_not : "no such preset";
}

const char *view_presets_colour_why_not(int family, int preset, int colour)
{
    const struct view_preset_family *f = view_family(family);
    const struct view_preset *p = view_preset(family, preset);

    if (colour < 0 || colour >= VIEW_NB_COLOURS ||
        !view_colours[colour].needs_deep_rgb)
        return NULL;
    if (!f || !p)
        return "no such preset";

    /* Linear light in an integer codec is two problems at once: ten bits
     * of it bands visibly in the shadows, because the code values are
     * evenly spaced in light and the eye is not, and *any* unorm format
     * clips at 1.0, which throws away everything above diffuse white -
     * which on a raw clip is most of the highlights. So the scene-referred
     * spaces are offered only where the samples are float, and here that
     * is FFV1's gbrpf32. */
    if (f->scene_fmt == AV_PIX_FMT_NONE)
        return "linear light needs float samples, which of these families "
               "only the two FFv1 ones can write";
    if (!p->scene_ok)
        return "linear light needs FFV1 version 4 for float samples: pick "
               "an L4 preset";

    return NULL;
}

bool view_presets_probe_step(AVBufferRef *hw_device, bool verbose)
{
    const struct view_preset_family *f = NULL;
    const AVCodec *codec;
    AVCodecContext *avc = NULL;
    AVBufferRef *frames = NULL;
    AVHWFramesContext *fc;
    AVDictionary *opts = NULL;
    struct view_preset_sel sel = { 0 };
    char buf[512];
    int idx = -1, ret;
    int64_t t0;

    for (int i = 0; i < view_nb_preset_families &&
                    i < (int)CRAW_ARRAY_ELEMS(view_probe); i++) {
        if (view_preset_families[i].probe && !view_probe[i].done) {
            idx = i;
            f = &view_preset_families[i];
            break;
        }
    }
    if (!f)
        return false;

    /* No device, no answer; say so rather than claiming the family works. */
    codec = avcodec_find_encoder_by_name(f->encoder);
    if (!codec) {
        view_probe[idx].done = true;        /* the build check says it */
        return true;
    }
    if (!hw_device) {
        snprintf(view_probe[idx].why, sizeof(view_probe[idx].why),
                 "%s needs a Vulkan device and there is none", f->encoder);
        view_probe[idx].done = true;
        return true;
    }

    t0 = av_gettime_relative();

    frames = av_hwframe_ctx_alloc(hw_device);
    if (!frames) {
        view_probe[idx].done = true;
        return true;
    }
    fc = (AVHWFramesContext *)frames->data;
    fc->format    = AV_PIX_FMT_VULKAN;
    fc->sw_format = AV_PIX_FMT_YUV420P10;
    fc->width     = 1920;
    fc->height    = 1080;
    fc->initial_pool_size = 0;

    ret = av_hwframe_ctx_init(frames);
    if (ret < 0)
        goto out;

    avc = avcodec_alloc_context3(codec);
    if (!avc) {
        ret = AVERROR(ENOMEM);
        goto out;
    }
    avc->width     = fc->width;
    avc->height    = fc->height;
    avc->pix_fmt   = AV_PIX_FMT_VULKAN;
    avc->sw_pix_fmt = fc->sw_format;
    avc->time_base = (AVRational) { 1, 25 };
    avc->framerate = (AVRational) { 25, 1 };
    avc->hw_frames_ctx = av_buffer_ref(frames);
    if (!avc->hw_frames_ctx) {
        ret = AVERROR(ENOMEM);
        goto out;
    }

    /* The family's own default preset, so that the probe answers the
     * question the menu is about to ask. */
    sel.family = idx;
    sel.preset = f->def;
    view_presets_family_defaults(&sel);
    view_presets_opts(&sel, buf, sizeof(buf));
    if (buf[0])
        av_dict_parse_string(&opts, buf, "=", ",", 0);

    ret = avcodec_open2(avc, codec, &opts);

out:
    if (ret < 0)
        snprintf(view_probe[idx].why, sizeof(view_probe[idx].why),
                 "%s: %s", f->encoder,
                 ret == AVERROR(ENOSYS) || ret == AVERROR(EINVAL)
                 ? "this GPU has no encode engine for it" : av_err2str(ret));
    view_probe[idx].done = true;

    if (verbose)
        fprintf(stderr, "probed %s in %.0f ms: %s\n", f->encoder,
                (double)(av_gettime_relative() - t0) / 1000.0,
                view_probe[idx].why[0] ? view_probe[idx].why : "usable");

    av_dict_free(&opts);
    avcodec_free_context(&avc);
    av_buffer_unref(&frames);

    return true;
}

/* ------------------------------------------------------- the names */

static bool view_name_eq(const char *a, const char *b)
{
    return a && b && !strcasecmp(a, b);
}

void view_presets_list(void)
{
    char buf[512];

    printf("video export presets. --preset \"Family/Preset\"\n\n");

    for (int i = 0; i < view_nb_preset_families; i++) {
        const struct view_preset_family *f = &view_preset_families[i];
        const char *why = view_presets_family_why_not(i);

        printf("%s (%s, .%s)%s%s\n", f->name, f->encoder, f->container,
               why ? "  unusable: " : "", why ? why : "");

        for (int j = 0; j < f->nb_presets; j++) {
            struct view_preset_sel sel = { .family = i, .preset = j };

            view_presets_family_defaults(&sel);
            view_presets_opts(&sel, buf, sizeof(buf));
            printf("  %c %-34s %s%s%s\n", j == f->def ? '*' : ' ',
                   f->presets[j].name, buf,
                   f->presets[j].why_not ? "   unusable: " : "",
                   f->presets[j].why_not ? f->presets[j].why_not : "");
        }
        printf("\n");
    }

    printf("colour spaces. --colour NAME\n\n");
    for (int i = 0; i < VIEW_NB_COLOURS; i++)
        printf("  %-12s %s%s\n", view_colours[i].key, view_colours[i].name,
               view_colours[i].scene
               ? "  (scene referred, written straight from the pipeline: "
                 "float samples only, so FFv1 (Vulkan) with an L4 preset, "
                 "native resolution, no look and no tone mapping)" : "");

    printf("\nsizes. --size NAME\n\n");
    for (int i = 0; i < VIEW_NB_SIZES; i++)
        printf("  %-12s %s\n", view_size_names[i],
               i == VIEW_SIZE_TYPED ? "WxH, given as e.g. --size 2048x1080"
                                    : "of the crop rectangle");
}

int view_presets_parse(struct view_preset_sel *s, const char *arg)
{
    const char *slash = arg ? strchr(arg, '/') : NULL;
    char fam[64];
    size_t n;

    if (!arg || !arg[0]) {
        fprintf(stderr, "--preset: expected Family/Preset\n");
        return CRAW_ERROR(EINVAL);
    }

    n = slash ? (size_t)(slash - arg) : strlen(arg);
    while (n && arg[n - 1] == ' ')
        n--;
    if (n >= sizeof(fam))
        n = sizeof(fam) - 1;
    memcpy(fam, arg, n);
    fam[n] = '\0';

    for (int i = 0; i < view_nb_preset_families; i++) {
        const struct view_preset_family *f = &view_preset_families[i];

        if (!view_name_eq(f->name, fam) && !view_name_eq(f->encoder, fam))
            continue;

        s->family = i;
        s->preset = f->def;
        view_presets_family_defaults(s);
        s->set = true;

        if (!slash)
            return 0;

        while (*++slash == ' ')
            ;
        for (int j = 0; j < f->nb_presets; j++) {
            if (!view_name_eq(f->presets[j].name, slash))
                continue;
            s->preset = j;
            return 0;
        }

        fprintf(stderr, "--preset: %s has no preset called `%s`. It has:\n",
                f->name, slash);
        for (int j = 0; j < f->nb_presets; j++)
            fprintf(stderr, "  %s\n", f->presets[j].name);

        return CRAW_ERROR(EINVAL);
    }

    fprintf(stderr, "--preset: no family called `%s`. There are:\n", fam);
    for (int i = 0; i < view_nb_preset_families; i++)
        fprintf(stderr, "  %-18s (%s)\n", view_preset_families[i].name,
                view_preset_families[i].encoder);

    return CRAW_ERROR(EINVAL);
}

int view_presets_parse_colour(struct view_preset_sel *s, const char *arg)
{
    if (!arg || !arg[0]) {
        fprintf(stderr, "--colour: expected a name\n");
        return CRAW_ERROR(EINVAL);
    }

    for (int i = 0; i < VIEW_NB_COLOURS; i++) {
        if (view_name_eq(view_colours[i].key, arg) ||
            view_name_eq(view_colours[i].name, arg)) {
            s->colour = i;
            return 0;
        }
    }

    fprintf(stderr, "--colour: no colour space called `%s`. There are:\n", arg);
    for (int i = 0; i < VIEW_NB_COLOURS; i++)
        fprintf(stderr, "  %-12s %s\n", view_colours[i].key,
                view_colours[i].name);

    return CRAW_ERROR(EINVAL);
}
