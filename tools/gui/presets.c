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

#include <libavutil/cpu.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>

#include "craw/craw.h"
#include "presets.h"

/* ------------------------------------------------------------- ProRes */

/* Kept to three wrapped lines: the panel gives a refusal three rows of
 * height and a wrapped label clips at the bottom of the rectangle it is
 * given. */
#define VIEW_PRORES_422_WHY \
    "proxy, LT, standard and HQ are the 4:2:2 ProRes profiles, and " \
    "prores_ks reads the layout off the frame, not off the profile"
#define VIEW_PRORES_444_WHY \
    "4444 and 4444 XQ are the 4:4:4 ProRes profiles, and prores_ks reads " \
    "the layout off the frame, not off the profile"

/* auto is left unpinned on purpose: with no format of its own it gets the
 * engine's first choice for a ProRes-shaped codec, which is 4:2:2 ten bit,
 * and the encoder then autoselects HQ from it. */
static const struct view_preset view_presets_prores[] = {
    { "auto",     "profile=auto",     AV_PIX_FMT_NONE },
    { "proxy",    "profile=proxy",    AV_PIX_FMT_YUV422P10,
      .fmt_why = VIEW_PRORES_422_WHY },
    { "lt",       "profile=lt",       AV_PIX_FMT_YUV422P10,
      .fmt_why = VIEW_PRORES_422_WHY },
    { "standard", "profile=standard", AV_PIX_FMT_YUV422P10,
      .fmt_why = VIEW_PRORES_422_WHY },
    { "hq",       "profile=hq",       AV_PIX_FMT_YUV422P10,
      .fmt_why = VIEW_PRORES_422_WHY },
    { "4444",     "profile=4444",     AV_PIX_FMT_YUV444P10,
      .fmt_why = VIEW_PRORES_444_WHY },
    { "4444xq",   "profile=4444xq",   AV_PIX_FMT_YUV444P10,
      .fmt_why = VIEW_PRORES_444_WHY },
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

#define VIEW_AV1_MAIN_WHY \
    "AV1 Main profile is 4:2:0: 4:2:2 is not an AV1 chroma format at all, " \
    "and 4:4:4 is High"
#define VIEW_AV1_HIGH_WHY "AV1 High profile is 4:4:4"

/* AV1 High profile is 4:4:4, which libsvtav1 has no pixel format for: it
 * lists yuv420p and yuv420p10le and nothing else, and SVT itself answers
 * "Profile 1 requires 4:4:4 color format". Both were run. */
static const struct view_preset view_presets_svtav1[] = {
    { "Main", "profile=main", AV_PIX_FMT_YUV420P10,
      .fmt_why = VIEW_AV1_MAIN_WHY },
    { "High", "profile=high", AV_PIX_FMT_YUV444P10, .why_not =
      "libsvtav1 takes only 4:2:0 in this build, and AV1 High profile is "
      "4:4:4", .fmt_why = VIEW_AV1_HIGH_WHY },
};

static const struct view_preset view_presets_av1_vk[] = {
    { "Main", "profile=main", AV_PIX_FMT_YUV420P10,
      .fmt_why = VIEW_AV1_MAIN_WHY },
    { "High", "profile=high", AV_PIX_FMT_YUV444P10,
      .fmt_why = VIEW_AV1_HIGH_WHY },
};

/* ---------------------------------------------------------------- x264 */

/* x264's own preset names, which are what the family's presets are. The
 * pinned format is eight bit 4:2:0 as NV12, which x264 writes as High and
 * which is the one H.264 every player and every hardware decoder takes;
 * NV12 rather than yuv420p because it is x264's own internal layout, so
 * the encoder takes it without a conversion of its own. The profile is not
 * set alongside it, because x264 picks the lowest profile that can carry
 * the format it is given and the pixel format row would otherwise be a row
 * that refuses everything it offers. Checked: with no `profile` at all,
 * yuv420p10 comes out High 10 and yuv444p10 High 4:4:4 Predictive. */
static const struct view_preset view_presets_x264[] = {
    { "veryfast", "preset=veryfast", AV_PIX_FMT_NV12 },
    { "fast",     "preset=fast",     AV_PIX_FMT_NV12 },
    { "medium",   "preset=medium",   AV_PIX_FMT_NV12 },
    { "slow",     "preset=slow",     AV_PIX_FMT_NV12 },
    { "veryslow", "preset=veryslow", AV_PIX_FMT_NV12 },
};
#define VIEW_X264_DEF 4         /* veryslow */

/* ------------------------------------------------------ H.264 (Vulkan) */

/* The profiles h264_vulkan names, which on a hardware engine is the choice
 * there is to make: High for anything made today, Main for the players
 * that stop there. Both are eight bit 4:2:0, the one layout the engine
 * takes on RADV. */
static const struct view_preset view_presets_h264_vk[] = {
    { "High", "profile=high", AV_PIX_FMT_NV12 },
    { "Main", "profile=main", AV_PIX_FMT_NV12 },
};

/* ------------------------------------------------------- HEVC (Vulkan) */

/* hevc_vulkan picks Main or Main 10 from the format it is handed and
 * refuses a profile that says otherwise ("hevc profile "Main" not
 * supported" for p010 under profile=main), so the preset sets none and
 * the pixel format row is the choice: p010 is Main 10, which every HEVC
 * hardware decoder plays, nv12 is Main. RExt's 4:4:4 does not open on
 * RADV. */
static const struct view_preset view_presets_hevc_vk[] = {
    { "standard", "", AV_PIX_FMT_P010 },
};

/* --------------------------------------------------------- the families */

/* The parameters every SVT-AV1 encode carries, whatever the mode. */
#define VIEW_SVT_PARAMS \
    "hierarchical-levels=5:aq-mode=2:variance-boost-strength=2:" \
    "variance-octile=6:enable-qm=1:qm-min=0:qm-max=15:tune=0"

/* The formats each family's encoder takes, fixed here rather than asked
 * for at run time. A software encoder declares its list, but a Vulkan one
 * declares only "vulkan" and can only be asked by opening it with a frames
 * context of each candidate, which cost 136 to 540 ms an open on the
 * RX 6900 XT and over a hundred opens for the whole table. These are what
 * those opens found there on 2026-09-16, trimmed to the layouts a graded
 * picture is written in: no gray, no alpha, no packed 8-bit RGB. The
 * pinned default of every preset is in its family's list. ffv1_vulkan has
 * no 4:2:0 because its 32x32 slice grid has to divide the chroma planes;
 * av1_vulkan's pair is what FFmpeg's Vulkan AV1 encoder takes elsewhere,
 * this GPU having no AV1 engine to confirm it with. */
static const enum AVPixelFormat view_fmts_prores[] = {
    AV_PIX_FMT_YUV422P10, AV_PIX_FMT_YUV444P10,
};
static const enum AVPixelFormat view_fmts_ffv1_vk[] = {
    AV_PIX_FMT_GBRP16, AV_PIX_FMT_GBRP12, AV_PIX_FMT_GBRP10, AV_PIX_FMT_GBRPF32,
    AV_PIX_FMT_YUV444P16, AV_PIX_FMT_YUV444P12, AV_PIX_FMT_YUV444P10, AV_PIX_FMT_YUV444P,
    AV_PIX_FMT_YUV422P16, AV_PIX_FMT_YUV422P12, AV_PIX_FMT_YUV422P10, AV_PIX_FMT_YUV422P,
};
static const enum AVPixelFormat view_fmts_ffv1[] = {
    AV_PIX_FMT_GBRP16, AV_PIX_FMT_GBRP12, AV_PIX_FMT_GBRP10, AV_PIX_FMT_GBRP, AV_PIX_FMT_GBRPF32,
    AV_PIX_FMT_YUV444P16, AV_PIX_FMT_YUV444P12, AV_PIX_FMT_YUV444P10, AV_PIX_FMT_YUV444P,
    AV_PIX_FMT_YUV422P16, AV_PIX_FMT_YUV422P12, AV_PIX_FMT_YUV422P10, AV_PIX_FMT_YUV422P,
    AV_PIX_FMT_YUV420P16, AV_PIX_FMT_YUV420P12, AV_PIX_FMT_YUV420P10, AV_PIX_FMT_YUV420P,
};
static const enum AVPixelFormat view_fmts_apv[] = {
    AV_PIX_FMT_YUV422P10, AV_PIX_FMT_YUV422P12, AV_PIX_FMT_YUV444P10, AV_PIX_FMT_YUV444P12,
};
static const enum AVPixelFormat view_fmts_svtav1[] = {
    AV_PIX_FMT_YUV420P10, AV_PIX_FMT_YUV420P,
};
static const enum AVPixelFormat view_fmts_x264[] = {
    AV_PIX_FMT_NV12, AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV420P10,
    AV_PIX_FMT_YUV422P10, AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV444P10, AV_PIX_FMT_YUV444P,
};
static const enum AVPixelFormat view_fmts_av1_vk[] = {
    AV_PIX_FMT_P010, AV_PIX_FMT_NV12,
};
static const enum AVPixelFormat view_fmts_h264_vk[] = {
    AV_PIX_FMT_NV12,
};
static const enum AVPixelFormat view_fmts_hevc_vk[] = {
    AV_PIX_FMT_P010, AV_PIX_FMT_NV12,
};

#define VIEW_FMTS(l) .fmts = (l), .nb_fmts = (int)CRAW_ARRAY_ELEMS(l)

const struct view_preset_family view_preset_families[] = {
    {
        .name = "ProRes (Vulkan)", .encoder = "prores_ks_vulkan",
        VIEW_FMTS(view_fmts_prores),
        .presets = view_presets_prores,
        .nb_presets = (int)CRAW_ARRAY_ELEMS(view_presets_prores),
        .def = VIEW_PRORES_DEF,
        .ctrl = VIEW_CTRL_ASYNC, .container = "mov",
        .scene_fmt = AV_PIX_FMT_NONE,
        .async_min = 1, .async_max = 4, .async_def = 2,
    },
    {
        .name = "FFv1 (Vulkan)", .encoder = "ffv1_vulkan",
        VIEW_FMTS(view_fmts_ffv1_vk),
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
        VIEW_FMTS(view_fmts_apv),
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
        VIEW_FMTS(view_fmts_ffv1),
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
        VIEW_FMTS(view_fmts_prores),
        .presets = view_presets_prores,
        .nb_presets = (int)CRAW_ARRAY_ELEMS(view_presets_prores),
        .def = VIEW_PRORES_DEF,
        .container = "mov",
        .scene_fmt = AV_PIX_FMT_NONE,
    },
    {
        .name = "SVT-AV1", .encoder = "libsvtav1",
        VIEW_FMTS(view_fmts_svtav1),
        .presets = view_presets_svtav1,
        .nb_presets = (int)CRAW_ARRAY_ELEMS(view_presets_svtav1),
        .def = 0,
        .ctrl = VIEW_CTRL_CPU | VIEW_CTRL_RC, .container = "mp4",
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
        VIEW_FMTS(view_fmts_x264),
        .presets = view_presets_x264,
        .nb_presets = (int)CRAW_ARRAY_ELEMS(view_presets_x264),
        .def = VIEW_X264_DEF,
        .ctrl = VIEW_CTRL_RC, .container = "mp4",
        .scene_fmt = AV_PIX_FMT_NONE,
        .rc = {
            { "CRF", false, 0, 51, 18 },
            { "VBR", true },
            { "CBR", true },
        },
        .nb_rc = 3, .rc_def = 0, .kbit_def = 40000,
    },
    {
        .name = "H.264 (Vulkan)", .encoder = "h264_vulkan",
        VIEW_FMTS(view_fmts_h264_vk),
        .presets = view_presets_h264_vk,
        .nb_presets = (int)CRAW_ARRAY_ELEMS(view_presets_h264_vk),
        .def = 0,
        .ctrl = VIEW_CTRL_ASYNC | VIEW_CTRL_RC, .container = "mp4",
        .scene_fmt = AV_PIX_FMT_NONE,
        .async_min = 1, .async_max = 4, .async_def = 2,
        .rc = {
            { "VBR", true },
            { "CBR", true },
            /* The engine's constant quantiser, H.264's own 0 to 51. */
            { "CQP", false, 0, 51, 20 },
        },
        .nb_rc = 3, .rc_def = 0, .kbit_def = 40000,
    },
    {
        .name = "HEVC (Vulkan)", .encoder = "hevc_vulkan",
        VIEW_FMTS(view_fmts_hevc_vk),
        .presets = view_presets_hevc_vk,
        .nb_presets = (int)CRAW_ARRAY_ELEMS(view_presets_hevc_vk),
        .def = 0,
        .ctrl = VIEW_CTRL_ASYNC | VIEW_CTRL_RC, .container = "mp4",
        .scene_fmt = AV_PIX_FMT_NONE,
        .async_min = 1, .async_max = 4, .async_def = 2,
        .rc = {
            { "VBR", true },
            { "CBR", true },
            /* Honoured on RADV now: qp 24 and qp 40 are 2.7 and 0.5 Mbit/s
             * of the same 4K range. */
            { "CQP", false, 0, 51, 24 },
        },
        .nb_rc = 3, .rc_def = 0, .kbit_def = 40000,
    },
    {
        .name = "AV1 (Vulkan)", .encoder = "av1_vulkan",
        VIEW_FMTS(view_fmts_av1_vk),
        .presets = view_presets_av1_vk,
        .nb_presets = (int)CRAW_ARRAY_ELEMS(view_presets_av1_vk),
        .def = 0,
        .ctrl = VIEW_CTRL_ASYNC | VIEW_CTRL_RC, .container = "mp4",
        .scene_fmt = AV_PIX_FMT_NONE,
        .async_min = 1, .async_max = 4, .async_def = 2,
        .rc = {
            { "VBR", true },
            { "CBR", true },
            /* The hardware's own mode: a constant quantiser, 0 to 255. */
            { "CQP", false, 0, 255, 100 },
        },
        .nb_rc = 3, .rc_def = 0, .kbit_def = 40000,
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

/* ----------------------------------------------------------- the audio */

/* The bitrate modes start at 160 kbit/s, which is where both codecs are
 * transparent for a stereo location recording; flac's level 5 is the
 * encoder's own default. `wav` names no encoder because the depth is the
 * source's: pcm_s24le, or pcm_s16le for a 16 bit clip, decided by the
 * engine when it has the stream in hand. */
const struct view_audio_spec view_audio_modes[VIEW_NB_AUDIO_MODES] = {
    [VIEW_AUDIO_NONE] = { "none" },
    [VIEW_AUDIO_COPY] = { "copy" },
    [VIEW_AUDIO_OPUS] = { "opus", "libopus", "kbit/s", 6, 512, 160 },
    [VIEW_AUDIO_AAC]  = { "aac",  "aac",     "kbit/s", 8, 512, 160 },
    [VIEW_AUDIO_FLAC] = { "flac", "flac",    "level",  0, 12, 5 },
    [VIEW_AUDIO_WAV]  = { "wav" },
};

int view_presets_parse_audio(struct view_preset_sel *s, const char *arg)
{
    const char *colon = arg ? strchr(arg, ':') : NULL;
    size_t n;

    if (!arg || !arg[0]) {
        fprintf(stderr, "--audio: expected none, copy, opus[:KBPS], aac[:KBPS], "
                        "flac[:LEVEL] or wav\n");
        return CRAW_ERROR(EINVAL);
    }

    n = colon ? (size_t)(colon - arg) : strlen(arg);

    for (int i = 0; i < VIEW_NB_AUDIO_MODES; i++) {
        const struct view_audio_spec *m = &view_audio_modes[i];
        char *end;
        long v;

        if (strlen(m->name) != n || strncasecmp(m->name, arg, n))
            continue;

        s->audio_mode = i;
        if (!colon)
            return 0;

        /* A number on a mode that takes none is a typo worth stopping
         * on: `wav:24` is not a depth this program chooses. */
        if (!m->field) {
            fprintf(stderr, "--audio: %s takes no number\n", m->name);
            return CRAW_ERROR(EINVAL);
        }

        v = strtol(colon + 1, &end, 10);
        if (end == colon + 1 || *end || v < m->min || v > m->max) {
            fprintf(stderr, "--audio: %s takes a %s from %d to %d, got `%s`\n",
                    m->name, m->field, m->min, m->max, colon + 1);
            return CRAW_ERROR(EINVAL);
        }
        s->audio_q[i] = (int)v;

        return 0;
    }

    fprintf(stderr, "--audio: no mode called `%.*s`. There are none, copy, "
                    "opus[:KBPS], aac[:KBPS], flac[:LEVEL] and wav\n",
            (int)n, arg);

    return CRAW_ERROR(EINVAL);
}

const char *view_presets_audio_label(const struct view_preset_sel *s,
                                     char *buf, size_t n)
{
    int mode = CRAW_CLAMP(s->audio_mode, 0, VIEW_NB_AUDIO_MODES - 1);
    const struct view_audio_spec *m = &view_audio_modes[mode];

    if (!m->field)
        snprintf(buf, n, "%s", m->name);
    else if (!strcmp(m->field, "level"))
        snprintf(buf, n, "%s level %d", m->name, s->audio_q[mode]);
    else
        snprintf(buf, n, "%s %d %s", m->name, s->audio_q[mode], m->field);

    return buf;
}

/* --------------------------------------------------------- the threads */

int view_presets_max_threads(void)
{
    return CRAW_MAX(av_cpu_count(), 1);
}

bool view_presets_codec_is_vulkan(const AVCodec *c)
{
    const AVCodecHWConfig *hw;

    if (!c)
        return false;

    for (int i = 0; (hw = avcodec_get_hw_config(c, i)); i++)
        if (hw->device_type == AV_HWDEVICE_TYPE_VULKAN ||
            hw->pix_fmt == AV_PIX_FMT_VULKAN)
            return true;

    /* A name is not evidence on its own, but an encoder that calls itself
     * Vulkan and reports nothing is not one we would drive by uploading
     * system memory into it either. */
    return strstr(c->name, "_vulkan") != NULL;
}

bool view_presets_family_is_vulkan(int family)
{
    if (family < 0 || family >= view_nb_preset_families)
        return false;

    return view_presets_codec_is_vulkan(
        avcodec_find_encoder_by_name(view_preset_families[family].encoder));
}

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
    /* Spelled out: AV_PIX_FMT_NONE is -1 and the memset above made this
     * field AV_PIX_FMT_YUV420P. */
    s->pix_fmt = AV_PIX_FMT_NONE;
    s->pix_fmt_family = -1;
    s->size = VIEW_SIZE_FULL;
    /* The clip's own sound, remuxed: a range of a take without its sound
     * is rarely what anyone wanted. Every encoding mode's number starts on
     * its own default, so the field is right the moment the combo lands on
     * it. */
    s->audio_mode = VIEW_AUDIO_COPY;
    for (int i = 0; i < VIEW_NB_AUDIO_MODES; i++)
        s->audio_q[i] = view_audio_modes[i].def;
    s->threads = 0;
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

    /* The Vulkan encoders say which mode they are in, through the option
     * their common code shares; the two software encoders are put into one
     * by what is and is not set, which is what their code reads. */
    if (!strcmp(f->encoder, "av1_vulkan") || !strcmp(f->encoder, "h264_vulkan") ||
        !strcmp(f->encoder, "hevc_vulkan")) {
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

/* --------------------------------------------------------- pixel formats */

const char *view_presets_fmt_label(enum AVPixelFormat fmt, char *buf, size_t n)
{
    const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(fmt);
    const char *name = av_get_pix_fmt_name(fmt);
    char layout[16];

    if (!d || !name) {
        snprintf(buf, n, "none");
        return buf;
    }

    /* The layout is read off the descriptor rather than looked up: the
     * chroma planes' subsampling *is* the layout, one in x being 4:2:2,
     * one in both 4:2:0, two in x 4:1:1, and a format with no chroma
     * planes at all has no layout to say. */
    if (d->flags & AV_PIX_FMT_FLAG_RGB)
        snprintf(layout, sizeof(layout), "RGB");
    else if (d->nb_components < 3)
        snprintf(layout, sizeof(layout), "gray");
    else
        snprintf(layout, sizeof(layout), "4:%d:%d", 4 >> d->log2_chroma_w,
                 d->log2_chroma_h ? 0 : 4 >> d->log2_chroma_w);

    snprintf(buf, n, "%s%s %d-bit%s (%s)", layout,
             d->flags & AV_PIX_FMT_FLAG_ALPHA ? "+A" : "",
             d->comp[0].depth,
             d->flags & AV_PIX_FMT_FLAG_FLOAT ? " float" : "", name);

    return buf;
}

enum AVPixelFormat view_presets_sel_fmt(const struct view_preset_sel *s)
{
    if (s->pix_fmt != AV_PIX_FMT_NONE && s->pix_fmt_family == s->family)
        return (enum AVPixelFormat)s->pix_fmt;

    return view_presets_pix_fmt(s);
}

const char *view_presets_fmt_why_not(int family, int preset,
                                     enum AVPixelFormat fmt)
{
    const struct view_preset *p = view_preset(family, preset);
    const AVPixFmtDescriptor *want, *got;

    if (!p || !p->fmt_why || p->fmt == AV_PIX_FMT_NONE ||
        fmt == AV_PIX_FMT_NONE)
        return NULL;

    want = av_pix_fmt_desc_get(p->fmt);
    got  = av_pix_fmt_desc_get(fmt);
    if (!want || !got)
        return NULL;

    /* The profile fixes the layout and nothing else: a 4:2:2 profile is as
     * happy at twelve bits as at ten, and as happy with an alpha plane as
     * without one, but not with 4:4:4. RGB counts as a layout of its own,
     * and so does having no chroma planes - gray subsamples by nothing,
     * which would otherwise read as 4:4:4. */
    if (want->log2_chroma_w == got->log2_chroma_w &&
        want->log2_chroma_h == got->log2_chroma_h &&
        !!(want->flags & AV_PIX_FMT_FLAG_RGB) ==
        !!(got->flags & AV_PIX_FMT_FLAG_RGB) &&
        (want->nb_components >= 3) == (got->nb_components >= 3))
        return NULL;

    return p->fmt_why;
}

int view_presets_parse_fmt(struct view_preset_sel *s, const char *arg)
{
    enum AVPixelFormat f;

    if (!arg || !arg[0]) {
        fprintf(stderr, "--pix-fmt: expected a pixel format, e.g. nv12\n");
        return CRAW_ERROR(EINVAL);
    }

    /* av_get_pix_fmt() takes both spellings: "gbrp16le" and the bare
     * "gbrp16", which it resolves to this machine's endianness. */
    f = av_get_pix_fmt(arg);
    if (f == AV_PIX_FMT_NONE) {
        fprintf(stderr, "--pix-fmt: `%s` is not a pixel format FFmpeg knows. "
                        "--list-presets prints the ones each family accepts.\n", arg);
        return CRAW_ERROR(EINVAL);
    }

    s->pix_fmt = f;
    s->pix_fmt_family = s->family;
    s->set = true;

    return 0;
}

/* --------------------------------------------------------- availability */

/* The three hardware engines a Vulkan encoder can be, each an extension of
 * the device, and whether this device has them once view_presets_device_exts()
 * has said. Everything else a Vulkan encoder can be is compute shaders. */
static const struct {
    enum AVCodecID id;
    const char *ext;
    const char *name;
} view_engines[] = {
    { AV_CODEC_ID_H264, "VK_KHR_video_encode_h264", "H.264" },
    { AV_CODEC_ID_HEVC, "VK_KHR_video_encode_h265", "HEVC" },
    { AV_CODEC_ID_AV1,  "VK_KHR_video_encode_av1",  "AV1" },
};

static struct {
    bool known;
    bool have[CRAW_ARRAY_ELEMS(view_engines)];
    /* The largest picture each engine encodes and the granularity it
     * takes pictures in, as the driver said at device open; 0 when it was
     * not asked or did not say. */
    int max_w[CRAW_ARRAY_ELEMS(view_engines)];
    int max_h[CRAW_ARRAY_ELEMS(view_engines)];
    int gran_w[CRAW_ARRAY_ELEMS(view_engines)];
    int gran_h[CRAW_ARRAY_ELEMS(view_engines)];
} view_dev;

void view_presets_engine_extent(enum AVCodecID id, int max_w, int max_h,
                                int gran_w, int gran_h)
{
    for (size_t i = 0; i < CRAW_ARRAY_ELEMS(view_engines); i++) {
        if (view_engines[i].id != id)
            continue;
        view_dev.max_w[i] = max_w;
        view_dev.max_h[i] = max_h;
        view_dev.gran_w[i] = CRAW_MAX(gran_w, 1);
        view_dev.gran_h[i] = CRAW_MAX(gran_h, 1);
    }
}

/* Up to the next multiple of `g`, which need not be a power of two. */
static int view_align(int v, int g)
{
    return (v + g - 1) / g * g;
}

const char *view_presets_extent_why_not(const AVCodec *c, int w, int h)
{
    static char buf[160];

    if (!c || !view_presets_codec_is_vulkan(c))
        return NULL;

    for (size_t i = 0; i < CRAW_ARRAY_ELEMS(view_engines); i++) {
        /* The engine takes pictures in whole blocks of its granularity -
         * 1080 lines are 1088 to an H.264 engine - and FFmpeg rounds up to
         * them before it checks the limit, so this does too. */
        if (view_engines[i].id != c->id || !view_dev.max_w[i] ||
            (view_align(w, view_dev.gran_w[i]) <= view_dev.max_w[i] &&
             view_align(h, view_dev.gran_h[i]) <= view_dev.max_h[i]))
            continue;
        snprintf(buf, sizeof(buf), "%s on this GPU encodes at most %dx%d, "
                 "and this is %dx%d; pick a smaller size", c->name,
                 view_dev.max_w[i], view_dev.max_h[i], w, h);
        return buf;
    }

    return NULL;
}

void view_presets_device_exts(const char *const *exts, int n)
{
    view_dev.known = true;

    for (size_t i = 0; i < CRAW_ARRAY_ELEMS(view_engines); i++) {
        view_dev.have[i] = false;
        for (int j = 0; j < n; j++)
            if (exts[j] && !strcmp(exts[j], view_engines[i].ext))
                view_dev.have[i] = true;
    }
}

const char *view_presets_engine_why_not(const AVCodec *c)
{
    static char buf[160];

    if (!c || !view_dev.known || !view_presets_codec_is_vulkan(c))
        return NULL;

    for (size_t i = 0; i < CRAW_ARRAY_ELEMS(view_engines); i++) {
        if (view_engines[i].id != c->id || view_dev.have[i])
            continue;
        snprintf(buf, sizeof(buf), "this GPU has no %s encode engine (no %s)",
                 view_engines[i].name, view_engines[i].ext);
        return buf;
    }

    return NULL;
}

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

    return view_presets_engine_why_not(avcodec_find_encoder_by_name(f->encoder));
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

/* Large context is the one thing the FFV1 presets differ in besides the
 * level, so a move to the level that carries float keeps it. */
static bool view_preset_large_context(const struct view_preset *p)
{
    return p && p->opts && strstr(p->opts, "context=1");
}

/* The preset of `family` that can write `colour`, the one with the same
 * context as `big` if there is one, or -1. */
static int view_presets_carrier(int family, int colour, bool big)
{
    const struct view_preset_family *f = view_family(family);
    int found = -1;

    if (!f)
        return -1;

    for (int i = 0; i < f->nb_presets; i++) {
        if (view_presets_preset_why_not(family, i) ||
            view_presets_colour_why_not(family, i, colour))
            continue;
        if (view_preset_large_context(&f->presets[i]) == big)
            return i;
        if (found < 0)
            found = i;
    }

    return found;
}

bool view_presets_take_colour(struct view_preset_sel *s, int colour)
{
    int family = s->family, preset = s->preset;

    if (colour < 0 || colour >= VIEW_NB_COLOURS)
        return false;

    /* The family in use first, so that FFv1 stays FFv1; then the table's
     * order, which puts FFv1 (Vulkan) ahead of the software one. */
    if (view_presets_colour_why_not(family, preset, colour)) {
        bool big = view_preset_large_context(view_preset(family, preset));

        preset = view_presets_carrier(family, colour, big);
        for (int i = 0; preset < 0 && i < view_nb_preset_families; i++)
            preset = view_presets_carrier(family = i, colour, big);
        if (preset < 0)
            return false;
    }

    if (family != s->family) {
        s->family = family;
        s->preset = preset;
        view_presets_family_defaults(s);
    } else {
        s->preset = preset;
    }

    s->colour = colour;
    /* The direct path writes the source's own size and nothing else. */
    if (view_colours[colour].scene)
        s->size = VIEW_SIZE_FULL;
    s->set = true;

    return true;
}

/* ------------------------------------------------------ the formats */

int view_presets_fmts(int family, const enum AVPixelFormat **out)
{
    const struct view_preset_family *f = view_family(family);

    if (!f) {
        *out = NULL;
        return 0;
    }

    *out = f->fmts;

    return f->nb_fmts;
}

/* ------------------------------------------------------- the names */

static bool view_name_eq(const char *a, const char *b)
{
    return a && b && !strcasecmp(a, b);
}

void view_presets_list(void)
{
    char buf[512], fmt[64];

    printf("video export presets. --preset \"Family/Preset\"\n\n");

    for (int i = 0; i < view_nb_preset_families; i++) {
        const struct view_preset_family *f = &view_preset_families[i];
        const char *why = view_presets_family_why_not(i);
        const enum AVPixelFormat *fmts = NULL;
        int nb = view_presets_fmts(i, &fmts);

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
            /* What the row starts on for this preset, and what it refuses:
             * the pinned format is a default everywhere and a requirement
             * where the profile is a chroma layout. */
            printf("      pixel format %s%s\n",
                   f->presets[j].fmt == AV_PIX_FMT_NONE
                   ? "chosen by the engine"
                   : view_presets_fmt_label(f->presets[j].fmt, fmt,
                                            sizeof(fmt)),
                   f->presets[j].fmt_why ? ", and only that layout" : "");
        }

        /* What the encoder takes: the fixed list at the top of this file. */
        printf("    accepts:");
        for (int j = 0; j < nb; j++)
            printf("%s%s", j ? ", " : " ",
                   view_presets_fmt_label(fmts[j], fmt, sizeof(fmt)));
        printf("\n\n");
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

    printf("\npixel formats. --pix-fmt NAME, by FFmpeg's names, out of the "
           "\"accepts\" line\nof the family. A preset's own format is what "
           "the row starts on.\n");
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
        /* A command line is one request rather than a sequence of
         * gestures, so a --pix-fmt typed before --preset still means this
         * family's picture; in the panel a family change voids the row. */
        s->pix_fmt_family = s->family;
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
