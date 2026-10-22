/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#ifndef CINEVIEW_PRESETS_H
#define CINEVIEW_PRESETS_H

/* The video export's preset table: which encoders the panel offers, what
 * each preset means in AVDictionary terms, which knobs a family shows, and
 * what colour the picture is written in.
 *
 * Everything the panel's cascade and `--preset Family/Preset` need is in
 * one place so that a click and a command line produce the same file. The
 * options here are the encoders' real ones, read out of
 * /home/lynne/projects/ffmpeg/libavcodec and `ffmpeg -h encoder=NAME`;
 * where a name is not obvious the table says where it came from.
 *
 * Nothing in here talks to libplacebo or nuklear: presets.c is plain
 * libavcodec, so ui.c, main.c and encode.c can all include it. */

#include <stdbool.h>

#include <libavcodec/avcodec.h>
#include <libavutil/dict.h>
#include <libavutil/pixfmt.h>

/* ------------------------------------------------------------- controls */

/* Which rows a family puts under the codec line. A family that declares
 * none shows nothing but its name. */
enum {
    VIEW_CTRL_ASYNC   = 1 << 0,     /* async_depth, the Vulkan encoders' */
    VIEW_CTRL_QP      = 1 << 1,     /* APV's own quantiser */
    VIEW_CTRL_CPU     = 1 << 2,     /* SVT-AV1's speed preset */
    VIEW_CTRL_RC      = 1 << 3,     /* the rate control switch and its field */
};

/* One state of a family's rate control switch. The three a family declares
 * are not the same three everywhere: the hardware AV1 encoder offers a
 * constant quantiser, SVT-AV1 and x264 offer the constant-quality mode
 * their adaptive quantisation is actually tuned for. */
struct view_rc_spec {
    const char *label;          /* "VBR", "CBR", "CQP", "CRF" */
    bool bitrate;               /* takes a bitrate rather than a quantiser */
    int qmin, qmax, qdef;       /* the quantiser's range, when it takes one */
};

/* ------------------------------------------------------------- presets */

struct view_preset {
    const char *name;
    /* KEY=VALUE,... in av_dict_parse_string() form, applied to the
     * encoder. Empty for a preset that only pins a pixel format. */
    const char *opts;
    /* The format the profile implies, or AV_PIX_FMT_NONE to let the engine
     * pick the deepest one the encoder lists. A ProRes profile is a chroma
     * format as much as a bitrate, and prores_ks takes the chroma format
     * from the *input* rather than from the profile, so a 4:4:4 frame with
     * -profile hq comes out labelled HQ and subsampled 4:4:4, which is not
     * a ProRes file anything else will read the way it was written. */
    enum AVPixelFormat fmt;
    /* Whether this preset can carry scene-referred linear light, which
     * needs float samples and therefore, for FFV1, version 4. */
    bool scene_ok;
    /* Why this preset cannot be used in this build, or NULL. Static
     * strings; the menu greys the row and shows this. */
    const char *why_not;
    /* Set when `fmt` above is a requirement of the standard rather than a
     * default: the profile this preset selects *is* a chroma layout, so
     * the pixel format row refuses every format laid out differently and
     * shows this string as the reason. The depth stays free - a 4:2:2
     * profile is as happy at twelve bits as at ten - and a preset that
     * leaves this NULL takes anything its encoder accepts. */
    const char *fmt_why;
};

struct view_preset_family {
    const char *name;           /* "FFv1 (Vulkan)" */
    const char *encoder;        /* "ffv1_vulkan" */
    const struct view_preset *presets;
    int nb_presets;
    int def;                    /* what a plain click on the family picks */

    unsigned ctrl;              /* VIEW_CTRL_* */

    /* The container the panel's default file name gets when the user has
     * not typed an extension. A typed one always wins, and a pair the
     * muxer will not carry is refused before the encode starts. */
    const char *container;

    /* Options every preset of the family carries, in the same form as a
     * preset's. */
    const char *always;

    /* The formats a scene-referred export can use instead of the preset's,
     * the default first; none when this family cannot carry linear light. */
    const enum AVPixelFormat *scene_fmts;
    int nb_scene_fmts;

    int async_min, async_max, async_def;
    int qp_min, qp_max, qp_def;
    int cpu_min, cpu_max, cpu_def;

    struct view_rc_spec rc[3];
    int nb_rc;
    int rc_def;                 /* the mode the switch starts in */
    int kbit_def;               /* the bitrate field's default, kbit/s */


    /* The pixel formats the encoder takes, fixed in presets.c: a Vulkan
     * encoder cannot be asked short of opening it with each candidate,
     * which is far too dear to do at run time. */
    const enum AVPixelFormat *fmts;
    int nb_fmts;
};

/* The ten families, in the order the menu's first column shows them. */
extern const struct view_preset_family view_preset_families[];
extern const int view_nb_preset_families;

/* --------------------------------------------------------- the colour */

/* What the export is written in.
 *
 * The display-referred entries are render targets: the export renderer
 * tone maps and gamut maps into them exactly as it does for the screen,
 * with libplacebo's own defaults, so the file is the picture as the window
 * shows it. PQ's target peak is the one parameter of that mapping the
 * selection holds, because it is part of what the target is.
 *
 * The scene-referred three never touch the renderer at all. The pipeline is
 * run a second time for the frame with CRAWFrameParams.out_space set to the
 * space asked for, into a half or a single float texture, and those floats
 * are read back, split into the encoder's planes and, for a Vulkan encoder,
 * uploaded - no pl_renderer, no pl_shader, no pl_dispatch. Nothing
 * display-referred can therefore reach them: no tone curve, no gamut
 * mapping, no film simulation, no grain, which is why those sections grey
 * out while one is selected. `needs_deep_rgb`, the family's `scene_fmts`
 * and a preset's `scene_ok` are what that path keys on: FFV1 writes
 * gbrpf16 and gbrpf32 at version 4 and refuses them below, so the only
 * families that can carry linear light are the FFv1 ones with an L4
 * preset. */
enum view_colour_id {
    VIEW_COLOUR_709 = 0,
    VIEW_COLOUR_2020_SDR,
    VIEW_COLOUR_2020_PQ,
    VIEW_COLOUR_2020_HLG,
    VIEW_COLOUR_ACES,
    VIEW_COLOUR_ACESCG,
    VIEW_COLOUR_ACESCCT,
    VIEW_COLOUR_XYZ,
    VIEW_NB_COLOURS,
};

struct view_colour {
    const char *name;           /* "BT.2020 PQ" */
    const char *key;            /* "bt2020-pq", for --colour */
    /* The menu is two deep: a family row and, where there is more than
     * one, the transfers under it. `group` is the first column's entry. */
    const char *group;          /* "BT.2020" */

    /* Scene referred: written by the direct path above, never rendered.
     * Every consequence of that - the greyed sections, the refusal of a
     * size, the tone mapping settings not applying - is this one bit. */
    bool scene;
    bool hdr;                   /* tone mapped to an HDR target */
    bool needs_deep_rgb;        /* linear light: float RGB codecs only */
    /* Scene referred too, but through the ACEScct curve over ACEScg rather
     * than as linear light, which is what makes it an integer encoding:
     * any family's own formats of ten bits or more, ProRes among them. */
    bool log;

    /* The stream tags. H.273 has no code points for ACES or ACEScg, so
     * those two are written as unspecified primaries with a linear
     * transfer plus a container tag naming the space. */
    enum AVColorPrimaries pri;
    enum AVColorTransferCharacteristic trc;
    enum AVColorSpace spc;      /* AVCOL_SPC_RGB for the RGB ones */
    const char *tag;            /* container metadata, or NULL */
};

extern const struct view_colour view_colours[VIEW_NB_COLOURS];

/* --------------------------------------------------------- a selection */

/* What the panel holds and what an export is started from. Lives in the
 * look state so that it persists from clip to clip like every other
 * setting. */
struct view_preset_sel {
    int family;                 /* index into view_preset_families[] */
    int preset;                 /* index into that family's presets */

    int async;                  /* the sliders, whether the family shows them */
    int qp;
    int cpu;

    int rc;                     /* index into the family's rc[] */
    int kbit;                   /* the bitrate field, kbit/s */
    int quant;                  /* the QP or CRF field */

    int colour;                 /* enum view_colour_id */
    /* The display-referred colour that was in use when a scene-referred
     * one was picked, which picking a codec that cannot write the scene
     * one goes back to. */
    int display_colour;
    /* PQ's target peak in nits: the one mapping parameter that stays,
     * because it is part of what the target *is*. How the picture is
     * mapped into it is libplacebo's defaults, the same as the window's. */
    int hdr_peak;

    int size;                   /* enum view_size_id */
    int size_w, size_h;         /* VIEW_SIZE_TYPED's own, in pixels */

    /* The pixel format row. AV_PIX_FMT_NONE is an untouched row, which
     * means the preset's own pinned format; anything else was chosen
     * explicitly and outranks the pin. Held as an int because that is what
     * AV_PIX_FMT_NONE being -1 makes a memset of this struct produce, and
     * view_presets_default() has to spell it out for the same reason.
     *
     * `pix_fmt_family` is the family the choice was made under. A format
     * means a different thing in another family - the encoders do not even
     * accept the same ones - so the choice is void once the family moves,
     * and the row falls back to the new preset's pin. Within a family it
     * survives a preset change, like every other export setting. */
    int pix_fmt;
    int pix_fmt_family;
    /* The same row's choice under a scene-referred colour, kept apart from
     * the one above so that each survives a trip through the other:
     * AV_PIX_FMT_NONE, or a format the family does not list, is the
     * family's default, 16 bit float. */
    int scene_pix_fmt;

    /* What the export does with the clip's sound, enum view_audio_mode,
     * and the one number each encoding mode takes - a bitrate in kbit/s
     * for opus and aac, a compression level for flac - kept per mode so
     * that switching the combo back and forth never lands a bitrate in a
     * level field. The two modes with no number leave their entry alone. */
    int audio_mode;
    int audio_q[6];

    /* AVCodecContext.thread_count for a software encoder: 0 is auto, which
     * is what FFmpeg's own `-threads 0` means, and the panel's range runs
     * to the machine's core count. The Vulkan encoders have no threads and
     * ignore it. */
    int threads;

    bool set;                   /* false until something chose: use the defaults */
};

/* -------------------------------------------------------------- the audio */

/* What the video export does with the clip's sound. `copy` is the source
 * track remuxed and cut to the sample; the four encoders take the decoded
 * samples, through libswresample only where the encoder wants another
 * sample format or rate, and cut them to the same range. `wav` is PCM in
 * whatever container the file is: 24 bit little endian, or 16 bit when the
 * source is 16 bit. */
enum view_audio_mode {
    VIEW_AUDIO_NONE = 0,
    VIEW_AUDIO_COPY,
    VIEW_AUDIO_OPUS,
    VIEW_AUDIO_AAC,
    VIEW_AUDIO_FLAC,
    VIEW_AUDIO_WAV,
    VIEW_NB_AUDIO_MODES,
};

struct view_audio_spec {
    const char *name;           /* "opus": --audio's word and the combo's */
    const char *encoder;        /* libavcodec's encoder, NULL where none */
    /* What the numeric field beside the combo is - "kbit/s" or "level" -
     * and NULL for a mode that takes no number. */
    const char *field;
    int min, max, def;
};

extern const struct view_audio_spec view_audio_modes[VIEW_NB_AUDIO_MODES];

/* `--audio none|copy|opus[:KBPS]|aac[:KBPS]|flac[:LEVEL]|wav`. */
int view_presets_parse_audio(struct view_preset_sel *s, const char *arg);

/* "opus 160 kbit/s", "flac level 5", "copy": the mode and its number in one
 * string, for the panel's status and the log. Returns `buf`. */
const char *view_presets_audio_label(const struct view_preset_sel *s,
                                     char *buf, size_t n);

/* ------------------------------------------------------------ the threads */

/* The top of the threads row's range: the machine's core count, which is
 * what FFmpeg resolves `-threads 0` to. */
int view_presets_max_threads(void);

/* Whether a codec takes Vulkan frames, which is the one thing that decides
 * whether the threads row means anything: a Vulkan encoder has no threads
 * to set. The family form asks about the family's own encoder. */
bool view_presets_codec_is_vulkan(const AVCodec *c);
bool view_presets_family_is_vulkan(int family);

/* --------------------------------------------------------------- the size */

/* What the exported picture is scaled to. The crop rectangle's own size is
 * the default and everything else is a fraction or a width of it; a typed
 * WxH is the escape hatch. A software encoder at 5.8K wants three and a half
 * gigabytes, which is the reason this row exists. */
enum view_size_id {
    VIEW_SIZE_FULL = 0,
    VIEW_SIZE_HALF,
    VIEW_SIZE_QUARTER,
    VIEW_SIZE_3840,
    VIEW_SIZE_1920,
    VIEW_SIZE_TYPED,
    VIEW_NB_SIZES,
};

extern const char *const view_size_names[VIEW_NB_SIZES];

/* The size a selection means for a crop of `cw` x `ch`, both even. A width
 * larger than the crop's own never upscales: the row shrinks, it does not
 * enlarge. */
void view_presets_size(const struct view_preset_sel *s, int cw, int ch,
                       int *w, int *h);

/* `--size full|1/2|1/4|3840|1920|WxH`. */
int view_presets_parse_size(struct view_preset_sel *s, const char *arg);

/* One line for the panel and the log: "1/2 (2880x1520)". */
const char *view_presets_size_label(const struct view_preset_sel *s,
                                    int cw, int ch, char *buf, size_t n);

/* Fills a selection with the table's defaults. */
void view_presets_default(struct view_preset_sel *s);

/* Moves the sliders and the switch to what `s->family` declares. Called
 * whenever the family changes so the rows start on the family's own
 * defaults rather than on the last family's numbers. */
void view_presets_family_defaults(struct view_preset_sel *s);

/* "FFv1 (Vulkan) / L4 (experimental)", for the panel row and the log. */
const char *view_presets_label(const struct view_preset_sel *s,
                               char *buf, size_t n);

/* `--preset Family/Preset`, case-insensitive, either half on its own
 * allowed ("x264" picks x264's default preset). Returns 0, or a negative
 * errno having printed what was wrong and what the names are. */
int view_presets_parse(struct view_preset_sel *s, const char *arg);

/* `--colour NAME`, by the keys in view_colours[]. */
int view_presets_parse_colour(struct view_preset_sel *s, const char *arg);

/* Every family and preset, one per line, for --list-presets. */
void view_presets_list(void);

/* The encoder a selection names, and the pixel format its preset pins,
 * which is what the format row starts on (AV_PIX_FMT_NONE to let the
 * engine choose). view_presets_sel_fmt() below is what an export uses:
 * the same thing, with the row's own choice on top. */
const char *view_presets_encoder(const struct view_preset_sel *s);
enum AVPixelFormat view_presets_pix_fmt(const struct view_preset_sel *s);
const char *view_presets_container(const struct view_preset_sel *s);

/* ------------------------------------------------- the pixel format */

/* "4:2:0 8-bit (nv12)", "RGB 16-bit (gbrp16le)": the chroma layout, the
 * depth and FFmpeg's own name, all three read off the AVPixFmtDescriptor
 * rather than written down anywhere. Returns `buf`. */
const char *view_presets_fmt_label(enum AVPixelFormat fmt, char *buf, size_t n);

/* The formats the family's encoder takes: the family's own fixed list. */
int  view_presets_fmts(int family, const enum AVPixelFormat **out);

/* The format a selection writes: the row's explicit choice while it still
 * belongs to this family, and the preset's own pin otherwise. */
enum AVPixelFormat view_presets_sel_fmt(const struct view_preset_sel *s);

/* The float formats a scene-referred export of the family can be written
 * in, the default first, and the one a selection writes: the row's choice
 * if the family lists it, the default otherwise, AV_PIX_FMT_NONE for a
 * family that cannot carry linear light. */
int  view_presets_scene_fmts(int family, const enum AVPixelFormat **out);
enum AVPixelFormat view_presets_sel_scene_fmt(const struct view_preset_sel *s);

/* The format an ACEScct export writes: the row's own, or, for a preset
 * that leaves it to the engine (ProRes auto), the family's first of ten
 * bits or more. An 8 bit row is not moved: the colour is greyed under it. */
enum AVPixelFormat view_presets_sel_log_fmt(const struct view_preset_sel *s);

/* The format the selection's colour writes, whichever kind it is: the
 * float choice for linear light, the above for ACEScct, the row's own for
 * the rest. */
enum AVPixelFormat view_presets_sel_out_fmt(const struct view_preset_sel *s);

/* Why this preset cannot be written in this format, or NULL. The encoder
 * would take it - a format it does not take is never offered at all - but
 * the profile the preset selects is a chroma layout in the standard, and
 * writing one layout under another profile's label makes a file no decoder
 * reads the way it was written. */
const char *view_presets_fmt_why_not(int family, int preset,
                                     enum AVPixelFormat fmt);

/* `--pix-fmt NAME`, by FFmpeg's own names ("nv12", "yuv422p10le", and the
 * unsuffixed "gbrp16" for the native-endian one). A float format is the
 * scene-referred choice as well, for `--colour aces` and the other two. */
int view_presets_parse_fmt(struct view_preset_sel *s, const char *arg);

/* The whole option string a selection means: the family's always-options,
 * the preset's, and the sliders and the switch the family declares, in
 * that order. Returns `buf`. */
const char *view_presets_opts(const struct view_preset_sel *s,
                              char *buf, size_t n);

/* The same as an AVDictionary, with `extra` (the --enc string) parsed on
 * top so that what was asked for wins over the table. Returns 0 or a
 * negative errno. */
int view_presets_dict(const struct view_preset_sel *s, const char *extra,
                      AVDictionary **out);

/* Why a family or one of its presets cannot be used here, or NULL when it
 * can. The family form answers for the GPU as well once
 * view_presets_device_exts() has been told what it has; until then it
 * answers only from this build. */
const char *view_presets_family_why_not(int family);
const char *view_presets_preset_why_not(int family, int preset);
/* And with a colour space in hand, since three of them need float RGB and
 * ACEScct ten bits or more. */
const char *view_presets_colour_why_not(int family, int preset, int colour);
/* Why `colour` cannot be written in `fmt`, or NULL: ACEScct in fewer than
 * ten bits. */
const char *view_presets_colour_fmt_why_not(int colour, enum AVPixelFormat fmt);
/* The same with the whole selection in hand, its pixel format included:
 * what greys a colour in the menu. */
const char *view_presets_sel_colour_why_not(const struct view_preset_sel *s,
                                            int colour);
/* Selects `colour`, and nothing else: one the codec, the preset or the
 * pixel format in use cannot write is refused, changing nothing, and the
 * menu greys it. Returns whether it was taken. */
bool view_presets_take_colour(struct view_preset_sel *s, int colour);
/* Selects a family's preset, and with it the family's own defaults for the
 * sliders. The other way round from the above: a scene-referred colour the
 * new preset cannot write goes back to the display-referred one it
 * replaced, so the colour row never holds a space the codec row cannot
 * write. */
void view_presets_take_preset(struct view_preset_sel *s, int family, int preset);

/* The device extensions the viewer's VkDevice was created with, told once
 * at device open. A Vulkan H.264, HEVC or AV1 encoder is a hardware engine
 * and the engine is an extension - VK_KHR_video_encode_h264, _h265, _av1 -
 * so whether this GPU has one is whether the extension is among them,
 * which is all a GPU has to say about it. The compute-shader encoders
 * (FFV1, ProRes, APV) run wherever the shaders do and need no check. */
void view_presets_device_exts(const char *const *exts, int n);

/* Why this encoder has no engine on this GPU, or NULL: for a Vulkan codec
 * whose engine is an extension the device was not created with. Answers
 * NULL for everything else, and for every codec until the device has been
 * told about. */
const char *view_presets_engine_why_not(const AVCodec *c);

/* The largest picture the device's engine for `id` encodes and the
 * granularity it takes pictures in, told once at device open from the
 * driver's video capabilities. */
void view_presets_engine_extent(enum AVCodecID id, int max_w, int max_h,
                                int gran_w, int gran_h);

/* Why this encoder cannot write a `w` by `h` picture on this GPU, or NULL:
 * a hardware engine's maximum, which RADV puts at 4096 for H.264 - under a
 * 5.8K clip at full size. The panel refuses with it before the press, and
 * the engine once the size is known, rather than the encoder failing to
 * open. */
const char *view_presets_extent_why_not(const AVCodec *c, int w, int h);

#endif /* CINEVIEW_PRESETS_H */
