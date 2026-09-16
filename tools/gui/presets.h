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

    /* The format a scene-referred export uses instead of the preset's, or
     * AV_PIX_FMT_NONE when this family cannot carry linear light. */
    enum AVPixelFormat scene_fmt;

    int async_min, async_max, async_def;
    int qp_min, qp_max, qp_def;
    int cpu_min, cpu_max, cpu_def;

    struct view_rc_spec rc[3];
    int nb_rc;
    int rc_def;                 /* the mode the switch starts in */
    int kbit_def;               /* the bitrate field's default, kbit/s */

    /* True when the family may be missing an engine on this particular
     * device rather than from this build: the answer needs the encoder to
     * be opened once, which view_presets_probe_step() does. */
    bool probe;
};

/* The eight families, in the order the menu's first column shows them. */
extern const struct view_preset_family view_preset_families[];
extern const int view_nb_preset_families;

/* --------------------------------------------------------- the colour */

/* What the export is written in.
 *
 * The display-referred entries are render targets: the export renderer
 * tone maps and gamut maps into them exactly as it does for the screen.
 *
 * The scene-referred three are a selection the panel and the command line
 * accept and remember, but not yet an export: linear light has to reach
 * the encoder from libcraw's own output rather than through the display
 * pipeline, and that path is not in this build. Asking for one is refused
 * with a status line. The bookkeeping around them - `needs_deep_rgb`, the
 * family's `scene_fmt` and a preset's `scene_ok` - is what that path will
 * key on, and it is already measured: FFV1 writes gbrpf32 at version 4 and
 * refuses it below, so the only family that can carry linear light here is
 * FFv1 (Vulkan) with an L4 preset. */
enum view_colour_id {
    VIEW_COLOUR_709 = 0,
    VIEW_COLOUR_2020_SDR,
    VIEW_COLOUR_2020_PQ,
    VIEW_COLOUR_2020_HLG,
    VIEW_COLOUR_ACES,
    VIEW_COLOUR_ACESCG,
    VIEW_COLOUR_XYZ,
    VIEW_NB_COLOURS,
};

struct view_colour {
    const char *name;           /* "BT.2020 PQ" */
    const char *key;            /* "bt2020-pq", for --colour */
    /* The menu is two deep: a family row and, where there is more than
     * one, the transfers under it. `group` is the first column's entry. */
    const char *group;          /* "BT.2020" */

    bool scene;                 /* scene referred: not a render target */
    bool hdr;                   /* tone mapped to an HDR target */
    bool needs_deep_rgb;        /* linear light: float RGB codecs only */

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
    int hdr_peak;               /* PQ's target peak in nits */

    bool set;                   /* false until something chose: use the defaults */
};

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

/* The encoder a selection names, and the pixel format its preset pins
 * (AV_PIX_FMT_NONE to let the engine choose). */
const char *view_presets_encoder(const struct view_preset_sel *s);
enum AVPixelFormat view_presets_pix_fmt(const struct view_preset_sel *s);
const char *view_presets_container(const struct view_preset_sel *s);

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
 * can. The family form needs the probe below to have run for the families
 * that declare it; until it has, it answers only from this build. */
const char *view_presets_family_why_not(int family);
const char *view_presets_preset_why_not(int family, int preset);
/* And with a colour space in hand, since three of them need 16 bit RGB. */
const char *view_presets_colour_why_not(int family, int preset, int colour);

/* Opens at most one unprobed family's encoder, so that a device without an
 * AV1 engine greys that family rather than failing at export time. Cheap
 * to call every frame; returns true while there is still something left to
 * probe. `hw_device` is the viewer's AVHWDeviceContext, or NULL. */
bool view_presets_probe_step(struct AVBufferRef *hw_device, bool verbose);

#endif /* CINEVIEW_PRESETS_H */
