/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#ifndef CINEVIEW_ENCODE_H
#define CINEVIEW_ENCODE_H

/* The video exporter.
 *
 * A range of frames, the look exactly as the still export renders it, and
 * one file. The name picks the muxer through libavformat the same way a
 * still's does; `codec` picks the encoder, and the two paths behind it are
 * a Vulkan one and a software one:
 *
 *  - A Vulkan encoder (h264_vulkan, hevc_vulkan, ffv1_vulkan, apv_vulkan,
 *    prores_ks_vulkan, ...) is handed frames from an AVHWFramesContext
 *    built on the viewer's own AVVulkanDeviceContext, so the pictures never
 *    leave the device: libplacebo renders straight into the encoder's
 *    images, converting to YCbCr and subsampling on the way, and the frame
 *    is handed over through the AVVkFrame timeline semaphore rather than a
 *    copy or a wait.
 *
 *  - A software encoder gets the same RGBA16 read-back the still export
 *    uses, converted by libswscale to the deepest format the encoder takes.
 *    The read-back is double buffered so that the GPU renders the next
 *    frame while the scaler and the encoder chew on this one.
 *
 * The clip's sound goes with the picture as the selection says: remuxed
 * and cut to the sample, or decoded and encoded again with opus, aac, flac
 * or PCM, cut to the same range and interleaved with the video.
 *
 * Everything is stepped: one call is at most one frame, so the caller can
 * draw a progress bar, pump its event loop, and stop.
 *
 * Included from view.h, which owns the struct below; the engine itself is
 * opaque. */

#include <stdbool.h>
#include <stdint.h>

struct view_state;
struct view_encode;
struct view_preset_sel;

/* What one run is asked for. The size, the sound and the threads are the
 * selection's, which are export settings like the rest of them. */
struct view_encode_opts {
    int64_t start, end;         /* inclusive frame range */
    const char *path;
    /* The preset selection: the encoder, its options, the pixel format the
     * preset pins and the colour space the picture is written in. `codec`
     * overrides the encoder it names, for a --codec that is not one of the
     * families, and `enc` overrides individual options. */
    const struct view_preset_sel *sel;
    const char *codec;          /* encoder name, NULL: the preset's */
    const char *enc;            /* KEY=VALUE,... on top of the preset's */
};

/* The panel's and the command line's side of a run: what was asked for,
 * how far it got, and the engine while it is alive. */
struct view_video {
    bool active;
    int64_t start, end;
    /* nuklear's property fields want ints, and the panel keeps them across
     * a run so that the next export starts where the last one did. */
    int start_ui, end_ui;
    bool range_set;             /* the fields have been touched or set */

    char path[1024];
    char codec[64];
    char enc[256];
    int out_w, out_h;

    /* The panel's file field and the default it was last given. The field
     * follows the default - the clip's name, the range and the family's
     * container - until it differs from it, after which it is the user's
     * and is left alone, across clips too. */
    char field[1024];
    char field_def[1024];

    /* The frame that was on screen when the run started, and whether there
     * was one: the export walks the decoder, and when it is done the
     * picture goes back to where it was. */
    int64_t resume_index;
    bool resume;

    bool stop;                  /* the panel's stop button */
    char status[192];
    double progress;            /* 0 to 1 */

    struct view_encode *e;
};

/* --list-encoders: every video encoder this build has, which of them take
 * Vulkan frames, and the pixel formats each one accepts. */
void view_encode_list(void);

/* Next to the clip, named after it and the range, with the extension the
 * selected family names: .mov for what a camera writes, .mkv for FFV1 and
 * the archival ones, .mp4 for x264. `sel` may be NULL, in which case the
 * extension is guessed from `codec`'s name. */
void view_encode_default_path(const struct view_state *st, int64_t start,
                              int64_t end, const struct view_preset_sel *sel,
                              const char *codec, char *out, size_t size);

/* Whether the open clip has a sound track the export could carry at all,
 * by the same rule that picks it. A clip without one is exported without
 * one, whatever the mode says; it is not a reason to refuse the picture. */
bool view_encode_has_audio(const struct view_state *st);

/* Why the sound cannot be written this way into the file `path` names, or
 * NULL when it can: the container the name picks will not carry the
 * codec the mode means, or the clip has no sound to carry. Asked of the
 * muxer by writing a header into memory once per container and mode, so
 * the answer is this FFmpeg's own and not a table's. The panel greys the
 * mode with it; view_encode_begin() refuses with it. */
const char *view_encode_audio_why_not(const struct view_state *st,
                                      const char *path, int mode);

/* Why the file `path` names cannot hold the selection's video, or NULL:
 * no muxer for the name, or a muxer that does not carry the encoder.
 * `codec` is the --codec override or NULL. */
const char *view_encode_video_why_not(const struct view_state *st,
                                      const char *path,
                                      const struct view_preset_sel *sel,
                                      const char *codec);

/* Starts a run. An empty `path` means the default one above. Nothing about
 * the file is written here - the size is not known until a frame has been
 * decoded - but everything the name decides is checked, so a refusal is a
 * sentence in `st->video.status` rather than a half written file. */
int  view_encode_begin(struct view_state *st, const struct view_encode_opts *o);

/* One iteration: decodes, renders and encodes at most one frame, and
 * writes whatever audio belongs before it. Returns 0 while there is more
 * to do or nothing to do yet, 1 when the file is finished, negative on an
 * error. `st->video.active` is cleared either way. */
int  view_encode_step(struct view_state *st);

/* Stops a run early, writing what has been encoded so far. */
void view_encode_cancel(struct view_state *st);

/* Releases the engine. Safe on a run that never started. */
void view_encode_close(struct view_state *st);

#endif /* CINEVIEW_ENCODE_H */
