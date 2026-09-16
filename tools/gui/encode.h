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

/* What one run is asked for. Sizes of zero mean the crop's own size. */
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
    int width, height;          /* 0: the crop rectangle's size */
    bool audio;                 /* remux the clip's own audio */
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
    bool audio;

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

/* Starts a run. An empty `path` means the default one above. */
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
