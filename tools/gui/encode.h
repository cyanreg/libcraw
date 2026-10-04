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
 * In the window the run is a thread of its own, so that nothing about the
 * window - the compositor's frame callbacks, a window on another workspace
 * that gets none - sets the pace of an export: the window draws the
 * progress and the picture the run last developed when there is something
 * new to draw, and the run goes as fast as the device and the encoder go.
 * The two share one lock (view_encode_lock()), which the run holds while
 * it walks the decoder, develops and renders, and the window while it
 * builds the panel and draws the picture; the encoder, the muxer and the
 * sound run outside it. A headless run is the same loop on the only thread
 * there is.
 *
 * Included from view.h, which owns the struct below; the engine itself is
 * opaque. */

#include <stdbool.h>
#include <stdint.h>

#include <SDL3/SDL_atomic.h>
#include <SDL3/SDL_mutex.h>
#include <SDL3/SDL_thread.h>

struct view_state;
struct view_encode;
struct view_preset_sel;
struct AVFrame;
struct CRAWFrameParams;

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

    /* The panel's file field, the default it was last given and the clip
     * that was on. The field follows the default - the clip's name, the
     * range and the family's container - except for what was typed into
     * it: a typed extension for good, a typed name on its own clip (ui.c,
     * view_field_follow()). */
    char field[1024];
    char field_def[1024];
    int field_clip;             /* the clip the default was last given on */
    bool field_active;          /* being typed into, as of the last frame */

    /* The frame that was on screen when the run started, and whether there
     * was one: the export walks the decoder, and when it is done the
     * picture goes back to where it was. */
    int64_t resume_index;
    bool resume;

    /* The panel's stop button, q and the window closing, all on the
     * window's thread, read by the run's. */
    SDL_AtomicInt stop;
    /* The run's side of the panel, written by the run under the lock;
     * `updates` counts the writes, so that the window draws when one has
     * happened and not otherwise. */
    char status[192];
    double progress;            /* 0 to 1 */
    uint64_t updates;
    uint64_t updates_seen;      /* the window's own */

    /* The run's thread, from the first view_encode_poll() after the press
     * to the one that finds it finished, and the lock it shares with the
     * window. Both are the window's to create and destroy, and NULL on a
     * headless run, where the lock is no lock at all. `finished` and
     * `result` are the thread's last words, under the lock. */
    SDL_Thread *thread;
    SDL_Mutex *lock;
    bool finished;
    int result;

    struct view_encode *e;
};

/* --list-encoders: every video encoder this build has, which of them take
 * Vulkan frames, and the pixel formats each one accepts. */
void view_encode_list(void);

/* Next to the clip, named after it and the range, with the extension the
 * selected family names: .mov for what a camera writes, .mkv for FFV1,
 * .mp4 for the delivery codecs - H.264, HEVC and AV1. `sel` may be NULL,
 * in which case the extension is guessed from `codec`'s name. */
void view_encode_default_path(const struct view_state *st, int64_t start,
                              int64_t end, const struct view_preset_sel *sel,
                              const char *codec, char *out, size_t size);

/* Gives a name that picks no muxer the extension the selection writes by
 * default, in place: "awb" is awb.mkv under FFV1 and awb.mov under ProRes.
 * A name that already picks one is left alone. `size` is the buffer's. */
void view_encode_complete_path(const struct view_preset_sel *sel,
                               const char *codec, char *path, size_t size);

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

/* Sets a run up. An empty `path` means the default one above. Nothing about
 * the file is written here - the size is not known until a frame has been
 * decoded - but everything the name decides is checked, so a refusal is a
 * sentence in `st->video.status` rather than a half written file. Nothing
 * runs yet either: view_encode_poll() or view_encode_run() starts it. */
int  view_encode_begin(struct view_state *st, const struct view_encode_opts *o);

/* The window's side of a run, once an iteration while `st->video.active`
 * is set. The first call starts the run on its own thread - the clip's
 * sound has to be stopped by then, since the run takes the decoder and the
 * sound with it - and the call that finds it finished joins the thread and
 * clears `active`. Returns 0 while it runs, 1 when the file is finished,
 * negative on an error; `*moved` says whether the run has changed what the
 * panel shows since the last call. */
int  view_encode_poll(struct view_state *st, bool *moved);

/* A headless run, from the first frame to the trailer on the calling
 * thread. 1 when the file is finished, negative on an error; `active` is
 * cleared either way. */
int  view_encode_run(struct view_state *st);

/* The lock a run on its thread shares with the window. Does nothing when
 * no run is on a thread. */
void view_encode_lock(struct view_state *st);
void view_encode_unlock(struct view_state *st);

/* A frame the seek to the first frame of the range develops ahead of it
 * for the temporal denoise's state (main.c), which a scene-referred run's
 * own pipeline has to take in as well: the pipeline is built for it if it
 * is not yet. NULL for `f` says the pre-roll was abandoned and the display
 * starts the average at the first frame, which this pipeline then does
 * too. Does nothing for any other run or once the output is open. */
int  view_encode_preroll(struct view_state *st, const struct AVFrame *f,
                         const struct CRAWFrameParams *fp);

/* Stops a run early, writing what has been encoded so far. */
void view_encode_cancel(struct view_state *st);

/* Releases the engine, stopping a run that is still on its thread and
 * waiting for it first. Safe on a run that never started. */
void view_encode_close(struct view_state *st);

#endif /* CINEVIEW_ENCODE_H */
