/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#ifndef CINEVIEW_H
#define CINEVIEW_H

/* cineview: interactive viewer and exporter for camera raw clips.
 *
 * The whole program runs on a single VkDevice that libplacebo creates: the
 * decoder (FFmpeg's Vulkan hwaccel), the raw pipeline (libcraw) and the
 * display (libplacebo's renderer) all touch the same images, so a frame is
 * never copied to the host between the bitstream and the screen. */

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* libplacebo only needs vkGetInstanceProcAddr; everything this tool calls
 * itself comes from the loader we link against. The Vulkan header has to be
 * included before libplacebo's libav helper for AVVkFrame support. */
#include <vulkan/vulkan.h>

#include <SDL3/SDL.h>

#include <libplacebo/log.h>
#include <libplacebo/vulkan.h>
#include <libplacebo/renderer.h>
#include <libplacebo/swapchain.h>
#include <libplacebo/shaders/lut.h>

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>

#include "audio.h"

#include <craw/craw.h>
#include <craw/vulkan.h>
#include <craw/pipeline.h>
#include <craw/profile.h>
#include <craw/awb.h>
#include <craw/lens.h>

#define VIEW_MAX_PATH 1024

/* The video exporter's own state, which the panel and the command line both
 * write and the engine in encode.c reads. */
#include "encode.h"

/* The video export's preset table: the eight encoder families, what each
 * preset means, and the colour spaces an export can be written in. Both the
 * panel's cascade and --preset read it. */
#include "presets.h"

/* The panel sits on the right at a fixed width; the picture gets whatever
 * is left of the window. */
#define VIEW_PANEL_WIDTH  360.0f
#define VIEW_PANEL_MARGIN 12.0f

/* Hooks handed to one pl_render_params: the non-raw path's exposure on the
 * LINEAR stage, then the film simulation LUT, vibrance, soft light and film
 * grain, all four at PL_HOOK_PRE_OUTPUT and in that order. The export uses
 * the same chain as the display, so a frame of the movie and the still of
 * it are graded and grained identically. */
#define VIEW_MAX_HOOKS 5

/* Zoom steps by a quarter octave per notch, stops at fit and at eight times
 * one-to-one, and lands exactly on one-to-one when it passes it. */
#define VIEW_ZOOM_STEP 1.189207115f     /* 2^(1/4) */
#define VIEW_ZOOM_MAX  8.0f

/* Queue families enabled on top of the one this tool uses. The compute and
 * transfer ones exist so that FFmpeg sees more than one family and
 * allocates its frames with VK_SHARING_MODE_CONCURRENT: with a single
 * family it tracks an exclusive owner and then emits barriers that release
 * ownership to VK_QUEUE_FAMILY_IGNORED, which is not a legal transfer. The
 * video decode one is what lets FFmpeg's Vulkan HEVC hwaccel start at all,
 * and the encode one is what the video export's Vulkan encoders need. */
#define VIEW_EXTRA_QUEUES (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT | \
                           VK_QUEUE_VIDEO_DECODE_BIT_KHR | \
                           VK_QUEUE_VIDEO_ENCODE_BIT_KHR)

/* ------------------------------------------------------------- the look */

enum view_wb {
    VIEW_WB_AS_SHOT = 0,
    VIEW_WB_AUTO,
    VIEW_WB_MANUAL,
};

/* Everything the user can change that alters the picture. Both the GPU
 * pipeline and the display path read it, so it lives in one struct and the
 * panel writes it in place. */
struct view_look {
    enum view_wb wb;
    CRAWIlluminant manual;      /* for VIEW_WB_MANUAL */

    float ei;                   /* working exposure index, 0: as shot */
    float exposure;             /* bias in stops on top of ei */

    enum CRAWDebayer debayer;

    char lut_path[VIEW_MAX_PATH];
    /* The embedded film simulation, or -1 for none. Mutually exclusive
     * with lut_path: one LUT is active at a time, so picking a film
     * simulation clears the path and loading a path clears this. */
    int film_sim;
    enum pl_lut_type lut_type;
    /* Index into view_lut_spaces[]; anything but 0 makes the LUT a
     * conversion from that space to a BT.709 display. */
    int lut_space;

    /* The tone mapping curve the window is mapped with, as an index into
     * libplacebo's own list (view_tone_map_name() and friends). It is a
     * display setting and not an export setting: a display-referred export
     * is mapped exactly as the window is, so it follows this rather than
     * carrying a curve of its own, and the file keeps matching the frame on
     * screen whichever curve is picked. */
    int tone_map;

    /* Which encoder, which preset, which knobs, which colour space, what
     * happens to the sound: everything the export decides. It lives here
     * rather than beside the run in struct view_video because it is a
     * setting - it survives a finished export and it carries from clip to
     * clip, the way the grade does.
     *
     * How the picture is mapped into the target is not in here, because it
     * is not a decision the export gets to make: a display-referred export
     * is the picture as the window shows it, mapped into the chosen space
     * with the tone map above and libplacebo's defaults for the rest, the
     * way the window is mapped into whatever the compositor negotiated. */
    struct view_preset_sel video_preset;
};

/* Everything the raw pipeline pass depends on, kept for the picture on
 * screen: once the look differs from it while paused, the frame goes
 * through the pipeline again. */
struct view_pipeline_look {
    enum view_wb wb;
    CRAWIlluminant manual;
    CRAWIlluminant illum;       /* the auto track that was applied */
    bool have_illum;
    float ei, exposure, ambience;
    enum CRAWDebayer debayer;
    /* The strength the pass actually ran at, 0 when it did not: a toggle
     * while paused has to put the frame through again. */
    float denoise;
    /* Whether the lens warp ran, for the same reason. The record itself is
     * the frame's own and changes with it, so a replay of the *same* frame
     * only has to notice the switch. `lens_keep` is the same for the
     * fill / keep all row: the two produce different pictures from the same
     * record. */
    bool lens;
    bool lens_keep;
};

struct view_name_map {
    const char *name;
    int value;
};

/* The cascading film simulation menu: one column per level, each opening
 * the next beside it, the way the Windows XP start menu and RawTherapee's
 * film simulation picker do. Four levels is the deepest it goes:
 *
 *   column 0   Films | Looks
 *   column 1   the branch's categories
 *   column 2   Films: the category's film stocks   Looks: renderings (leaf)
 *   column 3   Films: the film's renderings (leaf)
 *
 * The geometry is kept in plain floats rather than nuklear rectangles so
 * that this header stays free of nuklear, which only ui.c includes. */
#define VIEW_MENU_COLS 4

/* Four panel rows hang a cascade off this one struct: the film
 * simulation's, the video export's codec preset, the export's colour space
 * and the export's pixel format. Only one is ever open, so the geometry,
 * the highlight and the hover timers are shared and `kind` says what the
 * columns mean. */
enum view_menu_kind {
    VIEW_MENU_FILM = 0,
    VIEW_MENU_PRESET,
    VIEW_MENU_COLOUR,
    VIEW_MENU_PIXFMT,
    VIEW_MENU_KINDS,
};

struct view_lut_menu {
    bool open;
    enum view_menu_kind kind;
    /* The highlighted entry of each column. Column 0 chooses the branch
     * (0 Films, 1 Looks) and the rest follow from it; -1 means nothing is
     * highlighted yet, which keeps every column to its right shut. */
    int sel[VIEW_MENU_COLS];
    int cols;                   /* how many are on screen */
    float scroll[VIEW_MENU_COLS];
    float x[VIEW_MENU_COLS], y[VIEW_MENU_COLS];
    float w[VIEW_MENU_COLS], h[VIEW_MENU_COLS];

    /* The panel row the menu hangs off, in window pixels. Written by that
     * row as it is laid out, so a row that is not laid out this frame -
     * its section collapsed under it - leaves the menu with nothing to
     * hang off and it closes. */
    float ax, ay, aw, ah;
    bool anchored;

    /* Hover hysteresis, so the cascade behaves like a real menu rather
     * than snapping to whatever the pointer crossed last. An entry has to
     * be rested on before its submenu opens, and once one is open the
     * entries the pointer sweeps past on its way there are ignored for a
     * moment: the "safe triangle" every cascading menu needs. */
    int hover_col, hover_entry;     /* what the pointer is on, -1 for none */
    int64_t hover_since;            /* when it landed there, microseconds */
    int guard_col;                  /* the column whose submenu is protected */
    int64_t guard_until;            /* until when, microseconds */
    float prev_x, prev_y;           /* the pointer last frame, for direction */
    int dir;                        /* +1 cascading right, -1 left */
};

/* The standard third-stop ISO list Apple's plug-in restricts the working
 * index to; see DESIGN.md 9b. */
extern const float view_iso_list[];
extern const int view_nb_iso;

extern const char *const view_debayer_names[];
extern const char *const view_lut_space_names[];
extern const int view_nb_lut_spaces;

/* ------------------------------------------------------------- decoding */

/* Demuxed video packets waiting for the decoder. A 5.8K ProRes RAW HQ
 * packet is five to eight megabytes and the medium may be a card reader, so
 * the queue is bounded by both count and bytes and holds roughly a second
 * and a half of playback. */
#define VIEW_PKTQ_MAX   48
#define VIEW_PKTQ_BYTES (512u << 20)

struct view_reader {
    SDL_Thread *thread;
    SDL_Mutex *lock;
    SDL_Condition *wake;             /* the reader waits here for room */
    SDL_Condition *ready;            /* the main loop waits here for a seek */

    AVPacket *q[VIEW_PKTQ_MAX];
    /* The lens record of each queued packet, read by the reader thread as
     * it queues it: the record lives in the frame's vendor metadata, which
     * the decoder does not expose, so it has to come off the packet, and
     * the reader thread is where the packet already is. Invalid for a
     * packet that carries none. */
    CRAWLensCorrection lens[VIEW_PKTQ_MAX];
    /* When the reader put each packet in the queue, so that --trace can say
     * whether a late picture was late because the medium was slow or
     * because everything after the medium was. */
    int64_t arrive[VIEW_PKTQ_MAX];
    int head, count;
    size_t bytes;

    /* Whether to look for records at all. Written once before the thread
     * starts and read-only afterwards. */
    bool scan;

    bool eof;                   /* the file is exhausted */
    bool quit;
    bool seek_pending;          /* the main loop asked for a seek */
    int64_t seek_ts;
    int64_t serial;             /* bumped on every seek, to drop stale packets */

    /* Queue depth averaged over the reader's own pushes: sampling it from
     * the render loop only ever catches the moment after a packet has been
     * taken, which is always near empty and says nothing. */
    int64_t depth_sum, depth_n;
};

struct view_decoder {
    AVFormatContext *fmt;
    AVCodecContext *dec;
    AVBufferRef *hw_device;
    AVPacket *pkt;
    int stream_index;
    bool draining;
    bool eof;

    /* Time the main loop spent waiting for the reader thread since the last
     * report. Reading itself happens off the render loop entirely. */
    double read_ms;

    struct view_reader reader;
    bool pkt_held;              /* d->pkt holds one the decoder refused */
    CRAWLensCorrection pkt_lens;  /* and its record */
    int64_t pkt_arrive;           /* and when the reader queued it */

    /* The queue time of the packets in flight, matched to the picture that
     * comes out by presentation timestamp, the way lens_ring below is -
     * except that this one is kept for every clip and never consumed, since
     * two pictures may legitimately want to read the same entry. */
    struct {
        int64_t pts, t;
    } arrive_ring[16];
    uint64_t arrive_pushed;
    int64_t frame_arrive;       /* of the picture view_decoder_next() returned */

    /* Records of the packets in flight, matched back to the picture that
     * comes out by presentation timestamp. ProRes RAW is all intra and
     * never reorders, so the match is always the oldest entry; going by
     * pts anyway costs nothing and keeps a reordering decoder from ever
     * putting one frame's lens state on another's picture. */
    struct {
        int64_t pts;
        CRAWLensCorrection lc;
        bool used;
    } lens_ring[8];
    uint64_t lens_pushed;

    /* What the clip turned out to carry, decided as pictures come out:
     * `lens_any` is what makes the panel's row live and the pass get
     * built. */
    bool lens_any;
    bool lens_none;
    /* Set when the codec context was rebuilt: its frame pool is new, so
     * every VkImageView cached for the old images is now dangling. */
    bool frames_reset;

    /* Bayer source that goes through libcraw, as opposed to a camera clip
     * that is already demosaiced and encoded. Decided by codec id: nothing
     * in the container metadata is trustworthy for this, since a ProRes RAW
     * clip's Panasonic XML describes the same capture gamma and gamut as a
     * V-Log recording would. */
    bool raw;
    const char *codec_name;
    /* Whether the frames actually came back on the GPU. Only known once one
     * has been decoded. */
    bool hwaccel;
    bool know_hwaccel;
    bool verbose;

    AVRational time_base;
    AVRational frame_rate;
    int64_t nb_frames;          /* best estimate, 0 if unknown */

    /* Container metadata cineraw also reads. */
    float meta_as_shot_ei;
    char model[256];

    /* What the Panasonic clip XML (`com.panasonic.Semi-Pro.metadata.xml`)
     * says the camera was recording, verbatim: "V-Log" and "V-Gamut" on
     * every clip this camera writes in that mode. Empty when the tag is
     * absent or does not carry the element. */
    char capture_gamma[32];
    char capture_gamut[32];
    /* The essence list's own name for what was recorded - "H265_420_LongGOP"
     * or "ProRes_RAW" - which nothing is decided by and which -v prints
     * because it is the clearest confirmation that `raw` below and the
     * capture gamma above are two different questions. */
    char essence_codec[32];

    /* The rest of the acquisition record, for the panel's Clip block and
     * -v: nothing below decides anything about the picture. All of it
     * comes through FFmpeg's demuxer - the Apple and RDD 18 format tags a
     * ProRes RAW clip carries, the stream's own tags, and the Panasonic
     * XML - and a field the file does not carry stays empty or zero, which
     * is how the block knows to leave its row out. The shutter angle, the
     * focal length and the lens name are in the maker note the demuxer
     * does not export from these files, so they are not here at all. */
    char cam_make[64];          /* "Panasonic" */
    char cam_model[64];         /* "DC-S1M2", the camera's own words */
    /* The codec as a person names it - "ProRes RAW", "ProRes 422 HQ",
     * "HEVC" - with the depth and the coded size beside it. The depth is
     * the XML's when it has one, else the stream's, and 0 when neither
     * says. */
    char format_name[32];
    int bit_depth;
    int coded_w, coded_h;
    /* The XML's own frame rate, "29.97p"; empty means the stream's. */
    char frame_rate_str[16];
    float aperture;             /* f-number, 0 when absent */
    /* Exif's exposure index, shown only when it differs from the ISO the
     * camera recorded at (meta_as_shot_ei above). */
    float exif_ei;
    int wb_kelvin;              /* the white balance as recorded, 0: none */
    char timecode[16];          /* at the clip's start, "01:36:02:00" */
    /* When the recording was made, as "2025-07-05 01:42:48 +09:00": the
     * XML's own date with the camera's zone when the clip carries it, else
     * the container's creation_time turned into this machine's zone. */
    char recorded[40];

    /* A developed clip whose XML says V-Log in V-Gamut, which is the one
     * case where the stream's own colour tags have to be overridden: the
     * camera writes a BT.709 VUI on a V-Log recording. Never true for a raw
     * clip - see `raw` above - so the same XML in a ProRes RAW file changes
     * nothing about how that clip is developed. */
    bool vlog;

    /* The track header's display matrix, reduced to the number of quarter
     * turns clockwise the picture is turned by before it is shown or
     * written: what mpv and ffmpeg's autorotate do with it. Zero for a clip
     * that carries no matrix, and for one whose matrix is not a multiple of
     * ninety degrees or carries a flip - a mirror is not a rotation, and
     * view_decoder_open() logs one rather than guessing at it. */
    pl_rotation rotation;
    /* The matrix's own angle, in the counter-clockwise degrees
     * av_display_rotation_get() reports, for the clip line and the log. */
    int rotation_deg;
};

/* --------------------------------------------------------------- the gpu */

/* Cached VkImageViews for the decoder's frame pool. FFmpeg recycles a
 * handful of AVVkFrames, so a tiny linear cache keyed on the VkImage saves
 * creating a view per frame without any lifetime tracking. */
#define VIEW_MAX_SRC_VIEWS 32

struct view_src_view {
    VkImage image;
    VkImageView view;
};

/* Frames in flight. Each slot owns an output texture, a command buffer and
 * a fence, so recording frame N + 1 never waits for frame N: the only fence
 * waited on is the one belonging to the slot about to be reused, which by
 * then is a whole frame old. libcraw's intermediates need no such thing -
 * successive recordings on one queue are ordered by the pipeline's own
 * barriers - and the histogram is kept in its own rotation of slots. */
#define VIEW_SLOTS 2

struct view_slot {
    pl_tex tex;
    VkImage image;
    VkImageView view;
    VkCommandBuffer cmd;
    VkFence fence;
    bool busy;
};

struct view_gpu {
    /* libplacebo owns the device. */
    pl_log log;
    pl_vk_inst inst;
    VkSurfaceKHR surface;
    pl_vulkan vk;
    pl_gpu gpu;
    pl_swapchain sw;
    pl_renderer rr;
    /* The window's picture is rendered into a texture of its own and only
     * when something about it changes; every swapchain frame then composes
     * that texture and the panel overlay, which is a copy and a blend
     * rather than a second pass over seventeen million source pixels. Its
     * own renderer, because alternating two unrelated renders on one
     * pl_renderer throws away the state it keeps between frames - the peak
     * detection track above all. */
    pl_renderer rr_compose;
    pl_tex pic_tex;
    int pic_w, pic_h;
    /* Bumped by every new developed or mapped picture; part of the
     * signature below, which is what decides that the texture is stale. */
    uint64_t pic_serial;
    uint64_t pic_sig;
    double pic_peak;            /* what the detector had settled on then */
    /* What the last swapchain frame described itself as, so that the
     * picture pass can run without one. */
    struct pl_color_repr pic_repr;
    struct pl_color_space pic_color;
    int pic_tw, pic_th;
    bool pic_have_target;
    bool pic_valid;
    /* Whether the last display actually ran the picture pass. The peak
     * detector converges over a few passes, so one redraw is a reason to
     * come back and look again. */
    bool pic_redrew;

    /* The single queue family everything runs on: libplacebo's async
     * transfer and compute are turned off so that graphics, compute and
     * transfer are all the same family and no image ever changes owner. */
    uint32_t qf;
    VkDeviceQueueCreateFlags queue_flags;
    VkQueue queue;
    VkCommandPool pool;
    VkQueryPool queries;
    double timestamp_period;    /* nanoseconds per timestamp tick */

    /* libcraw */
    CRAWContext *ctx;
    CRAWPipeline *pipe;
    uint32_t width, height;
    enum CRAWDebayer pipe_debayer;
    bool verbose;               /* -v, for the pipeline's own lines */

    /* Whether the denoise stage ran in each slot's submission, so that the
     * timestamp read back from a slot VIEW_SLOTS frames later is charged to
     * the right one of the panel's two averages. */
    bool slot_denoise[VIEW_SLOTS];
    bool last_gpu_denoise;
    /* The same for the lens warp. */
    bool slot_lens[VIEW_SLOTS];
    bool last_gpu_lens;

    /* The pipeline's output: one libplacebo texture per slot, held for the
     * duration of libcraw's submission into it. */
    struct view_slot slots[VIEW_SLOTS];
    int slot;                   /* the next one to record into */
    int display_slot;           /* the one holding the picture on screen */
    int64_t submitted;
    VkSemaphore out_sem;        /* timeline, chains hold -> libcraw -> release */
    uint64_t out_value;

    struct view_src_view src_views[VIEW_MAX_SRC_VIEWS];
    int nb_src_views;

    /* The non-raw path: libplacebo maps the decoded frame itself, whether
     * it came back as an AVVkFrame on this device or in system memory, and
     * the mapping stays alive until the next one replaces it so that a
     * paused window can keep redrawing it. */
    struct pl_frame mapped;
    pl_tex map_tex[4];
    bool have_mapped;

    /* Scene-linear exposure for the non-raw path, as an mpv user shader on
     * the LINEAR hook. Regenerated when the value changes. */
    const struct pl_hook *exposure_hook;
    float exposure_hook_gain;

    /* The nuklear overlay, composited by the renderer. One per slot, so the
     * backend's command buffer for a given index is never re-recorded while
     * it is still in flight and no full drain is needed between frames. The
     * previous set outlives a resize until nuklear has rebuilt the
     * framebuffers that still point at it. */
    pl_tex ui_tex[VIEW_SLOTS];
    VkImageView ui_views[VIEW_SLOTS];   /* nuklear keeps this array's address */
    pl_tex ui_tex_old[VIEW_SLOTS];
    VkImageView ui_views_old[VIEW_SLOTS];
    int ui_slot;
    VkFormat ui_format;
    int ui_w, ui_h;

    /* Offscreen export target, with a renderer of its own so that the peak
     * detection state an export walks through is not interleaved with the
     * one the window is driving. */
    pl_renderer rr_export;
    pl_tex export_tex;
    int export_w, export_h;

    /* The scene-referred export's direct path: a second pipeline writing
     * RGBA32F, a texture to write into and a command buffer and fence of
     * its own. Nothing here goes through pl_renderer, pl_shader or
     * pl_dispatch - the only libplacebo call on these pixels is the
     * pl_tex_download that reads them back - because linear light must
     * reach the encoder as the numbers libcraw produced and not as
     * something a display path had an opinion about.
     *
     * Built when such an export starts and freed when it ends: at 5.8K the
     * second pipeline's intermediates and this texture are most of a
     * gigabyte of device memory, which a viewer that is only playing has no
     * reason to be holding. */
    CRAWPipeline *direct_pipe;
    enum CRAWDebayer direct_debayer;
    uint32_t direct_w, direct_h;
    pl_tex direct_tex;
    VkImage direct_image;
    VkImageView direct_view;
    VkCommandBuffer direct_cmd;
    VkFence direct_fence;

    double last_gpu_ms;
    /* What libplacebo's peak detection measured for the last displayed
     * frame, in nits; 0 when it has nothing to report. */
    double measured_peak;

    /* Where --trace's current record lives, or NULL. gpu.c writes the
     * moments only it knows - when libcraw's submit returned, when its
     * fence was first seen signalled, when the swapchain took the frame -
     * straight into it rather than through another accumulator. */
    struct view_trace_rec *rec;
};

/* ---------------------------------------------------------- trace.c */

/* One record per iteration of the run loop that reached the swapchain. The
 * whole run is kept in memory and written out at exit: a file write inside
 * the loop would measure the writing rather than the playback, and what is
 * being asked of the trace is precisely where the milliseconds go. */
struct view_trace_rec {
    int64_t t_iter;         /* the iteration began */
    int64_t t_step;         /* view_step_frame returned */
    int64_t t_pkt;          /* the packet reached the reader's queue, 0: none */
    int64_t t_submit;       /* libcraw's vkQueueSubmit2 returned */
    int64_t t_fence;        /* that submission's fence was first seen signalled */
    int64_t t_disp;         /* view_gpu_display() began */
    int64_t t_start;        /* pl_swapchain_start_frame returned */
    int64_t t_image;        /* pl_render_image returned */
    int64_t t_hdr;          /* pl_renderer_get_hdr_metadata returned */
    int64_t t_render;       /* pl_swapchain_submit_frame returned */
    int64_t t_present;      /* the swap returned */
    int64_t index;          /* the picture this iteration showed, -1: none */
    double pts, clock;
    double craw_gpu_ms;     /* libcraw's own timestamps, filled in two frames late */
    double fence_ms;        /* blocked on a libcraw fence during this iteration */
    double ui_ms;
    bool stepped;           /* a new picture was decoded and developed */
    bool rendered;          /* the picture pass ran rather than being reused */
};

struct view_trace {
    struct view_trace_rec *recs;
    int nb, max;
    int64_t t0;
    /* The output's refresh, as SDL reports it. The summary counts how long
     * each picture was left on screen in refreshes, and that is the number
     * a person sees as judder; without it the same figures can only be
     * given in milliseconds. */
    double refresh_hz;
    /* The record being filled. Copied into recs[nb] at the end of the
     * iteration, so that a run longer than `max` still gets a summary. */
    struct view_trace_rec cur;
    bool on;
};

/* Allocates the ring; `path` is where view_trace_close() writes, and
 * `refresh_hz` is the output's own rate, 0 when it is not known. */
int  view_trace_open(struct view_state *st, const char *path, double refresh_hz);
/* Starts a record for this iteration. */
void view_trace_iter(struct view_state *st, int64_t now);
/* Ends it, at the moment the swap returned. */
void view_trace_push(struct view_state *st);
/* Writes the per-frame lines and the summary, and frees the ring. */
void view_trace_close(struct view_state *st);
/* Handed to pl_render_params.info_callback while tracing: totals the GPU
 * time of every pass libplacebo dispatches for the window, so that the
 * summary can say what the scaler, the peak detector, each hook and the
 * overlay blend each cost. */
void view_trace_pass(void *priv, const struct pl_render_info *info);

/* --------------------------------------------------------- program state */

struct view_cfg {
    /* The playlist: every non-option argument, in the order they were
     * given, and `input` the one clip that is open. `r` and `t` walk the
     * list; a headless export only ever looks at the first entry. */
    char *const *inputs;
    int nb_inputs;
    const char *input;

    const char *device;         /* physical device name substring */
    bool verbose;
    bool debug;                 /* validation layer */
    bool bench;                 /* decode as fast as the GPU allows */
    const char *trace;          /* --trace FILE: one line per present */

    const char *profile;        /* built-in per-CCT table set */
    const char *tables;         /* per-CCT tables from a text file */

    bool loop;                  /* start again at the end; --no-loop turns it off */
    int64_t loops;              /* stop and quit after N passes, 0: endless */
    const char *audio_dump;
    const char *audio_map;

    /* Headless export: run the loop without a window. */
    bool headless;
    int64_t export_frame;
    char export_path[VIEW_MAX_PATH];
    const char *export_codec;   /* --codec, NULL: the muxer's own */
    const char *export_enc;     /* --enc KEY=VALUE,... for the encoder */

    /* --export-video START END OUT, the same headless run for a range of
     * frames. The size is --size, which is an export setting and lives in
     * the selection with the rest of them. */
    bool export_video;
    int64_t video_start, video_end;
    char video_path[VIEW_MAX_PATH];

    /* --list-presets: after the arguments and with the device open. */
    bool list_presets;
};

struct view_awb {
    CRAWAwbSmoother *smoother;
    CRAWAwbParams params;
    CRAWAwbEstimate last;
    CRAWIlluminant illum;
    bool have_illum;
    /* Estimates still to throw away after a seek: the histogram read is
     * one frame behind, so the first one after a jump belongs to the
     * picture the seek left. The track itself is kept across seeks within
     * a clip; a camera original is one scene, and a track that restarted
     * from whatever frame came first jumped at every seek. */
    int skip;
    /* What the smoother is given as time: one frame period per estimate,
     * not the clip's pts. The filter smooths over what the eye sees, and a
     * seek's jump in pts is not elapsed time; fed the pts, the smoother
     * clamped the gap to a second and let one frame slew a second's worth. */
    double clock;
};

/* One frame to one file. The work is spread over the loop's iterations
 * only because the decoder may have to walk to the frame first; the write
 * itself happens in one go. */
struct view_export {
    bool active;
    int64_t frame;
    bool sought;                /* the seek towards `frame` has been issued */
    char path[VIEW_MAX_PATH];
    char codec[64];             /* empty: the muxer's own */
    char enc[256];              /* encoder options, from --enc */
    char status[128];

    /* The panel's file field and the default it was last given: the field
     * follows the frame on screen until it differs from that default, after
     * which it is the user's and is kept, across clips too. */
    char field[VIEW_MAX_PATH];
    char field_def[VIEW_MAX_PATH];
    bool field_active;          /* being typed into, as of the last frame */

    /* One RGBA16 frame, grown once and kept. */
    uint16_t *rgba;
    size_t rgba_values;
};

/* Where a frame's wall time goes. Accumulated per frame and averaged over
 * the reporting interval; printed with -v and --bench. */
struct view_timing {
    double read;        /* av_read_frame, i.e. the medium */
    double decode;      /* everything else in the decoder */
    double submit;      /* recording and submitting the raw pipeline */
    double fence;       /* waiting for that submission */
    double map;         /* handing a decoded frame to libplacebo */
    double ui;          /* building and drawing the panel */
    double finish;      /* pl_gpu_finish */
    double render;      /* pl_render_image and the swapchain submit */
    double present;     /* swap buffers */
    int64_t frames;
    int64_t starved;    /* iterations with nothing decoded yet */
};

/* Where the picture is shown and how much of it. `zoom` is screen pixels
 * per source pixel; `fit` means it is recomputed from the window every
 * frame. The centre is in source pixels. Nothing here reaches the pipeline
 * or the export: it only chooses the two rectangles handed to
 * pl_render_image. */
struct view_zoom {
    float zoom;
    bool fit;
    float cx, cy;
    bool dragging;
    float drag_x, drag_y;
};

/* The crop rectangle, in source pixels, so that it survives zoom, pan and
 * a resized window: it is a region of the frame, not of the screen. The
 * edges are even, which is what a subsampled encoder and a Bayer quad both
 * want. `on` is the mode a left drag draws in - outside it a drag pans as
 * it always has - and the rectangle outlives the mode being turned off.
 * Only the export reads it; the picture on screen is untouched, and the
 * shading over it is drawn on the nuklear overlay. */
struct view_crop {
    bool on;
    bool have;
    int x0, y0, x1, y1;
    bool dragging;
    float ax, ay;               /* the drag's anchor, in source pixels */

    /* A clip whose frames carry a crop starts with that rectangle, so that
     * what is on screen and what an export writes is the picture the camera
     * marked, while the sensels around it stay one `clear` away. It is a
     * default and nothing more: drawing a rectangle or clearing it drops
     * the flag, and the panel says "(clip)" only while it holds. */
    bool from_clip;
    bool clip_read;             /* the clip's own crop has been looked at */

    /* The frame the rectangle was last measured against. A rectangle is in
     * source pixels and means nothing on a frame of another size, so
     * switching to a clip that is not this size drops it; zero until the
     * first frame a rectangle has met, which is where one from --crop is
     * bound. */
    int frame_w, frame_h;
};

/* The spot white balance: a click on something that ought to be white, and
 * the illuminant that makes it so. The White balance section's "pick
 * white" button arms it, the next left click on the picture samples, and
 * the mode ends with that click; Escape or a second press ends it without
 * one. The pick itself - the read-back of the patch, the inversion of the
 * colour stage and the locus lookup - is pick.c, and what it finds lands
 * in look.manual with the mode set to manual, so that the sliders and the
 * paused replay do the rest. See VIEW_PICK_SIZE. */
struct view_pick {
    bool armed;                 /* the next left click on the picture samples */
    /* A pick waiting to be taken, in source pixels: a click that landed,
     * or --wb pick:X,Y until the first frame is held. It is taken on the
     * run loop rather than in the event handler because it puts the frame
     * through the pipeline twice and reads the device back, which wants
     * the frame settled and the export paths out of the way. */
    bool pending;
    int x, y;
    /* Why the last pick was refused, in the one word the section shows
     * beside the button - "clipped" or "too dark" - and empty when it took
     * or none has been made. Cleared when the next one is armed. */
    char refused[12];
    /* The crosshair shown while armed, created on arming and destroyed on
     * disarming; NULL headless. */
    SDL_Cursor *cursor;
};

/* The patch a pick averages: this many source pixels a side, centred on
 * the click and held inside the frame. 576 pixels is enough to average the
 * sensor's noise well below the locus's own resolution and small enough
 * to sit on a sheet of paper at 5.8K. */
#define VIEW_PICK_SIZE 24

/* How well the picture is keeping up with the master clock. */
struct view_sync {
    double diff;                /* clock minus frame pts at present time */
    double max_abs;             /* over every presented frame */
    double max_abs_settled;     /* ignoring the first few of each run */
    int64_t presented;
    int64_t dropped;
    int64_t run_frames;         /* since the last start or seek */
};

/* Soft light and vibrance, as the panel holds them: the two RawTherapee
 * tools, reimplemented in adjust.c. Like the grain, none of it reaches the
 * raw pipeline - all three hooks run at PL_HOOK_PRE_OUTPUT, after scaling
 * and colour management, which is where RawTherapee's own display-referred
 * stage sits too. The film simulation LUT joins them there so that vibrance
 * and soft light can be applied behind it, in RawTherapee's order. */
typedef struct craw_adjust craw_adjust;

struct view_adjust {
    /* Soft light strength as a percentage, 0 for off. */
    float softlight;

    /* Vibrance. Both amounts at zero is off, whatever the rest say. */
    float pastels;              /* -100..100 */
    float saturated;            /* -100..100 */
    float threshold;            /* 0..100, the chroma the two meet at */
    bool protect_skin;
    int protect_skin_flag;      /* nuklear's checkbox wants an int */
    bool avoid_shift;
    int avoid_shift_flag;

    /* One per renderer, the way the display and the export each own a
     * pl_renderer: the LUT hook holds the uploaded 3D texture between
     * frames and the two must not share it. Owned by gpu.c. */
    craw_adjust *state, *state_export;
};

/* Raw domain denoising, as the panel holds it. The model itself is the
 * camera's, read from the profile at the frame's as-shot exposure index, so
 * the panel holds only the switch, the strength and whatever the last frame
 * resolved, for the readout. See DESIGN.md 14.
 *
 * `strength` is kept across an off, so turning the section back on returns
 * to the number that was set rather than to the default. */
struct view_denoise {
    bool on;
    float strength;

    /* What the frame on screen resolved, for the readout. */
    CRAWNoiseModel model;
    float iso;
    bool have_model;

    /* libcraw's own pass time, smoothed separately for the frames that ran
     * the stage and the frames that did not, so that the panel can show
     * what it costs. Both only exist once the clip has been played both
     * ways, which is why the readout has to check. */
    double ms_on, ms_off;
};

/* Geometric lens correction, as the panel holds it. The correction itself
 * is the frame's own - it changes every frame while the lens zooms or
 * focuses - so the panel holds only the switch and what the picture on
 * screen carried, for the readout. See DESIGN.md 16.
 *
 * `on` is what the user asked for; the pass only runs when the frame also
 * has a record, which view_lens_active() is the single answer to. */
struct view_lens {
    bool on;

    /* The record of the picture on screen, copied out of the packet it came
     * from. Invalid when that frame carried none. */
    CRAWLensCorrection record;

    /* The corner correction of that record, per channel, in pixels, with
     * the crop divided out: what the lens actually does. Only meaningful
     * while `record.valid`. */
    float corner[3];

    /* Keep every source pixel instead of filling the frame, DESIGN.md 16.8,
     * and the scale that takes for the record on screen. `keep_sel` is the
     * panel row's index, 0 fill and 1 keep all, because a cycle row wants
     * an int. The scale is worked out once per frame beside `corner`, not
     * once per panel redraw. */
    bool keep;
    int keep_sel;
    float scale;

    /* libcraw's own pass time with and without the warp, smoothed
     * separately, the way the Denoise section measures its own. */
    double ms_on, ms_off;
};

/* Film grain, as the panel holds it. grain.c turns this into the parameters
 * libplacebo's grain shaders take and grain_ui.c does the turning; nothing
 * here reaches the raw pipeline, because the grain is applied at
 * PL_HOOK_PRE_OUTPUT, after scaling and colour management. `chroma` below
 * zero means "whatever the stock says". */
typedef struct craw_grain_hook craw_grain_hook;

struct view_grain {
    bool on;
    int gauge, stock, response;
    float amount, size, chroma;
    bool boil;
    int boil_flag;              /* nuklear's checkbox wants an int */

    /* What the last build worked out, for the panel's readout. */
    float blob_px;              /* what the gauge asked for */
    float blob_built;           /* what the model could build */
    float sigma;                /* amplitude it will deliver at mid grey */
    int model_used;
    int subsample;              /* 1, or the factor the grain is made at */

    /* One per renderer, the way the display and the export each own a
     * pl_renderer: the grain shaders keep state between frames and the two
     * must not share it. Owned by grain_ui.c. */
    craw_grain_hook *hook, *hook_export;
};

/* ------------------------------------------------------ the panel's sections */

/* The collapsible sections of the panel, in the order they are laid out.
 * Each one is a header - a power switch, then an expander - and a body that
 * only exists while the header is open. */
enum view_section {
    VIEW_SEC_WB = 0,
    VIEW_SEC_EXPOSURE,
    VIEW_SEC_DEBAYER,
    VIEW_SEC_DENOISE,
    VIEW_SEC_LENS,
    VIEW_SEC_LOOK,
    VIEW_SEC_ADJUST,
    VIEW_SEC_GRAIN,
    VIEW_SEC_CROP,
    VIEW_SEC_EXPORT,
    VIEW_NB_SECTIONS,
};

/* What each header carries, RawTherapee's arrangement: a switch at the left
 * of the title that bypasses the whole section while every one of its
 * parameters is kept, so switching it back on returns the picture it had.
 *
 * The open and closed state is here rather than in nuklear's own per-title
 * table because these headers are not nk_tree_push()es any more: a nuklear
 * tree header owns its whole row and cannot hold a switch beside its title.
 *
 * Four of the switches are the sections' own existing flags - denoise.on,
 * lens.on, grain.on and crop.on - which the header now shows instead of a
 * row inside the body. The three below are the new ones, and like every
 * other setting they carry from clip to clip. */
struct view_sections {
    bool open[VIEW_NB_SECTIONS];

    /* Which of the Export section's two tabs is up: the video export, or
     * the screenshot when false, which is where a session starts. Kept for
     * the session like the folds above; a running video export holds it on
     * Video, where the stop button and the progress are. */
    bool export_video;

    /* Off: the bias is zero and the working index is the as-shot one, with
     * the slider and the combo left where they were set. */
    bool exposure;
    /* Off: neither the film simulation nor a --lut cube is applied, and the
     * selection stays selected. */
    bool look;
    /* Off: no soft light and no vibrance, at whatever amounts they hold. */
    bool adjust;

    /* The three start off and switch themselves on when one of their
     * parameters is touched, the way RawTherapee's tools do, so that a lit
     * switch always means the section is doing something. --bypass pins
     * one off for the rest of the command line, whatever the other options
     * set; these remember that so their order does not matter. The panel
     * is not bound by it: touching a parameter later switches on. */
    bool exposure_forced, look_forced, adjust_forced;
};

struct view_state {
    struct view_cfg cfg;
    struct view_look look;
    struct view_gpu gpu;
    struct view_decoder dec;
    struct view_audio audio;
    struct view_awb awb;
    struct view_export exp;
    struct view_video video;
    struct view_sync sync;
    struct view_timing timing;
    struct view_trace trace;
    struct view_zoom view;
    struct view_crop crop;
    struct view_pick pick;
    struct view_grain grain;
    struct view_denoise denoise;
    struct view_lens lens;
    struct view_adjust adjustments;

    /* The panel is worked with the pointer alone: there is no keyboard
     * focus through its rows, and the keys are the transport's and the
     * view's. The one thing the panel takes from the keyboard is typing,
     * while a text field is being typed into. */
    bool ui_edit_active;        /* a text field is being typed into */
    struct view_lut_menu menu;
    struct view_sections sections;

    /* `q` during a video export: the run is cancelled and the program
     * quits once the file has its trailer, rather than leaving a file
     * nothing can play. */
    bool quit_pending;

    CRAWProfile base;           /* tables, no per-frame data */
    CRAWProfile prof;           /* base plus the current frame's side data */
    bool have_prof;

    /* Which of cfg.inputs is open. `r` and `t` step it, wrapping. */
    int clip;
    /* What the playlist has to say, on its own line under the clip line: a
     * file that would not open, a switch refused because an export is
     * running. Empty when there is nothing to report, and cleared by the
     * next switch that works. */
    char clip_status[160];
    /* Audio underruns of the clips that have been closed, so that the
     * figure printed at the end of a run is the session's and not just the
     * last clip's; the panel shows the open clip's own count. */
    uint64_t underruns, underrun_frames;

    void *win;                  /* SDL_Window, NULL when headless */
    bool sdl_init;
    int win_w, win_h;           /* window points, which mouse events use */

    /* The picture on screen stays referenced until the next decode, so a
     * grade change while paused can replay it without touching the file. */
    AVFrame *frame;
    int64_t index;              /* frame number of the picture on screen */
    double pts;
    bool have_frame;
    struct view_pipeline_look processed;    /* what it was processed with */
    bool playing;
    bool need_frame;            /* decode one more even while paused */
    int64_t seek_to;            /* >= 0: seek before the next decode */
    int64_t seek_target;        /* the frame a seek is still looking for */
    int seek_tries;

    struct pl_custom_lut *lut;
    char lut_error[128];

    /* Volume as the panel holds it; the callback ramps towards it.
     * nuklear's checkbox wants an int, so the flag is mirrored. */
    float volume;
    bool mute;
    int mute_flag;

    /* The wall clock that stands in for the audio clock on a silent clip. */
    double wall_pts;
    int64_t wall_t0;
    /* The master clock's smoother: see view_clock(). */
    double clk_t;
    int64_t clk_wall;
    bool clk_have;
    /* The first picture pays for the pipeline's shaders and libplacebo's;
     * the audio only starts once that is behind us, or the clock would run
     * away and the first second would be spent dropping frames. */
    bool warm;
    int64_t loops_done;
    /* How often the reader could not keep up, for -v. */
    int64_t starved;

    /* Timing, for -v and --bench. */
    double decode_ms;
    double fps;                 /* displayed frames per second */
    double proc_fps;            /* pictures pushed through the pipeline */
    double loop_period;         /* smoothed wall time between iterations */
    double step_cost;           /* smoothed decode plus pipeline time */

    /* The window is only drawn and presented when something on it would be
     * different: a new picture, a replayed one, any input at all, a picture
     * pass that has not settled yet, or the idle tick that keeps the
     * panel's own readouts alive. In between, the loop waits instead of
     * drawing the same frame again - which on a 60 Hz output used to be a
     * second full pass over seventeen million source pixels per picture,
     * and a second copy of the window back over the Thunderbolt link. */
    bool present_due;
    int64_t last_present;
};

/* ----------------------------------------------------------- gpu.c */

/* `sdl_window` may be NULL, which builds a headless device with no
 * surface, no swapchain and no UI. */
int  view_gpu_open(struct view_gpu *g, const struct view_cfg *cfg,
                   void *sdl_window);
void view_gpu_close(struct view_gpu *g);

/* Creates or resizes the pipeline and the output texture. */
int  view_gpu_ensure(struct view_gpu *g, uint32_t w, uint32_t h,
                     enum CRAWDebayer debayer);

/* Everything on the device that points at the closing clip's frames, while
 * its decoder is still alive to own them. */
void view_gpu_release_frames(struct view_gpu *g);

/* And what the arriving clip needs, once its kind is known: a developed one
 * cannot use the pipeline or its full-size output textures, so they go;
 * two raw clips of the same size keep both and view_gpu_ensure() then does
 * nothing at all. */
void view_gpu_new_clip(struct view_gpu *g, bool raw);

/* Records one decoded hardware frame through libcraw and submits it,
 * without waiting for the result. */
int  view_gpu_process(struct view_gpu *g, const AVFrame *frame,
                      const CRAWFrameParams *fp, struct view_timing *tm);

/* Waits for the submission before the last one, which is what the auto
 * white balance histogram of the previous frame needs. Normally returns at
 * once: that frame has had a whole frame period to retire. */
int  view_gpu_wait_prev(struct view_gpu *g, struct view_timing *tm);
/* Non-blocking: notes in --trace's record when the last submission's fence
 * was first seen signalled. */
void view_gpu_poll_fence(struct view_gpu *g);

/* The non-raw path: hands the decoded frame to libplacebo. Takes its own
 * reference, so the caller may unref immediately. `vlog` replaces the two
 * colour tags the camera writes wrongly; see view_gpu_map(). */
int  view_gpu_map(struct view_gpu *g, AVFrame *frame, AVStream *stream,
                  bool vlog, struct view_timing *tm);
void view_gpu_unmap(struct view_gpu *g);

/* Forgets the cached views of the decoder's frame pool, which a rebuilt
 * decoder replaces wholesale. */
void view_gpu_drop_src_views(struct view_gpu *g);

/* Points `lut` at the colour spaces the LUT input selector names, so that a
 * conversion LUT is applied in the space it was authored for. */
void view_lut_apply_space(struct pl_custom_lut *lut, int space);

/* libplacebo's tone mapping curves, by index into its own list, which is
 * what look.tone_map holds: nothing about them is typed into this program.
 * The default is whichever one pl_color_map_default_params names, and
 * view_tone_map_find() answers --tone-map with -1 for a name it does not
 * know. */
int  view_tone_map_count(void);
const char *view_tone_map_name(int i);
int  view_tone_map_default(void);
int  view_tone_map_find(const char *name);

/* What libplacebo makes of the current source's peak, in nits. */
float view_gpu_infer_ceiling(const struct view_gpu *g);

/* Builds the source pl_frame for the last processed picture. `ceiling` is
 * the highest value the pipeline can have produced, in nits. */
void view_gpu_image(const struct view_gpu *g, struct pl_frame *out,
                    float ceiling);

/* Forgets the peak detection track, for when continuity breaks. */
void view_gpu_reset_peak(struct view_gpu *g, bool export_renderer);

/* Renders to the swapchain, with `ui` composited on top when non-NULL. */
int  view_gpu_display(struct view_gpu *g, struct view_state *st);
/* Renders the window's picture into its own texture, off the present path,
 * so that presenting it later is a copy and a blend. Safe to call as often
 * as the loop likes: it does nothing when the texture already holds exactly
 * this picture. */
void view_gpu_prepare(struct view_gpu *g, struct view_state *st);

/* Renders to an offscreen sRGB texture and downloads it as 16 bit RGB. */
int  view_gpu_render_export(struct view_gpu *g, struct view_state *st,
                            uint16_t *rgba);

/* The same render into a target the caller owns, for the video export: the
 * export renderer, the export's own hook state and the crop rectangle as
 * the source, with everything about the destination - its planes, its
 * colour and its size - left to `target`. */
int  view_gpu_render_target(struct view_gpu *g, struct view_state *st,
                            struct pl_frame *target);

/* ------------------------------------------------ the direct export path */

/* The second pipeline and the RGBA32F texture a scene-referred export
 * writes through, for the size the clip is now. Idempotent; call again
 * after a clip switch. */
int  view_gpu_direct_open(struct view_gpu *g);
void view_gpu_direct_close(struct view_gpu *g);

/* Runs the picture on screen through that pipeline in `space` - no gamut
 * clamp, the reference gamut compression left to the space's own rule - and
 * reads the rectangle (x, y, w, h) back as w * h packed RGBA floats. The two
 * times are the device's own and the read-back's, in milliseconds. */
int  view_gpu_direct_frame(struct view_gpu *g, struct view_state *st,
                           enum CRAWColorSpace space, int x, int y, int w, int h,
                           float *rgba, double *gpu_ms, double *dl_ms);

/* Reads the rectangle (x, y, w, h) of the picture on screen back from the
 * display slot's own RGBA16F texture, as the texels libcraw wrote: w * h
 * packed RGBA halves, rows tightly packed. Waits for that slot's
 * submission first. Nothing about the picture changes; this is what the
 * spot white balance samples through. */
int  view_gpu_read_region(struct view_gpu *g, int x, int y, int w, int h,
                          uint16_t *rgba);

/* The overlay texture nuklear draws into. On a resize the previous texture
 * and view are kept until view_gpu_ui_drop_old(), which must be called once
 * nuklear has rebuilt its framebuffers from the new view. */
int  view_gpu_ui_resize(struct view_gpu *g, int w, int h);
void view_gpu_ui_drop_old(struct view_gpu *g);

/* The queue lock libplacebo, FFmpeg and we all go through. */
void view_gpu_lock(struct view_gpu *g);
void view_gpu_unlock(struct view_gpu *g);

/* ----------------------------------------------------------- decode.c */

int  view_decoder_open(struct view_decoder *d, const char *path,
                       struct view_gpu *g, bool verbose);
void view_decoder_close(struct view_decoder *d);
/* Once per program, after the last clip: the XML parser's globals. */
void view_decoder_cleanup(void);

/* 0: a frame, 1: end of stream, negative: an error. */
int  view_decoder_next(struct view_decoder *d, AVFrame *frame);

int  view_decoder_seek(struct view_decoder *d, int64_t frame_index);

/* Mean queue depth since the last call, as the reader itself saw it. */
double view_decoder_queued(struct view_decoder *d);

/* Fills `p` from the frame's AV_FRAME_DATA_RAW_COLOR_PARAMS side data. */
int  view_profile_from_frame(CRAWProfile *p, const AVFrame *frame);

/* The lens record of the picture with this presentation timestamp, read off
 * the packet it was decoded from. `out` is left invalid when that frame
 * carried none. */
void view_decoder_lens(struct view_decoder *d, int64_t pts,
                       CRAWLensCorrection *out);

/* The clip's acquisition record as the panel's Clip block shows it and -v
 * prints it: one label and one value per row, in the block's order, and
 * only the rows the file has something for. Fills up to `max` of `rows`
 * and returns how many. Formatted in one place so that the two never
 * disagree about what a clip carries. */
struct view_info_row {
    const char *label;
    /* Room for the longest row - a make and a model of 63 bytes each -
     * with the panel cutting what will not fit its column. */
    char value[160];
};
#define VIEW_INFO_MAX 12
int  view_decoder_info(const struct view_decoder *d, struct view_info_row *rows,
                       int max);

/* ----------------------------------------------------------- export.c */

/* Starts writing one frame to one file. An empty `path` means the default
 * one below; an empty `codec` means the one the file name implies. */
void view_export_begin(struct view_state *st, int64_t frame, const char *path,
                       const char *codec);
/* Renders and writes once the decoder has reached the frame. */
int  view_export_step(struct view_state *st);

/* Next to the clip, named after the clip and the frame, .png. */
void view_export_default_path(const struct view_state *st, int64_t frame,
                              char *out, size_t size);
/* Gives a name that picks no muxer the .png the default has, in place; a
 * name that already picks one is left alone. `size` is the buffer's. */
void view_export_complete_path(char *path, size_t size);

/* ----------------------------------------------------------- ui.c */

int  view_ui_open(struct view_state *st, void *sdl_window);
void view_ui_close(struct view_state *st);
/* True while nuklear owns the keyboard (a text field has focus). */
bool view_ui_wants_keys(const struct view_state *st);
/* True while a video export runs, which locks the whole panel and every
 * key, wheel and drag but the export's own stop button and `q`: the
 * engine owns the decoder and has copied the settings, and a slider that
 * moved would be changing a picture nobody is looking at and a file that
 * is half written. */
bool view_ui_locked(const struct view_state *st);
/* Feeds one SDL_Event; returns true when nuklear consumed it. */
bool view_ui_event(void *sdl_event);
void view_ui_input_begin(void);
void view_ui_input_end(void);
/* Lays out the panel, reading and writing `st`. */
void view_ui_build(struct view_state *st);
/* Draws the panel into the overlay texture, chaining libplacebo's
 * semaphores around nuklear's own submission. */
int  view_ui_render(struct view_state *st);
int  view_ui_resize(struct view_state *st, int w, int h);
/* Moves to the next overlay image, after the frame that used this one has
 * been handed to the swapchain. */
void view_ui_rotate(struct view_gpu *g);

/* The cascading menus - the film simulation's and the video export's
 * three. They own their own hit testing rather than going through
 * nuklear's widgets, because a cascade is not a thing nuklear has and the
 * columns have to keep taking the pointer while the panel below them is
 * what nuklear considers active. A cascade opens by a click on its row and
 * closes by a click elsewhere, a choice, or Escape. */
void view_ui_menu_open(struct view_state *st, enum view_menu_kind kind);
void view_ui_menu_close(struct view_state *st);
/* True when the open menu covers that point, given in window points the
 * way SDL reports them. The picture must not zoom or pan underneath it. */
bool view_ui_menu_covers(const struct view_state *st, float x, float y);

/* ----------------------------------------------------------- adjust.c */

/* Soft light off and vibrance neutral: every parameter at a value that
 * leaves the hooks unattached, so the defaults are an absence rather than
 * an identity pass. */
void view_adjust_defaults(struct view_adjust *a);

/* `--softlight PERCENT` and `--vibrance PASTELS[,SATURATED[,THRESHOLD]]`.
 * Return 0 or a negative errno, having printed what was wrong. */
int  view_adjust_parse_softlight(struct view_adjust *a, const char *arg);
int  view_adjust_parse_vibrance(struct view_adjust *a, const char *arg);

/* Releases both renderers' hook state. Call before the pl_gpu goes away. */
void view_adjust_close(struct view_adjust *a);

/* One line for the panel's readout. Returns `buf`. */
const char *view_adjust_summary(const struct view_adjust *a, char *buf,
                                size_t n);

/* ----------------------------------------------------------- main.c */

/* Total gain applied by the pipeline, i.e. what a pipeline value of 1.0 is
 * worth against the sensor clip. Used for the mastering metadata. */
float view_total_gain(const struct view_state *st);

/* The highest value the pipeline can produce, in nits: diffuse white times
 * the gain and the exposure bias, times the headroom the camera matrix adds
 * for a fully saturated clipped pixel. This is the ceiling libplacebo is
 * told about; peak detection measures the real per-frame peak below it. On
 * the non-raw path there is no such calculation and libplacebo's own
 * inference is reported instead. */
float view_peak_ceiling(const struct view_state *st);

/* Whether the selected export colour space is scene referred, which is the
 * one question the greying, the refusal of a size, the bypassed look hooks
 * and the direct write all turn on. */
bool view_export_is_scene(const struct view_state *st);

/* The colour space libcraw is asked for on the direct path, for the
 * selected export space. CRAW_CS_XYZ_D65 for anything that is not scene
 * referred, which nothing asks for. */
enum CRAWColorSpace view_export_craw_space(const struct view_state *st);

/* The exposure the picture is actually developed at: the panel's own two
 * numbers while the Exposure section is switched on, and a bias of zero at
 * the as-shot index while it is bypassed. Everything that has to agree on
 * what the frame was exposed with - the frame parameters, the ceiling, the
 * look the picture was processed with, the panel's readout - asks these. */
float view_exposure_bias(const struct view_state *st);
/* 0 means "as shot", the same as an unset --ei. */
float view_exposure_ei(const struct view_state *st);

/* Whether the film simulation or the .cube is applied, and whether soft
 * light and vibrance are: the Look and Adjustments switches, and with them
 * the scene-referred export that takes all three away. */
bool view_look_active(const struct view_state *st);
bool view_adjust_active(const struct view_state *st);

/* Linear gain the non-raw path's exposure hook applies: the bias in stops
 * times the working exposure index over the as-shot one, which is the same
 * model the raw path hands to libcraw. */
float view_video_exposure(const struct view_state *st);

/* Fills the per-frame parameters from the look and the current profile. */
void view_frame_params(const struct view_state *st, CRAWFrameParams *fp);

/* The camera's noise model for the frame on screen, looked up at its
 * as-shot exposure index, and false when there is none. Either pointer may
 * be NULL. */
bool view_denoise_model(const struct view_state *st, CRAWNoiseModel *nm, float *iso);

/* Whether the lens warp will run on the frame on screen: the switch is on
 * and that frame carried a record. Everything that has to agree on whether
 * the stage ran - the frame parameters, the look the picture was processed
 * with, the panel - asks this. */
bool view_lens_active(const struct view_state *st);

/* And whether it will run in the keep-everything mode of DESIGN.md 16.8,
 * which is the same question plus the panel's fill / keep all row. */
bool view_lens_keep_all(const struct view_state *st);

/* The strength the denoise stage will actually run at: zero unless the
 * switch is on, the strength is positive and a model exists. */
float view_denoise_strength(const struct view_state *st);

/* The pipeline-relevant part of the current look, and whether two of them
 * would give the same picture. */
struct view_pipeline_look view_pipeline_look(const struct view_state *st);
bool view_pipeline_look_equal(const struct view_pipeline_look *a,
                              const struct view_pipeline_look *b);

/* Runs the picture on screen through the pipeline again with the current
 * look. A no-op for an already developed clip, whose grade is applied at
 * display time, and while no picture is held. The second form adds
 * CRAW_FRAME_FLAG_* bits to the frame's parameters for this one pass,
 * which is how the spot white balance reads the colour stage's output
 * uncompressed; the look the picture is recorded as processed with does
 * not know about them, so the caller puts the frame through again. */
int  view_replay_frame(struct view_state *st);
int  view_replay_frame_flags(struct view_state *st, uint32_t flags);

void view_log(const struct view_state *st, const char *fmt, ...)
              craw_printf_format(2, 3);

/* Decodes the next picture and runs it through the pipeline.
 * 0: a frame, 1: end of stream, 2: the frame was too late to be worth
 * processing and was dropped, so call again, 3: the reader has not caught
 * up and there is nothing to decode yet; negative: an error. */
int  view_step_frame(struct view_state *st);

/* The master clock, in seconds on the clip's timeline: the audio clock
 * when there is audio, wall time since the last start or seek otherwise. */
double view_clock(struct view_state *st);

/* The .cube LUT named by look.lut_path. */
int  view_lut_load(struct view_state *st);
void view_lut_clear(struct view_state *st);

/* Selects one embedded film simulation by its index in the pack, which
 * materializes it as .cube text and hands that to the same parser a user
 * LUT goes through. Clears look.lut_path: one LUT is active at a time. */
int  view_lut_select_film(struct view_state *st, int index);
/* The name to print for the current selection, or "none". */
const char *view_lut_film_name(const struct view_state *st);

/* The directory the clip lives in, for the default export target. */
const char *view_clip_dir(const struct view_state *st);

/* The part of the window the picture gets: everything left of the panel. */
void view_picture_area(const struct view_state *st, pl_rect2df *out);

/* ------------------------------------------------------- the rotation */

/* The picture as it is looked at and written: the source turned by the
 * clip's display matrix. A 5760x3040 clip with a quarter turn is 3040 wide
 * and 5760 tall here, and every geometry below - fit, zoom, pan, the crop
 * rectangle as it is drawn, the export's size - is in these pixels.
 * Everything in source space stays in source space: libcraw's pipeline, the
 * lens correction, the histogram and the denoise never see the rotation. */
void view_disp_size(const struct view_state *st, int *w, int *h);

/* A rectangle of the picture as it is shown, in the source pixels it came
 * from, and back again. Both are exact: a quarter turn maps an even
 * coordinate to an even one whenever the frame's own dimensions are even,
 * which every sensor readout's are. */
void view_rot_to_src(const struct view_state *st, const pl_rect2df *in,
                     pl_rect2df *out);
void view_rot_to_disp(const struct view_state *st, const pl_rect2df *in,
                      pl_rect2df *out);

/* Screen pixels per source pixel at the current setting, resolving "fit". */
float view_zoom_scale(const struct view_state *st);

/* The rectangles the display path is built from: `src` in the rotated
 * picture's pixels, `dst` in window pixels. The renderer is handed `src`
 * through view_rot_to_src() and turns the picture itself. */
void view_zoom_rects(const struct view_state *st, pl_rect2df *src, pl_rect2df *dst);

/* Zoom by `notches` quarter-octaves about a window position, or about the
 * centre of the picture area when `at_cursor` is false. */
void view_zoom_by(struct view_state *st, float notches, int sx, int sy,
                  bool at_cursor);
void view_zoom_reset(struct view_state *st, bool one_to_one);
/* Pan by a fraction of the visible area, or by a screen-pixel delta. */
void view_pan_fraction(struct view_state *st, float fx, float fy);
void view_pan_pixels(struct view_state *st, int dx, int dy);

/* Window pixels to the rotated picture's pixels, through the same two
 * rectangles the display path is built from, so that whatever is placed by
 * the pointer lands on the pixels it is drawn over at any zoom and pan. */
void view_picture_pos(const struct view_state *st, int px, int py,
                      float *sx, float *sy);

/* The crop, drawn by a left drag in crop mode. The positions are in window
 * pixels, the way view_zoom_rects() gives them. */
void view_crop_begin(struct view_state *st, int px, int py);
void view_crop_drag(struct view_state *st, int px, int py);
void view_crop_end(struct view_state *st);
void view_crop_clear(struct view_state *st);

/* The region to export, in source pixels: the rectangle when there is one,
 * the whole frame when there is not, which is what the return value says.
 * Clamped to the frame, which is where a rectangle from the command line
 * meets the clip's real size for the first time. */
bool view_crop_rect(const struct view_state *st, int *x, int *y, int *w, int *h);

/* The same rectangle on the rotated picture, which is what the user drew
 * and what the export writes: the size every readout reports and the size
 * the export's target is made. */
bool view_crop_disp_rect(const struct view_state *st, int *x, int *y,
                         int *w, int *h);

/* Where that rectangle lands on screen, in window pixels, through the same
 * transform the renderer is handed. False when there is nothing to draw. */
bool view_crop_screen(const struct view_state *st, pl_rect2df *out);

/* ------------------------------------------------------------- pick.c */

/* Arms the spot white balance, or disarms it: the crosshair goes up or
 * comes down with it. Arming clears the last refusal's word. */
void view_pick_arm(struct view_state *st, bool on);

/* A left click on the picture while armed, in window pixels: maps it to
 * the source pixel under the pointer, queues the pick there, pauses the
 * transport - a pick is a decision about one frame - and disarms. */
void view_pick_click(struct view_state *st, int px, int py);

/* Queues a pick at a source pixel, which is what --wb pick:X,Y does. */
void view_pick_at(struct view_state *st, int x, int y);

/* Takes the queued pick on the frame that is held: puts it through the
 * pipeline uncompressed, reads the patch back, inverts the colour stage to
 * the raw ratios and looks the illuminant up on the profile's locus, then
 * sets the manual white balance to it and puts the frame through again
 * with the result. A refused pick - too much of the patch clipped or too
 * dark - leaves the look alone and says why in pick.refused. Returns a
 * negative error only when the device or the decoder failed; a refusal
 * is 0. */
int  view_pick_run(struct view_state *st);

#endif /* CINEVIEW_H */
