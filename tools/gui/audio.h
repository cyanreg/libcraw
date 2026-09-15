/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#ifndef CRAW_VIEW_AUDIO_H
#define CRAW_VIEW_AUDIO_H

/* Audio output and the master clock.
 *
 * The clip's PCM track is decoded, converted to interleaved float and
 * pushed into one ring buffer; the callback SDL fires before every pull off
 * the output stream is the only consumer of that ring and the only writer
 * of the played-sample counter, which is what the clock is derived from.
 * Everything that can click - starting, stopping, seeking, looping, running
 * dry, moving the volume slider - is a gain ramp applied inside the
 * callback on real samples, never a dropped or repeated one.
 *
 * SDL3 has no callback device: an SDL_AudioStream bound to the default
 * playback device is the whole output path, the callback renders into it,
 * and the stream says how much of what it was handed has not been taken
 * yet. That is the part SDL2 could only guess at, and it is what the clock
 * subtracts on top of the device's own buffer.
 *
 * The audio has its own AVFormatContext on the same file. Camera QuickTime
 * files interleave in chunks of half a second or so, with every video
 * packet of a chunk ahead of its audio, and a ProRes RAW packet is five
 * megabytes: reading far enough ahead through one demuxer to keep a quarter
 * of a second of audio buffered would mean holding a hundred megabytes of
 * video packets that are not wanted yet. Two readers over the page cache
 * cost nothing by comparison and decouple the two clocks entirely. */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include <SDL3/SDL.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>

#include <craw/utils.h>

/* What the callback hands over. The source is 48 kHz stereo and so is this,
 * so libswresample only ever changes the sample format and there is no
 * resampler of ours in the steady state to drift; if the hardware wants
 * something else SDL converts, and the clock reads the real device format
 * rather than assuming this one. */
#define VIEW_AUDIO_RATE     48000
#define VIEW_AUDIO_CHANNELS 2

/* The device buffer to ask SDL for, in sample frames. SDL3 picks the size
 * itself unless it is told, and 1024 frames is about 21 ms, which is short
 * enough for the clock and long enough not to starve. */
#define VIEW_AUDIO_DEVICE_FRAMES 1024

/* How far ahead of the callback the decoder keeps the ring, and how much
 * the ring can hold at all. */
#define VIEW_AUDIO_TARGET_MS 250
#define VIEW_AUDIO_RING_MS   2000

/* Boundary fades and the volume slider's own ramp. */
#define VIEW_AUDIO_FADE_MS   5
/* An underrun is continued from the last sample with a shorter ramp, so a
 * dry ring is a soft stop rather than a step to zero. */
#define VIEW_AUDIO_DECAY_MS  2

enum view_audio_state {
    /* Playing at full gain. */
    VIEW_AUDIO_RUN = 0,
    /* Ramping up from silence; the first samples of a run. */
    VIEW_AUDIO_FADE_IN,
    /* Ramping down; the producer is waiting for this to finish. */
    VIEW_AUDIO_FADE_OUT,
    /* The ramp down is done and the callback is emitting silence. */
    VIEW_AUDIO_IDLE,
};

struct view_audio {
    /* A stream, a decoder, a converter and an open device. */
    bool available;
    /* The device is unpaused and the clock is running. */
    bool playing;

    /* The stream and the device it opened alongside itself. */
    SDL_AudioStream *stream;
    int device_frames;          /* device buffer, in the device's own frames */
    double device_latency;      /* the same, in seconds */

    /* One or two sources. Two is the four-mono-track layout, where the last
     * two tracks are the stereo pair and have to be interleaved here; one
     * covers a stereo stream, a four channel stream (channels three and
     * four) and a lone mono track sent to both ears. */
    struct view_audio_src {
        int stream_index;
        AVCodecContext *dec;
        struct SwrContext *swr;
        AVFrame *frame;

        int out_ch;             /* channels libswresample produces */

        /* Decoded mono samples waiting for the other source to catch up. */
        float *pend;
        size_t pend_cap, pend_n;
        double pend_pts;
    } src[2];
    int nb_src;
    int chan[2];                /* input channels feeding L and R, one source */
    char mapping[64];           /* what the panel and -v show */

    AVFormatContext *fmt;       /* this module's own reader */
    AVPacket *pkt;
    AVRational time_base;
    int64_t start_time;         /* stream start_time, or 0 */
    bool demux_eof;
    bool draining;

    /* Everything below is shared with the callback. */
    SDL_Mutex *lock;

    float *ring;                /* interleaved, VIEW_AUDIO_CHANNELS wide */
    size_t ring_frames;         /* capacity in sample frames */
    size_t read, write;         /* ring positions in sample frames */
    size_t fill;                /* readable sample frames */

    /* The clock: the presentation time of the first sample of this run,
     * plus every sample frame the callback has taken from the ring. */
    double base_pts;
    bool have_base_pts;
    uint64_t played;
    /* Sample frames handed to the stream, which is every frame the callback
     * has produced - the ones it took from the ring and the silence and
     * underrun fill alike. Minus what the stream still holds, this is what
     * the device has actually taken. */
    uint64_t pushed;
    /* When the callback last handed a buffer over, so that the clock can be
     * interpolated across the device's buffer instead of stepping once per
     * pull. */
    uint64_t handover_ns;
    /* Once the track has been played to its end the cursor above stops; a
     * clip whose video outlasts its audio still needs the clock to move,
     * so from that moment it runs on wall time from where the audio left
     * it. `ended_clock` is the value at that moment plus whatever elapsed
     * while playing, `ended_ns` the wall time it was last anchored at. */
    bool ended;
    double ended_clock;
    uint64_t ended_ns;

    /* What the callback renders into before handing it to the stream,
     * grown once to the largest pull SDL has asked for. */
    float *mix;
    size_t mix_frames;

    enum view_audio_state state;
    uint32_t fade_pos, fade_len;
    float decay, decay_step;
    float last_out[VIEW_AUDIO_CHANNELS];

    float volume, volume_target, volume_step;
    bool mute;

    /* Samples whose presentation time is before this are thrown away, which
     * is how a seek lands on the frame the user asked for. */
    double drop_before;
    bool dropping;

    /* Set when the last packet has been decoded: the callback then spends
     * the tail on the ramp down instead of running dry. */
    bool eof;
    bool in_underrun;

    /* Scratch for libswresample, grown once. */
    float *convert;
    size_t convert_frames;

    /* Diagnostics for -v. */
    uint64_t underrun_frames;
    uint64_t underruns;

    FILE *dump;
};

/* Opens the clip's first audio stream and an SDL output stream for it. Never
 * fatal: without a stream, without libswresample or without a device the
 * viewer runs silent and falls back to a wall clock. */
/* `map` is the --audio-map argument or NULL: one or two audio stream
 * indices, in audio stream order, for left and right. */
int  view_audio_open(struct view_audio *a, const char *path, const char *map,
                     bool verbose);
void view_audio_close(struct view_audio *a);

/* Reads and decodes until the ring is VIEW_AUDIO_TARGET_MS ahead of the
 * callback, or the clip ends. Call it every iteration. */
int  view_audio_pump(struct view_audio *a);

double view_audio_buffered(struct view_audio *a);

/* Seconds on the clip's timeline; only meaningful while `available`. */
double view_audio_clock(struct view_audio *a);

/* Ramps up and unpauses, or ramps down and pauses. Both block for as long
 * as the fade takes (5 ms), so that no boundary is ever a hard cut. */
int  view_audio_play(struct view_audio *a);
void view_audio_pause(struct view_audio *a);

/* Fades out, empties the ring and the decoder, seeks and arms the run that
 * starts at `pts`: samples before it are discarded on the way in. */
void view_audio_seek(struct view_audio *a, double pts);

/* True once that ramp has been rendered and the ring is empty. */
bool view_audio_drained(struct view_audio *a);

void view_audio_set_volume(struct view_audio *a, float volume, bool mute);

/* Writes exactly what the callback hands to the stream. */
int  view_audio_dump_open(struct view_audio *a, const char *path);

#endif /* CRAW_VIEW_AUDIO_H */
