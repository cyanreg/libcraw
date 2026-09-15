/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Audio output and the master clock. See audio.h for the model.
 *
 * Without libswresample there is nothing here but stubs and the viewer runs
 * silent; craw-view is a picture tool first. */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libavutil/samplefmt.h>

#if CRAW_VIEW_HAVE_AUDIO
#include <libswresample/swresample.h>
#endif

#include "audio.h"

#if !CRAW_VIEW_HAVE_AUDIO

int view_audio_open(struct view_audio *a, const char *path, const char *map,
                    bool verbose)
{
    (void)path;
    (void)map;

    *a = (struct view_audio) { 0 };
    if (verbose)
        fprintf(stderr, "built without libswresample: no audio\n");

    return 0;
}

void view_audio_close(struct view_audio *a) { (void)a; }
int view_audio_pump(struct view_audio *a) { (void)a; return 0; }
double view_audio_buffered(struct view_audio *a) { (void)a; return 0.0; }
double view_audio_clock(struct view_audio *a) { (void)a; return 0.0; }
int view_audio_play(struct view_audio *a) { (void)a; return 0; }
void view_audio_pause(struct view_audio *a) { (void)a; }
void view_audio_seek(struct view_audio *a, double pts) { (void)a; (void)pts; }
bool view_audio_drained(struct view_audio *a) { (void)a; return true; }

void view_audio_set_volume(struct view_audio *a, float volume, bool mute)
{
    (void)a;
    (void)volume;
    (void)mute;
}

int view_audio_dump_open(struct view_audio *a, const char *path)
{
    (void)a;
    fprintf(stderr, "--audio-dump %s: built without libswresample\n", path);
    return CRAW_ERROR(ENOTSUP);
}

#else /* CRAW_VIEW_HAVE_AUDIO */

#define VIEW_AUDIO_FRAME_BYTES (VIEW_AUDIO_CHANNELS * sizeof(float))

/* ------------------------------------------------------------- the ring */

/* Caller holds the lock. */
static void view_ring_reset(struct view_audio *a)
{
    a->read = a->write = a->fill = 0;
    a->played = 0;
    a->have_base_pts = false;
    a->base_pts = 0.0;
    a->decay = 0.0f;
    memset(a->last_out, 0, sizeof(a->last_out));
}

/* Caller holds the lock. Drops what does not fit, which cannot happen while
 * the producer honours view_audio_hungry(). */
static void view_ring_push(struct view_audio *a, const float *src, size_t frames)
{
    size_t room = a->ring_frames - a->fill;

    if (frames > room)
        frames = room;

    for (size_t i = 0; i < frames; i++) {
        memcpy(a->ring + a->write * VIEW_AUDIO_CHANNELS,
               src + i * VIEW_AUDIO_CHANNELS, VIEW_AUDIO_FRAME_BYTES);
        a->write = (a->write + 1) % a->ring_frames;
    }

    a->fill += frames;
}

/* ---------------------------------------------------------- the callback */

/* Caller holds the lock. The largest pull SDL has asked for so far, kept. */
static bool view_audio_mix_room(struct view_audio *a, size_t frames)
{
    float *p;

    if (a->mix_frames >= frames)
        return true;

    p = realloc(a->mix, frames * VIEW_AUDIO_FRAME_BYTES);
    if (!p)
        return false;

    a->mix = p;
    a->mix_frames = frames;

    return true;
}

/* The only consumer of the ring and the only writer of `played`. SDL calls
 * it before every pull off the stream, asking for roughly what the device
 * still needs, and every boundary is a gain ramp over real samples: nothing
 * is ever skipped or repeated to make one. */
static void SDLCALL view_audio_callback(void *opaque, SDL_AudioStream *stream,
                                        int additional, int total)
{
    struct view_audio *a = opaque;
    int frames = additional / (int)VIEW_AUDIO_FRAME_BYTES;
    float *out;

    (void)total;

    /* SDL is entitled to ask for nothing, and does whenever the stream
     * already holds enough. */
    if (frames <= 0)
        return;

    SDL_LockMutex(a->lock);

    if (!view_audio_mix_room(a, (size_t)frames)) {
        SDL_UnlockMutex(a->lock);
        return;
    }
    out = a->mix;

    for (int i = 0; i < frames; i++) {
        float *dst = out + (size_t)i * VIEW_AUDIO_CHANNELS;
        float gain;

        /* Faded out and waiting to be paused or flushed: emit silence
         * without touching the ring, so the samples that were queued up
         * survive for whatever comes next. */
        if (a->state == VIEW_AUDIO_IDLE) {
            memset(dst, 0, VIEW_AUDIO_FRAME_BYTES);
            continue;
        }

        switch (a->state) {
        case VIEW_AUDIO_FADE_IN:
            gain = (float)a->fade_pos / (float)a->fade_len;
            if (++a->fade_pos >= a->fade_len)
                a->state = VIEW_AUDIO_RUN;
            break;
        case VIEW_AUDIO_FADE_OUT:
            gain = 1.0f - (float)a->fade_pos / (float)a->fade_len;
            if (++a->fade_pos >= a->fade_len)
                a->state = VIEW_AUDIO_IDLE;
            break;
        default:
            gain = 1.0f;
            break;
        }

        /* The slider's own ramp, so that dragging it does not step. */
        if (a->volume != a->volume_target) {
            float d = a->volume_target - a->volume;
            a->volume += CRAW_CLAMP(d, -a->volume_step, a->volume_step);
        }
        gain *= a->volume;

        if (a->fill) {
            const float *src = a->ring + a->read * VIEW_AUDIO_CHANNELS;

            for (int c = 0; c < VIEW_AUDIO_CHANNELS; c++) {
                dst[c] = src[c] * gain;
                a->last_out[c] = dst[c];
            }

            a->read = (a->read + 1) % a->ring_frames;
            a->fill--;
            a->played++;
            a->decay = 1.0f;
            a->in_underrun = false;

            /* End of stream: spend the last samples on the ramp down rather
             * than letting the ring run dry. */
            if (a->eof && a->state != VIEW_AUDIO_FADE_OUT &&
                a->fill <= a->fade_len) {
                a->state = VIEW_AUDIO_FADE_OUT;
                a->fade_pos = 0;
            }
        } else if (a->eof) {
            /* The track is simply over. The ramp above has already been
             * spent on its last samples, so there is nothing to continue
             * from and nothing has gone wrong. */
            a->state = VIEW_AUDIO_IDLE;
            memset(dst, 0, VIEW_AUDIO_FRAME_BYTES);
        } else {
            /* Dry with more to come: the producer fell behind. Continue
             * from the last sample and decay to silence; a hard cut here is
             * exactly the click this is all about. */
            for (int c = 0; c < VIEW_AUDIO_CHANNELS; c++)
                dst[c] = a->last_out[c] * a->decay;

            a->decay = CRAW_MAX(a->decay - a->decay_step, 0.0f);
            if (!a->in_underrun) {
                a->in_underrun = true;
                a->underruns++;
            }
            a->underrun_frames++;
        }
    }

    if (a->dump)
        fwrite(out, sizeof(*out), (size_t)frames * VIEW_AUDIO_CHANNELS, a->dump);

    /* Handing the buffer over takes the stream's own lock, which SDL is
     * already holding around this callback, so it nests safely here. The
     * one order that would not is the reverse: nothing on the transport
     * side may call into the stream while holding this lock, which is why
     * view_audio_clock() reads the queue before taking it. */
    if (SDL_PutAudioStreamData(stream, out, frames * (int)VIEW_AUDIO_FRAME_BYTES))
        a->pushed += (uint64_t)frames;
    a->handover_ns = SDL_GetTicksNS();

    SDL_UnlockMutex(a->lock);
}

/* ------------------------------------------------------------ the device */

/* The session's default driver first, then the ones that exist on a Linux
 * desktop, in the order they are likely to work. */
static const char *const view_audio_drivers[] = { NULL, "pipewire", "pulseaudio", "alsa" };

static int view_audio_device_open(struct view_audio *a, bool verbose)
{
    /* The format the callback hands over, not the hardware's: SDL3 always
     * takes a stream in whatever format the app finds convenient and
     * converts on its way to the device. */
    SDL_AudioSpec want = {
        .format   = SDL_AUDIO_F32,
        .channels = VIEW_AUDIO_CHANNELS,
        .freq     = VIEW_AUDIO_RATE,
    };
    char frames[16];

    /* SDL3 sizes the device buffer itself unless it is told; this is the
     * only way left to ask for the period the clock is tuned around. */
    snprintf(frames, sizeof(frames), "%d", VIEW_AUDIO_DEVICE_FRAMES);
    SDL_SetHint(SDL_HINT_AUDIO_DEVICE_SAMPLE_FRAMES, frames);

    for (size_t i = 0; i < CRAW_ARRAY_ELEMS(view_audio_drivers); i++) {
        SDL_AudioSpec dev;
        int dev_frames = 0;

        if (view_audio_drivers[i]) {
            SDL_QuitSubSystem(SDL_INIT_AUDIO);
            SDL_SetHint(SDL_HINT_AUDIO_DRIVER, view_audio_drivers[i]);
        }

        if (!SDL_InitSubSystem(SDL_INIT_AUDIO))
            continue;

        /* One call opens the default playback device and binds a stream to
         * it, and the callback fires just before every pull. The device
         * comes up paused, which is what view_audio_play() undoes. */
        a->stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK,
                                              &want, view_audio_callback, a);
        if (a->stream) {
            /* The real period and rate, which is what the latency term of
             * the clock is made of; the device need not run at ours. */
            if (SDL_GetAudioDeviceFormat(SDL_GetAudioStreamDevice(a->stream),
                                         &dev, &dev_frames) && dev.freq > 0) {
                a->device_frames = dev_frames;
                a->device_latency = (double)dev_frames / dev.freq;
            } else {
                a->device_frames = VIEW_AUDIO_DEVICE_FRAMES;
                a->device_latency = (double)VIEW_AUDIO_DEVICE_FRAMES
                                  / VIEW_AUDIO_RATE;
            }

            if (verbose)
                fprintf(stderr, "audio: %s, device %d Hz, %d channels, %s, "
                                "%d frame buffer (%.1f ms of output latency)\n",
                        SDL_GetCurrentAudioDriver(), dev.freq, dev.channels,
                        SDL_GetAudioFormatName(dev.format), a->device_frames,
                        1000.0 * a->device_latency);
            return 0;
        }

        if (verbose)
            fprintf(stderr, "audio: %s cannot open a device: %s\n",
                    SDL_GetCurrentAudioDriver(), SDL_GetError());
    }

    return CRAW_ERROR(ENODEV);
}

/* ------------------------------------------------------------- lifetime */

/* Picks which audio streams feed the left and right ears.
 *
 * The camera can record four independent mono tracks, and in that layout the
 * last two are the stereo pair while the first two are microphone feeds; the
 * same recording may arrive as four separate mono streams or as one four
 * channel stream, so both are handled. Everything else is the ordinary
 * single stereo stream. */
static int view_audio_select(struct view_audio *a, const char *map, bool verbose)
{
    int audio[64], nb_audio = 0;
    int mono = 0, stereo = 0, quad = -1;
    int pick[2] = { -1, -1 };
    int chan[2] = { 0, 1 };

    for (unsigned i = 0; i < a->fmt->nb_streams && nb_audio < 64; i++) {
        AVCodecParameters *cp = a->fmt->streams[i]->codecpar;

        if (cp->codec_type != AVMEDIA_TYPE_AUDIO)
            continue;
        if (cp->ch_layout.nb_channels == 1)
            mono++;
        else if (cp->ch_layout.nb_channels == 2)
            stereo++;
        else if (cp->ch_layout.nb_channels >= 4 && quad < 0)
            quad = nb_audio;
        audio[nb_audio++] = (int)i;
    }

    if (!nb_audio)
        return CRAW_ERROR(ENOENT);

    if (map) {
        char *end;
        long first = strtol(map, &end, 10);
        long second = first;

        if (end == map || (*end && *end != ',')) {
            fprintf(stderr, "--audio-map: expected A[,B], got `%s`\n", map);
            return CRAW_ERROR(EINVAL);
        }
        if (*end == ',') {
            const char *p = end + 1;
            second = strtol(p, &end, 10);
            if (end == p || *end) {
                fprintf(stderr, "--audio-map: cannot parse `%s`\n", map);
                return CRAW_ERROR(EINVAL);
            }
        }
        if (first < 0 || first >= nb_audio || second < 0 || second >= nb_audio) {
            fprintf(stderr, "--audio-map: the file has %d audio streams\n", nb_audio);
            return CRAW_ERROR(EINVAL);
        }

        pick[0] = audio[first];
        pick[1] = first == second ? -1 : audio[second];
        snprintf(a->mapping, sizeof(a->mapping),
                 pick[1] < 0 ? "stream %ld" : "streams %ld+%ld as L/R",
                 first, second);
    } else if (nb_audio == 1 && stereo == 1) {
        pick[0] = audio[0];
        snprintf(a->mapping, sizeof(a->mapping), "stereo stream 0");
    } else if (mono >= 4) {
        /* The last two mono tracks are the stereo pair. */
        pick[0] = audio[nb_audio - 2];
        pick[1] = audio[nb_audio - 1];
        snprintf(a->mapping, sizeof(a->mapping), "streams %d+%d as L/R",
                 nb_audio - 2, nb_audio - 1);
    } else if (quad >= 0) {
        pick[0] = audio[quad];
        chan[0] = 2;
        chan[1] = 3;
        snprintf(a->mapping, sizeof(a->mapping), "stream %d channels 3+4 as L/R", quad);
    } else if (stereo >= 2) {
        pick[0] = audio[0];
        snprintf(a->mapping, sizeof(a->mapping), "first of %d stereo streams", stereo);
        if (verbose)
            fprintf(stderr, "audio: %d stereo streams, using the first\n", stereo);
    } else {
        pick[0] = audio[0];
        snprintf(a->mapping, sizeof(a->mapping), "stream 0");
    }

    a->nb_src = pick[1] >= 0 ? 2 : 1;
    for (int i = 0; i < a->nb_src; i++)
        a->src[i].stream_index = pick[i];
    a->chan[0] = chan[0];
    a->chan[1] = chan[1];

    return 0;
}

/* Opens one source's decoder and its converter. A lone source produces
 * stereo directly, with a channel map picking which input channels reach
 * which ear; a pair produces mono each and is interleaved later. */
static int view_audio_src_open(struct view_audio *a, int idx, bool verbose)
{
    struct view_audio_src *sr = &a->src[idx];
    AVChannelLayout out_layout;
    const AVCodec *codec;
    AVStream *st = a->fmt->streams[sr->stream_index];
    int ret;

    codec = avcodec_find_decoder(st->codecpar->codec_id);
    if (!codec)
        return CRAW_ERROR(ENOSYS);

    sr->dec = avcodec_alloc_context3(codec);
    if (!sr->dec)
        return CRAW_ERROR(ENOMEM);

    ret = avcodec_parameters_to_context(sr->dec, st->codecpar);
    if (ret < 0)
        return CRAW_ERROR(EINVAL);

    ret = avcodec_open2(sr->dec, codec, NULL);
    if (ret < 0) {
        fprintf(stderr, "cannot open the %s decoder: %s\n", codec->name, av_err2str(ret));
        return CRAW_ERROR(EINVAL);
    }

    sr->frame = av_frame_alloc();
    if (!sr->frame)
        return CRAW_ERROR(ENOMEM);

    /* A pair of sources each give one channel. A single source keeps its own
     * channel count when it has more than two, so that nothing is remixed
     * and the two wanted channels can be picked out by hand; a mono or
     * stereo source is converted straight to the pair, which is also what
     * sends a lone mono track to both ears. */
    if (a->nb_src == 2)
        sr->out_ch = 1;
    else if (sr->dec->ch_layout.nb_channels > VIEW_AUDIO_CHANNELS)
        sr->out_ch = sr->dec->ch_layout.nb_channels;
    else
        sr->out_ch = VIEW_AUDIO_CHANNELS;

    av_channel_layout_default(&out_layout, sr->out_ch);

    /* Same rate in and out, so this only unpacks the sample format. */
    ret = swr_alloc_set_opts2(&sr->swr, &out_layout, AV_SAMPLE_FMT_FLT,
                              VIEW_AUDIO_RATE, &sr->dec->ch_layout,
                              sr->dec->sample_fmt, sr->dec->sample_rate, 0, NULL);
    if (ret < 0) {
        fprintf(stderr, "cannot set up the sample format conversion: %s\n",
                av_err2str(ret));
        return CRAW_ERROR(EINVAL);
    }

    ret = swr_init(sr->swr);
    if (ret < 0) {
        fprintf(stderr, "cannot initialise the sample conversion: %s\n",
                av_err2str(ret));
        return CRAW_ERROR(EINVAL);
    }

    if (verbose)
        fprintf(stderr, "audio stream %d: %s, %d Hz, %d channels, %s\n",
                sr->stream_index, codec->name, sr->dec->sample_rate,
                sr->dec->ch_layout.nb_channels,
                av_get_sample_fmt_name(sr->dec->sample_fmt));

    return 0;
}

int view_audio_open(struct view_audio *a, const char *path, const char *map,
                    bool verbose)
{
    AVStream *st;
    int ret;

    *a = (struct view_audio) {
        .volume = 1.0f,
        .volume_target = 1.0f,
    };
    a->src[0].stream_index = a->src[1].stream_index = -1;

    ret = avformat_open_input(&a->fmt, path, NULL, NULL);
    if (ret < 0) {
        fprintf(stderr, "%s: cannot open a second reader for the audio: %s\n",
                path, av_err2str(ret));
        return 0;
    }

    ret = avformat_find_stream_info(a->fmt, NULL);
    if (ret < 0) {
        avformat_close_input(&a->fmt);
        return 0;
    }

    ret = view_audio_select(a, map, verbose);
    if (ret < 0) {
        if (ret == CRAW_ERROR(ENOENT) && verbose)
            fprintf(stderr, "no audio stream; the clock falls back to wall time\n");
        avformat_close_input(&a->fmt);
        return ret == CRAW_ERROR(EINVAL) ? ret : 0;
    }

    /* This reader only wants the tracks it plays. Without discarding the
     * rest the demuxer would hand back every five megabyte video packet
     * just to have it thrown away, which is most of the file per pass. */
    for (unsigned i = 0; i < a->fmt->nb_streams; i++) {
        bool keep = false;

        for (int j = 0; j < a->nb_src; j++)
            keep |= (int)i == a->src[j].stream_index;
        if (!keep)
            a->fmt->streams[i]->discard = AVDISCARD_ALL;
    }

    st = a->fmt->streams[a->src[0].stream_index];
    a->time_base = st->time_base;
    a->start_time = st->start_time != AV_NOPTS_VALUE ? st->start_time : 0;

    for (int i = 0; i < a->nb_src; i++) {
        ret = view_audio_src_open(a, i, verbose);
        if (ret < 0)
            goto fail;
    }

    a->pkt = av_packet_alloc();
    if (!a->pkt) {
        ret = CRAW_ERROR(ENOMEM);
        goto fail;
    }

    a->lock = SDL_CreateMutex();
    if (!a->lock) {
        ret = CRAW_ERROR(ENOMEM);
        goto fail;
    }

    a->ring_frames = (size_t)VIEW_AUDIO_RATE * VIEW_AUDIO_RING_MS / 1000;
    a->ring = calloc(a->ring_frames, VIEW_AUDIO_FRAME_BYTES);
    if (!a->ring) {
        ret = CRAW_ERROR(ENOMEM);
        goto fail;
    }

    a->fade_len = (uint32_t)(VIEW_AUDIO_RATE * VIEW_AUDIO_FADE_MS / 1000);
    a->decay_step = 1.0f / (float)(VIEW_AUDIO_RATE * VIEW_AUDIO_DECAY_MS / 1000);
    a->volume_step = 1.0f / (float)a->fade_len;
    a->state = VIEW_AUDIO_IDLE;

    ret = view_audio_device_open(a, verbose);
    if (ret < 0) {
        fprintf(stderr, "no audio device could be opened; playing silent\n");
        goto fail;
    }

    if (verbose)
        fprintf(stderr, "audio: %s\n", a->mapping);

    a->available = true;

    return 0;

fail:
    view_audio_close(a);
    /* Silent playback is a degraded mode, not a failure. */
    return 0;
}

void view_audio_close(struct view_audio *a)
{
    /* Destroying the stream stops the callback and closes the device it
     * opened, so it has to happen before anything the callback touches. */
    if (a->stream) {
        SDL_PauseAudioStreamDevice(a->stream);
        SDL_DestroyAudioStream(a->stream);
        a->stream = NULL;
    }
    if (a->dump) {
        fclose(a->dump);
        a->dump = NULL;
    }
    if (a->lock) {
        SDL_DestroyMutex(a->lock);
        a->lock = NULL;
    }

    free(a->ring);
    a->ring = NULL;
    free(a->mix);
    a->mix = NULL;
    a->mix_frames = 0;
    free(a->convert);
    a->convert = NULL;

    for (int i = 0; i < 2; i++) {
        swr_free(&a->src[i].swr);
        av_frame_free(&a->src[i].frame);
        avcodec_free_context(&a->src[i].dec);
        free(a->src[i].pend);
        a->src[i].pend = NULL;
        a->src[i].pend_cap = a->src[i].pend_n = 0;
    }
    av_packet_free(&a->pkt);
    avformat_close_input(&a->fmt);

    a->available = false;
    a->playing = false;
}

int view_audio_dump_open(struct view_audio *a, const char *path)
{
    FILE *f;

    if (!a->available) {
        fprintf(stderr, "--audio-dump %s: there is no audio to dump\n", path);
        return CRAW_ERROR(ENODEV);
    }

    f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "--audio-dump %s: cannot open for writing\n", path);
        return CRAW_ERROR(EIO);
    }

    SDL_LockMutex(a->lock);
    a->dump = f;
    SDL_UnlockMutex(a->lock);

    return 0;
}

/* --------------------------------------------------------------- feeding */

double view_audio_buffered(struct view_audio *a)
{
    double s;

    if (!a->available)
        return 0.0;

    SDL_LockMutex(a->lock);
    s = (double)a->fill / VIEW_AUDIO_RATE;
    SDL_UnlockMutex(a->lock);

    return s;
}

static bool view_audio_hungry(struct view_audio *a)
{
    bool hungry;

    SDL_LockMutex(a->lock);
    hungry = a->fill < (size_t)VIEW_AUDIO_RATE * VIEW_AUDIO_TARGET_MS / 1000;
    SDL_UnlockMutex(a->lock);

    return hungry;
}

/* Pushes ready stereo frames into the ring, honouring the post-seek
 * discard. Caller must not hold the lock. */
static void view_audio_emit(struct view_audio *a, const float *src, size_t frames,
                            double pts)
{
    size_t skip = 0;

    if (!frames)
        return;

    SDL_LockMutex(a->lock);

    /* After a seek the container hands back whole packets from before the
     * target; the samples in front of it are dropped here so that the run
     * starts exactly where the picture does. */
    if (a->dropping) {
        double end = pts + (double)frames / VIEW_AUDIO_RATE;

        if (end <= a->drop_before) {
            SDL_UnlockMutex(a->lock);
            return;
        }
        if (pts < a->drop_before) {
            skip = (size_t)llrint((a->drop_before - pts) * VIEW_AUDIO_RATE);
            skip = CRAW_MIN(skip, frames);
            pts += (double)skip / VIEW_AUDIO_RATE;
        }
        a->dropping = false;
    }

    if (!a->have_base_pts) {
        a->base_pts = pts;
        a->have_base_pts = true;
    }

    view_ring_push(a, src + skip * VIEW_AUDIO_CHANNELS, frames - skip);

    SDL_UnlockMutex(a->lock);
}

static int view_audio_pend_room(struct view_audio_src *sr, size_t want)
{
    float *p;

    if (sr->pend_cap >= want)
        return 0;

    p = realloc(sr->pend, want * sizeof(*p));
    if (!p)
        return CRAW_ERROR(ENOMEM);

    sr->pend = p;
    sr->pend_cap = want;

    return 0;
}

/* Interleaves as much as both mono sources can supply and hands it on. */
static void view_audio_pair(struct view_audio *a)
{
    struct view_audio_src *l = &a->src[0], *r = &a->src[1];
    size_t n = CRAW_MIN(l->pend_n, r->pend_n);
    float *out;

    if (!n)
        return;

    out = malloc(n * VIEW_AUDIO_FRAME_BYTES);
    if (!out)
        return;

    for (size_t i = 0; i < n; i++) {
        out[i * 2 + 0] = l->pend[i];
        out[i * 2 + 1] = r->pend[i];
    }

    view_audio_emit(a, out, n, l->pend_pts);
    free(out);

    memmove(l->pend, l->pend + n, (l->pend_n - n) * sizeof(*l->pend));
    memmove(r->pend, r->pend + n, (r->pend_n - n) * sizeof(*r->pend));
    l->pend_n -= n;
    r->pend_n -= n;
    l->pend_pts += (double)n / VIEW_AUDIO_RATE;
    r->pend_pts += (double)n / VIEW_AUDIO_RATE;
}

/* Converts one decoded frame of one source. A lone source already produces
 * the stereo pair; a pair of mono sources is buffered until both sides have
 * samples to interleave. */
static int view_audio_convert(struct view_audio *a, int idx, double pts)
{
    struct view_audio_src *sr = &a->src[idx];
    size_t need = (size_t)swr_get_out_samples(sr->swr, sr->frame->nb_samples);
    uint8_t *out;
    int got;

    if (a->nb_src == 2) {
        if (view_audio_pend_room(sr, sr->pend_n + need) < 0)
            return CRAW_ERROR(ENOMEM);
        if (!sr->pend_n)
            sr->pend_pts = pts;
        out = (uint8_t *)(sr->pend + sr->pend_n);
    } else {
        if (need * (size_t)sr->out_ch > a->convert_frames) {
            float *p = realloc(a->convert, need * (size_t)sr->out_ch * sizeof(*p));
            if (!p)
                return CRAW_ERROR(ENOMEM);
            a->convert = p;
            a->convert_frames = need * (size_t)sr->out_ch;
        }
        out = (uint8_t *)a->convert;
    }

    got = swr_convert(sr->swr, &out, (int)need,
                      (const uint8_t * const *)sr->frame->extended_data,
                      sr->frame->nb_samples);
    if (got < 0)
        return CRAW_ERROR(EINVAL);
    if (!got)
        return 0;

    if (a->nb_src == 2) {
        sr->pend_n += (size_t)got;
        view_audio_pair(a);
    } else if (sr->out_ch == VIEW_AUDIO_CHANNELS) {
        view_audio_emit(a, a->convert, (size_t)got, pts);
    } else {
        /* More than two channels: take the two that are the stereo pair and
         * leave the rest alone. */
        int l = CRAW_CLAMP(a->chan[0], 0, sr->out_ch - 1);
        int r = CRAW_CLAMP(a->chan[1], 0, sr->out_ch - 1);
        float *pair = malloc((size_t)got * VIEW_AUDIO_FRAME_BYTES);

        if (!pair)
            return CRAW_ERROR(ENOMEM);

        for (int i = 0; i < got; i++) {
            pair[i * 2 + 0] = a->convert[(size_t)i * sr->out_ch + l];
            pair[i * 2 + 1] = a->convert[(size_t)i * sr->out_ch + r];
        }

        view_audio_emit(a, pair, (size_t)got, pts);
        free(pair);
    }

    return 0;
}

/* Drains whatever the decoders are holding into the ring. Returns 1 once
 * every source has reported end of stream. */
static int view_audio_receive(struct view_audio *a)
{
    int done = 0;

    for (int i = 0; i < a->nb_src; i++) {
        struct view_audio_src *sr = &a->src[i];

        while (1) {
            double pts;
            int ret = avcodec_receive_frame(sr->dec, sr->frame);

            if (ret == AVERROR(EAGAIN))
                break;
            if (ret == AVERROR_EOF) {
                done++;
                break;
            }
            if (ret < 0)
                return CRAW_ERROR(EINVAL);

            pts = sr->frame->pts != AV_NOPTS_VALUE
                ? (double)(sr->frame->pts - a->start_time) * av_q2d(a->time_base)
                : 0.0;

            ret = view_audio_convert(a, i, pts);
            av_frame_unref(sr->frame);
            if (ret < 0)
                return ret;
        }
    }

    return done == a->nb_src;
}

int view_audio_pump(struct view_audio *a)
{
    int ret;

    if (!a->available || a->eof)
        return 0;

    while (view_audio_hungry(a)) {
        ret = view_audio_receive(a);
        if (ret < 0)
            return ret;
        if (ret > 0)
            break;                  /* the decoder is done */

        if (!view_audio_hungry(a))
            return 0;

        if (a->draining)
            break;

        ret = av_read_frame(a->fmt, a->pkt);
        if (ret == AVERROR_EOF) {
            a->demux_eof = true;
            for (int i = 0; i < a->nb_src; i++)
                avcodec_send_packet(a->src[i].dec, NULL);
            a->draining = true;
            continue;
        }
        if (ret < 0) {
            fprintf(stderr, "reading the audio failed: %s\n", av_err2str(ret));
            return CRAW_ERROR(EIO);
        }

        ret = AVERROR(EAGAIN);
        for (int i = 0; i < a->nb_src; i++) {
            if (a->pkt->stream_index != a->src[i].stream_index)
                continue;
            ret = avcodec_send_packet(a->src[i].dec, a->pkt);
            break;
        }
        av_packet_unref(a->pkt);
        if (ret < 0 && ret != AVERROR(EAGAIN)) {
            fprintf(stderr, "audio decoding failed: %s\n", av_err2str(ret));
            return CRAW_ERROR(EINVAL);
        }
    }

    /* Everything the track holds is in the ring: the callback can now spend
     * the tail on its ramp down instead of running dry. */
    if (a->draining) {
        SDL_LockMutex(a->lock);
        a->eof = true;
        if (!a->fill && a->state != VIEW_AUDIO_IDLE) {
            a->state = VIEW_AUDIO_FADE_OUT;
            a->fade_pos = a->fade_len;
        }
        SDL_UnlockMutex(a->lock);
    }

    return 0;
}

/* ----------------------------------------------------------- the clock */

/* Sample frames handed to the stream that it has not been asked for yet.
 * Bytes in, bytes out: SDL_GetAudioStreamQueued() counts what was put in,
 * in the format it was put in, which is ours. Never called with the lock
 * held - see the note in the callback. */
static double view_audio_queued_frames(struct view_audio *a)
{
    int bytes = a->stream ? SDL_GetAudioStreamQueued(a->stream) : 0;

    return bytes > 0 ? (double)bytes / VIEW_AUDIO_FRAME_BYTES : 0.0;
}

double view_audio_clock(struct view_audio *a)
{
    double queued = view_audio_queued_frames(a);
    double consumed, backlog, t, since;

    SDL_LockMutex(a->lock);

    /* What the device has actually taken is everything handed to the stream
     * less what is still sitting in it; the difference from what was handed
     * over is the backlog waiting inside SDL, and in the running state each
     * of those frames took one frame out of the ring. So the ring position
     * the speakers are at is `played` less that backlog, and one device
     * buffer further back again, because the hardware is filled a buffer
     * ahead of what it is playing.
     *
     * SDL2's callback mode could report neither half: it assumed exactly a
     * buffer of latency and nothing queued behind it. */
    consumed = (double)a->pushed - queued;
    backlog = (double)a->pushed - consumed;

    /* That cursor only moves when SDL pulls, once per device period, which
     * on its own would make the clock a 21 ms staircase; wall time since
     * the last handover interpolates across it, clamped to the period. */
    since = a->playing
          ? (double)(SDL_GetTicksNS() - a->handover_ns) / 1.0e9 : 0.0;
    since = CRAW_CLAMP(since, 0.0, a->device_latency);

    t = a->base_pts + ((double)a->played - backlog) / VIEW_AUDIO_RATE
      - a->device_latency + since;

    SDL_UnlockMutex(a->lock);

    return t;
}

/* --------------------------------------------------------- transitions */

/* Blocks until the callback has rendered the ramp down and the stream has
 * handed it on to the device. Stopping the device with the ramp still
 * queued inside SDL would throw the ramp away and leave exactly the step to
 * zero it exists to avoid; SDL2's callback wrote straight into the device
 * buffer and had nothing to wait for here. Bounded, because a device that
 * has stopped pulling must not wedge the viewer. */
static void view_audio_wait_idle(struct view_audio *a)
{
    uint64_t ramp_end = 0;
    bool idle = false;

    for (int i = 0; i < 200; i++) {
        /* Read before the lock: this asks the stream, and the callback runs
         * with the stream's lock held while it waits for ours. */
        double queued = view_audio_queued_frames(a);
        uint64_t pushed;

        SDL_LockMutex(a->lock);
        if (!idle && a->state == VIEW_AUDIO_IDLE) {
            idle = true;
            ramp_end = a->pushed;
        }
        pushed = a->pushed;
        SDL_UnlockMutex(a->lock);

        /* The ramp is safe once the device has taken every frame up to the
         * one the state machine went idle on. */
        if (idle && (double)pushed - queued >= (double)ramp_end)
            return;

        SDL_Delay(1);
    }
}

int view_audio_play(struct view_audio *a)
{
    if (!a->available || a->playing)
        return 0;

    /* Whatever silence the stream was left holding when it was stopped is
     * not part of this run; dropping it keeps the start deterministic and
     * the clock's backlog term honest. The device is stopped, so the
     * callback cannot be running. */
    SDL_ClearAudioStream(a->stream);

    SDL_LockMutex(a->lock);
    a->state = VIEW_AUDIO_FADE_IN;
    a->fade_pos = 0;
    a->in_underrun = false;
    a->pushed = 0;
    a->handover_ns = SDL_GetTicksNS();
    SDL_UnlockMutex(a->lock);

    SDL_ResumeAudioStreamDevice(a->stream);
    a->playing = true;

    return 0;
}

void view_audio_pause(struct view_audio *a)
{
    if (!a->available || !a->playing)
        return;

    SDL_LockMutex(a->lock);

    switch (a->state) {
    case VIEW_AUDIO_RUN:
        a->state = VIEW_AUDIO_FADE_OUT;
        a->fade_pos = 0;
        break;
    case VIEW_AUDIO_FADE_IN:
        /* Reverse the ramp from wherever it had got to, so the gain is
         * continuous through the turn. */
        a->state = VIEW_AUDIO_FADE_OUT;
        a->fade_pos = a->fade_len - a->fade_pos;
        break;
    case VIEW_AUDIO_FADE_OUT:
        /* Already on the way down, most likely the end of the clip;
         * restarting it would ask for samples that do not exist. */
        break;
    case VIEW_AUDIO_IDLE:
        break;
    }

    /* Nothing left to ramp over. */
    if (!a->fill)
        a->state = VIEW_AUDIO_IDLE;

    SDL_UnlockMutex(a->lock);

    /* The device is only stopped once the ramp has actually been rendered
     * and taken; pausing first is what turns a fade into a cut. */
    view_audio_wait_idle(a);
    SDL_PauseAudioStreamDevice(a->stream);
    a->playing = false;
}

void view_audio_seek(struct view_audio *a, double pts)
{
    int64_t ts;

    if (!a->available)
        return;

    view_audio_pause(a);

    ts = (int64_t)llrint(pts / av_q2d(a->time_base)) + a->start_time;
    if (av_seek_frame(a->fmt, a->src[0].stream_index, ts, AVSEEK_FLAG_BACKWARD) < 0)
        fprintf(stderr, "cannot seek the audio to %.3f s\n", pts);

    for (int i = 0; i < a->nb_src; i++) {
        avcodec_flush_buffers(a->src[i].dec);
        /* Drain whatever the converter is holding rather than letting it
         * appear at the head of the new run. */
        swr_convert(a->src[i].swr, NULL, 0, NULL, 0);
        a->src[i].pend_n = 0;
    }

    SDL_LockMutex(a->lock);
    view_ring_reset(a);
    /* The run starts at the target, so the clock says so from here on
     * rather than from zero: the picture that lands on the seek is
     * compared against this before a single sample has been decoded, and
     * an unanchored clock makes that comparison nonsense. `have_base_pts`
     * stays clear, so the first samples still anchor it exactly. */
    a->base_pts = pts;
    a->drop_before = pts;
    a->dropping = true;
    a->state = VIEW_AUDIO_IDLE;
    SDL_UnlockMutex(a->lock);

    a->eof = false;
    a->demux_eof = false;
    a->draining = false;
}

bool view_audio_drained(struct view_audio *a)
{
    bool done;

    if (!a->available)
        return true;

    SDL_LockMutex(a->lock);
    done = a->state == VIEW_AUDIO_IDLE && !a->fill;
    SDL_UnlockMutex(a->lock);

    return done;
}

void view_audio_set_volume(struct view_audio *a, float volume, bool mute)
{
    if (!a->available)
        return;

    SDL_LockMutex(a->lock);
    a->mute = mute;
    a->volume_target = mute ? 0.0f : CRAW_CLAMP(volume, 0.0f, 1.0f);
    SDL_UnlockMutex(a->lock);
}

#endif /* CRAW_VIEW_HAVE_AUDIO */
