/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* cineraw: decode a camera raw clip or still with FFmpeg, run every frame
 * through libcraw on the GPU, write an image sequence or one image.
 *
 * This is the development driver for the library: it is the only place
 * where a real clip, a real decoder and the whole pipeline meet, so it
 * favours loud diagnostics over speed on the CPU side. The GPU side is the
 * same code a player would run.
 *
 * Everything the library needs about the camera arrives from two places:
 * the per-frame AV_FRAME_DATA_RAW_COLOR_PARAMS side data (levels, as-shot
 * white balance, matrix, gain), and the per-CCT tables, which the decoder
 * does not expose yet and which therefore come from tools/profiles.c or a
 * text file. A Panasonic RW2 still carries only the levels and the white
 * balance, and the rest of its profile comes from those tables
 * (craw_tool_profile_from_frame()). */

#include <getopt.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
#include <libavutil/time.h>

#include <craw/craw.h>
#include <craw/vulkan.h>
#include <craw/profile.h>
#include <craw/pipeline.h>
#include <craw/awb.h>
#include <craw/noise.h>
#include <craw/lens.h>

#include "vkdev.h"
#include "profiles.h"

/* ------------------------------------------------------------- options */

enum tool_encode {
    TOOL_ENC_LINEAR = 0,
    TOOL_ENC_SRGB,
    TOOL_ENC_SDR,
    TOOL_ENC_PQ,
    TOOL_ENC_HLG,
    TOOL_ENC_VLOG,
};

enum tool_writer {
    TOOL_WRITER_NONE = 0,
    TOOL_WRITER_PNG,
    TOOL_WRITER_PPM,
    TOOL_WRITER_PFM,
    TOOL_WRITER_TIFF,
};

enum tool_wb {
    TOOL_WB_AS_SHOT = 0,
    TOOL_WB_AUTO,
    TOOL_WB_FIXED,
};

/* Geometric lens correction. AUTO and ON both mean "apply the frame's own
 * record"; they differ only in whether a clip that carries none is worth a
 * warning, which it is when the user asked for the correction by name.
 * KEEP is ON with the keep-everything mode of DESIGN.md 16.8 on top. */
enum tool_lens {
    TOOL_LENS_AUTO = 0,
    TOOL_LENS_ON,
    TOOL_LENS_KEEP,
    TOOL_LENS_OFF,
};

struct tool_cfg {
    const char *input;
    const char *pattern;
    /* The pattern has a frame index in it. Without one it names a single
     * file, which only a run that writes one frame may use. */
    bool pattern_indexed;
    enum tool_writer writer;

    int64_t frames;             /* -1: to the end of the clip */
    double start;
    const char *device;
    bool verbose;

    enum CRAWDebayer debayer;
    enum CRAWColorSpace space;
    bool space_set;
    enum tool_encode encode;
    bool encode_set;

    enum tool_wb wb;
    CRAWIlluminant fixed_illum;
    float awb_ambience;

    float exposure;
    float ei;               /* working exposure index, 0 = as-shot */
    float meta_as_shot_ei;  /* from the container, 0 if absent */
    float meta_apple_ei;    /* Apple's stored ExposureIndex, 0 if absent */
    bool clamp_negative;
    bool no_gamut_compression;
    bool float32;
    bool bench;
    bool stats;

    /* Output crop in pixels, w = 0 for the whole frame. */
    int64_t crop_x, crop_y, crop_w, crop_h;

    float denoise;            /* 0 = off */
    float chroma;             /* chroma denoise strength, 0 = off */
    float temporal;           /* temporal denoise alpha_max, 0 = off */
    CRAWNoiseModel noise;     /* a > 0 overrides the profile's table */
    bool noise_estimate;

    uint32_t false_color;     /* median passes after the debayer, 0 = off */
    bool highlights;          /* rebuild clipped highlights, DESIGN.md 18 */
    float defringe;           /* DESIGN.md 19, 0 = off */
    float sharpen_radius;     /* DESIGN.md 20, 0 = off */
    uint32_t sharpen_iterations;
    float sharpen_contrast;

    enum tool_lens lens;

    const char *tables;
    const char *profile;
};

struct tool_name_map {
    const char *name;
    int value;
};

static const struct tool_name_map tool_switches[] = {
    { "on",  1 },
    { "off", 0 },
    { NULL,  0 },
};

static const struct tool_name_map tool_debayers[] = {
    { "bilinear", CRAW_DEBAYER_BILINEAR },
    { "mhc",      CRAW_DEBAYER_MHC      },
    { "rcd",      CRAW_DEBAYER_RCD      },
    { NULL,       0                     },
};

static const struct tool_name_map tool_spaces[] = {
    { "xyz",    CRAW_CS_XYZ_D65         },
    { "bt709",  CRAW_CS_BT709_LINEAR    },
    { "bt2020", CRAW_CS_BT2020_LINEAR   },
    { "p3d65",  CRAW_CS_P3D65_LINEAR    },
    { "vgamut", CRAW_CS_V_GAMUT_LINEAR  },
    { "ap0",    CRAW_CS_ACES_AP0_LINEAR },
    { "ap1",    CRAW_CS_ACES_AP1_LINEAR },
    /* The ACES names for the same two. tool_space_name() returns the first
     * match, so the short names above stay what -v prints. */
    { "aces",   CRAW_CS_ACES_AP0_LINEAR },
    { "acescg", CRAW_CS_ACES_AP1_LINEAR },
    { NULL,     0                       },
};

static const struct tool_name_map tool_lenses[] = {
    { "auto", TOOL_LENS_AUTO },
    { "on",   TOOL_LENS_ON   },
    { "keep", TOOL_LENS_KEEP },
    { "off",  TOOL_LENS_OFF  },
    { NULL,   0              },
};

static const struct tool_name_map tool_encodes[] = {
    { "linear", TOOL_ENC_LINEAR },
    { "srgb",   TOOL_ENC_SRGB   },
    { "sdr",    TOOL_ENC_SDR    },
    { "pq",     TOOL_ENC_PQ     },
    { "hlg",    TOOL_ENC_HLG    },
    { "vlog",   TOOL_ENC_VLOG   },
    { NULL,     0               },
};

static int tool_lookup(const struct tool_name_map *map, const char *opt,
                       const char *arg, int *out)
{
    for (const struct tool_name_map *m = map; m->name; m++) {
        if (!strcmp(m->name, arg)) {
            *out = m->value;
            return 0;
        }
    }

    fprintf(stderr, "%s: unknown value `%s`, expected one of:", opt, arg);
    for (const struct tool_name_map *m = map; m->name; m++)
        fprintf(stderr, " %s", m->name);
    fprintf(stderr, "\n");

    return CRAW_ERROR(EINVAL);
}

static const char *tool_space_name(enum CRAWColorSpace cs)
{
    for (const struct tool_name_map *m = tool_spaces; m->name; m++)
        if (m->value == (int)cs)
            return m->name;
    return "?";
}

static void tool_usage(FILE *f)
{
    fprintf(f,
"usage: cineraw [options] INPUT\n"
"\n"
"Decodes a camera raw clip (ProRes RAW) or still (Panasonic RW2) with\n"
"FFmpeg, runs every frame through libcraw on the GPU and writes an image\n"
"sequence, or one image.\n"
"\n"
"  -o PATTERN          output path with a printf frame index, e.g.\n"
"                      out_%%05d.png. The extension picks the writer: .png\n"
"                      (16 bit), .tif (16 bit, uncompressed), .ppm (8 bit\n"
"                      preview, dithered), .pfm (linear float). A still, or\n"
"                      a run of --frames 1, may name one file instead,\n"
"                      e.g. out.tif.\n"
"                      Without -o nothing is written, which is what --bench\n"
"                      and `--wb auto -v` want.\n"
"  --frames N          stop after N frames (default: the whole clip)\n"
"  --start SECONDS     seek to this timestamp before decoding\n"
"  --device SUBSTR     use the Vulkan device whose name contains SUBSTR;\n"
"                      the CRAW_TEST_DEVICE environment variable does the\n"
"                      same and wins over this option, and\n"
"                      CRAW_TEST_VALIDATION=1 turns the validation layer on\n"
"  -v                  verbose: per frame diagnostics with the decode and\n"
"                      develop times, container and still metadata, and\n"
"                      libcraw's own log at CRAW_LOG_VERBOSE\n"
"  -h                  this text\n"
"\n"
"  --debayer METHOD    bilinear, mhc or rcd (default: rcd)\n"
"  --space SPACE       xyz, bt709, bt2020, p3d65, vgamut, ap0 (also `aces`)\n"
"                      or ap1 (also `acescg`); the linear output space of\n"
"                      the pipeline (default: bt709).\n"
"                      ap1 and ap0 are the ACES pair: both are scene-linear\n"
"                      against the ACES white point (x 0.32168, y 0.33767,\n"
"                      adapted from the camera matrix\'s D65 with Bradford)\n"
"                      with 1.0 at diffuse white, not display referred, so\n"
"                      pair them with --encode linear and a float or 16-bit\n"
"                      writer. ap1 (ACEScg) is the rendering and display\n"
"                      hand-off and ends with the ACES 1.3 Reference Gamut\n"
"                      Compression, so it has no negative components; ap0\n"
"                      (ACES2065-1) is the archival encoding and is never\n"
"                      compressed. See DESIGN.md 8.1\n"
"  --encode TRC        transfer function applied on the CPU after download:\n"
"                      linear, srgb, sdr, pq, hlg or vlog (default: sdr, or\n"
"                      linear for .pfm). srgb clips at 1.0; sdr first rolls\n"
"                      each channel above 0.6 off towards 1.0, a filmic\n"
"                      shoulder that lets highlights go to white, then\n"
"                      applies the sRGB curve, which\n"
"                      is what a scene-linear frame needs to be judged on an\n"
"                      SDR screen. pq and hlg map a pipeline 1.0 to\n"
"                      203 cd/m2 per BT.2408; vlog forces --space vgamut.\n"
"                      Everything but linear clips negatives to zero.\n"
"  --wb MODE           as-shot, auto, or CCT[:TINT] in Kelvin to rebalance\n"
"                      to a fixed illuminant (default: as-shot)\n"
"  --awb-ambience F    0.0 to 1.0: how much of an illuminant warmer than\n"
"                      4000 K `--wb auto` deliberately leaves uncorrected\n"
"                      (default: 0.0, correct fully)\n"
"  --ei INDEX          working exposure index. The frame is scaled by\n"
"                      INDEX / as-shot ISO, the way Apple's decoder rates a\n"
"                      clip: as-shot ISO comes from the SMPTE RDD 18 key and\n"
"                      INDEX defaults to it, so camera originals scale by 1.0.\n"
"                      Apple offers the same ISO list plus an exposure bias\n"
"                      of at most one stop; --exposure is that bias, unclamped\n"
"  --exposure STOPS    exposure offset in stops on top of --ei (default: 0)\n"
"  --clamp-negative    clamp negative output components to zero\n"
"  --no-gamut-compression\n"
"                      skip the ACES 1.3 Reference Gamut Compression that\n"
"                      --space ap1 normally ends with, and write the raw\n"
"                      ACEScg values, negative components and all. No\n"
"                      effect on any other space\n"
"  --float32           RGBA32F pipeline output instead of RGBA16F\n"
"  --crop X,Y,W,H      write, and compute --stats over, only this rectangle\n"
"                      of the output. The pipeline still runs on the whole\n"
"                      frame, so a crop is what the full frame would have\n"
"                      looked like there, which is the point: it is how a\n"
"                      5.8K frame gets looked at 1:1 without a 5.8K file.\n"
"                      A clip whose frame headers mark a picture inside the\n"
"                      readout (ProRes RAW carries a crop; the whole readout\n"
"                      is still decoded and developed) is written cut to that\n"
"                      picture by default; --crop 0,0,W,H of the coded size\n"
"                      writes all of it\n"
"  --false-colour PASSES\n"
"                      false colour suppression after the debayer: PASSES\n"
"                      (1 to 5) of a 3x3 median and then a 3x3 blur of the\n"
"                      colour differences R-G and B-G, RawTherapee's steps,\n"
"                      which turn the coloured dots that glints smaller than\n"
"                      a CFA quad leave into neutral ones and leave luminance\n"
"                      alone. 0 (the default) is off. One or two are below\n"
"                      what a 4:2:0 delivery keeps of the chroma anyway;\n"
"                      five is a visible chroma blur at 1:1. DESIGN.md 17\n"
"  --highlights on|off\n"
"                      rebuild clipped highlights before the debayer\n"
"                      (default: on): every channel that clipped is\n"
"                      estimated from the channels that did not clip around\n"
"                      it and from the colour of the unclipped light around\n"
"                      the blown area, and may exceed the balanced clip\n"
"                      point, so a sky where only green clipped keeps its\n"
"                      clouds and a blown lamp keeps its colour. DESIGN.md 18\n"
"  --defringe STRENGTH\n"
"                      pull the colour of pixels along strong edges back into\n"
"                      the range the two sides of the edge span, 0 to 1\n"
"                      (default: 1): the purple, blue and pink rims chromatic\n"
"                      aberration leaves along lamps, windows and branches\n"
"                      against the sky. 0 is off. DESIGN.md 19\n"
"  --sharpen RADIUS[,ITERATIONS[,CONTRAST]]\n"
"                      capture sharpening, after every other stage on the\n"
"                      camera RGB: RADIUS is the sigma in pixels of the\n"
"                      Gaussian blur undone (0.8 is this camera's in-focus\n"
"                      blur, up to %g), ITERATIONS the Richardson-Lucy\n"
"                      iterations (default 20, at most %d, 0.4 ms each at\n"
"                      5.8K), CONTRAST the local contrast below which\n"
"                      nothing is sharpened, in units of the modelled noise\n"
"                      like --denoise (default 1; 0 sharpens noise too; with\n"
"                      --denoise, 0.5 sharpens more texture and none of the\n"
"                      noise it leaves). The luma is sharpened and every\n"
"                      channel scaled with it, so no colour moves, and no\n"
"                      pixel goes past the range of its neighbours, so a\n"
"                      lamp gets no dark ring. Off by default. DESIGN.md 20\n"
"  --denoise STRENGTH[,CHROMA[,TEMPORAL]]\n"
"                      raw domain denoising at this strength: 1.0 removes\n"
"                      noise at the level the camera's noise model predicts,\n"
"                      2.0 smooths twice as far, 0 (the default) is off. The\n"
"                      model comes from the profile's table at the clip's\n"
"                      as-shot ISO unless --noise overrides it. CHROMA, 0 to\n"
"                      1, is the chroma denoise after the debayer: how much\n"
"                      of the quarter resolution chroma band is replaced by\n"
"                      its luma-guided smoothing, which takes the colour\n"
"                      blotches out of a dark scene and the colour out of\n"
"                      the glints --false-colour leaves; 0 (the default) is\n"
"                      off, and it needs no noise model. DESIGN.md 14.4.\n"
"                      TEMPORAL, 0 to 1, averages every sample with the\n"
"                      previous frame's wherever the two agree within the\n"
"                      model's noise, before the spatial pass: the most of\n"
"                      the previous frame a sample keeps, 0.8 being about\n"
"                      five frames on a static shot; 0 (the default) is off.\n"
"                      It needs the noise model, and works with STRENGTH 0\n"
"                      too. DESIGN.md 14.7\n"
"  --noise A,B         override the noise model: variance = A * u + B with u\n"
"                      the signal above black over the sensor range. Print a\n"
"                      clip's own pair with --noise-estimate\n"
"  --lens on|off|keep  geometric lens correction from the frame's own\n"
"                      com.panasonic.Semi-Pro.optical_correction record: a\n"
"                      per-channel radial warp after the debayer that takes\n"
"                      the distortion out and the lateral chromatic\n"
"                      aberration with it. The default is on for every frame\n"
"                      that carries a record and off for every frame that\n"
"                      does not, so a clip from another camera is untouched;\n"
"                      `on` says so explicitly and warns when a clip carries\n"
"                      no record at all. With -v the corner correction per\n"
"                      channel is printed for the first frame and again\n"
"                      whenever it moves by more than half a pixel, which on\n"
"                      a clip that zooms while recording is once per step of\n"
"                      the zoom ring.\n"
"                      `keep` corrects the same way but keeps every source\n"
"                      pixel instead of filling the frame: the corrected\n"
"                      picture is scaled down until all of it fits and the\n"
"                      rest of the frame is black. -v prints the scale\n"
"  --noise-estimate    fit A and B on every decoded frame and print them,\n"
"                      then a summary; no GPU work and no output is written.\n"
"                      Point it at a flat, evenly lit, unclipped field (a\n"
"                      sheet of white paper filling the frame) for a\n"
"                      calibration, and check that the printed fit error is\n"
"                      small and the read noise plausible before believing it\n"
"  --bench             per frame GPU and decode time, plus the averages,\n"
"                      and a table of every pass's GPU time, average and\n"
"                      median over the run. The software decode keeps the\n"
"                      GPU idle between frames, so every frame runs on a\n"
"                      cold GPU: the passes compare with each other, but a\n"
"                      player's warm frames are faster (DESIGN.md 21)\n"
"  --stats             per frame min/mean/p50/p99/max of the output's Y\n"
"                      (XYZ output) or G channel\n"
"  --tables FILE       load the per-CCT tables from a text file of\n"
"                      `wb CCT R B` and `matrix CCT m00 .. m22` lines,\n"
"                      `#` comments, any order\n"
"  --profile NAME      use a built-in table set (%s). The default is to\n"
"                      match the camera model in the container metadata,\n"
"                      or in the first frame's Exif for a still, and to\n"
"                      carry on without tables if that fails. An RW2\n"
"                      still carries no colour matrix and is refused\n"
"                      without tables: its matrix and colour temperature\n"
"                      are where its as-shot white balance lies on them\n"
"                      (DESIGN.md 9c).\n"
"\n"
"`--wb auto` estimates the illuminant from the histogram of the frame it\n"
"has just processed and applies the smoothed result to the *next* frame, so\n"
"the correction lags the picture by one frame. That is deliberate for now:\n"
"correcting the frame itself would mean running the pipeline twice.\n",
    (double)CRAW_SHARPEN_MAX_RADIUS, CRAW_SHARPEN_MAX_ITERATIONS,
    craw_tool_profile_builtin_list());
}

/* ------------------------------------------------------------ encoding */

/* SMPTE ST 2084 inverse EOTF constants. */
#define TOOL_PQ_M1 0.1593017578125f
#define TOOL_PQ_M2 78.84375f
#define TOOL_PQ_C1 0.8359375f
#define TOOL_PQ_C2 18.8515625f
#define TOOL_PQ_C3 18.6875f

/* BT.2408 places diffuse white, i.e. a pipeline value of 1.0, at 203 nits. */
#define TOOL_REF_WHITE_NITS 203.0f

static float tool_encode_value(enum tool_encode enc, float v)
{
    switch (enc) {
    case TOOL_ENC_LINEAR:
        /* The only encode that passes negatives and highlights through. */
        return v;

    case TOOL_ENC_SRGB:
    case TOOL_ENC_SDR:
        /* For sdr the roll-off already happened in tool_rolloff(). */
        v = CRAW_CLAMP(v, 0.0f, 1.0f);
        return v <= 0.0031308f ? 12.92f * v
                               : 1.055f * powf(v, 1.0f / 2.4f) - 0.055f;

    case TOOL_ENC_PQ: {
        float y = CRAW_MAX(v, 0.0f) * (TOOL_REF_WHITE_NITS / 10000.0f);
        float yp = powf(CRAW_MIN(y, 1.0f), TOOL_PQ_M1);
        return powf((TOOL_PQ_C1 + TOOL_PQ_C2 * yp) / (1.0f + TOOL_PQ_C3 * yp), TOOL_PQ_M2);
    }

    case TOOL_ENC_HLG: {
        float e = CRAW_CLAMP(v * (TOOL_REF_WHITE_NITS / 1000.0f), 0.0f, 1.0f);
        if (e <= 1.0f / 12.0f)
            return 0.5f * sqrtf(e);
        return 0.17883277f * logf(12.0f * e - 0.28466892f) + 0.55991073f;
    }

    case TOOL_ENC_VLOG: {
        float x = CRAW_MAX(v, 0.0f);
        if (x < 0.01f)
            return 5.6f * x + 0.125f;
        return 0.241514f * log10f(x + 0.00873f) + 0.598206f;
    }
    }

    return v;
}

/* -------------------------------------------------------------- output */

static enum tool_writer tool_writer_from_path(const char *path)
{
    const char *dot = strrchr(path, '.');

    if (!dot)
        return TOOL_WRITER_NONE;
    if (!strcmp(dot, ".png"))
        return TOOL_WRITER_PNG;
    if (!strcmp(dot, ".ppm"))
        return TOOL_WRITER_PPM;
    if (!strcmp(dot, ".pfm"))
        return TOOL_WRITER_PFM;
    if (!strcmp(dot, ".tif") || !strcmp(dot, ".tiff"))
        return TOOL_WRITER_TIFF;

    return TOOL_WRITER_NONE;
}

/* The pattern is handed to snprintf(), so at most one integer conversion
 * may be there: two read past the end of the argument list. Returns how
 * many there are. None is a single file, which would be overwritten by
 * every frame of a clip, so main() lets only a run of one frame have it
 * (tool_check_single()). */
static int tool_check_pattern(const char *pattern)
{
    int nb = 0;

    for (const char *p = pattern; *p; p++) {
        if (*p != '%')
            continue;
        if (p[1] == '%') {
            p++;
            continue;
        }
        nb++;
        p++;
        while (*p && !strchr("diouxX", *p)) {
            if (strchr("eEfgGaAcspn%", *p)) {
                fprintf(stderr, "-o: `%s` has a non-integer conversion\n", pattern);
                return CRAW_ERROR(EINVAL);
            }
            p++;
        }
        if (!*p) {
            fprintf(stderr, "-o: `%s` has an unterminated conversion\n", pattern);
            return CRAW_ERROR(EINVAL);
        }
    }

    if (nb > 1) {
        fprintf(stderr, "-o: `%s` needs exactly one printf frame index "
                        "(e.g. out_%%05d.png), found %d\n", pattern, nb);
        return CRAW_ERROR(EINVAL);
    }

    return nb;
}

/* PFM as the convention has it: rows bottom-up, interleaved RGB, and a
 * negative scale to mean little endian samples, which is what a memcpy of
 * host floats gives on every machine this runs on. */
static int tool_write_pfm(const char *path, const float *rgba, uint32_t w,
                          uint32_t h, enum tool_encode enc)
{
    float *row = NULL;
    FILE *f = NULL;
    int ret = 0;

    row = malloc((size_t)w * 3 * sizeof(*row));
    if (!row)
        return CRAW_ERROR(ENOMEM);

    f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "%s: cannot open for writing\n", path);
        ret = CRAW_ERROR(EIO);
        goto fail;
    }

    fprintf(f, "PF\n%u %u\n-1.0\n", w, h);

    for (uint32_t y = 0; y < h; y++) {
        const float *src = rgba + (size_t)(h - 1 - y) * w * 4;

        for (uint32_t x = 0; x < w; x++)
            for (int c = 0; c < 3; c++)
                row[x * 3 + c] = tool_encode_value(enc, src[x * 4 + c]);

        if (fwrite(row, sizeof(*row), (size_t)w * 3, f) != (size_t)w * 3) {
            fprintf(stderr, "%s: short write\n", path);
            ret = CRAW_ERROR(EIO);
            goto fail;
        }
    }

fail:
    if (f)
        fclose(f);
    free(row);
    return ret;
}

/* Triangular dither of one quantisation step, for the 8 bit writer only.
 *
 * A denoised frame is exactly the case integer output handles worst. The
 * sensor noise that used to dither the steps has been removed on purpose,
 * so a smooth shadow gradient lands on one code for tens of pixels and the
 * boundaries between codes turn into visible contours. Measured on frame 0
 * of P1000387.MOV, DESIGN.md 14.3: the same dark gradient runs 1.22 pixels
 * per 8 bit code undenoised and 3.10, with runs up to 66 pixels, after
 * denoising, while the float field it came from got *smoother* (its
 * laplacian rms fell from 2.8% of the mean to 0.6%). Nothing is wrong with
 * the picture; the eight bit grid is too coarse for it once the noise is
 * gone.
 *
 * Two uniform variates summed give a triangular distribution two steps
 * peak to peak, which is the textbook amount: it decorrelates the
 * quantisation error from the signal completely, at the cost of raising the
 * total quantisation noise from 1/sqrt(12) to 1/2 of a step. At 16 bits the
 * step is 1/256 of an 8 bit code and never visible, so that path is left
 * alone: dither is added noise and only belongs where it buys something.
 * Dithering at 16 bits would not help a consumer that goes on to 8 bits
 * either, since it is the last quantisation that has to carry it.
 *
 * The pattern is a hash of the position, not a running PRNG, so the same
 * frame always produces the same file. */
static uint32_t tool_hash(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static float tool_dither(uint32_t x, uint32_t y, int c)
{
    uint32_t h = tool_hash(x * 73856093u ^ y * 19349663u ^ (uint32_t)c * 83492791u);

    return (float)(h & 0xFFFFu) * (1.0f / 65536.0f)
         + (float)(h >> 16)     * (1.0f / 65536.0f) - 1.0f;
}

/* PNG, TIFF and PPM all go through libavcodec, so no image writer of our
 * own and no libswscale: the pixel conversion is a clip and a scale. 16 bit
 * PNG is big endian by definition and the encoder takes nothing else; the
 * TIFF encoder takes only little endian, and is told not to compress,
 * which is the point of asking for a TIFF: a 5.8K frame is 0.4 s of
 * writing where PNG's deflate is ten. */
static int tool_write_avcodec(const char *path, enum AVCodecID codec_id,
                              enum AVPixelFormat pix_fmt, const float *rgba,
                              uint32_t w, uint32_t h, enum tool_encode enc)
{
    const bool be = pix_fmt == AV_PIX_FMT_RGB48BE;
    const int depth = be || pix_fmt == AV_PIX_FMT_RGB48LE ? 16 : 8;
    const float scale = depth == 16 ? 65535.0f : 255.0f;
    AVDictionary *opts = NULL;
    const AVCodec *codec;
    AVCodecContext *avc = NULL;
    AVFrame *frame = NULL;
    AVPacket *pkt = NULL;
    FILE *f = NULL;
    int ret;

    codec = avcodec_find_encoder(codec_id);
    if (!codec) {
        fprintf(stderr, "libavcodec has no %s encoder\n", avcodec_get_name(codec_id));
        ret = CRAW_ERROR(ENOSYS);
        goto fail;
    }

    avc = avcodec_alloc_context3(codec);
    if (!avc) {
        ret = CRAW_ERROR(ENOMEM);
        goto fail;
    }

    avc->width = (int)w;
    avc->height = (int)h;
    avc->pix_fmt = pix_fmt;
    avc->time_base = (AVRational) { 1, 1 };

    if (codec_id == AV_CODEC_ID_TIFF)
        av_dict_set(&opts, "compression_algo", "raw", 0);

    ret = avcodec_open2(avc, codec, &opts);
    av_dict_free(&opts);
    if (ret < 0) {
        fprintf(stderr, "avcodec_open2(%s): %s\n", codec->name, av_err2str(ret));
        ret = CRAW_ERROR(EINVAL);
        goto fail;
    }

    frame = av_frame_alloc();
    pkt = av_packet_alloc();
    if (!frame || !pkt) {
        ret = CRAW_ERROR(ENOMEM);
        goto fail;
    }

    frame->format = pix_fmt;
    frame->width = (int)w;
    frame->height = (int)h;

    ret = av_frame_get_buffer(frame, 0);
    if (ret < 0) {
        ret = CRAW_ERROR(ENOMEM);
        goto fail;
    }

    for (uint32_t y = 0; y < h; y++) {
        const float *src = rgba + (size_t)y * w * 4;
        uint8_t *dst = frame->data[0] + (ptrdiff_t)y * frame->linesize[0];

        for (uint32_t x = 0; x < w; x++) {
            for (int c = 0; c < 3; c++) {
                float v = tool_encode_value(enc, src[x * 4 + c]);
                long iq;
                unsigned q;

                v = CRAW_CLAMP(v, 0.0f, 1.0f);
                iq = lrintf(v * scale + (depth == 8 ? tool_dither(x, y, c) : 0.0f));
                q = (unsigned)CRAW_CLAMP(iq, 0, (long)scale);

                /* Written byte by byte so the byte order the format names
                 * is honoured whatever the host is. */
                if (depth == 16) {
                    dst[(x * 3 + c) * 2 + !be] = (uint8_t)(q >> 8);
                    dst[(x * 3 + c) * 2 +  be] = (uint8_t)(q & 0xFF);
                } else {
                    dst[x * 3 + c] = (uint8_t)q;
                }
            }
        }
    }

    ret = avcodec_send_frame(avc, frame);
    if (ret >= 0)
        ret = avcodec_send_frame(avc, NULL);
    if (ret < 0) {
        fprintf(stderr, "%s: avcodec_send_frame: %s\n", path, av_err2str(ret));
        ret = CRAW_ERROR(EINVAL);
        goto fail;
    }

    ret = avcodec_receive_packet(avc, pkt);
    if (ret < 0) {
        fprintf(stderr, "%s: avcodec_receive_packet: %s\n", path, av_err2str(ret));
        ret = CRAW_ERROR(EINVAL);
        goto fail;
    }

    f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "%s: cannot open for writing\n", path);
        ret = CRAW_ERROR(EIO);
        goto fail;
    }

    if (fwrite(pkt->data, 1, (size_t)pkt->size, f) != (size_t)pkt->size) {
        fprintf(stderr, "%s: short write\n", path);
        ret = CRAW_ERROR(EIO);
        goto fail;
    }

    ret = 0;

fail:
    if (f)
        fclose(f);
    av_packet_free(&pkt);
    av_frame_free(&frame);
    avcodec_free_context(&avc);
    return ret;
}

/* Rolls scene-linear highlights off for an SDR screen, per channel: linear
 * up to a knee, eased towards 1.0 above it. Doing it per channel rather
 * than on luminance lets a bright saturated highlight desaturate towards
 * white the way film and camera looks do; a luminance-only roll-off keeps
 * the chromaticity and leaves a lit wood surface as a flat fried orange
 * plateau once its channels clip. In place; the buffer is redownloaded
 * every frame. */
static void tool_rolloff(float *rgba, uint32_t w, uint32_t h)
{
    const float knee = 0.6f;

    for (size_t i = 0; i < (size_t)w * h; i++) {
        float *px = rgba + i * 4;

        for (int c = 0; c < 3; c++) {
            float v = px[c];
            if (v > knee)
                px[c] = knee + (1.0f - knee) * (1.0f - expf(-(v - knee) / (1.0f - knee)));
        }
    }
}

static int tool_write_frame(const struct tool_cfg *cfg, const char *path,
                            float *rgba, uint32_t w, uint32_t h)
{
    enum tool_encode enc = cfg->encode;

    if (enc == TOOL_ENC_SDR && cfg->writer != TOOL_WRITER_NONE) {
        tool_rolloff(rgba, w, h);
        enc = TOOL_ENC_SRGB;
    }

    switch (cfg->writer) {
    case TOOL_WRITER_PFM:
        return tool_write_pfm(path, rgba, w, h, enc);
    case TOOL_WRITER_PNG:
        return tool_write_avcodec(path, AV_CODEC_ID_PNG, AV_PIX_FMT_RGB48BE,
                                  rgba, w, h, enc);
    case TOOL_WRITER_TIFF:
        return tool_write_avcodec(path, AV_CODEC_ID_TIFF, AV_PIX_FMT_RGB48LE,
                                  rgba, w, h, enc);
    case TOOL_WRITER_PPM:
        return tool_write_avcodec(path, AV_CODEC_ID_PPM, AV_PIX_FMT_RGB24,
                                  rgba, w, h, enc);
    case TOOL_WRITER_NONE:
        break;
    }

    return 0;
}

/* --------------------------------------------------------------- stats */

struct tool_stats {
    float min, mean, p50, p99, max;
};

/* Quickselect; reorders `a`, which is a throwaway copy of one channel. */
static float tool_nth(float *a, size_t n, size_t k)
{
    ptrdiff_t lo = 0, hi = (ptrdiff_t)n - 1;

    while (lo < hi) {
        float pivot = a[lo + (hi - lo) / 2];
        ptrdiff_t i = lo, j = hi;

        while (i <= j) {
            float t;

            while (a[i] < pivot)
                i++;
            while (a[j] > pivot)
                j--;
            if (i > j)
                break;

            t = a[i];
            a[i] = a[j];
            a[j] = t;
            i++;
            j--;
        }

        if ((ptrdiff_t)k <= j)
            hi = j;
        else if ((ptrdiff_t)k >= i)
            lo = i;
        else
            break;
    }

    return a[k];
}

/* Channel 1 is Y in XYZ output and G in every RGB space, so the same index
 * answers both questions. */
static void tool_stats_compute(const float *rgba, size_t npix, float *scratch,
                               struct tool_stats *st)
{
    double sum = 0.0;

    st->min = INFINITY;
    st->max = -INFINITY;

    for (size_t i = 0; i < npix; i++) {
        float v = rgba[i * 4 + 1];

        scratch[i] = v;
        sum += v;
        st->min = CRAW_MIN(st->min, v);
        st->max = CRAW_MAX(st->max, v);
    }

    st->mean = (float)(sum / (double)npix);
    st->p50 = tool_nth(scratch, npix, npix / 2);
    st->p99 = tool_nth(scratch, npix, (size_t)(0.99 * (double)(npix - 1)));
}

/* --------------------------------------------------------------- gpu */

struct tool_gpu {
    CRAWTestDevice td;
    bool have_dev;
    CRAWContext *ctx;

    CRAWPipeline *pipe;
    CRAWTestImage src, dst;
    bool have_src, have_dst;
    uint32_t width, height;
    enum CRAWCfa cfa;
    bool lens;              /* the pipeline was built with the warp */
};

struct tool_record {
    CRAWPipeline *pipe;
    const CRAWImage *src, *dst;
    const CRAWFrameParams *fp;
};

static int tool_record_cb(VkCommandBuffer cmd, void *opaque)
{
    struct tool_record *r = opaque;
    return craw_pipeline_record(r->pipe, cmd, r->src, r->dst, r->fp);
}

static void tool_gpu_drop_size(struct tool_gpu *g)
{
    craw_pipeline_free(&g->pipe);
    if (g->have_dst) {
        craw_testdev_image_free(&g->td, &g->dst);
        g->have_dst = false;
    }
    if (g->have_src) {
        craw_testdev_image_free(&g->td, &g->src);
        g->have_src = false;
    }
    g->width = g->height = 0;
}

/* Sized to the frame that actually came out of the decoder, cropping and
 * all, and rebuilt whenever that changes mid-clip. */
static int tool_gpu_resize(struct tool_gpu *g, const struct tool_cfg *cfg,
                           uint32_t w, uint32_t h, enum CRAWCfa cfa, bool lens)
{
    VkFormat out_format = cfg->float32 ? VK_FORMAT_R32G32B32A32_SFLOAT
                                       : VK_FORMAT_R16G16B16A16_SFLOAT;
    CRAWPipelineParams pp;
    int ret;

    if (g->pipe && g->width == w && g->height == h && g->cfa == cfa && g->lens == lens)
        return 0;

    if (g->width)
        fprintf(stderr, "frame size changed from %ux%u to %ux%u, rebuilding\n",
                g->width, g->height, w, h);

    tool_gpu_drop_size(g);

    ret = craw_testdev_image_create(&g->td, &g->src, w, h, VK_FORMAT_R16_UNORM);
    if (ret < 0) {
        fprintf(stderr, "cannot create the %ux%u input image: %d\n", w, h, ret);
        return ret;
    }
    g->have_src = true;

    ret = craw_testdev_image_create(&g->td, &g->dst, w, h, out_format);
    if (ret < 0) {
        fprintf(stderr, "cannot create the %ux%u output image: %d\n", w, h, ret);
        return ret;
    }
    g->have_dst = true;

    craw_pipeline_params_init(&pp);
    pp.width = w;
    pp.height = h;
    /* The frame's own phase; the pipeline says so if it cannot develop
     * it. */
    pp.cfa = cfa;
    pp.debayer = cfg->debayer;
    pp.out_space = cfg->space;
    pp.out_format = out_format;
    pp.flags = cfg->wb == TOOL_WB_AUTO ? CRAW_PIPELINE_FLAG_AWB_HISTOGRAM : 0;
    if (cfg->denoise > 0.0f)
        pp.flags |= CRAW_PIPELINE_FLAG_DENOISE;
    if (cfg->temporal > 0.0f)
        pp.flags |= CRAW_PIPELINE_FLAG_TEMPORAL_DENOISE;
    if (cfg->chroma > 0.0f)
        pp.flags |= CRAW_PIPELINE_FLAG_CHROMA_DENOISE;
    if (cfg->false_color > 0)
        pp.flags |= CRAW_PIPELINE_FLAG_FALSE_COLOR;
    if (cfg->highlights)
        pp.flags |= CRAW_PIPELINE_FLAG_HIGHLIGHT_RECONSTRUCTION;
    if (cfg->defringe > 0.0f)
        pp.flags |= CRAW_PIPELINE_FLAG_DEFRINGE;
    if (cfg->sharpen_radius > 0.0f)
        pp.flags |= CRAW_PIPELINE_FLAG_SHARPEN;
    /* Only built once a frame has been seen to carry a record: a clip
     * without one never pays for the extra image. */
    if (lens)
        pp.flags |= CRAW_PIPELINE_FLAG_LENS;
    if (cfg->bench)
        pp.flags |= CRAW_PIPELINE_FLAG_PROFILE;

    ret = craw_pipeline_create(g->ctx, &pp, &g->pipe);
    if (ret < 0) {
        fprintf(stderr, "craw_pipeline_create(%ux%u) failed: %d\n", w, h, ret);
        return ret;
    }

    g->width = w;
    g->height = h;
    g->cfa = cfa;
    g->lens = lens;

    return 0;
}

static float tool_half_to_float(uint16_t h)
{
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    uint32_t bits;
    float f;

    if (e == 0) {
        /* Zero or subnormal: m * 2^-24, exact in fp32. */
        f = (float)m * 5.9604644775390625e-8f;
        return (h & 0x8000u) ? -f : f;
    }

    bits = (uint32_t)(h & 0x8000u) << 16;
    bits |= e == 31 ? (0x7F800000u | (m << 13)) : (((e + 112u) << 23) | (m << 13));
    memcpy(&f, &bits, sizeof(f));

    return f;
}

/* -------------------------------------------------------- host buffers */

/* One 5760x3040 frame is 35 MB of mosaic and 280 MB of float RGBA, so the
 * host side allocates once and grows only if the frame size does. */
struct tool_buffers {
    uint16_t *mosaic;
    size_t mosaic_samples;
    uint16_t *half;
    size_t half_values;
    float *rgba;
    size_t rgba_values;
    float *channel;
    size_t channel_values;
};

/* On failure the old buffer is released and NULL comes back, so the
 * caller's pointer ends up NULL and its cleanup path stays correct. */
static void *tool_grow(void *buf, size_t *have, size_t want, size_t elem)
{
    void *p;

    if (*have >= want)
        return buf;

    p = realloc(buf, want * elem);
    if (!p) {
        free(buf);
        *have = 0;
        return NULL;
    }

    *have = want;

    return p;
}

static void tool_buffers_free(struct tool_buffers *b)
{
    free(b->mosaic);
    free(b->half);
    free(b->rgba);
    free(b->channel);
    *b = (struct tool_buffers) { 0 };
}

/* craw_testdev_image_upload() and craw_noise_estimate() both want tightly
 * packed rows and the decoder pads its own, so the mosaic is repacked. */
static int tool_pack_mosaic(struct tool_buffers *b, const AVFrame *frame)
{
    uint32_t w = (uint32_t)frame->width, h = (uint32_t)frame->height;

    b->mosaic = tool_grow(b->mosaic, &b->mosaic_samples, (size_t)w * h,
                          sizeof(*b->mosaic));
    if (!b->mosaic)
        return CRAW_ERROR(ENOMEM);

    for (uint32_t y = 0; y < h; y++)
        memcpy(b->mosaic + (size_t)y * w,
               frame->data[0] + (ptrdiff_t)y * frame->linesize[0],
               (size_t)w * sizeof(*b->mosaic));

    return 0;
}

static int tool_upload_frame(struct tool_gpu *g, struct tool_buffers *b,
                             const AVFrame *frame)
{
    size_t npix = (size_t)frame->width * frame->height;
    int ret;

    ret = tool_pack_mosaic(b, frame);
    if (ret < 0)
        return ret;

    ret = craw_testdev_image_upload(&g->td, &g->src, b->mosaic,
                                    npix * sizeof(*b->mosaic));
    if (ret < 0)
        fprintf(stderr, "upload failed: %d\n", ret);

    return ret;
}

/* ------------------------------------------------------- noise estimate */

/* Per-frame fits, so that a clip of many frames can be summarized by the
 * median rather than by whichever frame happened to be last. */
struct tool_noise_run {
    float *a, *b;
    size_t nb, cap;
};

static int tool_noise_run_push(struct tool_noise_run *r, float a, float b)
{
    if (r->nb == r->cap) {
        size_t cap = r->cap ? r->cap * 2 : 64;
        float *na = realloc(r->a, cap * sizeof(*na));
        float *nb = na ? realloc(r->b, cap * sizeof(*nb)) : NULL;

        if (!nb) {
            free(na);
            return CRAW_ERROR(ENOMEM);
        }
        r->a = na;
        r->b = nb;
        r->cap = cap;
    }

    r->a[r->nb] = a;
    r->b[r->nb] = b;
    r->nb++;

    return 0;
}

static int tool_float_cmp(const void *x, const void *y)
{
    float a = *(const float *)x, c = *(const float *)y;
    return a < c ? -1 : (a > c ? 1 : 0);
}

static void tool_noise_run_report(struct tool_noise_run *r, const CRAWProfile *prof)
{
    float range;

    if (!r->nb)
        return;

    qsort(r->a, r->nb, sizeof(*r->a), tool_float_cmp);
    qsort(r->b, r->nb, sizeof(*r->b), tool_float_cmp);

    range = prof->white_level[0] - prof->black_level[0];

    printf("\n%zu frames, median of the per frame fits:\n", r->nb);
    printf("  noise %.0f %.6g %.6g   # profile table line, iso a b\n",
           (double)prof->as_shot_ei, (double)r->a[r->nb / 2], (double)r->b[r->nb / 2]);
    printf("  spread of a: %.4e .. %.4e (%.1f%% of the median)\n",
           (double)r->a[0], (double)r->a[r->nb - 1],
           r->a[r->nb / 2] > 0.0f ?
               100.0 * (r->a[r->nb - 1] - r->a[0]) / r->a[r->nb / 2] : 0.0);
    printf("  in 16-bit code values: a = %.4f per code value, read noise = %.2f\n",
           (double)(r->a[r->nb / 2] * range * 65535.0f),
           sqrt((double)r->b[r->nb / 2]) * (double)range * 65535.0);
}

/* ------------------------------------------------------ per pass times */

/* --bench: every pass's GPU time frame by frame, and the whole recording's,
 * for the table at the end. A pass that ran on only some frames (the lens
 * warp on a clip whose first frames carry no record) is averaged over the
 * frames it ran on. */
struct tool_series {
    double *v;
    size_t nb, cap;
};

struct tool_pass {
    char name[32];
    uint32_t count;               /* dispatches in the last frame it ran on */
    struct tool_series ms;
};

struct tool_prof {
    struct tool_pass *pass;
    size_t nb_passes;
    struct tool_series sum;       /* the passes, summed per frame */
    struct tool_series whole;     /* the recording, barriers and all */
};

static int tool_series_push(struct tool_series *s, double v)
{
    if (s->nb == s->cap) {
        size_t cap = s->cap ? 2 * s->cap : 64;
        double *nv = realloc(s->v, cap * sizeof(*nv));

        if (!nv)
            return CRAW_ERROR(ENOMEM);
        s->v = nv;
        s->cap = cap;
    }
    s->v[s->nb++] = v;

    return 0;
}

static int tool_double_cmp(const void *x, const void *y)
{
    double a = *(const double *)x, c = *(const double *)y;
    return a < c ? -1 : (a > c ? 1 : 0);
}

/* Sorts the series in place, which the table is the last reader of. */
static void tool_series_stats(struct tool_series *s, double *avg, double *median)
{
    double sum = 0.0;

    *avg = *median = 0.0;
    if (!s->nb)
        return;

    for (size_t i = 0; i < s->nb; i++)
        sum += s->v[i];
    qsort(s->v, s->nb, sizeof(*s->v), tool_double_cmp);

    *avg = sum / (double)s->nb;
    *median = s->nb & 1 ? s->v[s->nb / 2] : 0.5 * (s->v[s->nb / 2 - 1] + s->v[s->nb / 2]);
}

static int tool_prof_push(struct tool_prof *pr, CRAWPipeline *pipe, double whole_ms)
{
    CRAWStageTime st[64];
    double sum = 0.0;
    int n, ret;

    n = craw_pipeline_get_stage_times(pipe, st, CRAW_ARRAY_ELEMS(st));
    if (n < 0)
        return n;

    for (int i = 0; i < CRAW_MIN(n, (int)CRAW_ARRAY_ELEMS(st)); i++) {
        size_t j;

        for (j = 0; j < pr->nb_passes; j++)
            if (!strcmp(pr->pass[j].name, st[i].name))
                break;
        if (j == pr->nb_passes) {
            struct tool_pass *np = realloc(pr->pass, (j + 1) * sizeof(*np));

            if (!np)
                return CRAW_ERROR(ENOMEM);
            pr->pass = np;
            pr->pass[j] = (struct tool_pass) { 0 };
            snprintf(pr->pass[j].name, sizeof(pr->pass[j].name), "%s", st[i].name);
            pr->nb_passes++;
        }

        pr->pass[j].count = st[i].count;
        ret = tool_series_push(&pr->pass[j].ms, st[i].ms);
        if (ret < 0)
            return ret;
        sum += st[i].ms;
    }

    ret = tool_series_push(&pr->sum, sum);
    if (ret < 0)
        return ret;

    return tool_series_push(&pr->whole, whole_ms);
}

static void tool_prof_report(struct tool_prof *pr)
{
    double avg, med;

    if (!pr->sum.nb)
        return;

    printf("\nGPU time per pass over %zu frames, each run cold: the software decode\n"
           "idles the GPU between frames, so a warm player's frames are faster.\n",
           pr->sum.nb);
    printf("  %-20s %10s %10s %10s\n", "pass", "dispatches", "avg ms", "median ms");
    for (size_t i = 0; i < pr->nb_passes; i++) {
        tool_series_stats(&pr->pass[i].ms, &avg, &med);
        printf("  %-20s %10u %10.3f %10.3f\n", pr->pass[i].name, pr->pass[i].count, avg, med);
    }
    tool_series_stats(&pr->sum, &avg, &med);
    printf("  %-20s %10s %10.3f %10.3f\n", "sum of the passes", "", avg, med);
    tool_series_stats(&pr->whole, &avg, &med);
    printf("  %-20s %10s %10.3f %10.3f\n", "whole recording", "", avg, med);
}

static void tool_prof_free(struct tool_prof *pr)
{
    for (size_t i = 0; i < pr->nb_passes; i++)
        free(pr->pass[i].ms.v);
    free(pr->pass);
    free(pr->sum.v);
    free(pr->whole.v);
    *pr = (struct tool_prof) { 0 };
}

/* Brings the output image back as interleaved float RGBA, decoding fp16 on
 * the way unless the pipeline was asked for fp32 in the first place. */
static int tool_download_frame(struct tool_gpu *g, struct tool_buffers *b, bool float32)
{
    size_t nval = (size_t)g->width * g->height * 4;
    int ret;

    b->rgba = tool_grow(b->rgba, &b->rgba_values, nval, sizeof(*b->rgba));
    if (!b->rgba)
        return CRAW_ERROR(ENOMEM);

    if (float32) {
        ret = craw_testdev_image_download(&g->td, &g->dst, b->rgba,
                                          nval * sizeof(*b->rgba));
    } else {
        b->half = tool_grow(b->half, &b->half_values, nval, sizeof(*b->half));
        if (!b->half)
            return CRAW_ERROR(ENOMEM);

        ret = craw_testdev_image_download(&g->td, &g->dst, b->half,
                                          nval * sizeof(*b->half));
        if (ret >= 0)
            for (size_t i = 0; i < nval; i++)
                b->rgba[i] = tool_half_to_float(b->half[i]);
    }

    if (ret < 0)
        fprintf(stderr, "download failed: %d\n", ret);

    return ret;
}

/* ----------------------------------------------------------------- awb */

struct tool_awb {
    CRAWAwbSmoother *smoother;
    CRAWAwbParams params;
    CRAWIlluminant illum;
    bool have_illum;
};

/* Reads back the histogram the frame that was just submitted built, turns
 * it into an illuminant and feeds the temporal model. What comes out is
 * applied to the next frame, not to this one; see tool_usage(). */
static int tool_awb_step(struct tool_awb *awb, CRAWPipeline *pipe,
                         const CRAWProfile *prof, double pts_s,
                         int64_t index, bool verbose)
{
    CRAWAwbHistogram hist;
    CRAWAwbEstimate est;
    CRAWIlluminant out;
    int ret;

    ret = craw_pipeline_get_awb_histogram(pipe, &hist);
    if (ret < 0) {
        fprintf(stderr, "craw_pipeline_get_awb_histogram failed: %d\n", ret);
        return ret;
    }

    ret = craw_awb_estimate(prof, &hist, &awb->params, &est);
    if (ret < 0) {
        fprintf(stderr, "craw_awb_estimate failed: %d\n", ret);
        return ret;
    }

    ret = craw_awb_smoother_push(awb->smoother, pts_s, &est, false, &out);
    if (ret < 0 && ret != CRAW_ERROR(EAGAIN)) {
        fprintf(stderr, "craw_awb_smoother_push failed: %d\n", ret);
        return ret;
    }
    /* EAGAIN is "no track yet and this estimate cannot start one", so the
     * next frame stays on whatever the last good one left, or as shot. */
    if (ret >= 0) {
        awb->illum = out;
        awb->have_illum = true;
    }

    if (verbose) {
        char track[64];

        if (awb->have_illum)
            snprintf(track, sizeof(track), "cct %6.0f tint %+.4f",
                     awb->illum.cct, awb->illum.tint);
        else
            snprintf(track, sizeof(track), "none yet, staying as shot");

        fprintf(stderr, "frame %5" PRId64 " awb: est %s cct %6.0f tint %+.4f conf %.2f "
                        "zone %.3f spread %.1f mired -> smoothed %s\n",
                index, est.valid ? "ok " : "bad", est.illuminant.cct,
                est.illuminant.tint, est.confidence, est.in_zone_fraction,
                est.spread_mired, track);

        /* The two readings behind that number. A light source in frame wins
         * over the surfaces it lights, which is the right answer and a very
         * surprising one to read without the breakdown. */
        fprintf(stderr, "%14s from %s: bright %s cct %6.0f zone %.3f spread %.1f, "
                        "surfaces %s cct %6.0f zone %.3f spread %.1f\n", "",
                est.source == CRAW_AWB_SOURCE_BRIGHT ? "bright" :
                est.source == CRAW_AWB_SOURCE_ZONE ? "surfaces" : "nothing",
                est.bright.valid ? "ok " : "bad", est.bright.illuminant.cct,
                est.bright.in_zone_fraction, est.bright.spread_mired,
                est.zone.valid ? "ok " : "bad", est.zone.illuminant.cct,
                est.zone.in_zone_fraction, est.zone.spread_mired);
    }

    return 0;
}

/* ------------------------------------------------------------- profile */

/* Exposure indices from the container, the way Apple's decoder reads them:
 * the as-shot ISO from the SMPTE RDD 18 camera metadata, the working index
 * from Apple's own EXIF-style key. Either may be missing. */
static void tool_read_exposure_metadata(AVFormatContext *fmt, int stream_index,
                                        float *as_shot_ei, float *apple_ei)
{
    AVDictionary *const dicts[2] = { fmt->metadata, fmt->streams[stream_index]->metadata };

    *as_shot_ei = 0.0f;
    *apple_ei = 0.0f;

    for (int i = 0; i < 2; i++) {
        const AVDictionaryEntry *e;

        e = av_dict_get(dicts[i], "org.smpte.rdd18.camera.isosensitivity", NULL, 0);
        if (e && !(*as_shot_ei > 0.0f))
            *as_shot_ei = (float)atof(e->value);
        e = av_dict_get(dicts[i], "com.apple.proapps.exif.{Exif}.ExposureIndex", NULL, 0);
        if (e && !(*apple_ei > 0.0f))
            *apple_ei = (float)atof(e->value);
    }
}

/* The camera model hides in the container metadata under a different key in
 * every muxer's idea of QuickTime; rather than guess, look at every value. */
static const char *tool_detect_profile(AVFormatContext *fmt, int stream_index,
                                       bool verbose, const char **match_key)
{
    AVDictionary *const dicts[2] = { fmt->metadata, fmt->streams[stream_index]->metadata };
    const char *const labels[2] = { "container", "stream" };
    const char *found = NULL;

    for (int i = 0; i < 2; i++) {
        const AVDictionaryEntry *e = NULL;

        while ((e = av_dict_iterate(dicts[i], e))) {
            const char *name;

            if (verbose) {
                /* The Panasonic clip metadata is a whole XML document. */
                int len = (int)strlen(e->value);
                fprintf(stderr, "  %s metadata: %s = %.*s%s\n", labels[i], e->key,
                        CRAW_MIN(len, 120), e->value, len > 120 ? " [...]" : "");
            }

            name = craw_tool_profile_detect(e->value);
            if (name && !found) {
                found = name;
                *match_key = e->key;
            }
        }
    }

    return found;
}

/* A still's container says nothing - an RW2 is its own container, and the
 * file's tags are the frame's - so the camera is looked for in the Exif IFD
 * the decoder attaches to the first frame, the same way: every value. */
static const char *tool_detect_frame_profile(const AVFrame *frame, bool verbose,
                                             const char **match_key)
{
    const AVDictionaryEntry *e = NULL;
    const char *found = NULL;

    while ((e = av_dict_iterate(frame->metadata, e))) {
        const char *name;

        if (verbose) {
            int len = (int)strlen(e->value);
            fprintf(stderr, "  frame metadata: %s = %.*s%s\n", e->key,
                    CRAW_MIN(len, 120), e->value, len > 120 ? " [...]" : "");
        }

        name = craw_tool_profile_detect(e->value);
        if (name && !found) {
            found = name;
            *match_key = e->key;
        }
    }

    return found;
}

/* The rest of the camera's profile once its built-in tables, if any, are
 * in: a --tables file on top, and the check that a white balance other
 * than as shot has a table to work from. Run before the first frame when
 * the container named the camera, and at the first frame when only the
 * frame could. */
static int tool_base_finish(const struct tool_cfg *cfg, CRAWProfile *base)
{
    int ret;

    /* The file wins over anything built in: it is how a new camera gets
     * tried without a rebuild. */
    if (cfg->tables) {
        ret = craw_tool_profile_load_tables(base, cfg->tables);
        if (ret < 0)
            return ret;
        if (cfg->verbose)
            fprintf(stderr, "loaded %u wb and %u matrix knots from %s\n",
                    base->nb_wb_knots, base->nb_matrix_knots, cfg->tables);
    }

    if (cfg->wb != TOOL_WB_AS_SHOT && !base->nb_wb_knots) {
        fprintf(stderr, "--wb %s needs the camera's per-CCT white balance table, and "
                        "none was found.\nPass --profile NAME (have: %s) or --tables FILE.\n",
                cfg->wb == TOOL_WB_AUTO ? "auto" : "CCT", craw_tool_profile_builtin_list());
        return CRAW_ERROR(ENOENT);
    }

    return 0;
}

/* -------------------------------------------------------------- decode */

/* How many frames the stream holds, from its own count or else from its
 * duration and rate; 0 when neither says. An RW2 says 1 the second way. */
static int64_t tool_stream_frames(const AVStream *st)
{
    AVRational rate = st->avg_frame_rate.num ? st->avg_frame_rate : st->r_frame_rate;

    if (st->nb_frames > 0)
        return st->nb_frames;
    if (st->duration > 0 && rate.num > 0 && rate.den > 0)
        return av_rescale_q(st->duration, st->time_base, av_inv_q(rate));

    return 0;
}

/* -o without a frame index names one file, which every frame after the
 * first would overwrite: only a run that writes one frame - a still, or
 * --frames 1 - may use it. */
static int tool_check_single(const struct tool_cfg *cfg, const AVStream *st)
{
    int64_t n = tool_stream_frames(st);

    if (!cfg->pattern || cfg->pattern_indexed || cfg->frames == 1 || n == 1)
        return 0;

    fprintf(stderr, "-o: `%s` has no printf frame index, and this clip has %s%" PRId64
                    " frames; name one (e.g. out_%%05d%s) or pass --frames 1\n",
            cfg->pattern, n ? "" : "an unknown number of ", n,
            strrchr(cfg->pattern, '.') ? strrchr(cfg->pattern, '.') : "");

    return CRAW_ERROR(EINVAL);
}

/* The lens record travels with the packet, not with the frame: it lives in
 * the frame's vendor metadata, which the decoder does not expose, so it is
 * read off the packet before the packet is handed over and matched back to
 * the picture by presentation timestamp. ProRes RAW is all intra and never
 * reorders, so in practice the match is always the oldest entry; matching
 * by pts anyway costs nothing and stops a future reordering decoder from
 * silently applying the wrong lens state to a frame. */
#define TOOL_LENS_RING 8

struct tool_lens_slot {
    int64_t pts;
    CRAWLensCorrection lc;
    bool used;
};

struct tool_decoder {
    AVFormatContext *fmt;
    AVCodecContext *dec;
    AVPacket *pkt;
    int stream_index;
    bool draining;

    /* Lens records of the packets in flight. */
    struct tool_lens_slot lens[TOOL_LENS_RING];
    uint64_t lens_pushed;
    bool lens_scan;       /* look for records at all */
    bool lens_any;        /* at least one packet carried one */
    bool lens_none;       /* at least one packet did not */
};

/* Reads the record off a packet and files it under the packet's pts. */
static void tool_lens_push(struct tool_decoder *d, const AVPacket *pkt)
{
    struct tool_lens_slot *sl;

    if (!d->lens_scan)
        return;

    sl = &d->lens[d->lens_pushed++ % TOOL_LENS_RING];
    sl->pts = pkt->pts;
    sl->used = craw_lens_from_prores_raw(pkt->data, (size_t)pkt->size, &sl->lc) == 0;

    if (sl->used)
        d->lens_any = true;
    else
        d->lens_none = true;
}

/* The record filed under `pts`, or an invalid one when this frame had
 * none. The slot is cleared so that a pts the container repeats cannot
 * hand the same record out twice. */
static void tool_lens_take(struct tool_decoder *d, int64_t pts,
                           CRAWLensCorrection *out)
{
    *out = (CRAWLensCorrection) { 0 };

    for (uint32_t i = 0; i < TOOL_LENS_RING; i++) {
        struct tool_lens_slot *sl = &d->lens[i];

        if (sl->used && sl->pts == pts) {
            *out = sl->lc;
            sl->used = false;
            return;
        }
    }
}

static void tool_decoder_close(struct tool_decoder *d)
{
    av_packet_free(&d->pkt);
    avcodec_free_context(&d->dec);
    avformat_close_input(&d->fmt);
}

static int tool_decoder_open(struct tool_decoder *d, const char *path, bool verbose)
{
    const AVCodec *codec;
    int ret;

    ret = avformat_open_input(&d->fmt, path, NULL, NULL);
    if (ret < 0) {
        fprintf(stderr, "%s: %s\n", path, av_err2str(ret));
        return CRAW_ERROR(EIO);
    }

    ret = avformat_find_stream_info(d->fmt, NULL);
    if (ret < 0) {
        fprintf(stderr, "%s: cannot read the stream info: %s\n", path, av_err2str(ret));
        return CRAW_ERROR(EIO);
    }

    d->stream_index = -1;
    for (unsigned i = 0; i < d->fmt->nb_streams; i++) {
        if (d->fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            d->stream_index = (int)i;
            break;
        }
    }
    if (d->stream_index < 0) {
        fprintf(stderr, "%s: no video stream\n", path);
        return CRAW_ERROR(EINVAL);
    }

    codec = avcodec_find_decoder(d->fmt->streams[d->stream_index]->codecpar->codec_id);
    if (!codec) {
        fprintf(stderr, "%s: no decoder for %s\n", path,
                avcodec_get_name(d->fmt->streams[d->stream_index]->codecpar->codec_id));
        return CRAW_ERROR(ENOSYS);
    }

    d->dec = avcodec_alloc_context3(codec);
    if (!d->dec)
        return CRAW_ERROR(ENOMEM);

    ret = avcodec_parameters_to_context(d->dec, d->fmt->streams[d->stream_index]->codecpar);
    if (ret < 0)
        return CRAW_ERROR(EINVAL);

    /* Software decode only in this version; a Vulkan hwaccel would hand the
     * pipeline its input image directly and skip the round trip entirely. */
    if (codec->id == AV_CODEC_ID_PRORES_RAW) {
        /* One thread: ProRes RAW frame threading keeps a 35 MB Bayer frame
         * in flight per thread, which on a many-core machine is gigabytes
         * for a tool that hands every frame to the GPU anyway. */
        d->dec->thread_count = 1;
    } else {
        /* A still is one packet. Frame threads would get nothing out of it
         * but a picture held back until the stream ends; slice threads
         * split that one picture - an RW2's stripes - across the cores and
         * keep nothing extra in flight. 0.11 s for the 24 MP _1000406.RW2
         * against 0.15 s on one thread, 0.32 against 0.55 s at 96 MP. */
        d->dec->thread_type = FF_THREAD_SLICE;
        d->dec->thread_count = 0;
    }

    /* ProRes RAW carries a crop in the frame header (the Sony FX6 clips
     * mark 3840x2160 inside a 3872x2192 readout), and so does an RW2 (the
     * DC-S1M2's 6004x4000 picture in a 6008x4008 readout), and two things
     * are wrong with letting libavutil apply it. It refuses to, for one:
     * av_frame_apply_cropping() compares the number of low zero bits of
     * crop_left with that of the resulting byte offset and returns
     * AVERROR_BUG when the crop looks less aligned than the offset, which
     * for two bytes a sample it always does (16 samples is 32 bytes, 4
     * against 5), so a clip like that would not decode at all. And the
     * sensels outside the crop are real picture on this camera, which the
     * pipeline should develop and the user should be able to look at. So
     * the decoder is told not to crop: the tool sees the whole coded frame
     * with all four crop fields, develops every sensel, and cuts the
     * output to the marked picture unless --crop says otherwise. */
    d->dec->apply_cropping = 0;

    ret = avcodec_open2(d->dec, codec, NULL);
    if (ret < 0) {
        fprintf(stderr, "%s: cannot open the %s decoder: %s\n", path, codec->name,
                av_err2str(ret));
        return CRAW_ERROR(EINVAL);
    }

    d->pkt = av_packet_alloc();
    if (!d->pkt)
        return CRAW_ERROR(ENOMEM);

    if (verbose)
        fprintf(stderr, "input %s: stream %d, %s, %dx%d, %d threads\n", path,
                d->stream_index, codec->name, d->dec->width, d->dec->height,
                d->dec->thread_count);

    return 0;
}

/* 0: a frame, 1: end of stream, negative: an error. */
static int tool_decoder_next(struct tool_decoder *d, AVFrame *frame)
{
    int ret;

    while (1) {
        ret = avcodec_receive_frame(d->dec, frame);
        if (ret >= 0)
            return 0;
        if (ret == AVERROR_EOF)
            return 1;
        if (ret != AVERROR(EAGAIN)) {
            fprintf(stderr, "decoding failed: %s\n", av_err2str(ret));
            return CRAW_ERROR(EINVAL);
        }
        if (d->draining)
            return 1;

        ret = av_read_frame(d->fmt, d->pkt);
        if (ret == AVERROR_EOF) {
            avcodec_send_packet(d->dec, NULL);
            d->draining = true;
            continue;
        }
        if (ret < 0) {
            fprintf(stderr, "reading failed: %s\n", av_err2str(ret));
            return CRAW_ERROR(EIO);
        }

        if (d->pkt->stream_index != d->stream_index) {
            av_packet_unref(d->pkt);
            continue;
        }

        tool_lens_push(d, d->pkt);

        ret = avcodec_send_packet(d->dec, d->pkt);
        av_packet_unref(d->pkt);
        if (ret < 0) {
            fprintf(stderr, "feeding the decoder failed: %s\n", av_err2str(ret));
            return CRAW_ERROR(EINVAL);
        }
    }
}

/* ------------------------------------------------------------- logging */

static void tool_log_cb(void *opaque, enum CRAWLogLevel level, const char *fmt,
                        va_list args)
{
    (void)opaque;
    (void)level;
    vfprintf(stderr, fmt, args);
}

/* ------------------------------------------------------- option parsing */

static int tool_parse_wb(struct tool_cfg *cfg, const char *arg)
{
    char *end;
    double cct, tint = 0.0;

    if (!strcmp(arg, "as-shot")) {
        cfg->wb = TOOL_WB_AS_SHOT;
        return 0;
    }
    if (!strcmp(arg, "auto")) {
        cfg->wb = TOOL_WB_AUTO;
        return 0;
    }

    cct = strtod(arg, &end);
    if (end == arg || (*end && *end != ':')) {
        fprintf(stderr, "--wb: expected as-shot, auto or CCT[:TINT], got `%s`\n", arg);
        return CRAW_ERROR(EINVAL);
    }
    if (*end == ':') {
        const char *tp = end + 1;
        tint = strtod(tp, &end);
        if (end == tp || *end) {
            fprintf(stderr, "--wb: cannot parse the tint in `%s`\n", arg);
            return CRAW_ERROR(EINVAL);
        }
    }
    if (cct <= 0.0) {
        fprintf(stderr, "--wb: a colour temperature of %g K makes no sense\n", cct);
        return CRAW_ERROR(EINVAL);
    }

    cfg->wb = TOOL_WB_FIXED;
    cfg->fixed_illum = (CRAWIlluminant) { .cct = (float)cct, .tint = (float)tint };

    return 0;
}

/* `X,Y,W,H` in output pixels. */
/* --sharpen RADIUS[,ITERATIONS[,CONTRAST]]: the two left out keep the
 * defaults tool_parse_args() set. */
static int tool_parse_sharpen(struct tool_cfg *cfg, const char *arg)
{
    const char *at = arg;
    double v[3];
    char *end;
    int n = 0;

    while (n < 3) {
        v[n] = strtod(at, &end);
        if (end == at)
            break;
        n++;
        if (*end != ',')
            break;
        at = end + 1;
    }
    if (!n || *end) {
        fprintf(stderr, "--sharpen: expected RADIUS[,ITERATIONS[,CONTRAST]], got `%s`\n", arg);
        return CRAW_ERROR(EINVAL);
    }
    if (!(v[0] >= 0.0 && v[0] <= CRAW_SHARPEN_MAX_RADIUS)) {
        fprintf(stderr, "--sharpen: RADIUS must be 0 to %g, got %g\n",
                (double)CRAW_SHARPEN_MAX_RADIUS, v[0]);
        return CRAW_ERROR(EINVAL);
    }
    cfg->sharpen_radius = (float)v[0];
    if (n > 1) {
        if (!(v[1] >= 0.0 && v[1] <= CRAW_SHARPEN_MAX_ITERATIONS) || v[1] != (double)(int)v[1]) {
            fprintf(stderr, "--sharpen: ITERATIONS must be 0 to %d, got %g\n",
                    CRAW_SHARPEN_MAX_ITERATIONS, v[1]);
            return CRAW_ERROR(EINVAL);
        }
        cfg->sharpen_iterations = (uint32_t)v[1];
    }
    if (n > 2) {
        if (!(v[2] >= 0.0 && v[2] < 1e6)) {
            fprintf(stderr, "--sharpen: CONTRAST must be 0 or more, got %g\n", v[2]);
            return CRAW_ERROR(EINVAL);
        }
        cfg->sharpen_contrast = (float)v[2];
    }

    return 0;
}

static int tool_parse_crop(struct tool_cfg *cfg, const char *arg)
{
    double v[4];
    const char *p = arg;
    char *end;

    for (int i = 0; i < 4; i++) {
        v[i] = strtod(p, &end);
        if (end == p || (i < 3 && *end != ',') || (i == 3 && *end)) {
            fprintf(stderr, "--crop: expected X,Y,W,H, got `%s`\n", arg);
            return CRAW_ERROR(EINVAL);
        }
        p = end + 1;
    }

    if (v[0] < 0.0 || v[1] < 0.0 || v[2] < 1.0 || v[3] < 1.0) {
        fprintf(stderr, "--crop: the offset must not be negative and the size must be "
                        "at least one pixel\n");
        return CRAW_ERROR(EINVAL);
    }

    cfg->crop_x = (int64_t)v[0];
    cfg->crop_y = (int64_t)v[1];
    cfg->crop_w = (int64_t)v[2];
    cfg->crop_h = (int64_t)v[3];

    return 0;
}

/* The output crop for a frame: --crop when it was given, otherwise the
 * picture the frame header marks, if it marks one. The decoder is told not
 * to apply that crop (tool_decoder_open()), so the pipeline develops the
 * whole readout and this only decides what is written out; `--crop
 * 0,0,W,H` of the coded size is how to keep all of it. Returns false when
 * the whole frame is wanted. */
static bool tool_frame_crop(const struct tool_cfg *cfg, const AVFrame *frame,
                            int64_t crop[4])
{
    if (cfg->crop_w) {
        crop[0] = cfg->crop_x;
        crop[1] = cfg->crop_y;
        crop[2] = cfg->crop_w;
        crop[3] = cfg->crop_h;
        return true;
    }

    if (!frame->crop_left && !frame->crop_right &&
        !frame->crop_top && !frame->crop_bottom)
        return false;

    crop[0] = (int64_t)frame->crop_left;
    crop[1] = (int64_t)frame->crop_top;
    crop[2] = (int64_t)frame->width - (int64_t)frame->crop_left -
              (int64_t)frame->crop_right;
    crop[3] = (int64_t)frame->height - (int64_t)frame->crop_top -
              (int64_t)frame->crop_bottom;

    return crop[2] >= 1 && crop[3] >= 1;
}

/* Compacts the crop to the front of `rgba` and returns its size, clamped
 * to the frame. Done on the host after the download: the pipeline has no
 * notion of a region of interest and every stage's stencil should see the
 * same neighbours it would in a full frame. */
static void tool_crop(const int64_t crop[4], float *rgba, uint32_t *w, uint32_t *h)
{
    int64_t x0 = CRAW_MIN(crop[0], (int64_t)*w - 1);
    int64_t y0 = CRAW_MIN(crop[1], (int64_t)*h - 1);
    int64_t cw = CRAW_MIN(crop[2], (int64_t)*w - x0);
    int64_t ch = CRAW_MIN(crop[3], (int64_t)*h - y0);

    for (int64_t y = 0; y < ch; y++)
        memmove(rgba + y * cw * 4, rgba + ((y0 + y) * (int64_t)*w + x0) * 4,
                (size_t)cw * 4 * sizeof(*rgba));

    *w = (uint32_t)cw;
    *h = (uint32_t)ch;
}

/* `A,B` or `A B`: the two terms of variance = A * u + B. */
static int tool_parse_noise(struct tool_cfg *cfg, const char *arg)
{
    char *end;
    double a, b;

    a = strtod(arg, &end);
    if (end == arg || (*end != ',' && *end != ' ')) {
        fprintf(stderr, "--noise: expected A,B, got `%s`\n", arg);
        return CRAW_ERROR(EINVAL);
    }
    arg = end + 1;
    b = strtod(arg, &end);
    if (end == arg || *end) {
        fprintf(stderr, "--noise: cannot parse B in `%s`\n", arg);
        return CRAW_ERROR(EINVAL);
    }
    if (!(a > 0.0) || b < 0.0) {
        fprintf(stderr, "--noise: A must be positive and B must not be negative\n");
        return CRAW_ERROR(EINVAL);
    }

    cfg->noise = (CRAWNoiseModel) { .a = (float)a, .b = (float)b };

    return 0;
}

static int tool_parse_double(const char *opt, const char *arg, double *out)
{
    char *end;

    *out = strtod(arg, &end);
    if (end == arg || *end) {
        fprintf(stderr, "%s: `%s` is not a number\n", opt, arg);
        return CRAW_ERROR(EINVAL);
    }

    return 0;
}

enum {
    TOOL_OPT_FRAMES = 0x100,
    TOOL_OPT_START,
    TOOL_OPT_DEVICE,
    TOOL_OPT_DEBAYER,
    TOOL_OPT_SPACE,
    TOOL_OPT_ENCODE,
    TOOL_OPT_WB,
    TOOL_OPT_AMBIENCE,
    TOOL_OPT_EXPOSURE,
    TOOL_OPT_EI,
    TOOL_OPT_CLAMP_NEG,
    TOOL_OPT_NO_GAMUT,
    TOOL_OPT_FLOAT32,
    TOOL_OPT_BENCH,
    TOOL_OPT_STATS,
    TOOL_OPT_CROP,
    TOOL_OPT_DENOISE,
    TOOL_OPT_FALSE_COLOUR,
    TOOL_OPT_HIGHLIGHTS,
    TOOL_OPT_DEFRINGE,
    TOOL_OPT_SHARPEN,
    TOOL_OPT_NOISE,
    TOOL_OPT_NOISE_EST,
    TOOL_OPT_LENS,
    TOOL_OPT_TABLES,
    TOOL_OPT_PROFILE,
};

static int tool_parse_args(struct tool_cfg *cfg, int argc, char **argv)
{
    static const struct option opts[] = {
        { "frames",         required_argument, NULL, TOOL_OPT_FRAMES    },
        { "start",          required_argument, NULL, TOOL_OPT_START     },
        { "device",         required_argument, NULL, TOOL_OPT_DEVICE    },
        { "debayer",        required_argument, NULL, TOOL_OPT_DEBAYER   },
        { "space",          required_argument, NULL, TOOL_OPT_SPACE     },
        { "encode",         required_argument, NULL, TOOL_OPT_ENCODE    },
        { "wb",             required_argument, NULL, TOOL_OPT_WB        },
        { "awb-ambience",   required_argument, NULL, TOOL_OPT_AMBIENCE  },
        { "exposure",       required_argument, NULL, TOOL_OPT_EXPOSURE  },
        { "ei",             required_argument, NULL, TOOL_OPT_EI        },
        { "clamp-negative", no_argument,       NULL, TOOL_OPT_CLAMP_NEG },
        { "no-gamut-compression",
                            no_argument,       NULL, TOOL_OPT_NO_GAMUT  },
        { "float32",        no_argument,       NULL, TOOL_OPT_FLOAT32   },
        { "bench",          no_argument,       NULL, TOOL_OPT_BENCH     },
        { "stats",          no_argument,       NULL, TOOL_OPT_STATS     },
        { "crop",           required_argument, NULL, TOOL_OPT_CROP      },
        { "denoise",        required_argument, NULL, TOOL_OPT_DENOISE   },
        { "false-colour",   required_argument, NULL, TOOL_OPT_FALSE_COLOUR },
        { "highlights",     required_argument, NULL, TOOL_OPT_HIGHLIGHTS },
        { "defringe",       required_argument, NULL, TOOL_OPT_DEFRINGE },
        { "sharpen",        required_argument, NULL, TOOL_OPT_SHARPEN   },
        { "noise",          required_argument, NULL, TOOL_OPT_NOISE     },
        { "noise-estimate", no_argument,       NULL, TOOL_OPT_NOISE_EST },
        { "lens",           required_argument, NULL, TOOL_OPT_LENS      },
        { "tables",         required_argument, NULL, TOOL_OPT_TABLES    },
        { "profile",        required_argument, NULL, TOOL_OPT_PROFILE   },
        { "help",           no_argument,       NULL, 'h'                },
        { NULL,             0,                 NULL, 0                  },
    };
    double d;
    int v, c, ret;

    *cfg = (struct tool_cfg) {
        .frames   = -1,
        .debayer  = CRAW_DEBAYER_RCD,
        .space    = CRAW_CS_BT709_LINEAR,
        .encode   = TOOL_ENC_SDR,
        .wb       = TOOL_WB_AS_SHOT,
        /* A correction, like the lens: what clipped is rebuilt unless
         * --highlights off asks for the plain clip. */
        .highlights = true,
        /* On with it: the fringes it takes out are the ones the rebuilt
         * highlights stop hiding. */
        .defringe = 1.0f,
        /* Off, as in every raw converter; --sharpen RADIUS alone takes
         * these two. DESIGN.md 20.4. */
        .sharpen_iterations = 20,
        .sharpen_contrast = 1.5f,
    };

    while ((c = getopt_long(argc, argv, "o:vh", opts, NULL)) != -1) {
        switch (c) {
        case 'o':
            cfg->pattern = optarg;
            break;
        case 'v':
            cfg->verbose = true;
            break;
        case 'h':
            tool_usage(stdout);
            return 1;

        case TOOL_OPT_FRAMES:
            if ((ret = tool_parse_double("--frames", optarg, &d)) < 0)
                return ret;
            if (d < 1.0) {
                fprintf(stderr, "--frames: %g is not a frame count\n", d);
                return CRAW_ERROR(EINVAL);
            }
            cfg->frames = (int64_t)d;
            break;
        case TOOL_OPT_START:
            if ((ret = tool_parse_double("--start", optarg, &cfg->start)) < 0)
                return ret;
            break;
        case TOOL_OPT_DEVICE:
            cfg->device = optarg;
            break;
        case TOOL_OPT_DEBAYER:
            if ((ret = tool_lookup(tool_debayers, "--debayer", optarg, &v)) < 0)
                return ret;
            cfg->debayer = (enum CRAWDebayer)v;
            break;
        case TOOL_OPT_SPACE:
            if ((ret = tool_lookup(tool_spaces, "--space", optarg, &v)) < 0)
                return ret;
            cfg->space = (enum CRAWColorSpace)v;
            cfg->space_set = true;
            break;
        case TOOL_OPT_ENCODE:
            if ((ret = tool_lookup(tool_encodes, "--encode", optarg, &v)) < 0)
                return ret;
            cfg->encode = (enum tool_encode)v;
            cfg->encode_set = true;
            break;
        case TOOL_OPT_WB:
            if ((ret = tool_parse_wb(cfg, optarg)) < 0)
                return ret;
            break;
        case TOOL_OPT_AMBIENCE:
            if ((ret = tool_parse_double("--awb-ambience", optarg, &d)) < 0)
                return ret;
            if (d < 0.0 || d > 1.0) {
                fprintf(stderr, "--awb-ambience: %g is outside 0..1\n", d);
                return CRAW_ERROR(EINVAL);
            }
            cfg->awb_ambience = (float)d;
            break;
        case TOOL_OPT_EXPOSURE:
            if ((ret = tool_parse_double("--exposure", optarg, &d)) < 0)
                return ret;
            cfg->exposure = (float)d;
            break;
        case TOOL_OPT_EI:
            if ((ret = tool_parse_double("--ei", optarg, &d)) < 0)
                return ret;
            if (!(d > 0.0)) {
                fprintf(stderr, "--ei must be a positive exposure index\n");
                return CRAW_ERROR(EINVAL);
            }
            cfg->ei = (float)d;
            break;
        case TOOL_OPT_CLAMP_NEG:
            cfg->clamp_negative = true;
            break;
        case TOOL_OPT_NO_GAMUT:
            cfg->no_gamut_compression = true;
            break;
        case TOOL_OPT_FLOAT32:
            cfg->float32 = true;
            break;
        case TOOL_OPT_BENCH:
            cfg->bench = true;
            break;
        case TOOL_OPT_STATS:
            cfg->stats = true;
            break;
        case TOOL_OPT_CROP:
            if ((ret = tool_parse_crop(cfg, optarg)) < 0)
                return ret;
            break;
        case TOOL_OPT_DENOISE: {
            char *end;

            d = strtod(optarg, &end);
            if (end == optarg || d < 0.0 || (*end && *end != ',')) {
                fprintf(stderr, "--denoise: expected STRENGTH[,CHROMA[,TEMPORAL]], got `%s`\n",
                        optarg);
                return CRAW_ERROR(EINVAL);
            }
            cfg->denoise = (float)d;
            if (*end == ',') {
                const char *arg = end + 1;

                d = strtod(arg, &end);
                if (end == arg || (*end && *end != ',') || d < 0.0 || d > 1.0) {
                    fprintf(stderr, "--denoise: CHROMA must be 0 to 1, got `%s`\n", arg);
                    return CRAW_ERROR(EINVAL);
                }
                cfg->chroma = (float)d;
            }
            if (*end == ',') {
                const char *arg = end + 1;

                d = strtod(arg, &end);
                if (end == arg || *end || d < 0.0 || d > 1.0) {
                    fprintf(stderr, "--denoise: TEMPORAL must be 0 to 1, got `%s`\n", arg);
                    return CRAW_ERROR(EINVAL);
                }
                cfg->temporal = (float)d;
            }
            break;
        }
        case TOOL_OPT_FALSE_COLOUR:
            if ((ret = tool_parse_double("--false-colour", optarg, &d)) < 0)
                return ret;
            if (d < 0.0 || d > CRAW_FALSE_COLOR_MAX_PASSES || d != (double)(int)d) {
                fprintf(stderr, "--false-colour: expected 0 to %d passes, got %g\n",
                        CRAW_FALSE_COLOR_MAX_PASSES, d);
                return CRAW_ERROR(EINVAL);
            }
            cfg->false_color = (uint32_t)d;
            break;
        case TOOL_OPT_NOISE:
            if ((ret = tool_parse_noise(cfg, optarg)) < 0)
                return ret;
            break;
        case TOOL_OPT_NOISE_EST:
            cfg->noise_estimate = true;
            break;
        case TOOL_OPT_LENS:
            if ((ret = tool_lookup(tool_lenses, "--lens", optarg, &v)) < 0)
                return ret;
            cfg->lens = (enum tool_lens)v;
            break;
        case TOOL_OPT_DEFRINGE:
            if ((ret = tool_parse_double("--defringe", optarg, &d)) < 0)
                return ret;
            if (!(d >= 0.0 && d <= 1.0)) {
                fprintf(stderr, "--defringe: expected 0 to 1, got %g\n", d);
                return CRAW_ERROR(EINVAL);
            }
            cfg->defringe = (float)d;
            break;
        case TOOL_OPT_SHARPEN:
            if ((ret = tool_parse_sharpen(cfg, optarg)) < 0)
                return ret;
            break;
        case TOOL_OPT_HIGHLIGHTS:
            if ((ret = tool_lookup(tool_switches, "--highlights", optarg, &v)) < 0)
                return ret;
            cfg->highlights = v != 0;
            break;
        case TOOL_OPT_TABLES:
            cfg->tables = optarg;
            break;
        case TOOL_OPT_PROFILE:
            cfg->profile = optarg;
            break;

        default:
            tool_usage(stderr);
            return CRAW_ERROR(EINVAL);
        }
    }

    if (optind >= argc) {
        fprintf(stderr, "no input file\n");
        tool_usage(stderr);
        return CRAW_ERROR(EINVAL);
    }
    cfg->input = argv[optind++];
    if (optind < argc) {
        fprintf(stderr, "unexpected argument `%s`\n", argv[optind]);
        return CRAW_ERROR(EINVAL);
    }

    if (cfg->pattern) {
        if ((ret = tool_check_pattern(cfg->pattern)) < 0)
            return ret;
        cfg->pattern_indexed = ret == 1;

        cfg->writer = tool_writer_from_path(cfg->pattern);
        if (cfg->writer == TOOL_WRITER_NONE) {
            fprintf(stderr, "-o: `%s` has no .png, .tif, .ppm or .pfm extension\n",
                    cfg->pattern);
            return CRAW_ERROR(EINVAL);
        }

        /* A float file wants the pipeline's own numbers by default. */
        if (!cfg->encode_set && cfg->writer == TOOL_WRITER_PFM)
            cfg->encode = TOOL_ENC_LINEAR;
    }

    /* V-Log is defined on V-Gamut and nothing else. */
    if (cfg->encode == TOOL_ENC_VLOG && cfg->space != CRAW_CS_V_GAMUT_LINEAR) {
        if (cfg->space_set)
            fprintf(stderr, "warning: --encode vlog overrides --space %s with vgamut\n",
                    tool_space_name(cfg->space));
        cfg->space = CRAW_CS_V_GAMUT_LINEAR;
    }

    return 0;
}

/* ---------------------------------------------------------------- main */

int main(int argc, char **argv)
{
    struct tool_cfg cfg;
    struct tool_decoder dec = { .stream_index = -1 };
    struct tool_gpu gpu = { 0 };
    struct tool_buffers buf = { 0 };
    struct tool_awb awb = { 0 };
    struct tool_noise_run noise_run = { 0 };
    struct tool_prof prof_run = { 0 };
    CRAWContextOptions ctx_opts = { 0 };
    CRAWProfile base, prof;
    AVFrame *frame = NULL;
    AVRational time_base = { 1, 1 };
    double gpu_ms_total = 0.0, decode_ms_total = 0.0, decode_ms_warm = 0.0;
    float last_lens_dist = 0.0f;
    bool have_lens_report = false, warned_no_lens = false;
    /* The camera is to be found in the first frame's metadata rather than
     * the container's: a still. */
    bool detect_late = false;
    int64_t processed = 0;
    int ret;

    ret = tool_parse_args(&cfg, argc, argv);
    if (ret)
        return ret > 0 ? 0 : 1;

    /* --------------------------------------------------- the profile */

    craw_profile_init(&base);

    if (cfg.profile) {
        ret = craw_tool_profile_apply_builtin(&base, cfg.profile);
        if (ret < 0) {
            fprintf(stderr, "--profile: no built-in table set called `%s` (have: %s)\n",
                    cfg.profile, craw_tool_profile_builtin_list());
            goto fail;
        }
    }

    /* ---------------------------------------------------- the decoder */

    ret = tool_decoder_open(&dec, cfg.input, cfg.verbose);
    if (ret < 0)
        goto fail;

    time_base = dec.fmt->streams[dec.stream_index]->time_base;

    /* Reading the record is a few microseconds on a five megabyte packet
     * and it decodes nothing, but there is no point doing it at all when
     * the correction has been turned off. */
    dec.lens_scan = cfg.lens != TOOL_LENS_OFF &&
                    dec.fmt->streams[dec.stream_index]->codecpar->codec_id ==
                        AV_CODEC_ID_PRORES_RAW;

    ret = tool_check_single(&cfg, dec.fmt->streams[dec.stream_index]);
    if (ret < 0)
        goto fail;

    if (!cfg.profile) {
        const char *key = NULL;
        tool_read_exposure_metadata(dec.fmt, dec.stream_index, &cfg.meta_as_shot_ei, &cfg.meta_apple_ei);
        if (cfg.verbose)
            fprintf(stderr, "exposure metadata: as-shot ISO %.0f, EXIF exposure index %.0f (not applied), working %.0f\n",
                    cfg.meta_as_shot_ei, cfg.meta_apple_ei, cfg.ei > 0.0f ? cfg.ei : cfg.meta_as_shot_ei);
        const char *name = tool_detect_profile(dec.fmt, dec.stream_index,
                                               cfg.verbose, &key);
        if (name) {
            ret = craw_tool_profile_apply_builtin(&base, name);
            if (ret < 0)
                goto fail;
            if (cfg.verbose)
                fprintf(stderr, "detected `%s` from the metadata key %s\n", name, key);
        } else {
            /* Nothing in the container, which for a still is where nothing
             * ever is: the first frame's own metadata may yet name the
             * camera, and the rest of the profile waits for it. */
            detect_late = true;
        }
    }

    if (!detect_late) {
        ret = tool_base_finish(&cfg, &base);
        if (ret < 0)
            goto fail;
    }

    /* --------------------------------------------------------- seek */

    if (cfg.start > 0.0) {
        int64_t ts = (int64_t)llrint(cfg.start / av_q2d(time_base));

        ret = av_seek_frame(dec.fmt, dec.stream_index, ts, AVSEEK_FLAG_BACKWARD);
        if (ret < 0) {
            fprintf(stderr, "cannot seek to %g s: %s\n", cfg.start, av_err2str(ret));
            ret = CRAW_ERROR(EIO);
            goto fail;
        }
        avcodec_flush_buffers(dec.dec);
    }

    /* ---------------------------------------------------- the device */

    if (cfg.noise_estimate)
        goto have_device;

    ret = craw_testdev_open(&gpu.td, cfg.device, cfg.verbose);
    if (ret < 0) {
        fprintf(stderr, "no usable Vulkan device%s%s: %d\n", cfg.device ? " matching " : "",
                cfg.device ? cfg.device : "", ret);
        goto fail;
    }
    gpu.have_dev = true;

    ctx_opts.log_cb = tool_log_cb;
    ctx_opts.log_level = cfg.verbose ? CRAW_LOG_VERBOSE : CRAW_LOG_WARN;

    ret = craw_context_create(&gpu.ctx, &gpu.td.init, &ctx_opts);
    if (ret < 0) {
        fprintf(stderr, "craw_context_create failed: %d\n", ret);
        goto fail;
    }

    if (cfg.verbose)
        fprintf(stderr, "device %s, debayer %s, output %s %s, encode %s\n",
                gpu.td.name,
                cfg.debayer == CRAW_DEBAYER_BILINEAR ? "bilinear" :
                cfg.debayer == CRAW_DEBAYER_MHC ? "mhc" : "rcd",
                tool_space_name(cfg.space),
                cfg.float32 ? "RGBA32F" : "RGBA16F",
                tool_encodes[cfg.encode].name);

have_device:

    /* ------------------------------------------------- the smoother */

    craw_awb_params_init(&awb.params);
    awb.params.ambience = cfg.awb_ambience;

    if (cfg.wb == TOOL_WB_AUTO) {
        ret = craw_awb_smoother_create(NULL, &awb.smoother);
        if (ret < 0) {
            fprintf(stderr, "craw_awb_smoother_create failed: %d\n", ret);
            goto fail;
        }
    }

    /* ----------------------------------------------------- the loop */

    frame = av_frame_alloc();
    if (!frame) {
        ret = CRAW_ERROR(ENOMEM);
        goto fail;
    }

    while (cfg.frames < 0 || processed < cfg.frames) {
        CRAWFrameParams fp;
        CRAWLensCorrection lens;
        struct tool_record rec;
        double pts_s, decode_ms, gpu_ms = 0.0;
        double upload_ms, develop_ms, download_ms = 0.0, write_ms = 0.0;
        int64_t t0;
        int64_t crop[4];
        bool have_crop;
        uint32_t w, h;
        size_t npix;

        t0 = av_gettime_relative();
        ret = tool_decoder_next(&dec, frame);
        decode_ms = (double)(av_gettime_relative() - t0) / 1000.0;
        if (ret < 0)
            goto fail;
        if (ret > 0)
            break;

        pts_s = frame->pts == AV_NOPTS_VALUE ? 0.0 : (double)frame->pts * av_q2d(time_base);

        /* AVSEEK_FLAG_BACKWARD lands on or before the requested time. */
        if (cfg.start > 0.0 && pts_s < cfg.start - 1e-6) {
            av_frame_unref(frame);
            continue;
        }

        /* The upload copies the samples as they are, so they have to be in
         * the host's order; which of the four phases it is the profile
         * reads off the format, and the pipeline says whether it can
         * develop it. */
        if (frame->format != AV_PIX_FMT_BAYER_RGGB16LE &&
            frame->format != AV_PIX_FMT_BAYER_BGGR16LE &&
            frame->format != AV_PIX_FMT_BAYER_GRBG16LE &&
            frame->format != AV_PIX_FMT_BAYER_GBRG16LE) {
            const char *name = av_get_pix_fmt_name(frame->format);
            fprintf(stderr, "the decoder produced %s frames; cineraw needs a "
                            "16-bit little endian Bayer mosaic\n",
                    name ? name : "frames of an unknown format");
            ret = CRAW_ERROR(ENOTSUP);
            goto fail;
        }

        if (detect_late) {
            const char *key = NULL;
            const char *name = tool_detect_frame_profile(frame, cfg.verbose, &key);

            detect_late = false;
            if (name) {
                ret = craw_tool_profile_apply_builtin(&base, name);
                if (ret < 0)
                    goto fail;
                if (cfg.verbose)
                    fprintf(stderr, "detected `%s` from the frame's metadata key %s\n",
                            name, key);
            } else if (!cfg.tables) {
                fprintf(stderr, "warning: no camera recognized in the container or the "
                                "frame metadata, carrying on without per-CCT tables\n");
            }

            ret = tool_base_finish(&cfg, &base);
            if (ret < 0)
                goto fail;
        }

        /* The whole coded frame: nothing was cropped away, so the pipeline
         * sees every sensel the camera read out and the crop below only
         * picks what is written. */
        w = (uint32_t)frame->width;
        h = (uint32_t)frame->height;
        npix = (size_t)w * h;

        have_crop = tool_frame_crop(&cfg, frame, crop);

        prof = base;
        ret = craw_tool_profile_from_frame(&prof, frame, cfg.verbose && !processed);
        if (ret < 0)
            goto fail;

        /* Exposure model: gain * (working index / as-shot ISO) * 2^offset,
         * with the indices from the container when the decoder's side data
         * does not carry them. */
        if (!(prof.as_shot_ei > 0.0f))
            prof.as_shot_ei = cfg.meta_as_shot_ei;
        /* Apple's plug-in rates a camera original at its as-shot ISO and
         * never reads the EXIF ExposureIndex key, so that key is reported
         * but not applied. */
        if (cfg.ei > 0.0f)
            prof.exposure_index = cfg.ei;
        else if (!(prof.exposure_index > 0.0f))
            prof.exposure_index = prof.as_shot_ei;

        if (cfg.verbose && !processed) {
            fprintf(stderr, "frame: %ux%u coded, crop %zu left %zu right %zu top "
                            "%zu bottom, picture %zux%zu\n", w, h,
                    frame->crop_left, frame->crop_right, frame->crop_top,
                    frame->crop_bottom,
                    (size_t)w - frame->crop_left - frame->crop_right,
                    (size_t)h - frame->crop_top - frame->crop_bottom);
            if (have_crop)
                fprintf(stderr, "output crop: %" PRId64 "x%" PRId64 " at %" PRId64
                                ",%" PRId64 " (%s)\n", crop[2], crop[3], crop[0], crop[1],
                        cfg.crop_w ? "--crop" : "the frame's own");
        }

        if (cfg.verbose && !processed)
            fprintf(stderr, "profile `%s`: black %.6f white %.6f wb %.5f/%.5f "
                            "gain %.4f cct %.0f K, %u wb knots, %u matrix knots\n",
                    prof.name[0] ? prof.name : "(unnamed)", prof.black_level[0],
                    prof.white_level[0], prof.as_shot_wb[0], prof.as_shot_wb[2],
                    prof.gain, prof.as_shot_cct, prof.nb_wb_knots, prof.nb_matrix_knots);

        /* On a camera without per-CCT tables the frame header's matrix is
         * the only one there is, so -v shows it rather than only its
         * existence. */
        if (cfg.verbose && !processed)
            fprintf(stderr, "  as-shot camera matrix: [%7.4f %7.4f %7.4f] "
                            "[%7.4f %7.4f %7.4f] [%7.4f %7.4f %7.4f]\n",
                    prof.as_shot_matrix[0][0], prof.as_shot_matrix[0][1], prof.as_shot_matrix[0][2],
                    prof.as_shot_matrix[1][0], prof.as_shot_matrix[1][1], prof.as_shot_matrix[1][2],
                    prof.as_shot_matrix[2][0], prof.as_shot_matrix[2][1], prof.as_shot_matrix[2][2]);

        if (cfg.noise_estimate) {
            CRAWNoiseEstimate est;

            ret = tool_pack_mosaic(&buf, frame);
            if (ret < 0)
                goto fail;

            ret = craw_noise_estimate(buf.mosaic, w, h, w, &prof, NULL, &est);
            if (ret == CRAW_ERROR(ENODATA)) {
                fprintf(stderr, "frame %5" PRId64 ": too few levels in this frame to fit "
                        "anything. The fit needs at least three intensity bins holding "
                        "20000 samples each, which a frame of one flat level or of nothing "
                        "but texture will not have.\n", processed);
                goto fail;
            }
            if (ret < 0) {
                fprintf(stderr, "frame %5" PRId64 ": craw_noise_estimate failed: %d\n",
                        processed, ret);
                goto fail;
            }

            printf("frame %5" PRId64 ": a %.4e b %.4e   %u bins, %u fitted, "
                   "fit error %.1f%%, read noise %.2f code values\n",
                   processed, (double)est.model.a, (double)est.model.b,
                   est.nb_bins, est.nb_inliers, 100.0 * (double)est.fit_rms,
                   (double)est.read_noise_lsb);

            ret = tool_noise_run_push(&noise_run, est.model.a, est.model.b);
            if (ret < 0)
                goto fail;

            processed++;
            av_frame_unref(frame);
            continue;
        }

        /* The record was read off this frame's packet before the packet
         * went into the decoder; the pts matches it back to the picture. */
        tool_lens_take(&dec, frame->pts, &lens);

        if (cfg.verbose && lens.valid) {
            float dist[3], net[3];

            craw_lens_corner_shift(&lens, w, h, true, dist);
            craw_lens_corner_shift(&lens, w, h, false, net);

            /* Once for the first frame, then only when the lens has
             * actually moved: on a clip that zooms while recording this is
             * one line per step of the ring, and on a static one it is one
             * line for the whole clip. */
            if (!have_lens_report || fabsf(dist[1] - last_lens_dist) > 0.5f) {
                fprintf(stderr, "frame %5" PRId64 ": lens record, centre (%.4f, %.4f), "
                        "corner %+.2f px R %+.2f G %+.2f B (net %+.2f/%+.2f/%+.2f, "
                        "lateral CA R-B %+.2f px)\n", processed,
                        (double)lens.centre[0], (double)lens.centre[1],
                        (double)dist[0], (double)dist[1], (double)dist[2],
                        (double)net[0], (double)net[1], (double)net[2],
                        (double)(dist[0] - dist[2]));

                /* The scale is what the mode costs: the corrected picture
                 * is the filled one shrunk about the centre by exactly
                 * this much, which is what makes room for the corners fill
                 * mode cuts off. Under the same "only when the lens moved"
                 * rule as the line above, because it moves with the
                 * record. */
                if (cfg.lens == TOOL_LENS_KEEP) {
                    float scale = 0.0f;

                    if (craw_lens_keep_all_scale(&lens, w, h, &scale) == 0)
                        fprintf(stderr, "              keep all: scaled %.4f\n",
                                (double)scale);
                    else
                        fprintf(stderr, "              keep all: this record "
                                        "cannot be inverted\n");
                }

                last_lens_dist = dist[1];
                have_lens_report = true;
            }
        }

        if ((cfg.lens == TOOL_LENS_ON || cfg.lens == TOOL_LENS_KEEP) &&
            !lens.valid && !warned_no_lens) {
            fprintf(stderr, "warning: --lens %s, but this clip's frames carry no "
                            "optical_correction record; nothing is corrected\n",
                    cfg.lens == TOOL_LENS_KEEP ? "keep" : "on");
            warned_no_lens = true;
        }

        /* The pass is only built once a frame has been seen to carry a
         * record, so a clip without one never allocates the extra image. */
        ret = tool_gpu_resize(&gpu, &cfg, w, h, prof.cfa, dec.lens_any);
        if (ret < 0)
            goto fail;

        t0 = av_gettime_relative();
        ret = tool_upload_frame(&gpu, &buf, frame);
        upload_ms = (double)(av_gettime_relative() - t0) / 1000.0;
        if (ret < 0)
            goto fail;

        craw_frame_params_init(&fp);
        fp.profile = &prof;
        fp.exposure_stops = cfg.exposure;
        fp.flags = cfg.clamp_negative ? CRAW_FRAME_FLAG_CLAMP_NEGATIVE : 0;
        if (cfg.no_gamut_compression)
            fp.flags |= CRAW_FRAME_FLAG_NO_GAMUT_COMPRESSION;
        if (cfg.lens == TOOL_LENS_KEEP)
            fp.flags |= CRAW_FRAME_FLAG_LENS_KEEP_ALL;
        if (cfg.highlights)
            fp.flags |= CRAW_FRAME_FLAG_HIGHLIGHT_RECONSTRUCTION;
        fp.denoise_strength = cfg.denoise;
        fp.denoise_chroma_strength = cfg.chroma;
        /* Frames arrive one after another from one decoder, so each is the
         * next frame of the last. The first after --start, and the first
         * after a rebuild for a new frame size or a lens record, go to a
         * pipeline with no state yet, which starts the average by itself. */
        fp.denoise_temporal_strength = cfg.temporal;
        fp.noise_override = cfg.noise;
        fp.false_color_passes = cfg.false_color;
        fp.defringe_strength = cfg.defringe;
        fp.sharpen_radius = cfg.sharpen_radius;
        fp.sharpen_iterations = cfg.sharpen_iterations;
        fp.sharpen_contrast = cfg.sharpen_contrast;
        fp.lens = lens;

        if (cfg.wb == TOOL_WB_FIXED) {
            fp.wb_mode = CRAW_WB_REBALANCE;
            fp.illuminant = cfg.fixed_illum;
        } else if (cfg.wb == TOOL_WB_AUTO && awb.have_illum) {
            fp.wb_mode = CRAW_WB_REBALANCE;
            fp.illuminant = awb.illum;
        }

        rec = (struct tool_record) {
            .pipe = gpu.pipe, .src = &gpu.src.img, .dst = &gpu.dst.img, .fp = &fp,
        };

        t0 = av_gettime_relative();
        if (cfg.bench)
            ret = craw_testdev_run_timed(&gpu.td, tool_record_cb, &rec, &gpu_ms);
        else
            ret = craw_testdev_run(&gpu.td, tool_record_cb, &rec);
        develop_ms = (double)(av_gettime_relative() - t0) / 1000.0;
        if (ret < 0) {
            fprintf(stderr, "the submission failed: %d\n", ret);
            goto fail;
        }

        if (cfg.bench) {
            ret = tool_prof_push(&prof_run, gpu.pipe, gpu_ms);
            if (ret < 0) {
                fprintf(stderr, "cannot read the pass timings: %d\n", ret);
                goto fail;
            }
        }

        gpu_ms_total += gpu_ms;
        decode_ms_total += decode_ms;
        /* The first frame pays for opening the file and starting the
         * decoder's threads, which is not what a decode rate means. */
        if (processed)
            decode_ms_warm += decode_ms;

        /* ------------------------------------------------------ awb */

        if (cfg.wb == TOOL_WB_AUTO) {
            ret = tool_awb_step(&awb, gpu.pipe, &prof, pts_s, processed, cfg.verbose);
            if (ret < 0)
                goto fail;
        }

        /* ------------------------------------------------- download */

        /* 280 MB a frame, so only when something is going to read it. */
        if (cfg.writer != TOOL_WRITER_NONE || cfg.stats) {
            t0 = av_gettime_relative();
            ret = tool_download_frame(&gpu, &buf, cfg.float32);
            download_ms = (double)(av_gettime_relative() - t0) / 1000.0;
            if (ret < 0)
                goto fail;

            if (have_crop) {
                tool_crop(crop, buf.rgba, &w, &h);
                npix = (size_t)w * h;
            }
        }

        /* ---------------------------------------------------- stats */

        if (cfg.stats) {
            struct tool_stats st;

            buf.channel = tool_grow(buf.channel, &buf.channel_values, npix,
                                    sizeof(*buf.channel));
            if (!buf.channel) {
                ret = CRAW_ERROR(ENOMEM);
                goto fail;
            }

            tool_stats_compute(buf.rgba, npix, buf.channel, &st);
            printf("frame %5" PRId64 " %s: min %.5f mean %.5f p50 %.5f p99 %.5f max %.5f\n",
                   processed, cfg.space == CRAW_CS_XYZ_D65 ? "Y" : "G",
                   st.min, st.mean, st.p50, st.p99, st.max);
        }

        /* ---------------------------------------------------- write */

        if (cfg.writer != TOOL_WRITER_NONE) {
            char path[4096];

            /* A pattern without an index is formatted all the same, so
             * that a %% in it still means one percent sign. */
            snprintf(path, sizeof(path), cfg.pattern, (int)processed);

            t0 = av_gettime_relative();
            ret = tool_write_frame(&cfg, path, buf.rgba, w, h);
            write_ms = (double)(av_gettime_relative() - t0) / 1000.0;
            if (ret < 0)
                goto fail;
            if (cfg.verbose)
                fprintf(stderr, "frame %5" PRId64 " -> %s\n", processed, path);
        }

        /* Wall times of the frame's steps. The decode includes the read,
         * and on the first frame the decoder's own start-up; the develop
         * is the submission waited for, so on the first frame it pays for
         * the pipeline's first dispatches too. */
        if (cfg.verbose)
            fprintf(stderr, "frame %5" PRId64 ": decode %.1f ms, upload %.1f ms, develop "
                            "%.1f ms, download %.1f ms, write %.1f ms\n", processed,
                    decode_ms, upload_ms, develop_ms, download_ms, write_ms);

        if (cfg.bench)
            printf("frame %5" PRId64 " pts %8.3f s: gpu %7.3f ms, decode %7.1f ms\n",
                   processed, pts_s, gpu_ms, decode_ms);

        processed++;
        av_frame_unref(frame);
    }

    if (!processed) {
        fprintf(stderr, "no frames were processed\n");
        ret = CRAW_ERROR(EINVAL);
        goto fail;
    }

    if (cfg.noise_estimate)
        tool_noise_run_report(&noise_run, &prof);

    if (cfg.bench) {
        printf("%" PRId64 " frames: gpu %.3f ms/frame average, decode %.1f ms/frame average",
               processed, gpu_ms_total / (double)processed,
               decode_ms_total / (double)processed);
        if (processed > 1)
            printf(" (%.1f ms without the first frame's start-up)",
                   decode_ms_warm / (double)(processed - 1));
        printf("\n");
        tool_prof_report(&prof_run);
    }

    ret = 0;

fail:
    free(noise_run.a);
    free(noise_run.b);
    tool_prof_free(&prof_run);
    tool_buffers_free(&buf);
    av_frame_free(&frame);
    craw_awb_smoother_free(&awb.smoother);
    if (gpu.have_dev) {
        tool_gpu_drop_size(&gpu);
        craw_context_free(&gpu.ctx);
        craw_testdev_close(&gpu.td);
    }
    tool_decoder_close(&dec);

    return ret < 0 ? 1 : 0;
}
