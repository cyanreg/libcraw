/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libplacebo/dispatch.h>

#include "grain.h"

/* The map from gauge and stock onto AV1/H.274 grain parameters. grain.h
 * carries the derivation; this file carries the tables, the solver and the
 * two application paths. */

/* ------------------------------------------------------------- tables */

struct gauge_info {
    const char *name, *key;
    float width_mm;             /* camera aperture width */
    const char *source;
};

/* Camera apertures, because that is where the grain is recorded. The two
 * places the standards are not clean are flagged in `source`: nobody ever
 * standardised Super 35 or 2-perf, and secondary sources disagree over
 * 65 mm by 8%. Only the width enters the model. */
static const struct gauge_info gauges[CRAW_GRAIN_GAUGE_NB] = {
    [CRAW_GRAIN_GAUGE_REGULAR8]   = { "Regular 8",    "regular8",  4.80f,
        "SMPTE ST 231; secondary sources give 4.80-4.88 mm" },
    [CRAW_GRAIN_GAUGE_SUPER8]     = { "Super 8",      "super8",    5.69f,
        "SMPTE ST 157:1999, 5.69 x 4.22 mm" },
    [CRAW_GRAIN_GAUGE_16MM]       = { "16 mm",        "16mm",     10.26f,
        "SMPTE ST 7:2004 / ISO 69:1998, 10.26 x 7.49 mm" },
    [CRAW_GRAIN_GAUGE_SUPER16]    = { "Super 16",     "super16",  12.52f,
        "SMPTE ST 201:1996 / ISO 5768:1998, 12.52 x 7.41 mm" },
    [CRAW_GRAIN_GAUGE_35MM_2PERF] = { "35 mm 2-perf", "2perf",    22.00f,
        "Techniscope, 22.0 x 9.5 mm; no SMPTE or ISO standard exists" },
    [CRAW_GRAIN_GAUGE_35MM_3PERF] = { "35 mm 3-perf", "3perf",    24.89f,
        "ISO 2906:2002 style D, full aperture width" },
    [CRAW_GRAIN_GAUGE_35MM_4PERF] = { "35 mm Academy","academy",  21.95f,
        "SMPTE ST 59 / ISO 2906 style A, 21.95 x 16.00 mm" },
    [CRAW_GRAIN_GAUGE_SUPER35]    = { "Super 35",     "super35",  24.89f,
        "24.89 x 18.66 mm full aperture; never standardised by SMPTE" },
    [CRAW_GRAIN_GAUGE_65MM]       = { "65 mm 5-perf", "65mm",     52.63f,
        "Todd-AO 52.63 x 23.01 mm; other sources say 48.59 mm, unresolved" },
    [CRAW_GRAIN_GAUGE_IMAX]       = { "IMAX 15-perf", "imax",     70.41f,
        "IMAX camera aperture 70.41 x 52.63 mm; usable image 69.6 x 48.5" },
};

struct stock_info {
    const char *name, *key;
    float rms;                  /* diffuse RMS granularity x1000, 48 um */
    float blob_um;              /* dye cloud / grain clump diameter */
    float chroma;               /* default chroma grain, relative to luma */
    const char *source;
};

/* Diffuse RMS granularity is what Kodak publishes when it publishes
 * anything: the standard deviation of density read through a 48 um aperture
 * at a net density of 1.0, times 1000.
 *
 * The awkward finding behind this table is that Kodak does *not* publish a
 * number for any of the VISION3 colour negatives, or for cine Ektachrome.
 * Their data sheets carry a "Diffuse rms Granularity" curve and nothing
 * else - this was checked against the current H-1-5203/5207/5213/5219
 * sheets and back through VISION2 and the 1996 VISION 500T sheet. Four
 * numbers here are therefore read off published sheets and the rest are
 * estimates, and `source` says which is which. They are all in one place so
 * that a better number can replace a guess without touching anything else.
 *
 * The blob diameter is worse. A dye cloud is usually quoted at 10-15 um and
 * that figure could not be traced to any primary source; what could be
 * found is 0.2-2 um for individual silver halide grains (T-grain patent
 * literature), a single-source claim of "often 25 microns" for dye clouds,
 * and an unpinned patent range of 3-20 um. 12 um is the middle of the range
 * people repeat, it is the number every blob size in grain.h scales linearly
 * from, and it is a guess. */
static const struct stock_info stocks[CRAW_GRAIN_STOCK_NB] = {
    [CRAW_GRAIN_STOCK_FINE]   = { "Fine",   "fine",    3.0f, 10.0f, 0.55f,
        "generic, not a real stock" },
    [CRAW_GRAIN_STOCK_MEDIUM] = { "Medium", "medium",  6.0f, 12.0f, 0.60f,
        "generic, not a real stock" },
    [CRAW_GRAIN_STOCK_COARSE] = { "Coarse", "coarse", 12.0f, 16.0f, 0.65f,
        "generic, not a real stock" },

    [CRAW_GRAIN_STOCK_V3_50D]  = { "Vision3 50D",  "50d",  3.0f, 10.0f, 0.55f,
        "ESTIMATE: Kodak H-1-5203 gives a granularity curve, no number" },
    [CRAW_GRAIN_STOCK_V3_200T] = { "Vision3 200T", "200t", 4.0f, 11.5f, 0.60f,
        "ESTIMATE: Kodak H-1-5213 gives a granularity curve, no number" },
    [CRAW_GRAIN_STOCK_V3_250D] = { "Vision3 250D", "250d", 4.0f, 11.5f, 0.60f,
        "ESTIMATE: Kodak H-1-5207 gives a granularity curve, no number" },
    [CRAW_GRAIN_STOCK_V3_500T] = { "Vision3 500T", "500t", 6.0f, 13.0f, 0.65f,
        "ESTIMATE: Kodak H-1-5219 gives a granularity curve, no number" },

    [CRAW_GRAIN_STOCK_DOUBLE_X] = { "Double-X 5222", "doublex", 14.0f, 13.0f, 0.0f,
        "Kodak H-1-5222 (Mar 2022): RMS 14 at net density 1.0, 48 um. "
        "Monochrome, so no chroma grain" },
    [CRAW_GRAIN_STOCK_TRI_X]    = { "Tri-X 400",     "trix",    17.0f, 15.0f, 0.0f,
        "Kodak F-4017 (Feb 2016): RMS 17 at net density 1.0, 48 um, 12x. "
        "Monochrome, so no chroma grain" },
    [CRAW_GRAIN_STOCK_EKTACHROME] = { "Ektachrome",  "ektachrome", 8.0f, 10.0f, 0.50f,
        "Kodak E-4000 (Aug 2018) for still E100: RMS 8 at gross density 1.0. "
        "The cine 5294/7294 sheet publishes a curve only" },
    [CRAW_GRAIN_STOCK_PORTRA400]  = { "Portra 400",  "portra400",  9.0f, 12.5f, 0.60f,
        "ESTIMATE: Kodak publishes Print Grain Index, not RMS, for Portra. "
        "PGI 59 at 8x10 inverts through Kane & Cookingham (PICS 1998) "
        "Gi = 100.8*log10(sigma)-44.3 to a print sigma of 10.6, which is a "
        "print number at 8.8x, not the film's 48 um RMS" },
};

/* Mid grey, display encoded, and the display gamma the response shapes are
 * written against. sRGB is not exactly a 2.2 power law but is within a
 * percent of one over the range that matters here, and PQ is not a power
 * law at all - which is why the amplitude is checked by measurement in
 * tools/grain/grain_preview.c rather than trusted. */
#define GRAIN_MID   0.5f
#define GRAIN_GAMMA 2.2f

/* Response shapes, as grain amplitude against display-encoded picture level
 * - that is, as seen. Normalised so that mid grey gets exactly what the
 * physical model asked for, which is also where granularity is measured.
 * The shapes are modelled, not measured; see grain.h. */
static float response_weight(enum craw_grain_response r, float x)
{
    switch (r) {
    case CRAW_GRAIN_RESPONSE_NEGATIVE: {
        /* Nutting's law puts granularity at roughly sqrt(density), and on a
         * negative density follows scene exposure, so in a gamma 2.2 encoded
         * domain the amplitude goes as x^1.1. Capped because a real negative
         * shoulders off rather than continuing to get grainier. */
        float w = powf(x / 0.5f, 1.1f);
        return w > 1.8f ? 1.8f : w;
    }
    case CRAW_GRAIN_RESPONSE_PRINT: {
        /* The print's own curve has slope only through its middle, so the
         * negative's grain is carried to the screen there and swallowed in
         * the toe and the shoulder. */
        float t = (x - 0.5f) / 0.42f;
        return expf(-t * t);
    }
    case CRAW_GRAIN_RESPONSE_REVERSAL: {
        /* Reversal keeps silver where the scene was dark, so the peak sits
         * below mid grey and the highlights come out nearly clean. */
        float s = x < 0.30f ? 0.30f : 0.28f;
        float t = (x - 0.30f) / s;
        return expf(-t * t);
    }
    case CRAW_GRAIN_RESPONSE_FLAT:
    default:
        return 1.0f;
    }
}

/* ------------------------------------------- AV1 template, per the spec */

/* AV1 section 9.3: 2048 samples of a zero mean Gaussian with a standard
 * deviation of about 512, clipped to [-2048, 2047] and rounded to the
 * nearest multiple of 4. Copied from the specification. */
static const int16_t gaussian_sequence[2048] = {
  56,    568,   -180,  172,   124,   -84,   172,   -64,   -900,  24,   820,
  224,   1248,  996,   272,   -8,    -916,  -388,  -732,  -104,  -188, 800,
  112,   -652,  -320,  -376,  140,   -252,  492,   -168,  44,    -788, 588,
  -584,  500,   -228,  12,    680,   272,   -476,  972,   -100,  652,  368,
  432,   -196,  -720,  -192,  1000,  -332,  652,   -136,  -552,  -604, -4,
  192,   -220,  -136,  1000,  -52,   372,   -96,   -624,  124,   -24,  396,
  540,   -12,   -104,  640,   464,   244,   -208,  -84,   368,   -528, -740,
  248,   -968,  -848,  608,   376,   -60,   -292,  -40,   -156,  252,  -292,
  248,   224,   -280,  400,   -244,  244,   -60,   76,    -80,   212,  532,
  340,   128,   -36,   824,   -352,  -60,   -264,  -96,   -612,  416,  -704,
  220,   -204,  640,   -160,  1220,  -408,  900,   336,   20,    -336, -96,
  -792,  304,   48,    -28,   -1232, -1172, -448,  104,   -292,  -520, 244,
  60,    -948,  0,     -708,  268,   108,   356,   -548,  488,   -344, -136,
  488,   -196,  -224,  656,   -236,  -1128, 60,    4,     140,   276,  -676,
  -376,  168,   -108,  464,   8,     564,   64,    240,   308,   -300, -400,
  -456,  -136,  56,    120,   -408,  -116,  436,   504,   -232,  328,  844,
  -164,  -84,   784,   -168,  232,   -224,  348,   -376,  128,   568,  96,
  -1244, -288,  276,   848,   832,   -360,  656,   464,   -384,  -332, -356,
  728,   -388,  160,   -192,  468,   296,   224,   140,   -776,  -100, 280,
  4,     196,   44,    -36,   -648,  932,   16,    1428,  28,    528,  808,
  772,   20,    268,   88,    -332,  -284,  124,   -384,  -448,  208,  -228,
  -1044, -328,  660,   380,   -148,  -300,  588,   240,   540,   28,   136,
  -88,   -436,  256,   296,   -1000, 1400,  0,     -48,   1056,  -136, 264,
  -528,  -1108, 632,   -484,  -592,  -344,  796,   124,   -668,  -768, 388,
  1296,  -232,  -188,  -200,  -288,  -4,    308,   100,   -168,  256,  -500,
  204,   -508,  648,   -136,  372,   -272,  -120,  -1004, -552,  -548, -384,
  548,   -296,  428,   -108,  -8,    -912,  -324,  -224,  -88,   -112, -220,
  -100,  996,   -796,  548,   360,   -216,  180,   428,   -200,  -212, 148,
  96,    148,   284,   216,   -412,  -320,  120,   -300,  -384,  -604, -572,
  -332,  -8,    -180,  -176,  696,   116,   -88,   628,   76,    44,   -516,
  240,   -208,  -40,   100,   -592,  344,   -308,  -452,  -228,  20,   916,
  -1752, -136,  -340,  -804,  140,   40,    512,   340,   248,   184,  -492,
  896,   -156,  932,   -628,  328,   -688,  -448,  -616,  -752,  -100, 560,
  -1020, 180,   -800,  -64,   76,    576,   1068,  396,   660,   552,  -108,
  -28,   320,   -628,  312,   -92,   -92,   -472,  268,   16,    560,  516,
  -672,  -52,   492,   -100,  260,   384,   284,   292,   304,   -148, 88,
  -152,  1012,  1064,  -228,  164,   -376,  -684,  592,   -392,  156,  196,
  -524,  -64,   -884,  160,   -176,  636,   648,   404,   -396,  -436, 864,
  424,   -728,  988,   -604,  904,   -592,  296,   -224,  536,   -176, -920,
  436,   -48,   1176,  -884,  416,   -776,  -824,  -884,  524,   -548, -564,
  -68,   -164,  -96,   692,   364,   -692,  -1012, -68,   260,   -480, 876,
  -1116, 452,   -332,  -352,  892,   -1088, 1220,  -676,  12,    -292, 244,
  496,   372,   -32,   280,   200,   112,   -440,  -96,   24,    -644, -184,
  56,    -432,  224,   -980,  272,   -260,  144,   -436,  420,   356,  364,
  -528,  76,    172,   -744,  -368,  404,   -752,  -416,  684,   -688, 72,
  540,   416,   92,    444,   480,   -72,   -1416, 164,   -1172, -68,  24,
  424,   264,   1040,  128,   -912,  -524,  -356,  64,    876,   -12,  4,
  -88,   532,   272,   -524,  320,   276,   -508,  940,   24,    -400, -120,
  756,   60,    236,   -412,  100,   376,   -484,  400,   -100,  -740, -108,
  -260,  328,   -268,  224,   -200,  -416,  184,   -604,  -564,  -20,  296,
  60,    892,   -888,  60,    164,   68,    -760,  216,   -296,  904,  -336,
  -28,   404,   -356,  -568,  -208,  -1480, -512,  296,   328,   -360, -164,
  -1560, -776,  1156,  -428,  164,   -504,  -112,  120,   -216,  -148, -264,
  308,   32,    64,    -72,   72,    116,   176,   -64,   -272,  460,  -536,
  -784,  -280,  348,   108,   -752,  -132,  524,   -540,  -776,  116,  -296,
  -1196, -288,  -560,  1040,  -472,  116,   -848,  -1116, 116,   636,  696,
  284,   -176,  1016,  204,   -864,  -648,  -248,  356,   972,   -584, -204,
  264,   880,   528,   -24,   -184,  116,   448,   -144,  828,   524,  212,
  -212,  52,    12,    200,   268,   -488,  -404,  -880,  824,   -672, -40,
  908,   -248,  500,   716,   -576,  492,   -576,  16,    720,   -108, 384,
  124,   344,   280,   576,   -500,  252,   104,   -308,  196,   -188, -8,
  1268,  296,   1032,  -1196, 436,   316,   372,   -432,  -200,  -660, 704,
  -224,  596,   -132,  268,   32,    -452,  884,   104,   -1008, 424,  -1348,
  -280,  4,     -1168, 368,   476,   696,   300,   -8,    24,    180,  -592,
  -196,  388,   304,   500,   724,   -160,  244,   -84,   272,   -256, -420,
  320,   208,   -144,  -156,  156,   364,   452,   28,    540,   316,  220,
  -644,  -248,  464,   72,    360,   32,    -388,  496,   -680,  -48,  208,
  -116,  -408,  60,    -604,  -392,  548,   -840,  784,   -460,  656,  -544,
  -388,  -264,  908,   -800,  -628,  -612,  -568,  572,   -220,  164,  288,
  -16,   -308,  308,   -112,  -636,  -760,  280,   -668,  432,   364,  240,
  -196,  604,   340,   384,   196,   592,   -44,   -500,  432,   -580, -132,
  636,   -76,   392,   4,     -412,  540,   508,   328,   -356,  -36,  16,
  -220,  -64,   -248,  -60,   24,    -192,  368,   1040,  92,    -24,  -1044,
  -32,   40,    104,   148,   192,   -136,  -520,  56,    -816,  -224, 732,
  392,   356,   212,   -80,   -424,  -1008, -324,  588,   -1496, 576,  460,
  -816,  -848,  56,    -580,  -92,   -1372, -112,  -496,  200,   364,  52,
  -140,  48,    -48,   -60,   84,    72,    40,    132,   -356,  -268, -104,
  -284,  -404,  732,   -520,  164,   -304,  -540,  120,   328,   -76,  -460,
  756,   388,   588,   236,   -436,  -72,   -176,  -404,  -316,  -148, 716,
  -604,  404,   -72,   -88,   -888,  -68,   944,   88,    -220,  -344, 960,
  472,   460,   -232,  704,   120,   832,   -228,  692,   -508,  132,  -476,
  844,   -748,  -364,  -44,   1116,  -1104, -1056, 76,    428,   552,  -692,
  60,    356,   96,    -384,  -188,  -612,  -576,  736,   508,   892,  352,
  -1132, 504,   -24,   -352,  324,   332,   -600,  -312,  292,   508,  -144,
  -8,    484,   48,    284,   -260,  -240,  256,   -100,  -292,  -204, -44,
  472,   -204,  908,   -188,  -1000, -256,  92,    1164,  -392,  564,  356,
  652,   -28,   -884,  256,   484,   -192,  760,   -176,  376,   -524, -452,
  -436,  860,   -736,  212,   124,   504,   -476,  468,   76,    -472, 552,
  -692,  -944,  -620,  740,   -240,  400,   132,   20,    192,   -196, 264,
  -668,  -1012, -60,   296,   -316,  -828,  76,    -156,  284,   -768, -448,
  -832,  148,   248,   652,   616,   1236,  288,   -328,  -400,  -124, 588,
  220,   520,   -696,  1032,  768,   -740,  -92,   -272,  296,   448,  -464,
  412,   -200,  392,   440,   -200,  264,   -152,  -260,  320,   1032, 216,
  320,   -8,    -64,   156,   -1016, 1084,  1172,  536,   484,   -432, 132,
  372,   -52,   -256,  84,    116,   -352,  48,    116,   304,   -384, 412,
  924,   -300,  528,   628,   180,   648,   44,    -980,  -220,  1320, 48,
  332,   748,   524,   -268,  -720,  540,   -276,  564,   -344,  -208, -196,
  436,   896,   88,    -392,  132,   80,    -964,  -288,  568,   56,   -48,
  -456,  888,   8,     552,   -156,  -292,  948,   288,   128,   -716, -292,
  1192,  -152,  876,   352,   -600,  -260,  -812,  -468,  -28,   -120, -32,
  -44,   1284,  496,   192,   464,   312,   -76,   -516,  -380,  -456, -1012,
  -48,   308,   -156,  36,    492,   -156,  -808,  188,   1652,  68,   -120,
  -116,  316,   160,   -140,  352,   808,   -416,  592,   316,   -480, 56,
  528,   -204,  -568,  372,   -232,  752,   -344,  744,   -4,    324,  -416,
  -600,  768,   268,   -248,  -88,   -132,  -420,  -432,  80,    -288, 404,
  -316,  -1216, -588,  520,   -108,  92,    -320,  368,   -480,  -216, -92,
  1688,  -300,  180,   1020,  -176,  820,   -68,   -228,  -260,  436,  -904,
  20,    40,    -508,  440,   -736,  312,   332,   204,   760,   -372, 728,
  96,    -20,   -632,  -520,  -560,  336,   1076,  -64,   -532,  776,  584,
  192,   396,   -728,  -520,  276,   -188,  80,    -52,   -612,  -252, -48,
  648,   212,   -688,  228,   -52,   -260,  428,   -412,  -272,  -404, 180,
  816,   -796,  48,    152,   484,   -88,   -216,  988,   696,   188,  -528,
  648,   -116,  -180,  316,   476,   12,    -564,  96,    476,   -252, -364,
  -376,  -392,  556,   -256,  -576,  260,   -352,  120,   -16,   -136, -260,
  -492,  72,    556,   660,   580,   616,   772,   436,   424,   -32,  -324,
  -1268, 416,   -324,  -80,   920,   160,   228,   724,   32,    -516, 64,
  384,   68,    -128,  136,   240,   248,   -204,  -68,   252,   -932, -120,
  -480,  -628,  -84,   192,   852,   -404,  -288,  -132,  204,   100,  168,
  -68,   -196,  -868,  460,   1080,  380,   -80,   244,   0,     484,  -888,
  64,    184,   352,   600,   460,   164,   604,   -196,  320,   -64,  588,
  -184,  228,   12,    372,   48,    -848,  -344,  224,   208,   -200, 484,
  128,   -20,   272,   -468,  -840,  384,   256,   -720,  -520,  -464, -580,
  112,   -120,  644,   -356,  -208,  -608,  -528,  704,   560,   -424, 392,
  828,   40,    84,    200,   -152,  0,     -144,  584,   280,   -120, 80,
  -556,  -972,  -196,  -472,  724,   80,    168,   -32,   88,    160,  -688,
  0,     160,   356,   372,   -776,  740,   -128,  676,   -248,  -480, 4,
  -364,  96,    544,   232,   -1032, 956,   236,   356,   20,    -40,  300,
  24,    -676,  -596,  132,   1120,  -104,  532,   -1096, 568,   648,  444,
  508,   380,   188,   -376,  -604,  1488,  424,   24,    756,   -220, -192,
  716,   120,   920,   688,   168,   44,    -460,  568,   284,   1144, 1160,
  600,   424,   888,   656,   -356,  -320,  220,   316,   -176,  -724, -188,
  -816,  -628,  -348,  -228,  -380,  1012,  -452,  -660,  736,   928,  404,
  -696,  -72,   -268,  -892,  128,   184,   -344,  -780,  360,   336,  400,
  344,   428,   548,   -112,  136,   -228,  -216,  -820,  -516,  340,  92,
  -136,  116,   -300,  376,   -244,  100,   -316,  -520,  -284,  -12,  824,
  164,   -548,  -180,  -128,  116,   -924,  -828,  268,   -368,  -580, 620,
  192,   160,   0,     -1676, 1068,  424,   -56,   -360,  468,   -156, 720,
  288,   -528,  556,   -364,  548,   -148,  504,   316,   152,   -648, -620,
  -684,  -24,   -376,  -384,  -108,  -920,  -1032, 768,   180,   -264, -508,
  -1268, -260,  -60,   300,   -240,  988,   724,   -376,  -576,  -212, -736,
  556,   192,   1092,  -620,  -880,  376,   -56,   -4,    -216,  -32,  836,
  268,   396,   1332,  864,   -600,  100,   56,    -412,  -92,   356,  180,
  884,   -468,  -436,  292,   -388,  -804,  -704,  -840,  368,   -348, 140,
  -724,  1536,  940,   372,   112,   -372,  436,   -480,  1136,  296,  -32,
  -228,  132,   -48,   -220,  868,   -1016, -60,   -1044, -464,  328,  916,
  244,   12,    -736,  -296,  360,   468,   -376,  -108,  -92,   788,  368,
  -56,   544,   400,   -672,  -420,  728,   16,    320,   44,    -284, -380,
  -796,  488,   132,   204,   -596,  -372,  88,    -152,  -908,  -636, -572,
  -624,  -116,  -692,  -200,  -56,   276,   -88,   484,   -324,  948,  864,
  1000,  -456,  -184,  -276,  292,   -296,  156,   676,   320,   160,  908,
  -84,   -1236, -288,  -116,  260,   -372,  -644,  732,   -756,  -96,  84,
  344,   -520,  348,   -688,  240,   -84,   216,   -1044, -136,  -676, -396,
  -1500, 960,   -40,   176,   168,   1516,  420,   -504,  -344,  -364, -360,
  1216,  -940,  -380,  -212,  252,   -660,  -708,  484,   -444,  -152, 928,
  -120,  1112,  476,   -260,  560,   -148,  -344,  108,   -196,  228,  -288,
  504,   560,   -328,  -88,   288,   -1008, 460,   -228,  468,   -836, -196,
  76,    388,   232,   412,   -1168, -716,  -644,  756,   -172,  -356, -504,
  116,   432,   528,   48,    476,   -168,  -608,  448,   160,   -532, -272,
  28,    -676,  -12,   828,   980,   456,   520,   104,   -104,  256,  -344,
  -4,    -28,   -368,  -52,   -524,  -572,  -556,  -200,  768,   1124, -208,
  -512,  176,   232,   248,   -148,  -888,  604,   -600,  -304,  804,  -156,
  -212,  488,   -192,  -804,  -256,  368,   -360,  -916,  -328,  228,  -240,
  -448,  -472,  856,   -556,  -364,  572,   -12,   -156,  -368,  -340, 432,
  252,   -752,  -152,  288,   268,   -580,  -848,  -592,  108,   -76,  244,
  312,   -716,  592,   -80,   436,   360,   4,     -248,  160,   516,  584,
  732,   44,    -468,  -280,  -292,  -156,  -588,  28,    308,   912,  24,
  124,   156,   180,   -252,  944,   -924,  -772,  -520,  -428,  -624, 300,
  -212,  -1144, 32,    -724,  800,   -1128, -212,  -1288, -848,  180,  -416,
  440,   192,   -576,  -792,  -76,   -1080, 80,    -532,  -352,  -132, 380,
  -820,  148,   1112,  128,   164,   456,   700,   -924,  144,   -668, -384,
  648,   -832,  508,   552,   -52,   -100,  -656,  208,   -568,  748,  -88,
  680,   232,   300,   192,   -408,  -1012, -152,  -252,  -268,  272,  -876,
  -664,  -648,  -332,  -136,  16,    12,    1152,  -28,   332,   -536, 320,
  -672,  -460,  -316,  532,   -260,  228,   -40,   1052,  -816,  180,  88,
  -496,  -556,  -672,  -368,  428,   92,    356,   404,   -408,  252,  196,
  -176,  -556,  792,   268,   32,    372,   40,    96,    -332,  328,  120,
  372,   -900,  -40,   472,   -264,  -592,  952,   128,   656,   112,  664,
  -232,  420,   4,     -344,  -464,  556,   244,   -416,  -32,   252,  0,
  -412,  188,   -696,  508,   -476,  324,   -1096, 656,   -312,  560,  264,
  -136,  304,   160,   -64,   -580,  248,   336,   -720,  560,   -348, -288,
  -276,  -196,  -500,  852,   -544,  -236,  -1128, -992,  -776,  116,  56,
  52,    860,   884,   212,   -12,   168,   1020,  512,   -552,  924,  -148,
  716,   188,   164,   -340,  -520,  -184,  880,   -152,  -680,  -208, -1156,
  -300,  -528,  -472,  364,   100,   -744,  -1056, -32,   540,   280,  144,
  -676,  -32,   -232,  -280,  -224,  96,    568,   -76,   172,   148,  148,
  104,   32,    -296,  -32,   788,   -80,   32,    -16,   280,   288,  944,
  428,   -484
};

static int get_random_number(int bits, uint16_t *state)
{
    int r = *state;
    uint16_t bit = ((r >> 0) ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
    *state = (r >> 1) | (bit << 15);
    return (*state >> (16 - bits)) & ((1 << bits) - 1);
}

static int round2(int x, int shift)
{
    return shift ? (x + (1 << (shift - 1))) >> shift : x;
}

/* AV1 7.18.3.3 generate grain process, luma half. libplacebo runs exactly
 * this on the CPU before uploading, so measuring it here measures what the
 * shader will actually show. */
float craw_grain_template_y(int16_t buf[73][82],
                            const struct pl_av1_grain_data *d,
                            int bits, uint16_t seed)
{
    const int shift = 12 - bits + d->grain_scale_shift;
    const int centre = 128 << (bits - 8);
    const int gmin = -centre, gmax = (256 << (bits - 8)) - 1 - centre;

    for (int y = 0; y < 73; y++)
        for (int x = 0; x < 82; x++)
            buf[y][x] = round2(gaussian_sequence[get_random_number(11, &seed)],
                               shift);

    for (int y = 3; y < 73; y++) {
        for (int x = 3; x < 82 - 3; x++) {
            const int8_t *c = d->ar_coeffs_y;
            int sum = 0;
            for (int dy = -d->ar_coeff_lag; dy <= 0; dy++) {
                for (int dx = -d->ar_coeff_lag; dx <= d->ar_coeff_lag; dx++) {
                    if (!dx && !dy)
                        goto filtered;
                    sum += *(c++) * buf[y + dy][x + dx];
                }
            }
        filtered:;
            int g = buf[y][x] + round2(sum, d->ar_coeff_shift);
            buf[y][x] = g < gmin ? gmin : (g > gmax ? gmax : g);
        }
    }

    /* libplacebo uploads buf[9 + y][9 + x] for a 64x64 window, so that is
     * the only part whose statistics matter. */
    double mean = 0.0;
    for (int y = 0; y < 64; y++)
        for (int x = 0; x < 64; x++)
            mean += buf[y + 9][x + 9];
    mean /= 64 * 64;
    double var = 0.0;
    for (int y = 0; y < 64; y++)
        for (int x = 0; x < 64; x++) {
            double t = buf[y + 9][x + 9] - mean;
            var += t * t;
        }
    return (float)sqrt(var / (64 * 64));
}

/* AV1 7.18.3.3's chroma half, for the unsubsampled case that is all this
 * tool ever sees (one RGB plane, so every channel is full resolution).
 *
 * The chroma template needs measuring separately rather than reusing the
 * luma's standard deviation, because the centre tap mixes the luma template
 * into the *recursion*: the luma contribution comes out filtered twice and
 * so is amplified by the square of the filter's gain. With a weak filter
 * that is a few percent; with the strong filter a five pixel blob needs it
 * more than doubles the chroma amplitude, which is exactly the "colour
 * grain louder than luma grain" that measurement caught. */
static float template_uv(int16_t buf[73][82], const int16_t buf_y[73][82],
                         const struct pl_av1_grain_data *d, int channel,
                         int bits, uint16_t seed)
{
    const int shift = 12 - bits + d->grain_scale_shift;
    const int centre = 128 << (bits - 8);
    const int gmin = -centre, gmax = (256 << (bits - 8)) - 1 - centre;

    /* The spec's decorrelating constants, so that Cb and Cr are neither the
     * same grain as each other nor as luma. */
    seed ^= channel == 0 ? 0xb524u : 0x49d8u;

    for (int y = 0; y < 73; y++)
        for (int x = 0; x < 82; x++)
            buf[y][x] = round2(gaussian_sequence[get_random_number(11, &seed)],
                               shift);

    const int8_t *coeffs = d->ar_coeffs_uv[channel];
    for (int y = 3; y < 73; y++) {
        for (int x = 3; x < 82 - 3; x++) {
            const int8_t *c = coeffs;
            int sum = 0;
            for (int dy = -d->ar_coeff_lag; dy <= 0; dy++) {
                for (int dx = -d->ar_coeff_lag; dx <= d->ar_coeff_lag; dx++) {
                    if (!dx && !dy) {
                        if (d->num_points_y)
                            sum += buf_y[y][x] * (*c);
                        goto filtered;
                    }
                    sum += *(c++) * buf[y + dy][x + dx];
                }
            }
        filtered:;
            int g = buf[y][x] + round2(sum, d->ar_coeff_shift);
            buf[y][x] = (int16_t)(g < gmin ? gmin : (g > gmax ? gmax : g));
        }
    }

    double mean = 0.0, var = 0.0;
    for (int y = 0; y < 64; y++)
        for (int x = 0; x < 64; x++)
            mean += buf[y + 9][x + 9];
    mean /= 64 * 64;
    for (int y = 0; y < 64; y++)
        for (int x = 0; x < 64; x++) {
            double t = buf[y + 9][x + 9] - mean;
            var += t * t;
        }
    return (float)sqrt(var / (64 * 64));
}

/* The picture the shader will build out of a template, and its statistics.
 *
 * Neither the amplitude nor the blob size can be read off the 64x64 template
 * alone. The shader lays 32x32 blocks down, each reading the template from a
 * random offset of 0 to 30 in steps of two and blended into its neighbours
 * over two pixels, so adjacent blocks share most of their content and the
 * field has structure the template does not. More importantly, subtracting a
 * 64x64 window's own mean removes exactly the lowest frequencies a coarse
 * filter puts most of its energy in - measured against the GPU, that made
 * the standard deviation of a five pixel grain come out 40% low, and the
 * amplitude calibration with it. So synthesise a piece of the actual field,
 * exactly as AV1 7.18.3.5 and pl_shader_fg_av1() do, and measure that. */
#define FIELD 192

/* One fixed seed for every measurement, so that solving is deterministic
 * and the same parameters always produce the same answer. */
#define AR_SOLVE_SEED 0x4321u

struct field_stats {
    float sd;                   /* 8-bit grain units */
    float fwhm;                 /* pixels */
    float clip;                 /* fraction of the template at the clamp */
};

/* AV1 7.18.3.5's per-row LFSR walk over the block grid, and libplacebo's
 * packing of the four neighbouring offsets into one word. */
static void field_offsets(uint32_t *offs, int ow, int oh, uint16_t seed)
{
    for (int y = 0; y < oh; y++) {
        uint16_t st = seed;
        st ^= (uint16_t)(((y * 37 + 178) & 0xFF) << 8);
        st ^= (uint16_t)((y * 173 + 105) & 0xFF);
        for (int x = 0; x < ow; x++) {
            uint32_t *o = &offs[y * ow + x];
            uint8_t v = (uint8_t)get_random_number(8, &st);
            uint8_t vl = x ? (uint8_t)(o[-1] & 0xFF) : 0;
            uint8_t vt = y ? (uint8_t)(o[-ow] & 0xFF) : 0;
            uint8_t vtl = (x && y) ? (uint8_t)(o[-ow - 1] & 0xFF) : 0;
            *o = ((uint32_t)vtl << 24) | ((uint32_t)vt << 16) |
                 ((uint32_t)vl << 8) | v;
        }
    }
}

static float field_fetch(const int16_t buf[73][82], uint32_t data, int shift,
                         int lx, int ly, int bdx, int bdy)
{
    int ox = 2 * (int)((data >> (shift + 4)) & 0xF);
    int oy = 2 * (int)((data >> shift) & 0xF);
    int px = (ox + lx + 32 * bdx) & 63;
    int py = (oy + ly + 32 * bdy) & 63;
    return buf[py + 9][px + 9];
}

static void field_measure(const int16_t buf[73][82], bool overlap, int bits,
                          uint16_t seed, bool want_fwhm, struct field_stats *st)
{
    static float f[FIELD * FIELD];
    static uint32_t offs[(FIELD / 32 + 4) * (FIELD / 32 + 4)];
    const int ow = (FIELD + 127) / 128 * 4, oh = ow;

    field_offsets(offs, ow, oh, seed);

    const int centre = 128 << (bits - 8);
    const float gmin = (float)-centre;
    const float gmax = (float)((256 << (bits - 8)) - 1 - centre);

    for (int gy = 0; gy < FIELD; gy++) {
        for (int gx = 0; gx < FIELD; gx++) {
            int bx = gx / 32, by = gy / 32;
            int lx = gx - bx * 32, ly = gy - by * 32;
            uint32_t data = offs[by * ow + bx];
            float g = field_fetch(buf, data, 0, lx, ly, 0, 0);
            if (overlap) {
                /* The 27/17 and 17/27 weights of AV1 7.18.3.5. */
                if (bx > 0 && lx < 2) {
                    float w0 = (lx == 1 ? 17.0f : 27.0f) / 32.0f;
                    float w1 = (lx == 1 ? 27.0f : 17.0f) / 32.0f;
                    g = field_fetch(buf, data, 8, lx, ly, 1, 0) * w0 + g * w1;
                }
                if (by > 0 && ly < 2) {
                    float w0 = (ly == 1 ? 17.0f : 27.0f) / 32.0f;
                    float w1 = (ly == 1 ? 27.0f : 17.0f) / 32.0f;
                    float v;
                    if (bx > 0 && lx < 2) {
                        float u0 = (lx == 1 ? 17.0f : 27.0f) / 32.0f;
                        float u1 = (lx == 1 ? 27.0f : 17.0f) / 32.0f;
                        v = field_fetch(buf, data, 24, lx, ly, 1, 1) * u0 +
                            field_fetch(buf, data, 16, lx, ly, 0, 1) * u1;
                    } else {
                        v = field_fetch(buf, data, 16, lx, ly, 0, 1);
                    }
                    g = v * w0 + g * w1;
                }
                g = g < gmin ? gmin : (g > gmax ? gmax : g);
            }
            f[gy * FIELD + gx] = g;
        }
    }

    double mean = 0.0, var = 0.0;
    for (int i = 0; i < FIELD * FIELD; i++)
        mean += f[i];
    mean /= FIELD * FIELD;
    for (int i = 0; i < FIELD * FIELD; i++) {
        double t = f[i] - mean;
        var += t * t;
    }
    var /= FIELD * FIELD;
    st->sd = (float)sqrt(var);

    st->fwhm = 1.0f;
    if (want_fwhm && var > 0.0) {
        double half[2] = { 0.5, 0.5 };
        for (int axis = 0; axis < 2; axis++) {
            double prev = 1.0;
            /* Nothing above eight pixels is reachable, so there is no point
             * walking the autocorrelation further than that. */
            for (int lag = 1; lag < 16; lag++) {
                double acc = 0.0;
                long n = 0;
                for (int y = 0; y < FIELD - (axis ? lag : 0); y++)
                    for (int x = 0; x < FIELD - (axis ? 0 : lag); x++) {
                        double a = f[y * FIELD + x] - mean;
                        double b = axis ? f[(y + lag) * FIELD + x] - mean
                                        : f[y * FIELD + x + lag] - mean;
                        acc += a * b;
                        n++;
                    }
                double rho = acc / n / var;
                if (rho < 0.5) {
                    half[axis] = (lag - 1) + (prev - 0.5) / (prev - rho);
                    break;
                }
                prev = rho;
                half[axis] = lag;
            }
        }
        st->fwhm = (float)(half[0] + half[1]);
    }

    /* A filter strong enough to make big blobs runs away unless
     * grain_scale_shift divides the white noise down first, and a clipped
     * template looks like blotches rather than grain. */
    int hit = 0;
    for (int y = 0; y < 64; y++)
        for (int x = 0; x < 64; x++) {
            int v = buf[y + 9][x + 9];
            if (v <= -centre || v >= (256 << (bits - 8)) - 1 - centre)
                hit++;
        }
    st->clip = hit / (float)(64 * 64);
}

/* A separable first order filter, g = w + a*g[x-1] + a*g[y-1] - a^2*g[y-1][x-1],
 * whose autocorrelation is exactly a^|dx| * a^|dy|. AV1 orders lag 1
 * coefficients as (-1,-1) (-1,0) (-1,+1) (0,-1). Two things make this the
 * right family rather than a wide lag 3 filter: it is the only one whose
 * blob size can be steered by a single number, and measurement (grain.h)
 * says a weak lag 3 filter barely widens the autocorrelation at all. */
static void ar_separable(struct pl_av1_grain_data *d, float a)
{
    /* ar_coeff_shift 7 divides by 128, so coefficients reach 0.992 - more
     * than the 0.93 this solver ever asks for - in steps of 1/128. Shift 6
     * would reach 1.98 but in twice the step, and the step is what limits
     * how precisely a blob size can be hit; shift 8 tops out at 0.496 and
     * cannot make coarse grain at all. */
    const float q = 128.0f;
    memset(d->ar_coeffs_y, 0, sizeof(d->ar_coeffs_y));
    d->ar_coeff_lag = 1;
    d->ar_coeff_shift = 7;
    d->ar_coeffs_y[0] = (int8_t)lrintf(-a * a * q);
    d->ar_coeffs_y[1] = (int8_t)lrintf(a * q);
    d->ar_coeffs_y[2] = 0;
    d->ar_coeffs_y[3] = (int8_t)lrintf(a * q);
}

/* Evaluates one candidate filter strength: picks the smallest
 * grain_scale_shift that keeps the clamp out of the picture, and reports
 * what that template's blob size and standard deviation come to. */
#define AR_SOLVE_SEEDS 3

/* How far grain_scale_shift has to divide the white noise down before the
 * filter's gain runs the template into the +-128 clamp. A separable AR(1)
 * with coefficient a has a variance gain of 1/(1-a^2)^2, and a blob of b
 * pixels needs a of about 0.5^(2/b), so the standard deviation of the
 * template goes 42 at two pixels, 64 at four, 86 at six - against a clamp
 * at 128, i.e. barely three sigma at four pixels. Choosing the shift from
 * the target rather than from a clipping test keeps the blob size a
 * monotone function of the coefficient, which is what lets the search
 * below be a bisection. */
static int gss_for_blob(float blob)
{
    if (blob < 2.5f)
        return 0;
    if (blob < 4.5f)
        return 1;
    return 2;
}

/* Evaluates one achievable filter: the coefficients are 8-bit at
 * ar_coeff_shift, so only `coeff`/128 exists, and the search has to be over
 * integers rather than over a real number it then rounds. */
static void ar_eval(int coeff, int gss, int bits, struct pl_av1_grain_data *d,
                    float *fwhm, float *sd, float *clip)
{
    static int16_t buf[73][82];
    struct field_stats st;

    ar_separable(d, coeff / 128.0f);
    d->grain_scale_shift = gss;

    /* A 64x64 template holds only about (64/blob)^2 independent blobs, so at
     * the coarse end one seed's realised variance strays several percent
     * from the process's. Averaging a few seeds takes the bias out of the
     * solve; what remains is that any one frame still lands a few percent
     * either side, which is a property of the model and not of this code. */
    double var = 0.0;
    for (int i = 0; i < AR_SOLVE_SEEDS; i++) {
        struct field_stats s2;
        uint16_t seed = (uint16_t)(AR_SOLVE_SEED + i * 7919);
        craw_grain_template_y(buf, d, bits, seed);
        field_measure(buf, true, bits, seed, false, &s2);
        var += (double)s2.sd * s2.sd;
    }
    *sd = (float)sqrt(var / AR_SOLVE_SEEDS);

    craw_grain_template_y(buf, d, bits, AR_SOLVE_SEED);
    field_measure(buf, true, bits, AR_SOLVE_SEED, true, &st);
    *fwhm = st.fwhm;
    *clip = st.clip;
}

/* Finds the coefficient whose blob size comes closest to `target`. With the
 * shift fixed the blob size rises monotonically with the coefficient, so a
 * bisection over the integers converges; the two neighbours are then checked
 * because the last step of the bisection can land either side.
 *
 * grain_scale_shift is taken from the target rather than from a clipping
 * test at the strongest coefficient - testing there would find that a = 0.93
 * clips at every shift below 3 and then solve the whole problem at shift 3,
 * where the template's standard deviation is so small that the scaling
 * function runs out of range before it reaches the amplitude asked for. It
 * is only raised if the coefficient actually chosen turns out to clip. */
static void ar_solve(float target, int bits, struct pl_av1_grain_data *d,
                     float *sd_out, float *fwhm_out)
{
    /* The solve depends on nothing but the target blob size and the bit
     * depth, and costs a few dozen template generations, while the things
     * that change most often - amount, response, seed - do not touch it.
     * One cached answer is enough: the caller sweeping a parameter holds
     * the others still. Single threaded, like the rest of the viewer. */
    static struct {
        float target, sd, fwhm;
        int bits, valid;
        struct pl_av1_grain_data data;
    } memo;

    if (memo.valid && memo.bits == bits && memo.target == target) {
        *d = memo.data;
        *sd_out = memo.sd;
        *fwhm_out = memo.fwhm;
        return;
    }

    int gss = gss_for_blob(target);
    float fwhm = 1.0f, sd = 32.0f, clip = 0.0f;
    int best = 0;

    for (int attempt = 0; attempt < 3; attempt++) {
        /* 119/128 = 0.93. Above that the causal filter's transient is
         * longer than the 64x64 window libplacebo uploads and the template
         * stops being uniform: it turns into blotches rather than grain. */
        int lo = 0, hi = 119;

        ar_eval(hi, gss, bits, d, &fwhm, &sd, &clip);
        if (target >= fwhm) {
            best = hi;          /* coarser than the model can go */
        } else {
            while (hi - lo > 1) {
                int mid = (lo + hi) / 2;
                ar_eval(mid, gss, bits, d, &fwhm, &sd, &clip);
                if (fwhm < target)
                    lo = mid;
                else
                    hi = mid;
            }
            float best_err = INFINITY;
            best = lo;
            for (int c = lo; c <= hi; c++) {
                ar_eval(c, gss, bits, d, &fwhm, &sd, &clip);
                float err = fabsf(fwhm - target);
                if (err < best_err) {
                    best_err = err;
                    best = c;
                }
            }
        }

        ar_eval(best, gss, bits, d, &fwhm, &sd, &clip);
        if (clip < 0.001f || gss >= 3)
            break;
        gss++;
    }

    *sd_out = sd;
    *fwhm_out = fwhm;

    memo.target = target;
    memo.bits = bits;
    memo.sd = sd;
    memo.fwhm = fwhm;
    memo.data = *d;
    memo.valid = 1;
}

/* ------------------------------------------------- H.274 calibration */

/* What libplacebo 7.371's H.274 shader actually delivers for each cut-off
 * index, measured on a flat mid grey field by
 * `grain_preview --calibrate`: the standard deviation per unit of
 * comp_model_value scale, and the blob size. Band limiting removes energy,
 * so coarse grain arrives weaker and needs a larger scale to compensate.
 *
 * These are measurements and not predictions on purpose. Generating the
 * 64x64 slices on the CPU from the RDD 5 tables gives sigmas within 7% of
 * these, but the shader does not use a whole slice: it takes a 16x16 patch
 * at a 4-aligned x and 8-aligned y offset for every 16x16 region of the
 * picture, which both loses some of the lowest frequencies and narrows the
 * blob at the coarse end (6.7 px over a whole slice against 5.8 px as it
 * lands on screen). Re-run the calibration against a new libplacebo. */
static const struct {
    float sigma;                /* per unit scale, normalised to full scale */
    float fwhm;                 /* pixels */
} h274_cal[13] = {
    { 0.040298f, 5.82f }, { 0.053825f, 4.60f }, { 0.071522f, 3.62f },
    { 0.086426f, 3.12f }, { 0.098096f, 2.60f }, { 0.118852f, 2.36f },
    { 0.124885f, 2.07f }, { 0.151900f, 1.78f }, { 0.164376f, 1.58f },
    { 0.177391f, 1.33f }, { 0.196781f, 1.27f }, { 0.213219f, 1.17f },
    { 0.223599f, 1.06f },
};

/* Closest cut-off index to a wanted blob size. Index 0 is the coarsest. */
static int h274_index_for(float fwhm)
{
    int best = 0;
    float berr = INFINITY;
    for (int i = 0; i < 13; i++) {
        float e = fabsf(h274_cal[i].fwhm - fwhm);
        if (e < berr) {
            berr = e;
            best = i;
        }
    }
    return best;
}

/* ------------------------------------------------------------- tables */

struct craw_grain_params craw_grain_default_params(void)
{
    return (struct craw_grain_params) {
        .gauge    = CRAW_GRAIN_GAUGE_SUPER35,
        .stock    = CRAW_GRAIN_STOCK_V3_250D,
        .response = CRAW_GRAIN_RESPONSE_PRINT,
        .model    = CRAW_GRAIN_MODEL_AUTO,
        .path     = CRAW_GRAIN_PATH_OUTPUT,
        .amount   = 1.0f,
        .size     = 1.0f,
        .chroma   = -1.0f,      /* negative means "the stock's own value" */
    };
}

const char *craw_grain_gauge_name(enum craw_grain_gauge g)
{
    return (unsigned)g < CRAW_GRAIN_GAUGE_NB ? gauges[g].name : "?";
}

const char *craw_grain_gauge_key(enum craw_grain_gauge g)
{
    return (unsigned)g < CRAW_GRAIN_GAUGE_NB ? gauges[g].key : "?";
}

float craw_grain_gauge_width_mm(enum craw_grain_gauge g)
{
    return (unsigned)g < CRAW_GRAIN_GAUGE_NB ? gauges[g].width_mm : 0.0f;
}

const char *craw_grain_gauge_source(enum craw_grain_gauge g)
{
    return (unsigned)g < CRAW_GRAIN_GAUGE_NB ? gauges[g].source : "?";
}

const char *craw_grain_stock_name(enum craw_grain_stock s)
{
    return (unsigned)s < CRAW_GRAIN_STOCK_NB ? stocks[s].name : "?";
}

const char *craw_grain_stock_key(enum craw_grain_stock s)
{
    return (unsigned)s < CRAW_GRAIN_STOCK_NB ? stocks[s].key : "?";
}

float craw_grain_stock_rms(enum craw_grain_stock s)
{
    return (unsigned)s < CRAW_GRAIN_STOCK_NB ? stocks[s].rms : 0.0f;
}

float craw_grain_stock_blob_um(enum craw_grain_stock s)
{
    return (unsigned)s < CRAW_GRAIN_STOCK_NB ? stocks[s].blob_um : 0.0f;
}

const char *craw_grain_stock_source(enum craw_grain_stock s)
{
    return (unsigned)s < CRAW_GRAIN_STOCK_NB ? stocks[s].source : "?";
}

static const char *const response_names[CRAW_GRAIN_RESPONSE_NB] = {
    "Negative", "Print", "Reversal", "Flat",
};

static const char *const response_keys[CRAW_GRAIN_RESPONSE_NB] = {
    "negative", "print", "reversal", "flat",
};

const char *craw_grain_response_name(enum craw_grain_response r)
{
    return (unsigned)r < CRAW_GRAIN_RESPONSE_NB ? response_names[r] : "?";
}

const char *craw_grain_response_note(enum craw_grain_response r)
{
    switch (r) {
    case CRAW_GRAIN_RESPONSE_NEGATIVE:
        return "Grain grows with exposure and vanishes at black, the way an "
               "inverted scan of a camera negative behaves.";
    case CRAW_GRAIN_RESPONSE_PRINT:
        return "Strongest through the mid-tones, swallowed by the print "
               "curve's toe and shoulder at both ends.";
    case CRAW_GRAIN_RESPONSE_REVERSAL:
        return "Peaks below mid grey where a reversal stock keeps its "
               "silver, and leaves the highlights nearly clean.";
    case CRAW_GRAIN_RESPONSE_FLAT:
        return "The same everywhere. Not a film response - it is what a "
               "sensor does, and it makes measurement unambiguous.";
    default:
        return "?";
    }
}

const char *craw_grain_model_name(enum craw_grain_model m)
{
    switch (m) {
    case CRAW_GRAIN_MODEL_AUTO: return "Auto";
    case CRAW_GRAIN_MODEL_AV1:  return "AV1";
    case CRAW_GRAIN_MODEL_H274: return "H.274";
    default:                    return "?";
    }
}

static int key_eq(const char *a, const char *b)
{
    for (; *a && *b; a++, b++) {
        int ca = *a >= 'A' && *a <= 'Z' ? *a + 32 : *a;
        int cb = *b >= 'A' && *b <= 'Z' ? *b + 32 : *b;
        if (ca != cb)
            return 0;
    }
    return !*a && !*b;
}

int craw_grain_gauge_from_key(const char *key)
{
    if (!key)
        return -1;
    for (int i = 0; i < CRAW_GRAIN_GAUGE_NB; i++)
        if (key_eq(key, gauges[i].key))
            return i;
    return -1;
}

int craw_grain_stock_from_key(const char *key)
{
    if (!key)
        return -1;
    for (int i = 0; i < CRAW_GRAIN_STOCK_NB; i++)
        if (key_eq(key, stocks[i].key))
            return i;
    return -1;
}

int craw_grain_response_from_key(const char *key)
{
    if (!key)
        return -1;
    for (int i = 0; i < CRAW_GRAIN_RESPONSE_NB; i++)
        if (key_eq(key, response_keys[i]))
            return i;
    return -1;
}

/* ------------------------------------------------------------- build */

#define NUM_POINTS 14           /* the AV1 maximum, and what libaom uses */

/* Where the scaling function's x positions land. On the output path the
 * signal is display-encoded, so evenly spaced positions are evenly spaced
 * perceptually; on the source path it is scene-linear, so the same
 * perceptual positions have to be pushed through a gamma first or the whole
 * response shape would be squashed into the top of the range. */
static float point_x(int i, enum craw_grain_path path)
{
    float x = i / (float)(NUM_POINTS - 1);
    return path == CRAW_GRAIN_PATH_SOURCE ? powf(x, GRAIN_GAMMA) : x;
}

/* The grain the physical model asks for is a density fluctuation, which is
 * a *relative* modulation of linear light: sigma_L/L = ln(10)*sigma_D at
 * every level. What the grain shaders add is an *absolute* amount to the
 * stored signal, so the relative figure has to be turned into an absolute
 * one in whichever domain the signal is in at the point it is applied.
 *
 * With a display encoding v = L^(1/gamma), a relative linear modulation
 * rho becomes an encoded amplitude of rho*v/gamma; going the other way, an
 * encoded amplitude A becomes a linear amplitude A*dL/dv = A*gamma*v^(gamma-1).
 *
 * `shape` is the response curve, already normalised to 1 at mid grey, so the
 * amplitude at mid grey is exactly the physical value whichever response is
 * picked and whichever path applies it. */
static float signal_amp(float rel, float shape, float x,
                        enum craw_grain_path path)
{
    float enc = rel * GRAIN_MID / GRAIN_GAMMA * shape;
    if (path == CRAW_GRAIN_PATH_SOURCE)
        return enc * GRAIN_GAMMA * powf(x, GRAIN_GAMMA - 1.0f);
    return enc;
}

/* The AV1 shader's grain is the template (standard deviation `sd` in 8-bit
 * units) times the scaling LUT, which is point/2^scaling_shift. So a point
 * value of y delivers a standard deviation of y*sd/(2^shift * 255) of full
 * scale. Inverted: the point value for a wanted sigma. */
static float point_for_sigma(float sigma, float sd, int scaling_shift)
{
    if (sd <= 0.0f)
        return 0.0f;
    return sigma * 255.0f * (float)(1 << scaling_shift) / sd;
}

/* Amplitude, in the applied signal's own units, at sample point i. */
static float point_amp(const struct craw_grain_params *p, float rel, int i)
{
    const float norm = response_weight(p->response, GRAIN_MID);
    float x = i / (float)(NUM_POINTS - 1);
    float w = response_weight(p->response, x);
    if (norm > 0.0f)
        w /= norm;
    return signal_amp(rel, w, x, p->path);
}

static void build_points(uint8_t points[][2], int *num,
                         const struct craw_grain_params *p,
                         float rel, float sd, int scaling_shift, int max_pts)
{
    int n = 0;
    int prev_x = -1;

    for (int i = 0; i < NUM_POINTS && n < max_pts; i++) {
        int xi = (int)lrintf(point_x(i, p->path) * 255.0f);
        if (xi <= prev_x)       /* the spec wants strictly increasing x */
            continue;
        prev_x = xi;

        float y = point_for_sigma(point_amp(p, rel, i), sd, scaling_shift);
        int yi = (int)lrintf(y);
        points[n][0] = (uint8_t)xi;
        points[n][1] = (uint8_t)(yi < 0 ? 0 : (yi > 255 ? 255 : yi));
        n++;
    }
    *num = n;
}

/* Does the wanted amplitude fit in the scaling function at this shift?
 * Smaller shifts give more amplitude but coarser quantisation, so take the
 * largest shift that still fits, which puts the points as high up the 0..255
 * range as they will go. */
static int pick_scaling_shift(const struct craw_grain_params *p,
                              float rel, float sd)
{
    float peak = 0.0f;
    for (int i = 0; i < NUM_POINTS; i++) {
        float a = point_amp(p, rel, i);
        if (a > peak)
            peak = a;
    }
    for (int shift = 11; shift >= 8; shift--) {
        if (point_for_sigma(peak, sd, shift) <= 255.0f)
            return shift;
    }
    return 8;
}

static int build_av1(const struct craw_grain_params *p,
                     struct craw_grain_out *out, float sigma, float blob)
{
    struct pl_av1_grain_data *d = &out->data.params.av1;
    const int bits = 8;         /* our textures are float; libplacebo's
                                 * bit_depth() falls back to 8 for those */
    float sd = 0.0f, fwhm = 0.0f;

    *d = (struct pl_av1_grain_data) {0};

    if (blob <= 1.10f) {
        /* One pixel or finer: no filter at all, which is what libaom's
         * photon noise tables use and what a film grain smaller than the
         * pixel that samples it actually looks like. */
        static int16_t buf[73][82];
        struct field_stats st;
        d->ar_coeff_lag = 0;
        d->ar_coeff_shift = 6;
        d->grain_scale_shift = 0;
        craw_grain_template_y(buf, d, bits, AR_SOLVE_SEED);
        field_measure(buf, true, bits, AR_SOLVE_SEED, true, &st);
        sd = st.sd;
        fwhm = st.fwhm;
    } else {
        ar_solve(blob, bits, d, &sd, &fwhm);
    }
    out->blob_clamped = fwhm;

    d->overlap = true;          /* without it the 32x32 block grid shows */
    d->scaling_shift = pick_scaling_shift(p, sigma, sd);
    build_points(d->points_y, &d->num_points_y, p, sigma, sd,
                 d->scaling_shift, 14);

    /* The image is RGB at both hook points, and libplacebo maps RGB onto
     * the grain model as G->luma, B->Cb, R->Cr. Luma-only grain would
     * therefore be green-only grain, so the chroma channels always get
     * points too; `chroma` sets how loud they are. */
    float chroma = p->chroma;
    if (chroma < 0.0f)
        chroma = stocks[p->stock].chroma;
    if (chroma > 0.0f) {
        static int16_t buf_y[73][82], buf_uv[73][82];
        struct field_stats st;
        craw_grain_template_y(buf_y, d, bits, AR_SOLVE_SEED);
        for (int c = 0; c < 2; c++) {
            memcpy(d->ar_coeffs_uv[c], d->ar_coeffs_y, 24);
            /* The coefficient one past the luma set is the cross term: it
             * mixes the luma template into this channel's, which is what
             * decides whether the three channels' grain is the same grain
             * (monochrome, like a print) or three independent grains (like
             * three dye layers). Full independence at chroma 1. */
            float corr = 1.0f - (chroma > 1.0f ? 1.0f : chroma);
            d->ar_coeffs_uv[c][2 * d->ar_coeff_lag * (d->ar_coeff_lag + 1)] =
                (int8_t)lrintf(corr * (float)(1 << d->ar_coeff_shift));
            /* Solve this channel's points against its own template, not the
             * luma one - see template_uv(). */
            template_uv(buf_uv, buf_y, d, c, bits, AR_SOLVE_SEED);
            field_measure(buf_uv, d->overlap, bits, AR_SOLVE_SEED, false, &st);
            build_points(d->points_uv[c], &d->num_points_uv[c], p,
                         sigma * chroma, st.sd, d->scaling_shift, 10);
            /* Index the chroma scaling function by the channel's own value,
             * not by luma: each dye layer's grain follows its own density.
             * uv_mult is in 64ths, so 64 is unity. */
            d->uv_mult[c] = 64;
            d->uv_mult_luma[c] = 0;
            d->uv_offset[c] = 0;
        }
    }

    out->model_used = CRAW_GRAIN_MODEL_AV1;
    out->data.type = PL_FILM_GRAIN_AV1;
    out->data.seed = p->seed & 0xFFFFu;

    /* What the quantised points will actually deliver at the reference
     * level, read back off the scaling function the same way the shader
     * will read it. */
    const float ref = 255.0f * (p->path == CRAW_GRAIN_PATH_SOURCE
                                ? powf(GRAIN_MID, GRAIN_GAMMA) : GRAIN_MID);
    float scale = 0.0f;
    if (d->num_points_y > 0) {
        scale = d->points_y[0][1];
        for (int i = 0; i + 1 < d->num_points_y; i++) {
            float x0 = d->points_y[i][0], x1 = d->points_y[i + 1][0];
            if (ref >= x0 && ref <= x1) {
                float t = (ref - x0) / (x1 - x0);
                scale = d->points_y[i][1] +
                        t * (d->points_y[i + 1][1] - d->points_y[i][1]);
                break;
            }
            if (ref > x1)
                scale = d->points_y[i + 1][1];
        }
    }
    out->unit_sd = sd / ((float)(1 << d->scaling_shift) * 255.0f);
    out->sigma_built = scale * out->unit_sd;
    return 0;
}

/* Which model component reaches which picture channel. Luma first. See the
 * comment in build_h274(). */
static const int model_for_channel[3] = { 1, 0, 2 };

/* H.274 intensity intervals are piecewise constant rather than piecewise
 * linear, so the response curve has to be sampled into steps. Sixteen steps
 * of sixteen levels, offset by half a step so that one interval is centred
 * exactly on mid grey and the amplitude there is the physical value rather
 * than the response curve read half a step off; that leaves a half-width
 * interval at each end, hence seventeen. The count is a compromise:
 * libplacebo unrolls one branch and two uniforms per interval per component
 * in the shader, so the specification's maximum of 256 is not free. */
#define H274_INTERVALS 17
#define H274_STEP 16

static void h274_interval(int i, int *lo, int *hi)
{
    *lo = i == 0 ? 0 : i * H274_STEP - H274_STEP / 2;
    *hi = i == H274_INTERVALS - 1 ? 255
        : (i + 1) * H274_STEP - H274_STEP / 2 - 1;
}

static int build_h274(const struct craw_grain_params *p,
                      struct craw_grain_out *out, float sigma, float blob)
{
    struct pl_h274_grain_data *d = &out->data.params.h274;
    static uint8_t lower[3][H274_INTERVALS], upper[3][H274_INTERVALS];
    static int16_t values[3][H274_INTERVALS][6];

    const int idx = h274_index_for(blob);
    out->blob_clamped = h274_cal[idx].fwhm;

    float chroma = p->chroma;
    if (chroma < 0.0f)
        chroma = stocks[p->stock].chroma;

    *d = (struct pl_h274_grain_data) {
        /* libplacebo only implements model 0, the frequency filtering one;
         * pl_needs_fg_h274() rejects model 1 outright. */
        .model_id = 0,
        .blending_mode_id = 0,  /* additive. libplacebo ignores this field
                                 * and is always additive anyway. */
    };

    /* scale_factor in the shader is 1/2^(log2_scale_factor+6), and the
     * per-interval scale is an 8-bit value, so choose the shift that puts
     * the loudest interval near the top of 0..255. */
    const float norm = response_weight(p->response, GRAIN_MID);
    float peak = 0.0f;
    for (int i = 0; i < H274_INTERVALS; i++) {
        /* Intervals are indexed by the stored signal value, so on the
         * source path the perceptual position of interval i is the gamma
         * root of where it sits in the range. */
        int lo, hi;
        h274_interval(i, &lo, &hi);
        float pos = 0.5f * (lo + hi) / 255.0f;
        float x = p->path == CRAW_GRAIN_PATH_SOURCE
                ? powf(pos, 1.0f / GRAIN_GAMMA) : pos;
        float w = response_weight(p->response, x);
        if (norm > 0.0f)
            w /= norm;
        float a = signal_amp(sigma, w, x, p->path);
        if (a > peak)
            peak = a;
    }
    int log2s = 0;
    while (log2s < 15) {
        float s = peak * (float)(1 << (log2s + 6)) / h274_cal[idx].sigma;
        if (s > 255.0f)
            break;
        log2s++;
    }
    log2s = log2s > 0 ? log2s - 1 : 0;
    d->log2_scale_factor = log2s;

    /* Which model component libplacebo will apply to which picture channel.
     *
     * libplacebo 7.371's H.274 shader loops over the picture's components,
     * maps each to a grain channel with channel_map(), and then uses that
     * *channel* index to subscript `color[]` as well as to pick the model -
     * see the `color[%d] += scale * grain` at the end of pl_shader_fg_h274().
     * For YCbCr with the usual plane order the two indices coincide and
     * nothing is wrong, but for an RGB image channel_map sends R to CR, G to
     * Y and B to CB, so model component 0 lands on red, 1 on green and 2 on
     * blue. Measured, not deduced: with only component 0 present the green
     * channel came back with no grain at all and red carried it.
     *
     * libplacebo's AV1 shader has no such problem - it subscripts color[]
     * by the component index and only the LUT by the channel - so this
     * shuffle is H.274's alone. The luma parameters therefore go in the
     * slot that reaches green. If a later libplacebo fixes the indexing
     * this becomes wrong and has to go back to { 0, 1, 2 } with luma in 0. */
    for (int k = 0; k < 3; k++) {
        const int c = model_for_channel[k];
        float chan = k == 0 ? 1.0f : chroma;
        if (chan <= 0.0f)
            continue;
        d->component_model_present[c] = true;
        d->num_intensity_intervals[c] = H274_INTERVALS;
        d->num_model_values[c] = 3;     /* scale, then the two cut-offs */
        for (int i = 0; i < H274_INTERVALS; i++) {
            int lo, hi;
            h274_interval(i, &lo, &hi);
            lower[c][i] = (uint8_t)lo;
            upper[c][i] = (uint8_t)hi;
            float pos = 0.5f * (lo + hi) / 255.0f;
            float x = p->path == CRAW_GRAIN_PATH_SOURCE
                    ? powf(pos, 1.0f / GRAIN_GAMMA) : pos;
            float w = response_weight(p->response, x);
            if (norm > 0.0f)
                w /= norm;
            float s = signal_amp(sigma * chan, w, x, p->path) *
                      (float)(1 << (log2s + 6)) / h274_cal[idx].sigma;
            int si = (int)lrintf(s);
            values[c][i][0] = (int16_t)(si < 0 ? 0 : (si > 255 ? 255 : si));
            /* The cut-off index is offset by 2: libplacebo clamps
             * comp_model_value to [2,14] and subtracts 2 to get its 0..12
             * table index. */
            values[c][i][1] = (int16_t)(idx + 2);
            values[c][i][2] = (int16_t)(idx + 2);
            for (int j = 3; j < 6; j++)
                values[c][i][j] = 0;
        }
        d->intensity_interval_lower_bound[c] = lower[c];
        d->intensity_interval_upper_bound[c] = upper[c];
        d->comp_model_value[c] = (const int16_t (*)[6])values[c];
    }

    out->model_used = CRAW_GRAIN_MODEL_H274;
    out->data.type = PL_FILM_GRAIN_H274;
    out->data.seed = p->seed;

    /* The interval that contains the reference level. On the output path
     * that is interval 8, whose centre is mid grey exactly; on the source
     * path the reference sits wherever the gamma puts it and the reported
     * amplitude is a step's worth off. */
    float refpos = p->path == CRAW_GRAIN_PATH_SOURCE
                 ? powf(GRAIN_MID, GRAIN_GAMMA) : GRAIN_MID;
    const int ref255 = (int)lrintf(refpos * 255.0f);
    int mid = H274_INTERVALS - 1;
    for (int i = 0; i < H274_INTERVALS; i++) {
        int lo, hi;
        h274_interval(i, &lo, &hi);
        if (ref255 >= lo && ref255 <= hi) {
            mid = i;
            break;
        }
    }
    const int luma_c = model_for_channel[0];
    out->unit_sd = h274_cal[idx].sigma / (float)(1 << (log2s + 6));
    out->sigma_built = d->component_model_present[luma_c]
        ? values[luma_c][mid][0] * out->unit_sd : 0.0f;
    return 0;
}

int craw_grain_build(const struct craw_grain_params *p,
                     struct craw_grain_out *out)
{
    if (!p || !out)
        return -EINVAL;
    if ((unsigned)p->gauge >= CRAW_GRAIN_GAUGE_NB ||
        (unsigned)p->stock >= CRAW_GRAIN_STOCK_NB ||
        (unsigned)p->response >= CRAW_GRAIN_RESPONSE_NB ||
        (unsigned)p->model >= CRAW_GRAIN_MODEL_NB ||
        (unsigned)p->path >= CRAW_GRAIN_PATH_NB)
        return -EINVAL;
    if (p->width <= 0 || p->height <= 0)
        return -EINVAL;

    memset(out, 0, sizeof(*out));

    const struct gauge_info *g = &gauges[p->gauge];
    const struct stock_info *s = &stocks[p->stock];

    /* Section 1 of grain.h, in four lines. */
    out->pitch_um = 1000.0f * g->width_mm / (float)p->width;
    out->blob_um = s->blob_um * (p->size > 0.0f ? p->size : 1.0f);
    out->blob_px = out->blob_um / out->pitch_um;
    const float aperture = out->pitch_um > out->blob_um ? out->pitch_um
                                                        : out->blob_um;
    out->sigma_density = (s->rms / 1000.0f) * 48.0f / aperture;
    out->sigma_rel = logf(10.0f) * out->sigma_density * p->amount;
    out->sigma_signal = signal_amp(out->sigma_rel, 1.0f, GRAIN_MID, p->path);

    if (p->amount <= 0.0f || out->sigma_rel <= 0.0f) {
        out->enabled = false;
        out->data.type = PL_FILM_GRAIN_NONE;
        return 0;
    }

    /* A blob finer than a pixel is just white noise; the amplitude already
     * accounts for the extra averaging through Selwyn's law. */
    float blob = out->blob_px < 1.0f ? 1.0f : out->blob_px;

    enum craw_grain_model model = p->model;
    if (model == CRAW_GRAIN_MODEL_AUTO) {
        /* AV1, at every size. H.274's model 0 has the better handle on blob
         * size on paper - a ladder of thirteen cut-off frequencies instead
         * of one filter coefficient to tune - and it can vary the blob size
         * with intensity, which AV1 cannot. But libplacebo 7.371 does not
         * implement its deblocking (there is a `// TODO: Deblocking?` where
         * it would go), and it lays the grain down as 16x16 patches cut from
         * a 64x64 database with no blending between them. At one pixel that
         * is invisible; by three it is a quilt, and at five it is the only
         * thing you can see. The AV1 shader does implement the 27/17 block
         * overlap, so its grain stays seamless right up to its own ceiling
         * of about 6.4 px - which is coarser than H.274's 5.8 px anyway.
         * The pictures are in the scratchpad's sheet_blocks.png. */
        model = CRAW_GRAIN_MODEL_AV1;
    }

    int ret = model == CRAW_GRAIN_MODEL_H274
        ? build_h274(p, out, out->sigma_rel, blob)
        : build_av1(p, out, out->sigma_rel, blob);
    if (ret < 0)
        return ret;

    out->enabled = true;
    if (craw_grain_validate(&out->data)) {
        out->enabled = false;
        out->data.type = PL_FILM_GRAIN_NONE;
        return -ERANGE;
    }
    return 0;
}

/* --------------------------------------------------------- validation */

static const char *validate_av1(const struct pl_av1_grain_data *d)
{
    if (d->num_points_y < 0 || d->num_points_y > 14)
        return "num_points_y outside 0..14";
    for (int i = 1; i < d->num_points_y; i++)
        if (d->points_y[i][0] <= d->points_y[i - 1][0])
            return "points_y values are not strictly increasing";
    for (int c = 0; c < 2; c++) {
        if (d->num_points_uv[c] < 0 || d->num_points_uv[c] > 10)
            return "num_points_uv outside 0..10";
        for (int i = 1; i < d->num_points_uv[c]; i++)
            if (d->points_uv[c][i][0] <= d->points_uv[c][i - 1][0])
                return "points_uv values are not strictly increasing";
    }
    if (d->scaling_shift < 8 || d->scaling_shift > 11)
        return "scaling_shift outside 8..11";
    if (d->ar_coeff_lag < 0 || d->ar_coeff_lag > 3)
        return "ar_coeff_lag outside 0..3";
    if (d->ar_coeff_shift < 6 || d->ar_coeff_shift > 9)
        return "ar_coeff_shift outside 6..9";
    if (d->grain_scale_shift < 0 || d->grain_scale_shift > 3)
        return "grain_scale_shift outside 0..3";
    for (int c = 0; c < 2; c++)
        if (d->uv_offset[c] < -256 || d->uv_offset[c] > 255)
            return "uv_offset outside -256..255";
    /* libplacebo asserts 12 - bits + grain_scale_shift >= 0; with its
     * 8..12 bit clamp the worst case is 12 - 12 + 0, so this only fails on
     * a negative grain_scale_shift, which is caught above. Kept explicit
     * because it is the one constraint that is libplacebo's and not the
     * specification's. */
    if (12 - 12 + d->grain_scale_shift < 0)
        return "grain_scale_shift would make libplacebo's shift negative";
    /* Coefficients past the lag's own count are never read, but leaving
     * junk there would make two otherwise identical parameter sets compare
     * unequal in libplacebo's reuse check and force a pointless reupload
     * every frame. */
    const int used = 2 * d->ar_coeff_lag * (d->ar_coeff_lag + 1);
    for (int i = used; i < 24; i++)
        if (d->ar_coeffs_y[i])
            return "ar_coeffs_y has nonzero entries past ar_coeff_lag";
    return NULL;
}

static const char *validate_h274(const struct pl_h274_grain_data *d)
{
    if (d->model_id != 0)
        return "model_id must be 0; libplacebo does not implement model 1";
    if (d->blending_mode_id != 0 && d->blending_mode_id != 1)
        return "blending_mode_id outside 0..1";
    if (d->log2_scale_factor < 0 || d->log2_scale_factor > 15)
        return "log2_scale_factor outside 0..15";
    bool any = false;
    for (int c = 0; c < 3; c++) {
        if (!d->component_model_present[c])
            continue;
        any = true;
        if (d->num_intensity_intervals[c] < 1 ||
            d->num_intensity_intervals[c] > 256)
            return "num_intensity_intervals outside 1..256";
        if (d->num_model_values[c] < 1 || d->num_model_values[c] > 6)
            return "num_model_values outside 1..6";
        if (!d->intensity_interval_lower_bound[c] ||
            !d->intensity_interval_upper_bound[c] || !d->comp_model_value[c])
            return "component model present but its arrays are NULL";
        for (int i = 0; i < d->num_intensity_intervals[c]; i++) {
            if (d->intensity_interval_lower_bound[c][i] >
                d->intensity_interval_upper_bound[c][i])
                return "intensity interval lower bound above upper bound";
            if (i && d->intensity_interval_lower_bound[c][i] <=
                     d->intensity_interval_upper_bound[c][i - 1])
                return "intensity intervals overlap or are unordered";
            /* libplacebo packs the scale into the top 16 bits of a uint and
             * reads it back unsigned, so a negative scale would come out as
             * a huge positive one. */
            if (d->comp_model_value[c][i][0] < 0 ||
                d->comp_model_value[c][i][0] > 255)
                return "comp_model_value scale outside 0..255";
            if (d->num_model_values[c] > 1) {
                int16_t h = d->comp_model_value[c][i][1];
                if (h < 2 || h > 14)
                    return "horizontal cut-off outside 2..14";
            }
            if (d->num_model_values[c] > 2) {
                int16_t v = d->comp_model_value[c][i][2];
                if (v < 2 || v > 14)
                    return "vertical cut-off outside 2..14";
            }
        }
    }
    if (!any)
        return "no component model present";
    return NULL;
}

float craw_grain_sigma_at(const struct craw_grain_out *out, float level)
{
    if (!out || !out->enabled)
        return 0.0f;
    const float idx = level * 255.0f;
    /* Interval bounds are whole levels, so a query landing between two of
     * them has to round rather than fall through the crack. */
    const int idxi = (int)lrintf(idx);

    if (out->data.type == PL_FILM_GRAIN_AV1) {
        const struct pl_av1_grain_data *d = &out->data.params.av1;
        if (d->num_points_y < 1)
            return 0.0f;
        /* The scaling LUT is flat before the first point and after the
         * last, and linear in between: AV1 7.18.3.4. */
        if (idx <= d->points_y[0][0])
            return d->points_y[0][1] * out->unit_sd;
        for (int i = 0; i + 1 < d->num_points_y; i++) {
            float x0 = d->points_y[i][0], x1 = d->points_y[i + 1][0];
            if (idx <= x1) {
                float t = (idx - x0) / (x1 - x0);
                return (d->points_y[i][1] +
                        t * (d->points_y[i + 1][1] - d->points_y[i][1])) *
                       out->unit_sd;
            }
        }
        return d->points_y[d->num_points_y - 1][1] * out->unit_sd;
    }

    if (out->data.type == PL_FILM_GRAIN_H274) {
        const struct pl_h274_grain_data *d = &out->data.params.h274;
        const int c = model_for_channel[0];
        if (!d->component_model_present[c])
            return 0.0f;
        for (int i = 0; i < d->num_intensity_intervals[c]; i++) {
            if (idxi >= d->intensity_interval_lower_bound[c][i] &&
                idxi <= d->intensity_interval_upper_bound[c][i])
                return d->comp_model_value[c][i][0] * out->unit_sd;
        }
        /* Outside every interval the shader picks a scale of zero, which is
         * how H.274 leaves values above its top bound with no grain at all
         * rather than clipping them. */
        return 0.0f;
    }
    return 0.0f;
}

const char *craw_grain_validate(const struct pl_film_grain_data *d)
{
    if (!d)
        return "NULL grain data";
    switch (d->type) {
    case PL_FILM_GRAIN_NONE: return NULL;
    case PL_FILM_GRAIN_AV1:  return validate_av1(&d->params.av1);
    case PL_FILM_GRAIN_H274: return validate_h274(&d->params.h274);
    default:                 return "unknown film grain type";
    }
}

/* --------------------------------------------------------------- hook */

struct craw_grain_hook {
    pl_gpu gpu;
    struct pl_hook hook;
    pl_shader_obj state;
    struct craw_grain_params last;
    struct craw_grain_out out;
    bool valid;
};

static struct pl_hook_res grain_hook_fn(void *priv,
                                        const struct pl_hook_params *hp)
{
    craw_grain_hook *h = priv;
    struct pl_hook_res res = { .failed = true };

    if (!h->valid || !h->out.enabled || !hp->tex)
        return res;

    /* The image at PL_HOOK_PRE_OUTPUT has already been colour managed and
     * encoded for the target, so it is RGB in [0,1] whatever the source
     * was. That is why the grain shaders' clamps cost nothing here. */
    struct pl_color_repr repr = hp->repr;
    repr.sys = PL_COLOR_SYSTEM_RGB;
    repr.levels = PL_COLOR_LEVELS_FULL;

    struct pl_film_grain_params fg = {
        .data = h->out.data,
        .tex = hp->tex,
        .repr = &repr,
        .components = hp->components < 3 ? hp->components : 3,
        .component_mapping = { 0, 1, 2, -1 },
    };

    if (!pl_needs_film_grain(&fg))
        return res;

    pl_shader sh = pl_dispatch_begin(hp->dispatch);
    if (!pl_shader_film_grain(sh, &h->state, &fg)) {
        pl_dispatch_abort(hp->dispatch, &sh);
        return res;
    }

    return (struct pl_hook_res) {
        .output = PL_HOOK_SIG_COLOR,
        .sh = sh,
        .repr = repr,
        .color = hp->color,
        .components = hp->components,
        .rect = hp->rect,
    };
}

craw_grain_hook *craw_grain_hook_create(pl_gpu gpu)
{
    craw_grain_hook *h = calloc(1, sizeof(*h));
    if (!h)
        return NULL;
    h->gpu = gpu;
    h->hook = (struct pl_hook) {
        /* After colour management and the main scaler, before blending and
         * rotation: target resolution, display encoding, and the video only
         * - an overlay drawn later does not get grained. */
        .stages = PL_HOOK_PRE_OUTPUT,
        .input = PL_HOOK_SIG_TEX,
        .priv = h,
        .hook = grain_hook_fn,
        .signature = 0x637261776772616eULL,   /* "crawgran" */
    };
    return h;
}

void craw_grain_hook_destroy(craw_grain_hook **ph)
{
    craw_grain_hook *h = ph ? *ph : NULL;
    if (!h)
        return;
    pl_shader_obj_destroy(&h->state);
    free(h);
    *ph = NULL;
}

const struct pl_hook *craw_grain_hook_update(craw_grain_hook *h,
                                             const struct craw_grain_params *p,
                                             struct craw_grain_out *out)
{
    if (!h || !p)
        return NULL;

    /* Solving for the filter strength costs a couple of dozen template
     * generations, so only redo it when something other than the seed has
     * moved. The seed is the one thing that changes every frame. */
    struct craw_grain_params key = *p;
    key.seed = 0;
    struct craw_grain_params lastkey = h->last;
    lastkey.seed = 0;

    if (!h->valid || memcmp(&key, &lastkey, sizeof(key))) {
        h->valid = craw_grain_build(p, &h->out) == 0;
        h->last = *p;
    }
    if (h->valid) {
        h->out.data.seed = h->out.data.type == PL_FILM_GRAIN_AV1
            ? (p->seed & 0xFFFFu) : p->seed;
    }
    if (out)
        *out = h->out;
    if (!h->valid || !h->out.enabled)
        return NULL;
    return &h->hook;
}
