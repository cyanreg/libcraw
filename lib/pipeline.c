/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* The GPU pipeline: owns the compute shaders, the intermediate images and
 * the auto white balance buffers, and records one frame's worth of
 * dispatches and barriers into a command buffer the caller owns. See
 * DESIGN.md sections 4, 5 and 10. */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <craw/pipeline.h>

#include "vk_runtime.h"
#include "gamut.h"
#include "sharpen.h"

#include "debayer_bilinear.spv.h"
#include "debayer_mhc.spv.h"
#include "rcd_dir.spv.h"
#include "rcd_green.spv.h"
#include "rcd_rb.spv.h"
#include "color.spv.h"
#include "awb_hist.spv.h"
#include "denoise_nlm.spv.h"
#include "lens.spv.h"
#include "false_color.spv.h"
#include "chroma_down.spv.h"
#include "chroma_blur.spv.h"
#include "chroma_up.spv.h"
#include "highlight_field.spv.h"
#include "highlight_pull.spv.h"
#include "highlight_push.spv.h"
#include "highlight_fill.spv.h"
#include "defringe_prep.spv.h"
#include "defringe.spv.h"
#include "sharpen_prep.spv.h"
#include "sharpen.spv.h"
#include "sharpen_rows.spv.h"

/* Every debayer pass shares this block; it mirrors CRAWPrepPushBuf in
 * lib/shaders/common.glsl field for field. Scalar layout gives every member
 * 4-byte alignment and no padding, which is exactly what C gives these
 * types, so the static asserts below are the whole ABI check.
 *
 * inv_range is 1 / (white - black) rather than the white level itself: the
 * prep of DESIGN.md 4.1 divides by the range at every one of the up to 49
 * samples an RCD invocation reads, and a reciprocal computed once on the
 * host costs at most an ulp against the tolerances of section 11.
 *
 * `clip` is 1.0, the balanced clip of 4.1, except on the mosaic the
 * highlight reconstruction of DESIGN.md 18 rebuilt, which is read with an
 * identity prep and no clip at all. */
typedef struct CRAWPrepPush {
    int32_t size[2];
    float   black[4];
    float   inv_range[4];
    float   wb[4];        /* as-shot gains in [0..2], [3] unused */
    float   clip;
} CRAWPrepPush;

static_assert(sizeof(CRAWPrepPush) == 60, "CRAWPrepPush layout");
static_assert(offsetof(CRAWPrepPush, black) == 8, "CRAWPrepPush layout");
static_assert(offsetof(CRAWPrepPush, inv_range) == 24, "CRAWPrepPush layout");
static_assert(offsetof(CRAWPrepPush, wb) == 40, "CRAWPrepPush layout");
static_assert(offsetof(CRAWPrepPush, clip) == 56, "CRAWPrepPush layout");

/* Mirrors CRAWHighlightPushBuf in lib/shaders/highlight.glsl, which the
 * colour field's level 0 and the fill share: the prep's constants without
 * its clip, and the field's size. */
typedef struct CRAWHighlightPush {
    int32_t size[2];      /* mosaic */
    int32_t fsize[2];     /* field level 0, ceil(size / 4) */
    float   black[4];
    float   inv_range[4];
    float   wb[4];        /* the frame's gains in [0..2], [3] unused */
} CRAWHighlightPush;

static_assert(sizeof(CRAWHighlightPush) == 64, "CRAWHighlightPush layout");
static_assert(offsetof(CRAWHighlightPush, fsize) == 8, "CRAWHighlightPush layout");
static_assert(offsetof(CRAWHighlightPush, black) == 16, "CRAWHighlightPush layout");
static_assert(offsetof(CRAWHighlightPush, wb) == 48, "CRAWHighlightPush layout");

/* Mirrors CRAWHighlightLevelPushBuf in lib/shaders/highlight.glsl, for one
 * step of the pull or the push between two adjacent levels. */
typedef struct CRAWHighlightLevelPush {
    int32_t fine[2];
    int32_t coarse[2];
    int32_t top;          /* push: `coarse` is the top level, still sums */
} CRAWHighlightLevelPush;

static_assert(sizeof(CRAWHighlightLevelPush) == 20, "CRAWHighlightLevelPush layout");
static_assert(offsetof(CRAWHighlightLevelPush, coarse) == 8, "CRAWHighlightLevelPush layout");
static_assert(offsetof(CRAWHighlightLevelPush, top) == 16, "CRAWHighlightLevelPush layout");

/* Mirrors CRAWDenoisePushBuf in lib/shaders/denoise_nlm.comp.glsl. The
 * transform constants are all derived from the frame's (a, b) on the host,
 * so the shader never divides by a value it cannot check and the CPU
 * reference in lib/ref/ref_denoise.c can be handed the same numbers. The
 * last three are the temporal pass of DESIGN.md 14.7 and which of the two
 * filters run. */
typedef struct CRAWDenoisePush {
    int32_t size[2];       /* plane size, i.e. the mosaic halved */
    float   black[4];      /* black level per quad site, UNORM */
    float   inv_range[4];  /* 1 / (white - black) per quad site */
    float   range[4];      /* white - black per quad site */
    float   gat_a;         /* noise model a */
    float   gat_c;         /* 3 a^2 / 8 + b */
    float   gat_sc;        /* sqrt(gat_c) */
    float   gat_fwd;       /* 2 / a */
    float   gat_inv;       /* a / 4 */
    float   gat_off;       /* gat_c / a */
    float   inv_h2;        /* 1 / strength^2 */
    float   guard_lo;      /* highlight ramp start, in u */
    float   guard_inv;     /* 1 / (guard_hi - guard_lo) */
    float   t_alpha;       /* alpha_max */
    float   t_inv_h2;      /* 1 / ht^2 */
    int32_t mode;          /* CRAW_DN_MODE_* */
} CRAWDenoisePush;

static_assert(sizeof(CRAWDenoisePush) == 104, "CRAWDenoisePush layout");
static_assert(offsetof(CRAWDenoisePush, black) == 8, "CRAWDenoisePush layout");
static_assert(offsetof(CRAWDenoisePush, range) == 40, "CRAWDenoisePush layout");
static_assert(offsetof(CRAWDenoisePush, gat_a) == 56, "CRAWDenoisePush layout");
static_assert(offsetof(CRAWDenoisePush, inv_h2) == 80, "CRAWDenoisePush layout");
static_assert(offsetof(CRAWDenoisePush, t_alpha) == 92, "CRAWDenoisePush layout");
static_assert(offsetof(CRAWDenoisePush, mode) == 100, "CRAWDenoisePush layout");

/* Mirrors CRAW_DN_* in lib/shaders/denoise_nlm.comp.glsl: which of the two
 * filters of the raw stage run, and whether the temporal one has a
 * previous frame to read. */
#define CRAW_DN_MODE_SPATIAL  1
#define CRAW_DN_MODE_TEMPORAL 2
#define CRAW_DN_MODE_HISTORY  4

/* Mirrors CRAWColorPushBuf in lib/shaders/color.comp.glsl. `matrix` is row
 * major, as everything in libcraw is, and the shader reads it as three
 * vec3 rows and does the three dot products by hand: a GLSL mat3 would be
 * three *columns*, so this avoids a transpose that would have to be undone
 * somewhere. Scalar layout makes the GLSL array stride 12, matching
 * float[3][3]. */
typedef struct CRAWColorPush {
    int32_t  size[2];
    float    matrix[3][3];   /* M_out * cam_to_xyz * gain * 2^exposure_stops */
    float    wb_ratio[3];    /* frame wb / as-shot wb */
    CRAWGamutCompress gc;    /* ACES 1.3 RGC constants, only read when asked */
    uint32_t flags;
} CRAWColorPush;

static_assert(sizeof(CRAWColorPush) == 92, "CRAWColorPush layout");
static_assert(offsetof(CRAWColorPush, matrix) == 8, "CRAWColorPush layout");
static_assert(offsetof(CRAWColorPush, wb_ratio) == 44, "CRAWColorPush layout");
static_assert(offsetof(CRAWColorPush, gc) == 56, "CRAWColorPush layout");
static_assert(offsetof(CRAWColorPush, flags) == 88, "CRAWColorPush layout");

/* Mirrors CRAW_COLOR_GAMUT_COMPRESS in lib/shaders/color.comp.glsl. It is
 * not CRAW_FRAME_FLAG_NO_GAMUT_COMPRESSION: the public flag is a veto and
 * this is the positive answer the host works out from it and the output
 * space together. */
#define CRAW_COLOR_PUSH_GAMUT_COMPRESS 2u

/* Mirrors CRAWLensPushBuf in lib/shaders/lens.comp.glsl. The centre and
 * the normalizing radius are turned into pixels on the host, because they
 * depend only on the frame size and the record's own centre and a shader
 * that recomputed them per invocation would be doing the same divide
 * seventeen million times. Scalar layout gives the vec4 array a stride of
 * 16, matching float[3][4]. */
typedef struct CRAWLensPush {
    int32_t size[2];
    float   centre[2];      /* optical centre in pixels, pixel centre origin */
    float   inv_half_diag;  /* 1 / the corner distance from `centre` */
    float   k[3][4];        /* k0..k3 per channel, R G B */
    float   inv_scale;      /* 1 / s of DESIGN.md 16.8, exactly 1.0 to fill */
    int32_t keep_all;       /* 1: black outside the mapped source */
} CRAWLensPush;

static_assert(sizeof(CRAWLensPush) == 76, "CRAWLensPush layout");
static_assert(offsetof(CRAWLensPush, centre) == 8, "CRAWLensPush layout");
static_assert(offsetof(CRAWLensPush, inv_half_diag) == 16, "CRAWLensPush layout");
static_assert(offsetof(CRAWLensPush, k) == 20, "CRAWLensPush layout");
static_assert(offsetof(CRAWLensPush, inv_scale) == 68, "CRAWLensPush layout");
static_assert(offsetof(CRAWLensPush, keep_all) == 72, "CRAWLensPush layout");

/* Mirrors CRAWFalseColorPushBuf in lib/shaders/false_color.comp.glsl. */
typedef struct CRAWFalseColorPush {
    int32_t size[2];
    int32_t passes;        /* 1 to CRAW_FALSE_COLOR_MAX_PASSES */
} CRAWFalseColorPush;

static_assert(sizeof(CRAWFalseColorPush) == 12, "CRAWFalseColorPush layout");
static_assert(offsetof(CRAWFalseColorPush, passes) == 8, "CRAWFalseColorPush layout");

/* Mirrors CRAWDefringePushBuf in lib/shaders/defringe.glsl, which both
 * defringe passes read. */
typedef struct CRAWDefringePush {
    int32_t size[2];
    float   strength;      /* 0 < strength <= 1 */
} CRAWDefringePush;

static_assert(sizeof(CRAWDefringePush) == 12, "CRAWDefringePush layout");
static_assert(offsetof(CRAWDefringePush, strength) == 8, "CRAWDefringePush layout");

/* Mirrors CRAWSharpenPushBuf in lib/shaders/sharpen.glsl, which both
 * sharpening passes read. The taps and the noise coefficients come from
 * lib/sharpen.h, which the CPU reference uses too. */
typedef struct CRAWSharpenPush {
    int32_t size[2];
    int32_t stride;        /* a plane's row, in chunks of eight pixels */
    float   k[CRAW_SH_RMAX + 1];  /* taps up to the reach, which is the
                                   * iteration's specialization constant */
    float   contrast;      /* the mask's threshold in noise units, 0: none */
    float   noise_a;       /* the contrast statistic's modelled variance */
    float   noise_b;       /* at level L is noise_a * L + noise_b */
} CRAWSharpenPush;

static_assert(sizeof(CRAWSharpenPush) == 52, "CRAWSharpenPush layout");
static_assert(offsetof(CRAWSharpenPush, stride) == 8, "CRAWSharpenPush layout");
static_assert(offsetof(CRAWSharpenPush, k) == 12, "CRAWSharpenPush layout");
static_assert(offsetof(CRAWSharpenPush, contrast) == 40, "CRAWSharpenPush layout");
static_assert(offsetof(CRAWSharpenPush, noise_b) == 48, "CRAWSharpenPush layout");

/* Mirrors CRAWChromaPushBuf in the three lib/shaders/chroma_*.comp.glsl;
 * each pass reads the fields it needs. */
typedef struct CRAWChromaPush {
    int32_t size[2];       /* full resolution */
    int32_t qsize[2];      /* ceil(size / 4) */
    float   strength;      /* 0 < strength <= 1 */
} CRAWChromaPush;

static_assert(sizeof(CRAWChromaPush) == 20, "CRAWChromaPush layout");
static_assert(offsetof(CRAWChromaPush, qsize) == 8, "CRAWChromaPush layout");
static_assert(offsetof(CRAWChromaPush, strength) == 16, "CRAWChromaPush layout");

/* Mirrors CRAWAwbPushBuf in lib/shaders/awb_hist.comp.glsl. */
typedef struct CRAWAwbPush {
    int32_t blocks[2];
    float   wb[3];           /* as-shot gains, divided back out */
    int32_t bins;
    float   log_min, log_max;
} CRAWAwbPush;

static_assert(sizeof(CRAWAwbPush) == 32, "CRAWAwbPush layout");
static_assert(offsetof(CRAWAwbPush, wb) == 8, "CRAWAwbPush layout");
static_assert(offsetof(CRAWAwbPush, bins) == 20, "CRAWAwbPush layout");

/* The shader's clamp flag is the public frame flag. */
static_assert(CRAW_FRAME_FLAG_CLAMP_NEGATIVE == 1, "colour shader flag mirror");

/* Both option structs end in a padding tail written as arithmetic on the
 * fields in front of it, so a field added without shrinking the tail
 * changes the size and breaks the ABI silently. These catch that. */
static_assert(sizeof(CRAWPipelineParams) == 256, "CRAWPipelineParams size");
static_assert(sizeof(CRAWFrameParams) == 264, "CRAWFrameParams size");

/* One sample per 8x8 block of the intermediate, DESIGN.md 4.4. */
#define CRAW_AWB_BLOCK 8

/* Denoise geometry, DESIGN.md 14.3. The 7x7 search and 3x3 patch are what
 * fits the frame budget on the reference device; a patch radius of 0 makes
 * the same shader the bilateral fallback. */
#define CRAW_DN_SEARCH_R 3
#define CRAW_DN_PATCH_R  1

/* The shader's shared tile is a fixed 28x28 floats, which is a 16x16
 * workgroup plus a halo of search + patch on each side. Both of these are
 * compile-time here, so the check is too. */
#define CRAW_DN_TILE_MAX 28
#define CRAW_DN_WG       16

static_assert(CRAW_DN_WG + 2 * (CRAW_DN_SEARCH_R + CRAW_DN_PATCH_R) <= CRAW_DN_TILE_MAX,
              "denoise tile does not fit the shader's shared array");
static_assert((2 * CRAW_DN_PATCH_R + 1) * (2 * CRAW_DN_PATCH_R + 1) <= 25,
              "denoise patch does not fit the shader's register array");

/* Where the highlight guard of DESIGN.md 14.3 ramps the filter off, in
 * units of the sensor range above black. */
#define CRAW_DN_GUARD_LO 0.90f
#define CRAW_DN_GUARD_HI 1.00f

/* The false colour shader's shared tile is the workgroup plus an apron of
 * two pixels per pass on every side, one for the median and one for the
 * blur, sized through specialization constant 4 from the workgroup the
 * device gets. Two copies of a vec2 per entry: for a 16x16 workgroup and
 * five passes that is 36x36 and 20 KB, more than the 16 KB Vulkan
 * guarantees, so a device with only the guarantee gets an 8x8 workgroup
 * and a 28x28 tile of 12.5 KB. DESIGN.md 17.3. */
#define CRAW_FC_WG_MAX 16

static_assert(CRAW_FALSE_COLOR_MAX_PASSES == 5, "false colour passes must match the shader's");

static uint32_t craw_fc_tile(uint32_t wg)
{
    return wg + 4 * CRAW_FALSE_COLOR_MAX_PASSES;
}

static uint32_t craw_fc_shared_bytes(uint32_t wg)
{
    return 2 * craw_fc_tile(wg) * craw_fc_tile(wg) * 2 * (uint32_t)sizeof(float);
}

/* The chroma denoise bilateral's radius on the quarter grid, DESIGN.md
 * 14.4. The quarter grid mirrors its own borders, one reflection, so the
 * grid must be at least radius + 1 wide, i.e. the frame at least this. */
#define CRAW_CD_RADIUS 4
#define CRAW_CD_MIN_SIZE (4 * (CRAW_CD_RADIUS + 1))

/* The highlight reconstruction's colour field, DESIGN.md 18.2: one texel
 * per 4x4 block of the mosaic at level 0, and levels halved (rounding up)
 * until one is 1x1. Sixteen levels reach a frame 131072 pixels on its
 * longer side. */
#define CRAW_HL_BLOCK      4
#define CRAW_HL_MAX_LEVELS 16

/* The capture sharpening's planes are fp16 in chunks of eight pixels,
 * DESIGN.md 20.3. */
#define CRAW_SH_CHUNK    8

/* The rows one subgroup streams down in an iteration of
 * sharpen_rows.comp.glsl: each strip spends 4R rows filling its rings. */
#define CRAW_SH_STRIP_ROWS 32

/* The shared memory an iteration of reach r takes on a square workgroup
 * of side wg, mirroring the tile sizes in lib/shaders/sharpen.comp.glsl. */
static uint32_t craw_sh_shared_bytes(uint32_t wg, uint32_t r)
{
    const uint32_t nb = 4;
    uint32_t rd = CRAW_MAX(r, (uint32_t)CRAW_SH_MASK_R + 1);
    uint32_t cu0 = (2 * r + 7) / 8, ncu = cu0 + (wg + 2 * r + 7) / 8;
    uint32_t cd0 = (rd + 7) / 8, ncd = cd0 + (wg + rd + 7) / 8;
    uint32_t uy = wg + 4 * r, bx = wg + 2 * r, dy = wg + 2 * rd;
    uint32_t hx = (bx + nb - 1) / nb * nb;
    uint32_t a = CRAW_MAX(uy * (ncu * 8 + nb), bx * bx);
    uint32_t h = CRAW_MAX((uy + nb) * hx, bx * wg);

    return (a + h + dy * ncd * 8) * (uint32_t)sizeof(float);
}

#define CRAW_AWB_HIST_ENTRIES (CRAW_AWB_HIST_BINS * CRAW_AWB_HIST_BINS)
#define CRAW_AWB_HIST_BYTES   (CRAW_AWB_HIST_ENTRIES * (uint32_t)sizeof(uint32_t))

struct CRAWPipeline {
    CRAWContext *ctx;
    CRAWPipelineParams params;

    bool rcd;
    bool awb;
    bool has_denoise;
    bool has_lens;
    bool has_fc;
    bool has_chroma;
    bool has_hl;
    bool has_df;
    bool has_temporal;
    bool has_sh;

    /* One of `debayer` or the three RCD passes is created, never both.
     * rcd_rb exists twice, once per phase of DESIGN.md 5.3 pass C. */
    CRAWVkShader debayer;
    CRAWVkShader rcd_dir;
    CRAWVkShader rcd_green;
    CRAWVkShader rcd_rb[2];
    CRAWVkShader color;
    CRAWVkShader awb_hist;
    CRAWVkShader denoise;
    CRAWVkShader lens;
    CRAWVkShader false_color;
    CRAWVkShader chroma_down;
    CRAWVkShader chroma_blur;
    CRAWVkShader chroma_up;
    CRAWVkShader hl_field;
    CRAWVkShader hl_pull;
    CRAWVkShader hl_push;
    CRAWVkShader hl_fill;
    CRAWVkShader df_prep;
    CRAWVkShader df;
    CRAWVkShader sh_prep;
    /* Sharpening: [1] is the last iteration, and one of each per reach. */
    CRAWVkShader sh_iter[2][CRAW_SH_RMAX];
    /* The iterations before the last streamed by subgroups, one per reach,
     * with the subgroup size they were built for; a reach whose strip
     * would not fit the subgroup has none and uses sh_iter[0]. */
    CRAWVkShader sh_rows[CRAW_SH_RMAX];
    uint32_t sh_lanes;

    CRAWVkImage inter;   /* camera RGB, RGBA16F */
    CRAWVkImage aux;     /* RCD: (L, wv, wp, unused), RGBA16F */
    CRAWVkImage green;   /* RCD: pass B green plane, R16F */
    CRAWVkImage clean;   /* denoise: the filtered mosaic, R16_UNORM */
    CRAWVkImage scratch; /* a second intermediate for the stages that cannot
                          * run in place (false colour, chroma denoise) or
                          * need one to prepare into (defringe), RGBA16F,
                          * only without RCD, whose aux is borrowed */
    CRAWVkImage chroma_q;  /* chroma denoise: quarter resolution (d_r, d_b, G) */
    CRAWVkImage chroma_qf; /* the same, differences filtered */
    CRAWVkImage hl_mosaic;  /* highlights: the rebuilt mosaic, R16F, balanced */
    CRAWVkImage hl_level[CRAW_HL_MAX_LEVELS]; /* highlights: the colour field
                                               * and its pyramid, RGBA32F */
    uint32_t nb_hl_levels;
    CRAWVkBuffer sh_d;      /* sharpening: the observed luma, fp16 */
    CRAWVkBuffer sh_u[2];   /* sharpening: the estimate, ping-pong, fp16 */
    int32_t sh_stride;      /* their rows, in chunks of eight pixels */

    /* Temporal denoise, DESIGN.md 14.7: two sets of state, each the
     * temporally filtered level of every mosaic sample of one frame
     * (R32F, full resolution) and the square root of the noise variance
     * left in it (R8_UNORM). `tn_cur` is the set the last recording wrote,
     * which the next frame reads, and `tn_prev` the one that recording
     * read, which a replay of it reads again; -1 is "nothing valid", which
     * is where both start and where a reset puts them. The two are never
     * the same set, so a recording always writes the set it does not read. */
    CRAWVkImage tn_state[2];
    CRAWVkImage tn_var[2];
    int tn_prev, tn_cur;

    CRAWVkBuffer hist_count;
    CRAWVkBuffer hist_weight;
    CRAWVkBuffer hist_bright;
    CRAWVkBuffer hist_readback[4];
    uint32_t     nb_slots;
    uint64_t     nb_recorded;

    int32_t blocks[2];   /* whole AWB blocks */
    int32_t qsize[2];    /* chroma denoise quarter grid, ceil(size / 4) */

    /* The last keep-everything scale and the record it was worked out for.
     * craw_lens_keep_all_scale() is a boundary walk of a few hundred
     * microseconds, and a clip that is not zooming hands over byte for byte
     * the same record on every frame; the compare is 64 bytes. A miss only
     * costs the walk, so a caller whose reserved bytes are not zeroed loses
     * nothing but the cache. */
    CRAWLensCorrection keep_lc;
    float keep_scale;
    bool have_keep_scale;

    /* Profiling, CRAW_PIPELINE_FLAG_PROFILE and DESIGN.md 21: a timestamp
     * pair per dispatch, in CRAW_PIPELINE_PROFILE_SLOTS runs of 2 * prof_max
     * queries, one run per recording in rotation. prof_sh holds which
     * shader each pair timed, prof_max entries per slot, and prof_nb how
     * many pairs each slot's recording wrote. prof_ts and prof_agg are
     * scratch for the readback. */
    VkQueryPool prof_pool;
    uint32_t prof_max;
    uint32_t prof_slot;
    uint32_t prof_nb[CRAW_PIPELINE_PROFILE_SLOTS];
    const CRAWVkShader **prof_sh;
    uint64_t prof_recorded;
    uint64_t prof_mask;
    double prof_period;       /* ns per tick */
    uint64_t *prof_ts;
    CRAWStageTime *prof_agg;
};

static const CRAWVkBinding craw_bind_images[4] = {
    { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE },
    { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE },
    { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE },
    { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE },
};

/* The raw denoise: mosaic, output, and the temporal state's two images
 * read and two written. */
static const CRAWVkBinding craw_bind_dn[6] = {
    { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE },
    { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE },
    { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE },
    { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE },
    { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE },
    { 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE },
};

/* The sharpening's iteration: the estimate, the observed luma, the next
 * estimate, and the camera RGB intermediate the last iteration writes.
 * Its first pass takes the prefix of craw_bind_awb: the intermediate, and
 * the luma it writes. */
static const CRAWVkBinding craw_bind_sharpen[4] = {
    { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },
    { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },
    { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },
    { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE },
};

static const CRAWVkBinding craw_bind_awb[4] = {
    { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE },
    { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },
    { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },
    { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },
};

/* Specialization constants below 253: 0 is the CFA phase (common.glsl), 1
 * the RCD pass C phase (rcd_rb.comp.glsl), 2 and 3 the denoise search and
 * patch radii (denoise_nlm.comp.glsl), 4 the false colour tile side
 * (false_color.comp.glsl), 5 and 6 the sharpening's last iteration and
 * its kernel's reach (sharpen.comp.glsl), 7 and 8 the subgroup size and
 * the strip height of its streamed iterations (sharpen_rows.comp.glsl). A
 * shader that does not declare one of these ignores its map entry, so
 * every shader is given the prefix of this array it needs. */
static const VkSpecializationMapEntry craw_spec_map[9] = {
    { .constantID = 0, .offset = 0,                    .size = sizeof(uint32_t) },
    { .constantID = 1, .offset = 1 * sizeof(uint32_t), .size = sizeof(uint32_t) },
    { .constantID = 2, .offset = 2 * sizeof(uint32_t), .size = sizeof(uint32_t) },
    { .constantID = 3, .offset = 3 * sizeof(uint32_t), .size = sizeof(uint32_t) },
    { .constantID = 4, .offset = 4 * sizeof(uint32_t), .size = sizeof(uint32_t) },
    { .constantID = 5, .offset = 5 * sizeof(uint32_t), .size = sizeof(uint32_t) },
    { .constantID = 6, .offset = 6 * sizeof(uint32_t), .size = sizeof(uint32_t) },
    { .constantID = 7, .offset = 7 * sizeof(uint32_t), .size = sizeof(uint32_t) },
    { .constantID = 8, .offset = 8 * sizeof(uint32_t), .size = sizeof(uint32_t) },
};

void craw_pipeline_params_init(CRAWPipelineParams *p)
{
    if (!p)
        return;

    *p = (CRAWPipelineParams) {
        .cfa        = CRAW_CFA_RGGB,
        .debayer    = CRAW_DEBAYER_RCD,
        .out_space  = CRAW_CS_XYZ_D65,
        .out_format = VK_FORMAT_R16G16B16A16_SFLOAT,
    };
    p->nb_awb_slots = 1;
}

void craw_frame_params_init(CRAWFrameParams *fp)
{
    if (!fp)
        return;

    *fp = (CRAWFrameParams) {
        .wb_mode = CRAW_WB_AS_SHOT,
    };
}

/* Largest square workgroup the device allows, capped at `want`. Vulkan
 * guarantees 128 invocations and 128 in x and y, so a device reporting less
 * than that never had its properties filled in and only the guarantee is
 * trusted. */
static uint32_t craw_wg_dim(const CRAWContext *ctx, uint32_t want)
{
    const VkPhysicalDeviceLimits *l = &ctx->props.properties.limits;

    if (l->maxComputeWorkGroupInvocations < 128)
        return CRAW_MIN(want, 8);

    while (want > 1 && (want * want > l->maxComputeWorkGroupInvocations ||
                        want > l->maxComputeWorkGroupSize[0] ||
                        want > l->maxComputeWorkGroupSize[1]))
        want /= 2;

    return want;
}

/* A shader with a square workgroup of side `wg`, or, with `sg` set, a
 * workgroup of exactly one subgroup of that size, `sg` x 1, which the
 * device is required to run in full subgroups of `sg`. */
static int craw_pipeline_shader_sg(CRAWPipeline *p, CRAWVkShader *sh, const char *name,
                                   const uint8_t *spirv, size_t spirv_len,
                                   const CRAWVkBinding *bindings, uint32_t nb_bindings,
                                   uint32_t push_size, uint32_t wg, uint32_t sg,
                                   const uint32_t *spec, uint32_t nb_spec)
{
    CRAWVkShaderDesc desc = {
        .name           = name,
        .spirv          = spirv,
        .spirv_len      = spirv_len,
        .bindings       = bindings,
        .nb_bindings    = nb_bindings,
        .push_size      = push_size,
        .wg             = { sg ? sg : wg, sg ? 1 : wg, 1 },
        .spec_entries   = nb_spec ? craw_spec_map : NULL,
        .nb_spec_entries = nb_spec,
        .spec_data      = nb_spec ? spec : NULL,
        .spec_size      = nb_spec * sizeof(uint32_t),
        .required_subgroup_size = sg,
    };

    return craw_vk_shader_create(p->ctx, sh, &desc);
}

static int craw_pipeline_shader(CRAWPipeline *p, CRAWVkShader *sh, const char *name,
                                const uint8_t *spirv, size_t spirv_len,
                                const CRAWVkBinding *bindings, uint32_t nb_bindings,
                                uint32_t push_size, uint32_t wg,
                                const uint32_t *spec, uint32_t nb_spec)
{
    return craw_pipeline_shader_sg(p, sh, name, spirv, spirv_len, bindings, nb_bindings,
                                   push_size, wg, 0, spec, nb_spec);
}

/* The most dispatches one recording of this pipeline can make: every pass
 * it was built with, the highlight pyramid at its depth and the most
 * sharpening iterations a frame may ask for. */
static uint32_t craw_pipeline_max_dispatches(const CRAWPipeline *p)
{
    uint32_t n = 1;                         /* colour */

    n += p->rcd ? 4 : 1;                    /* debayer */
    if (p->has_denoise || p->has_temporal)
        n += 1;
    if (p->has_hl)
        n += 2 + 2 * (p->nb_hl_levels - 1); /* field, pull, push, fill */
    if (p->has_fc)
        n += 1;
    if (p->has_lens)
        n += 1;
    if (p->has_df)
        n += 2;
    if (p->has_chroma)
        n += 3;
    if (p->has_sh)
        n += 1 + CRAW_SHARPEN_MAX_ITERATIONS;
    if (p->awb)
        n += 1;

    return n;
}

/* CRAW_PIPELINE_FLAG_PROFILE: the query pool and the host side of it. */
static int craw_pipeline_profile_init(CRAWPipeline *p)
{
    CRAWContext *ctx = p->ctx;
    VkQueueFamilyProperties *qf;
    uint32_t nb_qf = 0, bits;
    VkResult res;

    ctx->fn.GetPhysicalDeviceQueueFamilyProperties(ctx->phys_device, &nb_qf, NULL);
    if (ctx->queue_family_index >= nb_qf)
        return CRAW_ERROR(EINVAL);
    qf = calloc(nb_qf, sizeof(*qf));
    if (!qf)
        return CRAW_ERROR(ENOMEM);
    ctx->fn.GetPhysicalDeviceQueueFamilyProperties(ctx->phys_device, &nb_qf, qf);
    bits = qf[ctx->queue_family_index].timestampValidBits;
    free(qf);

    if (!bits) {
        craw_log(ctx, CRAW_LOG_ERROR, "Profiling needs timestamps, and queue family %u "
                 "has none\n", ctx->queue_family_index);
        return CRAW_ERROR(ENOTSUP);
    }

    p->prof_mask = bits >= 64 ? UINT64_MAX : (UINT64_C(1) << bits) - 1;
    p->prof_period = ctx->props.properties.limits.timestampPeriod;
    p->prof_max = craw_pipeline_max_dispatches(p);

    p->prof_sh = calloc((size_t)CRAW_PIPELINE_PROFILE_SLOTS * p->prof_max, sizeof(*p->prof_sh));
    p->prof_ts = calloc(2 * (size_t)p->prof_max, sizeof(*p->prof_ts));
    p->prof_agg = calloc(p->prof_max, sizeof(*p->prof_agg));
    if (!p->prof_sh || !p->prof_ts || !p->prof_agg)
        return CRAW_ERROR(ENOMEM);

    VkQueryPoolCreateInfo info = {
        .sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType  = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = CRAW_PIPELINE_PROFILE_SLOTS * 2 * p->prof_max,
    };
    res = ctx->fn.CreateQueryPool(ctx->device, &info, ctx->alloc, &p->prof_pool);
    if (res != VK_SUCCESS) {
        craw_log(ctx, CRAW_LOG_ERROR, "Unable to create the profiling query pool: %s\n",
                 craw_vk_ret2str(res));
        p->prof_pool = VK_NULL_HANDLE;
        return craw_vk_ret2err(res);
    }

    return 0;
}

int craw_pipeline_create(CRAWContext *ctx, const CRAWPipelineParams *params,
                         CRAWPipeline **pipe)
{
    CRAWPipeline *p;
    uint32_t spec[9];
    uint32_t wg2d, wgrcd, wgdn, wgfc, shared_max;
    int ret;

    if (!ctx || !params || !pipe)
        return CRAW_ERROR(EINVAL);

    if (params->cfa != CRAW_CFA_RGGB) {
        craw_log(ctx, CRAW_LOG_ERROR, "Only the RGGB CFA phase is implemented\n");
        return CRAW_ERROR(ENOTSUP);
    }

    /* Odd sizes would break the quad geometry every stage assumes, and the
     * mirrored borders of DESIGN.md 3 only support one reflection, i.e. a
     * frame at least as wide as the largest stencil radius (3, in RCD). */
    if (!params->width || !params->height ||
        (params->width & 1) || (params->height & 1) ||
        params->width < 8 || params->height < 8) {
        craw_log(ctx, CRAW_LOG_ERROR, "Invalid frame size %ux%u: both must be even and at least 8\n",
                 params->width, params->height);
        return CRAW_ERROR(EINVAL);
    }

    if (params->out_format != VK_FORMAT_R16G16B16A16_SFLOAT &&
        params->out_format != VK_FORMAT_R32G32B32A32_SFLOAT) {
        craw_log(ctx, CRAW_LOG_ERROR, "Output format must be RGBA16F or RGBA32F\n");
        return CRAW_ERROR(EINVAL);
    }

    if (params->debayer < 0 || params->debayer >= CRAW_DEBAYER_NB ||
        params->out_space < 0 || params->out_space >= CRAW_CS_NB)
        return CRAW_ERROR(EINVAL);

    p = calloc(1, sizeof(*p));
    if (!p)
        return CRAW_ERROR(ENOMEM);

    p->ctx = ctx;
    p->params = *params;
    p->rcd = params->debayer == CRAW_DEBAYER_RCD;
    p->awb = !!(params->flags & CRAW_PIPELINE_FLAG_AWB_HISTOGRAM);
    p->has_denoise = !!(params->flags & CRAW_PIPELINE_FLAG_DENOISE);
    p->has_lens = !!(params->flags & CRAW_PIPELINE_FLAG_LENS);
    p->has_fc = !!(params->flags & CRAW_PIPELINE_FLAG_FALSE_COLOR);
    p->has_chroma = !!(params->flags & CRAW_PIPELINE_FLAG_CHROMA_DENOISE);
    p->has_hl = !!(params->flags & CRAW_PIPELINE_FLAG_HIGHLIGHT_RECONSTRUCTION);
    p->has_df = !!(params->flags & CRAW_PIPELINE_FLAG_DEFRINGE);
    p->has_temporal = !!(params->flags & CRAW_PIPELINE_FLAG_TEMPORAL_DENOISE);
    p->tn_prev = p->tn_cur = -1;
    p->has_sh = !!(params->flags & CRAW_PIPELINE_FLAG_SHARPEN);
    p->qsize[0] = (int32_t)((params->width + 3) / 4);
    p->qsize[1] = (int32_t)((params->height + 3) / 4);
    p->blocks[0] = (int32_t)(params->width / CRAW_AWB_BLOCK);
    p->blocks[1] = (int32_t)(params->height / CRAW_AWB_BLOCK);

    /* 16x16 for the pointwise and small-stencil stages; RCD pass A reads a
     * 7x7 patch per invocation, so it gets half the occupancy pressure. */
    wg2d  = craw_wg_dim(ctx, 16);
    wgrcd = craw_wg_dim(ctx, 8);
    /* The denoise tile is sized for CRAW_DN_WG; a device that cannot run
     * that many invocations gets a smaller one, which only shrinks the
     * tile. */
    wgdn  = craw_wg_dim(ctx, CRAW_DN_WG);
    /* The false colour tile has to fit the device's shared memory, and a
     * smaller workgroup only shrinks it. A limit below the 16 KB Vulkan
     * guarantees means the properties were never filled in, and only the
     * guarantee is trusted. */
    shared_max = CRAW_MAX(ctx->props.properties.limits.maxComputeSharedMemorySize, 16384u);
    wgfc  = craw_wg_dim(ctx, CRAW_FC_WG_MAX);
    while (wgfc > 4 && craw_fc_shared_bytes(wgfc) > shared_max)
        wgfc /= 2;

    ret = craw_vk_image_create(ctx, &p->inter, params->width, params->height,
                               VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT);
    if (ret < 0)
        goto fail;

    spec[0] = (uint32_t)params->cfa;
    spec[1] = 0;
    spec[2] = CRAW_DN_SEARCH_R;
    spec[3] = CRAW_DN_PATCH_R;
    spec[4] = craw_fc_tile(wgfc);
    spec[5] = 0;
    spec[6] = 1;
    spec[7] = 0;
    spec[8] = 0;

    /* The temporal pass is part of the raw stage's dispatch (DESIGN.md
     * 14.7), so either flag builds the stage. */
    if (p->has_denoise || p->has_temporal) {
        /* The denoise stage mirrors inside each CFA plane, and one
         * reflection only reaches a stencil radius of plane size - 1, so a
         * frame smaller than this would read outside the plane. Every real
         * frame is orders of magnitude bigger; this exists so that the
         * rule of DESIGN.md 3 is enforced rather than assumed. The temporal
         * pass's 3x3 comparison reaches one sample further. */
        const uint32_t dn_min = 2 * (CRAW_DN_SEARCH_R + CRAW_DN_PATCH_R + 1 +
                                     (p->has_temporal ? 1 : 0));

        if (params->width < dn_min || params->height < dn_min) {
            craw_log(ctx, CRAW_LOG_ERROR,
                     "Denoising needs a frame of at least %ux%u, got %ux%u\n",
                     dn_min, dn_min, params->width, params->height);
            ret = CRAW_ERROR(EINVAL);
            goto fail;
        }

        /* A second mosaic rather than a filter in place: no pass in this
         * library reads an image it also writes, and the caller's source
         * image is its own. */
        ret = craw_vk_image_create(ctx, &p->clean, params->width, params->height,
                                   VK_FORMAT_R16_UNORM, VK_IMAGE_USAGE_STORAGE_BIT);
        if (ret < 0)
            goto fail;

        ret = craw_pipeline_shader(p, &p->denoise, "denoise_nlm",
                                   craw_denoise_nlm_spv_data, craw_denoise_nlm_spv_len,
                                   craw_bind_dn, 6, sizeof(CRAWDenoisePush), wgdn,
                                   spec, 4);
        if (ret < 0)
            goto fail;
    }

    if (p->has_temporal) {
        /* fp32 for the reason the stabilised domain needs it (DESIGN.md
         * 14.3), and two sets so that a frame can be developed again from
         * the state it was first developed from. The variance is a number
         * between 0 and 1 that only scales thresholds, and eight bits of
         * its square root are a step of 2.4% of it at an alpha_max of 0.8. */
        for (int i = 0; i < 2; i++) {
            ret = craw_vk_image_create(ctx, &p->tn_state[i], params->width, params->height,
                                       VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT);
            if (ret < 0)
                goto fail;
            ret = craw_vk_image_create(ctx, &p->tn_var[i], params->width, params->height,
                                       VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT);
            if (ret < 0)
                goto fail;
        }
    }

    if (p->has_hl) {
        uint32_t lw = (params->width + CRAW_HL_BLOCK - 1) / CRAW_HL_BLOCK;
        uint32_t lh = (params->height + CRAW_HL_BLOCK - 1) / CRAW_HL_BLOCK;

        /* The rebuilt mosaic holds balanced values above 1.0, which no
         * UNORM encoding of the raw range can. fp16 rounds by at most
         * 2^-11 relative, at least seven times below the sensor's own
         * noise at any level (it is 7.4 / sqrt(u) times smaller), and the intermediate after the debayer is fp16
         * anyway. DESIGN.md 18.3. */
        ret = craw_vk_image_create(ctx, &p->hl_mosaic, params->width, params->height,
                                   VK_FORMAT_R16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT);
        if (ret < 0)
            goto fail;

        /* The colour field and its pyramid, down to 1x1. */
        while (1) {
            if (p->nb_hl_levels == CRAW_HL_MAX_LEVELS) {
                craw_log(ctx, CRAW_LOG_ERROR, "Frame too large for the highlight field's pyramid\n");
                ret = CRAW_ERROR(EINVAL);
                goto fail;
            }
            ret = craw_vk_image_create(ctx, &p->hl_level[p->nb_hl_levels], lw, lh,
                                       VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT);
            if (ret < 0)
                goto fail;
            p->nb_hl_levels++;
            if (lw == 1 && lh == 1)
                break;
            lw = (lw + 1) / 2;
            lh = (lh + 1) / 2;
        }

        /* The field pass keeps the clip state of the quads its workgroup
         * covers in a shared tile sized for a square workgroup of at most
         * 16, which is what wg2d is. */
        ret = craw_pipeline_shader(p, &p->hl_field, "highlight_field",
                                   craw_highlight_field_spv_data, craw_highlight_field_spv_len,
                                   craw_bind_images, 2, sizeof(CRAWHighlightPush), wg2d, spec, 1);
        if (ret < 0)
            goto fail;
        ret = craw_pipeline_shader(p, &p->hl_pull, "highlight_pull",
                                   craw_highlight_pull_spv_data, craw_highlight_pull_spv_len,
                                   craw_bind_images, 2, sizeof(CRAWHighlightLevelPush), wg2d,
                                   spec, 1);
        if (ret < 0)
            goto fail;
        ret = craw_pipeline_shader(p, &p->hl_push, "highlight_push",
                                   craw_highlight_push_spv_data, craw_highlight_push_spv_len,
                                   craw_bind_images, 2, sizeof(CRAWHighlightLevelPush), wg2d,
                                   spec, 1);
        if (ret < 0)
            goto fail;
        ret = craw_pipeline_shader(p, &p->hl_fill, "highlight_fill",
                                   craw_highlight_fill_spv_data, craw_highlight_fill_spv_len,
                                   craw_bind_images, 3, sizeof(CRAWHighlightPush), wg2d, spec, 1);
        if (ret < 0)
            goto fail;
    }

    if (!p->rcd) {
        const bool mhc = params->debayer == CRAW_DEBAYER_MHC;
        ret = craw_pipeline_shader(p, &p->debayer, mhc ? "debayer_mhc" : "debayer_bilinear",
                                   mhc ? craw_debayer_mhc_spv_data : craw_debayer_bilinear_spv_data,
                                   mhc ? craw_debayer_mhc_spv_len : craw_debayer_bilinear_spv_len,
                                   craw_bind_images, 2, sizeof(CRAWPrepPush), wg2d, spec, 1);
        if (ret < 0)
            goto fail;
    } else {
        ret = craw_vk_image_create(ctx, &p->aux, params->width, params->height,
                                   VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT);
        if (ret < 0)
            goto fail;

        ret = craw_vk_image_create(ctx, &p->green, params->width, params->height,
                                   VK_FORMAT_R16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT);
        if (ret < 0)
            goto fail;

        ret = craw_pipeline_shader(p, &p->rcd_dir, "rcd_dir",
                                   craw_rcd_dir_spv_data, craw_rcd_dir_spv_len,
                                   craw_bind_images, 2, sizeof(CRAWPrepPush), wgrcd, spec, 1);
        if (ret < 0)
            goto fail;

        ret = craw_pipeline_shader(p, &p->rcd_green, "rcd_green",
                                   craw_rcd_green_spv_data, craw_rcd_green_spv_len,
                                   craw_bind_images, 3, sizeof(CRAWPrepPush), wg2d, spec, 1);
        if (ret < 0)
            goto fail;

        for (uint32_t i = 0; i < 2; i++) {
            spec[1] = i;
            ret = craw_pipeline_shader(p, &p->rcd_rb[i], i ? "rcd_rb_g" : "rcd_rb_rb",
                                       craw_rcd_rb_spv_data, craw_rcd_rb_spv_len,
                                       craw_bind_images, 4, sizeof(CRAWPrepPush), wg2d, spec, 2);
            if (ret < 0)
                goto fail;
        }
    }

    /* Neither the false colour stage, the lens warp nor the chroma denoise
     * can run in place, so each needs a second intermediate to write, and
     * the two ping-pong between the debayer's and this one; the defringe
     * keeps its first pass there. With RCD it is the aux image: pass A
     * writes it, pass C is its last reader, and nothing touches it again
     * until the next frame's pass A, so the stages borrow it and the
     * pipeline costs no more memory than without the flags. The other two
     * methods have no such image and get one. DESIGN.md 17.3. */
    if ((p->has_fc || p->has_lens || p->has_chroma || p->has_df) && !p->rcd) {
        ret = craw_vk_image_create(ctx, &p->scratch, params->width, params->height,
                                   VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT);
        if (ret < 0)
            goto fail;
    }

    if (p->has_chroma) {
        if (params->width < CRAW_CD_MIN_SIZE || params->height < CRAW_CD_MIN_SIZE) {
            craw_log(ctx, CRAW_LOG_ERROR,
                     "Chroma denoising needs a frame of at least %ux%u, got %ux%u\n",
                     (unsigned)CRAW_CD_MIN_SIZE, (unsigned)CRAW_CD_MIN_SIZE,
                     params->width, params->height);
            ret = CRAW_ERROR(EINVAL);
            goto fail;
        }

        ret = craw_vk_image_create(ctx, &p->chroma_q, (uint32_t)p->qsize[0], (uint32_t)p->qsize[1],
                                   VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT);
        if (ret < 0)
            goto fail;
        ret = craw_vk_image_create(ctx, &p->chroma_qf, (uint32_t)p->qsize[0], (uint32_t)p->qsize[1],
                                   VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT);
        if (ret < 0)
            goto fail;

        ret = craw_pipeline_shader(p, &p->chroma_down, "chroma_down",
                                   craw_chroma_down_spv_data, craw_chroma_down_spv_len,
                                   craw_bind_images, 2, sizeof(CRAWChromaPush), wg2d, NULL, 0);
        if (ret < 0)
            goto fail;
        ret = craw_pipeline_shader(p, &p->chroma_blur, "chroma_blur",
                                   craw_chroma_blur_spv_data, craw_chroma_blur_spv_len,
                                   craw_bind_images, 2, sizeof(CRAWChromaPush), wg2d, NULL, 0);
        if (ret < 0)
            goto fail;
        ret = craw_pipeline_shader(p, &p->chroma_up, "chroma_up",
                                   craw_chroma_up_spv_data, craw_chroma_up_spv_len,
                                   craw_bind_images, 4, sizeof(CRAWChromaPush), wg2d, NULL, 0);
        if (ret < 0)
            goto fail;
    }

    if (p->has_df) {
        ret = craw_pipeline_shader(p, &p->df_prep, "defringe_prep",
                                   craw_defringe_prep_spv_data, craw_defringe_prep_spv_len,
                                   craw_bind_images, 2, sizeof(CRAWDefringePush), wg2d, NULL, 0);
        if (ret < 0)
            goto fail;
        ret = craw_pipeline_shader(p, &p->df, "defringe",
                                   craw_defringe_spv_data, craw_defringe_spv_len,
                                   craw_bind_images, 2, sizeof(CRAWDefringePush), wg2d, NULL, 0);
        if (ret < 0)
            goto fail;
    }

    if (p->has_sh) {
        if (params->width < CRAW_SH_MIN_SIZE || params->height < CRAW_SH_MIN_SIZE) {
            craw_log(ctx, CRAW_LOG_ERROR,
                     "Sharpening needs a frame of at least %ux%u, got %ux%u\n",
                     (unsigned)CRAW_SH_MIN_SIZE, (unsigned)CRAW_SH_MIN_SIZE,
                     params->width, params->height);
            ret = CRAW_ERROR(EINVAL);
            goto fail;
        }

        /* The observed luma and two estimates to ping-pong between, fp16:
         * a rounding of at most 2^-11 relative, 7.4 / sqrt(u) times below
         * the sensor's own noise and so seven times below it even at the
         * clip (DESIGN.md 18.3 has the same argument). Storage buffers
         * rather than images, eight pixels to a 16-byte chunk, so that an
         * iteration loads its tiles a chunk at a time. */
        p->sh_stride = (int32_t)((params->width + CRAW_SH_CHUNK - 1) / CRAW_SH_CHUNK);
        {
            VkDeviceSize bytes = (VkDeviceSize)p->sh_stride * params->height * 16;

            ret = craw_vk_buffer_create(ctx, &p->sh_d, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                        CRAW_MEM_DEVICE);
            for (int i = 0; i < 2 && ret >= 0; i++)
                ret = craw_vk_buffer_create(ctx, &p->sh_u[i], bytes,
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, CRAW_MEM_DEVICE);
            if (ret < 0)
                goto fail;
        }

        ret = craw_pipeline_shader(p, &p->sh_prep, "sharpen_prep",
                                   craw_sharpen_prep_spv_data, craw_sharpen_prep_spv_len,
                                   craw_bind_awb, 2, sizeof(CRAWSharpenPush), wg2d, NULL, 0);
        if (ret < 0)
            goto fail;

        /* A pipeline per reach and per kind of iteration, twelve in all:
         * with the reach a constant, every tile size, loop and divisor in
         * the shader is one. The tile has to fit the device's shared
         * memory, which at the widest reach and 16x16 is a few hundred
         * bytes over the 16 KB Vulkan guarantees; a smaller workgroup only
         * shrinks it, and the chunked loads need at least 8. */
        for (uint32_t r = 1; r <= CRAW_SH_RMAX; r++) {
            uint32_t wg = craw_wg_dim(ctx, 16);

            while (wg > 8 && craw_sh_shared_bytes(wg, r) > shared_max)
                wg /= 2;
            if (wg < 8 || craw_sh_shared_bytes(wg, r) > shared_max) {
                craw_log(ctx, CRAW_LOG_ERROR, "The device cannot run the sharpening's tiles\n");
                ret = CRAW_ERROR(ENOTSUP);
                goto fail;
            }

            for (uint32_t i = 0; i < 2; i++) {
                spec[5] = i;
                spec[6] = r;
                ret = craw_pipeline_shader(p, &p->sh_iter[i][r - 1],
                                           i ? "sharpen_last" : "sharpen",
                                           craw_sharpen_spv_data, craw_sharpen_spv_len,
                                           craw_bind_sharpen, 4, sizeof(CRAWSharpenPush), wg,
                                           spec, 7);
                if (ret < 0)
                    goto fail;
            }
        }

        /* The iterations before the last, streamed down strips by
         * subgroups of 32 (DESIGN.md 20.3 and 21), which on RDNA is what
         * makes a shuffle one swizzle rather than three permutes. A device
         * that cannot run 32 gets the nearest size it can; a strip that
         * then has no room for output (four columns a lane, 2R of aprons
         * on each side) leaves that reach to the tiled shader. */
        p->sh_lanes = CRAW_MIN(CRAW_MAX(32u, ctx->caps.min_subgroup_size),
                               ctx->caps.max_subgroup_size);
        for (uint32_t r = 1; r <= CRAW_SH_RMAX; r++) {
            if (!p->sh_lanes || 4 * p->sh_lanes < 4 * r + 8)
                continue;
            spec[6] = r;
            spec[7] = p->sh_lanes;
            spec[8] = CRAW_SH_STRIP_ROWS;
            ret = craw_pipeline_shader_sg(p, &p->sh_rows[r - 1], "sharpen",
                                          craw_sharpen_rows_spv_data, craw_sharpen_rows_spv_len,
                                          craw_bind_sharpen, 4, sizeof(CRAWSharpenPush), 0,
                                          p->sh_lanes, spec, 9);
            if (ret < 0)
                goto fail;
        }
        spec[5] = 0;
        spec[6] = 1;
        spec[7] = 0;
        spec[8] = 0;
    }

    if (p->has_fc) {
        ret = craw_pipeline_shader(p, &p->false_color, "false_color",
                                   craw_false_color_spv_data, craw_false_color_spv_len,
                                   craw_bind_images, 2, sizeof(CRAWFalseColorPush), wgfc,
                                   spec, 5);
        if (ret < 0)
            goto fail;
    }

    ret = craw_pipeline_shader(p, &p->color, "color", craw_color_spv_data, craw_color_spv_len,
                               craw_bind_images, 2, sizeof(CRAWColorPush), wg2d, NULL, 0);
    if (ret < 0)
        goto fail;

    if (p->has_lens) {
        ret = craw_pipeline_shader(p, &p->lens, "lens", craw_lens_spv_data, craw_lens_spv_len,
                                   craw_bind_images, 2, sizeof(CRAWLensPush), wg2d, NULL, 0);
        if (ret < 0)
            goto fail;
    }

    if (p->awb) {
        ret = craw_pipeline_shader(p, &p->awb_hist, "awb_hist",
                                   craw_awb_hist_spv_data, craw_awb_hist_spv_len,
                                   craw_bind_awb, 4, sizeof(CRAWAwbPush), wg2d, NULL, 0);
        if (ret < 0)
            goto fail;

        ret = craw_vk_buffer_create(ctx, &p->hist_count, CRAW_AWB_HIST_BYTES,
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT, CRAW_MEM_DEVICE);
        if (ret < 0)
            goto fail;

        ret = craw_vk_buffer_create(ctx, &p->hist_weight, CRAW_AWB_HIST_BYTES,
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT, CRAW_MEM_DEVICE);
        if (ret < 0)
            goto fail;

        ret = craw_vk_buffer_create(ctx, &p->hist_bright, CRAW_AWB_HIST_BYTES,
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT, CRAW_MEM_DEVICE);
        if (ret < 0)
            goto fail;

        p->nb_slots = params->nb_awb_slots ? params->nb_awb_slots : 1;
        if (p->nb_slots > CRAW_ARRAY_ELEMS(p->hist_readback)) {
            craw_log(ctx, CRAW_LOG_ERROR, "At most %zu awb histogram slots are supported\n",
                     CRAW_ARRAY_ELEMS(p->hist_readback));
            ret = CRAW_ERROR(EINVAL);
            goto fail;
        }
        for (uint32_t i = 0; i < p->nb_slots; i++) {
            ret = craw_vk_buffer_create(ctx, &p->hist_readback[i], 3 * CRAW_AWB_HIST_BYTES,
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT, CRAW_MEM_HOST_READBACK);
            if (ret < 0)
                goto fail;
        }
    }

    if (params->flags & CRAW_PIPELINE_FLAG_PROFILE) {
        ret = craw_pipeline_profile_init(p);
        if (ret < 0)
            goto fail;
    }

    craw_log(ctx, CRAW_LOG_VERBOSE, "Pipeline: %ux%u, debayer %d, workgroup %ux%u%s%s%s%s%s%s%s%s%s%s\n",
             params->width, params->height, (int)params->debayer, wg2d, wg2d,
             p->awb ? ", awb histogram" : "",
             p->has_temporal ? ", temporal denoise" : "",
             p->has_denoise ? ", denoise" : "",
             p->has_hl ? ", highlight reconstruction" : "",
             p->has_fc ? (wgfc == CRAW_FC_WG_MAX ? ", false colour" : ", false colour (small tile)") : "",
             p->has_df ? ", defringe" : "",
             p->has_chroma ? ", chroma denoise" : "",
             p->has_lens ? ", lens" : "",
             p->has_sh ? ", sharpen" : "",
             p->prof_pool ? ", profiled" : "");

    *pipe = p;
    return 0;

fail:
    craw_pipeline_free(&p);
    return ret;
}

void craw_pipeline_free(CRAWPipeline **pipe)
{
    CRAWPipeline *p;
    CRAWContext *ctx;

    if (!pipe || !*pipe)
        return;

    p = *pipe;
    ctx = p->ctx;

    if (p->debayer.pipeline)
        craw_vk_shader_free(ctx, &p->debayer);
    if (p->rcd_dir.pipeline)
        craw_vk_shader_free(ctx, &p->rcd_dir);
    if (p->rcd_green.pipeline)
        craw_vk_shader_free(ctx, &p->rcd_green);
    for (int i = 0; i < 2; i++)
        if (p->rcd_rb[i].pipeline)
            craw_vk_shader_free(ctx, &p->rcd_rb[i]);
    if (p->color.pipeline)
        craw_vk_shader_free(ctx, &p->color);
    if (p->awb_hist.pipeline)
        craw_vk_shader_free(ctx, &p->awb_hist);
    if (p->denoise.pipeline)
        craw_vk_shader_free(ctx, &p->denoise);
    if (p->lens.pipeline)
        craw_vk_shader_free(ctx, &p->lens);
    if (p->false_color.pipeline)
        craw_vk_shader_free(ctx, &p->false_color);
    if (p->chroma_down.pipeline)
        craw_vk_shader_free(ctx, &p->chroma_down);
    if (p->chroma_blur.pipeline)
        craw_vk_shader_free(ctx, &p->chroma_blur);
    if (p->chroma_up.pipeline)
        craw_vk_shader_free(ctx, &p->chroma_up);
    if (p->hl_field.pipeline)
        craw_vk_shader_free(ctx, &p->hl_field);
    if (p->hl_pull.pipeline)
        craw_vk_shader_free(ctx, &p->hl_pull);
    if (p->hl_push.pipeline)
        craw_vk_shader_free(ctx, &p->hl_push);
    if (p->hl_fill.pipeline)
        craw_vk_shader_free(ctx, &p->hl_fill);
    if (p->df_prep.pipeline)
        craw_vk_shader_free(ctx, &p->df_prep);
    if (p->df.pipeline)
        craw_vk_shader_free(ctx, &p->df);
    if (p->sh_prep.pipeline)
        craw_vk_shader_free(ctx, &p->sh_prep);
    for (int i = 0; i < 2; i++)
        for (int r = 0; r < CRAW_SH_RMAX; r++)
            if (p->sh_iter[i][r].pipeline)
                craw_vk_shader_free(ctx, &p->sh_iter[i][r]);
    for (int r = 0; r < CRAW_SH_RMAX; r++)
        if (p->sh_rows[r].pipeline)
            craw_vk_shader_free(ctx, &p->sh_rows[r]);

    if (p->inter.image)
        craw_vk_image_free(ctx, &p->inter);
    if (p->aux.image)
        craw_vk_image_free(ctx, &p->aux);
    if (p->green.image)
        craw_vk_image_free(ctx, &p->green);
    if (p->clean.image)
        craw_vk_image_free(ctx, &p->clean);
    for (int i = 0; i < 2; i++) {
        if (p->tn_state[i].image)
            craw_vk_image_free(ctx, &p->tn_state[i]);
        if (p->tn_var[i].image)
            craw_vk_image_free(ctx, &p->tn_var[i]);
    }
    if (p->scratch.image)
        craw_vk_image_free(ctx, &p->scratch);
    if (p->chroma_q.image)
        craw_vk_image_free(ctx, &p->chroma_q);
    if (p->chroma_qf.image)
        craw_vk_image_free(ctx, &p->chroma_qf);
    if (p->hl_mosaic.image)
        craw_vk_image_free(ctx, &p->hl_mosaic);
    for (uint32_t i = 0; i < p->nb_hl_levels; i++)
        if (p->hl_level[i].image)
            craw_vk_image_free(ctx, &p->hl_level[i]);
    if (p->sh_d.buf)
        craw_vk_buffer_free(ctx, &p->sh_d);
    for (int i = 0; i < 2; i++)
        if (p->sh_u[i].buf)
            craw_vk_buffer_free(ctx, &p->sh_u[i]);

    if (p->hist_count.buf)
        craw_vk_buffer_free(ctx, &p->hist_count);
    if (p->hist_weight.buf)
        craw_vk_buffer_free(ctx, &p->hist_weight);
    if (p->hist_bright.buf)
        craw_vk_buffer_free(ctx, &p->hist_bright);
    for (uint32_t i = 0; i < CRAW_ARRAY_ELEMS(p->hist_readback); i++) {
        if (p->hist_readback[i].buf)
            craw_vk_buffer_free(ctx, &p->hist_readback[i]);
    }

    if (p->prof_pool)
        ctx->fn.DestroyQueryPool(ctx->device, p->prof_pool, ctx->alloc);
    free(p->prof_sh);
    free(p->prof_ts);
    free(p->prof_agg);

    free(p);
    *pipe = NULL;
}

/* out = a * b, both row major. */
static void craw_mat3_mul(float out[3][3], const float a[3][3], const float b[3][3])
{
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
}

static int craw_pipeline_prep_push(CRAWPipeline *p, const CRAWProfile *prof,
                                   const CRAWColorParams *par, CRAWPrepPush *pp)
{
    *pp = (CRAWPrepPush) {
        .size = { (int32_t)p->params.width, (int32_t)p->params.height },
        .clip = 1.0f,
    };

    for (int i = 0; i < 4; i++) {
        float range = prof->white_level[i] - prof->black_level[i];
        if (!(range > 0.0f)) {
            craw_log(p->ctx, CRAW_LOG_ERROR, "Profile site %d has white level %f <= black level %f\n",
                     i, prof->white_level[i], prof->black_level[i]);
            return CRAW_ERROR(EINVAL);
        }
        pp->black[i] = prof->black_level[i];
        pp->inv_range[i] = 1.0f / range;
    }

    for (int i = 0; i < 3; i++)
        pp->wb[i] = par->wb[i];

    return 0;
}

/* Resolves the frame's noise model, for the denoise and for the
 * sharpening's mask alike: the override when it carries a positive a, else
 * the profile's table. The model is looked up at the sensor's own exposure
 * index (as_shot_ei) and not at the working one: rating a clip at a
 * different index scales the picture afterwards, it does not change how
 * the sensels were read. */
static int craw_pipeline_noise_model(CRAWPipeline *p, const CRAWProfile *prof,
                                     const CRAWFrameParams *fp, const char *what,
                                     CRAWNoiseModel *nm)
{
    int ret;

    if (fp->noise_override.a > 0.0f) {
        *nm = fp->noise_override;
        return 0;
    }

    ret = craw_profile_noise_eval(prof, prof->as_shot_ei, nm);
    if (ret < 0)
        craw_log(p->ctx, CRAW_LOG_ERROR,
                 "%s needs a noise model: the profile `%s` has no table and "
                 "CRAWFrameParams.noise_override was not set\n", what,
                 prof->name[0] ? prof->name : "(unnamed)");
    return ret;
}

/* Turns the frame's noise model into the denoise shader's constants,
 * DESIGN.md 14.3. `mode` says which of the two filters run; the temporal
 * one's history bit is the caller's to add. */
static int craw_pipeline_denoise_push(CRAWPipeline *p, const CRAWProfile *prof,
                                      const CRAWFrameParams *fp, int32_t mode,
                                      CRAWDenoisePush *dp)
{
    CRAWNoiseModel nm;
    float a, b, c, sc, h;
    int ret;

    ret = craw_pipeline_noise_model(p, prof, fp, "Denoising", &nm);
    if (ret < 0)
        return ret;

    a = nm.a;
    b = CRAW_MAX(nm.b, 0.0f);
    if (!(a > 0.0f)) {
        craw_log(p->ctx, CRAW_LOG_ERROR, "Noise model a = %g is not positive\n", (double)a);
        return CRAW_ERROR(EINVAL);
    }

    /* The 3 a^2 / 8 of the Anscombe transform, plus the read term. */
    c = 3.0f * a * a / 8.0f + b;
    sc = sqrtf(c);

    /* The temporal threshold is the spatial one when the spatial filter
     * runs, so that one strength means one thing in both, and the modelled
     * noise level itself when it does not. DESIGN.md 14.7. */
    h = (mode & CRAW_DN_MODE_SPATIAL) ? fp->denoise_strength : 1.0f;

    *dp = (CRAWDenoisePush) {
        .size      = { (int32_t)(p->params.width / 2), (int32_t)(p->params.height / 2) },
        .gat_a     = a,
        .gat_c     = c,
        .gat_sc    = sc,
        .gat_fwd   = 2.0f / a,
        .gat_inv   = a / 4.0f,
        .gat_off   = c / a,
        .inv_h2    = 1.0f / (h * h),
        .guard_lo  = CRAW_DN_GUARD_LO,
        .guard_inv = 1.0f / (CRAW_DN_GUARD_HI - CRAW_DN_GUARD_LO),
        .t_alpha   = (mode & CRAW_DN_MODE_TEMPORAL) ? fp->denoise_temporal_strength : 0.0f,
        .t_inv_h2  = 1.0f / (h * h),
        .mode      = mode,
    };

    for (int i = 0; i < 4; i++) {
        float range = prof->white_level[i] - prof->black_level[i];

        dp->black[i] = prof->black_level[i];
        dp->inv_range[i] = 1.0f / range;
        dp->range[i] = range;
    }

    return 0;
}

/* The sharpening's push constants, DESIGN.md 20: the Gaussian's taps for
 * the frame's radius and, when the mask is on, the luma's modelled noise
 * from the same model the denoise uses, under the frame's own gains. */
static int craw_pipeline_sharpen_push(CRAWPipeline *p, const CRAWProfile *prof,
                                      const CRAWFrameParams *fp, const float wb[3],
                                      CRAWSharpenPush *sp, int *reach)
{
    CRAWNoiseModel nm;
    int ret;

    if (!(fp->sharpen_radius <= CRAW_SHARPEN_MAX_RADIUS)) {
        craw_log(p->ctx, CRAW_LOG_ERROR, "sharpen_radius = %g; at most %g\n",
                 (double)fp->sharpen_radius, (double)CRAW_SHARPEN_MAX_RADIUS);
        return CRAW_ERROR(EINVAL);
    }
    if (fp->sharpen_iterations > CRAW_SHARPEN_MAX_ITERATIONS) {
        craw_log(p->ctx, CRAW_LOG_ERROR, "sharpen_iterations = %u; at most %u\n",
                 fp->sharpen_iterations, (unsigned)CRAW_SHARPEN_MAX_ITERATIONS);
        return CRAW_ERROR(EINVAL);
    }
    if (!(fp->sharpen_contrast >= 0.0f) || !isfinite(fp->sharpen_contrast)) {
        craw_log(p->ctx, CRAW_LOG_ERROR, "sharpen_contrast = %g is not a threshold\n",
                 (double)fp->sharpen_contrast);
        return CRAW_ERROR(EINVAL);
    }

    *sp = (CRAWSharpenPush) {
        .size     = { (int32_t)p->params.width, (int32_t)p->params.height },
        .stride   = p->sh_stride,
        .contrast = fp->sharpen_contrast,
    };
    *reach = craw_sharpen_kernel(fp->sharpen_radius, sp->k);

    if (fp->sharpen_contrast > 0.0f) {
        ret = craw_pipeline_noise_model(p, prof, fp, "The sharpening's contrast mask", &nm);
        if (ret < 0)
            return ret;
        if (!(nm.a > 0.0f)) {
            craw_log(p->ctx, CRAW_LOG_ERROR, "Noise model a = %g is not positive\n",
                     (double)nm.a);
            return CRAW_ERROR(EINVAL);
        }
        craw_sharpen_noise(&nm, wb, &sp->noise_a, &sp->noise_b);
    }

    return 0;
}

/* Turns the frame's correction into the warp's push constants, DESIGN.md
 * 16. The record states the centre in normalized frame coordinates and
 * normalizes its radius by the corner distance, so both become pixels here
 * against the pipeline's own frame size: the pipeline develops the whole
 * coded frame, which is the frame the camera normalized against. */
static int craw_pipeline_lens_push(CRAWPipeline *p, const CRAWLensCorrection *lc,
                                   bool keep_all, CRAWLensPush *lp)
{
    float hx, hy, half_diag, scale;
    int ret;

    if (lc->centre[0] < 0.0f || lc->centre[0] > 1.0f ||
        lc->centre[1] < 0.0f || lc->centre[1] > 1.0f) {
        craw_log(p->ctx, CRAW_LOG_ERROR, "Lens centre (%g, %g) is outside the frame\n",
                 (double)lc->centre[0], (double)lc->centre[1]);
        return CRAW_ERROR(EINVAL);
    }

    *lp = (CRAWLensPush) {
        .size   = { (int32_t)p->params.width, (int32_t)p->params.height },
        .centre = { lc->centre[0] * (float)p->params.width,
                    lc->centre[1] * (float)p->params.height },
        /* Filling the frame is a scale of exactly one, and the shader's
         * multiply by it is exact, so the frame it produces is bit for bit
         * the one it produced before the mode existed. */
        .inv_scale = 1.0f,
    };

    /* DESIGN.md 16.8: the scale is a property of the record and the frame
     * size, so it is worked out here, once per recording, rather than by
     * seventeen million invocations that would each get the same answer.
     * It changes frame by frame on a clip that is zooming. */
    if (keep_all) {
        if (p->have_keep_scale && !memcmp(&p->keep_lc, lc, sizeof(*lc))) {
            scale = p->keep_scale;
        } else {
            ret = craw_lens_keep_all_scale(lc, p->params.width, p->params.height, &scale);
            if (ret < 0) {
                craw_log(p->ctx, CRAW_LOG_ERROR, "Lens polynomial cannot be inverted; "
                         "the keep-everything scale is undefined\n");
                return ret;
            }
            p->keep_lc = *lc;
            p->keep_scale = scale;
            p->have_keep_scale = true;
        }
        lp->inv_scale = 1.0f / scale;
        lp->keep_all = 1;
    }

    /* r = 1 is the frame corner as seen from the stated centre, i.e. the
     * farther of the two sides on each axis. */
    hx = CRAW_MAX(lc->centre[0], 1.0f - lc->centre[0]) * (float)p->params.width;
    hy = CRAW_MAX(lc->centre[1], 1.0f - lc->centre[1]) * (float)p->params.height;
    half_diag = sqrtf(hx * hx + hy * hy);
    if (!(half_diag > 0.0f))
        return CRAW_ERROR(EINVAL);

    lp->inv_half_diag = 1.0f / half_diag;
    memcpy(lp->k, lc->k, sizeof(lp->k));

    return 0;
}

/* `gz` is 1 for every pass; the denoise dispatches its four CFA planes as
 * four times the columns of workgroups instead.
 *
 * With profiling on, the dispatch sits between two timestamps written once
 * everything before each has left the compute stage: the first when the
 * previous pass is done, the second when this one is. Off, it is one
 * well predicted branch per dispatch. DESIGN.md 21. */
static void craw_pipeline_dispatch(CRAWPipeline *p, VkCommandBuffer cmd, const CRAWVkShader *sh,
                                   CRAWVkDescWrite *writes, uint32_t nb_writes,
                                   const void *push, uint32_t push_size,
                                   uint32_t gx, uint32_t gy, uint32_t gz)
{
    uint32_t q = UINT32_MAX;

    if (p->prof_pool && p->prof_nb[p->prof_slot] < p->prof_max) {
        uint32_t i = p->prof_nb[p->prof_slot]++;

        p->prof_sh[p->prof_slot * p->prof_max + i] = sh;
        q = 2 * (p->prof_slot * p->prof_max + i);
        p->ctx->fn.CmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                      p->prof_pool, q);
    }

    craw_vk_shader_bind(p->ctx, cmd, sh);
    craw_vk_shader_push_desc(p->ctx, cmd, sh, writes, nb_writes);
    craw_vk_shader_push_const(p->ctx, cmd, sh, push, push_size);
    craw_vk_shader_dispatch(p->ctx, cmd, sh, gx, gy, gz);

    if (q != UINT32_MAX)
        p->ctx->fn.CmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                      p->prof_pool, q + 1);
}

/* The highlight reconstruction of DESIGN.md 18: the colour field's level 0
 * from `mosaic`, the pull up the pyramid, the push back down it, and the
 * fill into the rebuilt mosaic. `pp` is the prep the debayer would have
 * applied to `mosaic`; the stage applies the same black, range and gains,
 * without the clip. */
static void craw_pipeline_record_highlights(CRAWPipeline *p, VkCommandBuffer cmd,
                                            CRAWVkImage *mosaic, const CRAWPrepPush *pp)
{
    CRAWContext *ctx = p->ctx;
    CRAWVkDescWrite writes[3];
    CRAWHighlightPush hp;
    CRAWHighlightLevelPush lp;
    const uint32_t n = p->nb_hl_levels;

    hp = (CRAWHighlightPush) {
        .size  = { pp->size[0], pp->size[1] },
        .fsize = { (int32_t)p->hl_level[0].width, (int32_t)p->hl_level[0].height },
    };
    memcpy(hp.black, pp->black, sizeof(hp.black));
    memcpy(hp.inv_range, pp->inv_range, sizeof(hp.inv_range));
    memcpy(hp.wb, pp->wb, sizeof(hp.wb));

    craw_vk_image_barrier(ctx, cmd, &p->hl_level[0], VK_IMAGE_LAYOUT_GENERAL,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    craw_vk_desc_image(&writes[0], 0, mosaic);
    craw_vk_desc_image(&writes[1], 1, &p->hl_level[0]);
    craw_pipeline_dispatch(p, cmd, &p->hl_field, writes, 2, &hp, sizeof(hp),
                           p->hl_level[0].width, p->hl_level[0].height, 1);

    /* Pull: every level the sums of the one below it. */
    for (uint32_t l = 0; l + 1 < n; l++) {
        CRAWVkImage *fine = &p->hl_level[l], *coarse = &p->hl_level[l + 1];

        craw_vk_image_barrier(ctx, cmd, fine, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        craw_vk_image_barrier(ctx, cmd, coarse, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        lp = (CRAWHighlightLevelPush) {
            .fine   = { (int32_t)fine->width, (int32_t)fine->height },
            .coarse = { (int32_t)coarse->width, (int32_t)coarse->height },
        };
        craw_vk_desc_image(&writes[0], 0, fine);
        craw_vk_desc_image(&writes[1], 1, coarse);
        craw_pipeline_dispatch(p, cmd, &p->hl_pull, writes, 2, &lp, sizeof(lp),
                               coarse->width, coarse->height, 1);
    }

    /* Push: every level, from the one under the top down to level 0, takes
     * what it lacks from the level above, which is already a colour except
     * at the top. Each level is rewritten in place by invocations that each
     * touch only their own texel. */
    for (uint32_t l = n - 1; l-- > 0;) {
        CRAWVkImage *fine = &p->hl_level[l], *coarse = &p->hl_level[l + 1];

        craw_vk_image_barrier(ctx, cmd, coarse, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        craw_vk_image_barrier(ctx, cmd, fine, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                              VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        lp = (CRAWHighlightLevelPush) {
            .fine   = { (int32_t)fine->width, (int32_t)fine->height },
            .coarse = { (int32_t)coarse->width, (int32_t)coarse->height },
            .top    = l + 2 == n,
        };
        craw_vk_desc_image(&writes[0], 0, coarse);
        craw_vk_desc_image(&writes[1], 1, fine);
        craw_pipeline_dispatch(p, cmd, &p->hl_push, writes, 2, &lp, sizeof(lp),
                               fine->width, fine->height, 1);
    }

    craw_vk_image_barrier(ctx, cmd, &p->hl_level[0], VK_IMAGE_LAYOUT_GENERAL,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    craw_vk_image_barrier(ctx, cmd, &p->hl_mosaic, VK_IMAGE_LAYOUT_GENERAL,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    craw_vk_desc_image(&writes[0], 0, mosaic);
    craw_vk_desc_image(&writes[1], 1, &p->hl_level[0]);
    craw_vk_desc_image(&writes[2], 2, &p->hl_mosaic);
    /* One invocation per quad. */
    craw_pipeline_dispatch(p, cmd, &p->hl_fill, writes, 3, &hp, sizeof(hp),
                           p->params.width / 2, p->params.height / 2, 1);

    craw_vk_image_barrier(ctx, cmd, &p->hl_mosaic, VK_IMAGE_LAYOUT_GENERAL,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
}

int craw_pipeline_record(CRAWPipeline *p, VkCommandBuffer cmd,
                         const CRAWImage *src, const CRAWImage *dst,
                         const CRAWFrameParams *fp)
{
    CRAWContext *ctx;
    CRAWVkImage src_img, dst_img;
    CRAWVkImage *mosaic, *inter, *other;
    CRAWVkDescWrite writes[4];
    CRAWColorParams frame;
    CRAWPrepPush pp;
    CRAWColorPush cp;
    CRAWAwbPush ap;
    CRAWDenoisePush dp;
    CRAWLensPush lp;
    CRAWFalseColorPush fcp;
    CRAWChromaPush cdp;
    CRAWDefringePush dfp;
    CRAWSharpenPush shp;
    int sh_reach = 1;
    enum CRAWColorSpace out_space;
    bool denoise, lens, false_color, chroma, highlights, defringe, sharpen;
    bool spatial, temporal, replay;
    int t_read, t_write;
    int32_t dn_mode;
    float m_out[3][3], scale;
    uint32_t w, h;
    int ret;

    if (!p || !cmd || !src || !dst || !fp || !fp->profile)
        return CRAW_ERROR(EINVAL);

    ctx = p->ctx;
    w = p->params.width;
    h = p->params.height;

    if (src->format != VK_FORMAT_R16_UNORM) {
        craw_log(ctx, CRAW_LOG_ERROR, "Source must be VK_FORMAT_R16_UNORM (got %d)\n",
                 (int)src->format);
        return CRAW_ERROR(EINVAL);
    }
    if (dst->format != p->params.out_format) {
        craw_log(ctx, CRAW_LOG_ERROR, "Destination format %d does not match the pipeline's %d\n",
                 (int)dst->format, (int)p->params.out_format);
        return CRAW_ERROR(EINVAL);
    }
    if (src->width != w || src->height != h || dst->width != w || dst->height != h) {
        craw_log(ctx, CRAW_LOG_ERROR, "Frame size mismatch: src %ux%u dst %ux%u, pipeline %ux%u\n",
                 src->width, src->height, dst->width, dst->height, w, h);
        return CRAW_ERROR(EINVAL);
    }
    if (!src->image || !src->view || !dst->image || !dst->view)
        return CRAW_ERROR(EINVAL);
    if (src->image == dst->image) {
        craw_log(ctx, CRAW_LOG_ERROR, "Source and destination must not alias\n");
        return CRAW_ERROR(EINVAL);
    }

    /* The prep of DESIGN.md 4.1 applies the frame's own gains, so that the
     * clamp at the balanced clip point keeps blown highlights neutral under
     * whatever illuminant was chosen; re-balancing after the clamp would
     * turn every clipped lamp the colour of the gain ratio. */
    ret = craw_profile_eval(fp->profile, fp->wb_mode, &fp->illuminant, &frame);
    if (ret < 0)
        return ret;

    ret = craw_pipeline_prep_push(p, fp->profile, &frame, &pp);
    if (ret < 0)
        return ret;

    /* A strength of zero is the documented way to turn either filter off
     * for one frame, so it is not an error on a pipeline that was never
     * built with it either. More than all of last frame's value is. */
    spatial = p->has_denoise && fp->denoise_strength > 0.0f;
    temporal = p->has_temporal && fp->denoise_temporal_strength > 0.0f;
    if (temporal && !(fp->denoise_temporal_strength <= 1.0f)) {
        craw_log(ctx, CRAW_LOG_ERROR, "denoise_temporal_strength = %g; at most 1.0\n",
                 (double)fp->denoise_temporal_strength);
        return CRAW_ERROR(EINVAL);
    }

    /* Which state this recording reads and which it writes, DESIGN.md
     * 14.7: the next frame reads what the last recording wrote, a replay
     * what the last recording read, and a reset nothing. The set written
     * is the other one, or for a replay the one the last recording wrote,
     * which is then never the one being read. Worked out here and only
     * committed once nothing can fail. */
    replay = fp->denoise_temporal_replay;
    t_read = replay ? p->tn_prev : p->tn_cur;
    if (fp->flags & CRAW_FRAME_FLAG_TEMPORAL_RESET)
        t_read = -1;
    t_write = -1;
    if (temporal)
        t_write = replay && p->tn_cur >= 0 ? p->tn_cur : (t_read == 0 ? 1 : 0);

    denoise = spatial || temporal;
    if (denoise) {
        dn_mode = (spatial ? CRAW_DN_MODE_SPATIAL : 0) |
                  (temporal ? CRAW_DN_MODE_TEMPORAL : 0) |
                  (temporal && t_read >= 0 ? CRAW_DN_MODE_HISTORY : 0);
        ret = craw_pipeline_denoise_push(p, fp->profile, fp, dn_mode, &dp);
        if (ret < 0)
            return ret;
    }

    /* Zero passes is the documented way to leave one frame alone, on a
     * pipeline built with the stage or without it. More passes than the
     * shader's tile has apron for is an error rather than a clamp: a
     * caller who asked for five would not get what it asked for. */
    false_color = p->has_fc && fp->false_color_passes > 0;
    if (false_color) {
        if (fp->false_color_passes > CRAW_FALSE_COLOR_MAX_PASSES) {
            craw_log(ctx, CRAW_LOG_ERROR,
                     "false_color_passes = %u; at most %u passes are supported\n",
                     fp->false_color_passes, (unsigned)CRAW_FALSE_COLOR_MAX_PASSES);
            return CRAW_ERROR(EINVAL);
        }
        fcp = (CRAWFalseColorPush) {
            .size   = { (int32_t)w, (int32_t)h },
            .passes = (int32_t)fp->false_color_passes,
        };
    }

    /* Zero is the documented way to leave one frame alone; a colour cannot
     * be pulled back further than all the way. */
    defringe = p->has_df && fp->defringe_strength > 0.0f;
    if (defringe) {
        if (!(fp->defringe_strength <= 1.0f)) {
            craw_log(ctx, CRAW_LOG_ERROR, "defringe_strength = %g; at most 1.0\n",
                     (double)fp->defringe_strength);
            return CRAW_ERROR(EINVAL);
        }
        dfp = (CRAWDefringePush) {
            .size     = { (int32_t)w, (int32_t)h },
            .strength = fp->defringe_strength,
        };
    }

    /* Zero is the documented way to leave one frame alone; the band is
     * replaced by at most all of itself. */
    chroma = p->has_chroma && fp->denoise_chroma_strength > 0.0f;
    if (chroma) {
        if (!(fp->denoise_chroma_strength <= 1.0f)) {
            craw_log(ctx, CRAW_LOG_ERROR, "denoise_chroma_strength = %g; at most 1.0\n",
                     (double)fp->denoise_chroma_strength);
            return CRAW_ERROR(EINVAL);
        }
        cdp = (CRAWChromaPush) {
            .size     = { (int32_t)w, (int32_t)h },
            .qsize    = { p->qsize[0], p->qsize[1] },
            .strength = fp->denoise_chroma_strength,
        };
    }

    /* No iterations or no radius is the documented way to leave one frame
     * alone. The mask's noise is the luma's under the frame's own gains,
     * which is what the intermediate holds. */
    sharpen = p->has_sh && fp->sharpen_iterations > 0 && fp->sharpen_radius > 0.0f;
    if (sharpen) {
        ret = craw_pipeline_sharpen_push(p, fp->profile, fp, frame.wb, &shp, &sh_reach);
        if (ret < 0)
            return ret;
    }

    /* A correction marked invalid is the documented way to leave one frame
     * alone, so the whole warp drops out and the result is bit for bit the
     * uncorrected one. */
    lens = p->has_lens && fp->lens.valid;
    if (lens) {
        ret = craw_pipeline_lens_push(p, &fp->lens,
                                      !!(fp->flags & CRAW_FRAME_FLAG_LENS_KEEP_ALL), &lp);
        if (ret < 0)
            return ret;
    }

    /* The output space is per frame when the frame asks for one: nothing
     * the pipeline owns depends on it, only this 3x3 and the compression
     * below, so alternating spaces costs a different push constant and
     * nothing else. DESIGN.md 4.3. */
    out_space = fp->out_space_set ? fp->out_space : p->params.out_space;

    ret = craw_colorspace_from_xyz_d65(out_space, m_out);
    if (ret < 0) {
        craw_log(ctx, CRAW_LOG_ERROR, "Unknown output colour space %d\n", (int)out_space);
        return ret;
    }

    cp = (CRAWColorPush) {
        .size  = { (int32_t)w, (int32_t)h },
        .flags = fp->flags & CRAW_FRAME_FLAG_CLAMP_NEGATIVE,
    };

    /* DESIGN.md 8.1: ACEScg is the working space, so what leaves it has to
     * fit inside AP1; ACES2065-1 is the archive and is never touched. */
    craw_gamut_compress_params(&cp.gc);
    if (out_space == CRAW_CS_ACES_AP1_LINEAR &&
        !(fp->flags & CRAW_FRAME_FLAG_NO_GAMUT_COMPRESSION))
        cp.flags |= CRAW_COLOR_PUSH_GAMUT_COMPRESS;

    craw_mat3_mul(cp.matrix, m_out, frame.cam_to_xyz);
    scale = frame.gain * exp2f(fp->exposure_stops);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            cp.matrix[i][j] *= scale;
        /* The gains are already in the intermediate; kept as a field so
         * the shader layout stays put. */
        cp.wb_ratio[i] = 1.0f;
    }

    craw_vk_image_wrap(&src_img, src);
    craw_vk_image_wrap(&dst_img, dst);

    /* Nothing can fail from here on, so the slot this recording times is
     * taken now: its queries are reset in this command buffer, ahead of
     * the timestamps that rewrite them, which query commands on one queue
     * are ordered by without a barrier. */
    if (p->prof_pool) {
        p->prof_slot = (uint32_t)(p->prof_recorded % CRAW_PIPELINE_PROFILE_SLOTS);
        p->prof_nb[p->prof_slot] = 0;
        ctx->fn.CmdResetQueryPool(cmd, p->prof_pool, 2 * p->prof_slot * p->prof_max,
                                  2 * p->prof_max);
    }

    craw_vk_image_barrier(ctx, cmd, &src_img, VK_IMAGE_LAYOUT_GENERAL,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

    if (p->awb) {
        ctx->fn.CmdFillBuffer(cmd, p->hist_count.buf, 0, VK_WHOLE_SIZE, 0);
        ctx->fn.CmdFillBuffer(cmd, p->hist_weight.buf, 0, VK_WHOLE_SIZE, 0);
        ctx->fn.CmdFillBuffer(cmd, p->hist_bright.buf, 0, VK_WHOLE_SIZE, 0);
        /* vkCmdFillBuffer is a clear, not a copy: with synchronization2 it
         * executes in CLEAR, which COPY does not cover. */
        craw_vk_buffer_barrier(ctx, cmd, &p->hist_count,
                               VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        craw_vk_buffer_barrier(ctx, cmd, &p->hist_weight,
                               VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        craw_vk_buffer_barrier(ctx, cmd, &p->hist_bright,
                               VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }

    /* The denoise pass writes a second mosaic in the same raw encoding, so
     * everything downstream, the prep of 4.1 included, is untouched. */
    mosaic = &src_img;
    if (denoise) {
        /* The state images are only ever touched here, by consecutive
         * recordings in submission order, so these barriers are what order
         * this frame's read after the last one's write and this frame's
         * write after the last one's read, across submissions in flight
         * too. Bindings the mode does not use get stand-ins the shader
         * never touches: the source for the reads, the output for the
         * writes. */
        CRAWVkImage *t_prev = &src_img, *t_next = &p->clean;
        CRAWVkImage *v_prev = &src_img, *v_next = &p->clean;
        CRAWVkDescWrite dn_writes[6];

        craw_vk_image_barrier(ctx, cmd, &p->clean, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        if (temporal) {
            if (t_read >= 0) {
                t_prev = &p->tn_state[t_read];
                v_prev = &p->tn_var[t_read];
                craw_vk_image_barrier(ctx, cmd, t_prev, VK_IMAGE_LAYOUT_GENERAL,
                                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                      VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
                craw_vk_image_barrier(ctx, cmd, v_prev, VK_IMAGE_LAYOUT_GENERAL,
                                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                      VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
            }
            t_next = &p->tn_state[t_write];
            v_next = &p->tn_var[t_write];
            craw_vk_image_barrier(ctx, cmd, t_next, VK_IMAGE_LAYOUT_GENERAL,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            craw_vk_image_barrier(ctx, cmd, v_next, VK_IMAGE_LAYOUT_GENERAL,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }

        craw_vk_desc_image(&dn_writes[0], 0, &src_img);
        craw_vk_desc_image(&dn_writes[1], 1, &p->clean);
        craw_vk_desc_image(&dn_writes[2], 2, t_prev);
        craw_vk_desc_image(&dn_writes[3], 3, t_next);
        craw_vk_desc_image(&dn_writes[4], 4, v_prev);
        craw_vk_desc_image(&dn_writes[5], 5, v_next);
        /* Half resolution, four workgroups per tile, one per CFA plane,
         * side by side in x so that they run together. */
        {
            uint32_t wgx = p->denoise.wg[0];

            craw_pipeline_dispatch(p, cmd, &p->denoise, dn_writes, 6, &dp, sizeof(dp),
                                   4 * wgx * ((w / 2 + wgx - 1) / wgx), h / 2, 1);
        }

        craw_vk_image_barrier(ctx, cmd, &p->clean, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        /* Left as read, so that the next write to them, which may be a
         * replay's to these same images, gets a barrier of its own. */
        if (temporal) {
            craw_vk_image_barrier(ctx, cmd, t_next, VK_IMAGE_LAYOUT_GENERAL,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
            craw_vk_image_barrier(ctx, cmd, v_next, VK_IMAGE_LAYOUT_GENERAL,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        }
        mosaic = &p->clean;
    }

    /* The rebuilt mosaic replaces whichever the debayer would have read,
     * and already holds what the prep would have made of it, gains
     * included; the debayer's prep becomes the identity with no clip, so
     * the rebuilt highlights reach the colour stage above 1.0. */
    highlights = p->has_hl && (fp->flags & CRAW_FRAME_FLAG_HIGHLIGHT_RECONSTRUCTION);
    if (highlights) {
        craw_pipeline_record_highlights(p, cmd, mosaic, &pp);
        mosaic = &p->hl_mosaic;
        for (int i = 0; i < 4; i++) {
            pp.black[i] = 0.0f;
            pp.inv_range[i] = 1.0f;
            pp.wb[i] = 1.0f;
        }
        pp.clip = FLT_MAX;
    }

    craw_vk_image_barrier(ctx, cmd, &p->inter, VK_IMAGE_LAYOUT_GENERAL,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    if (!p->rcd) {
        craw_vk_desc_image(&writes[0], 0, mosaic);
        craw_vk_desc_image(&writes[1], 1, &p->inter);
        craw_pipeline_dispatch(p, cmd, &p->debayer, writes, 2, &pp, sizeof(pp), w, h, 1);
    } else {
        /* Pass A: low-pass and directional weights into the aux image. */
        craw_vk_image_barrier(ctx, cmd, &p->aux, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        craw_vk_desc_image(&writes[0], 0, mosaic);
        craw_vk_desc_image(&writes[1], 1, &p->aux);
        craw_pipeline_dispatch(p, cmd, &p->rcd_dir, writes, 2, &pp, sizeof(pp), w, h, 1);

        /* Pass B: the green plane. */
        craw_vk_image_barrier(ctx, cmd, &p->aux, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        craw_vk_image_barrier(ctx, cmd, &p->green, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        craw_vk_desc_image(&writes[2], 2, &p->green);
        craw_pipeline_dispatch(p, cmd, &p->rcd_green, writes, 3, &pp, sizeof(pp), w, h, 1);

        /* Pass C phase 0: the R and B sites, which phase 1 then reads. */
        craw_vk_image_barrier(ctx, cmd, &p->green, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        craw_vk_desc_image(&writes[3], 3, &p->inter);
        craw_pipeline_dispatch(p, cmd, &p->rcd_rb[0], writes, 4, &pp, sizeof(pp), w, h, 1);

        /* Pass C phase 1: the G sites. */
        craw_vk_image_barrier(ctx, cmd, &p->inter, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                              VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        craw_pipeline_dispatch(p, cmd, &p->rcd_rb[1], writes, 4, &pp, sizeof(pp), w, h, 1);
    }

    craw_vk_image_barrier(ctx, cmd, &p->inter, VK_IMAGE_LAYOUT_GENERAL,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

    /* The intermediate everything downstream reads: the debayer's own, or
     * what the false colour stage, the lens warp and the chroma denoise
     * made of it. Each of those writes the other full resolution image and
     * the two swap, so `inter` is always the latest and `other` free to
     * write. The second image is RCD's aux when there is one (its last
     * reader was pass C, just above) and the pipeline's own scratch
     * otherwise. DESIGN.md 17.3. */
    inter = &p->inter;
    other = p->rcd ? &p->aux : (p->scratch.image ? &p->scratch : NULL);

    if (false_color) {
        craw_vk_image_barrier(ctx, cmd, other, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        craw_vk_desc_image(&writes[0], 0, inter);
        craw_vk_desc_image(&writes[1], 1, other);
        craw_pipeline_dispatch(p, cmd, &p->false_color, writes, 2, &fcp, sizeof(fcp), w, h, 1);

        craw_vk_image_barrier(ctx, cmd, other, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        CRAWVkImage *t = inter;
        inter = other;
        other = t;
    }

    if (lens) {
        /* On the camera RGB, whose channels the record's polynomials are
         * for, and before the defringe, which must see them registered.
         * DESIGN.md 16.3. */
        craw_vk_image_barrier(ctx, cmd, other, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        craw_vk_desc_image(&writes[0], 0, inter);
        craw_vk_desc_image(&writes[1], 1, other);
        craw_pipeline_dispatch(p, cmd, &p->lens, writes, 2, &lp, sizeof(lp), w, h, 1);

        craw_vk_image_barrier(ctx, cmd, other, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        CRAWVkImage *t = inter;
        inter = other;
        other = t;
    }

    if (defringe) {
        /* DESIGN.md 19: l, c and E into the second image, then the clamp in
         * place, each invocation touching only its own texel of the
         * intermediate, so the two images keep their roles. */
        craw_vk_image_barrier(ctx, cmd, other, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        craw_vk_desc_image(&writes[0], 0, inter);
        craw_vk_desc_image(&writes[1], 1, other);
        craw_pipeline_dispatch(p, cmd, &p->df_prep, writes, 2, &dfp, sizeof(dfp), w, h, 1);

        craw_vk_image_barrier(ctx, cmd, other, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        craw_vk_image_barrier(ctx, cmd, inter, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                              VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        craw_vk_desc_image(&writes[0], 0, other);
        craw_vk_desc_image(&writes[1], 1, inter);
        craw_pipeline_dispatch(p, cmd, &p->df, writes, 2, &dfp, sizeof(dfp), w, h, 1);

        craw_vk_image_barrier(ctx, cmd, inter, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    }

    if (chroma) {
        /* Down to the quarter grid, the bilateral there, and the band back
         * into every pixel: DESIGN.md 14.4. */
        craw_vk_image_barrier(ctx, cmd, &p->chroma_q, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        craw_vk_desc_image(&writes[0], 0, inter);
        craw_vk_desc_image(&writes[1], 1, &p->chroma_q);
        craw_pipeline_dispatch(p, cmd, &p->chroma_down, writes, 2, &cdp, sizeof(cdp),
                               (uint32_t)p->qsize[0], (uint32_t)p->qsize[1], 1);

        craw_vk_image_barrier(ctx, cmd, &p->chroma_q, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        craw_vk_image_barrier(ctx, cmd, &p->chroma_qf, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        craw_vk_desc_image(&writes[0], 0, &p->chroma_q);
        craw_vk_desc_image(&writes[1], 1, &p->chroma_qf);
        craw_pipeline_dispatch(p, cmd, &p->chroma_blur, writes, 2, &cdp, sizeof(cdp),
                               (uint32_t)p->qsize[0], (uint32_t)p->qsize[1], 1);

        craw_vk_image_barrier(ctx, cmd, &p->chroma_qf, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        craw_vk_image_barrier(ctx, cmd, other, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        craw_vk_desc_image(&writes[0], 0, inter);
        craw_vk_desc_image(&writes[1], 1, &p->chroma_q);
        craw_vk_desc_image(&writes[2], 2, &p->chroma_qf);
        craw_vk_desc_image(&writes[3], 3, other);
        craw_pipeline_dispatch(p, cmd, &p->chroma_up, writes, 4, &cdp, sizeof(cdp), w, h, 1);

        craw_vk_image_barrier(ctx, cmd, other, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        CRAWVkImage *t = inter;
        inter = other;
        other = t;
    }

    if (sharpen) {
        /* DESIGN.md 20.3: the observed luma, then the iterations, the first
         * reading the luma as its estimate and each writing the plane the
         * one before it did not, and the last multiplying the
         * intermediate's own texels in place, so the two full resolution
         * images keep their roles. The planes are buffers, whose barriers
         * are spelled out: each write waits for the reads before it, in
         * this recording or the last one. */
        const VkPipelineStageFlags2 cs = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        const VkAccessFlags2 rd = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        const VkAccessFlags2 wr = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        CRAWVkBuffer *u = &p->sh_d;

        craw_vk_buffer_barrier(ctx, cmd, &p->sh_d, cs, rd, cs, wr);
        craw_vk_desc_image(&writes[0], 0, inter);
        craw_vk_desc_buffer(&writes[1], 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &p->sh_d, 0, 0);
        craw_pipeline_dispatch(p, cmd, &p->sh_prep, writes, 2, &shp, sizeof(shp),
                               (uint32_t)p->sh_stride, h, 1);
        craw_vk_buffer_barrier(ctx, cmd, &p->sh_d, cs, wr, cs, rd);

        craw_vk_image_barrier(ctx, cmd, inter, VK_IMAGE_LAYOUT_GENERAL, cs, rd | wr);
        for (uint32_t it = 0; it < fp->sharpen_iterations; it++) {
            bool last = it + 1 == fp->sharpen_iterations;
            CRAWVkBuffer *out = &p->sh_u[it & 1];

            /* The last iteration writes no plane, but the binding is part
             * of the shader and is given the one it would have written. */
            if (!last)
                craw_vk_buffer_barrier(ctx, cmd, out, cs, rd, cs, wr);
            craw_vk_desc_buffer(&writes[0], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, u, 0, 0);
            craw_vk_desc_buffer(&writes[1], 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &p->sh_d, 0, 0);
            craw_vk_desc_buffer(&writes[2], 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, out, 0, 0);
            craw_vk_desc_image(&writes[3], 3, inter);
            if (!last && p->sh_rows[sh_reach - 1].pipeline) {
                /* One subgroup per strip: its output width across, its
                 * rows down. */
                uint32_t ow = 4 * p->sh_lanes - 4 * (uint32_t)sh_reach;

                craw_pipeline_dispatch(p, cmd, &p->sh_rows[sh_reach - 1], writes, 4,
                                       &shp, sizeof(shp), p->sh_lanes * ((w + ow - 1) / ow),
                                       (h + CRAW_SH_STRIP_ROWS - 1) / CRAW_SH_STRIP_ROWS, 1);
            } else {
                craw_pipeline_dispatch(p, cmd, &p->sh_iter[last][sh_reach - 1], writes, 4,
                                       &shp, sizeof(shp), w, h, 1);
            }
            if (!last)
                craw_vk_buffer_barrier(ctx, cmd, out, cs, wr, cs, rd);
            u = out;
        }
        craw_vk_image_barrier(ctx, cmd, inter, VK_IMAGE_LAYOUT_GENERAL, cs, rd);
    }

    if (p->awb) {
        ap = (CRAWAwbPush) {
            .blocks  = { p->blocks[0], p->blocks[1] },
            .wb      = { frame.wb[0], frame.wb[1], frame.wb[2] },
            .bins    = CRAW_AWB_HIST_BINS,
            .log_min = CRAW_AWB_HIST_LOG_MIN,
            .log_max = CRAW_AWB_HIST_LOG_MAX,
        };
        craw_vk_desc_image(&writes[0], 0, inter);
        craw_vk_desc_buffer(&writes[1], 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            &p->hist_count, 0, CRAW_AWB_HIST_BYTES);
        craw_vk_desc_buffer(&writes[2], 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            &p->hist_weight, 0, CRAW_AWB_HIST_BYTES);
        craw_vk_desc_buffer(&writes[3], 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            &p->hist_bright, 0, CRAW_AWB_HIST_BYTES);
        craw_pipeline_dispatch(p, cmd, &p->awb_hist, writes, 4, &ap, sizeof(ap),
                               (uint32_t)p->blocks[0], (uint32_t)p->blocks[1], 1);
    }

    craw_vk_image_barrier(ctx, cmd, &dst_img, VK_IMAGE_LAYOUT_GENERAL,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    craw_vk_desc_image(&writes[0], 0, inter);
    craw_vk_desc_image(&writes[1], 1, &dst_img);
    craw_pipeline_dispatch(p, cmd, &p->color, writes, 2, &cp, sizeof(cp), w, h, 1);

    if (p->awb) {
        VkBufferCopy region = { .srcOffset = 0, .dstOffset = 0, .size = CRAW_AWB_HIST_BYTES };

        craw_vk_buffer_barrier(ctx, cmd, &p->hist_count,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                               VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        craw_vk_buffer_barrier(ctx, cmd, &p->hist_weight,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                               VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        craw_vk_buffer_barrier(ctx, cmd, &p->hist_bright,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                               VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

        CRAWVkBuffer *rb = &p->hist_readback[p->nb_recorded % p->nb_slots];

        ctx->fn.CmdCopyBuffer(cmd, p->hist_count.buf, rb->buf, 1, &region);
        region.dstOffset = CRAW_AWB_HIST_BYTES;
        ctx->fn.CmdCopyBuffer(cmd, p->hist_weight.buf, rb->buf, 1, &region);
        region.dstOffset = 2 * CRAW_AWB_HIST_BYTES;
        ctx->fn.CmdCopyBuffer(cmd, p->hist_bright.buf, rb->buf, 1, &region);

        craw_vk_buffer_barrier(ctx, cmd, rb,
                               VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                               VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
        p->nb_recorded++;
    }

    /* Hand both of the caller's images back in the layout it stated. */
    craw_vk_image_barrier(ctx, cmd, &dst_img, dst->layout,
                          VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                          VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    craw_vk_image_barrier(ctx, cmd, &src_img, src->layout,
                          VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                          VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);

    if (p->prof_pool)
        p->prof_recorded++;

    /* The recording is made; only now does the temporal state move. The
     * set this one read is what a replay of it reads. A next frame without
     * the pass leaves nothing for the frame after it, while a replay
     * without it leaves the frame's earlier development in place. */
    if (p->has_temporal) {
        p->tn_prev = t_read;
        if (temporal)
            p->tn_cur = t_write;
        else if (!replay)
            p->tn_cur = -1;
    }

    return 0;
}

int craw_pipeline_get_awb_histogram_n(CRAWPipeline *p, uint32_t frames_back,
                                      CRAWAwbHistogram *out)
{
    CRAWVkBuffer *rb;
    void *base;
    int ret;

    if (!p || !out)
        return CRAW_ERROR(EINVAL);

    if (!p->awb) {
        craw_log(p->ctx, CRAW_LOG_ERROR,
                 "Pipeline was created without CRAW_PIPELINE_FLAG_AWB_HISTOGRAM\n");
        return CRAW_ERROR(EINVAL);
    }
    if (frames_back >= p->nb_slots || frames_back >= p->nb_recorded)
        return CRAW_ERROR(EINVAL);

    rb = &p->hist_readback[(p->nb_recorded - 1 - frames_back) % p->nb_slots];

    ret = craw_vk_buffer_invalidate(p->ctx, rb);
    if (ret < 0)
        return ret;

    base = rb->mapped;
    if (!base)
        return CRAW_ERROR(EINVAL);

    *out = (CRAWAwbHistogram) {
        .bins    = CRAW_AWB_HIST_BINS,
        .log_min = CRAW_AWB_HIST_LOG_MIN,
        .log_max = CRAW_AWB_HIST_LOG_MAX,
        .count   = base,
        .weight  = (const uint32_t *)((const unsigned char *)base + CRAW_AWB_HIST_BYTES),
        .weight_bright = (const uint32_t *)((const unsigned char *)base + 2 * CRAW_AWB_HIST_BYTES),
    };

    return 0;
}

int craw_pipeline_get_awb_histogram(CRAWPipeline *p, CRAWAwbHistogram *out)
{
    return craw_pipeline_get_awb_histogram_n(p, 0, out);
}

int craw_pipeline_get_stage_times_n(CRAWPipeline *p, uint32_t frames_back,
                                    CRAWStageTime *out, uint32_t max)
{
    CRAWContext *ctx;
    uint32_t slot, nb, nb_agg = 0;
    VkResult res;

    if (!p || (max && !out))
        return CRAW_ERROR(EINVAL);

    ctx = p->ctx;
    if (!p->prof_pool) {
        craw_log(ctx, CRAW_LOG_ERROR, "Pipeline was created without CRAW_PIPELINE_FLAG_PROFILE\n");
        return CRAW_ERROR(EINVAL);
    }
    if (frames_back >= CRAW_PIPELINE_PROFILE_SLOTS || frames_back >= p->prof_recorded)
        return CRAW_ERROR(EINVAL);

    slot = (uint32_t)((p->prof_recorded - 1 - frames_back) % CRAW_PIPELINE_PROFILE_SLOTS);
    nb = p->prof_nb[slot];
    if (!nb)
        return 0;

    /* No WAIT_BIT: a submission that has not completed is the caller's
     * mistake, and it should hear about it rather than block on it. */
    res = ctx->fn.GetQueryPoolResults(ctx->device, p->prof_pool, 2 * slot * p->prof_max,
                                      2 * nb, 2 * nb * sizeof(*p->prof_ts), p->prof_ts,
                                      sizeof(*p->prof_ts), VK_QUERY_RESULT_64_BIT);
    if (res == VK_NOT_READY)
        return CRAW_ERROR(EAGAIN);
    if (res != VK_SUCCESS)
        return craw_vk_ret2err(res);

    for (uint32_t i = 0; i < nb; i++) {
        const char *name = p->prof_sh[slot * p->prof_max + i]->name;
        uint64_t ticks = (p->prof_ts[2 * i + 1] - p->prof_ts[2 * i]) & p->prof_mask;
        double ms = (double)ticks * p->prof_period / 1e6;
        uint32_t j;

        for (j = 0; j < nb_agg; j++)
            if (!strcmp(p->prof_agg[j].name, name))
                break;
        if (j == nb_agg)
            p->prof_agg[nb_agg++] = (CRAWStageTime) { .name = name };
        p->prof_agg[j].count++;
        p->prof_agg[j].ms += ms;
    }

    if (max)
        memcpy(out, p->prof_agg, CRAW_MIN(max, nb_agg) * sizeof(*out));

    return (int)nb_agg;
}

int craw_pipeline_get_stage_times(CRAWPipeline *p, CRAWStageTime *out, uint32_t max)
{
    return craw_pipeline_get_stage_times_n(p, 0, out, max);
}
