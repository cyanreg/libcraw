/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* The GPU pipeline: owns the compute shaders, the intermediate images and
 * the auto white balance buffers, and records one frame's worth of
 * dispatches and barriers into a command buffer the caller owns. See
 * DESIGN.md sections 4, 5 and 10. */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <craw/pipeline.h>

#include "vk_runtime.h"
#include "gamut.h"

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

/* Every debayer pass shares this block; it mirrors CRAWPrepPushBuf in
 * lib/shaders/common.glsl field for field. Scalar layout gives every member
 * 4-byte alignment and no padding, which is exactly what C gives these
 * types, so the static asserts below are the whole ABI check.
 *
 * inv_range is 1 / (white - black) rather than the white level itself: the
 * prep of DESIGN.md 4.1 divides by the range at every one of the up to 49
 * samples an RCD invocation reads, and a reciprocal computed once on the
 * host costs at most an ulp against the tolerances of section 11. */
typedef struct CRAWPrepPush {
    int32_t size[2];
    float   black[4];
    float   inv_range[4];
    float   wb[4];        /* as-shot gains in [0..2], [3] unused */
} CRAWPrepPush;

static_assert(sizeof(CRAWPrepPush) == 56, "CRAWPrepPush layout");
static_assert(offsetof(CRAWPrepPush, black) == 8, "CRAWPrepPush layout");
static_assert(offsetof(CRAWPrepPush, inv_range) == 24, "CRAWPrepPush layout");
static_assert(offsetof(CRAWPrepPush, wb) == 40, "CRAWPrepPush layout");

/* Mirrors CRAWDenoisePushBuf in lib/shaders/denoise_nlm.comp.glsl. The
 * transform constants are all derived from the frame's (a, b) on the host,
 * so the shader never divides by a value it cannot check and the CPU
 * reference in lib/ref/ref_denoise.c can be handed the same numbers. */
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
} CRAWDenoisePush;

static_assert(sizeof(CRAWDenoisePush) == 92, "CRAWDenoisePush layout");
static_assert(offsetof(CRAWDenoisePush, black) == 8, "CRAWDenoisePush layout");
static_assert(offsetof(CRAWDenoisePush, range) == 40, "CRAWDenoisePush layout");
static_assert(offsetof(CRAWDenoisePush, gat_a) == 56, "CRAWDenoisePush layout");
static_assert(offsetof(CRAWDenoisePush, inv_h2) == 80, "CRAWDenoisePush layout");

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

    CRAWVkImage inter;   /* camera RGB, RGBA16F */
    CRAWVkImage aux;     /* RCD: (L, wv, wp, unused), RGBA16F */
    CRAWVkImage green;   /* RCD: pass B green plane, R16F */
    CRAWVkImage clean;   /* denoise: the filtered mosaic, R16_UNORM */
    CRAWVkImage fc;      /* false colour: the filtered intermediate, RGBA16F,
                          * only without RCD, whose aux is borrowed instead */
    CRAWVkImage linear;  /* lens: the colour stage's output, out_format */

    CRAWVkBuffer hist_count;
    CRAWVkBuffer hist_weight;
    CRAWVkBuffer hist_bright;
    CRAWVkBuffer hist_readback[4];
    uint32_t     nb_slots;
    uint64_t     nb_recorded;

    int32_t blocks[2];   /* whole AWB blocks */

    /* The last keep-everything scale and the record it was worked out for.
     * craw_lens_keep_all_scale() is a boundary walk of a few hundred
     * microseconds, and a clip that is not zooming hands over byte for byte
     * the same record on every frame; the compare is 64 bytes. A miss only
     * costs the walk, so a caller whose reserved bytes are not zeroed loses
     * nothing but the cache. */
    CRAWLensCorrection keep_lc;
    float keep_scale;
    bool have_keep_scale;
};

static const CRAWVkBinding craw_bind_images[4] = {
    { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE },
    { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE },
    { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE },
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
 * (false_color.comp.glsl). A shader that does not declare one of these
 * ignores its map entry, so every shader is given the prefix of this
 * array it needs. */
static const VkSpecializationMapEntry craw_spec_map[5] = {
    { .constantID = 0, .offset = 0,                    .size = sizeof(uint32_t) },
    { .constantID = 1, .offset = 1 * sizeof(uint32_t), .size = sizeof(uint32_t) },
    { .constantID = 2, .offset = 2 * sizeof(uint32_t), .size = sizeof(uint32_t) },
    { .constantID = 3, .offset = 3 * sizeof(uint32_t), .size = sizeof(uint32_t) },
    { .constantID = 4, .offset = 4 * sizeof(uint32_t), .size = sizeof(uint32_t) },
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

static int craw_pipeline_shader(CRAWPipeline *p, CRAWVkShader *sh, const char *name,
                                const uint8_t *spirv, size_t spirv_len,
                                const CRAWVkBinding *bindings, uint32_t nb_bindings,
                                uint32_t push_size, uint32_t wg,
                                const uint32_t *spec, uint32_t nb_spec)
{
    CRAWVkShaderDesc desc = {
        .name           = name,
        .spirv          = spirv,
        .spirv_len      = spirv_len,
        .bindings       = bindings,
        .nb_bindings    = nb_bindings,
        .push_size      = push_size,
        .wg             = { wg, wg, 1 },
        .spec_entries   = nb_spec ? craw_spec_map : NULL,
        .nb_spec_entries = nb_spec,
        .spec_data      = nb_spec ? spec : NULL,
        .spec_size      = nb_spec * sizeof(uint32_t),
    };

    return craw_vk_shader_create(p->ctx, sh, &desc);
}

int craw_pipeline_create(CRAWContext *ctx, const CRAWPipelineParams *params,
                         CRAWPipeline **pipe)
{
    CRAWPipeline *p;
    uint32_t spec[5];
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

    if (p->has_denoise) {
        /* The denoise stage mirrors inside each CFA plane, and one
         * reflection only reaches a stencil radius of plane size - 1, so a
         * frame smaller than this would read outside the plane. Every real
         * frame is orders of magnitude bigger; this exists so that the
         * rule of DESIGN.md 3 is enforced rather than assumed. */
        const uint32_t dn_min = 2 * (CRAW_DN_SEARCH_R + CRAW_DN_PATCH_R + 1);

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
                                   craw_bind_images, 2, sizeof(CRAWDenoisePush), wgdn,
                                   spec, 4);
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

    if (p->has_fc) {
        /* The stage cannot run in place, so it needs a second intermediate
         * to write. With RCD that is the aux image: pass A writes it, pass
         * C is its last reader, and nothing touches it again until the
         * next frame's pass A, so the stage borrows it and the pipeline
         * costs no more memory than without the flag. The other two
         * methods have no such image and get one. DESIGN.md 17.3. */
        if (!p->rcd) {
            ret = craw_vk_image_create(ctx, &p->fc, params->width, params->height,
                                       VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT);
            if (ret < 0)
                goto fail;
        }

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
        /* The warp reads what the colour stage wrote, so with the stage
         * built the colour stage writes here instead of the caller's
         * destination. The format follows the pipeline's output rather
         * than being fixed at RGBA16F: a caller who asked for RGBA32F did
         * so to keep 32 bits all the way, and an internal image is no
         * place to quantize them away. With the default output this is
         * exactly the RGBA16F intermediate DESIGN.md 16 describes. */
        ret = craw_vk_image_create(ctx, &p->linear, params->width, params->height,
                                   params->out_format, VK_IMAGE_USAGE_STORAGE_BIT);
        if (ret < 0)
            goto fail;

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

    craw_log(ctx, CRAW_LOG_VERBOSE, "Pipeline: %ux%u, debayer %d, workgroup %ux%u%s%s%s%s\n",
             params->width, params->height, (int)params->debayer, wg2d, wg2d,
             p->awb ? ", awb histogram" : "",
             p->has_denoise ? ", denoise" : "",
             p->has_fc ? (wgfc == CRAW_FC_WG_MAX ? ", false colour" : ", false colour (small tile)") : "",
             p->has_lens ? ", lens" : "");

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

    if (p->inter.image)
        craw_vk_image_free(ctx, &p->inter);
    if (p->aux.image)
        craw_vk_image_free(ctx, &p->aux);
    if (p->green.image)
        craw_vk_image_free(ctx, &p->green);
    if (p->clean.image)
        craw_vk_image_free(ctx, &p->clean);
    if (p->fc.image)
        craw_vk_image_free(ctx, &p->fc);
    if (p->linear.image)
        craw_vk_image_free(ctx, &p->linear);

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

/* Resolves the frame's noise model and turns it into the shader's
 * constants, DESIGN.md 14.3. The model is looked up at the sensor's own
 * exposure index (as_shot_ei) and not at the working one: rating a clip at
 * a different index scales the picture afterwards, it does not change how
 * the sensels were read. */
static int craw_pipeline_denoise_push(CRAWPipeline *p, const CRAWProfile *prof,
                                      const CRAWFrameParams *fp, CRAWDenoisePush *dp)
{
    CRAWNoiseModel nm;
    float a, b, c, sc;
    int ret;

    if (fp->noise_override.a > 0.0f) {
        nm = fp->noise_override;
    } else {
        ret = craw_profile_noise_eval(prof, prof->as_shot_ei, &nm);
        if (ret < 0) {
            craw_log(p->ctx, CRAW_LOG_ERROR,
                     "Denoising needs a noise model: the profile `%s` has no table and "
                     "CRAWFrameParams.noise_override was not set\n",
                     prof->name[0] ? prof->name : "(unnamed)");
            return ret;
        }
    }

    a = nm.a;
    b = CRAW_MAX(nm.b, 0.0f);
    if (!(a > 0.0f)) {
        craw_log(p->ctx, CRAW_LOG_ERROR, "Noise model a = %g is not positive\n", (double)a);
        return CRAW_ERROR(EINVAL);
    }

    /* The 3 a^2 / 8 of the Anscombe transform, plus the read term. */
    c = 3.0f * a * a / 8.0f + b;
    sc = sqrtf(c);

    *dp = (CRAWDenoisePush) {
        .size      = { (int32_t)(p->params.width / 2), (int32_t)(p->params.height / 2) },
        .gat_a     = a,
        .gat_c     = c,
        .gat_sc    = sc,
        .gat_fwd   = 2.0f / a,
        .gat_inv   = a / 4.0f,
        .gat_off   = c / a,
        .inv_h2    = 1.0f / (fp->denoise_strength * fp->denoise_strength),
        .guard_lo  = CRAW_DN_GUARD_LO,
        .guard_inv = 1.0f / (CRAW_DN_GUARD_HI - CRAW_DN_GUARD_LO),
    };

    for (int i = 0; i < 4; i++) {
        float range = prof->white_level[i] - prof->black_level[i];

        dp->black[i] = prof->black_level[i];
        dp->inv_range[i] = 1.0f / range;
        dp->range[i] = range;
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

/* `gz` is 1 for every pass but the denoise, which puts the CFA plane index
 * there and dispatches all four at once. */
static void craw_pipeline_dispatch(CRAWPipeline *p, VkCommandBuffer cmd, const CRAWVkShader *sh,
                                   CRAWVkDescWrite *writes, uint32_t nb_writes,
                                   const void *push, uint32_t push_size,
                                   uint32_t gx, uint32_t gy, uint32_t gz)
{
    craw_vk_shader_bind(p->ctx, cmd, sh);
    craw_vk_shader_push_desc(p->ctx, cmd, sh, writes, nb_writes);
    craw_vk_shader_push_const(p->ctx, cmd, sh, push, push_size);
    craw_vk_shader_dispatch(p->ctx, cmd, sh, gx, gy, gz);
}

int craw_pipeline_record(CRAWPipeline *p, VkCommandBuffer cmd,
                         const CRAWImage *src, const CRAWImage *dst,
                         const CRAWFrameParams *fp)
{
    CRAWContext *ctx;
    CRAWVkImage src_img, dst_img;
    CRAWVkImage *mosaic, *inter, *color_dst;
    CRAWVkDescWrite writes[4];
    CRAWColorParams frame;
    CRAWPrepPush pp;
    CRAWColorPush cp;
    CRAWAwbPush ap;
    CRAWDenoisePush dp;
    CRAWLensPush lp;
    CRAWFalseColorPush fcp;
    enum CRAWColorSpace out_space;
    bool denoise, lens, false_color;
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

    /* A strength of zero is the documented way to turn the pass off for one
     * frame, so it is not an error on a pipeline that was never built with
     * it either. */
    denoise = p->has_denoise && fp->denoise_strength > 0.0f;
    if (denoise) {
        ret = craw_pipeline_denoise_push(p, fp->profile, fp, &dp);
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

    /* A correction marked invalid is the documented way to leave one frame
     * alone, so the whole warp drops out: the colour stage writes the
     * caller's destination as it would without the pass and the result is
     * bit for bit the uncorrected one. */
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
        craw_vk_image_barrier(ctx, cmd, &p->clean, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        craw_vk_desc_image(&writes[0], 0, &src_img);
        craw_vk_desc_image(&writes[1], 1, &p->clean);
        /* Half resolution, one workgroup slice per CFA plane. */
        craw_pipeline_dispatch(p, cmd, &p->denoise, writes, 2, &dp, sizeof(dp),
                               w / 2, h / 2, 4);

        craw_vk_image_barrier(ctx, cmd, &p->clean, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        mosaic = &p->clean;
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
     * the filtered copy the false colour stage makes of it. The copy is
     * RCD's aux image when there is one (its last reader was pass C, just
     * above) and the stage's own otherwise. DESIGN.md 17.3. */
    inter = &p->inter;
    if (false_color) {
        CRAWVkImage *filtered = p->rcd ? &p->aux : &p->fc;

        craw_vk_image_barrier(ctx, cmd, filtered, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        craw_vk_desc_image(&writes[0], 0, &p->inter);
        craw_vk_desc_image(&writes[1], 1, filtered);
        craw_pipeline_dispatch(p, cmd, &p->false_color, writes, 2, &fcp, sizeof(fcp), w, h, 1);

        craw_vk_image_barrier(ctx, cmd, filtered, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        inter = filtered;
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

    /* With the warp running, the colour stage lands in the intermediate the
     * warp reads and the warp writes the caller's destination; without it,
     * the colour stage is the last pass, exactly as before. */
    color_dst = lens ? &p->linear : &dst_img;

    craw_vk_image_barrier(ctx, cmd, color_dst, VK_IMAGE_LAYOUT_GENERAL,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    craw_vk_desc_image(&writes[0], 0, inter);
    craw_vk_desc_image(&writes[1], 1, color_dst);
    craw_pipeline_dispatch(p, cmd, &p->color, writes, 2, &cp, sizeof(cp), w, h, 1);

    if (lens) {
        craw_vk_image_barrier(ctx, cmd, &p->linear, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        craw_vk_image_barrier(ctx, cmd, &dst_img, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        craw_vk_desc_image(&writes[0], 0, &p->linear);
        craw_vk_desc_image(&writes[1], 1, &dst_img);
        craw_pipeline_dispatch(p, cmd, &p->lens, writes, 2, &lp, sizeof(lp), w, h, 1);
    }

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
