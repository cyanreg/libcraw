/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* GPU pipeline against the CPU reference, DESIGN.md 11.
 *
 * A synthetic 256x160 RGGB mosaic goes through craw_pipeline_record() and
 * through craw_ref_process(), and the two are compared per debayer method
 * and per output colour space. The first configuration of every method uses
 * a profile whose matrix is the identity and whose gain is 1.0 with XYZ D65
 * output, which makes the colour stage a no-op: the destination is then the
 * camera RGB intermediate itself and can be held to the intermediate
 * tolerance of DESIGN.md 11, and its GPU auto white balance histogram can be
 * compared against craw_awb_histogram_from_rgb() on the reference
 * intermediate.
 *
 * Set CRAW_TEST_FRAME to a raw little endian uint16 5760x3040 RGGB dump to
 * additionally time and check the RCD path on a real frame; the file is
 * never required. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>

#include <craw/craw.h>
#include <craw/vulkan.h>
#include <craw/profile.h>
#include <craw/pipeline.h>
#include <craw/awb.h>
#include <craw/noise.h>

#include "vkdev.h"
#include "gamut.h"
#include "ref/ref.h"

#define TEST_W 256
#define TEST_H 160

#define FRAME_W 5760
#define FRAME_H 3040

/* DESIGN.md 9, Panasonic DC-S1M2 clip P1000387.MOV. */
#define TEST_BLACK (256.0f / 65535.0f)
#define TEST_WHITE (63743.0f / 65535.0f)

static const float test_wb[3] = { 2.02539f, 1.0f, 1.66797f };

static const float test_matrix[3][3] = {
    { 0.695646f, 0.152540f,  0.102269f },
    { 0.303238f, 0.760187f, -0.063425f },
    { 0.005748f, 0.031326f,  1.051984f },
};

static const CRAWWbKnot test_wb_knots[] = {
    {  2000.0f, { 0.8454f, 1.0f, 3.9482f } },
    {  3200.0f, { 1.3763f, 1.0f, 2.4707f } },
    {  3400.0f, { 1.4477f, 1.0f, 2.2881f } },
    {  5000.0f, { 1.9499f, 1.0f, 1.7195f } },
    {  5500.0f, { 2.0253f, 1.0f, 1.6676f } },
    {  6300.0f, { 2.2051f, 1.0f, 1.5348f } },
    {  7300.0f, { 2.3299f, 1.0f, 1.4186f } },
    { 11500.0f, { 2.6469f, 1.0f, 1.2318f } },
};

static const CRAWMatrixKnot test_matrix_knots[] = {
    {  2300.0f, { { 0.6535f, 0.1947f, 0.1022f }, { 0.2831f, 0.6975f,  0.0193f }, { 0.0113f, -0.1074f, 1.1852f } } },
    {  2800.0f, { { 0.6601f, 0.1881f, 0.1022f }, { 0.2862f, 0.7073f,  0.0064f }, { 0.0104f, -0.0857f, 1.1644f } } },
    {  4000.0f, { { 0.6759f, 0.1723f, 0.1023f }, { 0.2938f, 0.7308f, -0.0246f }, { 0.0083f, -0.0337f, 1.1144f } } },
    {  5000.0f, { { 0.6891f, 0.1591f, 0.1023f }, { 0.3001f, 0.7504f, -0.0505f }, { 0.0066f,  0.0096f, 1.0728f } } },
    {  6500.0f, { { 0.7088f, 0.1394f, 0.1023f }, { 0.3095f, 0.7798f, -0.0893f }, { 0.0040f,  0.0747f, 1.0104f } } },
    {  7500.0f, { { 0.7218f, 0.1263f, 0.1023f }, { 0.3158f, 0.7991f, -0.1149f }, { 0.0023f,  0.1176f, 0.9692f } } },
    { 10000.0f, { { 0.7218f, 0.1263f, 0.1023f }, { 0.3158f, 0.7991f, -0.1149f }, { 0.0023f,  0.1176f, 0.9692f } } },
    { 15000.0f, { { 0.7218f, 0.1263f, 0.1023f }, { 0.3158f, 0.7991f, -0.1149f }, { 0.0023f,  0.1176f, 0.9692f } } },
};

static void test_profile_init(CRAWProfile *p, bool identity_color)
{
    craw_profile_init(p);

    p->cfa = CRAW_CFA_RGGB;
    for (int i = 0; i < 4; i++) {
        p->black_level[i] = TEST_BLACK;
        p->white_level[i] = TEST_WHITE;
    }
    memcpy(p->as_shot_wb, test_wb, sizeof(p->as_shot_wb));
    p->as_shot_cct = 5500.0f;

    if (identity_color) {
        /* Makes the colour stage the identity, so that the destination is
         * the camera RGB intermediate. */
        memset(p->as_shot_matrix, 0, sizeof(p->as_shot_matrix));
        for (int i = 0; i < 3; i++)
            p->as_shot_matrix[i][i] = 1.0f;
        p->gain = 1.0f;
    } else {
        memcpy(p->as_shot_matrix, test_matrix, sizeof(p->as_shot_matrix));
        p->gain = 23.04f;
    }

    memcpy(p->wb_knots, test_wb_knots, sizeof(test_wb_knots));
    p->nb_wb_knots = (uint32_t)(sizeof(test_wb_knots) / sizeof(test_wb_knots[0]));
    memcpy(p->matrix_knots, test_matrix_knots, sizeof(test_matrix_knots));
    p->nb_matrix_knots = (uint32_t)(sizeof(test_matrix_knots) / sizeof(test_matrix_knots[0]));

    snprintf(p->name, sizeof(p->name), "Panasonic DC-S1M2");
}

/* ---------------------------------------------------------------- scene */

static uint32_t test_hash(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static bool test_in_rect(int x, int y, int x0, int y0, int x1, int y1)
{
    return x >= x0 && x <= x1 && y >= y0 && y <= y1;
}

/* The scene is written in the prepped domain of DESIGN.md 4.1, i.e. as the
 * camera RGB a perfect debayer would produce, and inverted to raw samples
 * below. Away from the two hard rectangles nothing exceeds 0.9, so no 8x8
 * block mean lands near the histogram's 0.98 clip test or its 0.002 black
 * test and the GPU and the reference always agree on which blocks to skip.
 * The edges of those two rectangles bisect 8x8 blocks for the same reason:
 * a block is then either entirely clipped (or black) or roughly half, never
 * a hair away from a threshold. */
static void test_scene_rgb(int x, int y, float rgb[3])
{
    float fx = (float)x / (float)(TEST_W - 1);
    float fy = (float)y / (float)(TEST_H - 1);

    /* Each channel is affine in x and y, which bilinear and MHC reproduce
     * exactly away from the borders. */
    rgb[0] = 0.05f + 0.80f * fx;
    rgb[1] = 0.08f + 0.75f * fy;
    rgb[2] = 0.10f + 0.40f * fx + 0.35f * fy;

    if (test_in_rect(x, y, 16, 16, 47, 47)) {          /* warm flat patch */
        rgb[0] = 0.75f; rgb[1] = 0.30f; rgb[2] = 0.12f;
    }
    if (test_in_rect(x, y, 64, 96, 103, 127)) {        /* cool flat patch */
        rgb[0] = 0.12f; rgb[1] = 0.60f; rgb[2] = 0.80f;
    }
    if (test_in_rect(x, y, 160, 24, 199, 55)) {        /* neutral flat patch */
        rgb[0] = 0.45f; rgb[1] = 0.45f; rgb[2] = 0.45f;
    }
    if (test_in_rect(x, y, 208, 96, 247, 135)) {       /* 2x2 checkerboard */
        float v = (((x >> 1) + (y >> 1)) & 1) ? 0.82f : 0.08f;
        rgb[0] = v; rgb[1] = v; rgb[2] = v;
    }
}

static void test_gen_mosaic(uint16_t *m)
{
    for (int y = 0; y < TEST_H; y++) {
        for (int x = 0; x < TEST_W; x++) {
            int s = (y & 1) * 2 + (x & 1);
            int c = (s == 0) ? 0 : (s == 3) ? 2 : 1;
            uint16_t v;

            if (test_in_rect(x, y, 108, 4, 139, 35)) {
                v = 65535;                      /* above white: hard clip */
            } else if (test_in_rect(x, y, 44, 116, 75, 147)) {
                v = 256;                        /* exactly black */
            } else {
                float rgb[3];
                float raw;
                int iv;

                test_scene_rgb(x, y, rgb);
                raw = TEST_BLACK + (rgb[c] / test_wb[c]) * (TEST_WHITE - TEST_BLACK);
                iv = (int)lrintf(raw * 65535.0f);
                /* A deterministic +-8 count noise term so that no stage can
                 * pass by accident on a perfectly smooth signal. */
                iv += (int)(test_hash((uint32_t)(y * TEST_W + x) * 2654435761u) % 17u) - 8;
                iv = iv < 0 ? 0 : (iv > 65535 ? 65535 : iv);
                v = (uint16_t)iv;
            }

            m[y * TEST_W + x] = v;
        }
    }
}

/* ------------------------------------------------------------- plumbing */

static float test_half_to_float(uint16_t h)
{
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    uint32_t o;
    float f;

    if (e == 0) {
        if (m == 0) {
            o = s;
        } else {
            int sh = 0;
            while (!(m & 0x400u)) {
                m <<= 1;
                sh++;
            }
            o = s | ((uint32_t)(127 - 15 - sh) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (e == 31) {
        o = s | 0x7F800000u | (m << 13);
    } else {
        o = s | ((e - 15 + 127) << 23) | (m << 13);
    }

    memcpy(&f, &o, sizeof(f));
    return f;
}

struct test_run {
    CRAWPipeline *pipe;
    CRAWTestImage *src;
    CRAWTestImage *dst;
    const CRAWFrameParams *fp;
};

static int test_record(VkCommandBuffer cmd, void *opaque)
{
    struct test_run *r = opaque;
    return craw_pipeline_record(r->pipe, cmd, &r->src->img, &r->dst->img, r->fp);
}

/* Creates a pipeline, runs one frame and downloads the RGBA16F result as
 * floats. `hist_count` and `hist_weight`, when given, receive a copy of the
 * GPU histogram before the pipeline goes away. */
static int test_run_config(CRAWTestDevice *td, CRAWContext *ctx,
                           CRAWTestImage *src, CRAWTestImage *dst,
                           const CRAWPipelineParams *pp, const CRAWFrameParams *fp,
                           float *out, uint32_t *hist_count, uint32_t *hist_weight,
                           double *ms)
{
    size_t npix = (size_t)pp->width * pp->height;
    CRAWPipeline *pipe = NULL;
    uint16_t *half = NULL;
    struct test_run run;
    int ret;

    ret = craw_pipeline_create(ctx, pp, &pipe);
    if (ret < 0) {
        fprintf(stderr, "craw_pipeline_create failed: %d\n", ret);
        return ret;
    }

    run = (struct test_run) { .pipe = pipe, .src = src, .dst = dst, .fp = fp };
    ret = craw_testdev_run_timed(td, test_record, &run, ms);
    if (ret < 0) {
        fprintf(stderr, "submission failed: %d\n", ret);
        goto fail;
    }

    half = malloc(npix * 4 * sizeof(*half));
    if (!half) {
        ret = CRAW_ERROR(ENOMEM);
        goto fail;
    }

    ret = craw_testdev_image_download(td, dst, half, npix * 4 * sizeof(*half));
    if (ret < 0) {
        fprintf(stderr, "download failed: %d\n", ret);
        goto fail;
    }

    for (size_t i = 0; i < npix * 4; i++)
        out[i] = test_half_to_float(half[i]);

    if (hist_count) {
        CRAWAwbHistogram hist;
        ret = craw_pipeline_get_awb_histogram(pipe, &hist);
        if (ret < 0) {
            fprintf(stderr, "craw_pipeline_get_awb_histogram failed: %d\n", ret);
            goto fail;
        }
        if (hist.bins != CRAW_AWB_HIST_BINS) {
            fprintf(stderr, "histogram has %u bins, expected %u\n", hist.bins,
                    (unsigned)CRAW_AWB_HIST_BINS);
            ret = CRAW_ERROR(EINVAL);
            goto fail;
        }
        memcpy(hist_count, hist.count, CRAW_AWB_HIST_BINS * CRAW_AWB_HIST_BINS * sizeof(uint32_t));
        memcpy(hist_weight, hist.weight, CRAW_AWB_HIST_BINS * CRAW_AWB_HIST_BINS * sizeof(uint32_t));
    }

    ret = 0;

fail:
    free(half);
    craw_pipeline_free(&pipe);
    return ret;
}

/* Largest absolute difference over the three colour channels. */
static double test_max_abs(const float *gpu, const float *ref, size_t npix)
{
    double worst = 0.0;

    for (size_t i = 0; i < npix; i++) {
        for (int c = 0; c < 3; c++) {
            double e = fabs((double)gpu[i * 4 + c] - (double)ref[i * 4 + c]);
            if (e > worst)
                worst = e;
        }
    }

    return worst;
}

/* Largest error relative to the magnitude of the pixel rather than of the
 * single channel: individual XYZ (or BT.2020) components cross zero wherever
 * the scene leaves the camera's characterized gamut, and a per-channel
 * relative error would be unbounded there while the colour is still right.
 *
 * The per-pixel scale is floored at 1% of the frame's peak. Where the scene
 * is black the debayer amplifies fp16 storage noise relative to the pixel
 * (RCD divides by low-pass sums that are themselves near zero) while the
 * absolute error stays below 1e-3 of full scale, and a purely relative
 * bound there would measure nothing useful. Against the floor the dark
 * parts of the frame are still held to about 1e-4 of full scale. */
static double test_max_rel(const float *gpu, const float *ref, size_t npix)
{
    double worst = 0.0, floor_scale = 1e-9;

    for (size_t i = 0; i < npix; i++) {
        for (int c = 0; c < 3; c++) {
            double a = fabs((double)ref[i * 4 + c]);
            if (a * 0.01 > floor_scale)
                floor_scale = a * 0.01;
        }
    }

    for (size_t i = 0; i < npix; i++) {
        double scale = floor_scale;
        for (int c = 0; c < 3; c++) {
            double a = fabs((double)ref[i * 4 + c]);
            if (a > scale)
                scale = a;
        }
        for (int c = 0; c < 3; c++) {
            double e = fabs((double)gpu[i * 4 + c] - (double)ref[i * 4 + c]) / scale;
            if (e > worst)
                worst = e;
        }
    }

    return worst;
}

static const char *test_method_name(enum CRAWDebayer m)
{
    switch (m) {
    case CRAW_DEBAYER_BILINEAR: return "bilinear";
    case CRAW_DEBAYER_MHC:      return "mhc";
    case CRAW_DEBAYER_RCD:      return "rcd";
    default:                    return "?";
    }
}

static const char *test_space_name(enum CRAWColorSpace cs)
{
    switch (cs) {
    case CRAW_CS_XYZ_D65:         return "XYZ_D65";
    case CRAW_CS_BT2020_LINEAR:   return "BT2020";
    case CRAW_CS_ACES_AP0_LINEAR: return "AP0";
    case CRAW_CS_ACES_AP1_LINEAR: return "AP1";
    default:                      return "?";
    }
}

/* ------------------------------------------------------------ histogram */

/* Cancels every +1 against a -1 in one of the eight neighbouring bins. A
 * sample whose block mean moved by an fp16 ulp can only cross into an
 * adjacent bin, so those pairs cancel and what is left is a sample the GPU
 * put somewhere the reference did not. `scratch` holds bins*bins ints. */
static uint64_t test_hist_nonlocal(const uint32_t *gpu, const uint32_t *ref, int32_t *scratch)
{
    const int32_t n = CRAW_AWB_HIST_BINS;
    uint64_t left = 0;

    for (int32_t i = 0; i < n * n; i++)
        scratch[i] = (int32_t)gpu[i] - (int32_t)ref[i];

    for (int32_t y = 0; y < n; y++) {
        for (int32_t x = 0; x < n; x++) {
            int32_t i = y * n + x;
            for (int32_t dy = -1; dy <= 1 && scratch[i] > 0; dy++) {
                for (int32_t dx = -1; dx <= 1 && scratch[i] > 0; dx++) {
                    int32_t nx = x + dx, ny = y + dy, j, t;
                    if (nx < 0 || ny < 0 || nx >= n || ny >= n)
                        continue;
                    j = ny * n + nx;
                    if (scratch[j] >= 0)
                        continue;
                    t = CRAW_MIN(scratch[i], -scratch[j]);
                    scratch[i] -= t;
                    scratch[j] += t;
                }
            }
        }
    }

    for (int32_t i = 0; i < n * n; i++)
        left += (uint64_t)(scratch[i] < 0 ? -scratch[i] : scratch[i]);

    return left / 2;
}

static int test_check_histogram(const char *method, const uint32_t *gpu_count,
                                const uint32_t *gpu_weight, const uint32_t *ref_count,
                                const uint32_t *ref_weight, int32_t *scratch)
{
    const uint32_t nbins = CRAW_AWB_HIST_BINS * CRAW_AWB_HIST_BINS;
    uint64_t gpu_total = 0, ref_total = 0, moved = 0;
    uint64_t gpu_wtotal = 0, ref_wtotal = 0, nonlocal;
    double wdiff, allowed;

    for (uint32_t i = 0; i < nbins; i++) {
        gpu_total += gpu_count[i];
        ref_total += ref_count[i];
        gpu_wtotal += gpu_weight[i];
        ref_wtotal += ref_weight[i];
        moved += gpu_count[i] > ref_count[i] ? gpu_count[i] - ref_count[i]
                                             : ref_count[i] - gpu_count[i];
    }
    /* Every sample that landed in a different bin shows up twice. */
    moved /= 2;

    if (gpu_total != ref_total) {
        fprintf(stderr, "%s: awb histogram total count %llu, reference %llu\n", method,
                (unsigned long long)gpu_total, (unsigned long long)ref_total);
        return -1;
    }
    if (!gpu_total) {
        fprintf(stderr, "%s: awb histogram is empty\n", method);
        return -1;
    }

    /* fp16 storage moves a block mean by ~1e-4, which is enough to push a
     * sample sitting on a bin edge into the neighbouring bin: on the devices
     * this was written against that is 0.3% to 0.8% of the samples, and all
     * of it cancels against the adjacent bin. What must not happen is a
     * sample landing somewhere else (a wrong bin formula, a shifted grid, a
     * different block raster), so the hard limit is on the residual, and
     * `moved` only has to stay far away from the whole population, which a
     * systematic one-bin shift would not. */
    nonlocal = test_hist_nonlocal(gpu_count, ref_count, scratch);
    allowed = 0.01 * (double)gpu_total;
    wdiff = ref_wtotal ? fabs((double)gpu_wtotal - (double)ref_wtotal) / (double)ref_wtotal : 0.0;

    printf("  %-8s awb: %llu samples, %llu moved to an adjacent bin (%.2f%%), "
           "%llu elsewhere, weight %.3f%%\n", method,
           (unsigned long long)gpu_total, (unsigned long long)(moved - nonlocal),
           100.0 * (double)moved / (double)gpu_total, (unsigned long long)nonlocal,
           100.0 * wdiff);

    if ((double)nonlocal > allowed) {
        fprintf(stderr, "%s: %llu of %llu awb samples are in a bin that is not adjacent to the "
                "reference's, limit %.0f\n", method, (unsigned long long)nonlocal,
                (unsigned long long)gpu_total, allowed);
        return -1;
    }
    if ((double)moved > 0.05 * (double)gpu_total) {
        fprintf(stderr, "%s: %llu of %llu awb samples changed bin\n", method,
                (unsigned long long)moved, (unsigned long long)gpu_total);
        return -1;
    }
    if (wdiff > 0.01) {
        fprintf(stderr, "%s: awb weight total differs by %.3f%%\n", method, 100.0 * wdiff);
        return -1;
    }

    return 0;
}

/* ------------------------------------------------------------- denoise */

/* The noise the synthetic mosaic carries: a uniform +-8 count term, which
 * has a variance of (17^2 - 1) / 12 counts squared, expressed in the
 * normalized domain of DESIGN.md 14.1. The shot term is the real camera's,
 * so the model the stage is handed is a plausible one and the filter has
 * something to do at every level. */
#define TEST_NOISE_RANGE (TEST_WHITE - TEST_BLACK)
#define TEST_NOISE_B ((17.0 * 17.0 - 1.0) / 12.0 / \
                      (65535.0 * TEST_NOISE_RANGE * 65535.0 * TEST_NOISE_RANGE))
#define TEST_NOISE_A 1.3e-5

/* The denoise pass is a weighted mean of up to 49 stabilised samples,
 * inverted and re-encoded into 16 bits. What separates the GPU from the
 * reference is the order those 49 terms are summed in, an ulp or two in
 * exp(), and samples that land within an ulp of a 16-bit rounding
 * boundary and quantize the other way; the last of those is the largest
 * and is one code value of the mosaic, which the debayer and the white
 * balance turn into about 4e-5 of the intermediate. The bound below is the
 * intermediate tolerance of DESIGN.md 11 for the non-RCD methods, which
 * this comfortably clears. */
#define TEST_DENOISE_TOL 2e-3

/* The real frame goes through the colour stage of a real profile, so it is
 * held to the relative bound DESIGN.md 11 states for RCD output rather
 * than to an absolute one: its highlights sit far above 1.0, where fp16
 * storage alone is worth tens of millis absolute and means nothing. */
#define TEST_DENOISE_REAL_TOL 1.5e-2

/* The false colour stage is held to twice the debayer's own intermediate
 * tolerance on the synthetic frame: the output at a pixel is its G plus a
 * difference selected and averaged from its neighbours, so it carries
 * other pixels' fp16 and arithmetic disagreement, and the copy is stored
 * in fp16 once more. On the real frame it is held to this factor of the
 * plain chain's absolute error, for the reason test_real_frame() gives.
 * DESIGN.md 17.5. */
#define TEST_FC_TOL_FACTOR   2.0
#define TEST_FC_REAL_FACTOR  2.0

/* The defringe stage against the reference on the same input. Almost
 * every pixel agrees to an fp16 step; the exceptions are corners, where
 * the edge's normal turns fast and an E one fp16 step apart picks a
 * different side sample. DESIGN.md 19.5. */
#define TEST_DF_TOL          1e-2

/* The 3x3 [1 -2 1; -2 4 -2; 1 -2 1] / 6 high pass of one CFA plane has
 * unit gain on white noise variance and none at all on a bilinear signal,
 * so its median absolute value over a flat patch is 0.6745 noise standard
 * deviations whatever the patch's level. Used here to check that the stage
 * removes noise rather than merely producing the same numbers on both
 * sides. `x0, y0, x1, y1` are mosaic coordinates of a flat region. */
static int test_cmp_float(const void *a, const void *b)
{
    float x = *(const float *)a, y = *(const float *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

static double test_plane_noise(const uint16_t *m, int x0, int y0, int x1, int y1)
{
    static float res[1 << 14];
    size_t n = 0;

    for (int y = y0 + 2; y <= y1 - 2; y++) {
        for (int x = x0 + 2; x <= x1 - 2; x++) {
            double r;

            if (n >= CRAW_ARRAY_ELEMS(res))
                break;

            /* Steps of two stay inside one CFA plane. */
            r = (double)m[(y - 2) * TEST_W + x - 2] + m[(y - 2) * TEST_W + x + 2]
              + m[(y + 2) * TEST_W + x - 2] + m[(y + 2) * TEST_W + x + 2]
              - 2.0 * ((double)m[(y - 2) * TEST_W + x] + m[(y + 2) * TEST_W + x]
                     + m[y * TEST_W + x - 2] + m[y * TEST_W + x + 2])
              + 4.0 * m[y * TEST_W + x];
            res[n++] = (float)fabs(r / 6.0);
        }
    }

    if (!n)
        return 0.0;

    qsort(res, n, sizeof(*res), test_cmp_float);

    return res[n / 2] / 0.6744897502;
}

/* GPU against the reference with the denoise pass in front of the debayer,
 * plus the two ways the pass is switched off and the functional check that
 * it removes noise instead of merely producing the same numbers on both
 * sides.
 *
 * The first configuration is the plain chain, which fixes the floor every
 * other number is read against: the debayer's own GPU/CPU disagreement and
 * the fp16 storage of the intermediate are already there before this stage
 * adds anything. A denoised frame then has to stay inside the tolerance
 * against a denoised reference and, at the same time, sit well outside
 * that floor against an undenoised one, which is what a pass that silently
 * did not run could not manage. */
static int test_denoise(CRAWTestDevice *td, CRAWContext *ctx, const CRAWProfile *identity,
                        const uint16_t *mosaic, CRAWTestImage *src, CRAWTestImage *dst,
                        float *gpu, float *ref)
{
    static const float strengths[] = { 1.0f, 2.0f };
    const size_t npix = (size_t)TEST_W * TEST_H;
    uint16_t *clean = NULL;
    CRAWNoiseModel nm = { .a = (float)TEST_NOISE_A, .b = (float)TEST_NOISE_B };
    CRAWPipelineParams pp;
    CRAWFrameParams fp;
    double ms = 0.0, err, base, cross;
    int ret = -1;

    clean = malloc(npix * sizeof(*clean));
    if (!clean)
        return -1;

    /* Strength zero skips the pass, and so does a pipeline built without
     * the flag; both must reproduce the plain chain. The second of the two
     * leaves `ref` holding the undenoised reference and `base` the floor. */
    base = 0.0;
    for (int off = 0; off < 2; off++) {
        craw_pipeline_params_init(&pp);
        pp.width = TEST_W;
        pp.height = TEST_H;
        pp.debayer = CRAW_DEBAYER_MHC;
        pp.out_space = CRAW_CS_XYZ_D65;
        pp.flags = off ? 0 : CRAW_PIPELINE_FLAG_DENOISE;

        craw_frame_params_init(&fp);
        fp.profile = identity;
        fp.denoise_strength = off ? 1.0f : 0.0f;
        fp.noise_override = nm;

        if (craw_ref_debayer(mosaic, TEST_W, TEST_H, TEST_W, identity,
                             CRAW_DEBAYER_MHC, ref) < 0)
            goto done;
        if (test_run_config(td, ctx, src, dst, &pp, &fp, gpu, NULL, NULL, &ms) < 0)
            goto done;

        base = test_max_abs(gpu, ref, npix);
        printf("  denoise  %-14s: max abs %.3e\n",
               off ? "flag not set" : "strength 0", base);
        if (!(base <= TEST_DENOISE_TOL)) {
            fprintf(stderr, "the denoise pass ran when it should not have (%.3e)\n", base);
            goto done;
        }
    }

    for (size_t si = 0; si < CRAW_ARRAY_ELEMS(strengths); si++) {
        craw_pipeline_params_init(&pp);
        pp.width = TEST_W;
        pp.height = TEST_H;
        pp.debayer = CRAW_DEBAYER_MHC;
        pp.out_space = CRAW_CS_XYZ_D65;
        pp.flags = CRAW_PIPELINE_FLAG_DENOISE;

        craw_frame_params_init(&fp);
        fp.profile = identity;
        fp.denoise_strength = strengths[si];
        fp.noise_override = nm;

        if (test_run_config(td, ctx, src, dst, &pp, &fp, gpu, NULL, NULL, &ms) < 0)
            goto done;

        /* `ref` still holds the undenoised reference from the loop above. */
        cross = test_max_abs(gpu, ref, npix);

        if (craw_ref_process(mosaic, TEST_W, TEST_H, TEST_W, &pp, &fp, ref) < 0) {
            fprintf(stderr, "denoise: craw_ref_process failed\n");
            goto done;
        }
        err = test_max_abs(gpu, ref, npix);

        printf("  denoise  strength %.1f  : max abs %.3e (tol %.1e), %.3e against the "
               "undenoised chain (floor %.3e), %.3f ms\n",
               (double)strengths[si], err, TEST_DENOISE_TOL, cross, base, ms);
        if (!(err <= TEST_DENOISE_TOL)) {
            fprintf(stderr, "denoise at strength %.1f differs by %.3e, tolerance %.1e\n",
                    (double)strengths[si], err, TEST_DENOISE_TOL);
            goto done;
        }
        if (!(cross > 3.0 * base)) {
            fprintf(stderr, "a denoised frame is only %.3e from an undenoised one against "
                    "a floor of %.3e; the pass cannot have run\n", cross, base);
            goto done;
        }

        /* The reference's own denoised mosaic, so that the noise numbers
         * describe the definition rather than one of its implementations. */
        if (craw_ref_denoise(mosaic, TEST_W, TEST_H, TEST_W, identity, &nm,
                             strengths[si], clean, TEST_W) < 0) {
            fprintf(stderr, "craw_ref_denoise failed\n");
            goto done;
        }

        {
            /* The neutral flat patch of test_scene_rgb(), inset so that no
             * stencil reaches its edge. */
            double before = test_plane_noise(mosaic, 162, 26, 197, 53);
            double after = test_plane_noise(clean, 162, 26, 197, 53);

            printf("  denoise  strength %.1f  : flat patch noise %.2f -> %.2f counts "
                   "(%.0f%%, want <= 60%%)\n",
                   (double)strengths[si], before, after, 100.0 * after / before);
            if (!(after <= 0.60 * before)) {
                fprintf(stderr, "denoise at strength %.1f left %.0f%% of the noise\n",
                        (double)strengths[si], 100.0 * after / before);
                goto done;
            }
        }

        /* And it must not have flattened the frame: the 2x2 checkerboard
         * of test_scene_rgb() is real detail at the sampling limit, which
         * the patch distance is supposed to keep. */
        {
            double before = test_plane_noise(mosaic, 212, 100, 243, 131);
            double after = test_plane_noise(clean, 212, 100, 243, 131);

            printf("  denoise  strength %.1f  : checkerboard high pass %.0f -> %.0f "
                   "counts (%.0f%%, want >= 70%%)\n",
                   (double)strengths[si], before, after, 100.0 * after / before);
            if (!(after >= 0.70 * before)) {
                fprintf(stderr, "denoise at strength %.1f destroyed the checkerboard "
                        "(%.0f%% left)\n", (double)strengths[si], 100.0 * after / before);
                goto done;
            }
        }
    }

    /* Without a model the pass cannot know what to remove. */
    {
        CRAWPipeline *pipe = NULL;
        CRAWProfile bare = *identity;
        struct test_run run;
        int rec;

        bare.nb_noise_knots = 0;

        craw_pipeline_params_init(&pp);
        pp.width = TEST_W;
        pp.height = TEST_H;
        pp.flags = CRAW_PIPELINE_FLAG_DENOISE;

        craw_frame_params_init(&fp);
        fp.profile = &bare;
        fp.denoise_strength = 1.0f;

        if (craw_pipeline_create(ctx, &pp, &pipe) < 0)
            goto done;

        run = (struct test_run) { .pipe = pipe, .src = src, .dst = dst, .fp = &fp };
        rec = craw_testdev_run(td, test_record, &run);
        craw_pipeline_free(&pipe);
        if (rec != CRAW_ERROR(ENOTSUP)) {
            fprintf(stderr, "denoising without a noise model returned %d, wanted %d\n",
                    rec, CRAW_ERROR(ENOTSUP));
            goto done;
        }
    }

    ret = 0;

done:
    free(clean);
    return ret;
}

/* ----------------------------------------------------------- rejections */

/* The configurations craw_pipeline_create() must refuse, plus the histogram
 * accessor on a pipeline that was never asked for one. */
/* ----------------------------------------------------------------- lens */

/* DESIGN.md 16. Three things are checked, and the geometry one is checked
 * against arithmetic rather than against the other implementation:
 *
 *  - the GPU warp and craw_ref_lens() agree on the synthetic frame, and
 *    the warped frame is far enough from the unwarped one that the
 *    agreement means something;
 *  - a correction marked invalid, on a pipeline built with the pass, gives
 *    back exactly the bits a pipeline built without it does;
 *  - a bright mark put through a polynomial whose inverse is worked out by
 *    bisection lands where that inverse says, per channel, to a fraction of
 *    a pixel.
 */

/* The warp is a weighted sum of sixteen fp16 samples whose weights sum to
 * one but whose absolute values sum to as much as 1.25 per axis, so it
 * carries at most 1.56 times the error of the image it reads into its own
 * output. Against the 5e-3 DESIGN.md 11 allows the RCD intermediate that is
 * 7.8e-3; the bound below is that rounded up. The geometry itself is fp32
 * on both sides and agrees to a millionth of a pixel, so none of this is
 * positional error. */
#define TEST_LENS_TOL 8e-3

/* The analytic test's frame. Bigger than the shared one so that a 3 pixel
 * Gaussian near the corner has room on every side of it after being moved
 * outwards by a tenth of the radius. */
#define TEST_LENS_W 512
#define TEST_LENS_H 320

/* The mark is measured over a square this far either side of where it is
 * expected; at 14 pixels a sigma 3 Gaussian is down by e^-10.9. */
#define TEST_LENS_WIN 14

/* P1000387.MOV frame 0, tools/prr_lens.md 5: the LUMIX S 24-60 at 60 mm,
 * +81 px of pincushion corrected at the corner of a 5760x3040 frame. */
static const CRAWLensCorrection test_lens_real = {
    .k = {
        { 0.9732558749542047f,  1.3826185943196782e-02f,  1.5989100451823418e-02f, -5.451906036893745e-03f },
        { 0.9736971220864274f,  1.3698028399765133e-02f,  1.6482908551221100e-02f, -5.962137811706458e-03f },
        { 0.9743091685055746f,  1.3363908702330294e-02f,  1.7107910468175024e-02f, -6.633721811149271e-03f },
    },
    .centre = { 0.5f, 0.5f },
    .valid  = true,
};

/* A correction nobody's lens has: a tenth of the radius at the corner, two
 * hundred times what the real record does on a frame this small, and a
 * different coefficient per channel so that the three land visibly apart.
 * Purely quadratic, so its inverse is a cubic that bisects in a dozen
 * steps and the expected position is arithmetic, not a second
 * implementation of the warp. */
static const CRAWLensCorrection test_lens_synth = {
    .k = {
        { 1.0f, -0.090f, 0.0f, 0.0f },
        { 1.0f, -0.100f, 0.0f, 0.0f },
        { 1.0f, -0.110f, 0.0f, 0.0f },
    },
    .centre = { 0.5f, 0.5f },
    .valid  = true,
};

/* Where the mark sits in the source, as a fraction of the half diagonal. */
static const double test_lens_radii[] = { 0.35, 0.78 };

static double test_lens_poly(const float k[4], double r)
{
    double r2 = r * r;

    return k[0] + r2 * (k[1] + r2 * (k[2] + r2 * k[3]));
}

/* The destination radius whose source radius is `r_src`, i.e. the inverse
 * of r * poly(r). Monotone over the range any real record covers, so plain
 * bisection over [0, 2] converges to the last bit of a double in 60 steps. */
static double test_lens_invert(const float k[4], double r_src)
{
    double lo = 0.0, hi = 2.0;

    for (int i = 0; i < 80; i++) {
        double mid = 0.5 * (lo + hi);

        if (mid * test_lens_poly(k, mid) < r_src)
            lo = mid;
        else
            hi = mid;
    }

    return 0.5 * (lo + hi);
}

/* A black frame with a Gaussian mark at each of `nb` positions, written in
 * the prepped domain of DESIGN.md 4.1 and inverted to raw samples. No noise
 * term: this frame is measured, not compared. */
static void test_lens_gen_dots(uint16_t *m, uint32_t w, uint32_t h,
                               const double (*dots)[2], int nb, double sigma,
                               double peak, double base)
{
    double inv2s2 = 1.0 / (2.0 * sigma * sigma);

    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            int s = (int)((y & 1) * 2 + (x & 1));
            int c = (s == 0) ? 0 : (s == 3) ? 2 : 1;
            double v = base;
            double raw;
            int iv;

            for (int i = 0; i < nb; i++) {
                double dx = (double)x + 0.5 - dots[i][0];
                double dy = (double)y + 0.5 - dots[i][1];

                v += peak * exp(-(dx * dx + dy * dy) * inv2s2);
            }

            raw = TEST_BLACK + (v / (double)test_wb[c]) * (TEST_WHITE - TEST_BLACK);
            iv = (int)lrint(raw * 65535.0);
            m[(size_t)y * w + x] = (uint16_t)(iv < 0 ? 0 : (iv > 65535 ? 65535 : iv));
        }
    }
}

/* First moment of one channel over a window, in the same pixel centre
 * coordinates the warp uses. The Catmull-Rom undershoot around a bright
 * mark is symmetric, so the samples go in signed rather than clamped:
 * clamping them would be the thing that moved the answer. `bg` is the flat
 * field the mark sits on, subtracted because a first moment of a mark plus
 * a constant is a first moment of the window, not of the mark. */
static bool test_lens_centroid(const float *img, uint32_t w, uint32_t h, int c,
                               double cx, double cy, int win, double bg,
                               double out[2])
{
    int x0 = (int)floor(cx) - win, x1 = (int)floor(cx) + win;
    int y0 = (int)floor(cy) - win, y1 = (int)floor(cy) + win;
    double sum = 0.0, sx = 0.0, sy = 0.0;

    if (x0 < 0 || y0 < 0 || x1 >= (int)w || y1 >= (int)h)
        return false;

    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            double v = img[((size_t)y * w + x) * 4 + c] - bg;

            sum += v;
            sx += v * ((double)x + 0.5);
            sy += v * ((double)y + 0.5);
        }
    }

    if (!(sum > 1e-3))
        return false;

    out[0] = sx / sum;
    out[1] = sy / sum;

    return true;
}

/* The geometry check. `off` and `on` are the same frame developed without
 * and with the warp; for each mark and each channel the measured
 * displacement is compared against the inverse of that channel's
 * polynomial. Measuring the mark twice rather than trusting where it was
 * put means the debayer's own reconstruction of a Bayer sampled Gaussian
 * cancels out and what is left is the warp. */
static int test_lens_geometry(const char *what, const float *off, const float *on,
                              uint32_t w, uint32_t h, const CRAWLensCorrection *lc,
                              const double (*dots)[2], const int *wins, int nb,
                              double scale, double bg, double tol)
{
    double cx = lc->centre[0] * (double)w;
    double cy = lc->centre[1] * (double)h;
    double hx = (double)CRAW_MAX(lc->centre[0], 1.0f - lc->centre[0]) * (double)w;
    double hy = (double)CRAW_MAX(lc->centre[1], 1.0f - lc->centre[1]) * (double)h;
    double half_diag = sqrt(hx * hx + hy * hy);
    double worst = 0.0;
    int fails = 0;

    for (int i = 0; i < nb; i++) {
        for (int c = 0; c < 3; c++) {
            double src[2], got[2], want[2];
            double r_src, r_dst, factor, err;

            if (!test_lens_centroid(off, w, h, c, dots[i][0], dots[i][1], wins[i],
                                    bg, src)) {
                fprintf(stderr, "%s: mark %d channel %c was not found in the "
                        "uncorrected frame\n", what, i, "RGB"[c]);
                return -1;
            }

            /* Where the warp has to put what sits at `src`. Filling the
             * frame is `scale` 1; keeping everything shrinks the whole
             * picture by the mode's own factor on top of the inverse. */
            r_src = hypot(src[0] - cx, src[1] - cy) / half_diag;
            r_dst = test_lens_invert(lc->k[c], r_src) * scale;
            factor = r_dst / r_src;
            want[0] = cx + (src[0] - cx) * factor;
            want[1] = cy + (src[1] - cy) * factor;

            if (!test_lens_centroid(on, w, h, c, want[0], want[1], wins[i], bg, got)) {
                fprintf(stderr, "%s: mark %d channel %c was not found near "
                        "(%.2f, %.2f) in the corrected frame\n", what, i,
                        "RGB"[c], want[0], want[1]);
                return -1;
            }

            err = hypot(got[0] - want[0], got[1] - want[1]);
            worst = CRAW_MAX(worst, err);

            printf("    %s mark %d %c: r_src %.4f -> r_dst %.4f, moved %.2f px, "
                   "landed %.3f px from where the polynomial says\n", what, i,
                   "RGB"[c], r_src, r_dst,
                   hypot(want[0] - src[0], want[1] - src[1]), err);

            if (!(err <= tol)) {
                fprintf(stderr, "%s: mark %d channel %c landed at (%.3f, %.3f), "
                        "the polynomial says (%.3f, %.3f), %.3f px off, "
                        "tolerance %.2f\n", what, i, "RGB"[c], got[0], got[1],
                        want[0], want[1], err, tol);
                fails++;
            }
        }
    }

    printf("  %s: worst placement error %.3f px (tol %.2f px)\n", what, worst, tol);

    return fails ? -1 : 0;
}

static int test_lens_analytic(CRAWTestDevice *td, CRAWContext *ctx,
                              const CRAWProfile *identity)
{
    const size_t npix = (size_t)TEST_LENS_W * TEST_LENS_H;
    const int nb_dots = (int)(sizeof(test_lens_radii) / sizeof(test_lens_radii[0]));
    /* Along the diagonal towards the bottom right corner, which is where a
     * radial correction has the most to do. */
    const double half_diag = sqrt((double)(TEST_LENS_W * TEST_LENS_W +
                                           TEST_LENS_H * TEST_LENS_H)) * 0.5;
    const double ux = (TEST_LENS_W * 0.5) / half_diag;
    const double uy = (TEST_LENS_H * 0.5) / half_diag;
    double dots[8][2];
    int wins[8];
    CRAWTestImage src = { 0 }, dst = { 0 };
    bool have_src = false, have_dst = false;
    CRAWPipelineParams pp;
    CRAWFrameParams fp;
    uint16_t *mosaic = NULL;
    float *off = NULL, *on = NULL;
    double ms = 0.0;
    int ret;

    for (int i = 0; i < nb_dots; i++) {
        dots[i][0] = TEST_LENS_W * 0.5 + ux * test_lens_radii[i] * half_diag;
        dots[i][1] = TEST_LENS_H * 0.5 + uy * test_lens_radii[i] * half_diag;
        wins[i] = TEST_LENS_WIN;
    }

    mosaic = malloc(npix * sizeof(*mosaic));
    off = malloc(npix * 4 * sizeof(*off));
    on = malloc(npix * 4 * sizeof(*on));
    if (!mosaic || !off || !on) {
        ret = CRAW_ERROR(ENOMEM);
        goto done;
    }

    test_lens_gen_dots(mosaic, TEST_LENS_W, TEST_LENS_H, dots, nb_dots, 3.0, 0.40, 0.0);

    ret = craw_testdev_image_create(td, &src, TEST_LENS_W, TEST_LENS_H, VK_FORMAT_R16_UNORM);
    if (ret < 0)
        goto done;
    have_src = true;
    ret = craw_testdev_image_create(td, &dst, TEST_LENS_W, TEST_LENS_H,
                                    VK_FORMAT_R16G16B16A16_SFLOAT);
    if (ret < 0)
        goto done;
    have_dst = true;
    ret = craw_testdev_image_upload(td, &src, mosaic, npix * sizeof(*mosaic));
    if (ret < 0)
        goto done;

    craw_pipeline_params_init(&pp);
    pp.width = TEST_LENS_W;
    pp.height = TEST_LENS_H;
    pp.debayer = CRAW_DEBAYER_RCD;
    pp.out_space = CRAW_CS_XYZ_D65;
    pp.flags = CRAW_PIPELINE_FLAG_LENS;

    craw_frame_params_init(&fp);
    fp.profile = identity;

    /* Uncorrected: the pass is built but this frame has no record. */
    ret = test_run_config(td, ctx, &src, &dst, &pp, &fp, off, NULL, NULL, &ms);
    if (ret < 0)
        goto done;

    fp.lens = test_lens_synth;
    ret = test_run_config(td, ctx, &src, &dst, &pp, &fp, on, NULL, NULL, &ms);
    if (ret < 0)
        goto done;

    ret = test_lens_geometry("gpu", off, on, TEST_LENS_W, TEST_LENS_H,
                             &test_lens_synth, dots, wins, nb_dots, 1.0, 0.0, 0.25);
    if (ret < 0)
        goto done;

    /* The same two frames from the CPU reference, so that the definition is
     * held to the arithmetic too and not only the shader. */
    craw_frame_params_init(&fp);
    fp.profile = identity;
    ret = craw_ref_process(mosaic, TEST_LENS_W, TEST_LENS_H, TEST_LENS_W, &pp, &fp, off);
    if (ret < 0)
        goto done;
    fp.lens = test_lens_synth;
    ret = craw_ref_process(mosaic, TEST_LENS_W, TEST_LENS_H, TEST_LENS_W, &pp, &fp, on);
    if (ret < 0)
        goto done;

    ret = test_lens_geometry("ref", off, on, TEST_LENS_W, TEST_LENS_H,
                             &test_lens_synth, dots, wins, nb_dots, 1.0, 0.0, 0.25);

done:
    if (ret < 0)
        fprintf(stderr, "the analytic lens check failed: %d\n", ret);
    if (have_dst)
        craw_testdev_image_free(td, &dst);
    if (have_src)
        craw_testdev_image_free(td, &src);
    free(on);
    free(off);
    free(mosaic);

    return ret;
}


/* ------------------------------------- keeping everything (DESIGN.md 16.8) */

/* Where the marks sit for the keep-everything test, as a fraction of the
 * half diagonal: one near the centre as a control, one as far out as a 14
 * pixel measuring window still fits, and one at 0.97, which fill mode
 * throws away outright - the output radius that would have to show it is
 * past the frame corner - and which keeping everything has to bring back. */
static const double test_keep_radii[] = { 0.35, 0.86, 0.95 };
static const int test_keep_wins[] = { TEST_LENS_WIN, TEST_LENS_WIN, 4 };

/* The first two are measured by centroid the way the fill-mode geometry
 * check measures its own. The last one sits too close to the frame edge
 * for a centroid window to fit around it in the source, so it is checked
 * by where its peak lands instead - which is all that mark is for: fill
 * mode does not have it at all. */
#define TEST_KEEP_MEASURED 2

/* A flat field under the marks, so that "exactly black" means the border
 * the mode paints and nothing else: on a frame that was black to begin with
 * the check would pass without the mode doing anything. */
#define TEST_KEEP_BASE 0.25

/* Is the source position of the output pixel (x, y) outside the source
 * frame in any channel, and by how far? In double, the same geometry the
 * shader and the reference do in float, from the record and the scale
 * rather than from either of them. `margin` comes back positive outside
 * the frame and negative inside, in pixels, so that the pixels the two
 * arithmetics can legitimately disagree about are the ones near zero. */
static bool test_keep_outside(const CRAWLensCorrection *lc, double w, double h,
                              double s, int x, int y, double *margin)
{
    double cx = lc->centre[0] * w, cy = lc->centre[1] * h;
    double hx = (double)CRAW_MAX(lc->centre[0], 1.0f - lc->centre[0]) * w;
    double hy = (double)CRAW_MAX(lc->centre[1], 1.0f - lc->centre[1]) * h;
    double half_diag = hypot(hx, hy);
    double dx = ((double)x + 0.5 - cx) / s;
    double dy = ((double)y + 0.5 - cy) / s;
    double r2 = (dx * dx + dy * dy) / (half_diag * half_diag);
    double worst = -HUGE_VAL;

    for (int c = 0; c < 3; c++) {
        double poly = test_lens_poly(lc->k[c], sqrt(r2));
        double sx = cx + dx * poly, sy = cy + dy * poly;
        double over = CRAW_MAX(CRAW_MAX(-sx, sx - w), CRAW_MAX(-sy, sy - h));

        worst = CRAW_MAX(worst, over);
    }

    *margin = worst;

    return worst > 0.0;
}

/* The mode end to end on the synthetic frame: the marks land where the
 * scaled inverse says, everything outside the mapped source is exactly
 * black, the GPU agrees with the reference, and filling the frame is
 * untouched by any of it. */
static int test_lens_keep(CRAWTestDevice *td, CRAWContext *ctx,
                          const CRAWProfile *identity)
{
    const size_t npix = (size_t)TEST_LENS_W * TEST_LENS_H;
    const int nb_dots = (int)CRAW_ARRAY_ELEMS(test_keep_radii);
    const double half_diag = sqrt((double)(TEST_LENS_W * TEST_LENS_W +
                                           TEST_LENS_H * TEST_LENS_H)) * 0.5;
    const double ux = (TEST_LENS_W * 0.5) / half_diag;
    const double uy = (TEST_LENS_H * 0.5) / half_diag;
    double dots[8][2], want_corner[2], r_fill, r_keep;
    int wins[8];
    CRAWTestImage src = { 0 }, dst = { 0 };
    bool have_src = false, have_dst = false;
    CRAWPipelineParams pp;
    CRAWFrameParams fp;
    uint16_t *mosaic = NULL;
    float *off = NULL, *fill = NULL, *fill2 = NULL, *keep = NULL, *ref = NULL;
    size_t black = 0, guard = 0, lit = 0;
    double ms = 0.0, err;
    float scale = 1.0f;
    int ret;

    ret = craw_lens_keep_all_scale(&test_lens_synth, TEST_LENS_W, TEST_LENS_H, &scale);
    if (ret < 0) {
        fprintf(stderr, "craw_lens_keep_all_scale failed: %d\n", ret);
        return ret;
    }

    for (int i = 0; i < nb_dots; i++) {
        dots[i][0] = TEST_LENS_W * 0.5 + ux * test_keep_radii[i] * half_diag;
        dots[i][1] = TEST_LENS_H * 0.5 + uy * test_keep_radii[i] * half_diag;
        wins[i] = test_keep_wins[i];
    }

    /* What the outermost mark does in each mode. Fill mode would have to
     * show it past the frame corner, which is what "the corner is thrown
     * away" means; keeping everything brings it to `want_corner`, and the
     * test insists that is well inside the frame rather than a pixel from
     * the edge. */
    r_fill = test_lens_invert(test_lens_synth.k[1], test_keep_radii[nb_dots - 1]);
    r_keep = r_fill * (double)scale;
    want_corner[0] = TEST_LENS_W * 0.5 + ux * r_keep * half_diag;
    want_corner[1] = TEST_LENS_H * 0.5 + uy * r_keep * half_diag;

    printf("  keep all: scale %.6f; the mark at r %.2f wants output radius %.4f, "
           "which fill has no room for, and lands at (%.1f, %.1f)\n",
           (double)scale, test_keep_radii[nb_dots - 1], r_fill,
           want_corner[0], want_corner[1]);

    if (!(r_fill > 1.0)) {
        fprintf(stderr, "the outermost mark is not actually lost by fill mode "
                        "(output radius %.4f); the test proves nothing\n", r_fill);
        return -1;
    }
    if (!(want_corner[0] < TEST_LENS_W - 8.0 && want_corner[1] < TEST_LENS_H - 8.0)) {
        fprintf(stderr, "the outermost mark lands at (%.1f, %.1f), not well inside "
                        "a %dx%d frame\n", want_corner[0], want_corner[1],
                        TEST_LENS_W, TEST_LENS_H);
        return -1;
    }

    mosaic = malloc(npix * sizeof(*mosaic));
    off = malloc(npix * 4 * sizeof(*off));
    fill = malloc(npix * 4 * sizeof(*fill));
    fill2 = malloc(npix * 4 * sizeof(*fill2));
    keep = malloc(npix * 4 * sizeof(*keep));
    ref = malloc(npix * 4 * sizeof(*ref));
    if (!mosaic || !off || !fill || !fill2 || !keep || !ref) {
        ret = CRAW_ERROR(ENOMEM);
        goto done;
    }

    test_lens_gen_dots(mosaic, TEST_LENS_W, TEST_LENS_H, dots, nb_dots, 3.0, 0.40,
                       TEST_KEEP_BASE);

    ret = craw_testdev_image_create(td, &src, TEST_LENS_W, TEST_LENS_H, VK_FORMAT_R16_UNORM);
    if (ret < 0)
        goto done;
    have_src = true;
    ret = craw_testdev_image_create(td, &dst, TEST_LENS_W, TEST_LENS_H,
                                    VK_FORMAT_R16G16B16A16_SFLOAT);
    if (ret < 0)
        goto done;
    have_dst = true;
    ret = craw_testdev_image_upload(td, &src, mosaic, npix * sizeof(*mosaic));
    if (ret < 0)
        goto done;

    craw_pipeline_params_init(&pp);
    pp.width = TEST_LENS_W;
    pp.height = TEST_LENS_H;
    pp.debayer = CRAW_DEBAYER_RCD;
    pp.out_space = CRAW_CS_XYZ_D65;
    pp.flags = CRAW_PIPELINE_FLAG_LENS;

    /* Uncorrected, so that each mark's source position is measured rather
     * than assumed, exactly as the fill-mode geometry check does it. */
    craw_frame_params_init(&fp);
    fp.profile = identity;
    ret = test_run_config(td, ctx, &src, &dst, &pp, &fp, off, NULL, NULL, &ms);
    if (ret < 0)
        goto done;

    fp.lens = test_lens_synth;
    ret = test_run_config(td, ctx, &src, &dst, &pp, &fp, fill, NULL, NULL, &ms);
    if (ret < 0)
        goto done;

    fp.flags = CRAW_FRAME_FLAG_LENS_KEEP_ALL;
    ret = test_run_config(td, ctx, &src, &dst, &pp, &fp, keep, NULL, NULL, &ms);
    if (ret < 0)
        goto done;

    /* Fill mode again, after the mode has run: the flag is the only thing
     * that may change the picture, so these two have to be equal bit for
     * bit. Anything the keep path caches or leaves behind shows up here. */
    fp.flags = 0;
    ret = test_run_config(td, ctx, &src, &dst, &pp, &fp, fill2, NULL, NULL, &ms);
    if (ret < 0)
        goto done;

    if (memcmp(fill, fill2, npix * 4 * sizeof(*fill))) {
        fprintf(stderr, "filling the frame changed after the keep-everything mode "
                        "ran; the flag has to be the only difference\n");
        ret = -1;
        goto done;
    }

    /* The geometry: each mark measured in the uncorrected frame, and where
     * the scaled inverse of its own channel's polynomial says it has to be
     * in the corrected one. */
    ret = test_lens_geometry("keep", off, keep, TEST_LENS_W, TEST_LENS_H,
                             &test_lens_synth, dots, wins, TEST_KEEP_MEASURED,
                             (double)scale, TEST_KEEP_BASE, 0.25);
    if (ret < 0)
        goto done;

    /* The corner mark. Fill mode would need an output radius past the frame
     * corner to show it at all - that is checked above, from the record's
     * own polynomial - so the only question left is whether keeping
     * everything brings it back, and its peak has to be within a pixel of
     * where the scaled inverse puts it. The same window of the filled frame
     * is printed beside it, and what shows up there is not this mark: it is
     * the *inner* mark, which fill mode has pushed out to about here. */
    {
        double peak_keep = -HUGE_VAL, peak_fill = -HUGE_VAL, at[2] = { 0, 0 };
        int w0 = (int)floor(want_corner[0]) - wins[nb_dots - 1];
        int w1 = (int)floor(want_corner[0]) + wins[nb_dots - 1];
        int h0 = (int)floor(want_corner[1]) - wins[nb_dots - 1];
        int h1 = (int)floor(want_corner[1]) + wins[nb_dots - 1];

        for (int y = h0; y <= h1; y++) {
            for (int x = w0; x <= w1; x++) {
                double vk = keep[((size_t)y * TEST_LENS_W + x) * 4 + 1];
                double vf = fill[((size_t)y * TEST_LENS_W + x) * 4 + 1];

                if (vk > peak_keep) {
                    peak_keep = vk;
                    at[0] = (double)x + 0.5;
                    at[1] = (double)y + 0.5;
                }
                peak_fill = CRAW_MAX(peak_fill, vf);
            }
        }

        printf("    keep mark %d G: peak %.3f at (%.1f, %.1f), %.2f px from the "
               "prediction (the filled frame's own %.3f there is the inner mark)\n",
               nb_dots - 1, peak_keep, at[0], at[1],
               hypot(at[0] - want_corner[0], at[1] - want_corner[1]), peak_fill);

        if (!(peak_keep > TEST_KEEP_BASE + 0.15)) {
            fprintf(stderr, "the corner mark is not in the kept frame: the window "
                            "peaks at %.3f over a flat %.2f\n", peak_keep, TEST_KEEP_BASE);
            ret = -1;
            goto done;
        }
        if (!(hypot(at[0] - want_corner[0], at[1] - want_corner[1]) <= 1.0)) {
            fprintf(stderr, "the corner mark peaks at (%.1f, %.1f), the scaled "
                            "inverse says (%.1f, %.1f)\n", at[0], at[1],
                            want_corner[0], want_corner[1]);
            ret = -1;
            goto done;
        }
    }

    /* The border. Every pixel whose source lies outside the source frame in
     * any channel has to be exactly black with alpha exactly 1, and every
     * pixel whose source lies inside has to have kept the flat field it was
     * given. The handful of pixels within a hundredth of a pixel of the
     * boundary are where double and float can legitimately disagree and are
     * only counted. */
    for (uint32_t y = 0; y < TEST_LENS_H; y++) {
        for (uint32_t x = 0; x < TEST_LENS_W; x++) {
            const float *px = &keep[((size_t)y * TEST_LENS_W + x) * 4];
            double margin;
            bool outside = test_keep_outside(&test_lens_synth, TEST_LENS_W,
                                             TEST_LENS_H, (double)scale,
                                             (int)x, (int)y, &margin);

            if (fabs(margin) < 0.01) {
                guard++;
                continue;
            }

            if (outside) {
                black++;
                if (px[0] != 0.0f || px[1] != 0.0f || px[2] != 0.0f) {
                    fprintf(stderr, "pixel (%u, %u) is %.4f px outside the source "
                            "and came out (%g, %g, %g), not black\n", x, y, margin,
                            (double)px[0], (double)px[1], (double)px[2]);
                    ret = -1;
                    goto done;
                }
                if (px[3] != 1.0f) {
                    fprintf(stderr, "pixel (%u, %u) is black with alpha %g; the "
                            "border is black, not transparent\n", x, y,
                            (double)px[3]);
                    ret = -1;
                    goto done;
                }
            } else {
                lit++;
                if (!(px[1] > 0.5 * TEST_KEEP_BASE)) {
                    fprintf(stderr, "pixel (%u, %u) is %.4f px inside the source "
                            "and came out at %g, well under the %.2f flat field\n",
                            x, y, -margin, (double)px[1], TEST_KEEP_BASE);
                    ret = -1;
                    goto done;
                }
            }
        }
    }

    printf("  keep all: %zu pixels black outside the mapped source (%.1f%% of the "
           "frame), %zu inside, %zu within a hundredth of a pixel of the boundary\n",
           black, 100.0 * (double)black / (double)npix, lit, guard);

    if (!black || !lit) {
        fprintf(stderr, "the border check saw %zu black and %zu lit pixels; one of "
                        "them is not a picture\n", black, lit);
        ret = -1;
        goto done;
    }

    /* And the CPU reference, which runs the same mode from the same
     * craw_lens_keep_all_scale(). */
    ret = craw_ref_process(mosaic, TEST_LENS_W, TEST_LENS_H, TEST_LENS_W, &pp, &fp, ref);
    if (ret < 0)
        goto done;
    fp.flags = CRAW_FRAME_FLAG_LENS_KEEP_ALL;
    ret = craw_ref_process(mosaic, TEST_LENS_W, TEST_LENS_H, TEST_LENS_W, &pp, &fp, ref);
    if (ret < 0)
        goto done;

    err = test_max_abs(keep, ref, npix);
    printf("  keep all: max abs %.3e against the reference (tol %.1e)\n",
           err, TEST_LENS_TOL);
    if (!(err <= TEST_LENS_TOL)) {
        fprintf(stderr, "the GPU keep-everything warp differs from the reference by "
                        "%.3e, tolerance %.1e\n", err, TEST_LENS_TOL);
        ret = -1;
        goto done;
    }

    /* The border has to be in the same place in both, to the pixel: a
     * tolerance on the values would hide a border drawn half a pixel out. */
    for (size_t i = 0; i < npix; i++) {
        bool gb = keep[i * 4] == 0.0f && keep[i * 4 + 1] == 0.0f && keep[i * 4 + 2] == 0.0f;
        bool rb = ref[i * 4] == 0.0f && ref[i * 4 + 1] == 0.0f && ref[i * 4 + 2] == 0.0f;
        double margin;

        if (gb == rb)
            continue;
        test_keep_outside(&test_lens_synth, TEST_LENS_W, TEST_LENS_H, (double)scale,
                          (int)(i % TEST_LENS_W), (int)(i / TEST_LENS_W), &margin);
        if (fabs(margin) < 0.01)
            continue;

        fprintf(stderr, "pixel %zu is %s on the GPU and %s in the reference, %.4f px "
                "from the boundary\n", i, gb ? "black" : "lit", rb ? "black" : "lit",
                margin);
        ret = -1;
        goto done;
    }

done:
    if (ret < 0)
        fprintf(stderr, "the keep-everything check failed: %d\n", ret);
    if (have_dst)
        craw_testdev_image_free(td, &dst);
    if (have_src)
        craw_testdev_image_free(td, &src);
    free(ref);
    free(keep);
    free(fill2);
    free(fill);
    free(off);
    free(mosaic);

    return ret;
}

static int test_lens(CRAWTestDevice *td, CRAWContext *ctx, const CRAWProfile *identity,
                     const uint16_t *mosaic, CRAWTestImage *src, CRAWTestImage *dst,
                     float *gpu, float *ref)
{
    const size_t npix = (size_t)TEST_W * TEST_H;
    CRAWPipelineParams pp;
    CRAWFrameParams fp;
    float *plain = NULL;
    double ms_off = 0.0, ms_on = 0.0, ms_skip = 0.0, err, cross;
    int ret;

    plain = malloc(npix * 4 * sizeof(*plain));
    if (!plain)
        return CRAW_ERROR(ENOMEM);

    /* 1. The chain with no lens pass at all, kept as the thing every "off"
     *    case has to reproduce. */
    craw_pipeline_params_init(&pp);
    pp.width = TEST_W;
    pp.height = TEST_H;
    pp.debayer = CRAW_DEBAYER_RCD;
    pp.out_space = CRAW_CS_XYZ_D65;

    craw_frame_params_init(&fp);
    fp.profile = identity;

    ret = test_run_config(td, ctx, src, dst, &pp, &fp, plain, NULL, NULL, &ms_off);
    if (ret < 0)
        goto done;

    /* 2. The pass built, this frame without a record: the warp drops out
     *    and the colour stage writes the caller's image as it always did,
     *    so the two frames have to be equal bit for bit. */
    pp.flags = CRAW_PIPELINE_FLAG_LENS;
    ret = test_run_config(td, ctx, src, dst, &pp, &fp, gpu, NULL, NULL, &ms_skip);
    if (ret < 0)
        goto done;

    if (memcmp(gpu, plain, npix * 4 * sizeof(*gpu))) {
        size_t bad = 0;

        for (size_t i = 0; i < npix * 4; i++)
            if (gpu[i] != plain[i])
                bad++;
        fprintf(stderr, "an invalid correction changed %zu of %zu samples; it has to "
                "leave the frame exactly alone\n", bad, npix * 4);
        ret = -1;
        goto done;
    }

    ret = craw_ref_process(mosaic, TEST_W, TEST_H, TEST_W, &pp, &fp, ref);
    if (ret < 0)
        goto done;
    err = test_max_abs(gpu, ref, npix);
    printf("  lens off (no record): identical to the plain chain, %.3e against the "
           "reference, %.3f ms against %.3f without the pass\n", err, ms_skip, ms_off);
    if (!(err <= TEST_LENS_TOL)) {
        fprintf(stderr, "the skipped warp differs from the reference by %.3e\n", err);
        ret = -1;
        goto done;
    }

    /* 3. The warp running, against the reference, and far enough from the
     *    unwarped frame that the agreement is worth something. */
    for (int real = 0; real < 2; real++) {
        const CRAWLensCorrection *lc = real ? &test_lens_real : &test_lens_synth;

        craw_frame_params_init(&fp);
        fp.profile = identity;
        fp.lens = *lc;

        ret = test_run_config(td, ctx, src, dst, &pp, &fp, gpu, NULL, NULL, &ms_on);
        if (ret < 0)
            goto done;

        cross = test_max_abs(gpu, plain, npix);

        ret = craw_ref_process(mosaic, TEST_W, TEST_H, TEST_W, &pp, &fp, ref);
        if (ret < 0)
            goto done;

        err = test_max_abs(gpu, ref, npix);
        printf("  lens on (%s record): max abs %.3e (tol %.1e), %.3e against the "
               "uncorrected frame, %.3f ms\n", real ? "real" : "synthetic", err,
               TEST_LENS_TOL, cross, ms_on);

        if (!(err <= TEST_LENS_TOL)) {
            fprintf(stderr, "the GPU warp differs from the reference by %.3e, "
                    "tolerance %.1e\n", err, TEST_LENS_TOL);
            ret = -1;
            goto done;
        }
        if (!(cross > 5.0 * err)) {
            fprintf(stderr, "the corrected frame is only %.3e from the uncorrected "
                    "one; the warp cannot have run\n", cross);
            ret = -1;
            goto done;
        }
    }

    ret = test_lens_analytic(td, ctx, identity);
    if (ret < 0)
        goto done;

    ret = test_lens_keep(td, ctx, identity);

done:
    free(plain);

    return ret;
}

/* ----------------------------------------------------- gamut compression */

/* DESIGN.md 8.1. First the ACES 1.3 curve on its own, without a GPU: its
 * shape, what it leaves alone and which way it moves what it does not.
 * Then the whole chain through four output spaces on the device, with the
 * negative components counted in each. */

/* An RGB triple of `src` carried into AP1 linear through XYZ D65. */
static int test_to_ap1(enum CRAWColorSpace src, const float in[3], float out[3])
{
    float to[3][3], from[3][3], xyz[3];
    int ret;

    ret = craw_colorspace_to_xyz_d65(src, to);
    if (ret < 0)
        return ret;

    ret = craw_colorspace_from_xyz_d65(CRAW_CS_ACES_AP1_LINEAR, from);
    if (ret < 0)
        return ret;

    for (int r = 0; r < 3; r++)
        xyz[r] = to[r][0] * in[0] + to[r][1] * in[1] + to[r][2] * in[2];
    for (int r = 0; r < 3; r++)
        out[r] = from[r][0] * xyz[0] + from[r][1] * xyz[1] + from[r][2] * xyz[2];

    return 0;
}

/* One channel's compressed distance, read back off the operator itself: a
 * triple whose other two channels sit at the achromatic value 1.0 and whose
 * channel `c` sits a distance `d` below it exercises exactly one curve. */
static double test_gamut_curve_at(const CRAWGamutCompress *g, int c, double d)
{
    float rgb[3] = { 1.0f, 1.0f, 1.0f };

    rgb[c] = (float)(1.0 - d);
    craw_gamut_compress(g, rgb);

    return 1.0 - (double)rgb[c];
}

/* The published ACES 1.3 parameters, repeated here so that the test is a
 * check against the specification and not against lib/gamut.c's arithmetic
 * spelled a second way. Order is the operator's: cyan is the red channel's
 * curve, magenta the green one's, yellow the blue one's. */
static const double test_rgc_lim[3] = { 1.147, 1.264, 1.312 };
static const double test_rgc_thr[3] = { 0.815, 0.803, 0.880 };

static int test_gamut_curve(void)
{
    static const char *chan[3] = { "cyan (R)", "magenta (G)", "yellow (B)" };
    CRAWGamutCompress g;
    int bad = 0;

    craw_gamut_compress_params(&g);

    if (!(fabs((double)g.pwr - 1.2) < 1e-6)) {
        fprintf(stderr, "gamut: power is %.9g, the specification says 1.2\n", (double)g.pwr);
        bad++;
    }

    for (int c = 0; c < 3; c++) {
        const double thr = test_rgc_thr[c], lim = test_rgc_lim[c];
        const double asym = thr + (double)g.scl[c];
        double worst_step = 0.0, worst_below = 0.0;
        double at_lim, prev, jump;
        int steps = 40000;

        if (!(fabs((double)g.thr[c] - thr) < 1e-6)) {
            fprintf(stderr, "gamut: %s threshold is %.9g, want %.9g\n",
                    chan[c], (double)g.thr[c], thr);
            bad++;
        }

        /* Nothing below the threshold moves at all: not "within 1e-5",
         * bit for bit, because the operator does not touch it. Compared on
         * the triple itself rather than on the distance, because turning a
         * float back into a distance and out again is what loses the last
         * bits, not the operator. The divisor is 201 so that the last
         * sample is still strictly below the threshold. */
        for (int i = 0; i <= 200; i++) {
            float rgb[3] = { 1.0f, 1.0f, 1.0f }, was[3];

            rgb[c] = (float)(1.0 - thr * (double)i / 201.0);
            memcpy(was, rgb, sizeof(was));
            craw_gamut_compress(&g, rgb);

            for (int k = 0; k < 3; k++)
                worst_below = fmax(worst_below, fabs((double)rgb[k] - was[k]));
        }
        if (!(worst_below == 0.0)) {
            fprintf(stderr, "gamut: %s moves a distance below the threshold by %.3e\n",
                    chan[c], worst_below);
            bad++;
        }

        /* A distance of exactly the limit lands on 1.0, which puts the
         * channel at exactly zero: that is what "distance limit" means and
         * it is the whole reason anything further out stops being
         * negative. */
        at_lim = test_gamut_curve_at(&g, c, lim);
        if (!(fabs(at_lim - 1.0) < 1e-5)) {
            fprintf(stderr, "gamut: %s at the limit gives %.9g, want 1.0\n", chan[c], at_lim);
            bad++;
        }

        /* Monotone, and never expanding: a step in the distance can only
         * produce a step no larger in the compressed distance. Sweeping
         * straight across the threshold is what makes this a continuity
         * check — a break there would show up as one oversized step. */
        prev = test_gamut_curve_at(&g, c, 0.0);
        for (int i = 1; i <= steps; i++) {
            double d = 2.0 * (double)i / (double)steps;
            double f = test_gamut_curve_at(&g, c, d);
            double h = 2.0 / (double)steps;

            if (!(f >= prev - 1e-9)) {
                fprintf(stderr, "gamut: %s is not monotone at d = %.6f\n", chan[c], d);
                bad++;
                break;
            }
            worst_step = fmax(worst_step, (f - prev) / h);
            prev = f;
        }
        if (!(worst_step <= 1.0 + 1e-3)) {
            fprintf(stderr, "gamut: %s expands by %.6f at its steepest\n", chan[c], worst_step);
            bad++;
        }

        /* And the same across the threshold at a step far finer than the
         * sweep, which is where a discontinuity would hide. */
        jump = fabs(test_gamut_curve_at(&g, c, thr + 1e-6) -
                    test_gamut_curve_at(&g, c, thr - 1e-6));
        if (!(jump < 1e-5)) {
            fprintf(stderr, "gamut: %s jumps by %.3e across the threshold\n", chan[c], jump);
            bad++;
        }

        /* However far out the colour is, the compressed distance cannot
         * pass thr + scl, so the channel cannot fall below -(scl + thr - 1)
         * of the achromatic value. Beyond the limit it is negative again,
         * which is the specification's own trade and worth stating. */
        if (!(test_gamut_curve_at(&g, c, 1e6) <= asym + 1e-6)) {
            fprintf(stderr, "gamut: %s passes its asymptote\n", chan[c]);
            bad++;
        }

        printf("  %-12s thr %.3f lim %.3f scl %.6f: limit -> %.6f, asymptote %.6f,"
               " steepest %.6f, jump %.1e\n",
               chan[c], thr, lim, (double)g.scl[c], at_lim, asym, worst_step, jump);
    }

    return bad ? CRAW_ERROR(EINVAL) : 0;
}

/* Is this AP1 colour one the operator promises not to touch? Every channel
 * has to sit closer to the achromatic value than its own threshold. */
static bool test_gamut_protected(const CRAWGamutCompress *g, const float rgb[3])
{
    float ach = CRAW_MAX(rgb[0], CRAW_MAX(rgb[1], rgb[2]));

    if (ach == 0.0f)
        return true;

    for (int c = 0; c < 3; c++) {
        if ((ach - rgb[c]) / fabsf(ach) >= g->thr[c])
            return false;
    }

    return true;
}

struct test_gamut_colour {
    const char *name;
    enum CRAWColorSpace space;
    float rgb[3];
};

static int test_gamut_invariance(void)
{
    static const struct test_gamut_colour colours[] = {
        { "neutral",      CRAW_CS_BT709_LINEAR,  { 1.0f, 1.0f, 1.0f } },
        { "709 red",      CRAW_CS_BT709_LINEAR,  { 1.0f, 0.0f, 0.0f } },
        { "709 green",    CRAW_CS_BT709_LINEAR,  { 0.0f, 1.0f, 0.0f } },
        { "709 blue",     CRAW_CS_BT709_LINEAR,  { 0.0f, 0.0f, 1.0f } },
        { "709 cyan",     CRAW_CS_BT709_LINEAR,  { 0.0f, 1.0f, 1.0f } },
        { "709 magenta",  CRAW_CS_BT709_LINEAR,  { 1.0f, 0.0f, 1.0f } },
        { "709 yellow",   CRAW_CS_BT709_LINEAR,  { 1.0f, 1.0f, 0.0f } },
        { "2020 red",     CRAW_CS_BT2020_LINEAR, { 1.0f, 0.0f, 0.0f } },
        { "2020 green",   CRAW_CS_BT2020_LINEAR, { 0.0f, 1.0f, 0.0f } },
        { "2020 blue",    CRAW_CS_BT2020_LINEAR, { 0.0f, 0.0f, 1.0f } },
        { "2020 cyan",    CRAW_CS_BT2020_LINEAR, { 0.0f, 1.0f, 1.0f } },
        { "2020 magenta", CRAW_CS_BT2020_LINEAR, { 1.0f, 0.0f, 1.0f } },
        { "2020 yellow",  CRAW_CS_BT2020_LINEAR, { 1.0f, 1.0f, 0.0f } },
    };
    /* Each colour at four levels, and each again pulled 40% of the way to
     * neutral: a real surface under a real light is nowhere near a display
     * primary, and the desaturated set is the one the specification's
     * thresholds were fitted to protect. */
    static const float levels[4] = { 0.05f, 0.25f, 0.60f, 1.00f };
    CRAWGamutCompress g;
    int bad = 0, nprot = 0, nmoved = 0;

    craw_gamut_compress_params(&g);

    for (size_t i = 0; i < CRAW_ARRAY_ELEMS(colours); i++) {
        for (int desat = 0; desat < 2; desat++) {
            double worst_move = 0.0, worst_scale = 0.0;
            bool protect = true, outward = false, ach_moved = false;
            float base[3];
            int ret;

            ret = test_to_ap1(colours[i].space, colours[i].rgb, base);
            if (ret < 0) {
                fprintf(stderr, "gamut: test_to_ap1 failed: %d\n", ret);
                return ret;
            }

            if (desat) {
                float mean = (base[0] + base[1] + base[2]) / 3.0f;
                for (int c = 0; c < 3; c++)
                    base[c] = 0.6f * base[c] + 0.4f * mean;
            }

            for (size_t li = 0; li < CRAW_ARRAY_ELEMS(levels); li++) {
                float in[3], out[3], unit[3];
                float ach_in, ach_out;

                for (int c = 0; c < 3; c++)
                    in[c] = out[c] = base[c] * levels[li];

                ach_in = CRAW_MAX(in[0], CRAW_MAX(in[1], in[2]));
                protect = protect && test_gamut_protected(&g, in);
                craw_gamut_compress(&g, out);
                ach_out = CRAW_MAX(out[0], CRAW_MAX(out[1], out[2]));

                if (ach_out != ach_in)
                    ach_moved = true;

                for (int c = 0; c < 3; c++) {
                    /* The operator only ever pulls a channel up towards
                     * the achromatic value; a channel moving the other way
                     * would mean the sign of the distance was wrong. */
                    if (out[c] < in[c] - 1e-7f * fabsf(ach_in))
                        outward = true;
                    worst_move = fmax(worst_move, fabs((double)out[c] - in[c]));
                }

                /* Scale invariance: the distances are ratios, so the whole
                 * operator has to commute with a change of exposure. This
                 * is what makes "at several levels" a real check rather
                 * than four repetitions of one. */
                for (int c = 0; c < 3; c++)
                    unit[c] = base[c];
                craw_gamut_compress(&g, unit);
                for (int c = 0; c < 3; c++) {
                    double want = (double)unit[c] * levels[li];
                    double rel = fabs((double)out[c] - want) /
                                 fmax(1e-6, fabs((double)ach_in));
                    worst_scale = fmax(worst_scale, rel);
                }
            }

            if (ach_moved) {
                fprintf(stderr, "gamut: %s%s moved its achromatic value\n",
                        colours[i].name, desat ? " (desaturated)" : "");
                bad++;
            }
            if (outward) {
                fprintf(stderr, "gamut: %s%s was pushed away from achromatic\n",
                        colours[i].name, desat ? " (desaturated)" : "");
                bad++;
            }
            if (!(worst_scale < 1e-5)) {
                fprintf(stderr, "gamut: %s%s is not scale invariant, off by %.3e\n",
                        colours[i].name, desat ? " (desaturated)" : "", worst_scale);
                bad++;
            }
            if (protect) {
                nprot++;
                /* The promise for a colour inside the thresholds is exact,
                 * not approximate. */
                if (!(worst_move == 0.0)) {
                    fprintf(stderr, "gamut: %s%s is inside the thresholds but moved by %.3e\n",
                            colours[i].name, desat ? " (desaturated)" : "", worst_move);
                    bad++;
                }
            } else {
                nmoved++;
            }

            printf("  %-13s%-15s %s, max move %.2e (AP1 %+.4f %+.4f %+.4f)\n",
                   colours[i].name, desat ? " desaturated" : "",
                   protect ? "untouched" : "compressed", worst_move,
                   (double)base[0], (double)base[1], (double)base[2]);
        }
    }

    printf("  %d of %d colour/saturation combinations sit inside the ACES thresholds\n",
           nprot, nprot + nmoved);

    return bad ? CRAW_ERROR(EINVAL) : 0;
}

/* How far out of AP1 this camera's matrix can possibly reach, and what is
 * left of that after the operator has pulled it in.
 *
 * The per-channel distance is (max(Mc) - (Mc)_i) / |max(Mc)| with M the
 * composed camera-RGB-to-AP1 matrix: a ratio of linear forms in c, so it is
 * invariant to the scale of c, and each of the three ratios it is a maximum
 * of attains its own maximum over the cone of non-negative camera RGB on one
 * of that cone's three extreme rays. Running the three camera primaries
 * through the real operator therefore bounds every colour the sensor can
 * produce, which is what makes "ACEScg output has no negative component" a
 * property of this camera rather than something the test scene happened not
 * to trip. Each of the profile's matrix knots is measured too, because
 * CRAW_WB_REBALANCE interpolates between them and a grade can land on any.
 *
 * The camera primaries themselves are imaginary colours — a camera's
 * spectral responses overlap, so no light gives (0, 0, 1) — and at the warm
 * end of the table one of them reaches a hair past the operator's magenta
 * limit, which is why the knots are held to a small residual rather than to
 * none. The as-shot matrix, which is the default path, clears it outright. */
static int test_gamut_reach(const CRAWProfile *profile)
{
    static const char *chan[3] = { "cyan (R)", "magenta (G)", "yellow (B)" };
    CRAWGamutCompress g;
    double worst_knot = 0.0, worst_shot = 0.0;
    double dist_shot[3] = { 0.0, 0.0, 0.0 }, dist_knot[3] = { 0.0, 0.0, 0.0 };
    float worst_cct = 0.0f;
    float ap1[3][3];
    int ret, bad = 0;

    craw_gamut_compress_params(&g);

    ret = craw_colorspace_from_xyz_d65(CRAW_CS_ACES_AP1_LINEAR, ap1);
    if (ret < 0)
        return ret;

    /* Index nb_matrix_knots is the as-shot matrix, which CRAW_WB_AS_SHOT
     * uses in place of the table. */
    for (uint32_t k = 0; k <= profile->nb_matrix_knots; k++) {
        bool as_shot = k == profile->nb_matrix_knots;
        const float (*cam)[3] = as_shot ? profile->as_shot_matrix
                                        : profile->matrix_knots[k].m;
        double worst = 0.0;

        for (int j = 0; j < 3; j++) {
            float prim[3];
            float ach;

            /* Column j of ap1 * cam is where camera primary j lands. */
            for (int i = 0; i < 3; i++) {
                prim[i] = ap1[i][0] * cam[0][j] + ap1[i][1] * cam[1][j]
                        + ap1[i][2] * cam[2][j];
            }

            ach = CRAW_MAX(prim[0], CRAW_MAX(prim[1], prim[2]));
            if (!(ach > 0.0f))
                continue;

            for (int i = 0; i < 3; i++) {
                double d = ((double)ach - prim[i]) / fabs(ach);
                if (as_shot)
                    dist_shot[i] = fmax(dist_shot[i], d);
                else
                    dist_knot[i] = fmax(dist_knot[i], d);
            }

            craw_gamut_compress(&g, prim);

            for (int i = 0; i < 3; i++)
                worst = fmax(worst, -(double)prim[i] / ach);
        }

        if (as_shot) {
            worst_shot = worst;
        } else if (worst > worst_knot) {
            worst_knot = worst;
            worst_cct = profile->matrix_knots[k].cct;
        }
    }

    for (int i = 0; i < 3; i++) {
        printf("  camera reach %-12s as-shot %.4f, worst knot %.4f, limit %.3f\n",
               chan[i], dist_shot[i], dist_knot[i], test_rgc_lim[i]);
    }
    printf("  camera primaries after compression: as-shot leaves %.3e of the pixel "
           "negative, the worst knot (%.0f K) %.3e\n", worst_shot, (double)worst_cct,
           worst_knot);

    /* The as-shot matrix is what every frame uses unless a grade rebalances
     * it, and it has room to spare. */
    if (!(worst_shot == 0.0)) {
        fprintf(stderr, "gamut: the as-shot matrix still leaves %.3e negative after "
                "compression\n", worst_shot);
        bad++;
    }

    /* Across the rebalance table the reach may graze the operator's limit;
     * what must not happen is a residual anyone could see. */
    if (!(worst_knot < 1e-3)) {
        fprintf(stderr, "gamut: a matrix knot leaves %.3e of the pixel negative after "
                "compression, which is more than rounding\n", worst_knot);
        bad++;
    }

    return bad ? CRAW_ERROR(EINVAL) : 0;
}

/* ------------------------------------------------ gamut compression, GPU */

/* A scene made of the camera's own extremes: eight vertical bands of
 * saturated camera RGB at four levels. Through this camera's matrix the
 * blue band in particular lands well outside both BT.2020 and AP1, which is
 * the point.
 *
 * The bands cross-fade into each other over the last fifth of their width
 * rather than butting up against one another. A one-pixel step from a
 * saturated blue to a saturated yellow is not a scene, it is a test of how
 * RCD behaves when its directional ratios divide by a low pass sum that is
 * itself nearly zero, and it pushes the GPU and the reference apart by more
 * than the tolerances of DESIGN.md 11 allow for reasons that have nothing
 * to do with colour. The ramp still gives the debayer plenty of gradient to
 * ring on while leaving the flat middle of every band fully saturated. */
static void test_gamut_gen_mosaic(uint16_t *m)
{
    static const float band[8][3] = {
        { 1.00f, 0.02f, 0.02f },   /* camera red */
        { 0.02f, 1.00f, 0.02f },   /* camera green */
        { 0.02f, 0.02f, 1.00f },   /* camera blue, the one far outside AP1 */
        { 1.00f, 1.00f, 0.02f },   /* yellow */
        { 0.02f, 1.00f, 1.00f },   /* cyan */
        { 1.00f, 0.02f, 1.00f },   /* magenta */
        { 1.00f, 1.00f, 1.00f },   /* neutral */
        { 0.18f, 0.06f, 1.00f },   /* an LED blue with a little red in it */
    };
    static const float level[4] = { 0.12f, 0.35f, 0.65f, 0.95f };

    for (int y = 0; y < TEST_H; y++) {
        float fy = ((float)y + 0.5f) / (float)TEST_H * 4.0f;
        int l0 = CRAW_MIN((int)fy, 3), l1 = CRAW_MIN(l0 + 1, 3);
        float ly = fy - (float)l0;
        float lw = ly < 0.8f ? 0.0f : (ly - 0.8f) / 0.2f;
        float lv = level[l0] + (level[l1] - level[l0]) * lw;

        for (int x = 0; x < TEST_W; x++) {
            int s = (y & 1) * 2 + (x & 1);
            int c = (s == 0) ? 0 : (s == 3) ? 2 : 1;
            float fx = ((float)x + 0.5f) / (float)TEST_W * 8.0f;
            int b0 = CRAW_MIN((int)fx, 7), b1 = CRAW_MIN(b0 + 1, 7);
            float bx = fx - (float)b0;
            float bw = bx < 0.8f ? 0.0f : (bx - 0.8f) / 0.2f;
            float v = (band[b0][c] + (band[b1][c] - band[b0][c]) * bw) * lv;
            float raw = TEST_BLACK + (v / test_wb[c]) * (TEST_WHITE - TEST_BLACK);
            int iv = (int)lrintf(raw * 65535.0f);

            /* The same deterministic +-8 count noise the other scene uses,
             * so that nothing here passes on a perfectly smooth signal. */
            iv += (int)(test_hash((uint32_t)(y * TEST_W + x) * 2654435761u) % 17u) - 8;
            iv = iv < 0 ? 0 : (iv > 65535 ? 65535 : iv);

            m[y * TEST_W + x] = (uint16_t)iv;
        }
    }
}

/* Pixels with any component below zero, and how far below the worst one
 * goes relative to that pixel's own magnitude. */
static uint64_t test_count_negative(const float *img, size_t npix, double *worst)
{
    uint64_t n = 0;

    *worst = 0.0;

    for (size_t i = 0; i < npix; i++) {
        const float *px = &img[i * 4];
        double ach = CRAW_MAX(px[0], CRAW_MAX(px[1], px[2]));
        bool neg = false;

        for (int c = 0; c < 3; c++) {
            if (px[c] < 0.0f) {
                neg = true;
                if (ach > 0.0)
                    *worst = fmax(*worst, -(double)px[c] / ach);
            }
        }

        n += neg;
    }

    return n;
}

/* What a space is allowed to leave negative on the scene above. */
enum test_gamut_expect {
    /* Negatives are the point of the case and are not constrained. */
    TEST_GAMUT_ANY = 0,
    /* Only where the debayer itself rang the camera RGB below zero. The
     * compression's reach is a property of the camera-to-AP1 matrix over
     * the cone of non-negative camera RGB (test_gamut_reach above); a
     * debayer with negative kernel weights leaves that cone at a hard edge
     * between two saturated colours, and outside it nothing is promised.
     * Bilinear, whose weights are all positive, never leaves it. */
    TEST_GAMUT_IN_CONE,
    /* ACES2065-1 holds every colour a real light can be, but the scene
     * below is not made of real light: camera RGB of (0.02, 0.02, 1.0) is
     * a response no spectrum gives, and such a colour can fall outside the
     * spectral locus and so outside AP0 as well. What is checked here is
     * the relation — ACES2065-1 must still be far cleaner than an
     * uncompressed ACEScg — with the breakdown printed for the record.
     * "AP0 has no negatives" is a claim about real footage and is
     * measured on real clips, not here. */
    TEST_GAMUT_REAL,
};

struct test_gamut_case {
    const char *name;
    enum CRAWColorSpace space;
    uint32_t frame_flags;
    enum test_gamut_expect expect;
};

static const struct test_gamut_case test_gamut_cases[] = {
    { "BT.2020",          CRAW_CS_BT2020_LINEAR,   0,                                    TEST_GAMUT_ANY },
    { "AP1 uncompressed", CRAW_CS_ACES_AP1_LINEAR, CRAW_FRAME_FLAG_NO_GAMUT_COMPRESSION, TEST_GAMUT_ANY },
    { "AP1 compressed",   CRAW_CS_ACES_AP1_LINEAR, 0,                                    TEST_GAMUT_IN_CONE },
    { "AP0",              CRAW_CS_ACES_AP0_LINEAR, 0,                                    TEST_GAMUT_REAL },
    /* The flag is documented to do nothing outside ACEScg; this is the
     * pair that proves ACES2065-1 keeps every value the matrix produced. */
    { "AP0 (flag set)",   CRAW_CS_ACES_AP0_LINEAR, CRAW_FRAME_FLAG_NO_GAMUT_COMPRESSION, TEST_GAMUT_REAL },
};

static int test_gamut_gpu(CRAWTestDevice *td, CRAWContext *ctx, const CRAWProfile *profile,
                          CRAWTestImage *src, CRAWTestImage *dst, float *gpu, float *ref)
{
    const size_t npix = (size_t)TEST_W * TEST_H;
    uint16_t *mosaic = NULL;
    float *ap0_plain = NULL, *xyz = NULL, *cam = NULL;
    int ret;

    mosaic = malloc(npix * sizeof(*mosaic));
    ap0_plain = malloc(npix * 4 * sizeof(*ap0_plain));
    xyz = malloc(npix * 4 * sizeof(*xyz));
    cam = malloc(npix * 4 * sizeof(*cam));
    if (!mosaic || !ap0_plain || !xyz || !cam) {
        ret = CRAW_ERROR(ENOMEM);
        goto done;
    }

    test_gamut_gen_mosaic(mosaic);

    ret = craw_testdev_image_upload(td, src, mosaic, npix * sizeof(*mosaic));
    if (ret < 0) {
        fprintf(stderr, "gamut: upload failed: %d\n", ret);
        goto done;
    }

    for (int mi = 0; mi < CRAW_DEBAYER_NB; mi++) {
        enum CRAWDebayer method = (enum CRAWDebayer)mi;
        const char *mname = test_method_name(method);
        double tol_rel = method == CRAW_DEBAYER_RCD ? 1.5e-2 : 1e-2;
        CRAWPipelineParams xp;
        CRAWFrameParams xf;
        uint64_t uncompressed_neg = 0;

        /* The same scene in XYZ, on the CPU only: the luminance channel is
         * what says whether a pixel is a colour at all. */
        craw_pipeline_params_init(&xp);
        xp.width = TEST_W;
        xp.height = TEST_H;
        xp.debayer = method;
        xp.out_space = CRAW_CS_XYZ_D65;

        craw_frame_params_init(&xf);
        xf.profile = profile;

        ret = craw_ref_process(mosaic, TEST_W, TEST_H, TEST_W, &xp, &xf, xyz);
        if (ret < 0) {
            fprintf(stderr, "gamut: craw_ref_process (XYZ) failed: %d\n", ret);
            goto done;
        }

        /* And the camera RGB the colour stage was handed, which is where
         * the debayer's ringing shows up. */
        ret = craw_ref_debayer(mosaic, TEST_W, TEST_H, TEST_W, profile, method, cam);
        if (ret < 0) {
            fprintf(stderr, "gamut: craw_ref_debayer failed: %d\n", ret);
            goto done;
        }

        for (size_t ci = 0; ci < CRAW_ARRAY_ELEMS(test_gamut_cases); ci++) {
            const struct test_gamut_case *tc = &test_gamut_cases[ci];
            CRAWPipelineParams pp;
            CRAWFrameParams fp;
            uint64_t neg_gpu, neg_ref;
            double worst_gpu, worst_ref, err, ms = 0.0;


            craw_pipeline_params_init(&pp);
            pp.width = TEST_W;
            pp.height = TEST_H;
            pp.debayer = method;
            pp.out_space = tc->space;

            craw_frame_params_init(&fp);
            fp.profile = profile;
            fp.flags = tc->frame_flags;

            ret = craw_ref_process(mosaic, TEST_W, TEST_H, TEST_W, &pp, &fp, ref);
            if (ret < 0) {
                fprintf(stderr, "gamut: craw_ref_process failed: %d\n", ret);
                goto done;
            }

            ret = test_run_config(td, ctx, src, dst, &pp, &fp, gpu, NULL, NULL, &ms);
            if (ret < 0)
                goto done;

            err = test_max_rel(gpu, ref, npix);
            neg_gpu = test_count_negative(gpu, npix, &worst_gpu);
            neg_ref = test_count_negative(ref, npix, &worst_ref);

            if (tc->space == CRAW_CS_ACES_AP1_LINEAR && tc->frame_flags)
                uncompressed_neg = neg_ref;

            printf("  %-8s %-17s: %6" PRIu64 " negative px gpu (%5.2f%%, worst %.3f of the pixel),"
                   " %6" PRIu64 " ref, max rel %.3e\n",
                   mname, tc->name, neg_gpu, 100.0 * (double)neg_gpu / (double)npix,
                   worst_gpu, neg_ref, err);

            if (!(err <= tol_rel)) {
                fprintf(stderr, "gamut: %s %s differs by %.3e relative, tolerance %.1e\n",
                        mname, tc->name, err, tol_rel);
                ret = CRAW_ERROR(EINVAL);
                goto done;
            }

            /* Every negative the case does not excuse, counted on both
             * the GPU result and the reference so that neither can hide
             * one the other has. */
            if (tc->expect != TEST_GAMUT_ANY) {
                uint64_t unexplained = 0, rang = 0, imaginary = 0;

                for (size_t i = 0; i < npix; i++) {
                    bool neg = false;

                    for (int c = 0; c < 3; c++)
                        neg = neg || ref[i * 4 + c] < 0.0f || gpu[i * 4 + c] < 0.0f;
                    if (!neg)
                        continue;

                    /* A camera RGB the debayer rang below zero is no more a
                     * real camera response than a colour of no luminance is
                     * a real colour, so it excuses both kinds of case. */
                    if (cam[i * 4] < 0.0f || cam[i * 4 + 1] < 0.0f || cam[i * 4 + 2] < 0.0f)
                        rang++;
                    else if (tc->expect == TEST_GAMUT_REAL && !(xyz[i * 4 + 1] > 0.0f))
                        imaginary++;
                    else
                        unexplained++;
                }

                if (rang || imaginary || unexplained)
                    printf("      (%" PRIu64 " rang negative in the debayer, %" PRIu64
                           " have no positive luminance, %" PRIu64 " otherwise outside"
                           " the space)\n", rang, imaginary, unexplained);

                /* ACEScg after compression is held to the exact property:
                 * nothing negative anywhere the camera RGB stayed in the
                 * non-negative cone that test_gamut_reach bounds. */
                if (tc->expect == TEST_GAMUT_IN_CONE && unexplained) {
                    fprintf(stderr, "gamut: %s %s left %" PRIu64 " negative pixels on "
                            "camera RGB the debayer did not ring negative\n",
                            mname, tc->name, unexplained);
                    ret = CRAW_ERROR(EINVAL);
                    goto done;
                }

                /* ACES2065-1 is held to the relation instead, see the enum. */
                if (tc->expect == TEST_GAMUT_REAL && !(neg_ref * 2 < uncompressed_neg)) {
                    fprintf(stderr, "gamut: %s %s has %" PRIu64 " negative pixels against "
                            "%" PRIu64 " for an uncompressed ACEScg; it should be far "
                            "cleaner\n", mname, tc->name, neg_ref, uncompressed_neg);
                    ret = CRAW_ERROR(EINVAL);
                    goto done;
                }
            } else if (!neg_ref) {
                /* The uncompressed spaces are only interesting if they do
                 * produce negatives here; a scene that stayed in gamut
                 * would make the whole comparison vacuous. */
                fprintf(stderr, "gamut: %s %s produced no negatives at all, "
                        "the test scene is not out of gamut\n", mname, tc->name);
                ret = CRAW_ERROR(EINVAL);
                goto done;
            }

            /* Keep the plain ACES2065-1 result to compare the flagged one
             * against: the flag must not change a single bit there. */
            if (tc->space == CRAW_CS_ACES_AP0_LINEAR && !tc->frame_flags) {
                memcpy(ap0_plain, gpu, npix * 4 * sizeof(*ap0_plain));
            } else if (tc->space == CRAW_CS_ACES_AP0_LINEAR) {
                if (memcmp(ap0_plain, gpu, npix * 4 * sizeof(*ap0_plain)) != 0) {
                    fprintf(stderr, "gamut: %s the flag changed ACES2065-1 output\n", mname);
                    ret = CRAW_ERROR(EINVAL);
                    goto done;
                }
            }
        }
    }

    /* One pipeline, two spaces, frame by frame: this is the case the
     * viewer needs, where a display frame in ACEScg and an archival frame
     * in ACES2065-1 come out of the same pipeline with no rebuild. The
     * pipeline is created for a third space on purpose, so that a frame
     * silently falling back to the pipeline's own would be caught. */
    {
        CRAWPipelineParams pp;
        CRAWFrameParams fp;
        static const enum CRAWColorSpace per_frame[] = {
            CRAW_CS_ACES_AP1_LINEAR, CRAW_CS_ACES_AP0_LINEAR, CRAW_CS_ACES_AP1_LINEAR,
        };
        CRAWPipeline *pipe = NULL;
        struct test_run run;

        craw_pipeline_params_init(&pp);
        pp.width = TEST_W;
        pp.height = TEST_H;
        pp.debayer = CRAW_DEBAYER_RCD;
        pp.out_space = CRAW_CS_BT2020_LINEAR;

        ret = craw_pipeline_create(ctx, &pp, &pipe);
        if (ret < 0) {
            fprintf(stderr, "gamut: craw_pipeline_create failed: %d\n", ret);
            goto done;
        }

        for (size_t i = 0; i < CRAW_ARRAY_ELEMS(per_frame); i++) {
            uint16_t *half = NULL;
            uint64_t neg;
            double worst, err, ms = 0.0;

            craw_frame_params_init(&fp);
            fp.profile = profile;
            fp.out_space = per_frame[i];
            fp.out_space_set = true;

            ret = craw_ref_process(mosaic, TEST_W, TEST_H, TEST_W, &pp, &fp, ref);
            if (ret < 0) {
                fprintf(stderr, "gamut: craw_ref_process failed: %d\n", ret);
                craw_pipeline_free(&pipe);
                goto done;
            }

            run = (struct test_run) { .pipe = pipe, .src = src, .dst = dst, .fp = &fp };
            ret = craw_testdev_run_timed(td, test_record, &run, &ms);
            if (ret < 0) {
                fprintf(stderr, "gamut: submission failed: %d\n", ret);
                craw_pipeline_free(&pipe);
                goto done;
            }

            half = malloc(npix * 4 * sizeof(*half));
            if (!half) {
                ret = CRAW_ERROR(ENOMEM);
                craw_pipeline_free(&pipe);
                goto done;
            }

            ret = craw_testdev_image_download(td, dst, half, npix * 4 * sizeof(*half));
            if (ret == 0) {
                for (size_t k = 0; k < npix * 4; k++)
                    gpu[k] = test_half_to_float(half[k]);
            }
            free(half);
            if (ret < 0) {
                fprintf(stderr, "gamut: download failed: %d\n", ret);
                craw_pipeline_free(&pipe);
                goto done;
            }

            err = test_max_rel(gpu, ref, npix);
            neg = test_count_negative(gpu, npix, &worst);

            printf("  per frame #%zu %-8s on a BT.2020 pipeline: %" PRIu64
                   " negative px, max rel %.3e, %.3f ms\n",
                   i, test_space_name(per_frame[i]), neg, err, ms);

            /* ACEScg comes out clean because it was compressed, which is
             * the proof that the frame did not quietly fall back to the
             * pipeline's own BT.2020: that would leave thousands of
             * negatives here and fail the reference comparison too. */
            if (per_frame[i] == CRAW_CS_ACES_AP1_LINEAR && neg) {
                fprintf(stderr, "gamut: per frame %s left %" PRIu64 " negative pixels\n",
                        test_space_name(per_frame[i]), neg);
                ret = CRAW_ERROR(EINVAL);
                craw_pipeline_free(&pipe);
                goto done;
            }
            if (!(err <= 1.5e-2)) {
                fprintf(stderr, "gamut: per frame %s differs by %.3e relative\n",
                        test_space_name(per_frame[i]), err);
                ret = CRAW_ERROR(EINVAL);
                craw_pipeline_free(&pipe);
                goto done;
            }
        }

        craw_pipeline_free(&pipe);
    }

    ret = 0;

done:
    free(cam);
    free(xyz);
    free(ap0_plain);
    free(mosaic);

    return ret;
}

static int test_gamut(CRAWTestDevice *td, CRAWContext *ctx, const CRAWProfile *profile,
                      CRAWTestImage *src, CRAWTestImage *dst, float *gpu, float *ref)
{
    int ret;

    ret = test_gamut_curve();
    if (ret < 0)
        return ret;

    ret = test_gamut_reach(profile);
    if (ret < 0)
        return ret;

    ret = test_gamut_invariance();
    if (ret < 0)
        return ret;

    return test_gamut_gpu(td, ctx, profile, src, dst, gpu, ref);
}

static int test_validation(CRAWContext *ctx)
{
    CRAWAwbHistogram hist;
    CRAWPipelineParams pp;
    CRAWPipeline *pipe = NULL;
    int ret;

    craw_pipeline_params_init(&pp);
    pp.width = TEST_W;
    pp.height = TEST_H;
    pp.cfa = CRAW_CFA_BGGR;
    if (craw_pipeline_create(ctx, &pp, &pipe) != CRAW_ERROR(ENOTSUP)) {
        fprintf(stderr, "a non-RGGB CFA was not rejected with ENOTSUP\n");
        craw_pipeline_free(&pipe);
        return -1;
    }

    craw_pipeline_params_init(&pp);
    pp.width = TEST_W + 1;
    pp.height = TEST_H;
    if (craw_pipeline_create(ctx, &pp, &pipe) != CRAW_ERROR(EINVAL)) {
        fprintf(stderr, "an odd width was not rejected with EINVAL\n");
        craw_pipeline_free(&pipe);
        return -1;
    }

    craw_pipeline_params_init(&pp);
    pp.width = TEST_W;
    pp.height = TEST_H;
    pp.out_format = VK_FORMAT_R8G8B8A8_UNORM;
    if (craw_pipeline_create(ctx, &pp, &pipe) != CRAW_ERROR(EINVAL)) {
        fprintf(stderr, "an unsupported output format was not rejected with EINVAL\n");
        craw_pipeline_free(&pipe);
        return -1;
    }

    /* The denoise stage's plane mirror is good for one reflection, so it
     * has a minimum frame size of its own. */
    craw_pipeline_params_init(&pp);
    pp.width = 8;
    pp.height = 8;
    pp.flags = CRAW_PIPELINE_FLAG_DENOISE;
    if (craw_pipeline_create(ctx, &pp, &pipe) != CRAW_ERROR(EINVAL)) {
        fprintf(stderr, "an 8x8 denoise pipeline was not rejected with EINVAL\n");
        craw_pipeline_free(&pipe);
        return -1;
    }

    craw_pipeline_params_init(&pp);
    pp.width = TEST_W;
    pp.height = TEST_H;
    ret = craw_pipeline_create(ctx, &pp, &pipe);
    if (ret < 0) {
        fprintf(stderr, "craw_pipeline_create failed: %d\n", ret);
        return -1;
    }
    ret = craw_pipeline_get_awb_histogram(pipe, &hist);
    craw_pipeline_free(&pipe);
    if (ret != CRAW_ERROR(EINVAL)) {
        fprintf(stderr, "the histogram of a pipeline without the flag returned %d\n", ret);
        return -1;
    }

    return 0;
}

/* --------------------------------------------------------- false colour */

/* Largest |R - G| or |B - G| over a rectangle of a camera RGB buffer. */
static double test_patch_chroma(const float *rgb, int x0, int y0, int x1, int y1)
{
    double worst = 0.0;

    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            const float *px = &rgb[((size_t)y * TEST_W + x) * 4];
            double dr = fabs((double)px[0] - (double)px[1]);
            double db = fabs((double)px[2] - (double)px[1]);

            worst = CRAW_MAX(worst, CRAW_MAX(dr, db));
        }
    }

    return worst;
}

/* GPU against the reference with the false colour stage after the debayer,
 * DESIGN.md 17.5: both ways of switching it off, one and three passes on
 * MHC (the stage's own image) and on RCD (RCD's aux, borrowed), too many
 * passes refused, and the definition itself on the CPU.
 *
 * As in test_denoise(), every filtered frame is read against two numbers:
 * the reference of the same configuration, which it has to match within
 * the debayer's own tolerance, and the unfiltered reference, which it has
 * to sit well away from, at least three floors, or the pass did not run. */
static int test_false_color(CRAWTestDevice *td, CRAWContext *ctx, const CRAWProfile *identity,
                            const uint16_t *mosaic, CRAWTestImage *src, CRAWTestImage *dst,
                            float *gpu, float *ref)
{
    static const enum CRAWDebayer methods[] = { CRAW_DEBAYER_MHC, CRAW_DEBAYER_RCD };
    static const uint32_t passes[] = { 1, CRAW_FALSE_COLOR_MAX_PASSES };
    /* One spike per site colour, well inside the neutral flat patch of
     * test_scene_rgb() and well apart: (170, 30) is an R site, (181, 41) a
     * B site and (191, 34) a G site of an R row. */
    static const int spikes[3][2] = { { 170, 30 }, { 181, 41 }, { 191, 34 } };
    const size_t npix = (size_t)TEST_W * TEST_H;
    CRAWPipelineParams pp;
    CRAWFrameParams fp;
    uint16_t *spiked = NULL;
    float *plain = NULL;
    double ms = 0.0;
    int ret = -1;

    spiked = malloc(npix * sizeof(*spiked));
    plain = malloc(npix * 4 * sizeof(*plain));
    if (!spiked || !plain)
        goto done;

    /* Off both ways: passes set on a pipeline built without the flag, and
     * no passes on one built with it. Both must reproduce the plain chain. */
    for (int off = 0; off < 2; off++) {
        double base;

        craw_pipeline_params_init(&pp);
        pp.width = TEST_W;
        pp.height = TEST_H;
        pp.debayer = CRAW_DEBAYER_MHC;
        pp.out_space = CRAW_CS_XYZ_D65;
        pp.flags = off ? 0 : CRAW_PIPELINE_FLAG_FALSE_COLOR;

        craw_frame_params_init(&fp);
        fp.profile = identity;
        fp.false_color_passes = off ? CRAW_FALSE_COLOR_MAX_PASSES : 0;

        if (craw_ref_debayer(mosaic, TEST_W, TEST_H, TEST_W, identity,
                             CRAW_DEBAYER_MHC, ref) < 0)
            goto done;
        if (test_run_config(td, ctx, src, dst, &pp, &fp, gpu, NULL, NULL, &ms) < 0)
            goto done;

        base = test_max_abs(gpu, ref, npix);
        printf("  false colour %-12s: max abs %.3e\n", off ? "flag not set" : "passes 0", base);
        if (!(base <= 2e-3)) {
            fprintf(stderr, "the false colour pass ran when it should not have (%.3e)\n", base);
            goto done;
        }
    }

    for (size_t mi = 0; mi < CRAW_ARRAY_ELEMS(methods); mi++) {
        enum CRAWDebayer method = methods[mi];
        const char *mname = test_method_name(method);
        double tol = TEST_FC_TOL_FACTOR * (method == CRAW_DEBAYER_RCD ? 5e-3 : 2e-3);
        double base;

        craw_pipeline_params_init(&pp);
        pp.width = TEST_W;
        pp.height = TEST_H;
        pp.debayer = method;
        pp.out_space = CRAW_CS_XYZ_D65;
        pp.flags = CRAW_PIPELINE_FLAG_FALSE_COLOR;

        craw_frame_params_init(&fp);
        fp.profile = identity;

        /* The floor: this debayer's own GPU/CPU disagreement plus the fp16
         * intermediate, before the stage adds anything. */
        fp.false_color_passes = 0;
        if (craw_ref_debayer(mosaic, TEST_W, TEST_H, TEST_W, identity, method, ref) < 0)
            goto done;
        if (test_run_config(td, ctx, src, dst, &pp, &fp, gpu, NULL, NULL, &ms) < 0)
            goto done;
        base = test_max_abs(gpu, ref, npix);

        for (size_t pi = 0; pi < CRAW_ARRAY_ELEMS(passes); pi++) {
            double cross, err;

            fp.false_color_passes = passes[pi];
            if (test_run_config(td, ctx, src, dst, &pp, &fp, gpu, NULL, NULL, &ms) < 0)
                goto done;

            /* `ref` holds the unfiltered reference, rebuilt below. */
            cross = test_max_abs(gpu, ref, npix);

            if (craw_ref_process(mosaic, TEST_W, TEST_H, TEST_W, &pp, &fp, ref) < 0) {
                fprintf(stderr, "false colour: craw_ref_process failed\n");
                goto done;
            }
            err = test_max_abs(gpu, ref, npix);

            printf("  false colour %-8s %u pass%s: max abs %.3e (tol %.1e), %.3e against the "
                   "unfiltered chain (floor %.3e), %.3f ms\n",
                   mname, passes[pi], passes[pi] == 1 ? " " : "es", err, tol, cross, base, ms);
            if (!(err <= tol)) {
                fprintf(stderr, "false colour on %s with %u passes differs by %.3e, "
                        "tolerance %.1e\n", mname, passes[pi], err, tol);
                goto done;
            }
            if (!(cross > 3.0 * base)) {
                fprintf(stderr, "a filtered %s frame is only %.3e from an unfiltered one "
                        "against a floor of %.3e; the pass cannot have run\n",
                        mname, cross, base);
                goto done;
            }

            if (craw_ref_debayer(mosaic, TEST_W, TEST_H, TEST_W, identity, method, ref) < 0)
                goto done;
        }
    }

    /* The definition, on the CPU: a single site spike through the bilinear
     * debayer is a blob of its own colour, A at the centre, A/2 at the
     * edges and A/4 at the corners in the colour differences. The median
     * of one pass halves the centre and the blur then spreads it, leaving
     * a sixth; the maximum leaves nothing measurable whatever the debayer
     * (DESIGN.md 17.2). G is not the stage's to touch. */
    memcpy(spiked, mosaic, npix * sizeof(*spiked));
    for (size_t i = 0; i < CRAW_ARRAY_ELEMS(spikes); i++)
        spiked[spikes[i][1] * TEST_W + spikes[i][0]] = 40000;

    for (int mi = 0; mi < CRAW_DEBAYER_NB; mi++) {
        enum CRAWDebayer method = (enum CRAWDebayer)mi;
        double before, after1, after_max;

        if (craw_ref_debayer(spiked, TEST_W, TEST_H, TEST_W, identity, method, plain) < 0)
            goto done;

        before = test_patch_chroma(plain, 164, 28, 195, 51);

        memcpy(ref, plain, npix * 4 * sizeof(*ref));
        if (craw_ref_false_color(ref, TEST_W, TEST_H, 1) < 0) {
            fprintf(stderr, "craw_ref_false_color failed\n");
            goto done;
        }
        after1 = test_patch_chroma(ref, 164, 28, 195, 51);

        memcpy(ref, plain, npix * 4 * sizeof(*ref));
        if (craw_ref_false_color(ref, TEST_W, TEST_H, CRAW_FALSE_COLOR_MAX_PASSES) < 0) {
            fprintf(stderr, "craw_ref_false_color failed\n");
            goto done;
        }
        after_max = test_patch_chroma(ref, 164, 28, 195, 51);

        for (size_t i = 0; i < npix; i++) {
            if (ref[i * 4 + 1] != plain[i * 4 + 1] || ref[i * 4 + 3] != plain[i * 4 + 3]) {
                fprintf(stderr, "false colour moved G or alpha at pixel %zu\n", i);
                goto done;
            }
        }

        printf("  false colour %-8s spikes : chroma %.3f -> %.3f after one pass (%.0f%%%s), "
               "%.4f after %u (%.1f%%, want <= 10%%)\n", test_method_name(method),
               before, after1, 100.0 * after1 / before,
               method == CRAW_DEBAYER_BILINEAR ? ", want <= 65%" : "",
               after_max, (unsigned)CRAW_FALSE_COLOR_MAX_PASSES, 100.0 * after_max / before);
        if (!(after_max <= 0.10 * before) ||
            (method == CRAW_DEBAYER_BILINEAR && !(after1 <= 0.65 * before))) {
            fprintf(stderr, "false colour left %.0f%% of a %s spike's colour after one pass "
                    "and %.1f%% after %u\n", 100.0 * after1 / before,
                    test_method_name(method), 100.0 * after_max / before,
                    (unsigned)CRAW_FALSE_COLOR_MAX_PASSES);
            goto done;
        }
    }

    /* More passes than the tile has apron for is an error, not a clamp. */
    {
        CRAWPipeline *pipe = NULL;
        struct test_run run;
        int rec;

        craw_pipeline_params_init(&pp);
        pp.width = TEST_W;
        pp.height = TEST_H;
        pp.flags = CRAW_PIPELINE_FLAG_FALSE_COLOR;

        craw_frame_params_init(&fp);
        fp.profile = identity;
        fp.false_color_passes = CRAW_FALSE_COLOR_MAX_PASSES + 1;

        if (craw_pipeline_create(ctx, &pp, &pipe) < 0)
            goto done;

        run = (struct test_run) { .pipe = pipe, .src = src, .dst = dst, .fp = &fp };
        rec = craw_testdev_run(td, test_record, &run);
        craw_pipeline_free(&pipe);
        if (rec != CRAW_ERROR(EINVAL)) {
            fprintf(stderr, "%u false colour passes returned %d, wanted %d\n",
                    (unsigned)CRAW_FALSE_COLOR_MAX_PASSES + 1, rec, CRAW_ERROR(EINVAL));
            goto done;
        }
    }

    ret = 0;

done:
    free(plain);
    free(spiked);
    return ret;
}

/* -------------------------------------------------------- chroma denoise */

/* Mean of a channel's colour difference over a rectangle. */
static double test_patch_mean_diff(const float *rgb, int c, int x0, int y0, int x1, int y1)
{
    double acc = 0.0;
    int n = 0;

    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            const float *px = &rgb[((size_t)y * TEST_W + x) * 4];
            acc += (double)px[c] - (double)px[1];
            n++;
        }
    }

    return acc / n;
}

/* Spread of the 4x4 block means of R - G over a rectangle whose corners
 * lie on the block grid, the band the stage works on, with each block's
 * 3x3 neighbourhood mean taken out first so that a smooth trend across the
 * patch (what little of the surround's chroma the range terms let in) is
 * not counted as noise. */
static double test_patch_band_noise(const float *rgb, int x0, int y0, int x1, int y1)
{
    double m[16][16];
    double acc = 0.0, acc2 = 0.0;
    int nx = 0, ny = 0, n = 0;

    for (int by = y0; by + 3 <= y1 && ny < 16; by += 4, ny++) {
        nx = 0;
        for (int bx = x0; bx + 3 <= x1 && nx < 16; bx += 4, nx++)
            m[ny][nx] = test_patch_mean_diff(rgb, 0, bx, by, bx + 3, by + 3);
    }

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            double local = 0.0;
            int k = 0;

            for (int dj = -1; dj <= 1; dj++) {
                for (int di = -1; di <= 1; di++) {
                    int jj = CRAW_CLAMP(j + dj, 0, ny - 1), ii = CRAW_CLAMP(i + di, 0, nx - 1);
                    local += m[jj][ii];
                    k++;
                }
            }
            local = m[j][i] - local / k;
            acc += local;
            acc2 += local * local;
            n++;
        }
    }

    return sqrt(CRAW_MAX(acc2 / n - (acc / n) * (acc / n), 0.0));
}

/* GPU against the reference with the chroma denoise after the debayer,
 * DESIGN.md 14.6: both ways of switching it off, half and full strength on
 * MHC (its own scratch image) and on RCD (RCD's aux, borrowed), a strength
 * above one refused, and the definition on the CPU. The same three-floor
 * reading as the false colour test. */
static int test_chroma(CRAWTestDevice *td, CRAWContext *ctx, const CRAWProfile *identity,
                       const uint16_t *mosaic, CRAWTestImage *src, CRAWTestImage *dst,
                       float *gpu, float *ref)
{
    static const enum CRAWDebayer methods[] = { CRAW_DEBAYER_MHC, CRAW_DEBAYER_RCD };
    static const float strengths[] = { 0.5f, 1.0f };
    const size_t npix = (size_t)TEST_W * TEST_H;
    CRAWPipelineParams pp;
    CRAWFrameParams fp;
    float *plain = NULL;
    double ms = 0.0;
    int ret = -1;

    plain = malloc(npix * 4 * sizeof(*plain));
    if (!plain)
        goto done;

    for (int off = 0; off < 2; off++) {
        double base;

        craw_pipeline_params_init(&pp);
        pp.width = TEST_W;
        pp.height = TEST_H;
        pp.debayer = CRAW_DEBAYER_MHC;
        pp.out_space = CRAW_CS_XYZ_D65;
        pp.flags = off ? 0 : CRAW_PIPELINE_FLAG_CHROMA_DENOISE;

        craw_frame_params_init(&fp);
        fp.profile = identity;
        fp.denoise_chroma_strength = off ? 1.0f : 0.0f;

        if (craw_ref_debayer(mosaic, TEST_W, TEST_H, TEST_W, identity,
                             CRAW_DEBAYER_MHC, ref) < 0)
            goto done;
        if (test_run_config(td, ctx, src, dst, &pp, &fp, gpu, NULL, NULL, &ms) < 0)
            goto done;

        base = test_max_abs(gpu, ref, npix);
        printf("  chroma   %-12s: max abs %.3e\n", off ? "flag not set" : "strength 0", base);
        if (!(base <= 2e-3)) {
            fprintf(stderr, "the chroma denoise ran when it should not have (%.3e)\n", base);
            goto done;
        }
    }

    for (size_t mi = 0; mi < CRAW_ARRAY_ELEMS(methods); mi++) {
        enum CRAWDebayer method = methods[mi];
        const char *mname = test_method_name(method);
        double tol = TEST_FC_TOL_FACTOR * (method == CRAW_DEBAYER_RCD ? 5e-3 : 2e-3);
        double base;

        craw_pipeline_params_init(&pp);
        pp.width = TEST_W;
        pp.height = TEST_H;
        pp.debayer = method;
        pp.out_space = CRAW_CS_XYZ_D65;
        pp.flags = CRAW_PIPELINE_FLAG_CHROMA_DENOISE;

        craw_frame_params_init(&fp);
        fp.profile = identity;

        fp.denoise_chroma_strength = 0.0f;
        if (craw_ref_debayer(mosaic, TEST_W, TEST_H, TEST_W, identity, method, ref) < 0)
            goto done;
        if (test_run_config(td, ctx, src, dst, &pp, &fp, gpu, NULL, NULL, &ms) < 0)
            goto done;
        base = test_max_abs(gpu, ref, npix);

        for (size_t si = 0; si < CRAW_ARRAY_ELEMS(strengths); si++) {
            double cross, err;

            fp.denoise_chroma_strength = strengths[si];
            if (test_run_config(td, ctx, src, dst, &pp, &fp, gpu, NULL, NULL, &ms) < 0)
                goto done;

            cross = test_max_abs(gpu, ref, npix);

            if (craw_ref_process(mosaic, TEST_W, TEST_H, TEST_W, &pp, &fp, ref) < 0) {
                fprintf(stderr, "chroma: craw_ref_process failed\n");
                goto done;
            }
            err = test_max_abs(gpu, ref, npix);

            printf("  chroma   %-8s strength %.1f: max abs %.3e (tol %.1e), %.3e against the "
                   "unfiltered chain (floor %.3e), %.3f ms\n",
                   mname, (double)strengths[si], err, tol, cross, base, ms);
            if (!(err <= tol)) {
                fprintf(stderr, "chroma denoise on %s at %.1f differs by %.3e, tolerance %.1e\n",
                        mname, (double)strengths[si], err, tol);
                goto done;
            }
            if (!(cross > 3.0 * base)) {
                fprintf(stderr, "a chroma denoised %s frame is only %.3e from an unfiltered one "
                        "against a floor of %.3e; the pass cannot have run\n",
                        mname, cross, base);
                goto done;
            }

            if (craw_ref_debayer(mosaic, TEST_W, TEST_H, TEST_W, identity, method, ref) < 0)
                goto done;
        }
    }

    /* The definition, on the CPU: chroma blotches painted over the neutral
     * flat patch, +-0.02 per 8x8 block, the kind of thing a dark road
     * carries, lose most of their spread in the band the stage works on; a
     * flat coloured patch keeps its colour at its centre; and G is not the
     * stage's to touch. The synthetic mosaic's own chroma noise is a few
     * 1e-5, too small to measure the stage against. */
    {
        double before, after, warm_r, warm_b, warm_r2, warm_b2;

        if (craw_ref_debayer(mosaic, TEST_W, TEST_H, TEST_W, identity,
                             CRAW_DEBAYER_BILINEAR, plain) < 0)
            goto done;
        for (int y = 24; y <= 55; y++) {
            for (int x = 160; x <= 199; x++) {
                uint32_t hb = test_hash((uint32_t)((y / 8) * 64 + x / 8) * 2246822519u);
                float *px = &plain[((size_t)y * TEST_W + x) * 4];

                px[0] += (float)((int)(hb % 41u) - 20) * 0.001f;
                px[2] += (float)((int)((hb >> 8) % 41u) - 20) * 0.001f;
            }
        }
        memcpy(ref, plain, npix * 4 * sizeof(*ref));
        if (craw_ref_chroma_denoise(ref, TEST_W, TEST_H, 1.0f) < 0) {
            fprintf(stderr, "craw_ref_chroma_denoise failed\n");
            goto done;
        }

        for (size_t i = 0; i < npix; i++) {
            if (ref[i * 4 + 1] != plain[i * 4 + 1] || ref[i * 4 + 3] != plain[i * 4 + 3]) {
                fprintf(stderr, "chroma denoise moved G or alpha at pixel %zu\n", i);
                goto done;
            }
        }

        /* The neutral patch of test_scene_rgb() is 160..199 x 24..55; its
         * 4x4 blocks from (164, 28) are whole and inset. */
        before = test_patch_band_noise(plain, 164, 28, 195, 51);
        after = test_patch_band_noise(ref, 164, 28, 195, 51);
        /* The warm patch is 16..47 square; its central 8x8. */
        warm_r = test_patch_mean_diff(plain, 0, 28, 28, 35, 35);
        warm_b = test_patch_mean_diff(plain, 2, 28, 28, 35, 35);
        warm_r2 = test_patch_mean_diff(ref, 0, 28, 28, 35, 35);
        warm_b2 = test_patch_mean_diff(ref, 2, 28, 28, 35, 35);

        printf("  chroma   definition   : neutral patch blotches %.2e -> %.2e (%.0f%%, want "
               "<= 60%%), warm patch centre R-G %.3f -> %.3f, B-G %.3f -> %.3f (want within 5%%)\n",
               before, after, 100.0 * after / before, warm_r, warm_r2, warm_b, warm_b2);
        if (!(after <= 0.6 * before)) {
            fprintf(stderr, "chroma denoise left %.0f%% of the blotches\n", 100.0 * after / before);
            goto done;
        }
        if (!(fabs(warm_r2 - warm_r) <= 0.05 * fabs(warm_r)) ||
            !(fabs(warm_b2 - warm_b) <= 0.05 * fabs(warm_b))) {
            fprintf(stderr, "chroma denoise moved a flat colour by more than 5%%\n");
            goto done;
        }
    }

    /* A strength above one is an error, not a clamp. */
    {
        CRAWPipeline *pipe = NULL;
        struct test_run run;
        int rec;

        craw_pipeline_params_init(&pp);
        pp.width = TEST_W;
        pp.height = TEST_H;
        pp.flags = CRAW_PIPELINE_FLAG_CHROMA_DENOISE;

        craw_frame_params_init(&fp);
        fp.profile = identity;
        fp.denoise_chroma_strength = 1.5f;

        if (craw_pipeline_create(ctx, &pp, &pipe) < 0)
            goto done;

        run = (struct test_run) { .pipe = pipe, .src = src, .dst = dst, .fp = &fp };
        rec = craw_testdev_run(td, test_record, &run);
        craw_pipeline_free(&pipe);
        if (rec != CRAW_ERROR(EINVAL)) {
            fprintf(stderr, "a chroma strength of 1.5 returned %d, wanted %d\n",
                    rec, CRAW_ERROR(EINVAL));
            goto done;
        }
    }

    ret = 0;

done:
    free(plain);
    return ret;
}

/* ------------------------------------------------ highlight reconstruction */

/* The highlight scene of DESIGN.md 18.5, in the balanced camera RGB a
 * perfect debayer would produce: two Gaussian blobs of constant chroma on
 * a dim gradient. The first peaks at G = 1.4 with R and B below their
 * clips, so only G clips and the truth of what the stage has to rebuild is
 * known; the second peaks at G = 4.0, so its core is blown in every
 * channel. A pixel belongs to a blob wherever the blob's G is above 0.05
 * and to the background elsewhere, rather than the two being added, so the
 * chroma the field is built from is exactly the blob's. */
#define TEST_HL_PART_X 64.0f
#define TEST_HL_FULL_X 184.0f
#define TEST_HL_Y      80.0f

static const float test_hl_part[3] = { 1.30f, 1.0f, 0.90f };
static const float test_hl_full[3] = { 1.10f, 1.0f, 1.05f };

static void test_hl_truth(int x, int y, float rgb[3])
{
    float fx = (float)x / (float)(TEST_W - 1);
    float fy = (float)y / (float)(TEST_H - 1);
    float dp = ((float)x - TEST_HL_PART_X) * ((float)x - TEST_HL_PART_X) +
               ((float)y - TEST_HL_Y) * ((float)y - TEST_HL_Y);
    float df = ((float)x - TEST_HL_FULL_X) * ((float)x - TEST_HL_FULL_X) +
               ((float)y - TEST_HL_Y) * ((float)y - TEST_HL_Y);
    float gp = 1.4f * expf(-dp / (2.0f * 22.0f * 22.0f));
    float gf = 4.0f * expf(-df / (2.0f * 18.0f * 18.0f));

    if (gf >= 0.05f && gf >= gp) {
        for (int c = 0; c < 3; c++)
            rgb[c] = test_hl_full[c] * gf;
    } else if (gp >= 0.05f) {
        for (int c = 0; c < 3; c++)
            rgb[c] = test_hl_part[c] * gp;
    } else {
        rgb[0] = 0.02f + 0.02f * fx;
        rgb[1] = 0.02f + 0.02f * fy;
        rgb[2] = 0.03f + 0.01f * (fx + fy);
    }
}

/* Inverted to raw codes the way the sensor would record it: everything past
 * the white level saturates there, and the same deterministic +-8 count
 * noise as the other scenes rides on top, plateau included, as the codec's
 * does on a real clip. `scale` dims the whole scene, which with a small
 * enough value leaves nothing clipped at all. */
static void test_hl_gen_mosaic(uint16_t *m, float scale)
{
    for (int y = 0; y < TEST_H; y++) {
        for (int x = 0; x < TEST_W; x++) {
            int s = (y & 1) * 2 + (x & 1);
            int c = (s == 0) ? 0 : (s == 3) ? 2 : 1;
            float rgb[3], u;
            int iv;

            test_hl_truth(x, y, rgb);
            u = CRAW_MIN(rgb[c] * scale / test_wb[c], 1.0f);
            iv = (int)lrintf((TEST_BLACK + u * (TEST_WHITE - TEST_BLACK)) * 65535.0f);
            iv += (int)(test_hash((uint32_t)(y * TEST_W + x) * 2654435761u) % 17u) - 8;
            iv = iv < 0 ? 0 : (iv > 65535 ? 65535 : iv);

            m[y * TEST_W + x] = (uint16_t)iv;
        }
    }
}

/* u of DESIGN.md 18.2 for one sample, as the stage computes it. */
static float test_hl_u(const uint16_t *m, int x, int y)
{
    float raw = m[y * TEST_W + x] * (1.0f / 65535.0f);
    return CRAW_MAX((raw - TEST_BLACK) * (1.0f / (TEST_WHITE - TEST_BLACK)), 0.0f);
}

/* Whether every sample in the 3x3 around (x, y) whose channel is in `mask`
 * (bit c for channel c) satisfies lo <= u < hi. */
static bool test_hl_around(const uint16_t *m, int x, int y, int mask, float lo, float hi)
{
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx, ny = y + dy;
            int s = (ny & 1) * 2 + (nx & 1);
            int c = (s == 0) ? 0 : (s == 3) ? 2 : 1;
            float u;

            if (nx < 0 || ny < 0 || nx >= TEST_W || ny >= TEST_H || !(mask & (1 << c)))
                continue;
            u = test_hl_u(m, nx, ny);
            if (!(u >= lo && u < hi))
                return false;
        }
    }
    return true;
}

/* The definition on the CPU, DESIGN.md 18.5: what the reference's rebuilt
 * mosaic must be on the highlight scene. */
static int test_hl_definition(const uint16_t *m, const CRAWProfile *identity)
{
    const size_t npix = (size_t)TEST_W * TEST_H;
    float *v = malloc(npix * sizeof(*v));
    double worst_keep = 0.0, worst_part = 0.0, sum[3] = { 0 }, core_min = 1e9;
    int n_part = 0, n_core[3] = { 0 }, n_lower = 0;
    int ret = -1;

    if (!v)
        return -1;
    if (craw_ref_highlights(m, TEST_W, TEST_H, TEST_W, identity, test_wb, v) < 0) {
        fprintf(stderr, "craw_ref_highlights failed\n");
        goto done;
    }

    for (int y = 0; y < TEST_H; y++) {
        for (int x = 0; x < TEST_W; x++) {
            int s = (y & 1) * 2 + (x & 1);
            int c = (s == 0) ? 0 : (s == 3) ? 2 : 1;
            float u = test_hl_u(m, x, y);
            double xb = (double)u * test_wb[c];
            double out = v[y * TEST_W + x];
            float rgb[3];

            /* Below the ramp nothing moves beyond the fp16 store. */
            if (u < 0.92f)
                worst_keep = CRAW_MAX(worst_keep, fabs(out - xb) / CRAW_MAX(xb, 1e-3));

            /* A clipped sample is never lowered. */
            if (u >= 0.92f && out < xb * (1.0 - 1.0 / 1024.0))
                n_lower++;

            /* The first blob's G, clipped with R and B untouched around it:
             * rebuilt to the scene's own G. */
            test_hl_truth(x, y, rgb);
            if (c == 1 && x < 128 && u >= 0.98f && test_hl_around(m, x, y, 5, 0.0f, 0.92f)) {
                worst_part = CRAW_MAX(worst_part, fabs(out - rgb[1]) / rgb[1]);
                n_part++;
            }

            /* The second blob's core, saturated in every channel over the
             * whole 3x3 (the noise leaves the plateau within 1.3e-4 of the
             * white level): the plateau of 18.2, in the blob's colour. */
            if (x >= 128 && test_hl_around(m, x, y, 7, 0.999f, 2.0f)) {
                sum[c] += out;
                n_core[c]++;
                core_min = CRAW_MIN(core_min, out / test_wb[c]);
            }
        }
    }

    {
        double r = sum[0] / n_core[0], g = sum[1] / n_core[1], b = sum[2] / n_core[2];
        /* The lowest level at which the blob's colour clears every clip. */
        double want = CRAW_MAX(CRAW_MAX(test_wb[0] / test_hl_full[0], 1.0),
                               test_wb[2] / test_hl_full[2]);

        printf("  highlights definition: unclipped kept to %.1e, %d sites of G-only clip "
               "rebuilt to %.2f%% of the truth, blown core (%d/%d/%d sites) R/G %.4f "
               "(want %.2f), B/G %.4f (want %.2f), G %.4f (want %.4f), lowest %.4f of "
               "its clip, %d samples lowered\n",
               worst_keep, n_part, 100.0 * worst_part, n_core[0], n_core[1], n_core[2],
               r / g, test_hl_full[0], b / g, test_hl_full[2], g, want, core_min, n_lower);

        if (!(worst_keep <= 1.0 / 2048.0) || n_part < 200 || !(worst_part <= 0.02) ||
            n_core[1] < 100 || !(fabs(r / g / test_hl_full[0] - 1.0) <= 0.03) ||
            !(fabs(b / g / test_hl_full[2] - 1.0) <= 0.03) || !(fabs(g / want - 1.0) <= 0.03) ||
            !(core_min >= 0.999) || n_lower) {
            fprintf(stderr, "the rebuilt mosaic does not do what DESIGN.md 18.2 says\n");
            goto done;
        }
    }

    ret = 0;

done:
    free(v);
    return ret;
}

/* DESIGN.md 18.5: the definition on the CPU, both ways of switching the
 * stage off, a scene with nothing clipped, and the GPU against the
 * reference on the highlight scene with MHC and RCD. Leaves the highlight
 * scene in `src`. */
static int test_highlights(CRAWTestDevice *td, CRAWContext *ctx, const CRAWProfile *identity,
                           CRAWTestImage *src, CRAWTestImage *dst, float *gpu, float *ref)
{
    static const enum CRAWDebayer methods[] = { CRAW_DEBAYER_MHC, CRAW_DEBAYER_RCD };
    const size_t npix = (size_t)TEST_W * TEST_H;
    CRAWPipelineParams pp;
    CRAWFrameParams fp;
    uint16_t *m = NULL;
    float *plain = NULL;
    double ms = 0.0;
    int ret = -1;

    m = malloc(npix * sizeof(*m));
    plain = malloc(npix * 4 * sizeof(*plain));
    if (!m || !plain)
        goto done;

    test_hl_gen_mosaic(m, 1.0f);
    if (test_hl_definition(m, identity) < 0)
        goto done;

    if (craw_testdev_image_upload(td, src, m, npix * sizeof(*m)) < 0)
        goto done;

    /* Off both ways, bit for bit: the flag on a frame of a pipeline built
     * without the stage, and a pipeline built with it on a frame without
     * the flag. Neither dispatches anything the plain chain does not. */
    craw_pipeline_params_init(&pp);
    pp.width = TEST_W;
    pp.height = TEST_H;
    pp.debayer = CRAW_DEBAYER_RCD;
    pp.out_space = CRAW_CS_XYZ_D65;
    craw_frame_params_init(&fp);
    fp.profile = identity;
    if (test_run_config(td, ctx, src, dst, &pp, &fp, plain, NULL, NULL, &ms) < 0)
        goto done;

    for (int off = 0; off < 2; off++) {
        pp.flags = off ? 0 : CRAW_PIPELINE_FLAG_HIGHLIGHT_RECONSTRUCTION;
        fp.flags = off ? CRAW_FRAME_FLAG_HIGHLIGHT_RECONSTRUCTION : 0;
        if (test_run_config(td, ctx, src, dst, &pp, &fp, gpu, NULL, NULL, &ms) < 0)
            goto done;
        if (memcmp(gpu, plain, npix * 4 * sizeof(*gpu))) {
            fprintf(stderr, "highlights %s changed the frame\n",
                    off ? "flagged on a pipeline without the stage" : "unflagged");
            goto done;
        }
    }
    printf("  highlights off both ways: bit for bit the plain chain\n");

    for (size_t mi = 0; mi < CRAW_ARRAY_ELEMS(methods); mi++) {
        enum CRAWDebayer method = methods[mi];
        const char *mname = test_method_name(method);
        double tol = method == CRAW_DEBAYER_RCD ? 1.5e-2 : 1e-2;
        double err, cross;

        craw_pipeline_params_init(&pp);
        pp.width = TEST_W;
        pp.height = TEST_H;
        pp.debayer = method;
        pp.out_space = CRAW_CS_XYZ_D65;
        pp.flags = CRAW_PIPELINE_FLAG_HIGHLIGHT_RECONSTRUCTION;
        craw_frame_params_init(&fp);
        fp.profile = identity;
        fp.flags = CRAW_FRAME_FLAG_HIGHLIGHT_RECONSTRUCTION;

        if (test_run_config(td, ctx, src, dst, &pp, &fp, gpu, NULL, NULL, &ms) < 0)
            goto done;
        if (craw_ref_process(m, TEST_W, TEST_H, TEST_W, &pp, &fp, ref) < 0) {
            fprintf(stderr, "highlights: craw_ref_process failed\n");
            goto done;
        }
        err = test_max_rel(gpu, ref, npix);

        /* How far the stage moved the picture: G alone goes from the clip
         * at 1.0 to 1.4 at the first blob's peak. */
        fp.flags = 0;
        if (craw_ref_process(m, TEST_W, TEST_H, TEST_W, &pp, &fp, plain) < 0)
            goto done;
        cross = test_max_abs(gpu, plain, npix);

        printf("  highlights %-8s: max rel %.3e (tol %.1e), max abs %.3e, %.3f against "
               "the clipped chain, %.3f ms\n", mname, err, tol, test_max_abs(gpu, ref, npix),
               cross, ms);
        if (!(err <= tol)) {
            fprintf(stderr, "highlights on %s differ by %.3e relative, tolerance %.1e\n",
                    mname, err, tol);
            goto done;
        }
        if (!(cross > 0.3)) {
            fprintf(stderr, "a rebuilt %s frame is only %.3f from the clipped one; the pass "
                    "cannot have run\n", mname, cross);
            goto done;
        }
    }

    /* Nothing clipped: the stage is the fp16 rounding of the mosaic, which
     * the debayer's own bound of DESIGN.md 11 covers. */
    test_hl_gen_mosaic(m, 0.2f);
    if (craw_testdev_image_upload(td, src, m, npix * sizeof(*m)) < 0)
        goto done;

    for (size_t mi = 0; mi < CRAW_ARRAY_ELEMS(methods); mi++) {
        enum CRAWDebayer method = methods[mi];
        double tol = method == CRAW_DEBAYER_RCD ? 5e-3 : 2e-3;
        double err;

        craw_pipeline_params_init(&pp);
        pp.width = TEST_W;
        pp.height = TEST_H;
        pp.debayer = method;
        pp.out_space = CRAW_CS_XYZ_D65;
        pp.flags = CRAW_PIPELINE_FLAG_HIGHLIGHT_RECONSTRUCTION;
        craw_frame_params_init(&fp);
        fp.profile = identity;

        if (test_run_config(td, ctx, src, dst, &pp, &fp, plain, NULL, NULL, &ms) < 0)
            goto done;
        fp.flags = CRAW_FRAME_FLAG_HIGHLIGHT_RECONSTRUCTION;
        if (test_run_config(td, ctx, src, dst, &pp, &fp, gpu, NULL, NULL, &ms) < 0)
            goto done;

        err = test_max_abs(gpu, plain, npix);
        printf("  highlights %-8s with nothing clipped: max abs %.3e against the plain "
               "chain (tol %.1e)\n", test_method_name(method), err, tol);
        if (!(err <= tol)) {
            fprintf(stderr, "highlights moved an unclipped %s frame by %.3e\n",
                    test_method_name(method), err);
            goto done;
        }
    }

    ret = 0;

done:
    free(plain);
    free(m);
    return ret;
}

/* ---------------------------------------------------------------- defringe */

/* A smooth box: 1 inside [x0, x1] x [y0, y1], 0 outside, with a linear ramp
 * a pixel and a half wide on every side. */
static float test_df_box(float x, float y, float x0, float x1, float y0, float y1)
{
    float ax = CRAW_CLAMP(CRAW_MIN(x - x0, x1 - x) / 1.5f + 0.5f, 0.0f, 1.0f);
    float ay = CRAW_CLAMP(CRAW_MIN(y - y0, y1 - y) / 1.5f + 0.5f, 0.0f, 1.0f);
    return ax * ay;
}

/* The defringe scene of DESIGN.md 19.5, in balanced camera RGB: a bright
 * neutral bar whose R is imaged a pixel right and down and whose B a pixel
 * left and up, the way a lens misregisters them; a saturated red bar on
 * black, a hard edge whose colour is real; and an orange and a grey half of
 * equal level, a colour edge with no edge in level at all. */
static void test_df_truth(int x, int y, float rgb[3])
{
    static const float shift[3][2] = { { 1.0f, 1.0f }, { 0.0f, 0.0f }, { -1.0f, -1.0f } };
    const float bg = 0.03f;

    for (int c = 0; c < 3; c++) {
        float b = test_df_box((float)x - shift[c][0], (float)y - shift[c][1], 40, 110, 30, 70);
        rgb[c] = bg + (0.9f - bg) * b;
    }

    if (x >= 136 && x <= 210 && y >= 20 && y <= 80) {
        static const float red[3] = { 0.9f, 0.08f, 0.06f };
        float b = test_df_box((float)x, (float)y, 140, 200, 30, 70);
        for (int c = 0; c < 3; c++)
            rgb[c] = 0.01f + (red[c] - 0.01f) * b;
    }

    if (y >= 100 && y <= 150 && x >= 20 && x <= 240) {
        static const float orange[3] = { 0.8f, 0.4f, 0.1f }, grey[3] = { 0.425f, 0.425f, 0.425f };
        for (int c = 0; c < 3; c++)
            rgb[c] = x < 128 ? orange[c] : grey[c];
    }
}

static void test_df_gen_mosaic(uint16_t *m)
{
    for (int y = 0; y < TEST_H; y++) {
        for (int x = 0; x < TEST_W; x++) {
            int s = (y & 1) * 2 + (x & 1);
            int c = (s == 0) ? 0 : (s == 3) ? 2 : 1;
            float rgb[3];
            int iv;

            test_df_truth(x, y, rgb);
            iv = (int)lrintf((TEST_BLACK + rgb[c] / test_wb[c] * (TEST_WHITE - TEST_BLACK)) * 65535.0f);
            iv += (int)(test_hash((uint32_t)(y * TEST_W + x) * 2654435761u) % 17u) - 8;
            m[y * TEST_W + x] = (uint16_t)CRAW_CLAMP(iv, 0, 65535);
        }
    }
}

static int test_cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double test_df_chroma(const float *px)
{
    double r = log(CRAW_MAX(px[0], 1e-3) / CRAW_MAX(px[1], 1e-3));
    double b = log(CRAW_MAX(px[2], 1e-3) / CRAW_MAX(px[1], 1e-3));
    return hypot(r, b);
}

/* The definition on the CPU, DESIGN.md 19.5. */
static int test_df_definition(const uint16_t *m, const CRAWProfile *identity)
{
    const size_t npix = (size_t)TEST_W * TEST_H;
    float *before = malloc(npix * 4 * sizeof(*before)), *after = malloc(npix * 4 * sizeof(*after));
    double fringe_before = 0.0, *fringe_after = NULL, red_before = 0.0, red_after = 0.0;
    int nf = 0, nred = 0, moved_orange = 0, moved_far = 0, tinted = 0;
    int ret = -1;

    fringe_after = malloc(npix * sizeof(*fringe_after));
    if (!before || !after || !fringe_after)
        goto done;
    if (craw_ref_debayer(m, TEST_W, TEST_H, TEST_W, identity, CRAW_DEBAYER_RCD, before) < 0)
        goto done;
    memcpy(after, before, npix * 4 * sizeof(*after));
    if (craw_ref_defringe(after, TEST_W, TEST_H, 1.0f) < 0) {
        fprintf(stderr, "craw_ref_defringe failed\n");
        goto done;
    }

    for (int y = 0; y < TEST_H; y++) {
        for (int x = 0; x < TEST_W; x++) {
            const float *b = &before[((size_t)y * TEST_W + x) * 4];
            const float *a = &after[((size_t)y * TEST_W + x) * 4];
            bool moved = memcmp(a, b, 4 * sizeof(*a)) != 0;
            float yb = (b[0] + 2.0f * b[1] + b[2]) * 0.25f;
            /* Distance to the fringed bar's outline. */
            int dx = CRAW_MIN(abs(x - 40), abs(x - 110)), dy = CRAW_MIN(abs(y - 30), abs(y - 70));
            bool in_x = x >= 36 && x <= 114, in_y = y >= 26 && y <= 74;
            bool near_bar = (in_x && in_y) && (dx <= 3 || dy <= 3);

            if (near_bar && yb > 0.1f) {
                fringe_before = CRAW_MAX(fringe_before, test_df_chroma(b));
                fringe_after[nf++] = test_df_chroma(a);
            }

            /* The red bar's own edge pixels keep their colour. */
            if (x >= 138 && x <= 202 && y >= 28 && y <= 72 && yb > 0.1f &&
                (CRAW_MIN(abs(x - 140), abs(x - 200)) <= 2 || CRAW_MIN(abs(y - 30), abs(y - 70)) <= 2)) {
                red_before += log(CRAW_MAX(b[0], 1e-3) / CRAW_MAX(b[1], 1e-3));
                red_after += log(CRAW_MAX(a[0], 1e-3) / CRAW_MAX(a[1], 1e-3));
                nred++;
            }

            if (y >= 104 && y <= 146 && x >= 24 && x <= 236 && moved)
                moved_orange++;

            /* Far from every edge nothing moves. */
            if ((y <= 18 || (x >= 216 && y <= 90)) && moved)
                moved_far++;

            /* No neutral pixel of the dark background takes a colour. */
            if (yb < 0.06f && !(y >= 96 && y <= 154) && test_df_chroma(b) < 0.05 &&
                test_df_chroma(a) > 0.08)
                tinted++;
        }
    }

    {
        double *v = fringe_after, p99;
        qsort(v, (size_t)nf, sizeof(*v), test_cmp_double);
        p99 = v[(size_t)(0.99 * (nf - 1))];
        red_before /= nred;
        red_after /= nred;

        printf("  defringe definition: fringed bar's rim %d px, colour up to %.3f before, "
               "99th percentile %.3f after (bound 0.12); red bar's rim ln R/G %.3f -> %.3f; "
               "orange/grey edge %d px moved, far from every edge %d, neutral dark pixels "
               "tinted %d\n", nf, fringe_before, p99, red_before, red_after, moved_orange,
               moved_far, tinted);

        if (!(fringe_before > 0.3) || !(p99 <= 0.12) || !(red_after >= red_before - 0.05) ||
            moved_orange || moved_far || tinted) {
            fprintf(stderr, "the defringe does not do what DESIGN.md 19.2 says\n");
            goto done;
        }
    }
    ret = 0;

done:
    free(fringe_after);
    free(before);
    free(after);
    return ret;
}

/* DESIGN.md 19.5: the definition on the CPU, both ways of switching the
 * stage off, the GPU against the reference with MHC and RCD, and a
 * strength above one refused. Leaves the defringe scene in `src`. */
static int test_defringe(CRAWTestDevice *td, CRAWContext *ctx, const CRAWProfile *identity,
                         CRAWTestImage *src, CRAWTestImage *dst, float *gpu, float *ref)
{
    static const enum CRAWDebayer methods[] = { CRAW_DEBAYER_MHC, CRAW_DEBAYER_RCD };
    const size_t npix = (size_t)TEST_W * TEST_H;
    CRAWPipelineParams pp;
    CRAWFrameParams fp;
    uint16_t *m = malloc(npix * sizeof(*m));
    float *plain = malloc(npix * 4 * sizeof(*plain));
    double ms = 0.0;
    int ret = -1;

    if (!m || !plain)
        goto done;

    test_df_gen_mosaic(m);
    if (test_df_definition(m, identity) < 0)
        goto done;
    if (craw_testdev_image_upload(td, src, m, npix * sizeof(*m)) < 0)
        goto done;

    /* Off both ways, bit for bit. */
    craw_pipeline_params_init(&pp);
    pp.width = TEST_W;
    pp.height = TEST_H;
    pp.debayer = CRAW_DEBAYER_RCD;
    craw_frame_params_init(&fp);
    fp.profile = identity;
    if (test_run_config(td, ctx, src, dst, &pp, &fp, plain, NULL, NULL, &ms) < 0)
        goto done;
    for (int off = 0; off < 2; off++) {
        pp.flags = off ? 0 : CRAW_PIPELINE_FLAG_DEFRINGE;
        fp.defringe_strength = off ? 1.0f : 0.0f;
        if (test_run_config(td, ctx, src, dst, &pp, &fp, gpu, NULL, NULL, &ms) < 0)
            goto done;
        if (memcmp(gpu, plain, npix * 4 * sizeof(*gpu))) {
            fprintf(stderr, "defringe %s changed the frame\n",
                    off ? "on a pipeline without the stage" : "at strength 0");
            goto done;
        }
    }
    printf("  defringe off both ways: bit for bit the plain chain\n");

    /* The stage against the reference run on the GPU's own undefringed
     * frame: with identity colour the destination is the intermediate
     * itself. Against the whole reference chain the comparison would be of
     * the debayers: the side search is an argmin over fp16 E, and the
     * debayer's own disagreement of a millesimal moves it at a corner. */
    for (size_t mi = 0; mi < CRAW_ARRAY_ELEMS(methods); mi++) {
        enum CRAWDebayer method = methods[mi];
        double err, cross, chain;

        craw_pipeline_params_init(&pp);
        pp.width = TEST_W;
        pp.height = TEST_H;
        pp.debayer = method;
        pp.flags = CRAW_PIPELINE_FLAG_DEFRINGE;
        craw_frame_params_init(&fp);
        fp.profile = identity;

        if (test_run_config(td, ctx, src, dst, &pp, &fp, plain, NULL, NULL, &ms) < 0)
            goto done;
        fp.defringe_strength = 1.0f;
        if (test_run_config(td, ctx, src, dst, &pp, &fp, gpu, NULL, NULL, &ms) < 0)
            goto done;
        if (craw_ref_process(m, TEST_W, TEST_H, TEST_W, &pp, &fp, ref) < 0) {
            fprintf(stderr, "defringe: craw_ref_process failed\n");
            goto done;
        }
        chain = test_max_abs(gpu, ref, npix);
        memcpy(ref, plain, npix * 4 * sizeof(*ref));
        if (craw_ref_defringe(ref, TEST_W, TEST_H, 1.0f) < 0) {
            fprintf(stderr, "defringe: craw_ref_defringe failed\n");
            goto done;
        }
        err = test_max_abs(gpu, ref, npix);
        cross = test_max_abs(gpu, plain, npix);

        printf("  defringe %-8s: max abs %.3e (tol %.1e) on the same input, %.3e against the "
               "reference chain, %.3f against the plain chain, %.3f ms\n",
               test_method_name(method), err, TEST_DF_TOL, chain, cross, ms);
        if (!(err <= TEST_DF_TOL)) {
            fprintf(stderr, "defringe on %s differs by %.3e, tolerance %.1e\n",
                    test_method_name(method), err, TEST_DF_TOL);
            goto done;
        }
        if (!(cross > 0.05)) {
            fprintf(stderr, "a defringed %s frame is only %.3f from the plain one; the pass "
                    "cannot have run\n", test_method_name(method), cross);
            goto done;
        }
    }

    /* More than all the way is an error, not a clamp. */
    {
        CRAWPipeline *pipe = NULL;
        struct test_run run;
        int rec;

        craw_pipeline_params_init(&pp);
        pp.width = TEST_W;
        pp.height = TEST_H;
        pp.flags = CRAW_PIPELINE_FLAG_DEFRINGE;
        craw_frame_params_init(&fp);
        fp.profile = identity;
        fp.defringe_strength = 1.5f;
        if (craw_pipeline_create(ctx, &pp, &pipe) < 0)
            goto done;
        run = (struct test_run) { .pipe = pipe, .src = src, .dst = dst, .fp = &fp };
        rec = craw_testdev_run(td, test_record, &run);
        craw_pipeline_free(&pipe);
        if (rec != CRAW_ERROR(EINVAL)) {
            fprintf(stderr, "a defringe strength of 1.5 returned %d, wanted %d\n", rec,
                    CRAW_ERROR(EINVAL));
            goto done;
        }
    }

    ret = 0;

done:
    free(plain);
    free(m);
    return ret;
}

/* ----------------------------------------------------------- real frame */

static int test_real_frame(CRAWTestDevice *td, CRAWContext *ctx, const CRAWProfile *profile,
                           const CRAWProfile *identity)
{
    const char *path = getenv("CRAW_TEST_FRAME");
    const size_t npix = (size_t)FRAME_W * FRAME_H;
    char fallback[1024];
    uint16_t *mosaic = NULL;
    float *gpu = NULL, *ref = NULL;
    CRAWPipelineParams pp;
    CRAWFrameParams fp;
    CRAWTestImage src = { 0 }, dst = { 0 };
    bool have_src = false, have_dst = false;
    double ms = 0.0, plain_abs = 0.0;
    FILE *f;
    int ret = 0;

    if (!path) {
        /* DESIGN.md 11 also allows a sample next to the test sources. */
        snprintf(fallback, sizeof(fallback), "%s/../samples/frame.raw", TEST_SOURCE_DIR);
        f = fopen(fallback, "rb");
        if (!f)
            return 0;
        path = fallback;
    } else {
        f = fopen(path, "rb");
        if (!f) {
            fprintf(stderr, "CRAW_TEST_FRAME: cannot open %s\n", path);
            return 0;
        }
    }

    mosaic = malloc(npix * sizeof(*mosaic));
    if (!mosaic) {
        fclose(f);
        return 0;
    }

    if (fread(mosaic, sizeof(*mosaic), npix, f) != npix) {
        fprintf(stderr, "%s: not a %dx%d uint16 frame, skipping\n", path, FRAME_W, FRAME_H);
        fclose(f);
        free(mosaic);
        return 0;
    }
    fclose(f);

    printf("real frame %s (%dx%d)\n", path, FRAME_W, FRAME_H);

    craw_pipeline_params_init(&pp);
    pp.width = FRAME_W;
    pp.height = FRAME_H;
    pp.debayer = CRAW_DEBAYER_RCD;
    pp.out_space = CRAW_CS_XYZ_D65;

    craw_frame_params_init(&fp);
    fp.profile = profile;

    ret = craw_testdev_image_create(td, &src, FRAME_W, FRAME_H, VK_FORMAT_R16_UNORM);
    if (ret < 0)
        goto done;
    have_src = true;

    ret = craw_testdev_image_create(td, &dst, FRAME_W, FRAME_H, VK_FORMAT_R16G16B16A16_SFLOAT);
    if (ret < 0)
        goto done;
    have_dst = true;

    ret = craw_testdev_image_upload(td, &src, mosaic, npix * sizeof(*mosaic));
    if (ret < 0)
        goto done;

    gpu = malloc(npix * 4 * sizeof(*gpu));
    if (!gpu) {
        ret = CRAW_ERROR(ENOMEM);
        goto done;
    }

    ret = test_run_config(td, ctx, &src, &dst, &pp, &fp, gpu, NULL, NULL, &ms);
    if (ret < 0)
        goto done;

    ref = malloc(npix * 4 * sizeof(*ref));
    if (!ref) {
        ret = CRAW_ERROR(ENOMEM);
        goto done;
    }

    ret = craw_ref_process(mosaic, FRAME_W, FRAME_H, FRAME_W, &pp, &fp, ref);
    if (ret < 0)
        goto done;

    plain_abs = test_max_abs(gpu, ref, npix);
    printf("  rcd on %dx%d: %.2f ms, max abs %.3e, max rel %.3e\n", FRAME_W, FRAME_H, ms,
           plain_abs, test_max_rel(gpu, ref, npix));

    /* The same frame through the denoise pass, with the model fitted on
     * the frame itself so that the stage is handed the noise it really
     * carries. `cross` is against the undenoised reference still in `ref`,
     * i.e. how far the pass moved the picture; it has to be far larger
     * than the agreement between the two implementations of it. */
    {
        CRAWNoiseEstimate est;
        double cross, err_abs, err_rel;

        ret = craw_noise_estimate(mosaic, FRAME_W, FRAME_H, FRAME_W, profile, NULL, &est);
        if (ret < 0) {
            fprintf(stderr, "craw_noise_estimate failed: %d\n", ret);
            goto done;
        }
        printf("  noise model: a %.4e b %.4e, %u bins, fit error %.1f%%, "
               "read noise %.2f code values\n", (double)est.model.a, (double)est.model.b,
               est.nb_bins, 100.0 * (double)est.fit_rms, (double)est.read_noise_lsb);

        pp.flags = CRAW_PIPELINE_FLAG_DENOISE;
        fp.denoise_strength = 1.0f;
        fp.noise_override = est.model;

        ret = test_run_config(td, ctx, &src, &dst, &pp, &fp, gpu, NULL, NULL, &ms);
        if (ret < 0)
            goto done;

        cross = test_max_rel(gpu, ref, npix);

        ret = craw_ref_process(mosaic, FRAME_W, FRAME_H, FRAME_W, &pp, &fp, ref);
        if (ret < 0)
            goto done;

        err_abs = test_max_abs(gpu, ref, npix);
        err_rel = test_max_rel(gpu, ref, npix);
        printf("  rcd + denoise on %dx%d: %.2f ms, max rel %.3e (tol %.1e), max abs %.3e, "
               "%.3e relative against the undenoised chain\n", FRAME_W, FRAME_H, ms,
               err_rel, TEST_DENOISE_REAL_TOL, err_abs, cross);

        if (!(err_rel <= TEST_DENOISE_REAL_TOL)) {
            fprintf(stderr, "denoise on the real frame differs by %.3e relative, "
                    "tolerance %.1e\n", err_rel, TEST_DENOISE_REAL_TOL);
            ret = -1;
            goto done;
        }
        if (!(cross > 5.0 * err_rel)) {
            fprintf(stderr, "the denoised frame is only %.3e relative from the undenoised "
                    "one; the pass cannot have run\n", cross);
            ret = -1;
            goto done;
        }
    }

    /* And with three passes of false colour suppression on the plain RCD
     * chain, DESIGN.md 17.5. `ref` holds the denoised reference by now, so
     * the plain one is rebuilt for the cross check. */
    {
        double cross, err_abs, err_rel;

        pp.flags = CRAW_PIPELINE_FLAG_FALSE_COLOR;
        fp.denoise_strength = 0.0f;
        fp.false_color_passes = CRAW_FALSE_COLOR_MAX_PASSES;

        ret = test_run_config(td, ctx, &src, &dst, &pp, &fp, gpu, NULL, NULL, &ms);
        if (ret < 0)
            goto done;

        fp.false_color_passes = 0;
        ret = craw_ref_process(mosaic, FRAME_W, FRAME_H, FRAME_W, &pp, &fp, ref);
        if (ret < 0)
            goto done;
        cross = test_max_rel(gpu, ref, npix);

        fp.false_color_passes = CRAW_FALSE_COLOR_MAX_PASSES;
        ret = craw_ref_process(mosaic, FRAME_W, FRAME_H, FRAME_W, &pp, &fp, ref);
        if (ret < 0)
            goto done;

        err_abs = test_max_abs(gpu, ref, npix);
        err_rel = test_max_rel(gpu, ref, npix);
        printf("  rcd + false colour x%u on %dx%d: %.2f ms, max abs %.3e (plain %.3e, tol "
               "%.3e), max rel %.3e, %.3e relative against the unfiltered chain\n",
               (unsigned)CRAW_FALSE_COLOR_MAX_PASSES, FRAME_W, FRAME_H, ms, err_abs,
               plain_abs, TEST_FC_REAL_FACTOR * plain_abs, err_rel, cross);

        /* Absolute, against the plain chain's own disagreement: a median
         * moves by no more than its inputs do, so the stage cannot add to
         * the absolute error beyond what carrying a neighbour's error and
         * one more fp16 store costs. The relative bound of section 11 is
         * not the right yardstick here: the difference selected for a dark
         * pixel beside a clipped lamp is the lamp's, error included, and
         * the same absolute error is several times larger relative to the
         * dark pixel. DESIGN.md 17.5. */
        if (!(err_abs <= TEST_FC_REAL_FACTOR * plain_abs)) {
            fprintf(stderr, "false colour on the real frame differs by %.3e absolute, "
                    "tolerance %.3e (%.1f times the plain chain's %.3e)\n", err_abs,
                    TEST_FC_REAL_FACTOR * plain_abs, TEST_FC_REAL_FACTOR, plain_abs);
            ret = -1;
            goto done;
        }
        if (!(cross > 5.0 * err_rel)) {
            fprintf(stderr, "the filtered frame is only %.3e relative from the unfiltered "
                    "one; the pass cannot have run\n", cross);
            ret = -1;
            goto done;
        }
    }

    /* And the chroma denoise at full strength on top of two false colour
     * passes, the pairing DESIGN.md 14.4 describes. `ref` holds the false
     * colour reference from above, so the plain one is rebuilt first. */
    {
        double cross, err_abs, err_rel;

        pp.flags = CRAW_PIPELINE_FLAG_FALSE_COLOR | CRAW_PIPELINE_FLAG_CHROMA_DENOISE;
        fp.false_color_passes = 2;
        fp.denoise_chroma_strength = 1.0f;

        ret = test_run_config(td, ctx, &src, &dst, &pp, &fp, gpu, NULL, NULL, &ms);
        if (ret < 0)
            goto done;

        fp.false_color_passes = 0;
        fp.denoise_chroma_strength = 0.0f;
        ret = craw_ref_process(mosaic, FRAME_W, FRAME_H, FRAME_W, &pp, &fp, ref);
        if (ret < 0)
            goto done;
        cross = test_max_rel(gpu, ref, npix);

        fp.false_color_passes = 2;
        fp.denoise_chroma_strength = 1.0f;
        ret = craw_ref_process(mosaic, FRAME_W, FRAME_H, FRAME_W, &pp, &fp, ref);
        if (ret < 0)
            goto done;

        err_abs = test_max_abs(gpu, ref, npix);
        err_rel = test_max_rel(gpu, ref, npix);
        printf("  rcd + false colour x2 + chroma on %dx%d: %.2f ms, max abs %.3e (plain %.3e, "
               "tol %.3e), max rel %.3e, %.3e relative against the plain chain\n",
               FRAME_W, FRAME_H, ms, err_abs, plain_abs, TEST_FC_REAL_FACTOR * plain_abs,
               err_rel, cross);

        if (!(err_abs <= TEST_FC_REAL_FACTOR * plain_abs)) {
            fprintf(stderr, "chroma denoise on the real frame differs by %.3e absolute, "
                    "tolerance %.3e\n", err_abs, TEST_FC_REAL_FACTOR * plain_abs);
            ret = -1;
            goto done;
        }
        if (!(cross > 5.0 * err_rel)) {
            fprintf(stderr, "the chroma denoised frame is only %.3e relative from the plain "
                    "one; the pass cannot have run\n", cross);
            ret = -1;
            goto done;
        }
    }

    /* And the highlight reconstruction on the plain RCD chain, DESIGN.md
     * 18.5, timed. The relative bound of section 11: the rebuilt highlights
     * sit above the old clip, where fp16 is worth more in absolute terms. */
    {
        double cross, err_abs, err_rel;

        pp.flags = CRAW_PIPELINE_FLAG_HIGHLIGHT_RECONSTRUCTION;
        fp.false_color_passes = 0;
        fp.denoise_chroma_strength = 0.0f;
        fp.flags = CRAW_FRAME_FLAG_HIGHLIGHT_RECONSTRUCTION;

        ret = test_run_config(td, ctx, &src, &dst, &pp, &fp, gpu, NULL, NULL, &ms);
        if (ret < 0)
            goto done;

        fp.flags = 0;
        ret = craw_ref_process(mosaic, FRAME_W, FRAME_H, FRAME_W, &pp, &fp, ref);
        if (ret < 0)
            goto done;
        cross = test_max_rel(gpu, ref, npix);

        fp.flags = CRAW_FRAME_FLAG_HIGHLIGHT_RECONSTRUCTION;
        ret = craw_ref_process(mosaic, FRAME_W, FRAME_H, FRAME_W, &pp, &fp, ref);
        if (ret < 0)
            goto done;

        err_abs = test_max_abs(gpu, ref, npix);
        err_rel = test_max_rel(gpu, ref, npix);
        printf("  rcd + highlights on %dx%d: %.2f ms, max rel %.3e (tol %.1e), max abs %.3e, "
               "%.3e relative against the clipped chain\n", FRAME_W, FRAME_H, ms, err_rel,
               TEST_DENOISE_REAL_TOL, err_abs, cross);

        if (!(err_rel <= TEST_DENOISE_REAL_TOL)) {
            fprintf(stderr, "highlights on the real frame differ by %.3e relative, "
                    "tolerance %.1e\n", err_rel, TEST_DENOISE_REAL_TOL);
            ret = -1;
            goto done;
        }
    }

    /* And the defringe stage, DESIGN.md 19.5: timed on the plain RCD
     * chain, then held, as on the synthetic frame, against the reference
     * run on the GPU's own undefringed intermediate. `ref` takes that
     * intermediate and is defringed in place. A real frame has edges at
     * every angle and corners everywhere, so a few side searches go the
     * other way: held to a count of pixels past the tolerance, not to the
     * largest difference. */
    {
        double err, cross, total_ms;
        size_t moved = 0, beyond = 0;

        pp.flags = CRAW_PIPELINE_FLAG_DEFRINGE;
        fp.flags = 0;
        fp.defringe_strength = 1.0f;
        ret = test_run_config(td, ctx, &src, &dst, &pp, &fp, gpu, NULL, NULL, &total_ms);
        if (ret < 0)
            goto done;

        fp.profile = identity;
        fp.defringe_strength = 0.0f;
        ret = test_run_config(td, ctx, &src, &dst, &pp, &fp, ref, NULL, NULL, &ms);
        if (ret < 0)
            goto done;
        fp.defringe_strength = 1.0f;
        ret = test_run_config(td, ctx, &src, &dst, &pp, &fp, gpu, NULL, NULL, &ms);
        if (ret < 0)
            goto done;
        fp.profile = profile;

        cross = test_max_abs(gpu, ref, npix);
        for (size_t i = 0; i < npix; i++)
            moved += memcmp(&gpu[i * 4], &ref[i * 4], 3 * sizeof(*gpu)) != 0;

        ret = craw_ref_defringe(ref, FRAME_W, FRAME_H, 1.0f);
        if (ret < 0)
            goto done;
        err = test_max_abs(gpu, ref, npix);
        for (size_t i = 0; i < npix * 4; i += 4)
            beyond += fabsf(gpu[i] - ref[i]) > TEST_DF_TOL || fabsf(gpu[i + 1] - ref[i + 1]) > TEST_DF_TOL ||
                      fabsf(gpu[i + 2] - ref[i + 2]) > TEST_DF_TOL;

        printf("  rcd + defringe on %dx%d: %.2f ms, %zu px moved (%.2f%%) by up to %.3e; on "
               "the same intermediate %zu px past %.0e (at most %zu), max abs %.3e\n", FRAME_W,
               FRAME_H, total_ms, moved, 100.0 * (double)moved / (double)npix, cross, beyond,
               TEST_DF_TOL, moved / 1000, err);

        if (beyond > moved / 1000 || !moved) {
            fprintf(stderr, "defringe on the real frame differs at %zu px by more than %.0e, "
                    "of %zu moved\n", beyond, TEST_DF_TOL, moved);
            ret = -1;
            goto done;
        }
    }

done:
    if (ret < 0)
        fprintf(stderr, "real frame run failed: %d\n", ret);
    if (have_dst)
        craw_testdev_image_free(td, &dst);
    if (have_src)
        craw_testdev_image_free(td, &src);
    free(ref);
    free(gpu);
    free(mosaic);
    return ret;
}

/* ---------------------------------------------------------------- main */

int main(void)
{
    static const enum CRAWColorSpace spaces[] = { CRAW_CS_XYZ_D65, CRAW_CS_BT2020_LINEAR };
    const size_t npix = (size_t)TEST_W * TEST_H;
    const uint32_t nbins = CRAW_AWB_HIST_BINS * CRAW_AWB_HIST_BINS;
    bool verbose = getenv("CRAW_TEST_VERBOSE") != NULL;
    CRAWTestDevice td = { 0 };
    CRAWContext *ctx = NULL;
    CRAWProfile profile, identity;
    CRAWTestImage src = { 0 }, dst = { 0 };
    bool have_src = false, have_dst = false, have_dev = false;
    uint16_t *mosaic = NULL;
    float *gpu = NULL, *ref = NULL;
    uint32_t *gpu_count = NULL, *gpu_weight = NULL, *ref_count = NULL, *ref_weight = NULL;
    uint32_t *ref_bright = NULL;
    int32_t *hist_scratch = NULL;
    int ret, status = 1;

    ret = craw_testdev_open(&td, NULL, verbose);
    if (ret == CRAW_ERROR(ENODEV)) {
        printf("no suitable Vulkan device, skipping\n");
        return 77;
    }
    if (ret < 0) {
        fprintf(stderr, "craw_testdev_open failed: %d\n", ret);
        return 1;
    }
    have_dev = true;
    printf("device: %s%s\n", td.name, td.lavapipe ? " (lavapipe)" : "");

    ret = craw_context_create(&ctx, &td.init, NULL);
    if (ret < 0) {
        fprintf(stderr, "craw_context_create failed: %d\n", ret);
        goto done;
    }

    test_profile_init(&profile, false);
    test_profile_init(&identity, true);

    if (test_validation(ctx) < 0)
        goto done;

    mosaic = malloc(npix * sizeof(*mosaic));
    gpu = malloc(npix * 4 * sizeof(*gpu));
    ref = malloc(npix * 4 * sizeof(*ref));
    gpu_count = malloc(nbins * sizeof(*gpu_count));
    gpu_weight = malloc(nbins * sizeof(*gpu_weight));
    ref_count = malloc(nbins * sizeof(*ref_count));
    ref_weight = malloc(nbins * sizeof(*ref_weight));
    ref_bright = malloc(nbins * sizeof(*ref_bright));
    hist_scratch = malloc(nbins * sizeof(*hist_scratch));
    if (!mosaic || !gpu || !ref || !gpu_count || !gpu_weight || !ref_count || !ref_weight ||
        !hist_scratch) {
        fprintf(stderr, "out of memory\n");
        goto done;
    }

    test_gen_mosaic(mosaic);

    ret = craw_testdev_image_create(&td, &src, TEST_W, TEST_H, VK_FORMAT_R16_UNORM);
    if (ret < 0) {
        fprintf(stderr, "source image creation failed: %d\n", ret);
        goto done;
    }
    have_src = true;

    ret = craw_testdev_image_create(&td, &dst, TEST_W, TEST_H, VK_FORMAT_R16G16B16A16_SFLOAT);
    if (ret < 0) {
        fprintf(stderr, "destination image creation failed: %d\n", ret);
        goto done;
    }
    have_dst = true;

    ret = craw_testdev_image_upload(&td, &src, mosaic, npix * sizeof(*mosaic));
    if (ret < 0) {
        fprintf(stderr, "upload failed: %d\n", ret);
        goto done;
    }

    for (int mi = 0; mi < CRAW_DEBAYER_NB; mi++) {
        enum CRAWDebayer method = (enum CRAWDebayer)mi;
        const char *mname = test_method_name(method);
        double tol_abs = method == CRAW_DEBAYER_RCD ? 5e-3 : 2e-3;
        double tol_rel = method == CRAW_DEBAYER_RCD ? 1.5e-2 : 1e-2;
        CRAWPipelineParams pp;
        CRAWFrameParams fp;
        double ms = 0.0, err;

        /* 1. Identity colour: the destination is the intermediate itself. */
        craw_pipeline_params_init(&pp);
        pp.width = TEST_W;
        pp.height = TEST_H;
        pp.debayer = method;
        pp.out_space = CRAW_CS_XYZ_D65;
        pp.flags = CRAW_PIPELINE_FLAG_AWB_HISTOGRAM;

        craw_frame_params_init(&fp);
        fp.profile = &identity;

        ret = craw_ref_debayer(mosaic, TEST_W, TEST_H, TEST_W, &identity, method, ref);
        if (ret < 0) {
            fprintf(stderr, "%s: craw_ref_debayer failed: %d\n", mname, ret);
            goto done;
        }

        ret = test_run_config(&td, ctx, &src, &dst, &pp, &fp, gpu, gpu_count, gpu_weight, &ms);
        if (ret < 0)
            goto done;

        err = test_max_abs(gpu, ref, npix);
        printf("  %-8s intermediate: max abs %.3e (tol %.1e), %.3f ms\n", mname, err, tol_abs, ms);
        if (!(err <= tol_abs)) {
            fprintf(stderr, "%s: camera RGB intermediate differs by %.3e, tolerance %.1e\n",
                    mname, err, tol_abs);
            goto done;
        }

        ret = craw_awb_histogram_from_rgb(ref, TEST_W, TEST_H, (size_t)TEST_W * 4, 4,
                                          identity.as_shot_wb, ref_count, ref_weight,
                                          ref_bright);
        if (ret < 0) {
            fprintf(stderr, "%s: craw_awb_histogram_from_rgb failed: %d\n", mname, ret);
            goto done;
        }
        if (test_check_histogram(mname, gpu_count, gpu_weight, ref_count, ref_weight,
                                 hist_scratch) < 0)
            goto done;

        /* 2. The real profile, both output spaces, as shot and re-balanced. */
        for (size_t si = 0; si < sizeof(spaces) / sizeof(spaces[0]); si++) {
            for (int rebalance = 0; rebalance < 2; rebalance++) {
                craw_pipeline_params_init(&pp);
                pp.width = TEST_W;
                pp.height = TEST_H;
                pp.debayer = method;
                pp.out_space = spaces[si];
                pp.flags = CRAW_PIPELINE_FLAG_AWB_HISTOGRAM;

                craw_frame_params_init(&fp);
                fp.profile = &profile;
                if (rebalance) {
                    /* Exercises the wb ratio, the interpolated matrix, the
                     * exposure offset and the negative clamp at once. */
                    fp.wb_mode = CRAW_WB_REBALANCE;
                    fp.illuminant = (CRAWIlluminant) { .cct = 6500.0f, .tint = 0.02f };
                    fp.exposure_stops = -0.5f;
                    fp.flags = CRAW_FRAME_FLAG_CLAMP_NEGATIVE;
                }

                ret = craw_ref_process(mosaic, TEST_W, TEST_H, TEST_W, &pp, &fp, ref);
                if (ret < 0) {
                    fprintf(stderr, "%s: craw_ref_process failed: %d\n", mname, ret);
                    goto done;
                }

                ret = test_run_config(&td, ctx, &src, &dst, &pp, &fp, gpu, NULL, NULL, &ms);
                if (ret < 0)
                    goto done;

                err = test_max_rel(gpu, ref, npix);
                printf("  %-8s %-7s %-9s: max rel %.3e (tol %.1e), max abs %.3e, %.3f ms\n",
                       mname, test_space_name(spaces[si]), rebalance ? "rebalance" : "as-shot",
                       err, tol_rel, test_max_abs(gpu, ref, npix), ms);
                if (!(err <= tol_rel)) {
                    fprintf(stderr, "%s: %s %s output differs by %.3e relative, tolerance %.1e\n",
                            mname, test_space_name(spaces[si]),
                            rebalance ? "rebalance" : "as-shot", err, tol_rel);
                    goto done;
                }
            }
        }
    }

    if (test_denoise(&td, ctx, &identity, mosaic, &src, &dst, gpu, ref) < 0)
        goto done;

    if (test_false_color(&td, ctx, &identity, mosaic, &src, &dst, gpu, ref) < 0)
        goto done;

    if (test_chroma(&td, ctx, &identity, mosaic, &src, &dst, gpu, ref) < 0)
        goto done;

    if (test_lens(&td, ctx, &identity, mosaic, &src, &dst, gpu, ref) < 0)
        goto done;

    /* Both replace the scene in `src`; test_gamut() uploads its own. */
    if (test_highlights(&td, ctx, &identity, &src, &dst, gpu, ref) < 0)
        goto done;
    if (test_defringe(&td, ctx, &identity, &src, &dst, gpu, ref) < 0)
        goto done;

    if (test_gamut(&td, ctx, &profile, &src, &dst, gpu, ref) < 0)
        goto done;

    if (test_real_frame(&td, ctx, &profile, &identity) < 0)
        goto done;

    printf("test_pipeline: OK\n");
    status = 0;

done:
    if (have_dst)
        craw_testdev_image_free(&td, &dst);
    if (have_src)
        craw_testdev_image_free(&td, &src);
    free(hist_scratch);
    free(ref_weight);
    free(ref_bright);
    free(ref_count);
    free(gpu_weight);
    free(gpu_count);
    free(ref);
    free(gpu);
    free(mosaic);
    craw_context_free(&ctx);
    if (have_dev)
        craw_testdev_close(&td);
    return status;
}
