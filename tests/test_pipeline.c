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

#include <craw/craw.h>
#include <craw/vulkan.h>
#include <craw/profile.h>
#include <craw/pipeline.h>
#include <craw/awb.h>

#include "vkdev.h"
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
    case CRAW_CS_XYZ_D65:        return "XYZ_D65";
    case CRAW_CS_BT2020_LINEAR:  return "BT2020";
    default:                     return "?";
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

/* ----------------------------------------------------------- rejections */

/* The configurations craw_pipeline_create() must refuse, plus the histogram
 * accessor on a pipeline that was never asked for one. */
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

/* ----------------------------------------------------------- real frame */

static int test_real_frame(CRAWTestDevice *td, CRAWContext *ctx, const CRAWProfile *profile)
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
    double ms = 0.0;
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

    printf("  rcd on %dx%d: %.2f ms, max abs %.3e, max rel %.3e\n", FRAME_W, FRAME_H, ms,
           test_max_abs(gpu, ref, npix), test_max_rel(gpu, ref, npix));

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

    if (test_real_frame(&td, ctx, &profile) < 0)
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
