/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <craw/utils.h>

#if CRAW_TOOL_HAVE_LAVU
#include <libavutil/dict.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
#include <libavutil/raw_color_params.h>
#endif

#include "profiles.h"

/* Panasonic DC-S1M2, DESIGN.md section 9. Identical in every clip this
 * camera produces, so they belong to the camera and not to the clip.
 *
 * The 2500 K and 10000 K knots are not in the container's table: they are
 * the camera's own per-frame pairs from clips shot in Kelvin mode at those
 * settings (P1115219/222 and P1115220/223, 2026-10-01, the mean of the two
 * clips at each; the stills' EXIF agrees to 0.5%). The container table
 * interpolated linearly in Kelvin, which is how the camera's own decoder
 * reads it, lands 3.4% and 5.4% off the camera's pair at 2500 K and 1% and
 * 2% off at 10000 K, so between the container's knots the camera follows
 * a curve of its own, and each setting it was shot at is a sample of it.
 * The same pair varies by up to 1.5% from clip to clip at one setting,
 * which is still well inside the gap it closes. */
static const CRAWWbKnot s1m2_wb_knots[] = {
    {  2000.0f, { 0.8454f, 1.0f, 3.9482f } },
    {  2500.0f, { 1.1055f, 1.0f, 3.1675f } },
    {  3200.0f, { 1.3763f, 1.0f, 2.4707f } },
    {  3400.0f, { 1.4477f, 1.0f, 2.2881f } },
    {  5000.0f, { 1.9499f, 1.0f, 1.7195f } },
    {  5500.0f, { 2.0253f, 1.0f, 1.6676f } },
    {  6300.0f, { 2.2051f, 1.0f, 1.5348f } },
    {  7300.0f, { 2.3299f, 1.0f, 1.4186f } },
    { 10000.0f, { 2.5537f, 1.0f, 1.2813f } },
    { 11500.0f, { 2.6469f, 1.0f, 1.2318f } },
};

/* The last three knots are the same matrix: the camera stops changing it
 * above 7500 K. Kept as separate knots so the table range still reaches
 * 15000 K for craw_profile_eval()'s clamp. */
static const CRAWMatrixKnot s1m2_matrix_knots[] = {
    {  2300.0f, { { 0.6535f, 0.1947f, 0.1022f }, { 0.2831f, 0.6975f,  0.0193f }, { 0.0113f, -0.1074f, 1.1852f } } },
    {  2800.0f, { { 0.6601f, 0.1881f, 0.1022f }, { 0.2862f, 0.7073f,  0.0064f }, { 0.0104f, -0.0857f, 1.1644f } } },
    {  4000.0f, { { 0.6759f, 0.1723f, 0.1023f }, { 0.2938f, 0.7308f, -0.0246f }, { 0.0083f, -0.0337f, 1.1144f } } },
    {  5000.0f, { { 0.6891f, 0.1591f, 0.1023f }, { 0.3001f, 0.7504f, -0.0505f }, { 0.0066f,  0.0096f, 1.0728f } } },
    {  6500.0f, { { 0.7088f, 0.1394f, 0.1023f }, { 0.3095f, 0.7798f, -0.0893f }, { 0.0040f,  0.0747f, 1.0104f } } },
    {  7500.0f, { { 0.7218f, 0.1263f, 0.1023f }, { 0.3158f, 0.7991f, -0.1149f }, { 0.0023f,  0.1176f, 0.9692f } } },
    { 10000.0f, { { 0.7218f, 0.1263f, 0.1023f }, { 0.3158f, 0.7991f, -0.1149f }, { 0.0023f,  0.1176f, 0.9692f } } },
    { 15000.0f, { { 0.7218f, 0.1263f, 0.1023f }, { 0.3158f, 0.7991f, -0.1149f }, { 0.0023f,  0.1176f, 0.9692f } } },
};

/* Sensor noise model, DESIGN.md 14.2, fitted with `cineraw --noise-estimate
 * --frames 10` on every ProRes RAW clip on disk that records an ISO other
 * than 1000 and on fourteen that record 1000, 42 from ISO 640 to 12800. A
 * clip counts if its fit error is at most 15%, the fit keeps at least 90%
 * of its bins and a moves by at most 16% over the ten frames; its b counts
 * only if the fit resolved one (a read noise above half a code value: the
 * frame has shadows). Each knot is the median over the clips that count,
 * at 1000 with the frame 0 fits of four clips no longer on disk:
 *
 *     640    P1000111
 *     1000   P1000387 P1001123 P1094715 P1094725 P1094729 P1094753
 *            P1094844, and P1093870 P1093961 P1093971 P1094127
 *     1250   P1030266 P1094731
 *     1600   P1000255
 *     2000   P1000167 P1000168
 *     3200   P1094724 P1094727 P1094738
 *     10000  P1000165 P1001122 P1094352 P1094353
 *     12800  P1000164 P1000170 P1094756 P1094757
 *
 * P1094764 (4000) and P1094754 (8000) have no shadows and so no b; their
 * a sits 0.3% and 12% below the table there. Sixteen clips, textured
 * scenes and scenes that move, were discarded.
 *
 * From 1000 up the camera is one gain mode, the GainFactor 23.04 in every
 * clip: a goes as ISO^0.96, 12e-9 * ISO within 10%, and b as
 * 6.1e-10 + 2.3e-16 * ISO^2, a read noise of 1.6 code values after the
 * gain and 9.6e-4 * ISO before it. ISO 640 is another mode (GainFactor
 * 14.51): the same a as 1000 and 22 times the b, in all three 640 clips;
 * where between 640 and 1000 the camera switches no clip says.
 * Outside the table craw_noise_model_eval() carries on along that law: a
 * with the ISO both ways, b with its square above 12800 and unchanged
 * below 640. --noise overrides the whole table. */
static const CRAWNoiseKnot s1m2_noise_knots[] = {
    {   640.0f, 1.27e-5f, 1.72e-8f },
    {  1000.0f, 1.32e-5f, 7.8e-10f },
    {  1250.0f, 1.58e-5f, 9.7e-10f },
    {  1600.0f, 2.11e-5f, 1.22e-9f },
    {  2000.0f, 2.41e-5f, 1.92e-9f },
    {  3200.0f, 3.83e-5f, 2.05e-9f },
    { 10000.0f, 1.18e-4f, 2.65e-8f },
    { 12800.0f, 1.51e-4f, 3.82e-8f },
};

struct tool_builtin_profile {
    const char *name;        /* --profile argument */
    const char *model;       /* what the container calls the camera */
    const char *display;     /* CRAWProfile.name */

    const CRAWWbKnot *wb;
    uint32_t nb_wb;
    const CRAWMatrixKnot *matrix;
    uint32_t nb_matrix;
    const CRAWNoiseKnot *noise;
    uint32_t nb_noise;
};

static const struct tool_builtin_profile tool_builtin_profiles[] = {
    {
        .name      = "s1m2",
        .model     = "DC-S1M2",
        .display   = "Panasonic DC-S1M2",
        .wb        = s1m2_wb_knots,
        .nb_wb     = CRAW_ARRAY_ELEMS(s1m2_wb_knots),
        .matrix    = s1m2_matrix_knots,
        .nb_matrix = CRAW_ARRAY_ELEMS(s1m2_matrix_knots),
        .noise     = s1m2_noise_knots,
        .nb_noise  = CRAW_ARRAY_ELEMS(s1m2_noise_knots),
    },
};

int craw_tool_profile_apply_builtin(CRAWProfile *p, const char *name)
{
    for (size_t i = 0; i < CRAW_ARRAY_ELEMS(tool_builtin_profiles); i++) {
        const struct tool_builtin_profile *b = &tool_builtin_profiles[i];

        if (strcmp(b->name, name))
            continue;

        memcpy(p->wb_knots, b->wb, b->nb_wb * sizeof(*b->wb));
        p->nb_wb_knots = b->nb_wb;
        memcpy(p->matrix_knots, b->matrix, b->nb_matrix * sizeof(*b->matrix));
        p->nb_matrix_knots = b->nb_matrix;
        memcpy(p->noise_knots, b->noise, b->nb_noise * sizeof(*b->noise));
        p->nb_noise_knots = b->nb_noise;
        snprintf(p->name, sizeof(p->name), "%s", b->display);

        return 0;
    }

    return CRAW_ERROR(ENOENT);
}

const char *craw_tool_profile_builtin_list(void)
{
    /* One entry so far; the day there are more this becomes a join. */
    return tool_builtin_profiles[0].name;
}

const char *craw_tool_profile_detect(const char *text)
{
    if (!text)
        return NULL;

    for (size_t i = 0; i < CRAW_ARRAY_ELEMS(tool_builtin_profiles); i++) {
        if (strstr(text, tool_builtin_profiles[i].model))
            return tool_builtin_profiles[i].name;
    }

    return NULL;
}

/* ------------------------------------------------------------- loading */

static int tool_wb_knot_cmp(const void *a, const void *b)
{
    const CRAWWbKnot *ka = a, *kb = b;
    return (ka->cct > kb->cct) - (ka->cct < kb->cct);
}

static int tool_matrix_knot_cmp(const void *a, const void *b)
{
    const CRAWMatrixKnot *ka = a, *kb = b;
    return (ka->cct > kb->cct) - (ka->cct < kb->cct);
}

static int tool_noise_knot_cmp(const void *a, const void *b)
{
    const CRAWNoiseKnot *ka = a, *kb = b;
    return (ka->iso > kb->iso) - (ka->iso < kb->iso);
}

int craw_tool_profile_load_tables(CRAWProfile *p, const char *path)
{
    CRAWWbKnot wb[CRAW_PROFILE_MAX_KNOTS];
    CRAWMatrixKnot mat[CRAW_PROFILE_MAX_KNOTS];
    CRAWNoiseKnot noise[CRAW_PROFILE_MAX_NOISE_KNOTS];
    uint32_t nb_wb = 0, nb_mat = 0, nb_noise = 0;
    char line[512];
    unsigned lineno = 0;
    FILE *f;
    int ret = 0;

    f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "%s: cannot open table file\n", path);
        return CRAW_ERROR(ENOENT);
    }

    while (fgets(line, sizeof(line), f)) {
        char kind[16];
        float v[10];
        int n;

        lineno++;

        /* Comments and blank lines. */
        char *hash = strchr(line, '#');
        if (hash)
            *hash = '\0';
        if (sscanf(line, "%15s", kind) != 1)
            continue;

        if (!strcmp(kind, "wb")) {
            n = sscanf(line, "%15s %f %f %f", kind, &v[0], &v[1], &v[2]);
            if (n != 4) {
                fprintf(stderr, "%s:%u: expected `wb CCT R B`\n", path, lineno);
                ret = CRAW_ERROR(EINVAL);
                goto fail;
            }
            if (nb_wb >= CRAW_PROFILE_MAX_KNOTS) {
                fprintf(stderr, "%s:%u: more than %d wb knots\n", path, lineno,
                        CRAW_PROFILE_MAX_KNOTS);
                ret = CRAW_ERROR(EINVAL);
                goto fail;
            }
            wb[nb_wb++] = (CRAWWbKnot) {
                .cct = v[0],
                .wb  = { v[1], 1.0f, v[2] },
            };
        } else if (!strcmp(kind, "matrix")) {
            n = sscanf(line, "%15s %f %f %f %f %f %f %f %f %f %f", kind,
                       &v[0], &v[1], &v[2], &v[3], &v[4],
                       &v[5], &v[6], &v[7], &v[8], &v[9]);
            if (n != 11) {
                fprintf(stderr, "%s:%u: expected `matrix CCT m00 .. m22`\n", path, lineno);
                ret = CRAW_ERROR(EINVAL);
                goto fail;
            }
            if (nb_mat >= CRAW_PROFILE_MAX_KNOTS) {
                fprintf(stderr, "%s:%u: more than %d matrix knots\n", path, lineno,
                        CRAW_PROFILE_MAX_KNOTS);
                ret = CRAW_ERROR(EINVAL);
                goto fail;
            }
            mat[nb_mat++] = (CRAWMatrixKnot) {
                .cct = v[0],
                .m   = { { v[1], v[2], v[3] }, { v[4], v[5], v[6] }, { v[7], v[8], v[9] } },
            };
        } else if (!strcmp(kind, "noise")) {
            n = sscanf(line, "%15s %f %f %f", kind, &v[0], &v[1], &v[2]);
            if (n != 4) {
                fprintf(stderr, "%s:%u: expected `noise ISO A B`\n", path, lineno);
                ret = CRAW_ERROR(EINVAL);
                goto fail;
            }
            if (nb_noise >= CRAW_PROFILE_MAX_NOISE_KNOTS) {
                fprintf(stderr, "%s:%u: more than %d noise knots\n", path, lineno,
                        CRAW_PROFILE_MAX_NOISE_KNOTS);
                ret = CRAW_ERROR(EINVAL);
                goto fail;
            }
            noise[nb_noise++] = (CRAWNoiseKnot) {
                .iso = v[0], .a = v[1], .b = v[2],
            };
        } else {
            fprintf(stderr, "%s:%u: unknown keyword `%s`\n", path, lineno, kind);
            ret = CRAW_ERROR(EINVAL);
            goto fail;
        }
    }

    if (!nb_wb) {
        fprintf(stderr, "%s: no wb knots\n", path);
        ret = CRAW_ERROR(EINVAL);
        goto fail;
    }

    /* craw_profile_validate() insists on ascending knots, and the file is
     * allowed to be in any order. */
    qsort(wb, nb_wb, sizeof(*wb), tool_wb_knot_cmp);
    qsort(mat, nb_mat, sizeof(*mat), tool_matrix_knot_cmp);
    qsort(noise, nb_noise, sizeof(*noise), tool_noise_knot_cmp);

    memcpy(p->wb_knots, wb, nb_wb * sizeof(*wb));
    p->nb_wb_knots = nb_wb;
    memcpy(p->matrix_knots, mat, nb_mat * sizeof(*mat));
    p->nb_matrix_knots = nb_mat;
    /* A file without noise lines leaves whatever table was already there,
     * so --tables can add per-CCT knots to a built-in profile without
     * throwing its noise model away. */
    if (nb_noise) {
        memcpy(p->noise_knots, noise, nb_noise * sizeof(*noise));
        p->nb_noise_knots = nb_noise;
    }

fail:
    fclose(f);
    return ret;
}

/* ------------------------------------------------------- from a frame */

#if CRAW_TOOL_HAVE_LAVU

static float tool_q2f(AVRational r)
{
    return r.den ? (float)av_q2d(r) : 0.0f;
}

/* The mosaic's phase, from the pixel format the decoder chose: a software
 * frame's own, or the software format behind a Vulkan one. FFmpeg names a
 * Bayer format by its quad read left to right and top to bottom from
 * (0, 0), which is how enum CRAWCfa names it too. The byte order is the
 * uploader's business, not the phase's. */
static int tool_frame_cfa(const AVFrame *frame, enum CRAWCfa *cfa)
{
    enum AVPixelFormat fmt = (enum AVPixelFormat)frame->format;
    const char *name;

    if (fmt == AV_PIX_FMT_VULKAN && frame->hw_frames_ctx)
        fmt = ((const AVHWFramesContext *)frame->hw_frames_ctx->data)->sw_format;

    switch (fmt) {
    case AV_PIX_FMT_BAYER_RGGB16LE:
    case AV_PIX_FMT_BAYER_RGGB16BE:
        *cfa = CRAW_CFA_RGGB;
        return 0;
    case AV_PIX_FMT_BAYER_BGGR16LE:
    case AV_PIX_FMT_BAYER_BGGR16BE:
        *cfa = CRAW_CFA_BGGR;
        return 0;
    case AV_PIX_FMT_BAYER_GRBG16LE:
    case AV_PIX_FMT_BAYER_GRBG16BE:
        *cfa = CRAW_CFA_GRBG;
        return 0;
    case AV_PIX_FMT_BAYER_GBRG16LE:
    case AV_PIX_FMT_BAYER_GBRG16BE:
        *cfa = CRAW_CFA_GBRG;
        return 0;
    default:
        break;
    }

    name = av_get_pix_fmt_name(fmt);
    fprintf(stderr, "the decoder produced %s frames, which are not a 16-bit Bayer "
                    "mosaic\n", name ? name : "frames of an unknown format");

    return CRAW_ERROR(ENOTSUP);
}

static int tool_profile_prores_raw(CRAWProfile *p, const AVRawColorParams *rc)
{
    const AVProResRawColorParams *pr = &rc->codec.prores_raw;

    p->as_shot_wb[0] = tool_q2f(pr->wb_red);
    p->as_shot_wb[1] = 1.0f;
    p->as_shot_wb[2] = tool_q2f(pr->wb_blue);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            p->as_shot_matrix[i][j] = tool_q2f(pr->color_matrix[i][j]);

    p->gain = tool_q2f(pr->gain);
    p->as_shot_cct = (float)rc->wb_cct;

#if CRAW_TOOL_HAVE_EXPOSURE_INDEX
    p->as_shot_ei = tool_q2f(rc->as_shot_ei);
    p->exposure_index = tool_q2f(rc->exposure_index);
#endif

    return 0;
}

#if CRAW_TOOL_HAVE_RW2

/* The scene-linear scale of a still: what puts 18% grey at 0.18 and the
 * sensor clip at the scale itself (DESIGN.md 9b), which a ProRes RAW frame
 * carries as its GainFactor and an RW2 does not. The camera has two gain
 * modes and writes one factor for each in every clip, 23.04 from ISO 1000
 * up and 14.51 at ISO 640 (P1000111 and the other 640 clips of the noise
 * table above), and a still is the same sensor at the same gains - but
 * rated as a still, which the camera does three stops closer to the clip
 * than it rates a V-Log recording: developed with the video factor itself
 * a still came out three stops brighter in the mid-tones than the camera's
 * own JPEG of it, at ISO 100 and at 640 alike, 2.7 to 3.3 stops over four
 * stills (DESIGN.md 9c). So a still gets the video factor over eight, and
 * comes out as the camera's JPEG does; the clip shot at the same ISO
 * number and exposure is three stops brighter than it, because the two
 * ratings are three stops apart, and --exposure 3 is what puts the two
 * side by side. ISO 100 to 500 exist only for stills, so no clip has ever
 * said which mode they are; they are taken to be the low mode's, which is
 * an assumption, and where between 640 and 1000 the camera switches no
 * clip says either, so the line is drawn at 1000, where the high mode is
 * known to start. A still of another camera, developed with --tables,
 * gets the same numbers until that camera has been measured. */
#define TOOL_RW2_HIGH_MODE_ISO 1000.0f
#define TOOL_RW2_GAIN_HIGH     (23.04f / 8.0f)
#define TOOL_RW2_GAIN_LOW      (14.51f / 8.0f)

/* Exif values arrive as the decoder prints them, padded with spaces:
 * "    640". */
static float tool_frame_iso(const AVFrame *frame)
{
    const AVDictionaryEntry *e = av_dict_get(frame->metadata,
                                             "ExifIFD/ISOSpeedRatings", NULL, 0);

    return e ? (float)atof(e->value) : 0.0f;
}

static int tool_profile_rw2(CRAWProfile *p, const AVRawColorParams *rc,
                            const AVFrame *frame, bool verbose)
{
    const AVRW2RawColorParams *rw = &rc->codec.rw2;
    const AVDictionaryEntry *model;
    CRAWIlluminant illum;
    CRAWColorParams cp;
    float iso;
    int ret;

    p->as_shot_wb[0] = tool_q2f(rw->wb_red);
    p->as_shot_wb[1] = 1.0f;
    p->as_shot_wb[2] = tool_q2f(rw->wb_blue);

    if (!(p->as_shot_wb[0] > 0.0f) || !(p->as_shot_wb[2] > 0.0f)) {
        fprintf(stderr, "the RW2's white balance (R %.5f, B %.5f) is not usable\n",
                p->as_shot_wb[0], p->as_shot_wb[2]);
        return CRAW_ERROR(EINVAL);
    }

    /* The file has no matrix, and an identity in its place would be a
     * picture in the wrong colours that nothing warns about. The camera's
     * own tables are the same sensor's, so the matrix is the one they give
     * where the as-shot balance lies; without them there is none. */
    if (p->nb_wb_knots < 2 || !p->nb_matrix_knots) {
        model = av_dict_get(frame->metadata, "Model", NULL, 0);
        fprintf(stderr, "an RW2 carries no colour matrix, and there are no per-CCT white "
                        "balance and matrix tables for %s to take one from: name a "
                        "built-in set with --profile (have: %s) or load the camera's with "
                        "--tables FILE\n",
                model ? model->value : "this camera", craw_tool_profile_builtin_list());
        return CRAW_ERROR(ENOENT);
    }

    /* A neutral's raw ratios are the reciprocals of the gains that balance
     * it, and the colour temperature is the point of the camera's locus
     * those ratios lie at; whatever is left across the locus is the tint,
     * which the gains already carry. */
    ret = craw_profile_neutral_to_illuminant(p, 1.0f / p->as_shot_wb[0],
                                             1.0f / p->as_shot_wb[2], &illum);
    if (ret < 0) {
        fprintf(stderr, "the RW2's white balance does not map onto the camera's "
                        "locus: %d\n", ret);
        return ret;
    }

    ret = craw_profile_eval(p, CRAW_WB_REBALANCE, &illum, &cp);
    if (ret < 0) {
        fprintf(stderr, "the camera's tables do not evaluate at %.0f K: %d\n",
                illum.cct, ret);
        return ret;
    }

    memcpy(p->as_shot_matrix, cp.cam_to_xyz, sizeof(p->as_shot_matrix));
    p->as_shot_cct = illum.cct;

    /* The ISO the still was taken at is its as-shot index, so --ei re-rates
     * a still exactly as it re-rates a clip. */
    iso = tool_frame_iso(frame);
    p->as_shot_ei = iso;
    p->gain = iso >= TOOL_RW2_HIGH_MODE_ISO ? TOOL_RW2_GAIN_HIGH : TOOL_RW2_GAIN_LOW;

    if (verbose)
        fprintf(stderr, "rw2: as-shot gains R %.4f B %.4f lie at %.0f K, tint %+.4f, on "
                        "the %s locus; ISO %s%.0f, gain %.2f (the %s gain mode%s)\n",
                p->as_shot_wb[0], p->as_shot_wb[2], illum.cct, illum.tint,
                p->name[0] ? p->name : "loaded tables'", iso > 0.0f ? "" : "unknown, ",
                iso, p->gain, iso >= TOOL_RW2_HIGH_MODE_ISO ? "high" : "low",
                iso >= 640.0f && iso < TOOL_RW2_HIGH_MODE_ISO ? ", measured at 640"
                : iso < 640.0f ? ", assumed below 640" : "");

    return 0;
}

#endif /* CRAW_TOOL_HAVE_RW2 */

int craw_tool_profile_from_frame(CRAWProfile *p, const AVFrame *frame, bool verbose)
{
    const AVFrameSideData *sd;
    const AVRawColorParams *rc;
    int ret;

    sd = av_frame_get_side_data(frame, AV_FRAME_DATA_RAW_COLOR_PARAMS);
    if (!sd) {
        fprintf(stderr, "the decoder produced no AV_FRAME_DATA_RAW_COLOR_PARAMS "
                        "side data; only a raw camera frame can be developed\n");
        return CRAW_ERROR(EINVAL);
    }
    rc = (const AVRawColorParams *)sd->data;

    ret = tool_frame_cfa(frame, &p->cfa);
    if (ret < 0)
        return ret;

    /* The levels arrive normalized to the 16-bit sample range, which is
     * exactly the UNORM domain the pipeline reads. ProRes RAW has no
     * per-site variation, and an RW2's are the green sites' for all four,
     * which is what the decoder exports. An RW2's samples are the camera's
     * 14-bit codes in the 16-bit container, not scaled up, and its levels
     * are in the same units, so (sample - black) / (white - black) is right
     * for both without a conversion. */
    for (int i = 0; i < 4; i++) {
        p->black_level[i] = tool_q2f(rc->black_level);
        p->white_level[i] = tool_q2f(rc->white_level);
    }

    switch (rc->type) {
    case AV_RAW_COLOR_PARAMS_PRORES_RAW:
        ret = tool_profile_prores_raw(p, rc);
        break;
#if CRAW_TOOL_HAVE_RW2
    case AV_RAW_COLOR_PARAMS_RW2:
        ret = tool_profile_rw2(p, rc, frame, verbose);
        break;
#endif
    default:
        fprintf(stderr, "raw colour parameters of type %d are not supported\n",
                (int)rc->type);
        return CRAW_ERROR(ENOTSUP);
    }
    if (ret < 0)
        return ret;

    ret = craw_profile_validate(p);
    if (ret < 0)
        fprintf(stderr, "the frame's colour parameters do not make a valid profile "
                        "(black %.6f, white %.6f, wb %.5f/%.5f, gain %.4f): %d\n",
                p->black_level[0], p->white_level[0], p->as_shot_wb[0],
                p->as_shot_wb[2], p->gain, ret);

    return ret;
}

#endif /* CRAW_TOOL_HAVE_LAVU */
