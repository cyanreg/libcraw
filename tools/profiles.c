/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <craw/utils.h>

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
