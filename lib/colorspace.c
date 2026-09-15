/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#include <math.h>
#include <string.h>

#include <craw/profile.h>

/* Colour spaces are described by their CIE 1931 xy primaries and the white
 * they are defined against, and the matrices are derived from those at call
 * time in double precision. Storing nine hand transcribed constants per
 * space would be faster and much easier to get subtly wrong; this runs once
 * per pipeline creation. */

#define CRAW_D65_X  0.3127
#define CRAW_D65_Y  0.3290
#define CRAW_ACES_X 0.32168
#define CRAW_ACES_Y 0.33767

struct CRAWColorSpaceDesc {
    double prim[3][2];   /* R, G, B chromaticities */
    bool aces_white;     /* defined against the ACES white, not D65 */
};

static const struct CRAWColorSpaceDesc craw_cs_desc[CRAW_CS_NB] = {
    /* CRAW_CS_XYZ_D65 is the identity and never reaches the table. */
    [CRAW_CS_BT709_LINEAR] = {
        .prim = { { 0.64, 0.33 }, { 0.30, 0.60 }, { 0.15, 0.06 } },
    },
    [CRAW_CS_BT2020_LINEAR] = {
        .prim = { { 0.708, 0.292 }, { 0.170, 0.797 }, { 0.131, 0.046 } },
    },
    [CRAW_CS_P3D65_LINEAR] = {
        .prim = { { 0.680, 0.320 }, { 0.265, 0.690 }, { 0.150, 0.060 } },
    },
    /* Panasonic V-Gamut. The blue primary has a negative y: the gamut is
     * wider than the spectral locus there, which is legal for a working
     * space and only means the RGB to XYZ matrix has negative entries. */
    [CRAW_CS_V_GAMUT_LINEAR] = {
        .prim = { { 0.730, 0.280 }, { 0.165, 0.840 }, { 0.100, -0.030 } },
    },
    [CRAW_CS_ACES_AP0_LINEAR] = {
        .prim = { { 0.7347, 0.2653 }, { 0.0, 1.0 }, { 0.0001, -0.0770 } },
        .aces_white = true,
    },
    [CRAW_CS_ACES_AP1_LINEAR] = {
        .prim = { { 0.713, 0.293 }, { 0.165, 0.830 }, { 0.128, 0.044 } },
        .aces_white = true,
    },
};

/* CIE xy at unit luminance. */
static void craw_xy_to_xyz(double x, double y, double out[3])
{
    out[0] = x / y;
    out[1] = 1.0;
    out[2] = (1.0 - x - y) / y;
}

static void craw_mat3_mul(const double a[3][3], const double b[3][3],
                          double out[3][3])
{
    double t[3][3];

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            t[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }

    memcpy(out, t, sizeof(t));
}

static void craw_mat3_mul_vec(const double m[3][3], const double v[3],
                              double out[3])
{
    double t[3];

    for (int i = 0; i < 3; i++)
        t[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];

    memcpy(out, t, sizeof(t));
}

static double craw_mat3_det(const double m[3][3])
{
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
         - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
         + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

static int craw_mat3_invert(const double m[3][3], double out[3][3])
{
    double det = craw_mat3_det(m);
    double t[3][3];

    if (!(fabs(det) > 1e-12))
        return CRAW_ERROR(EINVAL);

    t[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det;
    t[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det;
    t[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det;
    t[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det;
    t[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det;
    t[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det;
    t[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det;
    t[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det;
    t[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det;

    memcpy(out, t, sizeof(t));
    return 0;
}

/* The classic RGB to XYZ construction: the primaries give the directions of
 * the three columns, and the scale of each column is whatever makes the
 * three of them add up to the white point. */
static int craw_rgb_to_xyz(const double prim[3][2], const double white[2],
                           double out[3][3])
{
    double p[3][3], inv[3][3], w[3], s[3];
    int ret;

    for (int c = 0; c < 3; c++) {
        double col[3];
        craw_xy_to_xyz(prim[c][0], prim[c][1], col);
        for (int i = 0; i < 3; i++)
            p[i][c] = col[i];
    }

    craw_xy_to_xyz(white[0], white[1], w);

    ret = craw_mat3_invert(p, inv);
    if (ret < 0)
        return ret;

    craw_mat3_mul_vec(inv, w, s);

    for (int i = 0; i < 3; i++) {
        for (int c = 0; c < 3; c++)
            out[i][c] = p[i][c] * s[c];
    }

    return 0;
}

/* Von Kries adaptation in the Bradford cone space, the transform ACES and
 * DNG both specify for moving a matrix between white points. */
static int craw_bradford(const double src[2], const double dst[2],
                         double out[3][3])
{
    static const double bfd[3][3] = {
        {  0.8951,  0.2664, -0.1614 },
        { -0.7502,  1.7135,  0.0367 },
        {  0.0389, -0.0685,  1.0296 },
    };
    double bfd_inv[3][3], scale[3][3] = { { 0 } }, tmp[3][3];
    double ws[3], wd[3], cs[3], cd[3];
    int ret;

    craw_xy_to_xyz(src[0], src[1], ws);
    craw_xy_to_xyz(dst[0], dst[1], wd);
    craw_mat3_mul_vec(bfd, ws, cs);
    craw_mat3_mul_vec(bfd, wd, cd);

    for (int i = 0; i < 3; i++) {
        if (!(fabs(cs[i]) > 1e-12))
            return CRAW_ERROR(EINVAL);
        scale[i][i] = cd[i] / cs[i];
    }

    ret = craw_mat3_invert(bfd, bfd_inv);
    if (ret < 0)
        return ret;

    craw_mat3_mul(scale, bfd, tmp);
    craw_mat3_mul(bfd_inv, tmp, out);

    return 0;
}

int craw_colorspace_from_xyz_d65(enum CRAWColorSpace cs, float m[3][3])
{
    static const double d65[2] = { CRAW_D65_X, CRAW_D65_Y };
    static const double aces[2] = { CRAW_ACES_X, CRAW_ACES_Y };
    const struct CRAWColorSpaceDesc *d;
    double to_xyz[3][3], from_xyz[3][3];
    int ret;

    if (!m || (unsigned)cs >= CRAW_CS_NB)
        return CRAW_ERROR(EINVAL);

    if (cs == CRAW_CS_XYZ_D65) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++)
                m[i][j] = i == j ? 1.0f : 0.0f;
        }
        return 0;
    }

    d = &craw_cs_desc[cs];

    ret = craw_rgb_to_xyz(d->prim, d->aces_white ? aces : d65, to_xyz);
    if (ret < 0)
        return ret;

    ret = craw_mat3_invert(to_xyz, from_xyz);
    if (ret < 0)
        return ret;

    /* The input is always relative to D65; a space defined against another
     * white needs the chromatic adaptation applied first. */
    if (d->aces_white) {
        double cat[3][3];

        ret = craw_bradford(d65, aces, cat);
        if (ret < 0)
            return ret;

        craw_mat3_mul(from_xyz, cat, from_xyz);
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            m[i][j] = (float)from_xyz[i][j];
    }

    return 0;
}
