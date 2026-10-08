/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Panasonic's per-frame optical correction record, read out of a ProRes RAW
 * frame. Both tools want it, so it lives in the library rather than in
 * either of them; nothing here touches Vulkan and nothing here decodes a
 * pixel. See DESIGN.md 16 and tools/prr_lens.md, which is the derivation
 * this file implements.
 *
 * The frame is walked the way libavcodec/prores_raw.c walks it, so the
 * vendor metadata list is found by arithmetic rather than by searching the
 * frame for the key string: a search would also find the bytes inside a
 * tile, and would say nothing about whether what follows them is a record. */

#include <math.h>
#include <stddef.h>
#include <string.h>

#include <craw/lens.h>

#include "lens_warp.h"

/* The `psim` key, without a terminator: the record's key is a Pascal
 * string. */
static const char craw_lens_key[] = "com.panasonic.Semi-Pro.optical_correction";
#define CRAW_LENS_KEY_LEN (sizeof(craw_lens_key) - 1)

/* version + 20 doubles: three channel blocks of six and the centre. */
#define CRAW_LENS_PAYLOAD (4 + 20 * 8)

/* The one version seen on every frame of every clip. A different one would
 * be a different layout and is refused rather than guessed at. */
#define CRAW_LENS_VERSION 3

static_assert(sizeof(CRAWLensCorrection) == 64, "CRAWLensCorrection size");
static_assert(offsetof(CRAWLensCorrection, centre) == 48, "CRAWLensCorrection layout");

static uint32_t craw_rb32(const uint8_t *p)
{
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8)  |  (uint32_t)p[3];
}

static uint32_t craw_rb16(const uint8_t *p)
{
    return ((uint32_t)p[0] << 8) | (uint32_t)p[1];
}

static double craw_rb64f(const uint8_t *p)
{
    uint64_t bits = 0;
    double v;

    for (int i = 0; i < 8; i++)
        bits = (bits << 8) | (uint64_t)p[i];

    memcpy(&v, &bits, sizeof(v));

    return v;
}

static uint32_t craw_popcount(uint32_t v)
{
    uint32_t n = 0;

    while (v) {
        v &= v - 1;
        n++;
    }

    return n;
}

/* Decodes the 164 byte payload. Returns 0, or EINVAL for a payload whose
 * version or shape is not the one documented in tools/prr_lens.md. */
static int craw_lens_payload(const uint8_t *p, size_t len, CRAWLensCorrection *out)
{
    double v[20];

    if (len != CRAW_LENS_PAYLOAD)
        return CRAW_ERROR(EINVAL);
    if (craw_rb32(p) != CRAW_LENS_VERSION)
        return CRAW_ERROR(EINVAL);

    for (int i = 0; i < 20; i++)
        v[i] = craw_rb64f(p + 4 + i * 8);

    /* Three blocks of six, R then G then B: k0..k3 and two slots that are
     * exactly +0.0 in every frame of every clip tried and are therefore
     * read as pad. Then the centre. */
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < 4; i++)
            out->k[c][i] = (float)v[c * 6 + i];

    out->centre[0] = (float)v[18];
    out->centre[1] = (float)v[19];

    /* A polynomial that is not finite, or a centre outside the frame, would
     * send the warp somewhere the mirrored border cannot follow. The camera
     * writes exactly (0.5, 0.5), so this only ever fires on a corrupt
     * frame, but the pass must not be handed one. */
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < 4; i++)
            if (!isfinite(out->k[c][i]))
                return CRAW_ERROR(EINVAL);

    if (!isfinite(out->centre[0]) || !isfinite(out->centre[1]) ||
        out->centre[0] < 0.0f || out->centre[0] > 1.0f ||
        out->centre[1] < 0.0f || out->centre[1] > 1.0f)
        return CRAW_ERROR(EINVAL);

    out->valid = true;

    return 0;
}

int craw_lens_from_prores_raw(const uint8_t *pkt, size_t pkt_size,
                              CRAWLensCorrection *out)
{
    size_t frame_size, hdr_len, tbl, off;
    uint32_t w, h, flags, align, tw16, nb_tw, nb_th, nb_tiles;
    const uint8_t *h_body;

    if (!pkt || !out)
        return CRAW_ERROR(EINVAL);

    *out = (CRAWLensCorrection) { 0 };

    /* be32 size, then 'prrf'. The size counts itself and any trailing pad,
     * so it is the bound everything below is checked against; a container
     * that hands over a longer buffer is not a reason to read past it. */
    if (pkt_size < 10)
        return CRAW_ERROR(EINVAL);

    frame_size = craw_rb32(pkt);
    if (frame_size < 10 || frame_size > pkt_size)
        return CRAW_ERROR(EINVAL);
    if (memcmp(pkt + 4, "prrf", 4))
        return CRAW_ERROR(EINVAL);

    /* be16 header length, counting itself: the body is hdr_len - 2 bytes
     * and starts at 10. The walk below reads 70 bytes of it. */
    hdr_len = craw_rb16(pkt + 8);
    if (hdr_len < 72 || 8 + hdr_len > frame_size)
        return CRAW_ERROR(EINVAL);

    h_body = pkt + 10;
    w = craw_rb16(h_body + 6);
    h = craw_rb16(h_body + 8);
    flags = craw_rb16(h_body + 68);
    align = (flags >> 1) & 7;

    if (!w || !h)
        return CRAW_ERROR(EINVAL);

    /* Tile count, exactly as prores_raw.c derives it: the width in 16 pixel
     * columns is split into groups of 1 << align, and every set bit below
     * that alignment is one more tile of its own. */
    tw16 = (w + 15) >> 4;
    nb_tw = (tw16 >> align) + craw_popcount(tw16 & ((1u << align) - 1u));
    nb_th = (h + 15) >> 4;
    nb_tiles = nb_tw * nb_th;

    tbl = 8 + hdr_len;
    if (!nb_tiles || nb_tiles > (frame_size - tbl) / 2)
        return CRAW_ERROR(EINVAL);

    /* The tile sizes are be16 and there are at most a few tens of
     * thousands of them, so the sum cannot overflow a size_t; it is still
     * checked against the frame after every addition rather than at the
     * end. */
    off = tbl + (size_t)nb_tiles * 2;
    for (uint32_t i = 0; i < nb_tiles; i++) {
        off += craw_rb16(pkt + tbl + (size_t)i * 2);
        if (off > frame_size)
            return CRAW_ERROR(EINVAL);
    }

    /* Everything after the last tile is vendor metadata: a list of records
     * framed as be32 size (counting itself and the fourcc) plus the
     * fourcc, terminated by an 8 byte `eomd`.
     *
     * Bytes that are not a walkable list end the search rather than failing
     * it, because not every camera writes a list at all: an FX6 recorded to
     * an Atomos Ninja V ends its frames with a few dozen bytes that are
     * neither records nor zero and carry no `eomd` anywhere. "This frame
     * has no such record" is the useful answer there, and it is the same
     * answer as for a frame whose list simply does not contain one. */
    while (off + 8 <= frame_size) {
        size_t rec_size = craw_rb32(pkt + off);
        const uint8_t *tag = pkt + off + 4;
        size_t klen;

        if (!memcmp(tag, "eomd", 4))
            break;
        if (rec_size < 9 || rec_size > frame_size - off)
            break;

        if (memcmp(tag, "psim", 4)) {
            off += rec_size;
            continue;
        }

        klen = pkt[off + 8];
        if (9 + klen > rec_size)
            break;

        /* The record is ours: from here a bad payload is an error and not
         * a miss, because there is nothing else it could be. */
        if (klen == CRAW_LENS_KEY_LEN &&
            !memcmp(pkt + off + 9, craw_lens_key, CRAW_LENS_KEY_LEN)) {
            int ret = craw_lens_payload(pkt + off + 9 + klen,
                                        rec_size - 9 - klen, out);
            if (ret < 0)
                *out = (CRAWLensCorrection) { 0 };
            return ret;
        }

        off += rec_size;
    }

    return CRAW_ERROR(ENOENT);
}

int craw_lens_corner_shift(const CRAWLensCorrection *lc, uint32_t width,
                           uint32_t height, bool relative, float px[3])
{
    float hx, hy, half_diag;

    if (!lc || !lc->valid || !px || !width || !height)
        return CRAW_ERROR(EINVAL);

    /* r = 1 is the frame corner as seen from the stated centre, which for
     * an off-centre optic is the farther of the two sides. */
    hx = CRAW_MAX(lc->centre[0], 1.0f - lc->centre[0]) * (float)width;
    hy = CRAW_MAX(lc->centre[1], 1.0f - lc->centre[1]) * (float)height;
    half_diag = sqrtf(hx * hx + hy * hy);

    for (int c = 0; c < 3; c++) {
        const float *k = lc->k[c];
        float f1 = k[0] + k[1] + k[2] + k[3];   /* poly(1) */

        px[c] = half_diag * (relative ? (k[0] != 0.0f ? f1 / k[0] - 1.0f : 0.0f)
                                      : f1 - 1.0f);
    }

    return 0;
}

/* ------------------------------------------- keeping every source pixel */

/* The boundary walk of DESIGN.md 16.8. 512 segments per side, each side
 * carrying its own starting corner, so 2048 points around the rectangle.
 * An even number of segments per side puts a sample exactly on every
 * corner and exactly on every edge midpoint, which for a monotone
 * polynomial are the only two places the constraint can bind; everything
 * between them is there for a polynomial that turns over. */
#define CRAW_LENS_KEEP_SEGS 512

/* The bisection's bracket and its iteration count. r * poly(r) is monotone
 * over this range for every record seen and for anything resembling a
 * lens; a polynomial that turns over inside it would make the inverse
 * ambiguous, and bisection would find one of the crossings, which is as
 * meaningful an answer as there is. 40 halvings of [0, 2] land within
 * 2e-12, far inside the float the answer comes back as. */
#define CRAW_LENS_KEEP_HI 2.0
#define CRAW_LENS_KEEP_ITERS 40

/* In double, and from coefficients already widened by the caller: the walk
 * evaluates this a quarter of a million times and a float array would
 * convert four coefficients on every one of them. */
static double craw_lens_poly(const double k[4], double r)
{
    double r2 = r * r;

    return k[0] + r2 * (k[1] + r2 * (k[2] + r2 * k[3]));
}

/* The output radius whose source radius is `r_src`, i.e. the inverse of
 * r * poly(r), by bisection, for the three channels at once. False when
 * `r_src` is out of the map's reach over [0, CRAW_LENS_KEEP_HI], which
 * takes a record twice as aggressive as any lens.
 *
 * Three brackets are halved in one loop and each is moved by a conditional
 * assignment rather than a branch, because this loop is the entire cost of
 * the scale: the three chains interleave where three separate searches
 * would each wait on their own compare, and a bisection's branch is
 * unpredictable by construction. Measured on this machine the whole walk
 * is 0.87 ms a frame this way and 2.07 ms written the obvious way, with a
 * branch and a float coefficient array. The bracket is carried as its low
 * end and its width, both dyadic and therefore exact, so the halving
 * introduces no rounding of its own. */
static bool craw_lens_invert3(const double k[3][4], double r_src, double u[3])
{
    double lo0 = 0.0, lo1 = 0.0, lo2 = 0.0, w = CRAW_LENS_KEEP_HI;

    for (int c = 0; c < 3; c++)
        if (CRAW_LENS_KEEP_HI * craw_lens_poly(k[c], CRAW_LENS_KEEP_HI) < r_src)
            return false;

    for (int i = 0; i < CRAW_LENS_KEEP_ITERS; i++) {
        double m0, m1, m2;

        w *= 0.5;
        m0 = lo0 + w;
        m1 = lo1 + w;
        m2 = lo2 + w;
        lo0 = (m0 * craw_lens_poly(k[0], m0) < r_src) ? m0 : lo0;
        lo1 = (m1 * craw_lens_poly(k[1], m1) < r_src) ? m1 : lo1;
        lo2 = (m2 * craw_lens_poly(k[2], m2) < r_src) ? m2 : lo2;
    }

    u[0] = lo0 + 0.5 * w;
    u[1] = lo1 + 0.5 * w;
    u[2] = lo2 + 0.5 * w;

    return true;
}

/* How far the ray (ux, uy) from (cx, cy) travels before it leaves the
 * w x h rectangle, in pixels. With the camera's centred record this is the
 * min(half_w / |cos|, half_h / |sin|) of DESIGN.md 16.8; written per side
 * so that an off-centre record measures against the edge it is actually
 * heading for. */
static double craw_lens_frame_radius(double ux, double uy, double cx, double cy,
                                     double w, double h)
{
    double t = HUGE_VAL;

    if (ux > 0.0)
        t = CRAW_MIN(t, (w - cx) / ux);
    else if (ux < 0.0)
        t = CRAW_MIN(t, cx / -ux);

    if (uy > 0.0)
        t = CRAW_MIN(t, (h - cy) / uy);
    else if (uy < 0.0)
        t = CRAW_MIN(t, cy / -uy);

    return t;
}

/* -------------------------------------------- turning the picture (16.9) */

#define CRAW_LENS_DEG (3.14159265358979323846 / 180.0)

const CRAWLensCorrection craw_lens_identity = {
    .k = { { 1.0f, 0.0f, 0.0f, 0.0f },
           { 1.0f, 0.0f, 0.0f, 0.0f },
           { 1.0f, 0.0f, 0.0f, 0.0f } },
    .centre = { 0.5f, 0.5f },
    .valid  = true,
};

void craw_lens_turn(float degrees, float scale, float turn[2])
{
    double rad = (double)degrees * CRAW_LENS_DEG;

    turn[0] = (float)(cos(rad) / (double)scale);
    turn[1] = (float)(sin(rad) / (double)scale);
}

/* Both scales of DESIGN.md 16.8 and 16.9 are one walk around the frame's
 * rectangle, which is both the source's outline and the output's. The map
 * is radial, so the corrected picture's outline along a direction is where
 * the source rectangle is left along that same direction, put through the
 * inverse of r * poly(r); the turn only changes which direction of the
 * output frame meets which direction of the corrected picture.
 *
 * Keeping everything walks the source's outline and asks how far out each
 * point lands, turned, against the output frame along the turned
 * direction: the smallest ratio is the largest shrink that keeps it all.
 * Filling walks the output frame's outline and asks how far the corrected
 * source reaches along the direction each point is turned onto: the
 * largest ratio of the point's radius to that reach is the smallest
 * magnification that leaves no output pixel without a source. Both are
 * exact for the same reason the keep walk was: the map is monotone along
 * each ray and both outlines are seen from the centre, which is inside
 * both, so a boundary inside means everything inside. */
static int craw_lens_walk(const CRAWLensCorrection *lc, uint32_t width,
                          uint32_t height, float degrees, bool keep_all,
                          float *scale)
{
    double k[3][4], w, h, cx, cy, hx, hy, half_diag, s = 1.0;
    double rad = (double)degrees * CRAW_LENS_DEG;
    double ct = cos(rad), st = sin(rad);

    if (lc->centre[0] < 0.0f || lc->centre[0] > 1.0f ||
        lc->centre[1] < 0.0f || lc->centre[1] > 1.0f)
        return CRAW_ERROR(EINVAL);

    for (int c = 0; c < 3; c++)
        for (int i = 0; i < 4; i++)
            k[c][i] = (double)lc->k[c][i];

    w = (double)width;
    h = (double)height;
    cx = (double)lc->centre[0] * w;
    cy = (double)lc->centre[1] * h;
    hx = (double)CRAW_MAX(lc->centre[0], 1.0f - lc->centre[0]) * w;
    hy = (double)CRAW_MAX(lc->centre[1], 1.0f - lc->centre[1]) * h;
    half_diag = sqrt(hx * hx + hy * hy);
    if (!(half_diag > 0.0))
        return CRAW_ERROR(EINVAL);

    /* Every source pixel lies inside the source rectangle, and the map is
     * radial and monotone along each ray, so an interior point never needs
     * a larger output radius than the boundary point on its own ray: the
     * boundary is the whole constraint. The same holds the other way round
     * for the output frame when filling. */
    for (int side = 0; side < 4; side++) {
        for (int i = 0; i < CRAW_LENS_KEEP_SEGS; i++) {
            double t = (double)i / CRAW_LENS_KEEP_SEGS;
            double px, py, dx, dy, len, ux, uy, r_src, r_frame, u[3];

            switch (side) {
            case 0:  px = t * w;         py = 0.0;           break;
            case 1:  px = w;             py = t * h;         break;
            case 2:  px = (1.0 - t) * w; py = h;             break;
            default: px = 0.0;           py = (1.0 - t) * h; break;
            }

            dx = px - cx;
            dy = py - cy;
            len = sqrt(dx * dx + dy * dy);
            /* An optical centre sitting exactly on the frame edge, which no
             * record has: that one point constrains nothing. */
            if (!(len > 0.0))
                continue;
            ux = dx / len;
            uy = dy / len;

            if (keep_all) {
                double needed = 0.0;

                /* A point of the source's outline: the channel that needs
                 * the largest output radius decides, so that one scale
                 * serves all three and the black border is one shape rather
                 * than three coloured ones. Unturned, the output direction
                 * is the source's own, exactly: one times it, less nothing. */
                r_src = len / half_diag;
                if (!craw_lens_invert3(k, r_src, u))
                    return CRAW_ERROR(EINVAL);
                for (int c = 0; c < 3; c++)
                    needed = CRAW_MAX(needed, u[c]);

                r_frame = craw_lens_frame_radius(ct * ux - st * uy, st * ux + ct * uy,
                                                 cx, cy, w, h) / half_diag;
                s = CRAW_MIN(s, r_frame / needed);
            } else {
                double reach;

                /* A point of the output frame's outline, turned onto the
                 * corrected picture: the source rectangle ends at r_src
                 * along that direction, and the corrected picture at the
                 * inverse of it. The channel that runs out first decides,
                 * so that no pixel is left with one channel off the
                 * source. */
                r_src = craw_lens_frame_radius(ct * ux + st * uy, ct * uy - st * ux,
                                               cx, cy, w, h) / half_diag;
                if (!craw_lens_invert3(k, r_src, u))
                    return CRAW_ERROR(EINVAL);
                reach = CRAW_MIN(CRAW_MIN(u[0], u[1]), u[2]);

                s = CRAW_MAX(s, (len / half_diag) / reach);
            }
        }
    }

    /* Bounded by the initial value either way. Keeping everything is capped
     * at 1: a correction that samples outward everywhere would otherwise be
     * magnified to fill the frame, which is not what keeping everything
     * means. Filling never goes below 1: the turn may use what the record's
     * own frame throws away, not show more of it than that frame does. */
    if (!(s > 0.0) || !isfinite(s))
        return CRAW_ERROR(EINVAL);

    *scale = (float)s;

    return 0;
}

int craw_lens_keep_all_scale(const CRAWLensCorrection *lc, uint32_t width,
                             uint32_t height, float *scale)
{
    if (!lc || !lc->valid || !scale || !width || !height)
        return CRAW_ERROR(EINVAL);

    return craw_lens_walk(lc, width, height, 0.0f, true, scale);
}

int craw_lens_rotation_scale(const CRAWLensCorrection *lc, uint32_t width,
                             uint32_t height, float degrees, bool keep_all,
                             float *scale)
{
    if (!scale || !width || !height || !(fabsf(degrees) < CRAW_LENS_MAX_ROTATION))
        return CRAW_ERROR(EINVAL);
    if (!lc || !lc->valid)
        lc = &craw_lens_identity;

    /* The record's own frame, the camera's choice and the one every other
     * decoder shows. Nothing to walk. */
    if (!keep_all && degrees == 0.0f) {
        *scale = 1.0f;
        return 0;
    }

    return craw_lens_walk(lc, width, height, degrees, keep_all, scale);
}
