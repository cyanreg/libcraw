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
