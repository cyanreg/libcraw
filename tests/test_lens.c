/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* craw_lens_from_prores_raw() against real camera bytes, DESIGN.md 16.
 *
 * tests/data/prr_lens_frame0.bin is frame 0 of P1000387.MOV (Panasonic
 * DC-S1M2, LUMIX S 24-60 at 60 mm) with the tile *payload* removed: it
 * holds the frame's first 8836 bytes verbatim — the be32 size, `prrf`, the
 * 88 byte header and all 4370 be16 tile sizes — followed by the 232 byte
 * vendor metadata trailer verbatim, which is the `psim` record and the
 * `eomd` that ends the list. 9068 bytes on disk instead of 5133984. The
 * test rebuilds the frame by putting sum(tile sizes) zero bytes back where
 * the tiles were, so every byte the parser actually reads is the camera's
 * own and the walk — header length, tile count from width, height and the
 * alignment field, tile table sum, record framing, payload decode — runs
 * end to end on it.
 *
 * With CRAW_TEST_CLIP pointing at a ProRes RAW clip, and libavformat
 * available at build time, the first packet of that clip is read and
 * parsed too. That path is skipped without both. */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <craw/craw.h>
#include <craw/lens.h>

#if CRAW_TEST_HAVE_LAVF
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#endif

#define FRAME_W 5760
#define FRAME_H 3040

/* tools/prr_lens.md section 5, P1000387 at 60 mm, decoded from the record's
 * doubles and narrowed to float. */
static const float test_k[3][4] = {
    { 0.9732558749542047f,  1.3826185943196782e-02f,  1.5989100451823418e-02f, -5.451906036893745e-03f },
    { 0.9736971220864274f,  1.3698028399765133e-02f,  1.6482908551221100e-02f, -5.962137811706458e-03f },
    { 0.9743091685055746f,  1.3363908702330294e-02f,  1.7107910468175024e-02f, -6.633721811149271e-03f },
};

/* Same section: the net corner shift and the distortion with the crop
 * divided out, in pixels on a 5760x3040 frame. */
static const float test_net_px[3]  = { -7.75f, -6.79f, -6.03f };
static const float test_dist_px[3] = { 81.52f, 81.00f, 79.68f };

static int test_fails;

#define TEST_CHECK(cond, ...) do {                                          \
    if (!(cond)) {                                                          \
        fprintf(stderr, "FAIL %s:%d: ", __func__, __LINE__);                \
        fprintf(stderr, __VA_ARGS__);                                       \
        fprintf(stderr, "\n");                                              \
        test_fails++;                                                       \
    }                                                                       \
} while (0)

static uint32_t test_rb32(const uint8_t *p)
{
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

static uint32_t test_rb16(const uint8_t *p)
{
    return ((uint32_t)p[0] << 8) | (uint32_t)p[1];
}

static void test_wb32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);
    p[3] = (uint8_t)v;
}

/* Rebuilds the full frame from the stored blob. The tile count is derived
 * here from the header's own fields, independently of the library, so that
 * a wrong count in either one moves the trailer and loses the record. */
static uint8_t *test_load_frame(size_t *out_size, size_t *out_tbl,
                                size_t *out_trailer_off)
{
    const char *path = TEST_SOURCE_DIR "/data/prr_lens_frame0.bin";
    uint8_t *blob = NULL, *frame = NULL;
    size_t blob_size, tbl, tile_bytes = 0, frame_size;
    uint32_t hdr_len, w, h, flags, align, tw16, nb_tw, nb_th, nb_tiles;
    long len;
    FILE *f = fopen(path, "rb");

    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    len = ftell(f);
    fseek(f, 0, SEEK_SET);
    blob = malloc((size_t)len);
    if (!blob || fread(blob, 1, (size_t)len, f) != (size_t)len) {
        fclose(f);
        free(blob);
        return NULL;
    }
    fclose(f);
    blob_size = (size_t)len;

    hdr_len = test_rb16(blob + 8);
    w     = test_rb16(blob + 10 + 6);
    h     = test_rb16(blob + 10 + 8);
    flags = test_rb16(blob + 10 + 68);
    align = (flags >> 1) & 7;

    tw16 = (w + 15) >> 4;
    nb_tw = tw16 >> align;
    for (uint32_t b = tw16 & ((1u << align) - 1u); b; b &= b - 1)
        nb_tw++;
    nb_th = (h + 15) >> 4;
    nb_tiles = nb_tw * nb_th;

    tbl = 8 + hdr_len;
    for (uint32_t i = 0; i < nb_tiles; i++)
        tile_bytes += test_rb16(blob + tbl + (size_t)i * 2);

    frame_size = tbl + (size_t)nb_tiles * 2 + tile_bytes +
                 (blob_size - tbl - (size_t)nb_tiles * 2);

    frame = calloc(1, frame_size);
    if (!frame) {
        free(blob);
        return NULL;
    }

    /* Header and tile table, then the zeroed tiles, then the trailer. */
    memcpy(frame, blob, tbl + (size_t)nb_tiles * 2);
    memcpy(frame + tbl + (size_t)nb_tiles * 2 + tile_bytes,
           blob + tbl + (size_t)nb_tiles * 2,
           blob_size - tbl - (size_t)nb_tiles * 2);

    if (out_tbl)
        *out_tbl = tbl;
    if (out_trailer_off)
        *out_trailer_off = tbl + (size_t)nb_tiles * 2 + tile_bytes;

    printf("frame %ux%u, header %u, %u tiles, %zu bytes of tile data, "
           "trailer at 0x%zx of 0x%zx\n", w, h, hdr_len, nb_tiles, tile_bytes,
           tbl + (size_t)nb_tiles * 2 + tile_bytes, frame_size);

    /* The blob's own size field has to be the frame's, or the walk is
     * checking against the wrong bound. */
    TEST_CHECK(test_rb32(frame) == frame_size,
               "the stored size field is %u, the rebuilt frame is %zu bytes",
               test_rb32(frame), frame_size);

    free(blob);
    *out_size = frame_size;

    return frame;
}

static void test_record(const CRAWLensCorrection *lc)
{
    float px[3];

    TEST_CHECK(lc->valid, "the record was not marked valid");
    TEST_CHECK(lc->centre[0] == 0.5f && lc->centre[1] == 0.5f,
               "centre (%g, %g), expected exactly (0.5, 0.5)",
               (double)lc->centre[0], (double)lc->centre[1]);

    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < 4; i++) {
            /* The library narrows the record's doubles to float, so the
             * two agree to the last bit of a float. */
            TEST_CHECK(lc->k[c][i] == test_k[c][i],
                       "k[%c][%d] = %.9g, expected %.9g", "RGB"[c], i,
                       (double)lc->k[c][i], (double)test_k[c][i]);
        }
    }

    TEST_CHECK(craw_lens_corner_shift(lc, FRAME_W, FRAME_H, false, px) == 0,
               "craw_lens_corner_shift failed");
    printf("  net corner shift  %+7.2f %+7.2f %+7.2f px\n",
           (double)px[0], (double)px[1], (double)px[2]);
    for (int c = 0; c < 3; c++)
        TEST_CHECK(fabsf(px[c] - test_net_px[c]) < 0.01f,
                   "net corner shift %c: %.2f px, expected %.2f",
                   "RGB"[c], (double)px[c], (double)test_net_px[c]);

    TEST_CHECK(craw_lens_corner_shift(lc, FRAME_W, FRAME_H, true, px) == 0,
               "craw_lens_corner_shift failed");
    printf("  distortion corrected %+7.2f %+7.2f %+7.2f px, lateral CA R-B %+.2f px\n",
           (double)px[0], (double)px[1], (double)px[2], (double)(px[0] - px[2]));
    for (int c = 0; c < 3; c++)
        TEST_CHECK(fabsf(px[c] - test_dist_px[c]) < 0.01f,
                   "distortion %c: %.2f px, expected %.2f",
                   "RGB"[c], (double)px[c], (double)test_dist_px[c]);
}

/* Every way of breaking the frame that the walk has to survive. The frame
 * is copied for each, so the checks are independent. */
static void test_malformed(const uint8_t *frame, size_t size, size_t tbl,
                           size_t trailer_off)
{
    CRAWLensCorrection lc;
    uint8_t *bad = malloc(size);

    if (!bad)
        return;

    TEST_CHECK(craw_lens_from_prores_raw(NULL, size, &lc) == CRAW_ERROR(EINVAL),
               "a NULL packet was not EINVAL");
    TEST_CHECK(craw_lens_from_prores_raw(frame, size, NULL) == CRAW_ERROR(EINVAL),
               "a NULL output was not EINVAL");

    /* Not a ProRes RAW frame. */
    memcpy(bad, frame, size);
    memcpy(bad + 4, "abcd", 4);
    TEST_CHECK(craw_lens_from_prores_raw(bad, size, &lc) == CRAW_ERROR(EINVAL),
               "a wrong fourcc was not EINVAL");
    TEST_CHECK(!lc.valid, "a failed parse left the correction valid");

    /* A size field that claims more than the caller handed over. */
    memcpy(bad, frame, size);
    test_wb32(bad, (uint32_t)size + 1);
    TEST_CHECK(craw_lens_from_prores_raw(bad, size, &lc) == CRAW_ERROR(EINVAL),
               "a size past the end of the packet was not EINVAL");

    /* A header that claims to run past the end of its own frame. */
    memcpy(bad, frame, size);
    test_wb32(bad, 200);
    bad[8] = 0xFF;
    bad[9] = 0xFF;
    TEST_CHECK(craw_lens_from_prores_raw(bad, size, &lc) == CRAW_ERROR(EINVAL),
               "a header longer than the frame was not EINVAL");

    /* A tile table that does not fit in what is left of the frame. */
    memcpy(bad, frame, size);
    test_wb32(bad, (uint32_t)(tbl + 8));
    TEST_CHECK(craw_lens_from_prores_raw(bad, size, &lc) == CRAW_ERROR(EINVAL),
               "a tile table past the end of the frame was not EINVAL");

    /* A header too short to hold the fields the walk reads. */
    memcpy(bad, frame, size);
    bad[8] = 0;
    bad[9] = 20;
    TEST_CHECK(craw_lens_from_prores_raw(bad, size, &lc) == CRAW_ERROR(EINVAL),
               "a 20 byte header was not EINVAL");

    /* A tile size that pushes the tile data past the end of the frame. */
    memcpy(bad, frame, size);
    bad[tbl + 0] = 0xFF;
    bad[tbl + 1] = 0xFF;
    TEST_CHECK(craw_lens_from_prores_raw(bad, size, &lc) == CRAW_ERROR(EINVAL),
               "a tile table overrunning the frame was not EINVAL");

    /* Damage inside the metadata list ends the search rather than failing
     * it: bytes after the last tile that are not a walkable list are how a
     * frame with no list at all looks, so the answer is "no record". */
    memcpy(bad, frame, size);
    test_wb32(bad + trailer_off, 0xFFFF);
    TEST_CHECK(craw_lens_from_prores_raw(bad, size, &lc) == CRAW_ERROR(ENOENT),
               "a record longer than the frame was not ENOENT");

    /* A key length that runs past the end of its own record. */
    memcpy(bad, frame, size);
    bad[trailer_off + 8] = 0xFF;
    TEST_CHECK(craw_lens_from_prores_raw(bad, size, &lc) == CRAW_ERROR(ENOENT),
               "a key longer than its record was not ENOENT");

    /* A payload version the layout is not known for. */
    memcpy(bad, frame, size);
    bad[trailer_off + 9 + 41 + 3] = 9;
    TEST_CHECK(craw_lens_from_prores_raw(bad, size, &lc) == CRAW_ERROR(EINVAL),
               "an unknown payload version was not EINVAL");

    /* A well formed frame with no such record: the key is changed, so the
     * list is walked to its `eomd` and nothing is found. */
    memcpy(bad, frame, size);
    bad[trailer_off + 9] = 'x';
    TEST_CHECK(craw_lens_from_prores_raw(bad, size, &lc) == CRAW_ERROR(ENOENT),
               "a frame without the record was not ENOENT");
    TEST_CHECK(!lc.valid, "a frame without the record left the correction valid");

    /* A record of a fourcc the parser does not know sits in front of ours
     * and must be stepped over, not tripped on. */
    memcpy(bad, frame, size);
    memcpy(bad + trailer_off + 4, "junk", 4);
    TEST_CHECK(craw_lens_from_prores_raw(bad, size, &lc) == CRAW_ERROR(ENOENT),
               "an unknown record type was not stepped over cleanly");

    free(bad);

    /* Truncation: every prefix of the frame must come back as one of the
     * three documented answers and must not read past what it was given.
     * Run under ASan or valgrind this is the bounds check. */
    for (size_t n = 0; n < 4096; n++) {
        int ret = craw_lens_from_prores_raw(frame, n, &lc);

        TEST_CHECK(ret == CRAW_ERROR(EINVAL),
                   "a %zu byte prefix returned %d, expected EINVAL", n, ret);
        if (ret != CRAW_ERROR(EINVAL))
            break;
    }
    for (size_t n = trailer_off - 16; n < size; n++) {
        int ret = craw_lens_from_prores_raw(frame, n, &lc);

        TEST_CHECK(ret == CRAW_ERROR(EINVAL),
                   "a %zu byte prefix of a %zu byte frame returned %d, "
                   "expected EINVAL", n, size, ret);
        if (ret != CRAW_ERROR(EINVAL))
            break;
    }
}


/* ---------------------------------------- the keep-everything scale (16.8) */

/* Everything below re-derives the scale a second way and checks the
 * library against it. The second way inverts r * poly(r) by Newton rather
 * than by bisection, and reaches the answer through the identity
 *
 *     frame_radius(theta) / needed(theta) = poly_c(invert_c(R(theta)))
 *
 * which holds because the source rectangle and the output frame are the
 * same rectangle: the boundary point at angle theta is exactly where the
 * output frame is hit along that ray. So the ratio depends on the boundary
 * only through its radius R, and for a polynomial that is monotone over
 * the range - which every real record is - the smallest ratio is at the
 * smallest or the largest R, i.e. at an edge midpoint or at a corner. The
 * test computes both candidates, says which one binds, and only then
 * compares against the library. */

/* The inverse of r * poly(r) by Newton from r_src itself, which is within
 * a few percent of the answer for anything lens-shaped. Not the library's
 * bisection, on purpose. */
static double test_keep_invert(const float k[4], double r_src)
{
    double r = r_src;

    for (int i = 0; i < 100; i++) {
        double r2 = r * r;
        double p = (double)k[0] + r2 * ((double)k[1] + r2 * ((double)k[2] + r2 * (double)k[3]));
        /* d/dr of r * poly(r) */
        double d = p + r2 * (2.0 * (double)k[1] + r2 * (4.0 * (double)k[2] + r2 * 6.0 * (double)k[3]));

        if (!(fabs(d) > 1e-12))
            break;
        r -= (r * p - r_src) / d;
    }

    return r;
}

/* The largest output radius the three channels need for a source point at
 * radius `r_src`, and the ratio that point allows. */
static double test_keep_ratio(const CRAWLensCorrection *lc, double r_src)
{
    double needed = 0.0;

    for (int c = 0; c < 3; c++)
        needed = fmax(needed, test_keep_invert(lc->k[c], r_src));

    return r_src / needed;
}

/* The mapping the mode defines, in pixels: where the source point at
 * (px, py) ends up once the picture has been scaled by `s`. */
static void test_keep_map(const CRAWLensCorrection *lc, double w, double h,
                          double s, double px, double py, double out[2])
{
    double cx = (double)lc->centre[0] * w;
    double cy = (double)lc->centre[1] * h;
    double hx = (double)(lc->centre[0] > 1.0f - lc->centre[0] ? lc->centre[0]
                                                              : 1.0f - lc->centre[0]) * w;
    double hy = (double)(lc->centre[1] > 1.0f - lc->centre[1] ? lc->centre[1]
                                                              : 1.0f - lc->centre[1]) * h;
    double half_diag = hypot(hx, hy);
    double dx = px - cx, dy = py - cy;
    double len = hypot(dx, dy);
    double needed = 0.0;

    for (int c = 0; c < 3; c++)
        needed = fmax(needed, test_keep_invert(lc->k[c], len / half_diag));

    out[0] = cx + dx / len * (s * needed * half_diag);
    out[1] = cy + dy / len * (s * needed * half_diag);
}

/* One record, one frame size: which of the two candidates binds, whether
 * the library agrees, and whether the answer actually keeps the source. */
static void test_keep_scale(const char *what, const CRAWLensCorrection *lc,
                            uint32_t width, uint32_t height, double want,
                            const char *want_where)
{
    double w = width, h = height;
    double hx = (double)(lc->centre[0] > 1.0f - lc->centre[0] ? lc->centre[0]
                                                              : 1.0f - lc->centre[0]) * w;
    double hy = (double)(lc->centre[1] > 1.0f - lc->centre[1] ? lc->centre[1]
                                                              : 1.0f - lc->centre[1]) * h;
    double half_diag = hypot(hx, hy);
    /* The three places a monotone ratio can bind: the corner, and the
     * midpoint of each pair of edges. */
    double r_corner = 1.0;
    double r_short = (w < h ? w : h) * 0.5 / half_diag;
    double r_long  = (w < h ? h : w) * 0.5 / half_diag;
    double cand[3] = { test_keep_ratio(lc, r_corner), test_keep_ratio(lc, r_short),
                       test_keep_ratio(lc, r_long) };
    const char *name[3] = { "corner", "short edge midpoint", "long edge midpoint" };
    int bind = 0;
    double expect, got = 0.0f, worst_in = 0.0;
    float s;
    int ret;

    for (int i = 1; i < 3; i++)
        if (cand[i] < cand[bind])
            bind = i;
    expect = cand[bind] < 1.0 ? cand[bind] : 1.0;

    /* The claim that only those three matter is the claim that the ratio is
     * monotone in R, so it is checked rather than assumed. */
    for (int i = 0; i <= 64; i++) {
        double r = r_short + (1.0 - r_short) * (double)i / 64.0;
        double v = test_keep_ratio(lc, r);

        TEST_CHECK(v >= cand[bind] - 1e-9,
                   "%s: the ratio at R %.4f is %.9f, below the %s's %.9f: the "
                   "polynomial turns over and the two candidates are not the whole "
                   "story", what, r, v, name[bind], cand[bind]);
    }

    ret = craw_lens_keep_all_scale(lc, width, height, &s);
    TEST_CHECK(ret == 0, "%s: craw_lens_keep_all_scale returned %d", what, ret);
    if (ret < 0)
        return;
    got = (double)s;

    printf("%s %ux%u: scale %.6f, binds at the %s (corner %.6f, short %.6f, "
           "long %.6f)\n", what, width, height, got, name[bind],
           cand[0], cand[1], cand[2]);

    TEST_CHECK(fabs(got - expect) <= 1e-6 * expect,
               "%s: the scale is %.9f, the %s says %.9f", what, got, name[bind], expect);
    if (want > 0.0)
        TEST_CHECK(fabs(got - want) <= 1e-6 * want,
                   "%s: the scale is %.9f, the closed form says %.9f", what, got, want);
    if (want_where)
        TEST_CHECK(!strcmp(name[bind], want_where),
                   "%s: the binding point is the %s, expected the %s", what,
                   name[bind], want_where);

    /* The contract itself: every source pixel has to land inside the output
     * frame. Sampled at points of the boundary the library does not use -
     * 331 a side, offset by half a step - so that agreeing with it takes
     * more than reproducing its sample positions. */
    for (int side = 0; side < 4; side++) {
        for (int i = 0; i < 331; i++) {
            double t = ((double)i + 0.5) / 331.0;
            double px, py, out[2], over;

            switch (side) {
            case 0:  px = t * w;         py = 0.0;           break;
            case 1:  px = w;             py = t * h;         break;
            case 2:  px = (1.0 - t) * w; py = h;             break;
            default: px = 0.0;           py = (1.0 - t) * h; break;
            }

            test_keep_map(lc, w, h, got, px, py, out);
            over = fmax(fmax(-out[0], out[0] - w), fmax(-out[1], out[1] - h));
            worst_in = fmax(worst_in, over);
            TEST_CHECK(over <= 1e-6 * w,
                       "%s: the source point (%.1f, %.1f) lands at (%.3f, %.3f), "
                       "%.4f px outside a %.0fx%.0f frame", what, px, py,
                       out[0], out[1], over, w, h);
        }
    }

    /* And that it is the *largest* such scale: a thousandth more has to put
     * something outside. Only meaningful when the cap at 1 is not what
     * decided the answer. */
    if (expect < 1.0) {
        double s_over = got * 1.001;
        bool escaped = false;

        for (int side = 0; side < 4 && !escaped; side++) {
            for (int i = 0; i <= 512; i++) {
                double t = (double)i / 512.0;
                double px, py, out[2];

                switch (side) {
                case 0:  px = t * w;         py = 0.0;           break;
                case 1:  px = w;             py = t * h;         break;
                case 2:  px = (1.0 - t) * w; py = h;             break;
                default: px = 0.0;           py = (1.0 - t) * h; break;
                }

                test_keep_map(lc, w, h, s_over, px, py, out);
                if (out[0] < -1e-9 || out[0] > w + 1e-9 ||
                    out[1] < -1e-9 || out[1] > h + 1e-9) {
                    escaped = true;
                    break;
                }
            }
        }

        TEST_CHECK(escaped, "%s: a scale 0.1%% larger still fits, so %.9f is not "
                   "the largest one", what, got);
    }

    printf("   worst source point lands %.4f px inside the frame edge\n", -worst_in);
}

/* r + k1 r^3 = 1 by Newton: the corner's output radius for a pure
 * quadratic barrel, whose reciprocal is the scale when the corner binds. */
static double test_keep_cubic_root(double k1)
{
    double r = 1.1;

    for (int i = 0; i < 200; i++)
        r -= (r + k1 * r * r * r - 1.0) / (1.0 + 3.0 * k1 * r * r);

    return r;
}

static void test_keep_all(const CRAWLensCorrection *real)
{
    /* A barrel correction ten times anything real, and a different
     * coefficient per channel so that the widest one decides. Purely
     * quadratic, so the corner's output radius is the root of a cubic and
     * the expected scale is a closed form. */
    static const CRAWLensCorrection barrel = {
        .k = { { 1.0f, -0.090f, 0.0f, 0.0f },
               { 1.0f, -0.100f, 0.0f, 0.0f },
               { 1.0f, -0.110f, 0.0f, 0.0f } },
        .centre = { 0.5f, 0.5f },
        .valid  = true,
    };
    /* No distortion at all, only the crop: every boundary point binds
     * equally and the scale is exactly k0. */
    static const CRAWLensCorrection pure_crop = {
        .k = { { 0.8f, 0.0f, 0.0f, 0.0f },
               { 0.8f, 0.0f, 0.0f, 0.0f },
               { 0.8f, 0.0f, 0.0f, 0.0f } },
        .centre = { 0.5f, 0.5f },
        .valid  = true,
    };
    /* Samples outward everywhere: the source already fits, and the mode
     * must not magnify it to fill the frame. */
    static const CRAWLensCorrection outward = {
        .k = { { 1.05f, 0.0f, 0.0f, 0.0f },
               { 1.05f, 0.0f, 0.0f, 0.0f },
               { 1.05f, 0.0f, 0.0f, 0.0f } },
        .centre = { 0.5f, 0.5f },
        .valid  = true,
    };
    CRAWLensCorrection invalid = { 0 };
    float s;

    /* The camera's own record, on the frame it was written for. A
     * pincushion correction pushes the corners out, so what is missing is
     * along the short edges and that is where it binds. */
    test_keep_scale("the real 60 mm record", real, FRAME_W, FRAME_H, 0.0,
                    "short edge midpoint");

    /* A barrel correction pulls the corners in, so the corner is the
     * hungriest point and the closed form is 1 / r_c with
     * r_c + k1 r_c^3 = 1 for the channel with the widest k1. */
    test_keep_scale("a 10x barrel", &barrel, FRAME_W, FRAME_H,
                    1.0 / test_keep_cubic_root(-0.110), "corner");
    test_keep_scale("a 10x barrel", &barrel, 512, 320,
                    1.0 / test_keep_cubic_root(-0.110), "corner");

    test_keep_scale("a pure crop", &pure_crop, FRAME_W, FRAME_H, 0.8, NULL);

    TEST_CHECK(craw_lens_keep_all_scale(&outward, FRAME_W, FRAME_H, &s) == 0 &&
               s == 1.0f, "a correction that samples outward everywhere gave a "
               "scale of %.6f, expected exactly 1", (double)s);

    TEST_CHECK(craw_lens_keep_all_scale(NULL, FRAME_W, FRAME_H, &s) == CRAW_ERROR(EINVAL),
               "a NULL correction was accepted");
    TEST_CHECK(craw_lens_keep_all_scale(real, FRAME_W, FRAME_H, NULL) == CRAW_ERROR(EINVAL),
               "a NULL output was accepted");
    TEST_CHECK(craw_lens_keep_all_scale(&invalid, FRAME_W, FRAME_H, &s) == CRAW_ERROR(EINVAL),
               "an invalid correction was accepted");
    TEST_CHECK(craw_lens_keep_all_scale(real, 0, FRAME_H, &s) == CRAW_ERROR(EINVAL),
               "a zero width was accepted");
}

/* ---------------------------------------------- turning the picture (16.9) */

/* The scale of a turned frame, both ways, checked against its contract
 * rather than against a second copy of the walk: filling, every point of
 * the output frame's outline has a source in all three channels and a
 * scale a thousandth smaller leaves one without; keeping everything, every
 * point of the source's outline lands inside the turned frame and a scale a
 * thousandth larger puts one outside. The outlines are sampled where the
 * library does not sample them, 331 points a side offset by half a step,
 * the way the keep-everything test above samples them. */

#define TEST_DEG (3.14159265358979323846 / 180.0)

/* tools/prr_lens.md section 5, P1000261 at 24 mm: the barrel end of the
 * 24-60, where the record's own frame throws away 209 px of the source at
 * each corner. */
static const CRAWLensCorrection test_w24 = {
    .k = { { 0.99965795f, -1.19016e-01f, 6.71727e-02f, -1.21451e-02f },
           { 0.99908949f, -1.18440e-01f, 6.67320e-02f, -1.19222e-02f },
           { 0.99902956f, -1.17697e-01f, 6.61432e-02f, -1.16425e-02f } },
    .centre = { 0.5f, 0.5f },
    .valid  = true,
};

/* No correction: what a frame without a record is turned with. */
static const CRAWLensCorrection test_none = {
    .k = { { 1.0f, 0.0f, 0.0f, 0.0f },
           { 1.0f, 0.0f, 0.0f, 0.0f },
           { 1.0f, 0.0f, 0.0f, 0.0f } },
    .centre = { 0.5f, 0.5f },
    .valid  = true,
};

static void test_outline(int side, double t, double w, double h, double *px, double *py)
{
    switch (side) {
    case 0:  *px = t * w;         *py = 0.0;           break;
    case 1:  *px = w;             *py = t * h;         break;
    case 2:  *px = (1.0 - t) * w; *py = h;             break;
    default: *px = 0.0;           *py = (1.0 - t) * h; break;
    }
}

static double test_half_diag(const CRAWLensCorrection *lc, double w, double h)
{
    double hx = (double)(lc->centre[0] > 1.0f - lc->centre[0] ? lc->centre[0]
                                                              : 1.0f - lc->centre[0]) * w;
    double hy = (double)(lc->centre[1] > 1.0f - lc->centre[1] ? lc->centre[1]
                                                              : 1.0f - lc->centre[1]) * h;

    return hypot(hx, hy);
}

/* How far outside the frame the furthest of an output point's three
 * sources lies, in pixels, filling at scale `s` turned `deg` clockwise:
 * the corrected position is the output offset turned back and divided by
 * the scale, and each channel's source is on its ray at r * poly(r). */
static double test_fill_over(const CRAWLensCorrection *lc, double w, double h,
                             double deg, double s, double px, double py)
{
    double cx = lc->centre[0] * w, cy = lc->centre[1] * h;
    double half_diag = test_half_diag(lc, w, h);
    double ct = cos(deg * TEST_DEG), st = sin(deg * TEST_DEG);
    double dx = px - cx, dy = py - cy;
    double qx = (ct * dx + st * dy) / s, qy = (ct * dy - st * dx) / s;
    double r = hypot(qx, qy) / half_diag, worst = -HUGE_VAL;

    for (int c = 0; c < 3; c++) {
        const float *k = lc->k[c];
        double r2 = r * r;
        double poly = k[0] + r2 * (k[1] + r2 * (k[2] + r2 * k[3]));
        double sx = cx + qx * poly, sy = cy + qy * poly;

        worst = fmax(worst, fmax(fmax(-sx, sx - w), fmax(-sy, sy - h)));
    }

    return worst;
}

/* The same for a source point keeping everything: where it lands in the
 * output, turned, and how far outside the frame that is. The channel that
 * needs the largest output radius is the one that has to fit. */
static double test_keep_over(const CRAWLensCorrection *lc, double w, double h,
                             double deg, double s, double px, double py)
{
    double cx = lc->centre[0] * w, cy = lc->centre[1] * h;
    double half_diag = test_half_diag(lc, w, h);
    double ct = cos(deg * TEST_DEG), st = sin(deg * TEST_DEG);
    double dx = px - cx, dy = py - cy, len = hypot(dx, dy), needed = 0.0;
    double qx, qy, ox, oy;

    if (!(len > 0.0))
        return -HUGE_VAL;

    for (int c = 0; c < 3; c++)
        needed = fmax(needed, test_keep_invert(lc->k[c], len / half_diag));

    /* Corrected position, then turned clockwise on screen: with y down
     * that is (x cos - y sin, x sin + y cos). */
    qx = dx / len * needed * half_diag;
    qy = dy / len * needed * half_diag;
    ox = cx + s * (ct * qx - st * qy);
    oy = cy + s * (st * qx + ct * qy);

    return fmax(fmax(-ox, ox - w), fmax(-oy, oy - h));
}

/* The worst point of an outline under one of the two contracts. */
static double test_turn_worst(const CRAWLensCorrection *lc, double w, double h,
                              double deg, double s, bool keep_all, int segs,
                              double offset)
{
    double worst = -HUGE_VAL;

    for (int side = 0; side < 4; side++) {
        for (int i = 0; i < segs; i++) {
            double px, py;

            test_outline(side, ((double)i + offset) / segs, w, h, &px, &py);
            worst = fmax(worst, keep_all ? test_keep_over(lc, w, h, deg, s, px, py)
                                         : test_fill_over(lc, w, h, deg, s, px, py));
        }
    }

    return worst;
}

static float test_turn_scale(const char *what, const CRAWLensCorrection *lc,
                             uint32_t width, uint32_t height, float deg, bool keep_all)
{
    double w = width, h = height, worst;
    float s = 0.0f;
    int ret;

    ret = craw_lens_rotation_scale(lc, width, height, deg, keep_all, &s);
    TEST_CHECK(ret == 0, "%s at %+.2f deg %s: craw_lens_rotation_scale returned %d",
               what, (double)deg, keep_all ? "keep" : "fill", ret);
    if (ret < 0)
        return 0.0f;

    /* The contract at the answer. */
    worst = test_turn_worst(lc, w, h, deg, s, keep_all, 331, 0.5);
    TEST_CHECK(worst <= 1e-6 * w, "%s at %+.2f deg %s: scale %.6f leaves a point %.4f px "
               "outside the frame", what, (double)deg, keep_all ? "keep" : "fill",
               (double)s, worst);

    /* And that it is the tightest one, unless the cap decided: a fill that
     * needs no magnification sits at exactly 1, and so does a keep that
     * needs no shrink. */
    if (s != 1.0f) {
        double tighter = keep_all ? (double)s * 1.001 : (double)s / 1.001;
        double over = test_turn_worst(lc, w, h, deg, tighter, keep_all, 512, 0.0);

        TEST_CHECK(over > 0.0, "%s at %+.2f deg %s: a scale 0.1%% %s than %.6f still "
                   "holds, so it is not the %s one", what, (double)deg,
                   keep_all ? "keep" : "fill", keep_all ? "larger" : "smaller",
                   (double)s, keep_all ? "largest" : "smallest");
    }

    return s;
}

/* No correction is a rectangle turned inside a rectangle, which has a
 * closed form: the turned frame's extent along each axis over the frame's,
 * whichever is larger, and keeping everything is its reciprocal. */
static double test_turn_closed(double w, double h, double deg)
{
    double ct = cos(deg * TEST_DEG), st = fabs(sin(deg * TEST_DEG));

    return fmax((w * st + h * ct) / h, (w * ct + h * st) / w);
}

static void test_rotation(const CRAWLensCorrection *real)
{
    static const float degs[] = { 0.01f, 0.5f, 2.0f, 5.0f, 12.5f, 30.0f, 44.9f };
    static const struct {
        const char *name;
        const CRAWLensCorrection *lc;
    } recs[] = {
        { "no correction", &test_none },
        { "the real 24 mm record", &test_w24 },
        { "the real 60 mm record", NULL },
    };
    CRAWLensCorrection invalid = { 0 };
    float s, s2;

    /* Every record, both ways, both directions: the contract. */
    for (size_t r = 0; r < CRAW_ARRAY_ELEMS(recs); r++) {
        const CRAWLensCorrection *lc = recs[r].lc ? recs[r].lc : real;

        for (size_t i = 0; i < CRAW_ARRAY_ELEMS(degs); i++) {
            for (int sign = -1; sign <= 1; sign += 2) {
                float deg = (float)sign * degs[i];
                float fill = test_turn_scale(recs[r].name, lc, FRAME_W, FRAME_H, deg, false);
                float keep = test_turn_scale(recs[r].name, lc, FRAME_W, FRAME_H, deg, true);

                if (sign > 0)
                    printf("%s, turned %5.2f deg: fill %.5f, keep %.5f (no correction "
                           "%.5f)\n", recs[r].name, (double)deg, (double)fill,
                           (double)keep, test_turn_closed(FRAME_W, FRAME_H, deg));
            }
        }
    }

    /* No correction against the closed form, which a walk of the outline
     * has no way to know: the binding point of a turned rectangle is a
     * corner, which the walk lands on, and nothing else. NULL and an
     * invalid record are both that. */
    for (size_t i = 0; i < CRAW_ARRAY_ELEMS(degs); i++) {
        double want = test_turn_closed(FRAME_W, FRAME_H, degs[i]);

        s = s2 = 0.0f;
        craw_lens_rotation_scale(NULL, FRAME_W, FRAME_H, degs[i], false, &s);
        craw_lens_rotation_scale(&invalid, FRAME_W, FRAME_H, -degs[i], true, &s2);
        TEST_CHECK(fabs(s - want) <= 1e-6 * want,
                   "no correction at %.2f deg fills at %.7f, the closed form says %.7f",
                   (double)degs[i], (double)s, want);
        TEST_CHECK(fabs(s2 - 1.0 / want) <= 1e-6 / want,
                   "no correction at %.2f deg keeps at %.7f, the closed form says %.7f",
                   -(double)degs[i], (double)s2, 1.0 / want);
    }

    /* Unturned: filling is the record's own frame, exactly 1 whatever the
     * record does, and keeping everything is craw_lens_keep_all_scale() to
     * the bit, so that a frame at 0 degrees is the frame it always was. */
    for (size_t r = 0; r < CRAW_ARRAY_ELEMS(recs); r++) {
        const CRAWLensCorrection *lc = recs[r].lc ? recs[r].lc : real;

        s = 0.0f;
        TEST_CHECK(craw_lens_rotation_scale(lc, FRAME_W, FRAME_H, 0.0f, false, &s) == 0 &&
                   s == 1.0f, "%s unturned fills at %.9f, not exactly 1", recs[r].name,
                   (double)s);
        TEST_CHECK(craw_lens_rotation_scale(lc, FRAME_W, FRAME_H, 0.0f, true, &s) == 0 &&
                   craw_lens_keep_all_scale(lc, FRAME_W, FRAME_H, &s2) == 0 && s == s2,
                   "%s unturned keeps at %.9f, craw_lens_keep_all_scale says %.9f",
                   recs[r].name, (double)s, (double)s2);

        /* And turning starts from there rather than jumping: every record
         * seen keeps its own frame on the source, so the first thousandth
         * of a degree costs nothing measurable either way. */
        craw_lens_rotation_scale(lc, FRAME_W, FRAME_H, 1e-3f, false, &s);
        craw_lens_rotation_scale(lc, FRAME_W, FRAME_H, -1e-3f, true, &s2);
        TEST_CHECK(s >= 1.0f && s - 1.0f < 1e-4f,
                   "%s at a thousandth of a degree fills at %.7f", recs[r].name, (double)s);
        craw_lens_rotation_scale(lc, FRAME_W, FRAME_H, 0.0f, true, &s);
        TEST_CHECK(fabsf(s2 - s) < 1e-4f,
                   "%s at a thousandth of a degree keeps at %.7f against %.7f unturned",
                   recs[r].name, (double)s2, (double)s);
    }

    /* What the whole thing is for. At the wide end the correction pulls
     * the source's corners out past the frame and the record's own frame
     * throws them away; a turn swings the frame's corners out into them.
     * Up to a couple of degrees that costs no magnification at all, where
     * the uncorrected frame already needs 6.6%; past it, it still costs
     * clearly less. */
    craw_lens_rotation_scale(&test_w24, FRAME_W, FRAME_H, 2.0f, false, &s);
    TEST_CHECK(s == 1.0f, "the 24 mm record turned 2 deg fills at %.6f; its thrown away "
               "corners should cover the turn", (double)s);
    craw_lens_rotation_scale(&test_w24, FRAME_W, FRAME_H, 5.0f, false, &s);
    TEST_CHECK(s < test_turn_closed(FRAME_W, FRAME_H, 5.0) - 0.05,
               "the 24 mm record turned 5 deg fills at %.6f, hardly better than the "
               "%.6f with no correction", (double)s, test_turn_closed(FRAME_W, FRAME_H, 5.0));
    /* The pincushion end throws away the bulge beyond each edge under k0's
     * crop, which the turned corners reach less of; it still has to help
     * and never hurt. */
    craw_lens_rotation_scale(real, FRAME_W, FRAME_H, 5.0f, false, &s);
    TEST_CHECK(s < test_turn_closed(FRAME_W, FRAME_H, 5.0),
               "the 60 mm record turned 5 deg fills at %.6f, no better than the %.6f "
               "with no correction", (double)s, test_turn_closed(FRAME_W, FRAME_H, 5.0));

    /* Either direction is the same magnification for a centred record, the
     * frame being symmetric about both axes. */
    for (size_t i = 0; i < CRAW_ARRAY_ELEMS(degs); i++) {
        craw_lens_rotation_scale(&test_w24, FRAME_W, FRAME_H, degs[i], false, &s);
        craw_lens_rotation_scale(&test_w24, FRAME_W, FRAME_H, -degs[i], false, &s2);
        TEST_CHECK(fabsf(s - s2) <= 1e-6f * s, "the 24 mm record fills at %.7f turned "
                   "%.2f deg and %.7f turned back", (double)s, (double)degs[i], (double)s2);
    }

    s = 0.0f;
    TEST_CHECK(craw_lens_rotation_scale(real, FRAME_W, FRAME_H, 45.0f, false, &s) ==
               CRAW_ERROR(EINVAL), "a turn of 45 degrees was accepted");
    TEST_CHECK(craw_lens_rotation_scale(real, FRAME_W, FRAME_H, -45.0f, true, &s) ==
               CRAW_ERROR(EINVAL), "a turn of -45 degrees was accepted");
    TEST_CHECK(craw_lens_rotation_scale(real, FRAME_W, FRAME_H, NAN, false, &s) ==
               CRAW_ERROR(EINVAL), "a turn that is not a number was accepted");
    TEST_CHECK(craw_lens_rotation_scale(real, FRAME_W, FRAME_H, 1.0f, false, NULL) ==
               CRAW_ERROR(EINVAL), "a NULL output was accepted");
    TEST_CHECK(craw_lens_rotation_scale(NULL, FRAME_W, 0, 1.0f, false, &s) ==
               CRAW_ERROR(EINVAL), "a zero height was accepted");
}

#if CRAW_TEST_HAVE_LAVF
/* The same parser on a packet straight out of libavformat, when a clip is
 * pointed at. Nothing is decoded. */
static void test_real_clip(const char *path)
{
    AVFormatContext *fmt = NULL;
    AVPacket *pkt = NULL;
    int stream = -1, nb = 0;

    if (avformat_open_input(&fmt, path, NULL, NULL) < 0) {
        fprintf(stderr, "CRAW_TEST_CLIP: cannot open %s, skipping\n", path);
        return;
    }
    avformat_find_stream_info(fmt, NULL);

    for (unsigned i = 0; i < fmt->nb_streams; i++) {
        if (fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            stream = (int)i;
            break;
        }
    }
    if (stream < 0 ||
        fmt->streams[stream]->codecpar->codec_id != AV_CODEC_ID_PRORES_RAW) {
        printf("%s is not a ProRes RAW clip, skipping the real packet check\n", path);
        avformat_close_input(&fmt);
        return;
    }

    pkt = av_packet_alloc();
    while (nb < 2 && av_read_frame(fmt, pkt) >= 0) {
        CRAWLensCorrection lc;
        int ret;

        if (pkt->stream_index != stream) {
            av_packet_unref(pkt);
            continue;
        }

        ret = craw_lens_from_prores_raw(pkt->data, (size_t)pkt->size, &lc);
        printf("%s packet %d (%d bytes, pts %" PRId64 "): %s\n", path, nb,
               pkt->size, pkt->pts,
               ret == 0 ? "record found" :
               ret == CRAW_ERROR(ENOENT) ? "no record in this clip" : "error");
        /* A ProRes RAW clip from another camera legitimately carries no
         * record; only a hard error is a failure. */
        TEST_CHECK(ret == 0 || ret == CRAW_ERROR(ENOENT),
                   "the real packet failed to walk: %d", ret);
        if (ret == 0 && !nb)
            test_record(&lc);
        nb++;
        av_packet_unref(pkt);
    }

    TEST_CHECK(nb > 0, "no video packet was read from %s", path);

    av_packet_free(&pkt);
    avformat_close_input(&fmt);
}
#endif

int main(void)
{
    CRAWLensCorrection lc;
    size_t size = 0, tbl = 0, trailer_off = 0;
    uint8_t *frame;
    int ret;

    frame = test_load_frame(&size, &tbl, &trailer_off);
    if (!frame) {
        fprintf(stderr, "cannot rebuild the stored frame\n");
        return 1;
    }

    ret = craw_lens_from_prores_raw(frame, size, &lc);
    if (ret < 0) {
        fprintf(stderr, "craw_lens_from_prores_raw on the stored frame: %d\n", ret);
        free(frame);
        return 1;
    }
    test_record(&lc);

    test_keep_all(&lc);

    test_rotation(&lc);

    test_malformed(frame, size, tbl, trailer_off);

    free(frame);

#if CRAW_TEST_HAVE_LAVF
    {
        const char *clip = getenv("CRAW_TEST_CLIP");

        if (clip)
            test_real_clip(clip);
        else
            printf("CRAW_TEST_CLIP is not set, skipping the real packet check\n");
    }
#else
    printf("built without libavformat, skipping the real packet check\n");
#endif

    printf("%s\n", test_fails ? "FAILED" : "all lens record checks passed");

    return test_fails ? 1 : 0;
}
