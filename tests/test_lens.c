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
