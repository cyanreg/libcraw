/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* The two records a Panasonic clip keeps where FFmpeg's demuxer does not
 * look, found by walking the file's boxes here.
 *
 * moov/udta/PANA is in every clip the camera writes - MOV and MP4, HEVC,
 * H.264 and ProRes RAW alike - and holds a fixed header, a JPEG of the
 * first frame, zero padding, and then at a fixed offset an Exif block
 * describing that JPEG: an SOI and an APP1 with no picture behind them.
 * Its Exif IFD has the ISO, the shutter, the aperture, the focal length
 * and the lens, and Panasonic's maker note in there has the white balance
 * the camera applied: the mode, the colour temperature - on an auto clip
 * the one it settled on for the take - and the multipliers. A developed
 * clip has none of that anywhere else. The offsets are per camera
 * generation and are exiftool's (its Panasonic::PANA table); each layout is
 * checked against the JPEG and Exif signatures before anything is read,
 * so a body with a layout not listed here reads as a clip without a maker
 * note. The TIFF itself goes to libavcodec's Exif parser, which knows
 * Panasonic's maker note.
 *
 * A top-level meta box with a 'psmd' handler ("Panasonic Static
 * Metadata"), in MP4 clips only, holds an ISO 'xml ' box with the same
 * clip XML a MOV carries as its com.panasonic.Semi-Pro.metadata.xml tag.
 * That document is where the capture gamma lives, so without it an MP4
 * V-Log clip would read as the BT.709 its VUI claims.
 *
 * The file is opened a second time, through FFmpeg's I/O layer so that
 * whatever the demuxer could open this can too, and nothing is read but
 * box headers, three bytes of JPEG, a dozen of APP1 and the Exif itself:
 * the PANA box is two megabytes and most of it is the picture. */

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include <libavcodec/exif.h>
#include <libavformat/avio.h>
#include <libavutil/intreadwrite.h>
#include <libavutil/macros.h>
#include <libavutil/mem.h>

#include "view.h"

/* Enough for any file a camera writes at one level - a MOV has seven boxes
 * at the top and fewer in udta - and a bound on a hostile one. */
#define VIEW_BOX_MAX 256
/* The clip XML is under two kilobytes; a megabyte is not a clip XML. */
#define VIEW_XML_MAX (1 << 20)

struct view_box {
    uint32_t type;
    int64_t start;              /* the payload */
    int64_t end;
};

/* The box at `pos`, which has to lie inside `end`. A size of 1 means a
 * 64-bit size follows the type; 0 means the box runs to the end. */
static bool view_box_at(AVIOContext *pb, int64_t pos, int64_t end,
                        struct view_box *b)
{
    uint64_t size;
    int hdr = 8;

    if (end - pos < 8 || avio_seek(pb, pos, SEEK_SET) < 0)
        return false;

    size = avio_rb32(pb);
    b->type = avio_rb32(pb);
    if (size == 1) {
        size = avio_rb64(pb);
        hdr = 16;
    } else if (!size) {
        size = (uint64_t)(end - pos);
    }

    if (avio_feof(pb) || size < (uint64_t)hdr || size > (uint64_t)(end - pos))
        return false;

    b->start = pos + hdr;
    b->end = pos + (int64_t)size;

    return true;
}

/* The first box of `type` among the ones from `pos` to `end`. */
static bool view_box_find(AVIOContext *pb, int64_t pos, int64_t end,
                          uint32_t type, struct view_box *b)
{
    for (int i = 0; i < VIEW_BOX_MAX && view_box_at(pb, pos, end, b); i++) {
        if (b->type == type)
            return true;
        pos = b->end;
    }

    return false;
}

static bool view_box_read(AVIOContext *pb, int64_t pos, void *buf, int size)
{
    return avio_seek(pb, pos, SEEK_SET) >= 0 && avio_read(pb, buf, size) == size;
}

/* A number out of an IFD whatever the camera declared it as. */
static bool view_exif_number(AVExifMetadata *ifd, uint16_t id, double *out)
{
    AVExifEntry *e = NULL;

    if (av_exif_get_entry(NULL, ifd, id, 0, &e) <= 0 || !e->count)
        return false;

    switch (e->type) {
    case AV_TIFF_SHORT:
    case AV_TIFF_LONG:
        *out = (double)e->value.uint[0];
        return true;
    case AV_TIFF_SSHORT:
    case AV_TIFF_SLONG:
        *out = (double)e->value.sint[0];
        return true;
    case AV_TIFF_RATIONAL:
    case AV_TIFF_SRATIONAL:
        if (!e->value.rat[0].den)
            return false;
        *out = av_q2d(e->value.rat[0]);
        return true;
    case AV_TIFF_FLOAT:
    case AV_TIFF_DOUBLE:
        *out = e->value.dbl[0];
        return true;
    default:
        return false;
    }
}

static void view_exif_string(AVExifMetadata *ifd, uint16_t id,
                             char *out, size_t size)
{
    AVExifEntry *e = NULL;
    size_t n;

    if (av_exif_get_entry(NULL, ifd, id, 0, &e) <= 0 ||
        e->type != AV_TIFF_STRING || !e->value.str)
        return;

    /* libavcodec terminates the value past its count; Exif pads its
     * strings with spaces as often as it ends them. */
    n = strlen(e->value.str);
    while (n && (unsigned char)e->value.str[n - 1] <= ' ')
        n--;
    if (n && n < size) {
        memcpy(out, e->value.str, n);
        out[n] = '\0';
    }
}

/* The Exif IFD, and in it the maker note, which is only read when it opens
 * with Panasonic's lead: another maker's note has the same tag numbers
 * meaning other things. */
static void view_pana_tiff(const uint8_t *tiff, int size, struct view_pana *p)
{
    AVExifMetadata ifd = { 0 };
    AVExifEntry *exif = NULL, *mn = NULL;
    double v, rgb[3];

    if (av_exif_parse_buffer(NULL, tiff, size, &ifd, AV_EXIF_TIFF_HEADER) < 0)
        return;

    if (av_exif_get_entry(NULL, &ifd, 0x8769, 0, &exif) <= 0 ||
        exif->type != AV_TIFF_IFD)
        goto end;

    if (view_exif_number(&exif->value.ifd, 0x8827, &v) && v > 0.0)
        p->iso = (float)v;
    if (view_exif_number(&exif->value.ifd, 0x829a, &v) && v > 0.0)
        p->exposure_s = (float)v;
    if (view_exif_number(&exif->value.ifd, 0x829d, &v) && v > 0.0)
        p->fnumber = (float)v;
    if (view_exif_number(&exif->value.ifd, 0x920a, &v) && v > 0.0)
        p->focal_mm = (float)v;
    view_exif_string(&exif->value.ifd, 0xa434, p->lens, sizeof(p->lens));

    if (av_exif_get_entry(NULL, &exif->value.ifd, 0x927c, 0, &mn) <= 0 ||
        mn->type != AV_TIFF_IFD || mn->ifd_offset < 9 || !mn->ifd_lead ||
        memcmp(mn->ifd_lead, "Panasonic", 9))
        goto end;

    if (view_exif_number(&mn->value.ifd, 0x0003, &v) && v > 0.0)
        p->wb_mode = (int)v;
    if (view_exif_number(&mn->value.ifd, 0x0044, &v) && v > 0.0)
        p->wb_kelvin = (int)v;

    /* WBRedLevel, WBGreenLevel, WBBlueLevel: green is 1024 on every clip
     * seen, and is divided out rather than assumed. */
    if (view_exif_number(&mn->value.ifd, 0x8004, &rgb[0]) &&
        view_exif_number(&mn->value.ifd, 0x8005, &rgb[1]) &&
        view_exif_number(&mn->value.ifd, 0x8006, &rgb[2]) &&
        rgb[0] > 0.0 && rgb[1] > 0.0 && rgb[2] > 0.0)
        for (int i = 0; i < 3; i++)
            p->wb_levels[i] = (float)(rgb[i] / rgb[1]);

end:
    av_exif_free(&ifd);
}

/* Where the preview and the Exif block sit in the PANA payload, by camera
 * generation, as exiftool has them. */
static const struct {
    int preview, exif;
} view_pana_layouts[] = {
    { 0x74, 0x200080 },         /* GH6 and later, which is the S1M2 */
    { 0x5c, 0x4080 },           /* FZ1000 */
    { 0x5c, 0x4068 },           /* older bodies */
};

static void view_pana_box(AVIOContext *pb, const struct view_box *b,
                          struct view_pana *p)
{
    uint8_t soi[3], app1[12], *tiff;
    int64_t at = -1;
    int size;

    for (int i = 0; i < FF_ARRAY_ELEMS(view_pana_layouts) && at < 0; i++) {
        int64_t pic = b->start + view_pana_layouts[i].preview;
        int64_t exif = b->start + view_pana_layouts[i].exif;

        if (exif + (int64_t)sizeof(app1) > b->end ||
            !view_box_read(pb, pic, soi, sizeof(soi)) ||
            !view_box_read(pb, exif, app1, sizeof(app1)))
            continue;

        if (AV_RB24(soi) == 0xffd8ff && AV_RB32(app1) == 0xffd8ffe1 &&
            !memcmp(app1 + 6, "Exif\0\0", 6))
            at = exif + (int64_t)sizeof(app1);
    }
    if (at < 0)
        return;

    /* The APP1 length counts itself and the six bytes of "Exif\0\0". */
    size = AV_RB16(app1 + 4) - 8;
    if (size < 8 || at + size > b->end)
        return;

    tiff = av_malloc(size);
    if (!tiff)
        return;
    if (view_box_read(pb, at, tiff, size))
        view_pana_tiff(tiff, size, p);
    av_free(tiff);
}

/* The MP4's clip XML: a meta full box whose handler is psmd, then an 'xml '
 * full box whose payload is the document. */
static char *view_pana_xml(AVIOContext *pb, const struct view_box *meta)
{
    struct view_box b;
    uint8_t hdlr[12];
    int64_t size;
    char *xml;

    if (!view_box_find(pb, meta->start + 4, meta->end, MKBETAG('h','d','l','r'), &b) ||
        b.end - b.start < (int64_t)sizeof(hdlr) ||
        !view_box_read(pb, b.start, hdlr, sizeof(hdlr)) ||
        AV_RB32(hdlr + 8) != MKBETAG('p','s','m','d'))
        return NULL;

    if (!view_box_find(pb, meta->start + 4, meta->end, MKBETAG('x','m','l',' '), &b))
        return NULL;

    size = b.end - b.start - 4;
    if (size <= 0 || size > VIEW_XML_MAX)
        return NULL;

    xml = av_malloc(size + 1);
    if (!xml)
        return NULL;
    if (!view_box_read(pb, b.start + 4, xml, (int)size)) {
        av_free(xml);
        return NULL;
    }
    xml[size] = '\0';

    return xml;
}

void view_pana_read(const char *path, struct view_pana *p, char **xml)
{
    AVIOContext *pb = NULL;
    struct view_box moov, udta, box;
    int64_t end;

    memset(p, 0, sizeof(*p));
    *xml = NULL;

    if (avio_open2(&pb, path, AVIO_FLAG_READ, NULL, NULL) < 0)
        return;

    end = avio_size(pb);
    if (end < 0)
        end = INT64_MAX;

    if (view_box_find(pb, 0, end, MKBETAG('m','o','o','v'), &moov) &&
        view_box_find(pb, moov.start, moov.end, MKBETAG('u','d','t','a'), &udta) &&
        view_box_find(pb, udta.start, udta.end, MKBETAG('P','A','N','A'), &box))
        view_pana_box(pb, &box, p);

    if (view_box_find(pb, 0, end, MKBETAG('m','e','t','a'), &box))
        *xml = view_pana_xml(pb, &box);

    avio_closep(&pb);
}
