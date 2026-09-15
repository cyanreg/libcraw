/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* The embedded film simulation collection: reader for the pack that
 * tools/luts_build.py produces. See luts.h for what the API is for.
 *
 * The pack is one contiguous blob compiled into the executable. Its header
 * points at four fixed-stride tables (renderings, films, film categories,
 * look categories), a region of u32 index lists the look categories point
 * into, a NUL terminated string table, and then one independently
 * compressed blob per rendering. init() only walks the tables and hands out
 * pointers into the string table, so opening the menu costs nothing; the
 * compressed data is touched exactly once, when a LUT is picked.
 *
 * Every field is little endian and read through memcpy rather than a cast:
 * the LUT table's stride is 80 bytes, so its fields are not naturally
 * aligned and a struct overlay would be undefined behaviour on the strict
 * targets and slower on the forgiving ones. */

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <zlib.h>
#if CRAW_VIEW_LUTS_HAVE_ZSTD
#include <zstd.h>
#endif

#include <craw/utils.h>

#include "luts.h"

/* The pack itself, generated at build time by blob_embed.py. */
#include "filmgrid_pack.h"

#define PACK_MAGIC   "CRAWLUT1"
#define PACK_VERSION 1

/* Fixed strides, kept in step with the struct formats in luts_build.py. */
#define HDR_SIZE  64
#define LUT_SIZE  80
#define FILM_SIZE 28
#define FCAT_SIZE 16
#define LCAT_SIZE 28

enum PackCompression {
    PACK_ZLIB = 0,
    PACK_ZSTD = 1,
};

static struct {
    bool ready;
    const uint8_t *base;
    size_t len;
    uint32_t compression;
    const uint8_t *dir, *films, *fcats, *lcats;
    const char *strtab;
    size_t strtab_len;

    int n_luts, n_films, n_fcats, n_lcats;

    /* The public views. Parsed once at init(): a few tens of kilobytes of
     * pointers, against which a per-call decode would buy nothing. */
    ViewLut *luts;
    ViewLutFilm *film_list;
    ViewLutFilmCategory *fcat_list;
    ViewLutLookCategory *lcat_list;
} pack;

static inline uint32_t rd32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline uint16_t rd16(const uint8_t *p)
{
    return (uint16_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8));
}

static inline float rdf32(const uint8_t *p)
{
    uint32_t u = rd32(p);
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

/* String table offsets come out of a file, so they are not trusted: an
 * offset past the end, or one whose string is not terminated inside the
 * table, yields the empty string rather than a read past the array. */
static const char *pack_str(uint32_t off)
{
    if (off >= pack.strtab_len)
        return "";
    if (!memchr(pack.strtab + off, '\0', pack.strtab_len - off))
        return "";
    return pack.strtab + off;
}

/* The look categories' index lists live in one region of the file; hand out
 * a pointer only when the whole array is inside it. */
static const uint32_t *pack_list(uint32_t off, uint32_t n, uint32_t lo,
                                 uint32_t hi)
{
    if (!n)
        return NULL;
    if (off < lo || off > hi || (hi - off) / 4 < n)
        return NULL;
    return (const uint32_t *)(const void *)(pack.base + off);
}

int view_luts_init(void)
{
    const uint8_t *p = craw_view_filmgrid_pack_data;
    size_t len = craw_view_filmgrid_pack_len;
    uint32_t dir_off, films_off, fcat_off, lcat_off, lists_off;
    uint32_t strtab_off, strtab_len, total;

    if (pack.ready)
        return 0;

    if (len < HDR_SIZE || memcmp(p, PACK_MAGIC, 8))
        return CRAW_ERROR(EINVAL);
    if (rd32(p + 8) != PACK_VERSION)
        return CRAW_ERROR(EINVAL);

    pack.compression = rd32(p + 12);
    pack.n_luts  = (int)rd32(p + 16);
    pack.n_films = (int)rd32(p + 20);
    pack.n_fcats = (int)rd32(p + 24);
    pack.n_lcats = (int)rd32(p + 28);
    dir_off    = rd32(p + 32);
    films_off  = rd32(p + 36);
    fcat_off   = rd32(p + 40);
    lcat_off   = rd32(p + 44);
    lists_off  = rd32(p + 48);
    strtab_off = rd32(p + 52);
    strtab_len = rd32(p + 56);
    total      = rd32(p + 60);

    if (total != len || pack.n_luts <= 0 || pack.n_films <= 0 ||
        pack.n_fcats <= 0 || pack.n_lcats <= 0 || pack.n_lcats > 32)
        return CRAW_ERROR(EINVAL);
    /* The tables are laid out in this order and must not overlap or run
     * past the end; checking it here means nothing below has to. */
    if (dir_off != HDR_SIZE ||
        films_off != dir_off + (size_t)pack.n_luts * LUT_SIZE ||
        fcat_off != films_off + (size_t)pack.n_films * FILM_SIZE ||
        lcat_off != fcat_off + (size_t)pack.n_fcats * FCAT_SIZE ||
        lists_off != lcat_off + (size_t)pack.n_lcats * LCAT_SIZE ||
        strtab_off < lists_off || strtab_off > len ||
        strtab_len > len - strtab_off)
        return CRAW_ERROR(EINVAL);

    if (pack.compression == PACK_ZSTD) {
#if !CRAW_VIEW_LUTS_HAVE_ZSTD
        /* Built without zstd but handed a zstd pack: the two halves of the
         * build disagree, which is a build bug, not a runtime one. */
        return CRAW_ERROR(ENOTSUP);
#endif
    } else if (pack.compression != PACK_ZLIB) {
        return CRAW_ERROR(EINVAL);
    }

    pack.base = p;
    pack.len = len;
    pack.dir = p + dir_off;
    pack.films = p + films_off;
    pack.fcats = p + fcat_off;
    pack.lcats = p + lcat_off;
    pack.strtab = (const char *)p + strtab_off;
    pack.strtab_len = strtab_len;

    pack.luts = calloc(pack.n_luts, sizeof(*pack.luts));
    pack.film_list = calloc(pack.n_films, sizeof(*pack.film_list));
    pack.fcat_list = calloc(pack.n_fcats, sizeof(*pack.fcat_list));
    pack.lcat_list = calloc(pack.n_lcats, sizeof(*pack.lcat_list));
    if (!pack.luts || !pack.film_list || !pack.fcat_list || !pack.lcat_list) {
        view_luts_uninit();
        return CRAW_ERROR(ENOMEM);
    }

    for (int i = 0; i < pack.n_luts; i++) {
        const uint8_t *e = pack.dir + (size_t)i * LUT_SIZE;
        ViewLut *l = &pack.luts[i];
        uint32_t blob_off = rd32(e + 32), comp = rd32(e + 36);

        l->name  = pack_str(rd32(e + 0));
        l->slug  = pack_str(rd32(e + 4));
        l->label = pack_str(rd32(e + 8));
        l->film  = (int)rd32(e + 12);
        l->url   = pack_str(rd32(e + 16));
        l->title = pack_str(rd32(e + 20));
        l->file  = pack_str(rd32(e + 24));
        l->look_mask = rd32(e + 28);
        l->size  = rd16(e + 44);
        l->flags = rd16(e + 46);
        l->compressed_bytes = comp;
        for (int c = 0; c < 3; c++) {
            l->domain_min[c] = rdf32(e + 56 + 4 * c);
            l->domain_max[c] = rdf32(e + 68 + 4 * c);
        }
        if (l->film < 0 || l->film >= pack.n_films || l->size < 2 ||
            blob_off < strtab_off + strtab_len || blob_off > len ||
            comp > len - blob_off) {
            view_luts_uninit();
            return CRAW_ERROR(EINVAL);
        }
    }

    for (int i = 0; i < pack.n_films; i++) {
        const uint8_t *e = pack.films + (size_t)i * FILM_SIZE;
        ViewLutFilm *f = &pack.film_list[i];

        f->name  = pack_str(rd32(e + 0));
        f->slug  = pack_str(rd32(e + 4));
        f->url   = pack_str(rd32(e + 8));
        f->about = pack_str(rd32(e + 12));
        f->category  = (int)rd32(e + 16);
        f->first_lut = (int)rd32(e + 20);
        f->n_luts    = (int)rd32(e + 24);
        if (f->category < 0 || f->category >= pack.n_fcats ||
            f->first_lut < 0 || f->n_luts < 0 ||
            f->first_lut + f->n_luts > pack.n_luts) {
            view_luts_uninit();
            return CRAW_ERROR(EINVAL);
        }
    }

    for (int i = 0; i < pack.n_fcats; i++) {
        const uint8_t *e = pack.fcats + (size_t)i * FCAT_SIZE;
        ViewLutFilmCategory *c = &pack.fcat_list[i];

        c->name = pack_str(rd32(e + 0));
        c->slug = pack_str(rd32(e + 4));
        c->first_film = (int)rd32(e + 8);
        c->n_films    = (int)rd32(e + 12);
        if (c->first_film < 0 || c->n_films < 0 ||
            c->first_film + c->n_films > pack.n_films) {
            view_luts_uninit();
            return CRAW_ERROR(EINVAL);
        }
    }

    for (int i = 0; i < pack.n_lcats; i++) {
        const uint8_t *e = pack.lcats + (size_t)i * LCAT_SIZE;
        ViewLutLookCategory *c = &pack.lcat_list[i];

        c->name = pack_str(rd32(e + 0));
        c->slug = pack_str(rd32(e + 4));
        c->url  = pack_str(rd32(e + 8));
        c->n_films = (int)rd32(e + 12);
        c->films = pack_list(rd32(e + 16), rd32(e + 12), lists_off, strtab_off);
        c->n_luts = (int)rd32(e + 20);
        c->luts = pack_list(rd32(e + 24), rd32(e + 20), lists_off, strtab_off);
        if ((c->n_films && !c->films) || (c->n_luts && !c->luts)) {
            view_luts_uninit();
            return CRAW_ERROR(EINVAL);
        }
    }

    pack.ready = true;
    return 0;
}

void view_luts_uninit(void)
{
    free(pack.luts);
    free(pack.film_list);
    free(pack.fcat_list);
    free(pack.lcat_list);
    memset(&pack, 0, sizeof(pack));
}

int view_luts_count(void)
{
    return pack.ready ? pack.n_luts : 0;
}

const ViewLut *view_lut_get(int index)
{
    if (!pack.ready || index < 0 || index >= pack.n_luts)
        return NULL;
    return &pack.luts[index];
}

int view_luts_film_count(void)
{
    return pack.ready ? pack.n_films : 0;
}

const ViewLutFilm *view_lut_film(int index)
{
    if (!pack.ready || index < 0 || index >= pack.n_films)
        return NULL;
    return &pack.film_list[index];
}

int view_luts_film_category_count(void)
{
    return pack.ready ? pack.n_fcats : 0;
}

const ViewLutFilmCategory *view_lut_film_category(int index)
{
    if (!pack.ready || index < 0 || index >= pack.n_fcats)
        return NULL;
    return &pack.fcat_list[index];
}

int view_luts_look_category_count(void)
{
    return pack.ready ? pack.n_lcats : 0;
}

const ViewLutLookCategory *view_lut_look_category(int index)
{
    if (!pack.ready || index < 0 || index >= pack.n_lcats)
        return NULL;
    return &pack.lcat_list[index];
}

/* strcasecmp is POSIX and this tool is already POSIX-only, but the
 * comparison also has to ignore the difference between the hyphens in a
 * slug and the spaces in a display name so that --lut "clear spring" and
 * --lut clear-spring both land. */
static bool name_eq(const char *a, const char *b)
{
    while (*a && *b) {
        int ca = tolower((unsigned char)*a++);
        int cb = tolower((unsigned char)*b++);
        if (ca == '-' || ca == '_')
            ca = ' ';
        if (cb == '-' || cb == '_')
            cb = ' ';
        if (ca != cb)
            return false;
    }
    return !*a && !*b;
}

int view_luts_find(const char *name)
{
    if (!pack.ready || !name || !*name)
        return CRAW_ERROR(EINVAL);
    /* Slugs are unique by construction, display names are not, so slugs win
     * and a name match only picks the first rendering that carries it. */
    for (int i = 0; i < pack.n_luts; i++)
        if (name_eq(pack.luts[i].slug, name))
            return i;
    for (int i = 0; i < pack.n_luts; i++)
        if (name_eq(pack.luts[i].name, name))
            return i;
    return CRAW_ERROR(ENOENT);
}

static int pack_decompress(int index, uint8_t **out, size_t *out_len)
{
    const uint8_t *e = pack.dir + (size_t)index * LUT_SIZE;
    uint32_t blob_off = rd32(e + 32);
    uint32_t comp_len = rd32(e + 36);
    uint32_t raw_len  = rd32(e + 40);
    uint8_t *buf;

    if (raw_len != (uint32_t)pack.luts[index].size *
                   pack.luts[index].size * pack.luts[index].size * 3 * 2)
        return CRAW_ERROR(EINVAL);

    buf = malloc(raw_len);
    if (!buf)
        return CRAW_ERROR(ENOMEM);

    if (pack.compression == PACK_ZSTD) {
#if CRAW_VIEW_LUTS_HAVE_ZSTD
        size_t n = ZSTD_decompress(buf, raw_len, pack.base + blob_off,
                                   comp_len);
        if (ZSTD_isError(n) || n != raw_len) {
            free(buf);
            return CRAW_ERROR(EINVAL);
        }
#else
        free(buf);
        return CRAW_ERROR(ENOTSUP);
#endif
    } else {
        uLongf n = raw_len;
        if (uncompress(buf, &n, pack.base + blob_off, comp_len) != Z_OK ||
            n != raw_len) {
            free(buf);
            return CRAW_ERROR(EINVAL);
        }
    }

    *out = buf;
    *out_len = raw_len;
    return 0;
}

int view_luts_cube_text(int index, char **text, size_t *len)
{
    const ViewLut *l = view_lut_get(index);
    const uint8_t *e;
    uint8_t *raw = NULL;
    size_t raw_len = 0, cap, used = 0;
    char *buf;
    float vmin, vmax, scale;
    int n, ret;

    if (!l || !text)
        return CRAW_ERROR(EINVAL);

    ret = pack_decompress(index, &raw, &raw_len);
    if (ret < 0)
        return ret;

    e = pack.dir + (size_t)index * LUT_SIZE;
    vmin = rdf32(e + 48);
    vmax = rdf32(e + 52);
    scale = (vmax - vmin) / 65535.0f;
    n = l->size;

    /* A sample prints as at most "-0.123456789 " — twelve significant
     * characters plus a separator — so thirteen per value is the ceiling,
     * and the header never reaches the slack left on top. */
    cap = (size_t)n * n * n * 3 * 13 + 512;
    buf = malloc(cap);
    if (!buf) {
        free(raw);
        return CRAW_ERROR(ENOMEM);
    }

    used += snprintf(buf + used, cap - used,
                     "# %s", l->name);
    if (l->label[0])
        used += snprintf(buf + used, cap - used, " (%s)", l->label);
    used += snprintf(buf + used, cap - used,
                     "\n# %s\n# Source: %s\n",
                     view_lut_film(l->film)->name, l->url);
    /* Only emit a TITLE we can quote as-is; nothing downstream needs it. */
    if (l->title[0] && !strchr(l->title, '"') && !strchr(l->title, '\n'))
        used += snprintf(buf + used, cap - used, "TITLE \"%s\"\n", l->title);
    used += snprintf(buf + used, cap - used, "LUT_3D_SIZE %d\n", n);
    used += snprintf(buf + used, cap - used,
                     "DOMAIN_MIN %.9g %.9g %.9g\n",
                     l->domain_min[0], l->domain_min[1], l->domain_min[2]);
    used += snprintf(buf + used, cap - used,
                     "DOMAIN_MAX %.9g %.9g %.9g\n",
                     l->domain_max[0], l->domain_max[1], l->domain_max[2]);

    /* Undo the delta coding and the quantization in one pass. The encoder
     * restarts the accumulator at every red run, so a row is self
     * contained and the u16 wraparound never has to be reasoned about
     * across more than n steps. */
    {
        const uint8_t *q = raw;
        int rows = n * n;
        for (int row = 0; row < rows; row++) {
            uint16_t acc[3] = { 0, 0, 0 };
            for (int r = 0; r < n; r++) {
                float v[3];
                for (int c = 0; c < 3; c++) {
                    acc[c] = (uint16_t)(acc[c] + rd16(q));
                    q += 2;
                    v[c] = vmin + (float)acc[c] * scale;
                }
                used += snprintf(buf + used, cap - used, "%.9g %.9g %.9g\n",
                                 v[0], v[1], v[2]);
            }
        }
    }

    free(raw);
    if (used >= cap) {
        /* Cannot happen with the bound above; treat it as corruption
         * rather than handing out a truncated LUT. */
        free(buf);
        return CRAW_ERROR(EINVAL);
    }

    *text = buf;
    if (len)
        *len = used;
    return 0;
}

void view_luts_cube_free(char *text)
{
    free(text);
}
