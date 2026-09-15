/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#ifndef CRAW_VIEW_LUTS_H
#define CRAW_VIEW_LUTS_H

/* The embedded film simulation collection.
 *
 * craw-view carries every rendering from filmgrid.org as one compressed
 * blob per LUT inside the executable (see tools/luts_build.py for the pack
 * and tools/gui/luts/LICENSE.md for where the data comes from). The user
 * picks one out of a cascading menu, so the two things this module has to
 * be good at are walking the category trees cheaply and materializing
 * exactly one LUT on demand: nothing is decompressed until a selection
 * happens, and the 295 that were not picked cost nothing but their bytes.
 *
 * Deliberately free of SDL, libplacebo and nuklear so that tests can link
 * it on its own. The only dependency past libc is the decompressor.
 *
 * The site groups the same renderings two ways and the menu offers both:
 *
 *   Films -> film category -> film -> rendering
 *   Looks -> look category -> rendering   (grouped by film if you want)
 *
 * Renderings are numbered in the Films tree's order, so a film's renderings
 * are always one contiguous run. */

#include <stddef.h>
#include <stdint.h>

enum ViewLutFlags {
    /* The .cube carried a DOMAIN_MIN/MAX other than 0..1. */
    VIEW_LUT_NON_UNIT_DOMAIN = 1 << 0,
    /* At least one output value falls outside 0..1, so clamping the LUT
     * output would change the look. */
    VIEW_LUT_OUT_OF_RANGE    = 1 << 1,
};

typedef struct ViewLut {
    const char *name;           /* "Clear Spring", as the site names it */
    const char *slug;           /* "clear-spring", unique, what --lut matches */
    const char *label;          /* "-1 stop", how it differs from the base */
    const char *title;          /* TITLE from the .cube, "" when absent */
    const char *url;            /* where the .cube came from */
    const char *file;           /* its path under tools/gui/luts/ in the tree */
    int film;                   /* index into the film array */
    uint32_t look_mask;         /* bit N set for look category N */
    int size;                   /* LUT_3D_SIZE */
    unsigned flags;             /* enum ViewLutFlags */
    float domain_min[3];
    float domain_max[3];
    uint32_t compressed_bytes;  /* what it costs in the executable */
} ViewLut;

typedef struct ViewLutFilm {
    const char *name;           /* "Kodak Portra 400" */
    const char *slug;           /* "kodak-portra-400" */
    const char *url;
    const char *about;          /* "colour negative · Kodak · ISO 400" */
    int category;               /* index into the film categories */
    int first_lut;              /* its renderings are [first_lut, +n_luts) */
    int n_luts;
} ViewLutFilm;

typedef struct ViewLutFilmCategory {
    const char *name;           /* "Colour Negative Film" */
    const char *slug;
    int first_film;             /* its films are [first_film, +n_films) */
    int n_films;
} ViewLutFilmCategory;

typedef struct ViewLutLookCategory {
    const char *name;           /* "Warm Film Looks" */
    const char *slug;
    const char *url;
    /* Look categories cut across films, so unlike the Films tree these are
     * explicit index lists rather than contiguous runs. Both are in the
     * site's order; films[] is the same set grouped one level up. */
    const uint32_t *films;
    int n_films;
    const uint32_t *luts;
    int n_luts;
} ViewLutLookCategory;

/* Reads the pack's header and tables. Cheap: it only points into the
 * embedded array, it does not copy or decompress anything. Returns 0, or a
 * negative errno when the pack is missing or malformed. Idempotent. */
int view_luts_init(void);
void view_luts_uninit(void);

/* Tree walking. Every getter returns NULL for an out of range index, so a
 * menu can walk with a plain loop and stop on NULL. */
int view_luts_count(void);
const ViewLut *view_lut_get(int index);

int view_luts_film_count(void);
const ViewLutFilm *view_lut_film(int index);

int view_luts_film_category_count(void);
const ViewLutFilmCategory *view_lut_film_category(int index);

int view_luts_look_category_count(void);
const ViewLutLookCategory *view_lut_look_category(int index);

/* Looks a rendering up by its slug first and its display name second, both
 * case insensitively; returns the index or a negative errno when nothing
 * matches. This is what `--lut NAME` goes through. */
int view_luts_find(const char *name);

/* Decompresses one LUT and writes it out as .cube text, which is what
 * libplacebo's pl_lut_parse_cube() eats. *text is NUL terminated and *len
 * is the length without the terminator; free it with view_luts_cube_free().
 * Returns 0 or a negative errno. This is the only call that touches the
 * compressed data, and it is only ever made when the user picks a LUT. */
int view_luts_cube_text(int index, char **text, size_t *len);
void view_luts_cube_free(char *text);

#endif /* CRAW_VIEW_LUTS_H */
