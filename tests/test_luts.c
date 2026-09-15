/*
 * Copyright © 2026, Lynne
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* Checks the embedded film simulation pack against the two things it was
 * built from: the .cube files in tools/gui/luts/films/, which are the
 * ground truth for every sample, and manifest.json, which is the ground
 * truth for the counts and the category trees.
 *
 * The interesting check is the last one: every LUT is materialized back to
 * .cube text, parsed with a parser this file owns (deliberately not the one
 * in luts.c, so a bug in the writer cannot hide behind the same bug in the
 * reader), and compared sample by sample against the original file. The
 * pack's quantization is the only thing allowed to move a value, so the
 * tolerance is half its step. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "luts.h"

static int failures;

/* C23's monotonic clock rather than POSIX's clock_gettime, so that the test
 * compiles under the project's strict -std=c23 without feature test macros.
 * Falls back to the wall clock where the implementation has no monotonic
 * base; this only ever times a few milliseconds of work. */
static double now_ms(void)
{
    struct timespec t;

    if (!timespec_get(&t, TIME_MONOTONIC) && !timespec_get(&t, TIME_UTC))
        return 0.0;
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

#define CHECK(cond, ...)                                                \
    do {                                                                \
        if (!(cond)) {                                                  \
            printf("FAIL %s:%d: ", __func__, __LINE__);                 \
            printf(__VA_ARGS__);                                        \
            printf("\n");                                               \
            failures++;                                                 \
        }                                                               \
    } while (0)

/* The source tree's copy of the collection, passed in by meson. */
#ifndef TEST_LUTS_DIR
#define TEST_LUTS_DIR "."
#endif

/* ------------------------------------------------------------------ cube */

typedef struct TestCube {
    int size;
    float dmin[3], dmax[3];
    float *v;                   /* size^3 * 3, file order */
    int n;
} TestCube;

static void cube_free(TestCube *c)
{
    free(c->v);
    memset(c, 0, sizeof(*c));
}

/* Parses .cube text from memory. Returns 0, or -1 with a message. */
static int cube_parse(const char *text, size_t len, TestCube *out,
                      const char *what)
{
    const char *p = text, *end = text + len;
    int cap = 0;

    memset(out, 0, sizeof(*out));
    out->size = -1;
    for (int i = 0; i < 3; i++) {
        out->dmin[i] = 0.0f;
        out->dmax[i] = 1.0f;
    }

    while (p < end) {
        const char *eol = memchr(p, '\n', (size_t)(end - p));
        size_t n = eol ? (size_t)(eol - p) : (size_t)(end - p);
        char line[512];

        if (n >= sizeof(line))
            n = sizeof(line) - 1;
        memcpy(line, p, n);
        line[n] = '\0';
        p = eol ? eol + 1 : end;

        char *s = line;
        while (*s == ' ' || *s == '\t' || *s == '\r')
            s++;
        if (!*s || *s == '#')
            continue;

        if (!strncmp(s, "TITLE", 5)) {
            continue;
        } else if (!strncmp(s, "LUT_3D_SIZE", 11)) {
            out->size = atoi(s + 11);
            if (out->size < 2 || out->size > 256) {
                printf("FAIL %s: bad LUT_3D_SIZE %d\n", what, out->size);
                return -1;
            }
            out->n = out->size * out->size * out->size * 3;
            cap = out->n;
            out->v = malloc((size_t)cap * sizeof(*out->v));
            if (!out->v)
                return -1;
            out->n = 0;
        } else if (!strncmp(s, "DOMAIN_MIN", 10)) {
            sscanf(s + 10, "%f %f %f", &out->dmin[0], &out->dmin[1],
                   &out->dmin[2]);
        } else if (!strncmp(s, "DOMAIN_MAX", 10)) {
            sscanf(s + 10, "%f %f %f", &out->dmax[0], &out->dmax[1],
                   &out->dmax[2]);
        } else if (*s == '-' || *s == '.' || (*s >= '0' && *s <= '9')) {
            float a, b, c;
            if (sscanf(s, "%f %f %f", &a, &b, &c) != 3) {
                printf("FAIL %s: cannot parse sample %.60s\n", what, s);
                return -1;
            }
            if (!out->v || out->n + 3 > cap) {
                printf("FAIL %s: more samples than LUT_3D_SIZE allows\n", what);
                return -1;
            }
            out->v[out->n++] = a;
            out->v[out->n++] = b;
            out->v[out->n++] = c;
        } else {
            printf("FAIL %s: unknown line %.60s\n", what, s);
            return -1;
        }
    }
    if (out->size < 2 || out->n != cap) {
        printf("FAIL %s: %d of %d samples\n", what, out->n, cap);
        return -1;
    }
    return 0;
}

static int cube_parse_file(const char *path, TestCube *out)
{
    FILE *f = fopen(path, "rb");
    char *buf;
    long len;
    int ret;

    if (!f) {
        printf("FAIL cannot open %s\n", path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    len = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = malloc((size_t)len + 1);
    if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) {
        fclose(f);
        free(buf);
        printf("FAIL cannot read %s\n", path);
        return -1;
    }
    fclose(f);
    buf[len] = '\0';
    ret = cube_parse(buf, (size_t)len, out, path);
    free(buf);
    return ret;
}

/* --------------------------------------------------------------- manifest */

static char *slurp(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    char *buf;
    long n;

    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = malloc((size_t)n + 1);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fclose(f);
        free(buf);
        return NULL;
    }
    fclose(f);
    buf[n] = '\0';
    if (len)
        *len = (size_t)n;
    return buf;
}

/* Counts occurrences of needle between two markers in the manifest. The
 * manifest is machine written with one space of indent and a fixed key
 * order, so anchoring a marker to "\n \"key\"" picks out a top level
 * section without dragging a JSON parser into the test — the same key names
 * also occur nested, at deeper indents, which is exactly what the anchor
 * excludes. */
static int count_between(const char *hay, const char *from, const char *to,
                         const char *needle)
{
    const char *s = strstr(hay, from);
    const char *e = s ? strstr(s, to) : NULL;
    int n = 0;

    if (!s)
        return -1;
    if (!e)
        e = hay + strlen(hay);
    for (const char *p = strstr(s, needle); p && p < e;
         p = strstr(p + 1, needle))
        n++;
    return n;
}

static void test_counts_against_manifest(void)
{
    char path[1024];
    char *man;

    snprintf(path, sizeof(path), "%s/manifest.json", TEST_LUTS_DIR);
    man = slurp(path, NULL);
    if (!man) {
        printf("FAIL cannot read %s\n", path);
        failures++;
        return;
    }

    /* One "category" key per entry of films_tree, one "about" per film,
     * one "category" per entry of looks_tree, one "source_url" per
     * rendering. The sections appear in this order. */
    int fcats = count_between(man, "\n \"films_tree\"", "\n \"looks_tree\"",
                              "\"category\":");
    int films = count_between(man, "\n \"films_tree\"", "\n \"looks_tree\"",
                              "\"about\":");
    int lcats = count_between(man, "\n \"looks_tree\"", "\n \"renderings\"",
                              "\"category\":");
    int luts  = count_between(man, "\n \"renderings\"", "\n \"__end__\"",
                              "\"source_url\":");

    CHECK(fcats > 0 && films > 0 && lcats > 0 && luts > 0,
          "the manifest's sections were not found; has its layout changed?");

    CHECK(fcats == view_luts_film_category_count(),
          "film categories: pack %d, manifest %d",
          view_luts_film_category_count(), fcats);
    CHECK(films == view_luts_film_count(),
          "films: pack %d, manifest %d", view_luts_film_count(), films);
    CHECK(lcats == view_luts_look_category_count(),
          "look categories: pack %d, manifest %d",
          view_luts_look_category_count(), lcats);
    CHECK(luts == view_luts_count(),
          "renderings: pack %d, manifest %d", view_luts_count(), luts);

    /* Every name the pack hands out must appear in the manifest, which
     * catches a string table that has slipped out of alignment. */
    for (int i = 0; i < view_luts_film_category_count(); i++) {
        const ViewLutFilmCategory *c = view_lut_film_category(i);
        CHECK(strstr(man, c->slug) != NULL,
              "film category %d slug %s is not in the manifest", i, c->slug);
    }
    for (int i = 0; i < view_luts_look_category_count(); i++) {
        const ViewLutLookCategory *c = view_lut_look_category(i);
        CHECK(strstr(man, c->slug) != NULL,
              "look category %d slug %s is not in the manifest", i, c->slug);
    }

    printf("  manifest: %d film categories, %d films, %d look categories, "
           "%d renderings\n", fcats, films, lcats, luts);
    free(man);
}

/* ------------------------------------------------------------------ trees */

static void test_films_tree(void)
{
    int n = view_luts_count();
    char *seen = calloc((size_t)n, 1);
    int films_seen = 0, total = 0;

    CHECK(seen != NULL, "out of memory");
    if (!seen)
        return;

    for (int ci = 0; ci < view_luts_film_category_count(); ci++) {
        const ViewLutFilmCategory *c = view_lut_film_category(ci);
        int cat_luts = 0;

        CHECK(c->name[0] && c->slug[0], "category %d has no name", ci);
        CHECK(c->n_films > 0, "category %s is empty", c->name);
        for (int fi = c->first_film; fi < c->first_film + c->n_films; fi++) {
            const ViewLutFilm *f = view_lut_film(fi);

            CHECK(f != NULL, "film %d is out of range", fi);
            if (!f)
                continue;
            films_seen++;
            CHECK(f->category == ci, "film %s says category %d, listed in %d",
                  f->slug, f->category, ci);
            CHECK(f->n_luts > 0, "film %s has no renderings", f->slug);
            for (int li = f->first_lut; li < f->first_lut + f->n_luts; li++) {
                const ViewLut *l = view_lut_get(li);

                CHECK(l != NULL, "rendering %d is out of range", li);
                if (!l)
                    continue;
                CHECK(l->film == fi, "%s says film %d, listed under %d",
                      l->slug, l->film, fi);
                CHECK(!seen[li], "%s appears twice in the Films tree", l->slug);
                seen[li] = 1;
                cat_luts++;
                total++;
            }
        }
        printf("  %-28s %2d films, %3d renderings\n", c->name, c->n_films,
               cat_luts);
    }
    CHECK(films_seen == view_luts_film_count(),
          "Films tree covers %d of %d films", films_seen,
          view_luts_film_count());
    CHECK(total == n, "Films tree covers %d of %d renderings", total, n);
    for (int i = 0; i < n; i++)
        CHECK(seen[i], "rendering %d is in no film", i);
    free(seen);
}

static void test_looks_tree(void)
{
    int n_luts = view_luts_count();
    int n_cats = view_luts_look_category_count();

    for (int ci = 0; ci < n_cats; ci++) {
        const ViewLutLookCategory *c = view_lut_look_category(ci);
        int from_films = 0;

        CHECK(c->name[0] && c->slug[0], "look category %d has no name", ci);
        CHECK(c->n_luts > 0, "look category %s is empty", c->name);

        /* The category's rendering list and the look bit on each rendering
         * are two encodings of the same membership; they must agree. */
        for (int i = 0; i < c->n_luts; i++) {
            uint32_t ix = c->luts[i];
            const ViewLut *l = view_lut_get((int)ix);

            CHECK(l != NULL, "look %s lists rendering %u, out of range",
                  c->slug, ix);
            if (!l)
                continue;
            CHECK(l->look_mask & (1u << ci),
                  "%s is listed under %s but its mask says otherwise",
                  l->slug, c->slug);
        }
        for (int i = 0; i < n_luts; i++) {
            const ViewLut *l = view_lut_get(i);
            if (!(l->look_mask & (1u << ci)))
                continue;
            int found = 0;
            for (int j = 0; j < c->n_luts && !found; j++)
                found = (int)c->luts[j] == i;
            CHECK(found, "%s claims %s but is not in its list", l->slug,
                  c->slug);
        }

        /* And the film grouping must cover exactly the same renderings. */
        for (int i = 0; i < c->n_films; i++) {
            const ViewLutFilm *f = view_lut_film((int)c->films[i]);
            CHECK(f != NULL, "look %s lists film %u, out of range", c->slug,
                  c->films[i]);
            if (f)
                from_films += f->n_luts;
        }
        CHECK(from_films == c->n_luts,
              "look %s: %d films hold %d renderings, list has %d", c->slug,
              c->n_films, from_films, c->n_luts);

        printf("  %-38s %2d films, %3d renderings\n", c->name, c->n_films,
               c->n_luts);
    }
}

static void test_lookup(void)
{
    int n = view_luts_count();

    for (int i = 0; i < n; i++) {
        const ViewLut *l = view_lut_get(i);
        int got = view_luts_find(l->slug);

        CHECK(got == i, "find(\"%s\") returned %d, want %d", l->slug, got, i);
        /* Slugs are unique, so a display name must at least resolve to a
         * rendering that carries the same name. */
        got = view_luts_find(l->name);
        CHECK(got >= 0 && !strcmp(view_lut_get(got)->name, l->name),
              "find(\"%s\") returned %d", l->name, got);
    }
    CHECK(view_luts_find("no such film look at all") < 0,
          "a nonexistent name must not resolve");
    CHECK(view_luts_find("") < 0, "the empty name must not resolve");
    CHECK(view_luts_find(NULL) < 0, "a NULL name must not resolve");

    /* Hyphens and spaces are interchangeable so that --lut takes either. */
    {
        const ViewLut *l = view_lut_get(0);
        char alt[256];
        size_t j = 0;
        for (const char *p = l->slug; *p && j + 1 < sizeof(alt); p++)
            alt[j++] = *p == '-' ? ' ' : *p;
        alt[j] = '\0';
        CHECK(view_luts_find(alt) == 0, "find(\"%s\") must match the slug",
              alt);
    }
}

/* ------------------------------------------------------------- round trip */

static void test_round_trip(void)
{
    int n = view_luts_count();
    double worst = 0.0, total_ms = 0.0, one_ms = 0.0;
    const char *worst_name = "";
    size_t text_total = 0;
    int checked = 0;

    for (int i = 0; i < n; i++) {
        const ViewLut *l = view_lut_get(i);
        char path[1024];
        TestCube want = { 0 }, got = { 0 };
        double t0, t1;
        char *text = NULL;
        size_t len = 0;
        float lo, hi, step, tol;
        int ret;

        snprintf(path, sizeof(path), "%s/%s", TEST_LUTS_DIR, l->file);
        if (cube_parse_file(path, &want) < 0) {
            failures++;
            continue;
        }

        t0 = now_ms();
        ret = view_luts_cube_text(i, &text, &len);
        t1 = now_ms();
        if (ret < 0) {
            CHECK(0, "%s: view_luts_cube_text failed with %d", l->slug, ret);
            cube_free(&want);
            continue;
        }
        double ms = t1 - t0;
        total_ms += ms;
        if (i == 0)
            one_ms = ms;
        text_total += len;

        CHECK(len == strlen(text), "%s: length %zu, strlen %zu", l->slug, len,
              strlen(text));

        if (cube_parse(text, len, &got, l->slug) < 0) {
            failures++;
            view_luts_cube_free(text);
            cube_free(&want);
            continue;
        }
        view_luts_cube_free(text);

        CHECK(got.size == want.size, "%s: size %d, original %d", l->slug,
              got.size, want.size);
        CHECK(got.size == l->size, "%s: size %d, directory says %d", l->slug,
              got.size, l->size);
        for (int c = 0; c < 3; c++) {
            CHECK(got.dmin[c] == want.dmin[c] && got.dmax[c] == want.dmax[c],
                  "%s: domain %d changed", l->slug, c);
            CHECK(l->domain_min[c] == want.dmin[c] &&
                  l->domain_max[c] == want.dmax[c],
                  "%s: directory domain %d disagrees with the file", l->slug,
                  c);
        }
        if (got.size != want.size) {
            cube_free(&got);
            cube_free(&want);
            continue;
        }

        /* The encoder quantizes over the LUT's own value range, so the step
         * is that range over 65535 and a correctly rounded sample lands
         * within half of it. The extra 1e-6 is for the float rounding in
         * the dequantization and in printing the text. */
        lo = hi = want.v[0];
        for (int k = 1; k < want.n; k++) {
            lo = want.v[k] < lo ? want.v[k] : lo;
            hi = want.v[k] > hi ? want.v[k] : hi;
        }
        step = (hi - lo) / 65535.0f;
        tol = step * 0.5f + 1e-6f;

        for (int k = 0; k < want.n; k++) {
            double d = fabs((double)got.v[k] - (double)want.v[k]);
            if (d > worst) {
                worst = d;
                worst_name = l->name;
            }
            if (d > tol) {
                CHECK(0, "%s: sample %d is %.9g, original %.9g, off by %.3g "
                      "(tol %.3g)", l->slug, k, got.v[k], want.v[k], d, tol);
                break;      /* one complaint per LUT is plenty */
            }
        }

        /* The flags in the directory have to describe the real data. */
        {
            bool out_of_range = lo < 0.0f || hi > 1.0f;
            bool non_unit = false;
            for (int c = 0; c < 3; c++)
                non_unit |= want.dmin[c] != 0.0f || want.dmax[c] != 1.0f;
            CHECK(!!(l->flags & VIEW_LUT_OUT_OF_RANGE) == out_of_range,
                  "%s: VIEW_LUT_OUT_OF_RANGE is wrong", l->slug);
            CHECK(!!(l->flags & VIEW_LUT_NON_UNIT_DOMAIN) == non_unit,
                  "%s: VIEW_LUT_NON_UNIT_DOMAIN is wrong", l->slug);
        }

        checked++;
        cube_free(&got);
        cube_free(&want);
    }

    printf("  materialized %d/%d LUTs, %.1f MB of .cube text\n", checked, n,
           text_total / 1e6);
    printf("  worst sample error %.3g (%s), quantization step %.3g for a "
           "full 0..1 range\n", worst, worst_name, 1.0 / 65535.0);
    printf("  decompress + write one LUT: %.3f ms (first), "
           "%.3f ms average over %d\n", one_ms, total_ms / (n ? n : 1), n);

    /* What selecting a LUT actually costs once the pages are warm, which is
     * every selection after the first. */
    if (n > 0) {
        const int reps = 50;
        double t0 = now_ms();
        for (int r = 0; r < reps; r++) {
            char *text = NULL;
            size_t len = 0;
            if (view_luts_cube_text(0, &text, &len) == 0)
                view_luts_cube_free(text);
        }
        printf("  warm: %.3f ms per selection (%s, %d samples, %u "
               "compressed bytes)\n", (now_ms() - t0) / reps,
               view_lut_get(0)->name,
               view_lut_get(0)->size * view_lut_get(0)->size *
               view_lut_get(0)->size, view_lut_get(0)->compressed_bytes);
    }
}

static void test_bounds(void)
{
    CHECK(view_lut_get(-1) == NULL, "index -1 must be NULL");
    CHECK(view_lut_get(view_luts_count()) == NULL, "one past the end is NULL");
    CHECK(view_lut_film(-1) == NULL, "film -1 must be NULL");
    CHECK(view_lut_film(view_luts_film_count()) == NULL, "film end is NULL");
    CHECK(view_lut_film_category(view_luts_film_category_count()) == NULL,
          "film category end is NULL");
    CHECK(view_lut_look_category(view_luts_look_category_count()) == NULL,
          "look category end is NULL");
    {
        char *text = NULL;
        size_t len = 0;
        CHECK(view_luts_cube_text(-1, &text, &len) < 0,
              "materializing index -1 must fail");
        CHECK(view_luts_cube_text(view_luts_count(), &text, &len) < 0,
              "materializing one past the end must fail");
    }
}

int main(void)
{
    int ret = view_luts_init();

    if (ret < 0) {
        printf("test_luts: view_luts_init failed with %d\n", ret);
        return 1;
    }
    /* init() is documented as idempotent; the GUI calls it from more than
     * one place. */
    CHECK(view_luts_init() == 0, "a second init must succeed");

    printf("test_luts: %d renderings, %d films, %d film categories, "
           "%d look categories\n", view_luts_count(), view_luts_film_count(),
           view_luts_film_category_count(), view_luts_look_category_count());

    test_counts_against_manifest();
    printf("Films:\n");
    test_films_tree();
    printf("Looks:\n");
    test_looks_tree();
    test_lookup();
    test_bounds();
    test_round_trip();

    view_luts_uninit();

    if (failures) {
        printf("test_luts: %d failure(s)\n", failures);
        return 1;
    }
    printf("test_luts: all checks passed\n");
    return 0;
}
