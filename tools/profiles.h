/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#ifndef CRAW_TOOL_PROFILES_H
#define CRAW_TOOL_PROFILES_H

#include <craw/profile.h>

/* Per-CCT white balance and matrix tables for the CLI tools.
 *
 * libcraw itself never parses a container, and FFmpeg's ProRes RAW decoder
 * exposes only the per-frame colour record, not the per-CCT tables that sit
 * in the movie header. Until those are plumbed through the decoder the tool
 * carries its own copy for the cameras it has been characterized against,
 * plus a text loader so a new camera needs no rebuild. */

/* Copies the built-in tables of `name` (e.g. "s1m2") into `p`, leaving
 * every other field alone, and sets p->name. Returns CRAW_ERROR(ENOENT) if
 * no such table set exists. */
int craw_tool_profile_apply_builtin(CRAWProfile *p, const char *name);

/* Comma separated list of the built-in names, for the usage text. */
const char *craw_tool_profile_builtin_list(void);

/* Returns the built-in name whose camera model appears anywhere in `text`
 * (a container metadata value), or NULL. */
const char *craw_tool_profile_detect(const char *text);

/* Loads tables from a text file into `p`, replacing whatever tables it
 * held. Lines are `wb CCT R B`, `matrix CCT m00 .. m22`, blank, or a `#`
 * comment; the two kinds may be interleaved and are sorted by CCT on load.
 * Returns a negative error and prints the offending line to stderr. */
int craw_tool_profile_load_tables(CRAWProfile *p, const char *path);

#endif /* CRAW_TOOL_PROFILES_H */
