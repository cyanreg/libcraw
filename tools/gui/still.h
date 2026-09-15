/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#ifndef CRAW_VIEW_STILL_H
#define CRAW_VIEW_STILL_H

#include <stddef.h>
#include <stdint.h>

/* Writes one rendered frame to `path` as a still image.
 *
 * The format comes from the file name: libavformat knows which muxer an
 * extension belongs to and which codec that muxer writes, so .png, .tif,
 * .jpg, .webp and .avif are all the same call. `codec_name` overrides the
 * encoder the name implies, and `enc_opts` is a KEY=VALUE,... list handed
 * to it (`threads=2`, `cpu-used=8`, `qmin=1`); both may be NULL.
 *
 * `rgba` is w * h interleaved RGBA samples, sixteen bits each, in host
 * order: exactly what the export target reads back as. The alpha is
 * dropped, since a still of an opaque frame has no use for it.
 *
 * `info`, when given, is filled with a short line naming the muxer, the
 * codec, the pixel format and the size of the file, for the panel and the
 * log to show. */
int view_still_write(const char *path, const char *codec_name,
                     const char *enc_opts, const uint16_t *rgba,
                     uint32_t w, uint32_t h, char *info, size_t info_size);

#endif /* CRAW_VIEW_STILL_H */
