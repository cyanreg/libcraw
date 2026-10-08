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
#ifndef CRAW_LENS_H
#define CRAW_LENS_H

#include <craw/utils.h>

/* Geometric lens correction: a per-channel radial polynomial, in the
 * destination-to-source convention. For an output pixel at normalized
 * radius `r` from the optical centre,
 *
 *     r_src = r * (k0 + k1 r^2 + k2 r^4 + k3 r^6)
 *
 * is the radius, on the same ray, the source is sampled at. `r` is
 * normalized so that r = 1 is the frame corner, i.e. by the half diagonal
 * measured from the centre this struct states. The three channels differ
 * by a pixel or two at the corner, which is the lens's lateral chromatic
 * aberration; k0 below 1.0 is the crop a pincushion correction needs to
 * keep its corners on the sensor. See DESIGN.md 16 for the whole
 * definition, including where the centre sits in pixels.
 *
 * Panasonic's cameras write exactly this, per frame, into every ProRes RAW
 * frame; craw_lens_from_prores_raw() below reads it out of one. The struct
 * is not tied to that source: a caller with a profile from anywhere else
 * fills it in itself. */
typedef struct CRAWLensCorrection {
    /* k[channel][term], channels in camera order R, G, B. The record's
     * doubles are narrowed to float: at r <= 1 with these magnitudes that
     * is a thousandth of a pixel on a 5.8K frame, and both the shader and
     * the CPU reference evaluate in fp32 anyway. */
    float k[3][4];

    /* Optical centre in normalized frame coordinates, (0.5, 0.5) for the
     * geometric centre. Panasonic reports exactly that on every frame seen;
     * the field exists because the record carries it. */
    float centre[2];

    /* False means "no correction for this frame": craw_pipeline_record()
     * then leaves the frame untouched, unless it is turned
     * (CRAWFrameParams.rotation). */
    bool valid;

    /* Padding to allow for future options. Must always be set to 0. */
    uint8_t reserved[64 - 3*4*4 - 2*4 - 1];
} CRAWLensCorrection;

/* Reads the per-frame optical correction record out of one ProRes RAW
 * frame, as handed over by a demuxer: `packet` points at the frame's be32
 * size field and `size` is the whole packet.
 *
 * The frame is walked the way libavcodec/prores_raw.c walks it — header,
 * tile size table, tile data — so that the vendor metadata list that
 * follows the last tile is found by arithmetic rather than by searching
 * for the key string, and every read is bounds checked against `size`.
 * The record is the `psim` entry whose key is
 * `com.panasonic.Semi-Pro.optical_correction` in the list `eomd`
 * terminates.
 *
 * Returns 0 and fills `out` (with valid = true) on success,
 * CRAW_ERROR(ENOENT) for a frame that carries no such record — including
 * one whose trailing bytes are not a record list at all, which is what an
 * FX6 recorded to an Atomos Ninja V writes — and CRAW_ERROR(EINVAL) for a
 * packet that is not a walkable ProRes RAW frame, or one whose
 * optical_correction record is there but the wrong shape. `out` is zeroed,
 * i.e. left invalid, on every failure, so the return value may be ignored
 * by a caller that only wants "correct it if you can".
 *
 * Nothing here touches the bitstream: no tile is decoded and no pixel is
 * read. It costs a few microseconds on a 5 MB frame. */
CRAW_API int craw_lens_from_prores_raw(const uint8_t *packet, size_t size,
                                       CRAWLensCorrection *out);

/* The correction's corner displacement in pixels for a frame of
 * `width` x `height`, per channel, into `px[3]`: how far the source is
 * from the destination at the frame corner, positive outwards. This is the
 * net figure, the crop k0 included, so it is a few pixels on a frame whose
 * k0 was chosen to absorb the correction.
 *
 * With `relative` set the centre scale is divided out first
 * (poly(1) / poly(0) - 1), which is the distortion the record actually
 * corrects: -207 px of barrel at 24 mm, +100 px of pincushion at 60 mm.
 * That is the number to print for a person.
 *
 * Returns CRAW_ERROR(EINVAL) on a NULL or invalid correction. */
CRAW_API int craw_lens_corner_shift(const CRAWLensCorrection *lc,
                                    uint32_t width, uint32_t height,
                                    bool relative, float px[3]);

/* The uniform scale the "keep everything" mode of DESIGN.md 16.8 needs for
 * this record on a frame of `width` x `height`: the largest s <= 1 for
 * which every source pixel, put through the correction, still lands inside
 * the output frame. One number for all three channels, so the chromatic
 * correction stays consistent; the shader divides the output position by it
 * before evaluating the polynomial, which shrinks the corrected picture
 * until the whole warped source fits and leaves black around it.
 *
 * It is computed by walking the source rectangle's boundary, inverting
 * r * poly(r) per channel at each point to find the output radius that
 * source pixel needs, and taking the smallest ratio of the output frame's
 * boundary radius to it. A few hundred microseconds of arithmetic, no
 * allocation and no Vulkan; the answer changes with the record, so it is a
 * per-frame number on a clip that zooms.
 *
 * Only a record whose polynomial cannot be inverted at all — one no lens
 * has — fails, with CRAW_ERROR(EINVAL), which is also what a NULL or
 * invalid correction and a zero frame size give.
 *
 * The pipeline works this out for itself when the frame carries
 * CRAW_FRAME_FLAG_LENS_KEEP_ALL; this is exported so that a tool can show
 * the number, which is the only way a person can see how much of the frame
 * the mode costs. It is craw_lens_rotation_scale() at 0 degrees with
 * `keep_all` set, to the bit. */
CRAW_API int craw_lens_keep_all_scale(const CRAWLensCorrection *lc,
                                      uint32_t width, uint32_t height,
                                      float *scale);

/* Turning the picture, DESIGN.md 16.9: the most CRAWFrameParams.rotation
 * may be either way, exclusive. Up to it the frame keeps its size and its
 * aspect ratio, and the picture is magnified until it covers the turned
 * frame. */
#define CRAW_LENS_MAX_ROTATION 45.0f

/* The scale the warp applies to the corrected picture on a frame of
 * `width` x `height` turned by `degrees` clockwise, about the record's
 * optical centre: the corrected picture is magnified by it, so above 1 is
 * a zoom in and below 1 a shrink. `lc` may be NULL or invalid, which is
 * the picture with no correction, turned about the frame's centre.
 *
 * Filling the frame, `keep_all` false, it is the smallest scale, never
 * below 1, at which every output pixel still has a source in all three
 * channels. The source is what the correction makes of the readout, not
 * the readout's rectangle: a barrel correction pulls the source's corners
 * out past the frame and a pincushion one, under the crop of its k0,
 * leaves a bulge of source beyond each edge, and filling throws both away.
 * A turned frame's corners swing out into exactly those places, so the
 * scale is measured against the corrected source's own outline, and a
 * small turn at the wide end of a zoom costs no magnification at all. At
 * 0 degrees it is exactly 1 without anything being walked: the frame is
 * the record's own, which is the camera's choice.
 *
 * Keeping everything, `keep_all` true, it is craw_lens_keep_all_scale()'s
 * scale for the turned frame: the largest one, at most 1, for which every
 * source pixel still lands inside the frame once turned.
 *
 * Both are boundary walks of the kind craw_lens_keep_all_scale() does,
 * under a millisecond, no allocation and no Vulkan. Fails with
 * CRAW_ERROR(EINVAL) for a NULL `scale`, a zero frame size, a turn of
 * CRAW_LENS_MAX_ROTATION or more either way, a valid record whose centre is
 * outside the frame, or one whose polynomial cannot be inverted. */
CRAW_API int craw_lens_rotation_scale(const CRAWLensCorrection *lc,
                                      uint32_t width, uint32_t height,
                                      float degrees, bool keep_all,
                                      float *scale);

#endif /* CRAW_LENS_H */
