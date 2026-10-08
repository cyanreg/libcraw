/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#ifndef CRAW_LENS_WARP_H
#define CRAW_LENS_WARP_H

/* The warp of DESIGN.md 16 as lib/pipeline.c and the CPU reference in
 * lib/ref/ref_lens.c both set it up, so that the two cannot drift apart.
 * lib/shaders/lens.comp.glsl carries the same modes. */

#include <craw/lens.h>

/* What a frame with no record is warped with when it is turned: no
 * correction at all, about the frame's centre. lib/lens.c. */
extern const CRAWLensCorrection craw_lens_identity;

/* Mirrors the shader's `mode`. FILL is the record's own frame, written
 * exactly as it was before anything else existed (16.8); TURN fills a
 * turned frame (16.9); KEEP keeps every source pixel, turned or not, and
 * paints black where there is none. */
#define CRAW_LENS_WARP_FILL 0
#define CRAW_LENS_WARP_TURN 1
#define CRAW_LENS_WARP_KEEP 2

/* The output offset to corrected offset map of 16.9,
 *
 *     q = (turn[0] d.x + turn[1] d.y,  turn[0] d.y - turn[1] d.x)
 *
 * which is (cos, sin) of the clockwise turn over the scale. Worked out in
 * double and narrowed once. At 0 degrees it is (1 / scale, 0), so keeping
 * everything unturned divides by the scale exactly as it always did. */
void craw_lens_turn(float degrees, float scale, float turn[2]);

#endif /* CRAW_LENS_WARP_H */
