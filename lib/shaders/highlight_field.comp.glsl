/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Highlight reconstruction, the colour field's level 0, DESIGN.md 18.2: one
 * texel per 4x4 block of the mosaic, holding the brightness-weighted sums
 * of the log colour ratios of the block's usable quads and the sum of their
 * weights,
 *
 *     .r = sum w ln(R/G)    .g = sum w ln(B/G)    .b = sum w    .a = 0
 *
 * in RGBA32F: the pull passes add these up the pyramid, and the sums over a
 * whole frame need more than fp16's range and precision.
 *
 * A quad is usable when no quad within CRAW_HL_DILATE of it is clipped: the
 * quads near a clipped area read off colour and, around a small light, off
 * colour differently in every frame (DESIGN.md 18.2). Each invocation needs
 * the clip state of the 7x7 quads around each of its two by two, so the
 * workgroup loads the state of all the quads it covers, plus that margin,
 * into shared memory once, and every invocation reads its windows from
 * there. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. */
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"
#include "highlight.glsl"

layout(binding = 0) readonly uniform image2D mosaic;
layout(binding = 1) writeonly uniform image2D field;

/* The largest a of every quad the workgroup covers and of the margin
 * around them. Sized for the largest workgroup the host hands out, 16x16
 * blocks, i.e. 32x32 quads; a smaller one uses the top left of it. */
#define CRAW_HL_TILE_MAX (2 * 16 + 2 * CRAW_HL_DILATE)
shared float craw_hl_qa[CRAW_HL_TILE_MAX * CRAW_HL_TILE_MAX];

void main()
{
    int wg = int(gl_WorkGroupSize.x);
    int tile = 2 * wg + 2 * CRAW_HL_DILATE;
    ivec2 b = ivec2(gl_GlobalInvocationID.xy);
    /* The quad at the tile's top left corner. */
    ivec2 q0 = 2 * ivec2(gl_WorkGroupID.xy) * wg - CRAW_HL_DILATE;

    for (int i = int(gl_LocalInvocationIndex); i < tile * tile; i += wg * wg) {
        /* Quad coordinates, doubled into the site of its top left corner. A
         * quad outside the frame does not exist and clips nothing. */
        ivec2 q = 2 * (q0 + ivec2(i % tile, i / tile));
        float amax = 0.0;

        if (all(greaterThanEqual(q, ivec2(0))) && all(lessThan(q, pc.size))) {
            for (int s = 0; s < 4; s++)
                amax = max(amax, craw_hl_ramp(craw_hl_u(imageLoad(mosaic, q + ivec2(s & 1, s >> 1)).x, s)));
        }
        craw_hl_qa[(i / tile) * CRAW_HL_TILE_MAX + i % tile] = amax;
    }

    barrier();

    if (any(greaterThanEqual(b, pc.fsize)))
        return;

    /* The block's own quads, top row first, left to right, the order of
     * the definition. The frame is even in both directions, so a quad is
     * either whole or entirely outside it. */
    ivec2 lb = 2 * ivec2(gl_LocalInvocationID.xy) + CRAW_HL_DILATE;
    vec3 acc = vec3(0.0);
    for (int qy = 0; qy < 2; qy++) {
        for (int qx = 0; qx < 2; qx++) {
            ivec2 q = 4 * b + 2 * ivec2(qx, qy);
            if (any(greaterThanEqual(q, pc.size)))
                continue;

            float amax = 0.0;
            for (int j = -CRAW_HL_DILATE; j <= CRAW_HL_DILATE; j++)
                for (int i = -CRAW_HL_DILATE; i <= CRAW_HL_DILATE; i++)
                    amax = max(amax, craw_hl_qa[(lb.y + qy + j) * CRAW_HL_TILE_MAX + lb.x + qx + i]);

            vec3 rgb = vec3(0.0);
            float umax = 0.0;
            for (int s = 0; s < 4; s++) {
                /* q is even, so the site of q + (s & 1, s >> 1) is s. */
                float u = craw_hl_u(imageLoad(mosaic, q + ivec2(s & 1, s >> 1)).x, s);
                int c = craw_site_channel(s);

                rgb[c] += u * pc.wb[c];
                umax = max(umax, u);
            }
            rgb.g *= 0.5;

            float m = min(umax, 1.0);
            float w = (1.0 - amax) * m * m;
            float g = max(rgb.g, CRAW_HL_EPS);
            float r = clamp(log(max(rgb.r, CRAW_HL_EPS) / g), -CRAW_HL_LOG_MAX, CRAW_HL_LOG_MAX);
            float bl = clamp(log(max(rgb.b, CRAW_HL_EPS) / g), -CRAW_HL_LOG_MAX, CRAW_HL_LOG_MAX);

            acc += vec3(w * r, w * bl, w);
        }
    }

    imageStore(field, b, vec4(acc, 0.0));
}
