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
 * A quad is usable when neither it nor any of its eight neighbouring quads
 * is clipped: the ring of quads touching a clipped area reads measurably off
 * colour (DESIGN.md 18.2), so the block's two quads are judged over the 4x4
 * quads around them, which is the 8x8 sites from two up and left of the
 * block to five down and right. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. */
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"
#include "highlight.glsl"

layout(binding = 0) readonly uniform image2D mosaic;
layout(binding = 1) writeonly uniform image2D field;

void main()
{
    ivec2 b = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(b, pc.fsize)))
        return;

    /* The largest a of each of the 4x4 quads around the block, [j * 4 + i]
     * for the quad 2 b - 1 + (i, j). A quad outside the frame does not
     * exist and clips nothing. */
    float qa[16];
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            ivec2 q = 2 * (2 * b - 1 + ivec2(i, j));
            float amax = 0.0;

            if (all(greaterThanEqual(q, ivec2(0))) && all(lessThan(q, pc.size))) {
                for (int s = 0; s < 4; s++)
                    amax = max(amax, craw_hl_ramp(craw_hl_u(imageLoad(mosaic, q + ivec2(s & 1, s >> 1)).x, s)));
            }
            qa[j * 4 + i] = amax;
        }
    }

    /* The block's own quads, top row first, left to right, the order of
     * the definition. The frame is even in both directions, so a quad is
     * either whole or entirely outside it. */
    vec3 acc = vec3(0.0);
    for (int qy = 0; qy < 2; qy++) {
        for (int qx = 0; qx < 2; qx++) {
            ivec2 q = 4 * b + 2 * ivec2(qx, qy);
            if (any(greaterThanEqual(q, pc.size)))
                continue;

            float amax = 0.0;
            for (int j = 0; j < 3; j++)
                for (int i = 0; i < 3; i++)
                    amax = max(amax, qa[(qy + j) * 4 + qx + i]);

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
