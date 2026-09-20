/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Chroma denoise, DESIGN.md 14.4, pass 1 of 3: the camera RGB intermediate
 * reduced to quarter resolution as the mean of each 4x4 block's colour
 * differences R - G and B - G, with the block's mean G beside them for the
 * bilateral's range term. The quarter grid is ceil(size / 4) on each axis,
 * and a block at the right or bottom edge averages the pixels it has. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. */
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(binding = 0) readonly  uniform image2D craw_src;   /* camera RGB, full resolution */
layout(binding = 1) writeonly uniform image2D craw_dst;   /* (d_r, d_b, G, 0), quarter resolution */

/* Mirrors CRAWChromaPush in lib/pipeline.c. */
layout(push_constant, scalar) uniform CRAWChromaPushBuf {
    ivec2 size;        /* full resolution */
    ivec2 qsize;       /* quarter resolution, ceil(size / 4) */
    float strength;    /* not read by this pass */
} pc;

void main()
{
    ivec2 q = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(q, pc.qsize)))
        return;

    ivec2 o = q * 4;
    vec3 sum = vec3(0.0);
    float n = 0.0;

    /* Row by row, left to right: the order the reference sums in. */
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            ivec2 p = o + ivec2(i, j);
            if (any(greaterThanEqual(p, pc.size)))
                continue;
            vec3 c = imageLoad(craw_src, p).rgb;
            sum += vec3(c.r - c.g, c.b - c.g, c.g);
            n += 1.0;
        }
    }

    imageStore(craw_dst, q, vec4(sum / n, 0.0));
}
