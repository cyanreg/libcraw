/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Chroma denoise, DESIGN.md 14.4, pass 3 of 3: the band the quarter grid
 * holds is replaced in every full resolution pixel's colour differences,
 * blended by the strength, and everything finer is kept as it was:
 *
 *     d' = d + strength * (up(filtered) - up(down(d)))
 *
 * with `up` the bilinear interpolation of a quarter image at the block
 * centre aligned position, coordinates clamped to the grid. G and alpha
 * pass through untouched. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. */
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(binding = 0) readonly  uniform image2D craw_src;    /* camera RGB, full resolution */
layout(binding = 1) readonly  uniform image2D craw_down;   /* quarter, unfiltered */
layout(binding = 2) readonly  uniform image2D craw_blur;   /* quarter, filtered */
layout(binding = 3) writeonly uniform image2D craw_dst;    /* camera RGB, full resolution */

/* Mirrors CRAWChromaPush in lib/pipeline.c. */
layout(push_constant, scalar) uniform CRAWChromaPushBuf {
    ivec2 size;
    ivec2 qsize;
    float strength;
} pc;

void main()
{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, pc.size)))
        return;

    vec4 c = imageLoad(craw_src, p);
    vec2 d = vec2(c.r - c.g, c.b - c.g);

    /* Quarter texel q holds the mean of full pixels [4q, 4q + 4), centred
     * at 4q + 1.5, so p sits at (p + 0.5) / 4 - 0.5 in texel centre units.
     * The four corners and their weights, top left first; the sums below
     * run in that order, as the reference does. */
    vec2 f = (vec2(p) + 0.5) * 0.25 - 0.5;
    vec2 fl = floor(f);
    vec2 t = f - fl;
    ivec2 i0 = clamp(ivec2(fl), ivec2(0), pc.qsize - 1);
    ivec2 i1 = clamp(ivec2(fl) + 1, ivec2(0), pc.qsize - 1);
    float w00 = (1.0 - t.x) * (1.0 - t.y), w10 = t.x * (1.0 - t.y);
    float w01 = (1.0 - t.x) * t.y,         w11 = t.x * t.y;

    vec2 lo = w00 * imageLoad(craw_down, ivec2(i0.x, i0.y)).xy
            + w10 * imageLoad(craw_down, ivec2(i1.x, i0.y)).xy
            + w01 * imageLoad(craw_down, ivec2(i0.x, i1.y)).xy
            + w11 * imageLoad(craw_down, ivec2(i1.x, i1.y)).xy;
    vec2 lf = w00 * imageLoad(craw_blur, ivec2(i0.x, i0.y)).xy
            + w10 * imageLoad(craw_blur, ivec2(i1.x, i0.y)).xy
            + w01 * imageLoad(craw_blur, ivec2(i0.x, i1.y)).xy
            + w11 * imageLoad(craw_blur, ivec2(i1.x, i1.y)).xy;

    d += pc.strength * (lf - lo);

    imageStore(craw_dst, p, vec4(c.g + d.x, c.g, c.g + d.y, c.a));
}
