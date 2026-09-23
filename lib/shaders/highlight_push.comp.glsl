/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Highlight reconstruction, one level of the push, DESIGN.md 18.2: every
 * texel of the finer level mixes its own colour with the coarser level's,
 * bilinearly upsampled, by how much weight it gathered itself, and is
 * rewritten in place as a colour,
 *
 *     .r = ln(R/G)    .g = ln(B/G)    .b = its own weight, kept    .a = 0
 *
 * Each invocation reads and writes only its own texel of the finer level,
 * so rewriting it in place needs no second image. The coarser level has
 * already been pushed, except at the very top, which the pull left as sums
 * and which is normalized here as it is read. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. */
#extension GL_GOOGLE_include_directive : require

#define CRAW_HL_LEVELS
#include "common.glsl"
#include "highlight.glsl"

layout(binding = 0) readonly uniform image2D coarse;
layout(binding = 1) uniform image2D fine;

/* The coarser level's colour at a texel, clamped to its edge. */
vec2 craw_hl_coarse(ivec2 q)
{
    vec4 v = imageLoad(coarse, clamp(q, ivec2(0), pc.coarse - 1));

    if (pc.top != 0)
        return v.z > 0.0 ? v.xy / v.z : vec2(0.0);
    return v.xy;
}

void main()
{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, pc.fine)))
        return;

    /* The texel's centre in the coarser level's texel coordinates. */
    vec2 pos = (vec2(p) + 0.5) * 0.5 - 0.5;
    vec2 i0f = floor(pos);
    vec2 f = pos - i0f;
    ivec2 i0 = ivec2(i0f);

    vec2 up = mix(mix(craw_hl_coarse(i0),               craw_hl_coarse(i0 + ivec2(1, 0)), f.x),
                  mix(craw_hl_coarse(i0 + ivec2(0, 1)), craw_hl_coarse(i0 + ivec2(1, 1)), f.x),
                  f.y);

    vec4 own = imageLoad(fine, p);
    vec2 c = up;
    if (own.z > 0.0) {
        float t = min(own.z / CRAW_HL_W_SAT, 1.0);
        c = up + t * (own.xy / own.z - up);
    }

    imageStore(fine, p, vec4(c, own.z, 0.0));
}
