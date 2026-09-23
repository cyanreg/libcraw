/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Highlight reconstruction, one level of the pull, DESIGN.md 18.2: every
 * texel of the coarser level is the sum of the (up to) four texels of the
 * finer one under it, component by component. Sums and not means, weights
 * included: the push decides how much a texel trusts itself by how much
 * weight it has gathered, and the brightness weighting of level 0 has to
 * survive all the way up. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. */
#extension GL_GOOGLE_include_directive : require

#define CRAW_HL_LEVELS
#include "common.glsl"
#include "highlight.glsl"

layout(binding = 0) readonly uniform image2D fine;
layout(binding = 1) writeonly uniform image2D coarse;

void main()
{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, pc.coarse)))
        return;

    vec3 acc = vec3(0.0);
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            ivec2 f = 2 * p + ivec2(dx, dy);
            if (all(lessThan(f, pc.fine)))
                acc += imageLoad(fine, f).xyz;
        }
    }

    imageStore(coarse, p, vec4(acc, 0.0));
}
