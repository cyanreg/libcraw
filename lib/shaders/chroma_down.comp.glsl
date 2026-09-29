/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Chroma denoise, DESIGN.md 14.4, pass 1 of 3: the camera RGB intermediate
 * reduced to quarter resolution as the mean of each 4x4 block's colour
 * differences R - G and B - G, with the block's mean G beside them for the
 * bilateral's range term. The quarter grid is ceil(size / 4) on each axis,
 * and a block at the right or bottom edge averages the pixels it has. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. The subgroup
 * extensions are the basic and quad operations lib/craw.c checks for. */
#extension GL_GOOGLE_include_directive : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_quad : require

#include "common.glsl"

layout(binding = 0) readonly  uniform image2D craw_src;   /* camera RGB, full resolution */
layout(binding = 1) writeonly uniform image2D craw_dst;   /* (d_r, d_b, G, 0), quarter resolution */

/* Mirrors CRAWChromaPush in lib/pipeline.c. */
layout(push_constant, scalar) uniform CRAWChromaPushBuf {
    ivec2 size;        /* full resolution */
    ivec2 qsize;       /* quarter resolution, ceil(size / 4) */
    float strength;    /* not read by this pass */
} pc;

/* A workgroup is a band of four block rows, 16 pixel rows, and as many
 * pixel columns as it has invocations over four: an invocation reads one
 * column of one block, its four pixels, so that each load of a subgroup is
 * one run of a pixel row, where an invocation per block reading its
 * sixteen pixels touched sixteen scattered texels per load: 0.65 ms at
 * 5.8K then, 0.49 now, at every width and wave size tried. The four
 * columns of a block are a quad (four consecutive
 * invocations of a subgroup), whose first invocation gathers the other
 * three's values with quad broadcasts (DPP on RDNA, no memory) and sums
 * all sixteen row by row, left to right: the order the reference sums
 * in, as before. Invocations are placed by their subgroup and their lane
 * in it, so that the quads are what the placement says. */
void main()
{
    const int NT = int(gl_WorkGroupSize.x * gl_WorkGroupSize.y);
    const int W  = NT / 4;           /* pixel columns in a workgroup */
    int l = int(gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID);
    ivec2 o = ivec2(int(gl_WorkGroupID.x) * W + l % W, int(gl_WorkGroupID.y) * 16 + l / W * 4);
    vec3 v[4];
    bool ok[4];

    for (int j = 0; j < 4; j++) {
        ivec2 p = o + ivec2(0, j);
        ok[j] = all(lessThan(p, pc.size));
        vec3 c = ok[j] ? imageLoad(craw_src, p).rgb : vec3(0.0);
        v[j] = vec3(c.r - c.g, c.b - c.g, c.g);
    }

    vec3 sum = vec3(0.0);
    float n = 0.0;

    /* Row by row, left to right: the order the reference sums in. Every
     * invocation takes part in the broadcasts; the first of each quad
     * keeps the result. */
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            vec3 c = subgroupQuadBroadcast(v[j], i);
            bool k = subgroupQuadBroadcast(ok[j], i);
            if (k) {
                sum += c;
                n += 1.0;
            }
        }
    }

    ivec2 q = o / 4;
    if ((gl_SubgroupInvocationID & 3u) == 0u && all(lessThan(q, pc.qsize)))
        imageStore(craw_dst, q, vec4(sum / n, 0.0));
}
