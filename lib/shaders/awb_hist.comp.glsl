/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Auto white balance histogram, DESIGN.md 4.4: one invocation per 8x8 block
 * of the camera RGB intermediate, the block mean reduced to a raw
 * chromaticity bin. Three uint storage buffers are accumulated with atomics:
 * a sample count, a sum of min(g, 0.5) weights and a sum of g^4 weights,
 * the last one letting light sources and glare outvote a dominant surface;
 * all in 16.16 fixed point. The pipeline zeroes them before the frame and
 * copies them to a host visible buffer after it.
 *
 * Only whole blocks are sampled: a frame whose width or height is not a
 * multiple of 8 ignores the remainder, which is what the CPU builder in
 * craw_awb_histogram_from_rgb() does with the same 8x8 grid.
 *
 * A block is a quad of four invocations, each reading two columns of it
 * row by row, so that the loads of a subgroup are runs of a row rather
 * than 64 loads per invocation scattered over a block each, which took
 * 0.39 ms at 5.8K in the full chain where this takes 0.31 (DESIGN.md 21).
 * The quad's first
 * invocation gathers the others' pixels with quad broadcasts (DPP) and
 * sums the 64 row by row, left to right, the order they were summed in
 * before; it alone does the atomics. A subgroup is a row of eight blocks,
 * and a workgroup as many rows as it has subgroups: the host builds it in
 * full subgroups of 32 and invocations are placed by their subgroup and
 * lane. The atomics are three per block, 0.8 million a frame, and were
 * never what bound the pass, so a reduction across the subgroup before
 * them would buy nothing. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. The subgroup
 * extensions are the basic and quad operations lib/craw.c checks for. */
#extension GL_GOOGLE_include_directive : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_quad : require

#include "common.glsl"

layout(binding = 0) readonly uniform image2D src;

layout(binding = 1, scalar) buffer CRAWAwbCountBuf {
    uint craw_awb_count[];
};

layout(binding = 2, scalar) buffer CRAWAwbWeightBuf {
    uint craw_awb_weight[];
};

layout(binding = 3, scalar) buffer CRAWAwbBrightBuf {
    uint craw_awb_weight_bright[];
};

/* Mirrors CRAWAwbPush in lib/pipeline.c. */
layout(push_constant, scalar) uniform CRAWAwbPushBuf {
    ivec2 blocks;      /* whole 8x8 blocks, x and y */
    vec3  wb;          /* as-shot gains, divided back out */
    int   bins;        /* per axis */
    float log_min, log_max;
} pc;

/* DESIGN.md 4.4 skip rules. */
const float CRAW_AWB_CLIP  = 0.98;
const float CRAW_AWB_BLACK = 0.002;
const float CRAW_AWB_MAX_WEIGHT = 0.5;

void main()
{
    int lane = int(gl_SubgroupInvocationID);
    int k = lane & 3;
    ivec2 b = ivec2(int(gl_WorkGroupID.x) * (int(gl_SubgroupSize) / 4) + lane / 4,
                    int(gl_WorkGroupID.y) * int(gl_NumSubgroups) + int(gl_SubgroupID));
    bool live = all(lessThan(b, pc.blocks));

    /* The mean of the block, i.e. a 4x4 arrangement of complete CFA quads,
     * so the mean is not biased towards any channel by the debayer. This
     * invocation's two columns first; every invocation then takes part in
     * the broadcasts. */
    ivec2 o = b * 8 + ivec2(2 * k, 0);
    vec3 px[8][2];
    for (int j = 0; j < 8; j++)
        for (int i = 0; i < 2; i++)
            px[j][i] = live ? imageLoad(src, o + ivec2(i, j)).rgb : vec3(0.0);

    vec3 sum = vec3(0.0);
    for (int j = 0; j < 8; j++)
        for (int i = 0; i < 8; i++)
            sum += subgroupQuadBroadcast(px[j][i & 1], i >> 1);
    vec3 m = sum * (1.0 / 64.0);

    if (!live || k != 0)
        return;

    if (max(max(m.r, m.g), m.b) >= CRAW_AWB_CLIP || m.g < CRAW_AWB_BLACK)
        return;

    /* Raw chromaticity: the as-shot white balance divided back out, so the
     * histogram lives in the same plane as the profile's locus. */
    float rr = (m.r / pc.wb.r) / m.g;
    float bb = (m.b / pc.wb.b) / m.g;

    vec2 lg = clamp(vec2(log(rr), log(bb)), vec2(pc.log_min), vec2(pc.log_max));
    vec2 f = (lg - pc.log_min) * (float(pc.bins) / (pc.log_max - pc.log_min));
    ivec2 idx = clamp(ivec2(floor(f)), ivec2(0), ivec2(pc.bins - 1));

    uint bin = uint(idx.y * pc.bins + idx.x);
    atomicAdd(craw_awb_count[bin], 1u);
    atomicAdd(craw_awb_weight[bin], uint(round(min(m.g, CRAW_AWB_MAX_WEIGHT) * 65536.0)));
    atomicAdd(craw_awb_weight_bright[bin], uint(round(m.g * m.g * m.g * m.g * 65536.0)));
}
