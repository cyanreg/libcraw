/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Chroma denoise, DESIGN.md 14.4, pass 2 of 3: a bilateral over the
 * quarter resolution colour differences, Gaussian in space over a 9x9
 * window, Gaussian in the difference of the quarter resolution G and
 * Gaussian in the difference of the colour differences themselves, the
 * widths of both range terms relative to the centre's own G so that they
 * mean the same thing in a shadow and in a highlight. The G term keeps
 * chroma from crossing a luminance edge and the chroma term from crossing
 * a colour edge at equal luminance; both are block means rather than
 * pixel values, so a glint smaller than a block does not fence its colour
 * off, which is the whole reason this stage runs after section 17's. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. */
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(binding = 0) readonly  uniform image2D craw_src;   /* (d_r, d_b, G, 0), quarter resolution */
layout(binding = 1) writeonly uniform image2D craw_dst;   /* the same with the differences filtered */

/* Mirrors CRAWChromaPush in lib/pipeline.c. */
layout(push_constant, scalar) uniform CRAWChromaPushBuf {
    ivec2 size;
    ivec2 qsize;
    float strength;    /* not read by this pass */
} pc;

/* DESIGN.md 14.4; lib/ref/ref_chroma.c holds the same five numbers. */
const int   CRAW_CD_RADIUS    = 4;
const float CRAW_CD_INV_2S2   = 1.0 / (2.0 * 2.0 * 2.0);   /* sigma_s = 2 quarter pixels */
const float CRAW_CD_RANGE_K   = 0.35;
const float CRAW_CD_CHROMA_K  = 0.35;
const float CRAW_CD_RANGE_EPS = 0.02;

void main()
{
    ivec2 q = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(q, pc.qsize)))
        return;

    vec4 c = imageLoad(craw_src, q);
    float level = max(c.z, 0.0) + CRAW_CD_RANGE_EPS;
    float inv_range = 1.0 / (CRAW_CD_RANGE_K * level);
    float inv_chroma = 1.0 / (CRAW_CD_CHROMA_K * level);
    vec2 acc = vec2(0.0);
    float wsum = 0.0;

    /* Row by row, left to right, the reference's order. Borders mirrored in
     * the quarter grid's own coordinates. */
    for (int dy = -CRAW_CD_RADIUS; dy <= CRAW_CD_RADIUS; dy++) {
        for (int dx = -CRAW_CD_RADIUS; dx <= CRAW_CD_RADIUS; dx++) {
            vec4 s = imageLoad(craw_src, craw_mirror(q + ivec2(dx, dy), pc.qsize));
            float dg = (s.z - c.z) * inv_range;
            vec2 dd = (s.xy - c.xy) * inv_chroma;
            float w = exp(-float(dx * dx + dy * dy) * CRAW_CD_INV_2S2 - dg * dg
                          - dd.x * dd.x - dd.y * dd.y);

            acc += w * s.xy;
            wsum += w;
        }
    }

    imageStore(craw_dst, q, vec4(acc / wsum, c.z, 0.0));
}
