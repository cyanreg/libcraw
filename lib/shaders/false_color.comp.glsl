/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* False colour suppression, DESIGN.md 17: on the two colour differences
 * R - G and B - G of the camera RGB intermediate, one to five passes of a
 * 3x3 median followed by a 3x3 box blur, with G left exactly as the
 * debayer made it. RawTherapee's "false colour suppression steps", on the
 * differences to G rather than in YIQ and with mirrored borders.
 *
 * A glint smaller than a CFA quad is sampled by one site and comes out of
 * every debayer as a coloured dot; in the colour differences it is an
 * isolated outlier, which the median removes, and a glint two or three
 * sensels wide is a small blob, which the blur spreads and desaturates.
 * All the passes run in one dispatch: the tile carries an apron of two
 * pixels per pass, one for each sweep, sweep s is computed on the part of
 * the tile that lies s inside its edge, and the result at the workgroup's
 * own pixels is what the sequential definition gives. The apron is loaded
 * through the mirror of DESIGN.md 3, and because that mirror is a
 * reflection and both stencils are symmetric under it, a sweep computed at
 * a mirrored position is the sweep at the position it mirrors, so the
 * fused result matches the sequential one at the borders too. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. */
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(binding = 0) readonly  uniform image2D craw_src;   /* camera RGB, RGBA16F */
layout(binding = 1) writeonly uniform image2D craw_dst;   /* the same, filtered */

/* Mirrors CRAWFalseColorPush in lib/pipeline.c. */
layout(push_constant, scalar) uniform CRAWFalseColorPushBuf {
    ivec2 size;
    int   passes;      /* 1 to CRAW_FC_MAX_PASSES; the host never sends 0 */
} pc;

#define CRAW_FC_MAX_PASSES 5

/* The tile's side: the workgroup's plus an apron of 2 * CRAW_FC_MAX_PASSES
 * on each side, chosen by the host to fit the device's shared memory
 * (lib/pipeline.c, CRAW_FC_WG_MAX). Two copies of both differences in one
 * vec2: a median cannot run in place, and the componentwise min and max
 * of the network below sort the two planes at once. */
layout (constant_id = 4) const int craw_fc_tile = 36;

shared vec2 craw_fc_mem[2 * craw_fc_tile * craw_fc_tile];

/* Median of nine by a 20 exchange selection network (Smith's, the one in
 * McGuire's median shader): each step discards the minimum and the maximum
 * of a shrinking set, neither of which can be the median, until three are
 * left. A selection, not an average: the result is one of the inputs, so
 * the reference gets the same bits from a sort. */
#define CRAW_S2(a, b) { vec2 t = min(a, b); b = max(a, b); a = t; }
#define CRAW_MN3(a, b, c) CRAW_S2(a, b) CRAW_S2(a, c)
#define CRAW_MX3(a, b, c) CRAW_S2(b, c) CRAW_S2(a, c)
#define CRAW_MNMX3(a, b, c)          CRAW_MX3(a, b, c) CRAW_S2(a, b)
#define CRAW_MNMX4(a, b, c, d)       CRAW_S2(a, b) CRAW_S2(c, d) CRAW_S2(a, c) CRAW_S2(b, d)
#define CRAW_MNMX5(a, b, c, d, e)    CRAW_S2(a, b) CRAW_S2(c, d) CRAW_MN3(a, c, e) CRAW_MX3(b, d, e)
#define CRAW_MNMX6(a, b, c, d, e, f) CRAW_S2(a, d) CRAW_S2(b, e) CRAW_S2(c, f) \
                                     CRAW_MN3(a, b, c) CRAW_MX3(d, e, f)

vec2 craw_fc_median9(vec2 v0, vec2 v1, vec2 v2, vec2 v3, vec2 v4,
                     vec2 v5, vec2 v6, vec2 v7, vec2 v8)
{
    CRAW_MNMX6(v0, v1, v2, v3, v4, v5);
    CRAW_MNMX5(v1, v2, v3, v4, v6);
    CRAW_MNMX4(v2, v3, v4, v7);
    CRAW_MNMX3(v3, v4, v8);
    return v4;
}

/* The box blur: the nine summed row by row, left to right, then scaled by
 * the float nearest 1/9, in that order, which is the order the reference
 * sums in. */
vec2 craw_fc_mean9(vec2 v0, vec2 v1, vec2 v2, vec2 v3, vec2 v4,
                   vec2 v5, vec2 v6, vec2 v7, vec2 v8)
{
    return ((((((((v0 + v1) + v2) + v3) + v4) + v5) + v6) + v7) + v8) * (1.0 / 9.0);
}

void main()
{
    int rad = 2 * clamp(pc.passes, 1, CRAW_FC_MAX_PASSES);
    ivec2 wgs = ivec2(gl_WorkGroupSize.xy);
    ivec2 lid = ivec2(gl_LocalInvocationID.xy);
    ivec2 org = ivec2(gl_WorkGroupID.xy) * wgs - rad;
    ivec2 tsz = wgs + 2 * rad;
    int tw = tsz.x;
    int cur = 0, nxt = craw_fc_tile * craw_fc_tile;

    for (int ty = lid.y; ty < tsz.y; ty += wgs.y) {
        for (int tx = lid.x; tx < tsz.x; tx += wgs.x) {
            ivec2 q = craw_mirror(org + ivec2(tx, ty), pc.size);
            vec3 c = imageLoad(craw_src, q).rgb;
            craw_fc_mem[cur + ty * tw + tx] = vec2(c.r - c.g, c.b - c.g);
        }
    }

    barrier();

    /* Sweep s, the median of pass (s + 1) / 2 when s is odd and the blur
     * of pass s / 2 when it is even, is wanted on [s, tsz - s) in each
     * direction: that is where its stencil stays inside what sweep s - 1
     * produced, and it is all that sweep s + 1, or the store below, will
     * read. The bound is a push constant, so the loop and the barriers in
     * it are uniform. */
    for (int s = 1; s <= rad; s++) {
        bool med = (s & 1) == 1;

        for (int ty = lid.y + s; ty < tsz.y - s; ty += wgs.y) {
            for (int tx = lid.x + s; tx < tsz.x - s; tx += wgs.x) {
                int i = cur + ty * tw + tx;
                vec2 v0 = craw_fc_mem[i - tw - 1], v1 = craw_fc_mem[i - tw], v2 = craw_fc_mem[i - tw + 1];
                vec2 v3 = craw_fc_mem[i - 1],      v4 = craw_fc_mem[i],      v5 = craw_fc_mem[i + 1];
                vec2 v6 = craw_fc_mem[i + tw - 1], v7 = craw_fc_mem[i + tw], v8 = craw_fc_mem[i + tw + 1];

                craw_fc_mem[nxt + ty * tw + tx] = med ? craw_fc_median9(v0, v1, v2, v3, v4, v5, v6, v7, v8)
                                                      : craw_fc_mean9(v0, v1, v2, v3, v4, v5, v6, v7, v8);
            }
        }

        barrier();

        int t = cur;
        cur = nxt;
        nxt = t;
    }

    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, pc.size)))
        return;

    vec4 c = imageLoad(craw_src, p);
    vec2 d = craw_fc_mem[cur + (lid.y + rad) * tw + lid.x + rad];

    imageStore(craw_dst, p, vec4(c.g + d.x, c.g, c.g + d.y, c.a));
}
