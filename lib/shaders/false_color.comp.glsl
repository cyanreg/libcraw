/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* False colour suppression, DESIGN.md 17: a 3x3 median of the two colour
 * differences R - G and B - G of the camera RGB intermediate, run one to
 * four times, with G left exactly as the debayer made it.
 *
 * A glint smaller than a CFA quad is sampled by one site and comes out of
 * every debayer as a coloured dot; in the colour differences it is an
 * isolated outlier, which is what a median removes. All the passes run in
 * one dispatch: the tile carries an apron of `passes` pixels, pass k is
 * computed on the part of the tile that pass k - 1 covered, and the result
 * at the workgroup's own pixels is what the sequential definition gives.
 * The apron is loaded through the mirror of DESIGN.md 3, and because that
 * mirror is a reflection and the stencil is symmetric under it, a pass
 * computed at a mirrored position is the pass at the position it mirrors,
 * so the fused result matches the sequential one at the borders too. */

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

/* The largest tile the shared arrays hold: a 16x16 workgroup with an apron
 * of four on every side. The host keeps to both; see CRAW_FC_WG_MAX in
 * lib/pipeline.c. */
#define CRAW_FC_MAX_PASSES 4
#define CRAW_FC_WG_MAX     16
#define CRAW_FC_TILE_MAX   (CRAW_FC_WG_MAX + 2 * CRAW_FC_MAX_PASSES)

/* Both differences in one vec2, so that the componentwise min and max of
 * the network below sort the two planes at once. Two copies, because a
 * median cannot run in place. */
shared vec2 craw_fc_tile[2][CRAW_FC_TILE_MAX * CRAW_FC_TILE_MAX];

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

void main()
{
    int rad = clamp(pc.passes, 1, CRAW_FC_MAX_PASSES);
    ivec2 wgs = ivec2(gl_WorkGroupSize.xy);
    ivec2 lid = ivec2(gl_LocalInvocationID.xy);
    ivec2 org = ivec2(gl_WorkGroupID.xy) * wgs - rad;
    ivec2 tsz = wgs + 2 * rad;
    int tw = tsz.x;
    int cur = 0;

    for (int ty = lid.y; ty < tsz.y; ty += wgs.y) {
        for (int tx = lid.x; tx < tsz.x; tx += wgs.x) {
            ivec2 q = craw_mirror(org + ivec2(tx, ty), pc.size);
            vec3 c = imageLoad(craw_src, q).rgb;
            craw_fc_tile[0][ty * tw + tx] = vec2(c.r - c.g, c.b - c.g);
        }
    }

    barrier();

    /* Pass k is wanted on [k, tsz - k) in each direction: that is where its
     * stencil stays inside what pass k - 1 produced, and it is all that
     * pass k + 1, or the store below, will read. The bound is a push
     * constant, so the loop and the barriers in it are uniform. */
    for (int k = 1; k <= rad; k++) {
        int nxt = cur ^ 1;

        for (int ty = lid.y + k; ty < tsz.y - k; ty += wgs.y) {
            for (int tx = lid.x + k; tx < tsz.x - k; tx += wgs.x) {
                int i = ty * tw + tx;

                craw_fc_tile[nxt][i] = craw_fc_median9(
                    craw_fc_tile[cur][i - tw - 1], craw_fc_tile[cur][i - tw], craw_fc_tile[cur][i - tw + 1],
                    craw_fc_tile[cur][i - 1],      craw_fc_tile[cur][i],      craw_fc_tile[cur][i + 1],
                    craw_fc_tile[cur][i + tw - 1], craw_fc_tile[cur][i + tw], craw_fc_tile[cur][i + tw + 1]);
            }
        }

        barrier();
        cur = nxt;
    }

    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, pc.size)))
        return;

    vec4 c = imageLoad(craw_src, p);
    vec2 d = craw_fc_tile[cur][(lid.y + rad) * tw + lid.x + rad];

    imageStore(craw_dst, p, vec4(c.g + d.x, c.g, c.g + d.y, c.a));
}
