/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Highlight reconstruction, the fill, DESIGN.md 18.2: every clipped site of
 * the mosaic rebuilt from the channels that did not clip around it and the
 * colour field, every other site passed through, all of it balanced and
 * unclipped into an R16F mosaic that the debayer then reads with an
 * identity prep.
 *
 * One invocation per quad. The 3x3 neighbourhoods of a quad's four sites
 * together cover the 4x4 block from one site up and left of the quad to two
 * down and right, so that block is loaded once and every site's stencil is
 * read out of it. Most quads have no clipped site at all and store their
 * four balanced samples without touching the field. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. */
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"
#include "highlight.glsl"

layout(binding = 0) readonly uniform image2D mosaic;
layout(binding = 1) readonly uniform image2D field;
layout(binding = 2) writeonly uniform image2D dst;

/* Level 0's colour at a texel, clamped to its edge. */
vec2 craw_hl_field(ivec2 t)
{
    return imageLoad(field, clamp(t, ivec2(0), pc.fsize - 1)).xy;
}

void main()
{
    ivec2 q = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(q, pc.size >> 1)))
        return;

    /* The 4x4 block, [j * 4 + i] at o + (i, j), borders mirrored. */
    ivec2 o = 2 * q - 1;
    float xv[16], av[16];
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            ivec2 p = craw_mirror(o + ivec2(i, j), pc.size);
            int s = craw_site(p);
            float u = craw_hl_u(imageLoad(mosaic, p).x, s);

            xv[j * 4 + i] = u * pc.wb[craw_site_channel(s)];
            av[j * 4 + i] = craw_hl_ramp(u);
        }
    }

    /* The quad itself is the middle 2x2. */
    if (av[5] == 0.0 && av[6] == 0.0 && av[9] == 0.0 && av[10] == 0.0) {
        for (int s = 0; s < 4; s++)
            imageStore(dst, 2 * q + ivec2(s & 1, s >> 1),
                       vec4(xv[(1 + (s >> 1)) * 4 + 1 + (s & 1)]));
        return;
    }

    /* The field at the quad's centre, (2 q + 1) / 4 - 0.5 in level 0's
     * texel coordinates. */
    vec2 pos = vec2(q) * 0.5 - 0.25;
    vec2 i0f = floor(pos);
    vec2 f = pos - i0f;
    ivec2 i0 = ivec2(i0f);
    vec2 lf = mix(mix(craw_hl_field(i0),               craw_hl_field(i0 + ivec2(1, 0)), f.x),
                  mix(craw_hl_field(i0 + ivec2(0, 1)), craw_hl_field(i0 + ivec2(1, 1)), f.x),
                  f.y);
    vec3 lrho = vec3(lf.x, 0.0, lf.y);

    for (int s = 0; s < 4; s++) {
        ivec2 d = ivec2(s & 1, s >> 1);
        ivec2 p = 2 * q + d;
        int ci = (1 + d.y) * 4 + 1 + d.x;
        int c = craw_site_channel(craw_site(p));
        float x = xv[ci], a = av[ci];
        float y = x;

        if (a > 0.0) {
            /* The other two channels over the 3x3, in stencil order. The
             * mirror preserves the CFA phase, so a neighbour's channel is
             * that of its unmirrored position. */
            vec3 sum = vec3(0.0), n = vec3(0.0), amax = vec3(0.0);
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int k = craw_site_channel(craw_site(p + ivec2(dx, dy)));
                    int ni = ci + dy * 4 + dx;

                    if (k == c)
                        continue;
                    sum[k] += xv[ni];
                    n[k] += 1.0;
                    amax[k] = max(amax[k], av[ni]);
                }
            }

            /* The other two, numbered in R, G, B order. */
            int k1 = c == 0 ? 1 : 0;
            int k2 = c == 2 ? 1 : 2;
            float a1 = 1.0 - amax[k1], a2 = 1.0 - amax[k2];
            float l1 = log(max(sum[k1] / n[k1], CRAW_HL_EPS)) - lrho[k1];
            float l2 = log(max(sum[k2] / n[k2], CRAW_HL_EPS)) - lrho[k2];
            float lc = log(max(x, CRAW_HL_EPS)) - lrho[c];

            float A = a1 + a2;
            float lfull = exp(max(lc, max(l1, l2)));
            float lpart = A > 0.0 ? exp((a1 * l1 + a2 * l2) / A) : lfull;
            float L = lfull + min(A, 1.0) * (lpart - lfull);

            y = x + a * max(L * exp(lrho[c]) - x, 0.0);
        }

        imageStore(dst, p, vec4(y));
    }
}
