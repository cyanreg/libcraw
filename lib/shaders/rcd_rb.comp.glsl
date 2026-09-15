/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* RCD pass C, DESIGN.md 5.3: red and blue from colour differences against
 * the green plane of pass B, and the two other channels of the finished
 * intermediate.
 *
 * 5.3 makes the G sites depend on the R/B sites of this same pass, so the
 * pass runs twice with specialization constant 1 selecting the phase:
 *
 *   phase 0: R and B sites. The missing colour comes from the diagonal
 *            neighbours' colour differences, blended with wp. The full RGB
 *            of those sites is written to the intermediate.
 *   phase 1: G sites. Both missing colours come from the four edge
 *            neighbours' colour differences, which phase 0 wrote: the
 *            neighbours on one axis hold their native colour (D in 5.3) and
 *            the ones on the other axis the diagonal estimate (E), and the
 *            two are blended with wv. Phase 1 reads only pixels phase 0
 *            wrote and writes only pixels phase 0 did not, so the barrier
 *            between the two dispatches is the whole dependency.
 *
 * See rcd_dir.comp.glsl for the auxiliary image layout. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. */
#extension GL_GOOGLE_include_directive : require

#define CRAW_PREP
#include "common.glsl"

/* 0: R and B sites, 1: G sites. */
layout (constant_id = 1) const uint craw_rcd_phase = 0u;

layout(binding = 1) readonly uniform image2D aux;     /* (L, wv, wp, -) */
layout(binding = 2) readonly uniform image2D green;   /* pass B */
layout(binding = 3)          uniform image2D dst;     /* intermediate, rw */

/* Colour difference at an R or B neighbour: its native sample minus its
 * green. Only phase 0 uses this, where the neighbour is a native site. */
float craw_diff(ivec2 p)
{
    ivec2 q = craw_mirror(p, pc.size);
    return craw_v(q) - imageLoad(green, q).x;
}

void main()
{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, pc.size)))
        return;

    int s = craw_site(p);
    int c = craw_site_channel(s);
    bool is_green = c == 1;

    /* Each phase owns half the sites. */
    if (is_green != (craw_rcd_phase == 1u))
        return;

    float g0 = imageLoad(green, p).x;
    vec3 rgb;
    rgb[1] = g0;

    if (!is_green) {
        rgb[c] = craw_v(p);
        float cp = g0 + 0.5 * (craw_diff(p + ivec2(-1,  1)) + craw_diff(p + ivec2( 1, -1)));
        float cq = g0 + 0.5 * (craw_diff(p + ivec2(-1, -1)) + craw_diff(p + ivec2( 1,  1)));
        float wp = imageLoad(aux, p).z;
        rgb[2 - c] = wp * cp + (1.0 - wp) * cq;
    } else {
        vec4 nw = imageLoad(dst, craw_mirror(p + ivec2(-1, 0), pc.size));
        vec4 ne = imageLoad(dst, craw_mirror(p + ivec2( 1, 0), pc.size));
        vec4 nn = imageLoad(dst, craw_mirror(p + ivec2(0, -1), pc.size));
        vec4 ns = imageLoad(dst, craw_mirror(p + ivec2(0,  1), pc.size));
        float wv = imageLoad(aux, p).y;

        /* R and B are symmetric here: on one axis the neighbours are native
         * sites of that colour (D) and on the other they carry the estimate
         * phase 0 made (E), and wv weights the vertical estimate either
         * way. */
        for (int ch = 0; ch < 3; ch += 2) {
            float chh = g0 + 0.5 * ((nw[ch] - nw.g) + (ne[ch] - ne.g));
            float cvv = g0 + 0.5 * ((nn[ch] - nn.g) + (ns[ch] - ns.g));
            rgb[ch] = wv * cvv + (1.0 - wv) * chh;
        }
    }

    imageStore(dst, p, vec4(rgb, 1.0));
}
