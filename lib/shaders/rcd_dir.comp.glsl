/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* RCD pass A, DESIGN.md 5.3: the 3x3 binomial low-pass of the prepped
 * mosaic and the directional weights.
 *
 * Auxiliary image layout, shared by the three RCD passes (RGBA16F, full
 * resolution):
 *
 *     .r = L    binomial 3x3 low-pass of v
 *     .g = wv   weight of the vertical estimate, Hs / (Hs + Vs + eps)
 *     .b = wp   weight of the P (NE-SW) estimate, Qs / (Ps + Qs + eps)
 *     .a = 0    unused
 *
 * Pass B (rcd_green) writes the green plane into a separate R16F image
 * rather than into .a so that no pass ever reads an image it also writes:
 * everything a pass needs was produced by an earlier dispatch, and the only
 * ordering is the barrier between dispatches. Arithmetic is fp32; only the
 * stored results are fp16. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. */
#extension GL_GOOGLE_include_directive : require

#define CRAW_PREP
#include "common.glsl"

layout(binding = 1) writeonly uniform image2D aux;

const float CRAW_RCD_EPS = 1e-5;

/* The directional values need a 5x5 stencil and are then box filtered over
 * 3x3, so one pixel depends on a 7x7 patch of prepped samples. Mirroring is
 * applied per absolute position while the patch is filled, which is exactly
 * what the neighbours' own stencils would have done. */
float craw_patch[49];

float craw_pat(int dx, int dy)
{
    return craw_patch[(dy + 3) * 7 + (dx + 3)];
}

/* Vg, Hg, Pg, Qg at the patch offset (ox, oy). P is NE-SW, Q is NW-SE. */
vec4 craw_rcd_dirs(int ox, int oy)
{
    float c2 = 2.0 * craw_pat(ox, oy);
    return vec4(abs(craw_pat(ox,     oy - 1) - craw_pat(ox,     oy + 1)) +
                abs(c2 - craw_pat(ox,     oy - 2) - craw_pat(ox,     oy + 2)),
                abs(craw_pat(ox - 1, oy    ) - craw_pat(ox + 1, oy    )) +
                abs(c2 - craw_pat(ox - 2, oy    ) - craw_pat(ox + 2, oy    )),
                abs(craw_pat(ox - 1, oy + 1) - craw_pat(ox + 1, oy - 1)) +
                abs(c2 - craw_pat(ox - 2, oy + 2) - craw_pat(ox + 2, oy - 2)),
                abs(craw_pat(ox - 1, oy - 1) - craw_pat(ox + 1, oy + 1)) +
                abs(c2 - craw_pat(ox - 2, oy - 2) - craw_pat(ox + 2, oy + 2)));
}

void main()
{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, pc.size)))
        return;

    for (int dy = -3; dy <= 3; dy++)
        for (int dx = -3; dx <= 3; dx++)
            craw_patch[(dy + 3) * 7 + (dx + 3)] = craw_v(p + ivec2(dx, dy));

    float l = (4.0 * craw_pat(0, 0) +
               2.0 * (craw_pat(0, -1) + craw_pat(0, 1) + craw_pat(-1, 0) + craw_pat(1, 0)) +
               craw_pat(-1, -1) + craw_pat(1, -1) + craw_pat(-1, 1) + craw_pat(1, 1)) * (1.0 / 16.0);

    /* 3x3 box filter of the four directional values. */
    vec4 d = vec4(0.0);
    for (int oy = -1; oy <= 1; oy++)
        for (int ox = -1; ox <= 1; ox++)
            d += craw_rcd_dirs(ox, oy);
    d *= 1.0 / 9.0;

    float wv = d.y / (d.y + d.x + CRAW_RCD_EPS);
    float wp = d.w / (d.z + d.w + CRAW_RCD_EPS);

    imageStore(aux, p, vec4(l, wv, wp, 0.0));
}
