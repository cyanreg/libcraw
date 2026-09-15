/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* RCD pass B, DESIGN.md 5.3: green at the R and B sites from the vertical
 * and horizontal neighbours corrected by the low-pass ratio, blended with
 * the directional weight from pass A. At green sites the native sample is
 * copied. Writes a full green plane (R16F) that pass C reads; see
 * rcd_dir.comp.glsl for the auxiliary image layout. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. */
#extension GL_GOOGLE_include_directive : require

#define CRAW_PREP
#include "common.glsl"

layout(binding = 1) readonly  uniform image2D aux;     /* (L, wv, wp, -) */
layout(binding = 2) writeonly uniform image2D green;

const float CRAW_RCD_EPS = 1e-5;

float craw_lp(ivec2 p)
{
    return imageLoad(aux, craw_mirror(p, pc.size)).x;
}

/* One neighbour's contribution: its own sample scaled by the ratio of the
 * low-pass at the centre to the low-pass at the neighbour's own colour site
 * two pixels out, written as 1 + (L0 - L2) / (L0 + L2 + eps) so that it
 * degrades to a plain copy when the low-pass is flat or near zero. */
float craw_rcd_lpf(float v, float l0, float l2)
{
    return v * (1.0 + (l0 - l2) / (l0 + l2 + CRAW_RCD_EPS));
}

void main()
{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, pc.size)))
        return;

    int s = craw_site(p);
    float g;

    if (craw_site_channel(s) == 1) {
        g = craw_v(p);
    } else {
        float l0 = craw_lp(p);
        float gv = 0.5 * (craw_rcd_lpf(craw_v(p + ivec2(0, -1)), l0, craw_lp(p + ivec2(0, -2))) +
                          craw_rcd_lpf(craw_v(p + ivec2(0,  1)), l0, craw_lp(p + ivec2(0,  2))));
        float gh = 0.5 * (craw_rcd_lpf(craw_v(p + ivec2(-1, 0)), l0, craw_lp(p + ivec2(-2, 0))) +
                          craw_rcd_lpf(craw_v(p + ivec2( 1, 0)), l0, craw_lp(p + ivec2( 2, 0))));
        float wv = imageLoad(aux, p).y;
        g = wv * gv + (1.0 - wv) * gh;
    }

    imageStore(green, p, vec4(g));
}
