/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Bilinear debayer, DESIGN.md 5.1, with the prep step of 4.1 fused into the
 * loads: reads the R16_UNORM mosaic, writes the RGBA16F camera RGB
 * intermediate. No clamping beyond the prep clip. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. */
#extension GL_GOOGLE_include_directive : require

#define CRAW_PREP
#include "common.glsl"

layout(binding = 1) writeonly uniform image2D dst;

void main()
{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, pc.size)))
        return;

    int s = craw_site(p);
    int c = craw_site_channel(s);

    vec3 rgb;
    rgb[c] = craw_v(p);

    if (c == 1) {
        /* Green site: the two other colours sit on the horizontal and the
         * vertical neighbours. Which is which follows from the phase, so
         * the site 1 / site 2 distinction of 5.1 needs no special case. */
        rgb[craw_site_channel(s ^ 1)] = 0.5 * (craw_v(p + ivec2(-1,  0)) +
                                               craw_v(p + ivec2( 1,  0)));
        rgb[craw_site_channel(s ^ 2)] = 0.5 * (craw_v(p + ivec2( 0, -1)) +
                                               craw_v(p + ivec2( 0,  1)));
    } else {
        /* R or B site: green from the four edge neighbours, the opposite
         * colour (R <-> B, i.e. 2 - c) from the four corners. */
        rgb[1] = 0.25 * (craw_v(p + ivec2( 0, -1)) + craw_v(p + ivec2( 0, 1)) +
                         craw_v(p + ivec2(-1,  0)) + craw_v(p + ivec2( 1, 0)));
        rgb[2 - c] = 0.25 * (craw_v(p + ivec2(-1, -1)) + craw_v(p + ivec2( 1, -1)) +
                             craw_v(p + ivec2(-1,  1)) + craw_v(p + ivec2( 1,  1)));
    }

    imageStore(dst, p, vec4(rgb, 1.0));
}
