/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Malvar-He-Cutler debayer, DESIGN.md 5.2: 5x5 gradient corrected kernels
 * over 8, with the prep step of 4.1 fused into the loads. Reads the
 * R16_UNORM mosaic, writes the RGBA16F camera RGB intermediate. The kernels
 * are applied verbatim, with no clamping: the colour stage sees whatever
 * they produce. */

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

    /* The 5x5 stencil, named by compass offset. Every kernel below is a
     * subset of it, so one set of loads serves all cases. */
    float v00 = craw_v(p);
    float vn1 = craw_v(p + ivec2( 0, -1)), vs1 = craw_v(p + ivec2( 0,  1));
    float vw1 = craw_v(p + ivec2(-1,  0)), ve1 = craw_v(p + ivec2( 1,  0));
    float vn2 = craw_v(p + ivec2( 0, -2)), vs2 = craw_v(p + ivec2( 0,  2));
    float vw2 = craw_v(p + ivec2(-2,  0)), ve2 = craw_v(p + ivec2( 2,  0));
    float vnw = craw_v(p + ivec2(-1, -1)), vne = craw_v(p + ivec2( 1, -1));
    float vsw = craw_v(p + ivec2(-1,  1)), vse = craw_v(p + ivec2( 1,  1));

    float cross1 = vn1 + vs1 + vw1 + ve1;      /* the four edge neighbours */
    float cross2 = vn2 + vs2 + vw2 + ve2;      /* the four at distance two */
    float diag   = vnw + vne + vsw + vse;      /* the four corners */

    vec3 rgb;
    rgb[c] = v00;

    if (c == 1) {
        /* Green site. The wanted colour is at the horizontal neighbours for
         * one of R/B and at the vertical ones for the other; the second
         * kernel is the transpose of the first, and the corner term is
         * shared because it is symmetric under that transpose. */
        rgb[craw_site_channel(s ^ 1)] =
            (5.0 * v00 + 4.0 * (vw1 + ve1) - (diag + vw2 + ve2) + 0.5 * (vn2 + vs2)) * (1.0 / 8.0);
        rgb[craw_site_channel(s ^ 2)] =
            (5.0 * v00 + 4.0 * (vn1 + vs1) - (diag + vn2 + vs2) + 0.5 * (vw2 + ve2)) * (1.0 / 8.0);
    } else {
        /* R or B site: green from the edge neighbours, the opposite colour
         * (2 - c) from the corners. */
        rgb[1]     = (4.0 * v00 + 2.0 * cross1 - cross2) * (1.0 / 8.0);
        rgb[2 - c] = (6.0 * v00 + 2.0 * diag - 1.5 * cross2) * (1.0 / 8.0);
    }

    imageStore(dst, p, vec4(rgb, 1.0));
}
