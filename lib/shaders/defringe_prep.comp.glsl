/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Defringe, the first pass, DESIGN.md 19.2: per pixel of the camera RGB
 * intermediate, the quantities the second pass reads around every pixel it
 * treats,
 *
 *     .r = l = ln(max(Y, YMIN) + Y0)    .g, .b = c, the log ratios to G
 *     .a = E, the edge strength |Sobel(l)| / 8
 *
 * into the pipeline's second intermediate. E needs l over the 3x3, so that
 * is computed here from the nine RGB samples rather than read back. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. */
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"
#include "defringe.glsl"

layout(binding = 0) readonly uniform image2D src;
layout(binding = 1) writeonly uniform image2D prep;

float craw_df_l(vec3 rgb)
{
    float y = (rgb.r + 2.0 * rgb.g + rgb.b) * 0.25;
    return log(max(y, CRAW_DF_YMIN) + CRAW_DF_Y0);
}

void main()
{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, pc.size)))
        return;

    float l[9];
    for (int j = 0; j < 3; j++)
        for (int i = 0; i < 3; i++)
            l[j * 3 + i] = craw_df_l(imageLoad(src, craw_mirror(p + ivec2(i - 1, j - 1), pc.size)).rgb);

    float gx = (l[2] + 2.0 * l[5] + l[8]) - (l[0] + 2.0 * l[3] + l[6]);
    float gy = (l[6] + 2.0 * l[7] + l[8]) - (l[0] + 2.0 * l[1] + l[2]);

    vec3 rgb = imageLoad(src, p).rgb;
    float g = max(rgb.g, CRAW_DF_YMIN);
    vec2 c = vec2(log(max(rgb.r, CRAW_DF_YMIN) / g), log(max(rgb.b, CRAW_DF_YMIN) / g));

    imageStore(prep, p, vec4(l[4], c, length(vec2(gx, gy)) * 0.125));
}
