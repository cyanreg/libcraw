/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Capture sharpening, the first pass, DESIGN.md 20.2: the observed luma
 * the deconvolution works on,
 *
 *     d = max((R + 2 G + B) / 4, 0) + PEDESTAL
 *
 * per pixel of the camera RGB intermediate, into the fp16 plane every
 * iteration reads. One invocation per chunk of eight pixels; the pixels of
 * the last chunk that lie past the frame's edge hold the pedestal and are
 * never read as themselves. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. */
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"
#include "sharpen.glsl"

layout(binding = 0) readonly uniform image2D craw_src;        /* camera RGB, RGBA16F */
layout(binding = 1, std430) writeonly buffer CRAWShD { uvec4 craw_d[]; };

void main()
{
    ivec2 c = ivec2(gl_GlobalInvocationID.xy);
    if (c.x >= pc.stride || c.y >= pc.size.y)
        return;

    float v[8];
    for (int i = 0; i < 8; i++) {
        int x = c.x * 8 + i;
        vec3 px = x < pc.size.x ? imageLoad(craw_src, ivec2(x, c.y)).rgb : vec3(0.0);
        v[i] = max((px.r + 2.0 * px.g + px.b) * 0.25, 0.0) + CRAW_SH_PEDESTAL;
    }

    craw_d[c.y * pc.stride + c.x] = craw_sh_pack(v);
}
