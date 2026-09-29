/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Capture sharpening, the first pass, DESIGN.md 20.2: the observed luma
 * the deconvolution works on,
 *
 *     d = max((R + 2 G + B) / 4, 0) + PEDESTAL
 *
 * per pixel of the camera RGB intermediate, into the fp16 plane every
 * iteration reads. One invocation per pixel of the plane's rows, which
 * are whole chunks of eight: the pixels of the last chunk that lie past
 * the frame's edge hold the pedestal and are never read as themselves.
 * One pixel an invocation, where it used to be a chunk, makes every load
 * and store of a subgroup one contiguous run of the row: eight loads a
 * chunk apart per invocation took 0.80 ms at 5.8K and this takes 0.50. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. The 16-bit storage
 * extension is storageBuffer16BitAccess, which libcraw requires: each
 * invocation stores its own half of the plane. */
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_16bit_storage : require

#include "common.glsl"
#include "sharpen.glsl"

layout(binding = 0) readonly uniform image2D craw_src;        /* camera RGB, RGBA16F */
layout(binding = 1, std430) writeonly buffer CRAWShD { uint16_t craw_d[]; };

void main()
{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= pc.stride * 8 || p.y >= pc.size.y)
        return;

    vec3 px = p.x < pc.size.x ? imageLoad(craw_src, p).rgb : vec3(0.0);
    float v = max((px.r + 2.0 * px.g + px.b) * 0.25, 0.0) + CRAW_SH_PEDESTAL;

    craw_d[p.y * pc.stride * 8 + p.x] = uint16_t(craw_sh_half(v));
}
