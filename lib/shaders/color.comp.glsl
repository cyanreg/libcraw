/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Colour stage, DESIGN.md 4.3: re-balance the camera RGB intermediate to
 * the frame's illuminant, apply one combined 3x3 and write the caller's
 * destination image. The matrix is M_out * cam_to_xyz * gain * 2^stops,
 * multiplied out on the host in fp32, so the shader is a wb multiply, three
 * dot products and an optional clamp. The destination is formatless: the
 * same binary writes RGBA16F and RGBA32F. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. */
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(binding = 0) readonly  uniform image2D src;   /* camera RGB, RGBA16F */
layout(binding = 1) writeonly uniform image2D dst;

/* Mirrors CRAW_FRAME_FLAG_CLAMP_NEGATIVE in <craw/pipeline.h>. */
#define CRAW_COLOR_CLAMP_NEGATIVE 1u

/* Mirrors CRAWColorPush in lib/pipeline.c. A GLSL mat3 in scalar layout is
 * three *columns* of vec3, while every matrix in libcraw is row major, so
 * the rows are passed as a vec3 array and multiplied by hand below: no
 * transpose happens anywhere between profile.c and here. Scalar layout
 * makes the array stride 12 bytes, matching float m[3][3] in C. */
layout(push_constant, scalar) uniform CRAWColorPushBuf {
    ivec2 size;
    vec3  m[3];       /* rows of M_out * cam_to_xyz * gain * 2^stops */
    vec3  wb_ratio;   /* frame wb / as-shot wb, componentwise */
    uint  flags;
} pc;

void main()
{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, pc.size)))
        return;

    vec3 c = imageLoad(src, p).rgb * pc.wb_ratio;
    vec3 o = vec3(dot(pc.m[0], c), dot(pc.m[1], c), dot(pc.m[2], c));

    /* Off by default: values outside the camera's characterized gamut are
     * meaningful, and so are values above 1.0. */
    if ((pc.flags & CRAW_COLOR_CLAMP_NEGATIVE) != 0u)
        o = max(o, vec3(0.0));

    imageStore(dst, p, vec4(o, 1.0));
}
