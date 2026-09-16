/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Colour stage, DESIGN.md 4.3: re-balance the camera RGB intermediate to
 * the frame's illuminant, apply one combined 3x3 and write the caller's
 * destination image. The matrix is M_out * cam_to_xyz * gain * 2^stops,
 * multiplied out on the host in fp32, so the shader is a wb multiply, three
 * dot products, an optional gamut compression and an optional clamp. The
 * destination is formatless: the same binary writes RGBA16F and RGBA32F. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. */
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(binding = 0) readonly  uniform image2D src;   /* camera RGB, RGBA16F */
layout(binding = 1) writeonly uniform image2D dst;

/* Mirrors CRAW_FRAME_FLAG_CLAMP_NEGATIVE in <craw/pipeline.h>. */
#define CRAW_COLOR_CLAMP_NEGATIVE 1u

/* Not a public frame flag: the host sets this when the output space is
 * ACEScg *and* the frame did not ask for CRAW_FRAME_FLAG_NO_GAMUT_COMPRESSION,
 * so the shader never has to know which colour space it is writing. */
#define CRAW_COLOR_GAMUT_COMPRESS 2u

/* Mirrors CRAWColorPush in lib/pipeline.c. A GLSL mat3 in scalar layout is
 * three *columns* of vec3, while every matrix in libcraw is row major, so
 * the rows are passed as a vec3 array and multiplied by hand below: no
 * transpose happens anywhere between profile.c and here. Scalar layout
 * makes the array stride 12 bytes, matching float m[3][3] in C. */
layout(push_constant, scalar) uniform CRAWColorPushBuf {
    ivec2 size;
    vec3  m[3];       /* rows of M_out * cam_to_xyz * gain * 2^stops */
    vec3  wb_ratio;   /* frame wb / as-shot wb, componentwise */
    vec3  gc_thr;     /* ACES 1.3 RGC, see lib/gamut.c: thresholds, */
    vec3  gc_scl;     /* the scales solved from the distance limits, */
    float gc_pwr;     /* the curve power and its reciprocal. */
    float gc_inv_pwr;
    uint  flags;
} pc;

/* ACES 1.3 Reference Gamut Compression, DESIGN.md 8.1. The same arithmetic
 * as craw_gamut_compress() in lib/gamut.c, which is the transcription of
 * the Academy's CTL; the constants arrive derived from the host so the two
 * cannot disagree about them. */
vec3 craw_gamut_compress(vec3 c)
{
    float ach = max(c.r, max(c.g, c.b));

    /* The CTL's own guard: pure black has no direction to be pulled in. */
    if (ach == 0.0)
        return c;

    float mag = abs(ach);
    vec3  d   = (vec3(ach) - c) / mag;

    /* Clamped at zero only so that pow() below never sees a negative base;
     * the select afterwards throws that branch's result away anyway. */
    vec3 nd = max(d - pc.gc_thr, vec3(0.0)) / pc.gc_scl;
    vec3 cd = pc.gc_thr + pc.gc_scl * nd / pow(vec3(1.0) + pow(nd, vec3(pc.gc_pwr)),
                                               vec3(pc.gc_inv_pwr));

    /* Below the threshold the original component is kept rather than
     * rebuilt from its own distance: ach - ((ach - c) / mag) * mag is a
     * float round trip that can land an ulp away, while the reference in
     * lib/gamut.c simply skips the channel. Keeping it makes the two agree
     * exactly, which matters for an RGBA32F destination. */
    return mix(vec3(ach) - cd * mag, c, lessThan(d, pc.gc_thr));
}

void main()
{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, pc.size)))
        return;

    vec3 c = imageLoad(src, p).rgb * pc.wb_ratio;
    vec3 o = vec3(dot(pc.m[0], c), dot(pc.m[1], c), dot(pc.m[2], c));

    /* Uniform across the dispatch, so the branch costs a scalar compare. */
    if ((pc.flags & CRAW_COLOR_GAMUT_COMPRESS) != 0u)
        o = craw_gamut_compress(o);

    /* Off by default: values outside the camera's characterized gamut are
     * meaningful, and so are values above 1.0. */
    if ((pc.flags & CRAW_COLOR_CLAMP_NEGATIVE) != 0u)
        o = max(o, vec3(0.0));

    imageStore(dst, p, vec4(o, 1.0));
}
