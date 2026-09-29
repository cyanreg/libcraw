/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Shared by the two capture sharpening passes of DESIGN.md 20: the
 * constants of 20.2, the push constant block and the planes' layout.
 * Include after common.glsl. lib/sharpen.h carries the same numbers for
 * the host and the CPU reference. */

#ifndef CRAW_SHARPEN_GLSL
#define CRAW_SHARPEN_GLSL

#define CRAW_SH_RMAX      6
#define CRAW_SH_MASK_R    2
#define CRAW_SH_RANGE_R   1

const float CRAW_SH_PEDESTAL = 6.103515625e-5;
const float CRAW_SH_CORR_MAX = 2.0;

/* Mirrors CRAWSharpenPush in lib/pipeline.c. */
layout(push_constant, scalar) uniform CRAWSharpenPushBuf {
    ivec2 size;
    int   stride;      /* a plane's row, in chunks of eight pixels */
    float k[CRAW_SH_RMAX + 1];  /* taps, k[0] the centre, k[i] at +-i up to R */
    float contrast;    /* the mask's threshold in noise units, 0 for none */
    float noise_a;     /* the statistic's modelled variance is */
    float noise_b;     /* noise_a * level + noise_b */
} pc;

/* The luma and the estimates are storage buffers of fp16 pixels, eight to
 * a 16-byte chunk and `stride` chunks to a row, so that a workgroup reads
 * its tile with one load per eight pixels; a formatless image load per
 * pixel took three quarters of an iteration. Pixel x of a chunk is half
 * (x & 1) of component (x & 7) >> 1, the even pixel in the low half, as
 * unpackHalf2x16() reads it.
 *
 * The conversion to fp16 rounds to nearest, ties to even, as an image
 * store does and as the reference does. packHalf2x16() does not promise a
 * rounding and on AMD truncates, which biased every iteration's estimate
 * low by a quarter of an fp16 step on average. Every value stored here is
 * at least the pedestal, the smallest normal fp16, so the normal case is
 * the only one: add just under half of the thirteen bits that go, plus
 * the lowest bit that stays, and let a carry run into the exponent. */
uint craw_sh_half(float f)
{
    uint b = floatBitsToUint(min(f, 65504.0));
    b += 0xFFFu + ((b >> 13) & 1u);
    return (b >> 13) - ((127u - 15u) << 10);
}

uvec4 craw_sh_pack(float v[8])
{
    uvec4 o;
    for (int i = 0; i < 4; i++)
        o[i] = craw_sh_half(v[2 * i]) | (craw_sh_half(v[2 * i + 1]) << 16);
    return o;
}

#endif /* CRAW_SHARPEN_GLSL */
