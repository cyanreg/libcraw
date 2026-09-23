/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Shared helpers for every libcraw compute shader: CFA geometry, mirrored
 * borders and the prep step of DESIGN.md 4.1.
 *
 * Shaders that read the Bayer mosaic define CRAW_PREP before including this
 * file. That declares the mosaic image at binding 0 and the push constant
 * block all five debayer passes share (CRAWPrepPush in lib/pipeline.c,
 * mirrored field for field), and defines craw_v(): one prepped sample with
 * mirrored borders. Shaders that only touch the intermediate (colour, awb)
 * leave CRAW_PREP undefined and declare their own bindings. */

#ifndef CRAW_COMMON_GLSL
#define CRAW_COMMON_GLSL

#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_image_load_formatted : require

/* The host picks the workgroup size (the 253/254/255 convention), so that
 * one shader binary can be tuned per device without a recompile. */
layout (local_size_x_id = 253, local_size_y_id = 254, local_size_z_id = 255) in;

/* CFA phase: 0 = RGGB, 1 = BGGR, 2 = GRBG, 3 = GBRG. Only RGGB is accepted
 * by craw_pipeline_create(); the others are plumbed so that enabling them
 * is a validation change and a test, not a shader change. */
layout (constant_id = 0) const uint craw_cfa = 0u;

/* Quad site index of a pixel, DESIGN.md 3: s = (y & 1) * 2 + (x & 1). */
int craw_site(ivec2 p)
{
    return ((p.y & 1) << 1) | (p.x & 1);
}

/* Channel (0 = R, 1 = G, 2 = B) native to a quad site, for the phase in
 * specialization constant 0. Two bits per site with site 0 in the low bits,
 * one byte per phase with phase 0 in the low byte: RGGB 0,1,1,2 -> 0x94;
 * BGGR 2,1,1,0 -> 0x16; GRBG 1,0,2,1 -> 0x61; GBRG 1,2,0,1 -> 0x49. A
 * packed constant rather than an array keeps the lookup a pair of shifts
 * even though the phase is a specialization constant. */
#define CRAW_CFA_CHANNELS 0x49611694u

int craw_site_channel(int site)
{
    uint row = (CRAW_CFA_CHANNELS >> (craw_cfa * 8u)) & 0xFFu;
    return int((row >> (uint(site) * 2u)) & 3u);
}

/* Mirrored border, DESIGN.md 3: x < 0 -> -x, x >= W -> 2W - 2 - x, same for
 * y. The edge sample is not repeated and both reflections are even, so the
 * CFA phase of the mirrored position matches the requested one. Valid for
 * one reflection, i.e. stencil radius <= W - 1 and <= H - 1. */
ivec2 craw_mirror(ivec2 p, ivec2 size)
{
    return mix(abs(p), 2 * size - 2 - p, greaterThanEqual(p, size));
}

#ifdef CRAW_PREP

layout(binding = 0) readonly uniform image2D craw_mosaic;

/* Mirrors CRAWPrepPush in lib/pipeline.c. */
layout(push_constant, scalar) uniform CRAWPrepPushBuf {
    ivec2 size;        /* mosaic size in pixels */
    vec4  black;       /* black level per quad site, UNORM */
    vec4  inv_range;   /* 1 / (white - black) per quad site */
    vec4  wb;          /* as-shot gains in .rgb, .a unused */
    float clip;        /* 1.0, or FLT_MAX on the rebuilt mosaic */
} pc;

/* DESIGN.md 4.1. The clip at 1.0 sits after the white balance so that blown
 * highlights stay neutral; in raw units the clip point therefore differs per
 * channel. On the mosaic the highlight reconstruction of DESIGN.md 18 wrote,
 * the host makes all of this the identity: black 0, range 1, gains 1 and a
 * clip nothing reaches, because that mosaic already holds balanced values
 * and its highlights are meant to exceed 1.0. */
float craw_prep(float raw, int site)
{
    float v = (raw - pc.black[site]) * pc.inv_range[site];
    v = max(v, 0.0);
    v *= pc.wb[craw_site_channel(site)];
    return min(v, pc.clip);
}

/* The prepped sample at an absolute position, borders mirrored. */
float craw_v(ivec2 p)
{
    ivec2 q = craw_mirror(p, pc.size);
    return craw_prep(imageLoad(craw_mosaic, q).x, craw_site(q));
}

#endif /* CRAW_PREP */

#endif /* CRAW_COMMON_GLSL */
