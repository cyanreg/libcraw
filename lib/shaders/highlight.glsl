/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Shared by the highlight reconstruction passes of DESIGN.md 18: the
 * constants of 18.2 and, for the two passes that read the mosaic (the field
 * and the fill), their push constant block and the per-site quantities u and
 * a. The pull and push passes over the field's pyramid define
 * CRAW_HL_LEVELS instead and get the level block. Include after
 * common.glsl. */

#ifndef CRAW_HIGHLIGHT_GLSL
#define CRAW_HIGHLIGHT_GLSL

/* DESIGN.md 18.2; lib/ref/ref_highlight.c carries the same numbers. */
const float CRAW_HL_U_LO    = 0.92;
const float CRAW_HL_U_HI    = 0.98;
const float CRAW_HL_EPS     = 1e-4;
const float CRAW_HL_LOG_MAX = 2.0;
const float CRAW_HL_W_SAT   = 1.0;

#ifdef CRAW_HL_LEVELS

/* Mirrors CRAWHighlightLevelPush in lib/pipeline.c. The pull reads `fine`
 * and writes `coarse`; the push reads `coarse` and rewrites `fine`. */
layout(push_constant, scalar) uniform CRAWHighlightLevelPushBuf {
    ivec2 fine;        /* size of the finer of the two levels */
    ivec2 coarse;      /* size of the coarser one */
    int   top;         /* push: the coarser level is the top, still unnormalized */
} pc;

#else

/* Mirrors CRAWHighlightPush in lib/pipeline.c. */
layout(push_constant, scalar) uniform CRAWHighlightPushBuf {
    ivec2 size;        /* mosaic size in pixels */
    ivec2 fsize;       /* field level 0, ceil(size / 4) */
    vec4  black;       /* black level per quad site, UNORM */
    vec4  inv_range;   /* 1 / (white - black) per quad site */
    vec4  wb;          /* the frame's gains in .rgb, .a unused */
} pc;

/* The fraction of the sensor range: the prep of DESIGN.md 4.1 without its
 * gain and its clip. */
float craw_hl_u(float raw, int site)
{
    return max((raw - pc.black[site]) * pc.inv_range[site], 0.0);
}

/* How clipped a sample is, 0 to 1. */
float craw_hl_ramp(float u)
{
    return smoothstep(CRAW_HL_U_LO, CRAW_HL_U_HI, u);
}

#endif /* CRAW_HL_LEVELS */

#endif /* CRAW_HIGHLIGHT_GLSL */
