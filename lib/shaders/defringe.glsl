/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Shared by the defringe of DESIGN.md 19 and its tests: the constants of 19.2
 * and the push constant block. Include after common.glsl. */

#ifndef CRAW_DEFRINGE_GLSL
#define CRAW_DEFRINGE_GLSL

/* DESIGN.md 19.2; lib/ref/ref_defringe.c carries the same numbers. */
const float CRAW_DF_YMIN   = 1e-3;
const float CRAW_DF_Y0     = 0.02;
const float CRAW_DF_E_LO   = 0.15;
const float CRAW_DF_E_HI   = 0.35;
const float CRAW_DF_B_LO   = 0.005;
const float CRAW_DF_B_HI   = 0.02;
const int   CRAW_DF_D_MIN  = 2;
const int   CRAW_DF_D_MAX  = 6;
const float CRAW_DF_DARK   = 0.1;
const float CRAW_DF_MARGIN = 0.03;

/* Mirrors CRAWDefringePush in lib/pipeline.c. */
layout(push_constant, scalar) uniform CRAWDefringePushBuf {
    ivec2 size;
    float strength;    /* 0 < strength <= 1 */
} pc;

#endif /* CRAW_DEFRINGE_GLSL */
