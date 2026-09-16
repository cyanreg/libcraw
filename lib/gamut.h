/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#ifndef CRAW_GAMUT_H
#define CRAW_GAMUT_H

/* ACES 1.3 Reference Gamut Compression, DESIGN.md 8.1.
 *
 * The parameters are constants of the specification, but the per-channel
 * scale factors have to be solved for from them, so they live in one struct
 * that craw_gamut_compress_params() fills. The pipeline pushes that struct
 * to the colour shader and the CPU reference runs the same numbers through
 * craw_gamut_compress(): there is exactly one place the constants are
 * derived, so the shader and the reference cannot drift apart. */

typedef struct CRAWGamutCompress {
    float thr[3];      /* threshold distance per channel: cyan, magenta, yellow */
    float scl[3];      /* solved so that thr + scl * f(lim) lands on 1.0 */
    float pwr;         /* curve aggressiveness */
    float inv_pwr;     /* 1 / pwr, so the shader does not divide per pixel */
} CRAWGamutCompress;

/* Fills `g` with the ACES 1.3 constants and the scales derived from them. */
void craw_gamut_compress_params(CRAWGamutCompress *g);

/* Applies the compression to one AP1 linear RGB triple in place. */
void craw_gamut_compress(const CRAWGamutCompress *g, float rgb[3]);

#endif /* CRAW_GAMUT_H */
