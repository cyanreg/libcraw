/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Raw domain denoise, DESIGN.md 14.3: non-local means on the four CFA
 * planes in a variance stabilised domain.
 *
 * One dispatch, one workgroup per tile of one plane, the plane index in
 * gl_WorkGroupID.z (0 = the (0,0) site of the quad, 1 = (1,0), 2 = (0,1),
 * 3 = (1,1), i.e. R, Gr, Gb, B for RGGB). Each plane is half resolution and
 * is filtered on its own: the noise is per sensel and independent between
 * sites, and filtering before the debayer is what keeps it from being
 * smeared into colour.
 *
 * Reads the R16_UNORM mosaic and writes a denoised R16_UNORM mosaic in the
 * same raw encoding, so every stage downstream is unchanged and the prep of
 * 4.1 still applies to it. Re-encoding costs a quantization step of 1/65535
 * of the sample range, which is a fraction of a code value against a read
 * noise of a couple of them.
 *
 * Arithmetic is fp32 throughout. The stabilised values sit on a large
 * offset with a small useful range around it (the whole point of the
 * transform is that one unit is one noise standard deviation), so fp16
 * would quantize the signal itself away. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. */
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(binding = 0) readonly  uniform image2D craw_mosaic;
layout(binding = 1) writeonly uniform image2D craw_denoised;

/* Search and patch radii, chosen by the host. A patch radius of 0 turns
 * the same binary into the bilateral fallback of DESIGN.md 14.3: the patch
 * distance is then the plain difference of two stabilised samples. */
layout (constant_id = 2) const int craw_dn_search = 3;
layout (constant_id = 3) const int craw_dn_patch  = 1;

/* Largest tile the shared array can hold: a 16x16 workgroup plus a halo of
 * search + patch on each side, with that sum capped at 6. The host checks
 * this; see CRAW_DN_TILE_MAX in lib/pipeline.c. */
#define CRAW_DN_TILE_MAX  28
#define CRAW_DN_PATCH_MAX 25   /* (2 * 2 + 1)^2 */

shared float craw_tile[CRAW_DN_TILE_MAX * CRAW_DN_TILE_MAX];

/* Mirrors CRAWDenoisePush in lib/pipeline.c. */
layout(push_constant, scalar) uniform CRAWDenoisePushBuf {
    ivec2 size;        /* plane size, mosaic size / 2 */
    vec4  black;       /* black level per quad site, UNORM */
    vec4  inv_range;   /* 1 / (white - black) per quad site */
    vec4  range;       /* white - black per quad site */
    float gat_a;       /* noise model a */
    float gat_c;       /* 3 a^2 / 8 + b */
    float gat_sc;      /* sqrt(gat_c) */
    float gat_fwd;     /* 2 / a */
    float gat_inv;     /* a / 4 */
    float gat_off;     /* gat_c / a */
    float inv_h2;      /* 1 / strength^2 */
    float guard_lo;    /* highlight ramp start, in u */
    float guard_inv;   /* 1 / (guard_hi - guard_lo) */
} pc;

/* The generalised Anscombe transform of DESIGN.md 14.3, with the square
 * root replaced by its own tangent below u = 0. Read noise puts samples
 * below the black level, where a * u + c goes negative and the textbook
 * transform has neither a value nor a bounded derivative; the tangent
 * continues it with the constant slope the variance there actually calls
 * for, and is exactly invertible. */
float craw_dn_gat(float u)
{
    if (u >= 0.0)
        return pc.gat_fwd * sqrt(pc.gat_a * u + pc.gat_c);

    return pc.gat_fwd * pc.gat_sc + u / pc.gat_sc;
}

float craw_dn_igat(float t)
{
    float t0 = pc.gat_fwd * pc.gat_sc;

    if (t >= t0)
        return pc.gat_inv * t * t - pc.gat_off;

    return (t - t0) * pc.gat_sc;
}

/* The mosaic sample of plane `pl` at plane position q, as a level above
 * black in units of the site's own range. No white balance and no clip:
 * the noise model is a property of the sensor, and both of those belong to
 * the prep the debayer does afterwards. */
float craw_dn_level(ivec2 q, int pl, int px, int py)
{
    float raw = imageLoad(craw_mosaic, ivec2(2 * q.x + px, 2 * q.y + py)).x;
    return (raw - pc.black[pl]) * pc.inv_range[pl];
}

void main()
{
    int pl = int(gl_WorkGroupID.z);
    int px = pl & 1, py = pl >> 1;
    int rad = craw_dn_search + craw_dn_patch;
    ivec2 wgs = ivec2(gl_WorkGroupSize.xy);
    ivec2 org = ivec2(gl_WorkGroupID.xy) * wgs - rad;
    ivec2 tsz = wgs + 2 * rad;
    ivec2 lid = ivec2(gl_LocalInvocationID.xy);

    /* The tile holds the stabilised plane, halo included. Borders are
     * mirrored in the plane's own coordinates (DESIGN.md 14.3), which is
     * what the neighbours' own tiles would have loaded. */
    for (int ty = lid.y; ty < tsz.y; ty += wgs.y) {
        for (int tx = lid.x; tx < tsz.x; tx += wgs.x) {
            ivec2 q = craw_mirror(org + ivec2(tx, ty), pc.size);
            craw_tile[ty * tsz.x + tx] = craw_dn_gat(craw_dn_level(q, pl, px, py));
        }
    }

    barrier();

    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, pc.size)))
        return;

    int tw = tsz.x;
    int ctr = (lid.y + rad) * tw + lid.x + rad;

    /* The centre patch is read once and kept in registers; every search
     * position then costs one pass over the patch in shared memory. */
    float cp[CRAW_DN_PATCH_MAX];
    int np = 0;
    for (int ky = -craw_dn_patch; ky <= craw_dn_patch; ky++)
        for (int kx = -craw_dn_patch; kx <= craw_dn_patch; kx++)
            cp[np++] = craw_tile[ctr + ky * tw + kx];

    float inv_np = 1.0 / float(np);
    float acc = 0.0, wsum = 0.0;

    for (int dy = -craw_dn_search; dy <= craw_dn_search; dy++) {
        for (int dx = -craw_dn_search; dx <= craw_dn_search; dx++) {
            int s = ctr + dy * tw + dx;
            float d2 = 0.0;
            int n = 0;

            for (int ky = -craw_dn_patch; ky <= craw_dn_patch; ky++) {
                for (int kx = -craw_dn_patch; kx <= craw_dn_patch; kx++) {
                    float e = cp[n++] - craw_tile[s + ky * tw + kx];
                    d2 += e * e;
                }
            }

            /* Two stabilised samples of the same true value differ by a
             * variance of 2, so that much of the distance is noise and is
             * subtracted before it can count against a neighbour. The
             * centre itself always lands on weight 1, so wsum >= 1. */
            float w = exp(-max(d2 * inv_np - 2.0, 0.0) * pc.inv_h2);

            wsum += w;
            acc += w * craw_tile[s];
        }
    }

    float u = craw_dn_igat(acc / wsum);

    /* Near the sensor's clip point the noise distribution is truncated and
     * the model no longer holds, so the filter is ramped off and the
     * original sample passes through. Pulling a clipped sample down would
     * also un-clip it in one channel only, which is exactly what the
     * balanced clip of 4.1 exists to prevent. */
    ivec2 mp = ivec2(2 * p.x + px, 2 * p.y + py);
    float raw0 = imageLoad(craw_mosaic, mp).x;
    float u0 = (raw0 - pc.black[pl]) * pc.inv_range[pl];
    float g = clamp((u0 - pc.guard_lo) * pc.guard_inv, 0.0, 1.0);

    u = mix(u, u0, g);

    imageStore(craw_denoised, mp, vec4(clamp(pc.black[pl] + u * pc.range[pl], 0.0, 1.0)));
}
