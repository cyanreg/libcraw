/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Raw domain denoise, DESIGN.md 14.3: non-local means on the four CFA
 * planes in a variance stabilised domain, and in front of it, in the same
 * domain, the temporal recursive average of DESIGN.md 14.7.
 *
 * One dispatch, one workgroup per tile of one plane, the plane index in
 * the low two bits of gl_WorkGroupID.x (0 = the (0,0) site of the quad,
 * 1 = (1,0), 2 = (0,1), 3 = (1,1), i.e. R, Gr, Gb, B for RGGB) and the
 * tile's column in the rest. Each plane is half resolution and is filtered
 * on its own: the noise is per sensel and independent between sites, and
 * filtering before the debayer is what keeps it from being smeared into
 * colour. The four planes of a tile are neighbours in the dispatch so that
 * they run together and share the cache lines of the mosaic and of the
 * state, which interleave all four: a plane at a time, each would stream
 * the whole frame for a quarter of every line it fetched.
 *
 * Reads the R16_UNORM mosaic and writes a denoised R16_UNORM mosaic in the
 * same raw encoding, so every stage downstream is unchanged and the prep of
 * 4.1 still applies to it. Re-encoding costs a quantization step of 1/65535
 * of the sample range, which is a fraction of a code value against a read
 * noise of a couple of them. With the temporal pass it also reads the
 * previous frame's state and writes this frame's; either filter runs
 * without the other.
 *
 * Arithmetic is fp32 throughout, and so is the state. The stabilised
 * values sit on a large offset with a small useful range around it (the
 * whole point of the transform is that one unit is one noise standard
 * deviation), so fp16 would quantize the signal itself away. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. */
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(binding = 0) readonly  uniform image2D craw_mosaic;
layout(binding = 1) writeonly uniform image2D craw_denoised;

/* The temporal state, DESIGN.md 14.7: the previous frame's temporally
 * filtered planes, and where this frame's go. Full resolution R32F images
 * indexed like the mosaic, holding the level u of every sample rather than
 * its stabilised value, so that a frame with another noise model reads
 * last frame's state through its own transform; and beside each, R8_UNORM,
 * the square root of the noise variance left in that sample, in
 * stabilised units, which is 1 for a frame taken as it came. Bound to
 * stand-ins that are never touched when the mode below does not ask for
 * them. */
layout(binding = 2) readonly  uniform image2D craw_state_prev;
layout(binding = 3) writeonly uniform image2D craw_state_next;
layout(binding = 4) readonly  uniform image2D craw_var_prev;
layout(binding = 5) writeonly uniform image2D craw_var_next;

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

/* With a history, the noise variance left in each sample of the tile. */
shared float craw_tvar[CRAW_DN_TILE_MAX * CRAW_DN_TILE_MAX];

/* The temporal pass compares 3x3 patches of this frame and the last, so
 * both are loaded one sample further out than the tile: 30x30 at most. */
#define CRAW_TN_TILE_MAX (CRAW_DN_TILE_MAX + 2)

shared float craw_tcur[CRAW_TN_TILE_MAX * CRAW_TN_TILE_MAX];
shared float craw_tprev[CRAW_TN_TILE_MAX * CRAW_TN_TILE_MAX];

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
    float t_alpha;     /* alpha_max of DESIGN.md 14.7 */
    float t_inv_h2;    /* 1 / ht^2 */
    int   mode;        /* CRAW_DN_* below */
} pc;

/* Mirrors the CRAW_DN_MODE_* bits of lib/pipeline.c. The host never sets
 * HISTORY without TEMPORAL, and never dispatches with neither filter. */
#define CRAW_DN_SPATIAL  1   /* the non-local means of 14.3 */
#define CRAW_DN_TEMPORAL 2   /* the recursive average of 14.7; writes the state */
#define CRAW_DN_HISTORY  4   /* the state holds the previous frame: read it */

/* The least noise variance a sample is taken to carry, a hundredth of a
 * noise standard deviation: the spatial filter divides by it, and the
 * recursion could otherwise approach 0 at an alpha_max of 1. Mirrors
 * CRAW_REF_DN_VAR_MIN in lib/ref/ref_denoise.c. */
#define CRAW_DN_VAR_MIN 1.0e-4

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

/* The mosaic sample of plane `pl` at mosaic position mp, as a level above
 * black in units of the site's own range. No white balance and no clip:
 * the noise model is a property of the sensor, and both of those belong to
 * the prep the debayer does afterwards. */
float craw_dn_level(ivec2 mp, int pl)
{
    float raw = imageLoad(craw_mosaic, mp).x;
    return (raw - pc.black[pl]) * pc.inv_range[pl];
}

/* The weight of last frame in the recursive average of DESIGN.md 14.7 at
 * the sample at `c` of the two temporal tiles, `w` wide: this frame's
 * stabilised value against last frame's filtered one at the same place,
 * weighted as 14.3 weights a spatial neighbour, with the 3x3 patch around
 * the sample in both frames as the patch, and capped at alpha_max so that
 * nothing survives for ever. The 2 is the same 2 sigma^2 as there. A patch
 * rather than the sample alone, because a sample alone is compared against
 * its own noise: every excursion past a sigma and a half reads as motion
 * and is passed through unaveraged, and those are exactly the samples the
 * average was for. */
float craw_dn_alpha(int c, int w)
{
    float d2 = 0.0;

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            float e = craw_tprev[c + ky * w + kx] - craw_tcur[c + ky * w + kx];
            d2 += e * e;
        }
    }

    return pc.t_alpha * exp(-max(d2 * (1.0 / 9.0) - 2.0, 0.0) * pc.t_inv_h2);
}

/* Near the sensor's clip point the noise distribution is truncated and the
 * model no longer holds, so the filter is ramped off and the original
 * sample passes through. Pulling a clipped sample down would also un-clip
 * it in one channel only, which is exactly what the balanced clip of 4.1
 * exists to prevent. */
void craw_dn_store(ivec2 mp, int pl, float u0, float u)
{
    float g = clamp((u0 - pc.guard_lo) * pc.guard_inv, 0.0, 1.0);

    u = mix(u, u0, g);

    imageStore(craw_denoised, mp, vec4(clamp(pc.black[pl] + u * pc.range[pl], 0.0, 1.0)));
}

void main()
{
    int pl = int(gl_WorkGroupID.x) & 3;
    int px = pl & 1, py = pl >> 1;
    ivec2 wid = ivec2(int(gl_WorkGroupID.x) >> 2, int(gl_WorkGroupID.y));
    bool spatial = (pc.mode & CRAW_DN_SPATIAL) != 0;
    bool history = (pc.mode & CRAW_DN_HISTORY) != 0;
    /* The spatial filter's halo; the temporal average alone needs none. */
    int rad = spatial ? craw_dn_search + craw_dn_patch : 0;
    ivec2 wgs = ivec2(gl_WorkGroupSize.xy);
    ivec2 org = wid * wgs - rad;
    ivec2 tsz = wgs + 2 * rad;
    ivec2 lid = ivec2(gl_LocalInvocationID.xy);

    /* The tile holds the stabilised plane, halo included, already
     * averaged with the previous frame when the temporal pass has one: the
     * spatial filter works on what the temporal one left. Borders are
     * mirrored in the plane's own coordinates (DESIGN.md 14.3), which is
     * what the neighbours' own tiles would have loaded, and the temporal
     * value of a sample is a function of the 3x3 around it in the two
     * frames, mirrored the same way, so every tile that holds it holds the
     * same number. */
    if (history) {
        ivec2 esz = tsz + 2;

        for (int ty = lid.y; ty < esz.y; ty += wgs.y) {
            for (int tx = lid.x; tx < esz.x; tx += wgs.x) {
                ivec2 q = craw_mirror(org - 1 + ivec2(tx, ty), pc.size);
                ivec2 mq = 2 * q + ivec2(px, py);

                craw_tcur[ty * esz.x + tx] = craw_dn_gat(craw_dn_level(mq, pl));
                craw_tprev[ty * esz.x + tx] = craw_dn_gat(imageLoad(craw_state_prev, mq).x);
            }
        }

        barrier();

        /* The variance a sample is left with follows from the weight: this
         * frame brings a variance of 1 and last frame's state what it
         * carried, which the state keeps as its square root. */
        for (int ty = lid.y; ty < tsz.y; ty += wgs.y) {
            for (int tx = lid.x; tx < tsz.x; tx += wgs.x) {
                int c = (ty + 1) * esz.x + tx + 1;
                ivec2 q = craw_mirror(org + ivec2(tx, ty), pc.size);
                float sp = imageLoad(craw_var_prev, 2 * q + ivec2(px, py)).x;
                float a = craw_dn_alpha(c, esz.x);

                craw_tile[ty * tsz.x + tx] = mix(craw_tcur[c], craw_tprev[c], a);
                craw_tvar[ty * tsz.x + tx] = max((1.0 - a) * (1.0 - a) + a * a * sp * sp,
                                                 CRAW_DN_VAR_MIN);
            }
        }
    } else {
        for (int ty = lid.y; ty < tsz.y; ty += wgs.y) {
            for (int tx = lid.x; tx < tsz.x; tx += wgs.x) {
                ivec2 q = craw_mirror(org + ivec2(tx, ty), pc.size);
                craw_tile[ty * tsz.x + tx] = craw_dn_gat(craw_dn_level(2 * q + ivec2(px, py), pl));
            }
        }
    }

    barrier();

    ivec2 p = wid * wgs + lid;
    if (any(greaterThanEqual(p, pc.size)))
        return;

    int tw = tsz.x;
    int ctr = (lid.y + rad) * tw + lid.x + rad;
    ivec2 mp = 2 * p + ivec2(px, py);

    /* The state is what the temporal pass made of this sample, before the
     * spatial filter: fed back through it, a static scene would be
     * filtered again every frame and smoothed further without end. It goes
     * to the other image of the pair (DESIGN.md 14.7): written over the
     * one being read, a neighbouring tile that holds this sample in its
     * halo could read it back already averaged once. */
    if ((pc.mode & CRAW_DN_TEMPORAL) != 0) {
        imageStore(craw_state_next, mp, vec4(craw_dn_igat(craw_tile[ctr])));
        imageStore(craw_var_next, mp, vec4(history ? sqrt(craw_tvar[ctr]) : 1.0));
    }

    /* The guard compares against the sample as it came in, not against
     * its temporal average. */
    if (!spatial) {
        craw_dn_store(mp, pl, craw_dn_level(mp, pl), craw_dn_igat(craw_tile[ctr]));
        return;
    }

    /* The centre patch is read once and kept in registers; every search
     * position then costs one pass over the patch in shared memory. */
    float cp[CRAW_DN_PATCH_MAX];
    int np = 0;
    for (int ky = -craw_dn_patch; ky <= craw_dn_patch; ky++)
        for (int kx = -craw_dn_patch; kx <= craw_dn_patch; kx++)
            cp[np++] = craw_tile[ctr + ky * tw + kx];

    float inv_np = 1.0 / float(np);
    float acc = 0.0, wsum = 0.0;

    /* After the temporal average a sample carries less noise than one
     * noise standard deviation, the static ones a third of it at an
     * alpha_max of 0.8, and a filter that still took two of them for noise
     * in every patch distance would take the detail the average has just
     * uncovered for noise too. So the distance loses the noise the two
     * samples actually carry, and the strength is taken in units of the
     * noise left in the sample being filtered: where nothing was averaged
     * this is exactly the loop below, and where much was the filter
     * reaches only as far into the signal as the remaining noise does.
     * DESIGN.md 14.7. */
    if (history) {
        float vc = craw_tvar[ctr];
        float inv_h2 = pc.inv_h2 / vc;

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

                float w = exp(-max(d2 * inv_np - (vc + craw_tvar[s]), 0.0) * inv_h2);

                wsum += w;
                acc += w * craw_tile[s];
            }
        }

        craw_dn_store(mp, pl, craw_dn_level(mp, pl), craw_dn_igat(acc / wsum));
        return;
    }

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

    craw_dn_store(mp, pl, craw_dn_level(mp, pl), craw_dn_igat(acc / wsum));
}
