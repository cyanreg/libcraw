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
 * The workgroup is a column of subgroups, each a band of M rows: a lane
 * owns one column of its band and filters M samples down it. For one
 * search offset the squared differences of the lane's column against the
 * offset one are formed once per row, summed along x with the neighbouring
 * lanes' through subgroupShuffleUp/Down and along y in registers, which
 * gives every patch distance of the band for 1 + 2P shuffles and 2P adds a
 * row instead of (2P + 1)^2 shared memory reads a sample; and the offset
 * column is read from the tile once for all 2S + 1 offsets that share its
 * dx. The P lanes at either end of a subgroup lack a neighbour on one side
 * and produce no output, so a band is L - 2P columns wide for L lanes.
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
 * no SPIR-V capability); the build passes -I for it. The subgroup
 * extensions are the basic and relative shuffle operations lib/craw.c
 * checks for, and the loop attribute unrolls the search's inner loop,
 * whose registers it indexes. */
#extension GL_GOOGLE_include_directive : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require
#extension GL_EXT_control_flow_attributes : require

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
 * distance is then the plain difference of two stabilised samples. The
 * subgroup size the host required, L, and the rows a lane filters, M. */
layout (constant_id = 2) const int craw_dn_search = 3;
layout (constant_id = 3) const int craw_dn_patch  = 1;
layout (constant_id = 7) const int craw_dn_lanes  = 32;
layout (constant_id = 8) const int craw_dn_rows   = 8;

/* The most the host asks for: a search radius of 3, a patch radius of 2,
 * 16 rows a lane. The register arrays are sized by these, the loops by the
 * constants above. */
#define CRAW_DN_SEARCH_MAX 3
#define CRAW_DN_PATCH_MAX  2
#define CRAW_DN_ROWS_MAX   16

const int S  = craw_dn_search;
const int P  = craw_dn_patch;
const int M  = craw_dn_rows;
const int OX = craw_dn_lanes - 2 * P;                 /* a tile's columns */
const int OY = int(gl_WorkGroupSize.y) * M;          /* and rows */
const int TX = OX + 2 * (S + P);                     /* with the halo */
const int TY = OY + 2 * (S + P);

/* The tile and, with a history, the noise variance left in each of its
 * samples. The temporal pass compares 3x3 patches of this frame and the
 * last, so both of those are loaded one sample further out than the tile. */
shared float craw_tile[TX * TY];
shared float craw_tvar[TX * TY];
shared float craw_tcur[(TX + 2) * (TY + 2)];
shared float craw_tprev[(TX + 2) * (TY + 2)];

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

/* The search of DESIGN.md 14.3 for the M samples of the lane's column of
 * its band, which starts at tile position (tx, ty0): res[m] is the
 * weighted mean of the stabilised values around sample m. With a history
 * the patch distances lose the noise the two samples still carry and the
 * strength is taken in units of the centre's (14.7). */
void craw_dn_search_band(bool history, int tx, int ty0, out float res[CRAW_DN_ROWS_MAX])
{
    const float inv_np = 1.0 / float((2 * P + 1) * (2 * P + 1));
    float own[CRAW_DN_ROWS_MAX + 2 * CRAW_DN_PATCH_MAX];
    float acc[CRAW_DN_ROWS_MAX], wsum[CRAW_DN_ROWS_MAX];

    /* The weight exp(-max(d2 / np - thr, 0) / h^2) is evaluated as
     * exp2(min(d2 * c1 + c0, 0)), with k = -log2(e) / h^2 folded into
     * c1 = k / np and c0 = -thr k once per sample rather than per
     * neighbour: the search is bound by its ALU, and those are two
     * multiplies of the seven instructions a neighbour costs. With a
     * history h^2 and thr depend on the samples, so c1 is per sample and
     * c0 = -(vc + v) k becomes kv * v + ck. */
    float c1[CRAW_DN_ROWS_MAX], ck[CRAW_DN_ROWS_MAX], kv[CRAW_DN_ROWS_MAX];
    const float k0 = -1.4426950408889634 * pc.inv_h2;

    /* The lane's own column over the band's patches. */
    for (int r = 0; r < M + 2 * P; r++)
        own[r] = craw_tile[(ty0 - P + r) * TX + tx];
    for (int m = 0; m < M; m++) {
        acc[m] = 0.0;
        wsum[m] = 0.0;
        if (history) {
            float vc = craw_tvar[(ty0 + m) * TX + tx];
            float k = k0 / vc;

            c1[m] = k * inv_np;
            kv[m] = -k;
            ck[m] = -vc * k;
        }
    }

    for (int dx = -S; dx <= S; dx++) {
        /* The column dx over, over every row an offset of this dx reads:
         * tc[r] is row ty0 - P - S + r. */
        float tc[CRAW_DN_ROWS_MAX + 2 * (CRAW_DN_PATCH_MAX + CRAW_DN_SEARCH_MAX)];
        float tv[CRAW_DN_ROWS_MAX + 2 * CRAW_DN_SEARCH_MAX];

        for (int r = 0; r < M + 2 * (P + S); r++)
            tc[r] = craw_tile[(ty0 - P - S + r) * TX + tx + dx];
        if (history) {
            for (int r = 0; r < M + 2 * S; r++)
                tv[r] = craw_tvar[(ty0 - S + r) * TX + tx + dx];
        }

        [[unroll]] for (int dy = -S; dy <= S; dy++) {
            /* The squared differences of a row of the patches, summed
             * along x across the lanes. */
            float hs[CRAW_DN_ROWS_MAX + 2 * CRAW_DN_PATCH_MAX];

            for (int r = 0; r < M + 2 * P; r++) {
                float e = own[r] - tc[r + S + dy];
                float e2 = e * e;
                float h = e2;

                /* Left to right, as the reference sums a row. */
                for (int k = 1; k <= P; k++)
                    h = subgroupShuffleUp(e2, k) + h + subgroupShuffleDown(e2, k);
                hs[r] = h;
            }

            /* And along y, into the patch distance of each sample. */
            for (int m = 0; m < M; m++) {
                float d2 = hs[m];
                for (int j = 1; j <= 2 * P; j++)
                    d2 += hs[m + j];

                float ts = tc[m + P + S + dy];
                float w;

                /* Two stabilised samples of the same true value differ by
                 * a variance of 2, so that much of the distance is noise
                 * and is subtracted before it can count against a
                 * neighbour. The centre itself always lands on weight 1, so
                 * wsum >= 1. */
                if (history)
                    w = exp2(min(d2 * c1[m] + (kv[m] * tv[m + S + dy] + ck[m]), 0.0));
                else
                    w = exp2(min(d2 * (k0 * inv_np) - 2.0 * k0, 0.0));

                wsum[m] += w;
                acc[m] += w * ts;
            }
        }
    }

    for (int m = 0; m < M; m++)
        res[m] = acc[m] / wsum[m];
}

/* Fills the tile of the plane `pl` whose first sample sits at `org`,
 * with a halo of `rad`: the stabilised plane, halo included, already
 * averaged with the previous frame when the temporal pass has one, since
 * the spatial filter works on what the temporal one left. Borders are
 * mirrored in the plane's own coordinates (DESIGN.md 14.3), which is what
 * the neighbours' own tiles would have loaded, and the temporal value of a
 * sample is a function of the 3x3 around it in the two frames, mirrored
 * the same way, so every tile that holds it holds the same number. `rad`
 * is always a literal at the call, so the tile's sides are constants and
 * the divisions that walk it are multiplies. */
void craw_dn_fill(bool history, const int rad, ivec2 org, int pl)
{
    const ivec2 tsz = ivec2(OX, OY) + 2 * rad;
    const ivec2 esz = tsz + 2;
    const int nt = int(gl_WorkGroupSize.x * gl_WorkGroupSize.y);
    int lin = int(gl_LocalInvocationIndex);
    ivec2 ph = ivec2(pl & 1, pl >> 1);

    /* The walks are unrolled to their constant trip counts, so that every
     * load of one is in flight before the first is waited for. */
    if (!history) {
        [[unroll]] for (int k = 0; k < (tsz.x * tsz.y + nt - 1) / nt; k++) {
            int i = lin + k * nt;
            ivec2 t = ivec2(i % tsz.x, i / tsz.x);
            ivec2 q = craw_mirror(org + t, pc.size);
            if (i < tsz.x * tsz.y)
                craw_tile[t.y * TX + t.x] = craw_dn_gat(craw_dn_level(2 * q + ph, pl));
        }
        return;
    }

    [[unroll]] for (int k = 0; k < (esz.x * esz.y + nt - 1) / nt; k++) {
        int i = lin + k * nt;
        ivec2 t = ivec2(i % esz.x, i / esz.x);
        ivec2 q = craw_mirror(org - 1 + t, pc.size);
        ivec2 mq = 2 * q + ph;

        if (i < esz.x * esz.y) {
            craw_tcur[i] = craw_dn_gat(craw_dn_level(mq, pl));
            craw_tprev[i] = craw_dn_gat(imageLoad(craw_state_prev, mq).x);
        }
    }

    barrier();

    /* The variance a sample is left with follows from the weight: this
     * frame brings a variance of 1 and last frame's state what it carried,
     * which the state keeps as its square root. */
    [[unroll]] for (int k = 0; k < (tsz.x * tsz.y + nt - 1) / nt; k++) {
        int i = lin + k * nt;
        if (i >= tsz.x * tsz.y)
            break;
        ivec2 t = ivec2(i % tsz.x, i / tsz.x);
        int c = (t.y + 1) * esz.x + t.x + 1;
        ivec2 q = craw_mirror(org + t, pc.size);
        float sp = imageLoad(craw_var_prev, 2 * q + ph).x;
        float a = craw_dn_alpha(c, esz.x);

        craw_tile[t.y * TX + t.x] = mix(craw_tcur[c], craw_tprev[c], a);
        craw_tvar[t.y * TX + t.x] = max((1.0 - a) * (1.0 - a) + a * a * sp * sp,
                                        CRAW_DN_VAR_MIN);
    }
}

void main()
{
    int pl = int(gl_WorkGroupID.x) & 3;
    int px = pl & 1, py = pl >> 1;
    ivec2 wid = ivec2(int(gl_WorkGroupID.x) >> 2, int(gl_WorkGroupID.y));
    bool spatial = (pc.mode & CRAW_DN_SPATIAL) != 0;
    bool history = (pc.mode & CRAW_DN_HISTORY) != 0;
    /* The spatial filter's halo; the temporal average alone needs none. */
    int rad = spatial ? S + P : 0;

    if (spatial) {
        if (history)
            craw_dn_fill(true, S + P, wid * ivec2(OX, OY) - (S + P), pl);
        else
            craw_dn_fill(false, S + P, wid * ivec2(OX, OY) - (S + P), pl);
    } else {
        if (history)
            craw_dn_fill(true, 0, wid * ivec2(OX, OY), pl);
        else
            craw_dn_fill(false, 0, wid * ivec2(OX, OY), pl);
    }

    barrier();

    /* The lane's column of the tile and the band's first row, and the
     * samples of the plane they are. The P lanes at either end filter
     * nothing, but have to take part in the shuffles of their neighbours. */
    int lane = int(gl_SubgroupInvocationID);
    int tx = rad + lane - P;
    int ty0 = rad + int(gl_SubgroupID) * M;
    ivec2 p0 = wid * ivec2(OX, OY) + ivec2(lane - P, int(gl_SubgroupID) * M);
    bool owns = lane >= P && lane < craw_dn_lanes - P && p0.x < pc.size.x;

    float res[CRAW_DN_ROWS_MAX];
    if (spatial) {
        if (history)
            craw_dn_search_band(true, tx, ty0, res);
        else
            craw_dn_search_band(false, tx, ty0, res);
    }

    if (!owns)
        return;

    /* The samples as they came in, for the guard, all loaded before any
     * store: the compiler cannot tell the images apart and would otherwise
     * wait for each load in turn. Rows past the plane read a mirrored row
     * and store nothing. */
    float u0[CRAW_DN_ROWS_MAX];
    for (int m = 0; m < M; m++)
        u0[m] = craw_dn_level(2 * craw_mirror(p0 + ivec2(0, m), pc.size) + ivec2(px, py), pl);

    for (int m = 0; m < M; m++) {
        ivec2 p = p0 + ivec2(0, m);
        int ctr = (ty0 + m) * TX + tx;
        ivec2 mp = 2 * p + ivec2(px, py);

        if (p.y >= pc.size.y)
            break;

        /* The state is what the temporal pass made of this sample, before
         * the spatial filter: fed back through it, a static scene would be
         * filtered again every frame and smoothed further without end. It
         * goes to the other image of the pair (DESIGN.md 14.7): written
         * over the one being read, a neighbouring tile that holds this
         * sample in its halo could read it back already averaged once. */
        if ((pc.mode & CRAW_DN_TEMPORAL) != 0) {
            imageStore(craw_state_next, mp, vec4(craw_dn_igat(craw_tile[ctr])));
            imageStore(craw_var_next, mp, vec4(history ? sqrt(craw_tvar[ctr]) : 1.0));
        }

        /* The guard compares against the sample as it came in, not against
         * its temporal average. */
        craw_dn_store(mp, pl, u0[m], craw_dn_igat(spatial ? res[m] : craw_tile[ctr]));
    }
}
