/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Capture sharpening, one Richardson-Lucy iteration, DESIGN.md 20.2:
 *
 *     b = K * u                  the estimate blurred by the PSF
 *     q = d / b                  how far that is from what was observed
 *     c = K * q                  the correction, blurred back (K is symmetric)
 *     u' = u * clamp(c, 1 / CORR_MAX, CORR_MAX), at least PEDESTAL
 *
 * with K the separable Gaussian whose taps the host sends, all of it in one
 * dispatch. The workgroup loads the estimate with an apron of 2R and the
 * observed luma with an apron of R (at least 3, for the mask below), blurs
 * the estimate along x and then y on the part of the tile that lies R
 * inside its edge, takes the ratio there, and blurs the ratio along x and y
 * down to its own pixels, all in shared memory. The planes are read once
 * per iteration instead of four times; the redundant blurring of the apron
 * is shared memory arithmetic.
 *
 * The tiles are loaded in whole chunks of eight pixels, and the part of a
 * tile that lies outside the frame is then filled from the positions the
 * mirror of DESIGN.md 3 maps it to, which the same tile always holds for
 * the positions a pixel inside the frame depends on. Since the mirror is a
 * reflection and the Gaussian symmetric, the blur at a mirrored position is
 * the blur at the position it mirrors, so the fused result is the
 * sequential definition at the borders too (the argument of 17.3).
 *
 * The reach R is specialization constant 6, one pipeline per reach, so
 * that every tile size, every loop bound and every divisor below is a
 * constant the compiler can unroll and strength-reduce.
 *
 * Specialization constant 5 makes the last iteration: instead of storing
 * u', it clamps it to the range of the observed luma around the pixel,
 * weighs the change by the local contrast mask and multiplies the camera
 * RGB intermediate's own texel by the resulting luma ratio, in place. A
 * pixel the mask leaves alone is not written at all, so it stays bit for
 * bit what came in. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. */
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"
#include "sharpen.glsl"

layout (constant_id = 5) const bool craw_sh_final = false;
layout (constant_id = 6) const int  craw_sh_r = 3;

layout(binding = 0, std430) readonly buffer CRAWShU { uvec4 craw_u[]; };  /* the estimate */
layout(binding = 1, std430) readonly buffer CRAWShD { uvec4 craw_d[]; };  /* the observed luma */
/* The next estimate; not written by the last iteration. */
layout(binding = 2, std430) writeonly buffer CRAWShO { uvec4 craw_o[]; };
/* The camera RGB intermediate, RGBA16F: only the last iteration reads and
 * writes it, each invocation its own texel. */
layout(binding = 3) uniform image2D craw_rgb;

const int R  = craw_sh_r;
/* The observed luma's apron: R for the ratio, and the mask's statistic
 * needs MASK_R + 1 around each pixel. */
const int RD = R > CRAW_SH_MASK_R + 1 ? R : CRAW_SH_MASK_R + 1;
const int WX = int(gl_WorkGroupSize.x);
const int WY = int(gl_WorkGroupSize.y);
const int NT = WX * WY;
const int UX = WX + 4 * R, UY = WY + 4 * R;     /* the estimate's tile */
const int BX = WX + 2 * R, BY = WY + 2 * R;     /* the first blur's, the ratio's */
const int DY = WY + 2 * RD;                     /* the observed luma's rows */
const int AP = 2 * R > RD ? 2 * R : RD;         /* the wider of the two aprons */

/* The chunks a tile's rows take, the tile's first pixel being `CU0 * 8 -
 * 2R` into the first of them (the workgroup's width is a multiple of
 * eight). */
const int CU0 = (2 * R + 7) / 8;
const int NCU = CU0 + (WX + 2 * R + 7) / 8;
const int LU  = NCU * 8;
const int OU  = CU0 * 8 - 2 * R;
const int CD0 = (RD + 7) / 8;
const int NCD = CD0 + (WX + RD + 7) / 8;
const int LD  = NCD * 8;
const int OD  = CD0 * 8 - RD;

/* The two blurs along x and the first along y take NB outputs to an
 * invocation, from one run of NB + 2R entries of shared memory held in
 * registers: at R = 3 a third of the reads of one output at a time. A run
 * that ends past a tile's last column or row reads the padding below, and
 * what it computes there is never read. */
const int NB  = 4;
const int GH1 = (BX + NB - 1) / NB;             /* the first blur along x */
const int HX  = GH1 * NB;
const int GV1 = (BY + NB - 1) / NB;             /* and along y */
const int GH2 = WX / NB;                        /* the second along x */
const int LUS = LU + NB;

/* The estimate's tile, and after the first blur the ratio's. */
shared float craw_sh_a[UY * LUS > BY * BX ? UY * LUS : BY * BX];
/* The pass along x of each blur. */
shared float craw_sh_h[(UY + NB) * HX > BY * WX ? (UY + NB) * HX : BY * WX];
shared float craw_sh_d[DY * LD];

/* The most chunks one invocation loads: all of them are issued into
 * registers before any is stored, so that the workgroup waits for memory
 * once. */
const int NLU = (UY * NCU + NT - 1) / NT;
const int NLD = (DY * NCD + NT - 1) / NT;

void craw_sh_unpack(uvec4 v, out float o[8])
{
    for (int i = 0; i < 4; i++) {
        vec2 h = unpackHalf2x16(v[i]);
        o[2 * i] = h.x;
        o[2 * i + 1] = h.y;
    }
}

/* NB outputs of the blur, from the run of NB + 2R inputs around them. The
 * centre tap first and then the pairs outwards, which is the order the
 * reference sums in. The run is sized for the widest reach, since a
 * parameter's size cannot be a specialization constant; the entries past
 * NB + 2R are never touched. */
#define CRAW_SH_RUN (4 + 2 * CRAW_SH_RMAX)

void craw_sh_blur(float v[CRAW_SH_RUN], out float o[NB])
{
    for (int j = 0; j < NB; j++) {
        float s = pc.k[0] * v[j + R];
        for (int k = 1; k <= R; k++)
            s += pc.k[k] * (v[j + R - k] + v[j + R + k]);
        o[j] = s;
    }
}

void main()
{
    const ivec2 base = ivec2(gl_WorkGroupID.xy) * ivec2(WX, WY);
    const ivec2 lid  = ivec2(gl_LocalInvocationID.xy);
    const ivec2 p    = base + lid;
    const int lin    = lid.y * WX + lid.x;
    const int cb     = base.x / 8;

    {
        uvec4 vu[NLU], vd[NLD];

        /* Chunks wholly outside the frame are not read; the fill below
         * writes over them. */
        for (int n = 0; n < NLU; n++) {
            int i = lin + n * NT;
            int y = base.y - 2 * R + i / NCU, c = cb - CU0 + i % NCU;
            vu[n] = i < UY * NCU && y >= 0 && y < pc.size.y && c >= 0 && c < pc.stride
                  ? craw_u[y * pc.stride + c] : uvec4(0);
        }
        for (int n = 0; n < NLD; n++) {
            int i = lin + n * NT;
            int y = base.y - RD + i / NCD, c = cb - CD0 + i % NCD;
            vd[n] = i < DY * NCD && y >= 0 && y < pc.size.y && c >= 0 && c < pc.stride
                  ? craw_d[y * pc.stride + c] : uvec4(0);
        }
        for (int n = 0; n < NLU; n++) {
            int i = lin + n * NT;
            if (i < UY * NCU) {
                float o[8];
                craw_sh_unpack(vu[n], o);
                for (int j = 0; j < 8; j++)
                    craw_sh_a[(i / NCU) * LUS + (i % NCU) * 8 + j] = o[j];
            }
        }
        for (int n = 0; n < NLD; n++) {
            int i = lin + n * NT;
            if (i < DY * NCD) {
                float o[8];
                craw_sh_unpack(vd[n], o);
                for (int j = 0; j < 8; j++)
                    craw_sh_d[(i / NCD) * LD + (i % NCD) * 8 + j] = o[j];
            }
        }
    }

    barrier();

    /* The mirror, on the workgroups whose tiles reach past the frame. A
     * position is filled from the one it mirrors to, clamped into the tile:
     * every position a pixel inside the frame depends on mirrors to one
     * inside the frame and inside the tile, and the clamp only keeps the
     * others, which nothing stores, from reading outside it. */
    if (base.x < AP || base.y < AP ||
        base.x + WX + AP > pc.size.x || base.y + WY + AP > pc.size.y) {
        for (int i = lin; i < UY * UX; i += NT) {
            ivec2 t = ivec2(i % UX, i / UX);
            ivec2 q = base - 2 * R + t;
            if (all(greaterThanEqual(q, ivec2(0))) && all(lessThan(q, pc.size)))
                continue;
            ivec2 s = clamp(craw_mirror(q, pc.size) - base + 2 * R, ivec2(0), ivec2(UX, UY) - 1);
            craw_sh_a[t.y * LUS + OU + t.x] = craw_sh_a[s.y * LUS + OU + s.x];
        }
        for (int i = lin; i < DY * (WX + 2 * RD); i += NT) {
            ivec2 t = ivec2(i % (WX + 2 * RD), i / (WX + 2 * RD));
            ivec2 q = base - RD + t;
            if (all(greaterThanEqual(q, ivec2(0))) && all(lessThan(q, pc.size)))
                continue;
            ivec2 s = clamp(craw_mirror(q, pc.size) - base + RD, ivec2(0),
                            ivec2(WX + 2 * RD, DY) - 1);
            craw_sh_d[t.y * LD + OD + t.x] = craw_sh_d[s.y * LD + OD + s.x];
        }
        barrier();
    }

    float u0 = craw_sh_a[(lid.y + 2 * R) * LUS + OU + 2 * R + lid.x];

    /* b along x, on every row of the estimate's tile and on the columns
     * that lie R inside it. */
    for (int i = lin; i < UY * GH1; i += NT) {
        int ty = i / GH1, g = i % GH1;
        float v[CRAW_SH_RUN], o[NB];
        for (int j = 0; j < NB + 2 * R; j++)
            v[j] = craw_sh_a[ty * LUS + OU + g * NB + j];
        craw_sh_blur(v, o);
        for (int j = 0; j < NB; j++)
            craw_sh_h[ty * HX + g * NB + j] = o[j];
    }

    barrier();

    /* b along y and the ratio, on the part of the tile R inside it. The
     * estimate's own tile is not read again, so the ratio takes its
     * place. */
    for (int i = lin; i < GV1 * BX; i += NT) {
        int g = i / BX, tx = i % BX;
        float v[CRAW_SH_RUN], o[NB];
        for (int j = 0; j < NB + 2 * R; j++)
            v[j] = craw_sh_h[(g * NB + j) * HX + tx];
        craw_sh_blur(v, o);
        for (int j = 0; j < NB; j++) {
            int ty = g * NB + j;
            if (ty < BY)
                craw_sh_a[ty * BX + tx] = craw_sh_d[(ty + RD - R) * LD + OD + RD - R + tx] / o[j];
        }
    }

    barrier();

    /* c along x, on the rows of the ratio's tile and the workgroup's own
     * columns. */
    for (int i = lin; i < BY * GH2; i += NT) {
        int ty = i / GH2, g = i % GH2;
        float v[CRAW_SH_RUN], o[NB];
        for (int j = 0; j < NB + 2 * R; j++)
            v[j] = craw_sh_a[ty * BX + g * NB + j];
        craw_sh_blur(v, o);
        for (int j = 0; j < NB; j++)
            craw_sh_h[ty * WX + g * NB + j] = o[j];
    }

    barrier();

    float c;
    {
        int i0 = (lid.y + R) * WX + lid.x;
        c = pc.k[0] * craw_sh_h[i0];
        for (int k = 1; k <= R; k++)
            c += pc.k[k] * (craw_sh_h[i0 - k * WX] + craw_sh_h[i0 + k * WX]);
    }
    float u = max(u0 * clamp(c, 1.0 / CRAW_SH_CORR_MAX, CRAW_SH_CORR_MAX), CRAW_SH_PEDESTAL);

    if (!craw_sh_final) {
        /* Eight pixels to a store, through shared memory: the ratio's tile
         * was last read before the barrier above. */
        craw_sh_a[lin] = u;
        barrier();
        if (lin < NT / 8) {
            int y = base.y + lin / (WX / 8), cx = cb + lin % (WX / 8);
            float v[8];
            for (int j = 0; j < 8; j++)
                v[j] = craw_sh_a[lin * 8 + j];
            if (y < pc.size.y && cx < pc.stride)
                craw_o[y * pc.stride + cx] = craw_sh_pack(v);
        }
        return;
    }

    if (any(greaterThanEqual(p, pc.size)))
        return;

    /* The last iteration. The target luma is held to the range the
     * observed luma spans around the pixel, which is what keeps an
     * overshoot from drawing a halo. */
    int dc = (lid.y + RD) * LD + OD + RD + lid.x;
    float d0 = craw_sh_d[dc], lo = d0, hi = d0;
    for (int j = -CRAW_SH_RANGE_R; j <= CRAW_SH_RANGE_R; j++) {
        for (int i = -CRAW_SH_RANGE_R; i <= CRAW_SH_RANGE_R; i++) {
            float v = craw_sh_d[dc + j * LD + i];
            lo = min(lo, v);
            hi = max(hi, v);
        }
    }

    /* The contrast mask: the mean squared central difference over the
     * window against what the noise model says the same statistic reads
     * on a flat patch at the window's level. */
    float m = 1.0;
    if (pc.contrast > 0.0) {
        const float inv_n = 1.0 / float((2 * CRAW_SH_MASK_R + 1) * (2 * CRAW_SH_MASK_R + 1));
        float e = 0.0, l = 0.0;

        for (int j = -CRAW_SH_MASK_R; j <= CRAW_SH_MASK_R; j++) {
            for (int i = -CRAW_SH_MASK_R; i <= CRAW_SH_MASK_R; i++) {
                int q = dc + j * LD + i;
                float gx = craw_sh_d[q + 1] - craw_sh_d[q - 1];
                float gy = craw_sh_d[q + LD] - craw_sh_d[q - LD];
                e += gx * gx + gy * gy;
                l += craw_sh_d[q];
            }
        }
        e *= 0.25 * inv_n;
        l = l * inv_n - CRAW_SH_PEDESTAL;

        float var = pc.noise_a * max(l, 0.0) + pc.noise_b;
        float z = sqrt(e / max(var, 1e-30));
        m = smoothstep(pc.contrast, 2.0 * pc.contrast, z);
    }

    float rho = 1.0 + m * (clamp(u, lo, hi) / d0 - 1.0);
    if (rho == 1.0)
        return;

    vec4 px = imageLoad(craw_rgb, p);
    imageStore(craw_rgb, p, vec4(px.rgb * rho, px.a));
}
