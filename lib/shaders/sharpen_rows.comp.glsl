/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Capture sharpening, one Richardson-Lucy iteration other than the last,
 * DESIGN.md 20.2 and 20.3:
 *
 *     b = K * u                  the estimate blurred by the PSF
 *     q = d / b                  how far that is from what was observed
 *     c = K * q                  the correction, blurred back
 *     u' = u * clamp(c, 1 / CORR_MAX, CORR_MAX), at least PEDESTAL
 *
 * sharpen.comp.glsl computes the same thing on shared memory tiles; this
 * one streams it through registers with subgroup operations. A subgroup
 * owns a vertical strip of the planes and walks down it a row at a time,
 * each lane holding N adjacent columns of the row:
 *
 *  - along x, a blur takes the R columns it needs on either side of the
 *    lane's own from the neighbouring lanes with subgroupShuffleUp/Down,
 *    which in wave32 on RDNA is a ds_swizzle each: the lanes' crossbar,
 *    not shared memory, and no barrier;
 *  - along y, the last 2R + 1 rows of the x blur stay in registers, a
 *    ring that shifts by one row per step, so the y blur of a row is ready
 *    as soon as the row R below it has been blurred along x.
 *
 * Row y of u enters; b and q come out for row y - R, and c and u' for row
 * y - 2R. The strip's first 4R rows only fill the two rings. The strip is
 * 2R columns wider than its output on either side, R for each blur, and
 * those columns are computed and thrown away, as the tiles' aprons are.
 *
 * Every sum is the one sharpen.comp.glsl does, in the same order (the
 * centre tap, then the pairs outwards, x before y), and the stored value
 * goes through the same rounding, so the two produce the same bits. At the
 * frame's edges the loads mirror (DESIGN.md 3); the blurs are symmetric,
 * so the blur at a mirrored position is the blur at the one it mirrors,
 * bit for bit, since a + b is b + a in any rounding.
 *
 * The reach R is specialization constant 6, the subgroup size the host
 * required 7, and the rows a strip outputs 8. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. The two subgroup
 * extensions are the basic and the relative shuffle operations of Vulkan
 * 1.1, which lib/craw.c checks for. */
#extension GL_GOOGLE_include_directive : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require
#extension GL_EXT_control_flow_attributes : require

#include "common.glsl"
#include "sharpen.glsl"

layout (constant_id = 6) const int craw_sh_r     = 3;
layout (constant_id = 7) const int craw_sh_lanes = 32;
layout (constant_id = 8) const int craw_sh_rows  = 64;

/* The planes of sharpen.glsl, read and written two pixels to a word. */
layout(binding = 0, std430) readonly  buffer CRAWShU { uint craw_u[]; };  /* the estimate */
layout(binding = 1, std430) readonly  buffer CRAWShD { uint craw_d[]; };  /* the observed luma */
layout(binding = 2, std430) writeonly buffer CRAWShO { uint craw_o[]; };  /* the next estimate */

const int R  = craw_sh_r;
const int N  = 4;                   /* columns a lane holds */
const int SW = craw_sh_lanes * N;   /* the strip, aprons included */
const int OW = SW - 4 * R;          /* its output columns */
const int P  = 2 * R + 1;           /* the rings' depth */

/* One word of the observed luma (`d`) or of the estimate. */
uint craw_sh_word(bool d, int i)
{
    return d ? craw_d[i] : craw_u[i];
}

/* Row y of a plane, mirrored, at the lane's N columns from c0. The strip
 * is `inside` when all of it lies inside the frame, which is uniform
 * across the subgroup; otherwise every column mirrors on its own, and one
 * past the second reflection (only the last strip of a frame narrower
 * than the strip has any, and nothing stored depends on them) is clamped
 * to the frame rather than read out of bounds. */
void craw_sh_load(bool d, int y, int c0, bool inside, out float v[N])
{
    int b = clamp(craw_mirror(ivec2(0, y), pc.size).y, 0, pc.size.y - 1) * pc.stride * 4;

    if (inside) {
        vec2 h0 = unpackHalf2x16(craw_sh_word(d, b + (c0 >> 1)));
        vec2 h1 = unpackHalf2x16(craw_sh_word(d, b + (c0 >> 1) + 1));
        v = float[N](h0.x, h0.y, h1.x, h1.y);
        return;
    }

    for (int j = 0; j < N; j++) {
        int c = clamp(craw_mirror(ivec2(c0 + j, 0), pc.size).x, 0, pc.size.x - 1);
        v[j] = unpackHalf2x16(craw_sh_word(d, b + (c >> 1)))[c & 1];
    }
}

/* The blur along x of the lane's N columns. Column -m of the lane is
 * column N * dl - m of the lane dl = ceil(m / N) to the left, and column
 * N - 1 + m the one (N - 1 + m) % N of the lane (N - 1 + m) / N to the
 * right; lanes at the strip's ends get undefined values there, which only
 * reach the columns that are thrown away. */
void craw_sh_hblur(float v[N], out float o[N])
{
    float e[N + 2 * CRAW_SH_RMAX];

    for (int j = 0; j < N; j++)
        e[R + j] = v[j];
    for (int m = 1; m <= R; m++) {
        const int dl = (m + N - 1) / N;
        const int dr = (N - 1 + m) / N;
        e[R - m] = subgroupShuffleUp(v[N * dl - m], dl);
        e[R + N - 1 + m] = subgroupShuffleDown(v[N - 1 + m - N * dr], dr);
    }

    for (int j = 0; j < N; j++) {
        float s = pc.k[0] * e[j + R];
        for (int k = 1; k <= R; k++)
            s += pc.k[k] * (e[j + R - k] + e[j + R + k]);
        o[j] = s;
    }
}

void main()
{
    const int lane = int(gl_SubgroupInvocationID);
    const int x0 = int(gl_WorkGroupID.x) * OW;       /* the first output column */
    const int y0 = int(gl_WorkGroupID.y) * craw_sh_rows;
    const int c0 = x0 - 2 * R + N * lane;            /* the lane's first column */
    const bool inside = x0 - 2 * R >= 0 && x0 - 2 * R + SW <= pc.size.x;

    /* The rings: the rows of u, of its x blur and of the x blur of q, row
     * t of the strip in slot t % P. The loop over a ring's P slots is
     * unrolled, so every slot is a register and nothing moves as the
     * strip advances. */
    float ua[P][N], hb[P][N], hc[P][N];
    const int T = craw_sh_rows + 4 * R;

    for (int t0 = 0; t0 < T; t0 += P) {
        [[unroll]] for (int ph = 0; ph < P; ph++) {
            int t = t0 + ph;
            int y = y0 - 2 * R + t;
            float dv[N], q[N], un[N];

            /* No break or continue in here: the loop has to unroll. */
            if (t >= T)
                t = -1;

            if (t >= 0) {
                craw_sh_load(false, y, c0, inside, ua[ph]);
                craw_sh_hblur(ua[ph], hb[ph]);
            }

            /* b, and the ratio, at row y - R, whose slot is (ph - R) % P. */
            if (t >= 2 * R) {
                craw_sh_load(true, y - R, c0, inside, dv);
                for (int j = 0; j < N; j++) {
                    float s = pc.k[0] * hb[(ph + P - R) % P][j];
                    for (int k = 1; k <= R; k++)
                        s += pc.k[k] * (hb[(ph + P - R - k) % P][j] + hb[(ph + P - R + k) % P][j]);
                    q[j] = dv[j] / s;
                }
                craw_sh_hblur(q, hc[(ph + P - R) % P]);
            }

            /* c, and the next estimate, at row y - 2R, in slot (ph + 1) % P
             * of u and of the x blur of q. */
            if (t >= 4 * R && y - 2 * R < pc.size.y) {
                int yo = y - 2 * R;

                for (int j = 0; j < N; j++) {
                    float c = pc.k[0] * hc[(ph + 1) % P][j];
                    for (int k = 1; k <= R; k++)
                        c += pc.k[k] * (hc[(ph + 1 + P - k) % P][j] + hc[(ph + 1 + k) % P][j]);
                    un[j] = max(ua[(ph + 1) % P][j] * clamp(c, 1.0 / CRAW_SH_CORR_MAX, CRAW_SH_CORR_MAX),
                                CRAW_SH_PEDESTAL);
                }

                /* Two pixels to a word, only where the strip owns the
                 * output. The strip's bounds and the frame's width are
                 * even, so a word is either wholly inside them or wholly
                 * outside. */
                int b = yo * pc.stride * 4;
                for (int m = 0; m < N / 2; m++) {
                    int c = c0 + 2 * m;
                    if (c >= x0 && c < x0 + OW && c < pc.size.x)
                        craw_o[b + (c >> 1)] = craw_sh_half(un[2 * m]) | (craw_sh_half(un[2 * m + 1]) << 16);
                }
            }
        }
    }
}
