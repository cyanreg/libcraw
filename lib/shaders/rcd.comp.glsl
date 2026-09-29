/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* RCD, DESIGN.md 5.3, the three passes in one dispatch. Each workgroup
 * loads the prepped mosaic of its tile with an apron of five pixels into
 * shared memory and works inwards, one ring of apron per step:
 *
 *     v               tile + 5    the prepped samples, mirrored
 *     L               tile + 4    pass A's binomial low-pass
 *     Vg Hg Pg Qg     tile + 3    pass A's directional values
 *     wv wp, G        tile + 2    pass A's weights, and pass B's green
 *     C               tile + 1    pass C's estimate at the R and B sites
 *     R G B           tile        pass C at the G sites, and the output
 *
 * where the four separate passes wrote an RGBA16F image of L and the
 * weights, an R16F green plane and the intermediate twice, and read them
 * back around every pixel: 1 GB of traffic at 5.8K for the 175 MB of the
 * mosaic in and the intermediate out, and a first pass bound by the 49
 * image loads each of its pixels made. What stays is the one read of the
 * mosaic, from the cache for the aprons, and the one write.
 *
 * The borders are those of the passes: every value a step reads at a
 * position outside the frame is the one it computed at the position that
 * mirrors to it (DESIGN.md 3), which the tile always holds, except for
 * the directional values, which pass A computed from the mirrored patch
 * at every offset and so are computed here at whatever position they are
 * wanted. Every sum is the passes' own in their order, and every value a
 * pass stored in an fp16 image is rounded to fp16 here where it was
 * stored, the way the image did, so the output is the four passes' bit
 * for bit. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. */
#extension GL_GOOGLE_include_directive : require

#define CRAW_PREP
#include "common.glsl"

layout(binding = 1) writeonly uniform image2D dst;   /* the intermediate */

const float CRAW_RCD_EPS = 1e-5;

const int WX = int(gl_WorkGroupSize.x);
const int WY = int(gl_WorkGroupSize.y);
const int NT = WX * WY;

/* Every array covers the tile and an apron of its own width, row major. */
shared float craw_v_t[(WX + 10) * (WY + 10)];   /* apron 5 */
shared float craw_l_t[(WX + 8) * (WY + 8)];     /* apron 4 */
shared vec4  craw_d_t[(WX + 6) * (WY + 6)];     /* apron 3 */
shared vec2  craw_w_t[(WX + 4) * (WY + 4)];     /* apron 2: (wv, wp) */
shared float craw_g_t[(WX + 4) * (WY + 4)];     /* apron 2 */
shared float craw_c_t[(WX + 2) * (WY + 2)];     /* apron 1 */

ivec2 craw_base;   /* the tile's first pixel */

/* The index of absolute position q in an array of apron a. */
int craw_at(ivec2 q, int a)
{
    ivec2 t = q - craw_base + a;
    return t.y * (WX + 2 * a) + t.x;
}

/* The position q mirrors to, which is where the passes computed what they
 * read at q: an array holds at a position outside the frame the value of
 * the position inside it that mirrors there, so that every read is
 * direct. Every position a pixel inside the frame depends on mirrors into
 * the array of apron a; the clamp only keeps the rest, on a tile hanging
 * far past a small frame's edge, inside it. */
bool craw_interior;   /* the tile and all of its aprons inside the frame */

ivec2 craw_mir(ivec2 q, int a)
{
    if (craw_interior)
        return q;
    return clamp(craw_mirror(q, pc.size), craw_base - a, craw_base + ivec2(WX, WY) + a - 1);
}

float craw_vv(ivec2 q)
{
    return craw_v_t[craw_at(q, 5)];
}

/* Pass A's binomial low-pass at q. */
float craw_rcd_lp(ivec2 q)
{
    return (4.0 * craw_vv(q) +
            2.0 * (craw_vv(q + ivec2(0, -1)) + craw_vv(q + ivec2(0, 1)) +
                   craw_vv(q + ivec2(-1, 0)) + craw_vv(q + ivec2(1, 0))) +
            craw_vv(q + ivec2(-1, -1)) + craw_vv(q + ivec2(1, -1)) +
            craw_vv(q + ivec2(-1, 1)) + craw_vv(q + ivec2(1, 1))) * (1.0 / 16.0);
}

/* Vg, Hg, Pg, Qg at q. P is NE-SW, Q is NW-SE. */
vec4 craw_rcd_dirs(ivec2 q)
{
    float c2 = 2.0 * craw_vv(q);
    return vec4(abs(craw_vv(q + ivec2( 0, -1)) - craw_vv(q + ivec2( 0,  1))) +
                abs(c2 - craw_vv(q + ivec2( 0, -2)) - craw_vv(q + ivec2( 0,  2))),
                abs(craw_vv(q + ivec2(-1,  0)) - craw_vv(q + ivec2( 1,  0))) +
                abs(c2 - craw_vv(q + ivec2(-2,  0)) - craw_vv(q + ivec2( 2,  0))),
                abs(craw_vv(q + ivec2(-1,  1)) - craw_vv(q + ivec2( 1, -1))) +
                abs(c2 - craw_vv(q + ivec2(-2,  2)) - craw_vv(q + ivec2( 2, -2))),
                abs(craw_vv(q + ivec2(-1, -1)) - craw_vv(q + ivec2( 1,  1))) +
                abs(c2 - craw_vv(q + ivec2(-2, -2)) - craw_vv(q + ivec2( 2,  2))));
}

/* A value as an fp16 image stored it. The passes stored L and the weights
 * in an RGBA16F image, the green in an R16F one and the R and B sites'
 * colour in the intermediate, and read them back from there. An image
 * store on RDNA2 rounds toward zero (measured: 1 + 7/8 of an fp16 step
 * stores as 1), so that is what this does, which keeps the output bit
 * for bit the four passes' on that hardware: the ten bits of mantissa
 * truncated, and below the smallest normal fp16 the step of 2^-24. It is
 * written out because a conversion to fp16 and back is folded away as
 * the identity it is not. The values here never come near the fp16
 * range's end. */
float craw_h(float x)
{
    if (abs(x) < 6.103515625e-5)
        return trunc(x * 16777216.0) * (1.0 / 16777216.0);

    return uintBitsToFloat(floatBitsToUint(x) & 0xFFFFE000u);
}

/* Pass B's neighbour: its own sample scaled by the ratio of the low-pass
 * at the centre to the low-pass at the neighbour's own colour site two
 * pixels out. */
float craw_rcd_lpf(float v, float l0, float l2)
{
    return v * (1.0 + (l0 - l2) / (l0 + l2 + CRAW_RCD_EPS));
}

float craw_ll(ivec2 q)
{
    return craw_l_t[craw_at(q, 4)];
}

float craw_gg(ivec2 q)
{
    return craw_g_t[craw_at(q, 2)];
}

/* Each step walks the positions of its array, the invocations taking
 * consecutive entries; the array's width is a constant, so the division
 * is a multiply. A walk in two dimensions with the workgroup's own shape
 * leaves half the invocations idle on the narrower arrays, and measured
 * 10% slower for it. */
#define CRAW_RCD_WALK(a)                                                      \
    for (int i_ = lin, tx, ty; i_ < (WX + 2 * (a)) * (WY + 2 * (a)) &&        \
         (tx = i_ % (WX + 2 * (a)), ty = i_ / (WX + 2 * (a)), true); i_ += NT)

void main()
{
    const ivec2 lid = ivec2(gl_LocalInvocationID.xy);
    const int lin = int(gl_LocalInvocationIndex);

    craw_base = ivec2(gl_WorkGroupID.xy) * ivec2(WX, WY);
    craw_interior = all(greaterThanEqual(craw_base, ivec2(5))) &&
                    all(lessThanEqual(craw_base + ivec2(WX, WY) + 5, pc.size));

    /* One reflection reaches every position a pixel inside the frame
     * depends on; the clamp keeps the others inside the image. */
    CRAW_RCD_WALK(5) {
        ivec2 q = craw_base - 5 + ivec2(tx, ty);
        craw_v_t[ty * (WX + 10) + tx] = craw_v(clamp(craw_mirror(q, pc.size), ivec2(0),
                                                     pc.size - 1));
    }

    barrier();

    /* Pass A: the low-pass, and the directional values, which pass A took
     * at every offset from the mirrored patch rather than mirrored. */
    CRAW_RCD_WALK(4) {
        ivec2 q = craw_base - 4 + ivec2(tx, ty);
        craw_l_t[ty * (WX + 8) + tx] = craw_h(craw_rcd_lp(craw_mir(q, 4)));
    }
    CRAW_RCD_WALK(3) {
        ivec2 q = craw_base - 3 + ivec2(tx, ty);
        craw_d_t[ty * (WX + 6) + tx] = craw_rcd_dirs(q);
    }

    barrier();

    /* Pass A's weights from the 3x3 box of the directional values, and
     * pass B's green, which needs the weight at its own position only. */
    CRAW_RCD_WALK(2) {
        ivec2 q = craw_mir(craw_base - 2 + ivec2(tx, ty), 2);
        vec4 d = vec4(0.0);

        for (int oy = -1; oy <= 1; oy++)
            for (int ox = -1; ox <= 1; ox++)
                d += craw_d_t[craw_at(q + ivec2(ox, oy), 3)];
        d *= 1.0 / 9.0;

        float wv = craw_h(d.y / (d.y + d.x + CRAW_RCD_EPS));
        float wp = craw_h(d.w / (d.z + d.w + CRAW_RCD_EPS));
        float g;

        if (craw_site_channel(craw_site(q)) == 1) {
            g = craw_vv(q);
        } else {
            float l0 = craw_ll(q);
            float gv = 0.5 * (craw_rcd_lpf(craw_vv(q + ivec2(0, -1)), l0, craw_ll(q + ivec2(0, -2))) +
                              craw_rcd_lpf(craw_vv(q + ivec2(0,  1)), l0, craw_ll(q + ivec2(0,  2))));
            float gh = 0.5 * (craw_rcd_lpf(craw_vv(q + ivec2(-1, 0)), l0, craw_ll(q + ivec2(-2, 0))) +
                              craw_rcd_lpf(craw_vv(q + ivec2( 1, 0)), l0, craw_ll(q + ivec2( 2, 0))));
            g = wv * gv + (1.0 - wv) * gh;
        }

        craw_w_t[ty * (WX + 4) + tx] = vec2(wv, wp);
        craw_g_t[ty * (WX + 4) + tx] = craw_h(g);
    }

    barrier();

    /* Pass C, first half: the colour an R or B site lacks, from the
     * diagonal neighbours' differences to green. */
    CRAW_RCD_WALK(1) {
        ivec2 q = craw_mir(craw_base - 1 + ivec2(tx, ty), 1);

        if (craw_site_channel(craw_site(q)) == 1)
            continue;

        float g0 = craw_gg(q);
        float dsw = craw_vv(q + ivec2(-1,  1)) - craw_gg(q + ivec2(-1,  1));
        float dne = craw_vv(q + ivec2( 1, -1)) - craw_gg(q + ivec2( 1, -1));
        float dnw = craw_vv(q + ivec2(-1, -1)) - craw_gg(q + ivec2(-1, -1));
        float dse = craw_vv(q + ivec2( 1,  1)) - craw_gg(q + ivec2( 1,  1));
        float cp = g0 + 0.5 * (dsw + dne);
        float cq = g0 + 0.5 * (dnw + dse);
        float wp = craw_w_t[craw_at(q, 2)].y;

        craw_c_t[ty * (WX + 2) + tx] = craw_h(wp * cp + (1.0 - wp) * cq);
    }

    barrier();

    ivec2 p = craw_base + lid;
    if (any(greaterThanEqual(p, pc.size)))
        return;

    int c = craw_site_channel(craw_site(p));
    float g0 = craw_gg(p);
    vec3 rgb;
    rgb[1] = g0;

    if (c != 1) {
        rgb[c] = craw_vv(p);
        rgb[2 - c] = craw_c_t[craw_at(p, 1)];
    } else {
        /* Pass C, second half: at a G site both colours, from the four
         * edge neighbours, R or B sites whose full colour the first half
         * made: on one axis they carry the colour natively (D in 5.3),
         * on the other its estimate (E), and wv weights the vertical
         * estimate either way. */
        vec3 nb[4];
        const ivec2 off[4] = ivec2[4](ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));

        for (int k = 0; k < 4; k++) {
            ivec2 q = p + off[k];
            int cn = craw_site_channel(craw_site(q));

            nb[k][1] = craw_gg(q);
            nb[k][cn] = craw_h(craw_vv(q));
            nb[k][2 - cn] = craw_c_t[craw_at(q, 1)];
        }

        float wv = craw_w_t[craw_at(p, 2)].x;

        for (int ch = 0; ch < 3; ch += 2) {
            float chh = g0 + 0.5 * ((nb[0][ch] - nb[0].g) + (nb[1][ch] - nb[1].g));
            float cvv = g0 + 0.5 * ((nb[2][ch] - nb[2].g) + (nb[3][ch] - nb[3].g));
            rgb[ch] = wv * cvv + (1.0 - wv) * chh;
        }
    }

    imageStore(dst, p, vec4(rgb, 1.0));
}
