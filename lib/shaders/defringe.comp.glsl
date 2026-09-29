/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Defringe, DESIGN.md 19.2, in one dispatch: at a pixel near a strong
 * edge, the colour clamped to the range the two sides of the edge span,
 * from the camera RGB intermediate into the second image.
 *
 * The workgroup loads its tile of the intermediate with an apron of nine
 * and works out, on the tile with an apron of eight, what the first pass
 * used to write into an RGBA16F image for the second to read around every
 * pixel:
 *
 *     l = ln(max(Y, YMIN) + Y0), the two log ratios c, E = |Sobel(l)| / 8
 *
 * rounded to fp16 as that image stored them (toward zero on RDNA2, see
 * craw_h()), so that the output is the two passes' bit for bit. Then each
 * invocation treats its pixel from shared memory: the side search reaches
 * six pixels along the normal and the 3x3 means around the sides two more,
 * which is the apron of eight, and the Sobel one more for l. At a position
 * outside the frame the tile holds what the first pass computed at the
 * one inside it that mirrors there (DESIGN.md 3), which is what the second
 * pass read. Every pixel is written, the ones the stage leaves alone as
 * they came in, so the two images swap roles like the other stages' that
 * cannot run in place; the two passes read and wrote the whole frame three
 * times where this reads it once and writes it once.
 *
 * Most pixels are nowhere near an edge that strong and leave after the
 * 5x5 of edge strengths. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. */
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"
#include "defringe.glsl"

layout(binding = 0) readonly  uniform image2D src;   /* the intermediate */
layout(binding = 1) writeonly uniform image2D dst;   /* the second image */

const int WX = int(gl_WorkGroupSize.x);
const int WY = int(gl_WorkGroupSize.y);
const int NT = WX * WY;
const int A  = 8;                       /* the apron of the first pass's values */
const int LX = WX + 2 * A + 2, LY = WY + 2 * A + 2;
const int PX = WX + 2 * A, PY = WY + 2 * A;

shared float craw_l9[LX * LY];          /* l, apron 9, as computed */
shared uint  craw_c8[PX * PY];          /* c, apron 8, fp16 pairs */
shared uvec2 craw_p8[PX * PY];          /* (l, c) and (c, E), apron 8, fp16 */

ivec2 craw_base;                        /* the tile's first pixel */

/* A value as the first pass's RGBA16F image stored it. An image store on
 * RDNA2 rounds toward zero (1 + 7/8 of an fp16 step stores as 1), so that
 * is what this does: the ten bits of mantissa truncated, and below the
 * smallest normal fp16 the step of 2^-24. The values never come near the
 * fp16 range's end. The result is exact in fp16, so packHalf2x16() packs
 * it whatever its own rounding. */
float craw_h(float x)
{
    if (abs(x) < 6.103515625e-5)
        return trunc(x * 16777216.0) * (1.0 / 16777216.0);

    return uintBitsToFloat(floatBitsToUint(x) & 0xFFFFE000u);
}

float craw_df_l(vec3 rgb)
{
    float y = (rgb.r + 2.0 * rgb.g + rgb.b) * 0.25;
    return log(max(y, CRAW_DF_YMIN) + CRAW_DF_Y0);
}

/* The first pass's (l, c, E) at q: the tile's, which at a position outside
 * the frame is the mirrored one's. */
vec4 craw_df_prep(ivec2 q)
{
    uvec2 v = craw_p8[(q.y - craw_base.y + A) * PX + q.x - craw_base.x + A];
    return vec4(unpackHalf2x16(v.x), unpackHalf2x16(v.y));
}

/* The level Y back from l. */
float craw_df_y(float l)
{
    return exp(l) - CRAW_DF_Y0;
}

/* The 3x3 mean of (Y, c) at a position between pixels: bilinear between
 * the means at the four integer positions around it, which is the 4x4
 * around it weighted (1 - f, 1, 1, f) / 3 on each axis. */
vec3 craw_df_side(vec2 pos)
{
    vec2 b = floor(pos);
    vec2 f = pos - b;
    ivec2 o = ivec2(b) - 1;
    float wx[4] = float[4](1.0 - f.x, 1.0, 1.0, f.x);
    float wy[4] = float[4](1.0 - f.y, 1.0, 1.0, f.y);
    vec3 acc = vec3(0.0);

    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            vec4 v = craw_df_prep(o + ivec2(i, j));
            acc += (wx[i] * wy[j]) * vec3(craw_df_y(v.r), v.gb);
        }
    }
    return acc * (1.0 / 9.0);
}

/* The treatment of the pixel p, DESIGN.md 19.2, of colour px: what it
 * becomes, which is px itself unless the clamp moves it. */
vec4 craw_df(ivec2 p, vec4 px)
{
    float en = 0.0;
    for (int j = -2; j <= 2; j++)
        for (int i = -2; i <= 2; i++)
            en = max(en, craw_df_prep(p + ivec2(i, j)).a);
    if (en <= CRAW_DF_E_LO)
        return px;

    float yx = 0.0;
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++)
            yx = max(yx, craw_df_y(craw_df_prep(p + ivec2(i, j)).r));

    float t = pc.strength * smoothstep(CRAW_DF_E_LO, CRAW_DF_E_HI, en) *
              smoothstep(CRAW_DF_B_LO, CRAW_DF_B_HI, yx);
    if (t <= 0.0)
        return px;

    /* The edge normal: l convolved with the 7x7 kernel of rows
     * (1, 3, 4, 4, 4, 3, 1) and columns (-1, -1, 0, 0, 0, 1, 1) for x, the
     * transpose for y: the Sobel gradient boxed over the 5x5. */
    const float kv[7] = float[7](1.0, 3.0, 4.0, 4.0, 4.0, 3.0, 1.0);
    const float kd[7] = float[7](-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0);
    vec2 g = vec2(0.0);
    for (int j = 0; j < 7; j++) {
        for (int i = 0; i < 7; i++) {
            float l = craw_df_prep(p + ivec2(i - 3, j - 3)).r;
            g += l * vec2(kd[i] * kv[j], kv[i] * kd[j]);
        }
    }
    float gl = length(g);
    if (!(gl > 0.0))
        return px;
    vec2 n = g / gl;

    /* Each side where the edge is weakest along the normal, 2 to 6 pixels
     * out, the nearer one on a tie: past the fringe, which must not vote on
     * its own colour, but inside a leaf thinner than that. */
    vec3 s1, s2;
    for (int s = 0; s < 2; s++) {
        float sgn = s == 1 ? -1.0 : 1.0, best = 1.0 / 0.0;
        int bk = CRAW_DF_D_MIN;

        for (int k = CRAW_DF_D_MIN; k <= CRAW_DF_D_MAX; k++) {
            float e = craw_df_prep(ivec2(floor(vec2(p) + sgn * float(k) * n + 0.5))).a;
            if (e < best) {
                best = e;
                bk = k;
            }
        }
        vec3 side = craw_df_side(vec2(p) + sgn * float(bk) * n);
        if (s == 0)
            s1 = side;
        else
            s2 = side;
    }
    float ys = max(s1.x, s2.x);
    bool ok1 = s1.x >= CRAW_DF_DARK * ys;
    bool ok2 = s2.x >= CRAW_DF_DARK * ys;
    /* A side too dark to have a colour counts as neutral, the colour of the
     * light: a grey pixel beside a red bar stays grey, a blue one beside a
     * white lamp is pulled to white. */
    vec2 c1 = ok1 ? s1.yz : vec2(0.0), c2 = ok2 ? s2.yz : vec2(0.0);
    vec2 lo = min(c1, c2), hi = max(c1, c2);

    vec2 c = craw_df_prep(p).gb;
    vec2 cl = clamp(c, lo - CRAW_DF_MARGIN, hi + CRAW_DF_MARGIN);
    if (cl == c)
        return px;

    vec2 cn = c + t * (cl - c);
    float y = (px.r + 2.0 * px.g + px.b) * 0.25;
    vec2 e = exp(cn);
    float gn = 4.0 * y / (e.x + 2.0 + e.y);

    return vec4(gn * e.x, gn, gn * e.y, px.a);
}

void main()
{
    const int lin = int(gl_LocalInvocationIndex);

    craw_base = ivec2(gl_WorkGroupID.xy) * ivec2(WX, WY);

    /* l on the apron of nine, and the rounded c on the eight inside it,
     * each of the intermediate at the mirrored position, as the first
     * pass read it. One reflection reaches every position a pixel inside
     * the frame depends on; the clamp keeps the others inside the image. */
    for (int i = lin; i < LX * LY; i += NT) {
        ivec2 t = ivec2(i % LX, i / LX);
        ivec2 q = clamp(craw_mirror(craw_base - A - 1 + t, pc.size), ivec2(0), pc.size - 1);
        vec3 rgb = imageLoad(src, q).rgb;

        craw_l9[i] = craw_df_l(rgb);
        if (t.x >= 1 && t.y >= 1 && t.x <= PX && t.y <= PY) {
            float g = max(rgb.g, CRAW_DF_YMIN);
            vec2 c = vec2(log(max(rgb.r, CRAW_DF_YMIN) / g), log(max(rgb.b, CRAW_DF_YMIN) / g));
            craw_c8[(t.y - 1) * PX + t.x - 1] = packHalf2x16(vec2(craw_h(c.x), craw_h(c.y)));
        }
    }

    barrier();

    /* The first pass at every position of the apron of eight, computed at
     * the position inside the frame that mirrors to it: the Sobel of its
     * 3x3 of l, the l at its centre and its c. */
    for (int i = lin; i < PX * PY; i += NT) {
        ivec2 q = craw_base - A + ivec2(i % PX, i / PX);
        ivec2 m = clamp(craw_mirror(q, pc.size), craw_base - A,
                        craw_base + ivec2(WX, WY) + A - 1) - craw_base + A + 1;
        float l[9];

        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                l[j * 3 + k] = craw_l9[(m.y + j - 1) * LX + m.x + k - 1];

        float gx = (l[2] + 2.0 * l[5] + l[8]) - (l[0] + 2.0 * l[3] + l[6]);
        float gy = (l[6] + 2.0 * l[7] + l[8]) - (l[0] + 2.0 * l[1] + l[2]);
        float e = length(vec2(gx, gy)) * 0.125;
        vec2 c = unpackHalf2x16(craw_c8[(m.y - 1) * PX + m.x - 1]);

        craw_p8[i] = uvec2(packHalf2x16(vec2(craw_h(l[4]), c.x)),
                           packHalf2x16(vec2(c.y, craw_h(e))));
    }

    barrier();

    ivec2 p = craw_base + ivec2(gl_LocalInvocationID.xy);
    if (any(greaterThanEqual(p, pc.size)))
        return;

    imageStore(dst, p, craw_df(p, imageLoad(src, p)));
}
