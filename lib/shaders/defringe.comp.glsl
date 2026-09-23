/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Defringe, the second pass, DESIGN.md 19.2: at a pixel near a strong edge,
 * the colour clamped to the range the two sides of the edge span, in place
 * in the intermediate. Each invocation reads the first pass's image around
 * its pixel and reads and writes only its own texel of the intermediate,
 * and it writes nothing at all unless it moves the colour, so every pixel
 * the stage leaves alone stays bit for bit what the debayer made.
 *
 * Most pixels are nowhere near an edge that strong and leave after the
 * 5x5 of edge strengths. */

/* GL_GOOGLE_include_directive is a preprocessor-only extension (it emits
 * no SPIR-V capability); the build passes -I for it. */
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"
#include "defringe.glsl"

layout(binding = 0) readonly uniform image2D prep;
layout(binding = 1) uniform image2D inter;

vec4 craw_df_prep(ivec2 q)
{
    return imageLoad(prep, craw_mirror(q, pc.size));
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

void main()
{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, pc.size)))
        return;

    float en = 0.0;
    for (int j = -2; j <= 2; j++)
        for (int i = -2; i <= 2; i++)
            en = max(en, craw_df_prep(p + ivec2(i, j)).a);
    if (en <= CRAW_DF_E_LO)
        return;

    float yx = 0.0;
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++)
            yx = max(yx, craw_df_y(craw_df_prep(p + ivec2(i, j)).r));

    float t = pc.strength * smoothstep(CRAW_DF_E_LO, CRAW_DF_E_HI, en) *
              smoothstep(CRAW_DF_B_LO, CRAW_DF_B_HI, yx);
    if (t <= 0.0)
        return;

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
        return;
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
        return;

    vec2 cn = c + t * (cl - c);
    vec4 px = imageLoad(inter, p);
    float y = (px.r + 2.0 * px.g + px.b) * 0.25;
    vec2 e = exp(cn);
    float gn = 4.0 * y / (e.x + 2.0 + e.y);

    imageStore(inter, p, vec4(gn * e.x, gn, gn * e.y, px.a));
}
