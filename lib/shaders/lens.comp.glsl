/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Lens correction, DESIGN.md 16: a per-channel radial resampling warp on
 * the linear output of the colour stage.
 *
 * Destination to source. For each output pixel the normalized radius from
 * the optical centre is computed once, each channel's polynomial is
 * evaluated at it, and the source is sampled on the same ray at
 * r * poly(r) with a 16 tap Catmull-Rom bicubic. The three channels differ
 * by up to a couple of pixels at the corner, which is the lens's lateral
 * chromatic aberration; near the centre they coincide to within a
 * thousandth of a pixel and the warp is within a few percent of the
 * identity everywhere, which is why the interpolator has to be good: a
 * bilinear resample at a displacement of a fraction of a pixel is a low
 * pass filter applied to the whole frame, and it would undo exactly the
 * detail RCD was chosen for.
 *
 * Alpha is read straight from the output pixel's own position, not warped:
 * the colour stage writes 1.0 and it stays exactly 1.0.
 *
 * With `keep_all` the frame is not filled: the output position is divided
 * by a scale worked out on the host so that the whole warped source fits
 * inside the frame, and an output pixel whose source lies outside the
 * source frame in any channel is written black. See DESIGN.md 16.8. */

#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(binding = 0) readonly  uniform image2D src;   /* the colour stage's output */
layout(binding = 1) writeonly uniform image2D dst;

/* Mirrors CRAWLensPush in lib/pipeline.c. */
layout(push_constant, scalar) uniform CRAWLensPushBuf {
    ivec2 size;
    vec2  centre;          /* optical centre in pixels, pixel centre origin */
    float inv_half_diag;   /* 1 / the corner distance from `centre` */
    vec4  k[3];            /* k0..k3 per channel, R G B */
    float inv_scale;       /* 1 / s of DESIGN.md 16.8; exactly 1.0 to fill */
    int   keep_all;        /* 1: black outside the mapped source */
} pc;

/* Catmull-Rom cubic weights for taps at -1, 0, +1, +2 around a sample
 * whose fractional position is `t` in [0, 1). The Mitchell-Netravali
 * family at B = 0, C = 1/2: it interpolates, it is C1, and its weights sum
 * to one for every t, so a flat field comes through flat. */
vec4 craw_catrom(float t)
{
    float t2 = t * t;
    float t3 = t2 * t;

    return vec4(-0.5 * t3 + 1.0 * t2 - 0.5 * t,
                 1.5 * t3 - 2.5 * t2           + 1.0,
                -1.5 * t3 + 2.0 * t2 + 0.5 * t,
                 0.5 * t3 - 0.5 * t2);
}

/* One channel of the source at a continuous position, in the same pixel
 * index space as `dst`: integer p is the centre of pixel p. Borders follow
 * the mirror of DESIGN.md 3. The warp keeps the source inside the frame on
 * every record seen, so only the outer ring of the 4x4 footprint ever
 * reaches past an edge. */
float craw_lens_tap(vec2 s, int c)
{
    ivec2 b = ivec2(floor(s));
    vec2  f = s - vec2(b);
    vec4  wx = craw_catrom(f.x);
    vec4  wy = craw_catrom(f.y);
    float acc = 0.0;

    for (int j = 0; j < 4; j++) {
        float row = 0.0;

        for (int i = 0; i < 4; i++) {
            ivec2 q = craw_mirror(ivec2(b.x + i - 1, b.y + j - 1), pc.size);
            row += wx[i] * imageLoad(src, q)[c];
        }

        acc += wy[j] * row;
    }

    return acc;
}

void main()
{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, pc.size)))
        return;

    /* Pixel centres: integer p sits at p + 0.5 in the continuous frame, so
     * a centre of (0.5, 0.5) of the frame is size * 0.5 and the geometry
     * is symmetric about it. */
    vec2 d = vec2(p) + 0.5 - pc.centre;
    vec4 o;

    /* The two modes are written out separately, and the filling one is
     * written exactly as it was before the other existed. `keep_all` is a
     * push constant, so the branch is uniform across the dispatch and costs
     * nothing; what it buys is that the compiler sees the same expressions
     * it always saw on the filling path and contracts them into the same
     * multiply-adds. Sharing the code instead moved 1557 samples of a
     * 5760x3040 frame by one unit in the last place of the fp16 output,
     * which is not a difference anyone can see and is exactly the kind of
     * difference a mode switch must not make. */
    if (pc.keep_all != 0) {
        /* DESIGN.md 16.8: the whole mode is the filling mapping evaluated
         * at p / s, plus a black border where that lands off the source. */
        vec2  q  = d * pc.inv_scale;
        float r2 = dot(q, q) * (pc.inv_half_diag * pc.inv_half_diag);
        vec2  sp[3];
        bool  inside = true;

        for (int c = 0; c < 3; c++) {
            vec4  k = pc.k[c];
            float poly = k.x + r2 * (k.y + r2 * (k.z + r2 * k.w));

            sp[c] = pc.centre + q * poly;

            /* One boundary for all three channels - the pixel is black
             * unless every channel has something to read - so the border is
             * one shape and not three, which is what would show as a
             * coloured fringe along it. Half a dozen comparisons against 49
             * image loads. */
            inside = inside && all(greaterThanEqual(sp[c], vec2(0.0))) &&
                               all(lessThanEqual(sp[c], vec2(pc.size)));
        }

        /* Black, not transparent: alpha is the output pixel's own, which
         * the colour stage wrote as 1.0. */
        if (!inside) {
            imageStore(dst, p, vec4(0.0, 0.0, 0.0, imageLoad(src, p).a));
            return;
        }

        /* Back from the continuous frame into pixel index space. */
        for (int c = 0; c < 3; c++)
            o[c] = craw_lens_tap(sp[c] - 0.5, c);
    } else {
        float r2 = dot(d, d) * (pc.inv_half_diag * pc.inv_half_diag);

        for (int c = 0; c < 3; c++) {
            vec4  k = pc.k[c];
            float poly = k.x + r2 * (k.y + r2 * (k.z + r2 * k.w));

            /* Back from the continuous frame into pixel index space. */
            o[c] = craw_lens_tap(pc.centre + d * poly - 0.5, c);
        }
    }

    o.a = imageLoad(src, p).a;

    imageStore(dst, p, o);
}
