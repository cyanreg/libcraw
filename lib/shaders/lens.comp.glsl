/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* Lens correction, DESIGN.md 16: a per-channel radial resampling warp on
 * the camera RGB intermediate, whose channels are the ones the record's
 * polynomials describe, ahead of the defringe that must see them
 * registered.
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
 * Alpha is written as 1.0, not warped: it is 1.0 throughout the
 * intermediate, which every debayer writes and every stage after it
 * keeps, and the colour stage writes 1.0 over it anyway. Reading it back
 * at the pixel's own position, as the warp used to, was a 140 MB read of
 * its own at 5.8K and a tenth of the pass.
 *
 * The other two modes put the output position through a turn and a scale
 * worked out on the host before the polynomial. TURN fills a frame turned
 * by up to 45 degrees, magnified just enough that every output pixel
 * still has a source, measured against the corrected source's own outline
 * rather than the readout's rectangle (DESIGN.md 16.9). KEEP shrinks it
 * instead, turned or not, until the whole warped source fits inside the
 * frame, and writes black wherever an output pixel's source lies outside
 * the source frame in any channel (16.8). */

#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(binding = 0) readonly  uniform image2D src;   /* the camera RGB intermediate */
layout(binding = 1) writeonly uniform image2D dst;

/* Mirrors CRAWLensPush in lib/pipeline.c. */
layout(push_constant, scalar) uniform CRAWLensPushBuf {
    ivec2 size;
    vec2  centre;          /* optical centre in pixels, pixel centre origin */
    float inv_half_diag;   /* 1 / the corner distance from `centre` */
    vec4  k[3];            /* k0..k3 per channel, R G B */
    vec2  turn;            /* (cos, sin) of the turn over the scale, 16.9 */
    int   mode;            /* CRAW_LENS_WARP_* in lib/lens_warp.h */
} pc;

#define CRAW_LENS_WARP_FILL 0
#define CRAW_LENS_WARP_TURN 1
#define CRAW_LENS_WARP_KEEP 2

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

    /* The record's own frame is written out separately, exactly as it was
     * before the other modes existed. `mode` is a push constant, so the
     * branch is uniform across the dispatch and costs nothing; what it buys
     * is that the compiler sees the same expressions it always saw on the
     * filling path and contracts them into the same multiply-adds. Sharing
     * the code instead moved 1557 samples of a 5760x3040 frame by one unit
     * in the last place of the fp16 output, which is not a difference
     * anyone can see and is exactly the kind of difference a mode switch
     * must not make. */
    if (pc.mode != CRAW_LENS_WARP_FILL) {
        /* DESIGN.md 16.8 and 16.9: the filling mapping evaluated at the
         * output position turned and divided by the scale, plus, keeping
         * everything, a black border where that lands off the source.
         * Unturned, `turn` is (1 / s, 0) and this is d times 1 / s to the
         * bit, which is what keeping everything always computed. */
        vec2  q  = vec2(pc.turn.x * d.x + pc.turn.y * d.y,
                        pc.turn.x * d.y - pc.turn.y * d.x);
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

        /* Black, not transparent, and only when keeping everything. A
         * filled turn is on the source by construction, with half a pixel
         * to spare, since the host measures the frame's edge and the
         * outermost pixel centres sit half a pixel inside it; the mirror
         * takes whatever rounding gets past that, where a test here would
         * paint a black dot in a corner. */
        if (pc.mode == CRAW_LENS_WARP_KEEP && !inside) {
            imageStore(dst, p, vec4(0.0, 0.0, 0.0, 1.0));
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

    o.a = 1.0;

    imageStore(dst, p, o);
}
