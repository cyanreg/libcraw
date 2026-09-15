/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#ifndef CRAW_VIEW_ADJUST_H
#define CRAW_VIEW_ADJUST_H

#include <stdbool.h>

#include <libplacebo/gpu.h>
#include <libplacebo/shaders/custom.h>
#include <libplacebo/shaders/lut.h>

/* The display-referred adjustments: the film simulation LUT, vibrance and
 * soft light, all three as hooks at PL_HOOK_PRE_OUTPUT.
 *
 *
 * 1. Why they are hooks, and why they are in this order
 * -----------------------------------------------------
 * RawTherapee runs film simulation in its RGB stage, vibrance in the Lab
 * stage after it, and soft light late in that same Lab stage. The order
 * matters - vibrance on a LUT'd picture is not the same picture as a LUT on
 * a vibrant one - so this file reproduces it: LUT, vibrance, soft light,
 * and then the film grain hook that already existed, which is last because
 * grain is a property of the recording medium and everything above it is a
 * property of the grade.
 *
 * libplacebo applies `pl_frame.lut` on the *target* frame at the very end of
 * pass_output_target(), after PL_HOOK_PRE_OUTPUT has already run, so a LUT
 * left there could never have anything in front of it. Moving it into the
 * first of these hooks puts all four in one ordered list.
 *
 * Nothing is lost by the move. Between the PRE_OUTPUT hook point and the
 * place the target LUT used to be applied, libplacebo runs only
 * pl_shader_encode_color(), and for an RGB target at full range whose
 * sample and colour depths agree - which is what both the window and the
 * export are - that function is a no-op by its own `skip` test. The hook
 * therefore holds exactly the values the target LUT used to be handed, and
 * calling pl_shader_custom_lut() on them is the same arithmetic on the same
 * numbers through the same tetrahedral interpolation.
 *
 *
 * 2. Why they are PL_HOOK_SIG_COLOR
 * ----------------------------------
 * All three are pure per-pixel functions: no neighbour is read, no
 * resolution changes. A PL_HOOK_SIG_TEX hook - what the grain needs, since
 * it does read a neighbourhood - would force the renderer to finish the
 * shader it is building into an intermediate texture and start another, one
 * round trip through an FBO per hook. PL_HOOK_SIG_COLOR instead hands over
 * the shader itself, so all three append into the output pass and cost
 * nothing but their own instructions. That is also what keeps the LUT's
 * arithmetic identical to what it was on the target frame: same shader,
 * same floats, no intermediate storage to quantise through.
 *
 *
 * 3. Where in colour these operate
 * ---------------------------------
 * At PL_HOOK_PRE_OUTPUT the picture has been tone mapped, gamut mapped and
 * encoded for the target, so it is display-encoded RGB in [0,1]: sRGB for
 * the window and for the export. That is the same kind of signal
 * RawTherapee's soft light works on - gamma-encoded RGB - so soft light
 * runs on it directly, untouched.
 *
 * Vibrance needs a perceptual space, so it linearizes through libplacebo's
 * own transfer function shader, works in Oklab, and re-encodes. See the
 * vibrance section below for why Oklab and not Lab.
 */

/* ----------------------------------------------------------- soft light */

/* RawTherapee's soft light is the Pegtop blend of the picture with itself,
 * mixed against the original by the strength.
 *
 * The Pegtop formula for a backdrop a and a blend layer b is
 *
 *     S(a, b) = 2*a*b + a^2 * (1 - 2*b)
 *
 * which is the continuous approximation to the W3C/SVG soft-light
 * compositing operator: same endpoints, same behaviour, but without the
 * piecewise branch at b = 0.5 whose derivative is discontinuous. Pegtop
 * published it for exactly that reason and it is what RawTherapee uses.
 *
 * Blending the picture with itself sets b = a, so the whole operator
 * collapses to one cubic in a single variable:
 *
 *     S(a, a) = 2*a^2 + a^2 * (1 - 2*a) = 3*a^2 - 2*a^3
 *
 * which is the smoothstep polynomial. It fixes 0 and 1, has zero slope at
 * both ends, and crosses the identity only at 0, 0.5 and 1: an S-curve that
 * darkens below mid grey and lightens above it. Applied per channel it
 * raises contrast and, because the channels move by different amounts, also
 * saturation - which is what the control is for.
 *
 * Strength is a percentage and enters as a plain mix:
 *
 *     out = a + (strength/100) * (S(a, a) - a)
 *
 * so 0 is the original and 100 is the full blend. Values outside [0,1] would
 * run away through the cubic, so they are clamped first; at this hook point
 * the picture is display-encoded and already inside it, which makes the
 * clamp free in practice.
 */

/* ------------------------------------------------------------- vibrance */

/* Vibrance raises chroma with more effect on pastel colours than on already
 * saturated ones. RawTherapee does this in CIE L*a*b*, where a hue rotation
 * of the Lab hue angle is not perceptually a constant hue - Lab's blue
 * region in particular bends badly - so it has to carry a Munsell hue
 * correction table to put the hue back where it started.
 *
 * This implementation works in Oklab instead. Oklab was fitted to keep
 * lines of constant hue straight, which is precisely the property the table
 * exists to restore, so scaling the (a, b) pair by a scalar already leaves
 * the hue alone and no table is needed. Nothing of RawTherapee's is
 * reproduced here but the shape of the controls.
 *
 * Chroma and hue are the polar form of Oklab's (a, b):
 *
 *     C = sqrt(a^2 + b^2),  h = atan2(b, a)
 *
 * and the whole filter is one number: the factor C is multiplied by. L and h
 * never move, so the result is hue-preserving and lightness-preserving by
 * construction.
 *
 * The two saturation sliders run -100..100 and become multiplicative
 * factors through
 *
 *     f(s) = 2^(s/50)
 *
 * i.e. +50 doubles chroma, -50 halves it, +100 quadruples and -100 quarters
 * it, and 0 is exactly 1. An exponential rather than a linear map is what
 * makes the slider behave the same either side of zero: a linear 1 + s/100
 * would reach 0 at -100 and only 2 at +100, so the two halves would not be
 * inverses and a -50 could not be undone by a +50.
 *
 * Which factor applies is decided by the pixel's own chroma against the
 * threshold. The threshold is a percentage of a reference chroma; Oklab
 * chroma of a fully saturated sRGB primary is about 0.3, so
 *
 *     C_t = (threshold/100) * 0.3
 *
 * and the transition between the two factors is a smoothstep over a band of
 * half a decade either side of it:
 *
 *     t = smoothstep(C_t / 1.6, C_t * 1.6, C)
 *     f = mix(f(pastels), f(saturated), t)
 *
 * so a pixel well below the threshold gets the pastel factor, one well
 * above gets the saturated factor, and the change between them has a
 * continuous derivative - which matters because a hard switch is visible as
 * a contour in any smooth gradient that crosses it.
 *
 * Protect skin tones is a smooth window in hue rather than a hard one, for
 * the same reason. Skin of every complexion sits in a narrow wedge of
 * orange-red: in Oklab its hue angle runs from roughly 25 to 70 degrees.
 * Inside that wedge the factor is pulled back towards 1, with a raised
 * cosine roll-off over 15 degrees at each edge so that no contour appears
 * where a cheek meets a background.
 *
 * Avoid colour shift is the gamut step. Multiplying C can push a colour
 * outside what the target's primaries can encode, and clipping the encoded
 * RGB afterwards would move the hue - clipping one channel and not the
 * others is exactly a hue rotation. Instead the chroma scale is walked back
 * towards 1 until the encoded RGB is inside [0,1], by bisection in the
 * shader, which keeps the hue and the lightness and gives up only the
 * chroma that could not be shown. Hue is preserved because every candidate
 * on that walk has the same h.
 *
 * The user curve over the pastel/saturated transition is not implemented in
 * this round. craw_vibrance_params carries the field it would enter
 * through, and DESIGN.md says what would go there.
 */

struct craw_vibrance_params {
    /* Both -100..100, 0 for no change. */
    float pastels;
    float saturated;
    /* 0..100, the chroma the pastel and saturated regimes meet at, as a
     * percentage of a saturated sRGB primary's Oklab chroma. */
    float threshold;
    bool protect_skin;
    bool avoid_shift;
};

/* True when these parameters cannot change a single pixel, which is what
 * lets the caller leave the hook off the list entirely rather than
 * dispatching an identity. */
bool craw_vibrance_is_noop(const struct craw_vibrance_params *p);

/* --------------------------------------------------------------- state */

/* Everything the three hooks keep between frames. One per pl_renderer - the
 * display and the export each own one, the way they each own a renderer -
 * because the LUT hook holds a pl_shader_obj and two renderers must not
 * share one. */
typedef struct craw_adjust craw_adjust;

craw_adjust *craw_adjust_create(pl_gpu gpu);
void craw_adjust_destroy(craw_adjust **a);

/* Each returns the pl_hook to hand to pl_render_params.hooks, or NULL when
 * that stage would do nothing and should not be attached at all. The
 * returned pointers stay valid until the next call for the same state.
 *
 * They must be added to the hook array in this order: LUT, vibrance, soft
 * light. libplacebo runs the hooks of a stage in array order. */
const struct pl_hook *craw_adjust_lut(craw_adjust *a,
                                      const struct pl_custom_lut *lut);
const struct pl_hook *craw_adjust_vibrance(craw_adjust *a,
                                           const struct craw_vibrance_params *p);
const struct pl_hook *craw_adjust_softlight(craw_adjust *a, float strength);

#endif /* CRAW_VIEW_ADJUST_H */
