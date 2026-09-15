/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#include <libplacebo/colorspace.h>
#include <libplacebo/common.h>
#include <libplacebo/dispatch.h>
#include <libplacebo/shaders/colorspace.h>

#include <craw/utils.h>

#include "adjust.h"
#include "view.h"

/* The three display-referred hooks. adjust.h carries the derivation and the
 * sources; this file is the shaders and the plumbing that hands them to
 * pl_renderer.
 *
 * All three are PL_HOOK_SIG_COLOR, so each one appends its instructions to
 * the shader the renderer is already building rather than finishing it into
 * a texture and starting another. Nothing here reads a neighbouring pixel,
 * so there is nothing to gain from a texture and a whole FBO round trip to
 * lose. */

/* Oklab, from Björn Ottosson's public derivation. The cube roots sit
 * between these two: XYZ -> LMS, cube root, LMS' -> Lab.
 *
 * They are kept in XYZ form rather than baked against sRGB because the
 * target of this hook is whatever the window or the file is - BT.709 for
 * both today, but a BT.2020 swapchain would otherwise silently get the
 * wrong hue angles. Composing with the target's own RGB <-> XYZ matrices
 * costs two matrix multiplies once per frame and makes the space exact for
 * any D65 primaries. */
static const pl_matrix3x3 oklab_xyz2lms = {{
    {  0.8189330101f,  0.3618667424f, -0.1288597137f },
    {  0.0329845436f,  0.9293118715f,  0.0361456387f },
    {  0.0482003018f,  0.2643662691f,  0.6338517070f },
}};

static const pl_matrix3x3 oklab_lms2xyz = {{
    {  1.2270138511f, -0.5577999807f,  0.2812561490f },
    { -0.0405801784f,  1.1122568696f, -0.0716766787f },
    { -0.0763812845f, -0.4214819784f,  1.5861632204f },
}};

/* Oklab chroma of a fully saturated sRGB primary, which is what the
 * threshold is a percentage of. Measured rather than derived: the six sRGB
 * primaries and secondaries come out between 0.15 (cyan) and 0.32
 * (magenta), so 0.30 puts a threshold of 100 at the edge of the gamut. */
#define ADJUST_CHROMA_REF 0.30f

/* How much of the effect skin keeps. RawTherapee's classifier emits 0.3 for
 * its core skin region, and that number is worth matching: it is the one
 * thing about the control a user carries over. */
#define ADJUST_SKIN_KEEP 0.3f

struct craw_adjust {
    pl_gpu gpu;

    /* The film simulation LUT. The shader object holds the uploaded 3D
     * texture between frames, which is why one of these belongs to each
     * renderer rather than to the program. */
    struct pl_hook lut_hook;
    pl_shader_obj lut_state;
    const struct pl_custom_lut *lut;

    struct pl_hook vib_hook;
    struct craw_vibrance_params vib;
    /* Derived once per frame, in the target's primaries. */
    enum pl_color_primaries vib_prim;
    float vib_rgb2lms[9], vib_lms2rgb[9];    /* column major, for GLSL */
    bool vib_have_mat;

    struct pl_hook soft_hook;
    float soft;
};

/* GLSL takes a mat3 as three columns; pl_matrix3x3 is m[row][column]. */
static void adjust_mat3(const pl_matrix3x3 *m, float out[9])
{
    for (int c = 0; c < 3; c++) {
        for (int r = 0; r < 3; r++)
            out[c * 3 + r] = m->m[r][c];
    }
}

/* ---------------------------------------------------------------- LUT */

static struct pl_hook_res adjust_lut_fn(void *priv,
                                        const struct pl_hook_params *hp)
{
    craw_adjust *a = priv;

    if (!a->lut || !hp->sh)
        return (struct pl_hook_res) { .failed = true };

    /* The same call libplacebo makes for `pl_frame.lut` at PL_LUT_NATIVE,
     * on the same values: between here and there it runs only
     * pl_shader_encode_color(), which for a full range RGB target whose
     * sample and colour depths agree skips itself entirely. */
    pl_shader_custom_lut(hp->sh, a->lut, &a->lut_state);

    return (struct pl_hook_res) {
        .output     = PL_HOOK_SIG_COLOR,
        .sh         = hp->sh,
        .repr       = hp->repr,
        .color      = hp->color,
        .components = hp->components,
        .rect       = hp->rect,
    };
}

const struct pl_hook *craw_adjust_lut(craw_adjust *a,
                                      const struct pl_custom_lut *lut)
{
    if (!a || !lut)
        return NULL;

    a->lut = lut;

    return &a->lut_hook;
}

/* --------------------------------------------------------- soft light */

/* The self-blended Pegtop soft light, which collapses to the smoothstep
 * polynomial, mixed against the original by the strength.
 *
 * The `step` pair reproduces one behaviour of RawTherapee's that a clamp
 * would not: a channel outside [0,1] is left exactly alone rather than
 * pulled to the boundary, and the decision is per channel. The blend is
 * only defined on [0,1] - the cubic runs away outside it - and at this hook
 * point a channel is out of range only where the gamut mapper left it
 * there, which is precisely where dragging it to the boundary would invent
 * a colour. */
static const char *const adjust_soft_body =
    "vec3 craw_v = color.rgb;                                            \n"
    "vec3 craw_b = craw_v * craw_v * (3.0 - 2.0 * craw_v);               \n"
    "vec3 craw_ok = step(vec3(0.0), craw_v) * step(craw_v, vec3(1.0));   \n"
    "color.rgb = craw_v + craw_ok * craw_soft * (craw_b - craw_v);       \n";

static struct pl_hook_res adjust_soft_fn(void *priv,
                                         const struct pl_hook_params *hp)
{
    craw_adjust *a = priv;
    const float mix = a->soft / 100.0f;
    const struct pl_shader_var vars[1] = {
        { .var = pl_var_float("craw_soft"), .data = &mix, .dynamic = true },
    };

    if (!hp->sh || hp->components < 3)
        return (struct pl_hook_res) { .failed = true };

    if (!pl_shader_custom(hp->sh, &(struct pl_custom_shader) {
            .description   = "craw soft light",
            .input         = PL_SHADER_SIG_COLOR,
            .output        = PL_SHADER_SIG_COLOR,
            .variables     = vars,
            .num_variables = 1,
            .body          = adjust_soft_body,
        }))
        return (struct pl_hook_res) { .failed = true };

    return (struct pl_hook_res) {
        .output     = PL_HOOK_SIG_COLOR,
        .sh         = hp->sh,
        .repr       = hp->repr,
        .color      = hp->color,
        .components = hp->components,
        .rect       = hp->rect,
    };
}

const struct pl_hook *craw_adjust_softlight(craw_adjust *a, float strength)
{
    if (!a || !(strength > 0.0f))
        return NULL;

    a->soft = strength;

    return &a->soft_hook;
}

/* ------------------------------------------------------------ vibrance */

bool craw_vibrance_is_noop(const struct craw_vibrance_params *p)
{
    /* The two amounts are the whole filter: with both at zero the chroma
     * factor is one everywhere and neither toggle has anything to act on.
     * RawTherapee returns from the same test, and it is what lets the
     * caller leave the hook off the list so that the defaults are not
     * merely an identity but an absence. */
    return !p || (p->pastels == 0.0f && p->saturated == 0.0f);
}

/* Lab -> linear RGB, as a helper so that the gamut walk and the final
 * conversion are the same three lines. The matrix comes in as an argument
 * rather than off the #define so that this text does not depend on what
 * pl_shader_custom named the uniform. */
static const char *const adjust_vib_header =
    "vec3 craw_ok2rgb(mat3 m, float L, float A, float B)                 \n"
    "{                                                                   \n"
    "    vec3 t = vec3(L + 0.3963377774 * A + 0.2158037573 * B,          \n"
    "                  L - 0.1055613458 * A - 0.0638541728 * B,          \n"
    "                  L - 0.0894841775 * A - 1.2914855480 * B);         \n"
    "    return m * (t * t * t);                                         \n"
    "}                                                                   \n"
    "                                                                    \n"
    "bool craw_okfits(vec3 q)                                            \n"
    "{                                                                   \n"
    "    return all(greaterThanEqual(q, vec3(-1e-4)))                    \n"
    "        && all(lessThanEqual(q, vec3(1.0001)));                     \n"
    "}                                                                   \n";

/* Everything up to the chroma factor. `craw_vib` is (pastel amount,
 * saturated amount, transition low, transition high) and `craw_skin` is how
 * much of the effect a skin hue loses, or zero when the toggle is off. */
static const char *const adjust_vib_head =
    "vec3 craw_l = craw_rgb2lms * color.rgb;                             \n"
    "craw_l = sign(craw_l) * pow(abs(craw_l), vec3(1.0 / 3.0));          \n"
    "float craw_L = dot(vec3( 0.2104542553,  0.7936177850,               \n"
    "                        -0.0040720468), craw_l);                    \n"
    "float craw_A = dot(vec3( 1.9779984951, -2.4285922050,               \n"
    "                         0.4505937099), craw_l);                    \n"
    "float craw_B = dot(vec3( 0.0259040371,  0.7827717662,               \n"
    "                        -0.8086757660), craw_l);                    \n"
    "float craw_C = sqrt(craw_A * craw_A + craw_B * craw_B);             \n"
    /* Pastel below the threshold, saturated above, and a smoothstep rather
     * than a corner across it: a hard switch is visible as a contour in
     * any gradient that crosses it. */
    "float craw_m = mix(craw_vib.x, craw_vib.y,                          \n"
    "                   smoothstep(craw_vib.z, craw_vib.w, craw_C));     \n"
    /* Skin is a wedge of warm hues, bounded above in chroma so that a
     * saturated orange is not mistaken for a cheek. Both edges roll off,
     * for the same reason the threshold does. */
    "float craw_h = degrees(atan(craw_B, craw_A));                       \n"
    "float craw_w = smoothstep(10.0, 25.0, craw_h)                       \n"
    "             * (1.0 - smoothstep(75.0, 90.0, craw_h))               \n"
    "             * (1.0 - smoothstep(0.12, 0.18, craw_C));              \n"
    "craw_m *= 1.0 - craw_skin * craw_w;                                 \n"
    /* The limiter is RawTherapee's: chroma may be cut to 7% of itself but
     * no further, which keeps a full negative slider from collapsing a
     * colour onto the neutral axis and losing its hue with it. */
    "float craw_f = 1.0 + clamp(craw_m, -0.93, 2.0);                     \n";

/* Avoid colour shift. Walking the scale back beats clipping the encoded RGB
 * because every candidate on the walk has the same Oklab hue and lightness,
 * so what is given up is only the chroma that could not be shown; clipping
 * one channel and not the others is a hue rotation by another name.
 *
 * The asked-for scale is tried first, and the walk only runs where it did
 * not fit. That is not just a saving: bisection converges on the boundary
 * from inside, so a colour that was always going to fit would come back
 * about a 256th of the interval short of what was asked for, and the filter
 * would quietly under-deliver everywhere rather than only at the boundary.
 *
 * Where the walk does run, eight bisections resolve the interval to a 256th
 * of it, which at the widest factor this filter can ask for is a chroma
 * error well below the quantisation of the window it ends up in. The
 * invariant is that `craw_lo` is always a scale that fits. */
static const char *const adjust_vib_gamut =
    "if (craw_f > 1.0 && !craw_okfits(craw_ok2rgb(craw_lms2rgb, craw_L,  \n"
    "                                             craw_A * craw_f,       \n"
    "                                             craw_B * craw_f))) {   \n"
    "    float craw_lo = 1.0, craw_hi = craw_f;                          \n"
    "    for (int craw_i = 0; craw_i < 8; craw_i++) {                    \n"
    "        float craw_t = 0.5 * (craw_lo + craw_hi);                   \n"
    "        bool craw_in = craw_okfits(craw_ok2rgb(craw_lms2rgb, craw_L, \n"
    "                                               craw_A * craw_t,     \n"
    "                                               craw_B * craw_t));   \n"
    "        craw_lo = craw_in ? craw_t : craw_lo;                       \n"
    "        craw_hi = craw_in ? craw_hi : craw_t;                       \n"
    "    }                                                               \n"
    "    craw_f = craw_lo;                                               \n"
    "}                                                                   \n";

static const char *const adjust_vib_tail =
    "color.rgb = craw_ok2rgb(craw_lms2rgb, craw_L,                       \n"
    "                        craw_A * craw_f, craw_B * craw_f);          \n";

static struct pl_hook_res adjust_vib_fn(void *priv,
                                        const struct pl_hook_params *hp)
{
    craw_adjust *a = priv;
    struct pl_color_space csp = hp->color;
    const struct craw_vibrance_params *p = &a->vib;
    /* Big enough for the longer of the two variants with room to spare;
     * the guard below is the one that matters. */
    char body[4096];
    float vib[4], skin;

    if (!hp->sh || hp->components < 3)
        return (struct pl_hook_res) { .failed = true };

    /* The matrices follow the target's primaries, which do not move once a
     * window is open, so they are rebuilt only when they do. */
    if (csp.primaries == PL_COLOR_PRIM_UNKNOWN)
        csp.primaries = PL_COLOR_PRIM_BT_709;

    if (!a->vib_have_mat || a->vib_prim != csp.primaries) {
        const struct pl_raw_primaries *prim =
            pl_raw_primaries_get(csp.primaries);
        pl_matrix3x3 rgb2lms = oklab_xyz2lms;
        pl_matrix3x3 lms2rgb = pl_get_xyz2rgb_matrix(prim);
        pl_matrix3x3 rgb2xyz = pl_get_rgb2xyz_matrix(prim);

        pl_matrix3x3_mul(&rgb2lms, &rgb2xyz);       /* xyz2lms * rgb2xyz */
        pl_matrix3x3_mul(&lms2rgb, &oklab_lms2xyz); /* xyz2rgb * lms2xyz */

        adjust_mat3(&rgb2lms, a->vib_rgb2lms);
        adjust_mat3(&lms2rgb, a->vib_lms2rgb);
        a->vib_prim = csp.primaries;
        a->vib_have_mat = true;
    }

    /* The sliders enter as a fraction, the way RawTherapee's do: +100 is
     * twice the chroma, -100 all of it the limiter will give up. */
    vib[0] = p->pastels / 100.0f;
    vib[1] = p->saturated / 100.0f;

    /* The threshold is a chroma, and each regime hands a quarter of its own
     * span to the transition, so the band is narrow where the regime is. */
    {
        float ct = CRAW_CLAMP(p->threshold, 0.0f, 100.0f)
                 / 100.0f * ADJUST_CHROMA_REF;

        vib[2] = ct * 0.75f;
        vib[3] = ct + (ADJUST_CHROMA_REF - ct) * 0.25f;
        /* A threshold of zero would make the smoothstep degenerate; nudging
         * the band open leaves every real pixel above it, which is what
         * "no pastel regime at all" should mean. */
        if (!(vib[3] > vib[2]))
            vib[3] = vib[2] + 1e-4f;
    }

    skin = p->protect_skin ? 1.0f - ADJUST_SKIN_KEEP : 0.0f;

    const struct pl_shader_var vars[4] = {
        { .var = pl_var_vec4("craw_vib"),     .data = vib,   .dynamic = true },
        { .var = pl_var_float("craw_skin"),   .data = &skin, .dynamic = true },
        { .var = pl_var_mat3("craw_rgb2lms"), .data = a->vib_rgb2lms },
        { .var = pl_var_mat3("craw_lms2rgb"), .data = a->vib_lms2rgb },
    };

    if (snprintf(body, sizeof(body), "%s%s%s", adjust_vib_head,
                 p->avoid_shift ? adjust_vib_gamut : "",
                 adjust_vib_tail) >= (int)sizeof(body)) {
        /* Said out loud: libplacebo answers a failed hook by disabling it
         * for the rest of the run, so this would otherwise show up as
         * vibrance quietly doing nothing at all. */
        fprintf(stderr, "the vibrance shader does not fit its buffer\n");
        return (struct pl_hook_res) { .failed = true };
    }

    /* Oklab is a linear-light space, so the display encoding has to come
     * off and go back on around it. libplacebo's own transfer function
     * shaders do it for whatever the target turned out to be. */
    pl_shader_linearize(hp->sh, &csp);

    if (!pl_shader_custom(hp->sh, &(struct pl_custom_shader) {
            .description   = "craw vibrance",
            .input         = PL_SHADER_SIG_COLOR,
            .output        = PL_SHADER_SIG_COLOR,
            .header        = adjust_vib_header,
            .variables     = vars,
            .num_variables = 4,
            .body          = body,
        }))
        return (struct pl_hook_res) { .failed = true };

    pl_shader_delinearize(hp->sh, &csp);

    return (struct pl_hook_res) {
        .output     = PL_HOOK_SIG_COLOR,
        .sh         = hp->sh,
        .repr       = hp->repr,
        .color      = hp->color,
        .components = hp->components,
        .rect       = hp->rect,
    };
}

const struct pl_hook *craw_adjust_vibrance(craw_adjust *a,
                                           const struct craw_vibrance_params *p)
{
    if (!a || craw_vibrance_is_noop(p))
        return NULL;

    a->vib = *p;

    return &a->vib_hook;
}

/* -------------------------------------------------------------- state */

craw_adjust *craw_adjust_create(pl_gpu gpu)
{
    craw_adjust *a = calloc(1, sizeof(*a));

    if (!a)
        return NULL;

    a->gpu = gpu;

    /* All three sit where the target LUT used to: after colour management
     * and the main scaler, before blending and rotation, so they land on
     * the picture in the encoding the window or the file will hold and on
     * the video only - an overlay drawn later is not graded. */
    a->lut_hook = (struct pl_hook) {
        .stages    = PL_HOOK_PRE_OUTPUT,
        .input     = PL_HOOK_SIG_COLOR,
        .priv      = a,
        .hook      = adjust_lut_fn,
        .signature = 0x6372617766696C6DULL,     /* "crawfilm" */
    };
    a->vib_hook = (struct pl_hook) {
        .stages    = PL_HOOK_PRE_OUTPUT,
        .input     = PL_HOOK_SIG_COLOR,
        .priv      = a,
        .hook      = adjust_vib_fn,
        .signature = 0x6372617776696272ULL,     /* "crawvibr" */
    };
    a->soft_hook = (struct pl_hook) {
        .stages    = PL_HOOK_PRE_OUTPUT,
        .input     = PL_HOOK_SIG_COLOR,
        .priv      = a,
        .hook      = adjust_soft_fn,
        .signature = 0x637261777366746CULL,     /* "crawsftl" */
    };

    return a;
}

void craw_adjust_destroy(craw_adjust **pa)
{
    craw_adjust *a = pa ? *pa : NULL;

    if (!a)
        return;

    pl_shader_obj_destroy(&a->lut_state);
    free(a);
    *pa = NULL;
}

/* ------------------------------------------------------ the viewer's side */

/* Everything below belongs to craw-view rather than to the model: the
 * defaults the panel starts from, the two command line options and the
 * readout. The hooks themselves are handed to pl_render_params by gpu.c. */

void view_adjust_defaults(struct view_adjust *a)
{
    *a = (struct view_adjust) {
        /* Soft light off. RawTherapee's slider defaults to 30 but its tool
         * starts disabled, so 0 is the same picture by a shorter route. */
        .softlight = 0.0f,

        /* Vibrance neutral. Both amounts at zero is what
         * craw_vibrance_is_noop() tests, so the hook is not attached at
         * all until one of them moves. */
        .pastels   = 0.0f,
        .saturated = 0.0f,
        /* The chroma the pastel and saturated regimes meet at, as a
         * percentage of a saturated sRGB primary's Oklab chroma. 45 puts
         * the crossover at a chroma of 0.135, which is where
         * RawTherapee's default threshold of 75 puts it: its abscissa is
         * the RGB saturation 1 - min/max, and over a large sample of sRGB
         * colours its default crossover of 0.7675 comes out at a median
         * Oklab chroma of 0.135. Pastels measure 0.04 to 0.07 and the
         * primaries 0.15 to 0.32, so the two regimes land either side. */
        .threshold = 45.0f,
        /* Both toggles on, the way somebody who reaches for vibrance at
         * all wants them: skin held back, and chroma kept inside the
         * gamut instead of clipped against it. RawTherapee ships avoid
         * colour shift on and skin protection off; the second is on here
         * because without RawTherapee's hue table a hue-window is all the
         * protection there is, and it costs nothing when there is no skin
         * in the frame. */
        .protect_skin      = true,
        .protect_skin_flag = 1,
        .avoid_shift       = true,
        .avoid_shift_flag  = 1,
    };
}

int view_adjust_parse_softlight(struct view_adjust *a, const char *arg)
{
    char *end;
    float v = strtof(arg, &end);

    if (end == arg || *end || !(v >= 0.0f) || v > 100.0f) {
        fprintf(stderr, "--softlight: expected a percentage 0 to 100, "
                        "got `%s`\n", arg);
        return CRAW_ERROR(EINVAL);
    }

    a->softlight = v;

    return 0;
}

int view_adjust_parse_vibrance(struct view_adjust *a, const char *arg)
{
    /* PASTELS[,SATURATED[,THRESHOLD]]. One number is the common case -
     * RawTherapee's own panel links the two amounts by default - so a lone
     * PASTELS sets both, which is what that link does. */
    float v[3] = { 0.0f, 0.0f, a->threshold };
    static const float lo[3] = { -100.0f, -100.0f,   0.0f };
    static const char *const name[3] = { "pastels", "saturated", "threshold" };
    const char *p = arg;
    char *end = (char *) arg;
    int n = 0;

    while (n < 3) {
        v[n] = strtof(p, &end);
        if (end == p || !(v[n] >= lo[n]) || v[n] > 100.0f) {
            fprintf(stderr, "--vibrance: %s must be %g to 100, got `%s`\n",
                    name[n], (double)lo[n], p);
            return CRAW_ERROR(EINVAL);
        }
        n++;
        if (*end != ',')
            break;
        p = end + 1;
    }

    /* Whatever the loop stopped on has to be the end of the string: a
     * fourth field, or anything the number parser did not eat, is a typo
     * and not something to quietly ignore. */
    if (*end) {
        fprintf(stderr, "--vibrance: expected "
                        "PASTELS[,SATURATED[,THRESHOLD]], got `%s`\n", arg);
        return CRAW_ERROR(EINVAL);
    }

    a->pastels   = v[0];
    a->saturated = n >= 2 ? v[1] : v[0];
    a->threshold = v[2];

    return 0;
}

void view_adjust_close(struct view_adjust *a)
{
    craw_adjust_destroy(&a->state);
    craw_adjust_destroy(&a->state_export);
}

const char *view_adjust_summary(const struct view_adjust *a, char *buf,
                                size_t n)
{
    struct craw_vibrance_params vp = {
        .pastels = a->pastels, .saturated = a->saturated,
    };
    bool vib  = !craw_vibrance_is_noop(&vp);
    bool soft = a->softlight > 0.0f;

    if (!vib && !soft)
        snprintf(buf, n, "off");
    else if (vib && soft)
        snprintf(buf, n, "vibrance %+.0f/%+.0f, soft light %.0f%%",
                 (double)a->pastels, (double)a->saturated,
                 (double)a->softlight);
    else if (vib)
        snprintf(buf, n, "vibrance %+.0f/%+.0f",
                 (double)a->pastels, (double)a->saturated);
    else
        snprintf(buf, n, "soft light %.0f%%", (double)a->softlight);

    return buf;
}
