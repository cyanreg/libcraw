/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#ifndef CRAW_TOOL_GRAIN_H
#define CRAW_TOOL_GRAIN_H

#include <stdbool.h>
#include <stdint.h>

#include <libplacebo/gpu.h>
#include <libplacebo/shaders/custom.h>
#include <libplacebo/shaders/film_grain.h>

/* Film grain emulation, built on libplacebo's AV1 (and H.274) film grain
 * synthesis shaders.
 *
 * The models themselves are codec grain-synthesis models: a small grain
 * template, a per-block random offset into it, and an amplitude that depends
 * on the local picture value. Nothing in either model knows about film. What
 * this file adds is the map from things a colourist names - a gauge, a stock,
 * "a bit more grain" - onto parameters those models can carry, by way of the
 * one physical fact that makes gauges differ at all: a smaller frame is
 * magnified more to fill the same screen, so the same dye cloud covers more
 * of it.
 *
 *
 * 1. The physical model
 * ---------------------
 * A colour negative's grain is a random field of dye clouds roughly 10-15 um
 * across. Two numbers describe how it lands on a screen:
 *
 *   blob size. Pixel pitch measured on the film is
 *
 *       p = 1000 * frame_width_mm / output_width_px    [um per pixel]
 *
 *   and a dye cloud of diameter d covers d/p pixels. Below one pixel the
 *   field is white noise - a pixel cannot show structure finer than itself.
 *
 *   amplitude. Granularity is published as diffuse RMS granularity: the
 *   standard deviation of density read through a 48 um circular aperture at
 *   a net density of 1.0, times 1000. Selwyn's law says that over a scanning
 *   aperture of area A the density fluctuation obeys sigma * sqrt(2A) = G, a
 *   constant of the emulsion, so sigma scales as 1/a for aperture diameter a:
 *
 *       sigma_D(a) = (rms48 / 1000) * 48 / a
 *
 *   with a = max(p, d): once the aperture is smaller than a single cloud you
 *   are no longer averaging clouds together and the fluctuation stops growing.
 *   That clamp is what keeps 8 mm at 4K from asking for impossible amplitudes.
 *
 * A density fluctuation is a ratio of transmittances, so for small sigma_D
 * the relative fluctuation of linear light is
 *
 *       sigma_L / L = ln(10) * sigma_D
 *
 * and that, times the `amount` control, is what the grain has to deliver.
 *
 *
 * 2. Where the grain is applied, and why there are two paths
 * ----------------------------------------------------------
 * `pl_frame.film_grain` is applied by pl_renderer inside the per-plane setup,
 * before colour conversion, before the main scaler and before tone mapping,
 * at the source plane's own resolution (renderer.c, plane_film_grain(), the
 * deband -> film grain -> user hooks sequence). For a 5760-wide S1II frame
 * shown at 1920 that is the wrong resolution to synthesise at, and for our
 * frames it is also the wrong range: the AV1 shader ends every component with
 * clamp(color, 0.0, 1.0) for full-range material, which would clip every
 * highlight above diffuse white in a scRGB RGBA16F frame. libplacebo's H.274
 * shader does not clamp, but its intensity intervals are matched against
 * 8-bit bounds in [0,1], so anything above 1.0 falls outside every interval
 * and silently receives no grain at all.
 *
 * So there are two paths, and the API makes the choice explicit:
 *
 *   CRAW_GRAIN_PATH_SOURCE   fills `pl_frame.film_grain` and lets pl_renderer
 *                            apply it. Physically the most correct place -
 *                            the display pixel averages film area by itself -
 *                            but limited to [0,1] material and to blob sizes
 *                            the model can reach at the source resolution.
 *
 *   CRAW_GRAIN_PATH_OUTPUT   a pl_hook at PL_HOOK_PRE_OUTPUT, which runs
 *                            after colour management and after the main
 *                            scaler: target resolution, display-encoded, and
 *                            therefore in [0,1] by construction, so the
 *                            model's clamps cost nothing. The hook calls
 *                            libplacebo's own pl_shader_film_grain(), so the
 *                            synthesis is identical; only the resolution and
 *                            the domain differ. This is the default.
 *
 * Measured reach of the two synthesis models, as the full width at half
 * maximum of the grain's autocorrelation in pixels at whatever resolution
 * the grain is generated at. These came out of grain_preview rendering
 * through libplacebo's own shaders and measuring the result, not out of
 * theory:
 *
 *   asked for   AV1 delivers  its amplitude   H.274 delivers  its amplitude
 *      1.00 px      1.00 px       1.02             1.06 px        1.00
 *      1.42         1.45          1.01             1.33           1.00
 *      2.01         2.16          1.00             2.07           1.00
 *      2.84         3.05          0.98             2.60           1.00
 *      4.01         4.15          0.97             3.62           1.00
 *      5.67         5.70          0.97             5.82           1.00
 *      8.02         7.69          0.95             5.82           1.00
 *
 * so AV1 covers 1 to about 7.7 px and H.274 1.06 to 5.82 px, and H.274's
 * ladder of thirteen cut-off frequencies cannot land between its rungs.
 *
 * Two things about AV1's half of that are worth keeping in mind, because
 * both are the opposite of what the model's shape suggests. A lag-3 filter
 * has a seven pixel wide support, but with coefficients small enough to
 * stay stable the autocorrelation it produces is still about one pixel
 * wide - libaom's own photon noise tables use lag 0 for exactly that
 * reason. What does reach multi-pixel blobs is a *strong low-order* filter:
 * a separable AR(1) at lag 1, whose single coefficient steers the blob size
 * directly. And that filter has a variance gain of 1/(1-a^2)^2, so it needs
 * grain_scale_shift to divide the white noise down first or its own output
 * runs into the +-128 clamp and turns into blotches.
 *
 * AV1 is the default for every gauge. H.274's model 0 is the better model
 * on paper - a cut-off ladder instead of a coefficient to tune, and blob
 * size that can vary with intensity, which AV1 cannot do at all - but
 * libplacebo 7.371 does not implement its deblocking (`// TODO: Deblocking?`
 * marks the spot) and lays the grain down as 16x16 patches cut from a 64x64
 * database with no blending between them. At one pixel that is invisible;
 * by three it is a quilt; at five it is the only thing you can see. The AV1
 * shader does implement the 27/17 block overlap, so its grain stays seamless
 * to its own ceiling.
 *
 *
 * 3. Gauge table
 * --------------
 * Blob size for a 13 um dye cloud - the VISION3 500T entry - and the Selwyn
 * amplitude multiplier 48/max(pitch, blob) that goes with it. Aperture
 * widths are camera apertures; craw_grain_gauge_source() names the standard
 * for each and flags the two that have none. "src" is the 5760-wide S1II
 * source.
 *
 *   gauge            width   blob in px            sigma multiplier
 *                     mm    src    4K   1080p    src    4K   1080p
 *   Regular 8         4.80  15.60 10.40   5.20   3.69  3.69   3.69
 *   Super 8           5.69  13.16  8.77   4.39   3.69  3.69   3.69
 *   16 mm            10.26   7.30  4.87   2.43   3.69  3.69   3.69
 *   Super 16         12.52   5.98  3.99   1.99   3.69  3.69   3.69
 *   35 mm 2-perf     22.00   3.40  2.27   1.13   3.69  3.69   3.69
 *   35 mm 3-perf     24.89   3.01  2.01   1.00   3.69  3.69   3.69
 *   35 mm Academy    21.95   3.41  2.27   1.14   3.69  3.69   3.69
 *   Super 35         24.89   3.01  2.01   1.00   3.69  3.69   3.69
 *   65 mm 5-perf     52.63   1.42  0.95   0.47   3.69  3.50   1.75
 *   IMAX 15-perf     70.41   1.06  0.71   0.35   3.69  2.62   1.31
 *
 * Read across the "blob in px" columns and the decision in section 2 makes
 * itself. At 1080p every gauge lands between 0.35 and 5.2 px, which AV1
 * covers with room to spare. At the source resolution Regular 8 and Super 8
 * ask for 13 to 16 px and no model reaches 8, so both come out about three
 * times too fine - and worse, a grain that coarse at 5760 is then reduced
 * 3:1 for a 1080p window, which aliases the 64x64 template's own repeat down
 * to a visible 10 px weave. That is in the pictures too, not just in the
 * arithmetic: see sheet_paths_grain.png.
 *
 * The sigma multipliers being nearly all the same number is the model
 * working, not a bug: below one dye cloud the Selwyn clamp holds the
 * per-pixel fluctuation at its saturated value, so what actually separates
 * the gauges at a given resolution is the blob size, and coarse grain looks
 * far stronger than fine grain of the same per-pixel standard deviation
 * because the eye integrates over an area.
 *
 *
 * 4. Calibration
 * --------------
 * With scaling_shift = 8 and grain_scale_shift = 0 at an effective depth of
 * 8 bits, the AV1 template is the Gaussian sequence (sigma 512) shifted right
 * by 4, so sigma = 32 in 8-bit units, and a scaling point of y contributes
 * a standard deviation of y * 32 / (256 * 255) = y / 2040 of full scale. The
 * point value for a wanted sigma is therefore y = 2040 * sigma. libaom's
 * examples/photon_noise_table.c independently arrives at
 * y = round(255 * 7.88 * sigma), i.e. y = 2009 * sigma, from an entirely
 * physical photon-shot-noise derivation; the 1.5% gap is its own empirical
 * trim. That agreement is the calibration anchor for this file.
 *
 * The AR filter changes the template's variance (a strong filter roughly
 * doubles it, grain_scale_shift divides it back down), so craw_grain_build()
 * does not use the 2040 constant directly: it generates the template exactly
 * as the spec does, measures its standard deviation, and solves for the
 * scaling points from that. tests/test_grain.c checks the measured value
 * against the intended one.
 */

enum craw_grain_gauge {
    CRAW_GRAIN_GAUGE_REGULAR8 = 0,
    CRAW_GRAIN_GAUGE_SUPER8,
    CRAW_GRAIN_GAUGE_16MM,
    CRAW_GRAIN_GAUGE_SUPER16,
    CRAW_GRAIN_GAUGE_35MM_2PERF,
    CRAW_GRAIN_GAUGE_35MM_3PERF,
    CRAW_GRAIN_GAUGE_35MM_4PERF,
    CRAW_GRAIN_GAUGE_SUPER35,
    CRAW_GRAIN_GAUGE_65MM,
    CRAW_GRAIN_GAUGE_IMAX,
    CRAW_GRAIN_GAUGE_NB,
};

enum craw_grain_stock {
    /* Generic, for when no particular stock is meant. */
    CRAW_GRAIN_STOCK_FINE = 0,
    CRAW_GRAIN_STOCK_MEDIUM,
    CRAW_GRAIN_STOCK_COARSE,
    /* Named stocks, with their published RMS granularity where one exists.
     * craw_grain_stock_source() says where each number came from. */
    CRAW_GRAIN_STOCK_V3_50D,
    CRAW_GRAIN_STOCK_V3_200T,
    CRAW_GRAIN_STOCK_V3_250D,
    CRAW_GRAIN_STOCK_V3_500T,
    CRAW_GRAIN_STOCK_DOUBLE_X,
    CRAW_GRAIN_STOCK_TRI_X,
    CRAW_GRAIN_STOCK_EKTACHROME,
    CRAW_GRAIN_STOCK_PORTRA400,
    CRAW_GRAIN_STOCK_NB,
};

/* How grain amplitude varies with picture level. The three film responses
 * are shapes, not measurements: the amplitude at the reference point is
 * calibrated (section 1), the way it falls away from there is modelled.
 * See craw_grain_response_note(). */
enum craw_grain_response {
    /* Grain grows with exposure. Nutting's law makes granularity rise as
     * roughly sqrt(D) over a negative's straight line, and on a camera
     * negative high density is a bright part of the scene, so a positive
     * made by simply inverting a negative scan is grainiest in its
     * highlights and clean at black. This is the modern scan-and-grade
     * look. */
    CRAW_GRAIN_RESPONSE_NEGATIVE = 0,
    /* Mid-tone peak. A print carries the negative's grain through its own
     * transfer curve, whose slope collapses in the print's toe and shoulder,
     * so the grain is strongest through the mid-tones and falls away at both
     * ends. The classic projected-print look. */
    CRAW_GRAIN_RESPONSE_PRINT,
    /* Peak pushed down into the shadows, highlights nearly clean. On a
     * reversal stock the retained silver - and so the grain - is densest
     * where the scene was dark. */
    CRAW_GRAIN_RESPONSE_REVERSAL,
    /* Constant amplitude everywhere. Not a film response; it is what a
     * sensor does, and it is the one that makes the measurement in
     * grain_preview.c unambiguous. */
    CRAW_GRAIN_RESPONSE_FLAT,
    CRAW_GRAIN_RESPONSE_NB,
};

enum craw_grain_model {
    /* Pick per gauge: H.274 where its cut-off frequencies reach a blob size
     * AV1's AR filter cannot, AV1 otherwise. craw_grain_build() reports what
     * it chose in `model_used`. */
    CRAW_GRAIN_MODEL_AUTO = 0,
    CRAW_GRAIN_MODEL_AV1,
    CRAW_GRAIN_MODEL_H274,
    CRAW_GRAIN_MODEL_NB,
};

enum craw_grain_path {
    /* A pl_hook at PL_HOOK_PRE_OUTPUT: target resolution, display-encoded. */
    CRAW_GRAIN_PATH_OUTPUT = 0,
    /* pl_frame.film_grain: source resolution, source range. */
    CRAW_GRAIN_PATH_SOURCE,
    CRAW_GRAIN_PATH_NB,
};

struct craw_grain_params {
    enum craw_grain_gauge gauge;
    enum craw_grain_stock stock;
    enum craw_grain_response response;
    enum craw_grain_model model;
    enum craw_grain_path path;

    /* Multiplies the amplitude the physical model asks for. 1.0 is the
     * stock's published granularity; 0 disables grain entirely. */
    float amount;
    /* Multiplies the blob diameter the gauge asks for. 1.0 is the stock's
     * dye cloud size at that gauge. */
    float size;
    /* Chroma grain relative to luma. Colour negative grain is largely
     * uncorrelated between the three dye layers, so the chroma channels do
     * carry grain, but the eye's chroma acuity is low and prints are
     * further smoothed by the paper, which is why this defaults below 1. */
    float chroma;

    /* Resolution the grain will be applied at. For CRAW_GRAIN_PATH_OUTPUT
     * this is the target size; for CRAW_GRAIN_PATH_SOURCE the source plane
     * size. Only the width enters the model - the film's aspect ratio is
     * not the sensor's, and it is the horizontal magnification that sets
     * the blob size. */
    int width, height;

    /* Seed for this frame. With boil, feed a new value every frame and the
     * grain reshuffles like film through a gate; without it, hold one value
     * and the grain is fixed in place, which is wrong for film but is what
     * makes a still comparison readable. Only the low 16 bits reach the AV1
     * template. */
    uint64_t seed;
};

/* Sensible starting point: Super 35 on 250D, print response, AV1 or H.274
 * chosen automatically, applied at the output. width/height and seed are
 * left at zero and must be filled in. */
struct craw_grain_params craw_grain_default_params(void);

/* What craw_grain_build() worked out, for the UI, the docs and the tests. */
struct craw_grain_out {
    /* Zero when `amount` is 0 or the stock has no grain: nothing to apply. */
    bool enabled;

    struct pl_film_grain_data data;
    enum craw_grain_model model_used;

    /* The physical intermediates, all at the resolution in params. */
    float pitch_um;         /* film um covered by one pixel */
    float blob_um;          /* dye cloud diameter of the stock */
    float blob_px;          /* what the gauge asks for */
    float blob_clamped;     /* what the model can actually build */
    float sigma_density;    /* Selwyn-scaled RMS granularity */
    float sigma_rel;        /* relative linear-light modulation, x amount */
    /* The same thing in the units of the signal the grain is added to, at
     * mid grey: a fraction of full scale of the display-encoded value on
     * the output path, of the scene-linear value on the source path. This
     * is what the scaling function is solved for. */
    float sigma_signal;
    /* What the built, quantised parameters will actually produce there.
     * Compare with sigma_signal; grain_preview.c compares both against a
     * measurement of the rendered picture. */
    float sigma_built;
    /* The standard deviation one unit of the model's own scale number
     * produces: point_y_scaling for AV1, comp_model_value[0] for H.274.
     * craw_grain_sigma_at() is the way to use it. */
    float unit_sd;
};

/* The standard deviation the built data will produce at picture level
 * `level`, which runs 0 to 1 in the units of whichever signal the chosen
 * path applies grain to. This is the response curve as the shader will
 * actually walk it, quantisation and all, and it is what a measurement of
 * a flat field at that level should come back with. */
float craw_grain_sigma_at(const struct craw_grain_out *out, float level);

/* Builds grain data from perceptual parameters. Returns 0, or a negative
 * errno on a parameter that makes no sense (a zero width, an out of range
 * enum). Never returns invalid grain data: everything it emits passes
 * craw_grain_validate(). */
int craw_grain_build(const struct craw_grain_params *p,
                     struct craw_grain_out *out);

/* Checks every constraint the AV1 and H.274 specifications place on the
 * data. Returns NULL when the data is valid, or a static string naming the
 * first violation. Cheap enough to call on every build; the tests do. */
const char *craw_grain_validate(const struct pl_film_grain_data *d);

/* Generates the AV1 luma grain template exactly as AV1 section 7.18.3.3
 * does, into an 82x73 array, and returns its standard deviation over the
 * 64x64 window libplacebo uploads. Exposed because the calibration in
 * section 4 needs it and because tests/test_grain.c checks it against the
 * spec. `bits` is the effective bit depth (8 for our float frames). */
float craw_grain_template_y(int16_t buf[73][82],
                            const struct pl_av1_grain_data *d,
                            int bits, uint16_t seed);

/* Names for the UI and the command line. Stable, lowercase, no spaces for
 * the *_key forms; the *_name forms are for display. */
const char *craw_grain_gauge_name(enum craw_grain_gauge g);
const char *craw_grain_gauge_key(enum craw_grain_gauge g);
const char *craw_grain_stock_name(enum craw_grain_stock s);
const char *craw_grain_stock_key(enum craw_grain_stock s);
const char *craw_grain_response_name(enum craw_grain_response r);
const char *craw_grain_model_name(enum craw_grain_model m);

/* -1 when no gauge/stock/response has that key. Case insensitive. */
int craw_grain_gauge_from_key(const char *key);
int craw_grain_stock_from_key(const char *key);
int craw_grain_response_from_key(const char *key);

/* Camera aperture width in mm, and which standard it comes from. */
float craw_grain_gauge_width_mm(enum craw_grain_gauge g);
const char *craw_grain_gauge_source(enum craw_grain_gauge g);

/* Published diffuse RMS granularity x1000, the dye cloud diameter in um,
 * and where each came from - including "estimated" where it is a guess. */
float craw_grain_stock_rms(enum craw_grain_stock s);
float craw_grain_stock_blob_um(enum craw_grain_stock s);
const char *craw_grain_stock_source(enum craw_grain_stock s);

/* Why a response has the shape it has, in one sentence, for a tooltip. */
const char *craw_grain_response_note(enum craw_grain_response r);

/* ---------------------------------------------------------------- hook */

/* The output-resolution path. Holds the shader state libplacebo's grain
 * shaders keep between frames, so it lives as long as the renderer does. */
typedef struct craw_grain_hook craw_grain_hook;

craw_grain_hook *craw_grain_hook_create(pl_gpu gpu);
void craw_grain_hook_destroy(craw_grain_hook **h);

/* Rebuilds the hook's grain data from `p` and returns the pl_hook to put in
 * pl_render_params.hooks, or NULL when grain is disabled or the build
 * failed. The returned pointer stays valid until the next call or until the
 * hook is destroyed. `out` may be NULL. Call this once per frame with a
 * fresh seed to make the grain boil. */
const struct pl_hook *craw_grain_hook_update(craw_grain_hook *h,
                                             const struct craw_grain_params *p,
                                             struct craw_grain_out *out);

#endif /* CRAW_TOOL_GRAIN_H */
