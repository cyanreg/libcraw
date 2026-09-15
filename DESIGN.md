# libcraw design

Cinema RAW processing library. Vulkan compute only. BSD-2-Clause, C23,
libavtransport conventions. This file is the specification that the
shaders, the CPU reference and the tests implement; when two of them
disagree, this file decides and the others get fixed.

The word for the reconstruction step is **debayer** everywhere in the code:
identifiers, file names, comments, docs. Not demosaic.

## 1. Scope of this version

In:

- Bayer input, RGGB only (other phases are plumbed as a specialization
  constant but rejected until tested), 16-bit UNORM samples.
- Black/white normalization, as-shot white balance, clip at the white
  balanced clip point.
- Three debayer methods: bilinear, Malvar-He-Cutler (MHC), and a
  ratio-corrected directional method (RCD).
- Colour stage: re-balance to a chosen illuminant, camera matrix, gain,
  exposure, conversion to an output colour space. Linear float output.
- Auto white balance: GPU histogram, CPU estimator on the profile's locus,
  temporal smoother.
- Camera profile model with per-CCT tables (as-shot exact vs re-balance).
- CPU reference of every stage, tests comparing it against the GPU.

Out, deliberately, with a slot reserved: denoising, scaling, learned
debayer, highlight reconstruction beyond the clip, OCIO-generated
transforms, LUTs, other CFA phases, ARRIRAW and BRAW profiles, and lens
correction. (The GUI, craw-view, exists under tools/gui as a consumer of
the library: it is not part of the library's scope.) On the last one: Panasonic ProRes RAW frames carry a
per-frame `com.panasonic.Semi-Pro.optical_correction` record in the
vendor metadata after the last tile (version 3: per channel R, G, B a
radial polynomial k0..k3 plus two zero pads, then the optical centre in
normalized frame coordinates, all big-endian doubles). The coefficients
follow zoom and focus frame by frame and encode distortion with lateral
chromatic aberration in the destination-to-source convention,
`r_src = r_dst * (k0 + k1 r^2 + k2 r^4 + k3 r^6)` per channel, with k0
below 1.0 acting as the crop that keeps a pincushion correction inside the
frame. Vignetting is not in the record. A lens stage would be a per-channel
resampling warp after the debayer and before any scaling, fed through the
profile or per-frame params.

## 2. Conventions

- Public API: `craw_` functions, `CRAW`-prefixed CamelCase types,
  `CRAW_` macros, `CRAW_API` on every exported function, negative errno
  returns (`CRAW_ERROR(EINVAL)`), double-pointer free functions that NULL
  the caller's handle, `goto fail` cleanup, designated initializers.
- Public headers carry the full BSD-2 block; internal files use the one
  line form `/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */`.
- 4-space indent, K&R braces, function opening brace on its own line,
  `while (1)` for infinite loops, never `for (;;)`.
- Padding tails on public option structs for ABI growth; always zeroed.
- Every allocation on the caller's VkDevice goes through the VMA wrapper in
  `lib/vk_runtime.h`. The library never creates instances, devices,
  queues, command pools or command buffers, and never submits.
- Shaders: GLSL 460 compute, compiled at build time by glslang to SPIR-V for
  Vulkan 1.4, embedded by `tools/spv_embed.py`. Workgroup size via
  specialization constants 253/254/255. All descriptors in set 0, pushed.
  Push constants at most 128 bytes. `#extension GL_EXT_scalar_block_layout`,
  `GL_EXT_shader_image_load_formatted` and `GL_GOOGLE_include_directive`
  (preprocessor only, for `#include "common.glsl"`) are allowed; anything
  else needs a note here.
- Precision: fp32 arithmetic in shaders unless a stage documents an fp16
  fast path; fp16 storage for the intermediate camera RGB image.

## 3. Data model

Coordinates: x is the column, y the row, (0, 0) top left. CFA quad site
index `s = (y & 1) * 2 + (x & 1)`. For RGGB: site 0 = R, 1 = G (in the R
row, "Gr"), 2 = G (in the B row, "Gb"), 3 = B.

Input image: `VK_FORMAT_R16_UNORM`, the mosaic, sample `raw(x, y)` in
[0, 1]. This is what FFmpeg's Vulkan hwcontext maps `AV_PIX_FMT_BAYER_RGGB16`
to.

Intermediate image (internal): `VK_FORMAT_R16G16B16A16_SFLOAT`, camera RGB
after prep and debayer, alpha unused.

Output image: `VK_FORMAT_R16G16B16A16_SFLOAT` or `R32G32B32A32_SFLOAT`,
linear light in the requested colour space, alpha 1.0.

Border handling, all stages: mirror without repeating the edge sample and
preserving CFA phase: `x < 0 -> -x`, `x >= W -> 2W - 2 - x`, same for y.

## 4. Stage math

### 4.1 Prep (fused into the debayer loads)

For a sample at site `s` with channel `c(s)`:

    v  = (raw - black_level[s]) / (white_level[s] - black_level[s])
    v  = max(v, 0)
    v  = v * wb_asshot[c(s)]
    v  = min(v, 1.0)

`wb_asshot` is a misnomer kept for the shader field: it is
`CRAWColorParams.wb` evaluated for the frame's own wb_mode and illuminant.
The clip at 1.0 after white balance keeps blown highlights neutral under
that illuminant; it is the placeholder for highlight reconstruction. Note
the clip point differs per channel in raw units. An earlier revision
balanced with the as-shot gains and re-balanced in the colour stage; that
turned every clipped lamp the colour of the gain ratio, so the debayer
output is now tied to the frame's white balance and a grade-time change
re-runs it, which costs a few milliseconds.

The debayer operates on `v`. A sample is "clipped" when
`raw >= white_level[s] - 1/1024`; this mask is not used by this version but
must be cheap to add.

### 4.2 Debayer

Produces R, G, B at every pixel from `v`. Definitions in section 5.

### 4.3 Colour

With `c` the camera RGB from the intermediate and the frame's
`CRAWColorParams` `p` (as-shot or re-balance) plus the as-shot params
`a`:

    c'  = c                               the gains are already in c
    xyz = p.cam_to_xyz * c'
    xyz = xyz * p.gain * 2^exposure_stops
    out = M_out * xyz                     M_out from craw_colorspace_from_xyz_d65()
    if (flags & CLAMP_NEGATIVE) out = max(out, 0)
    alpha = 1

No clamp of any kind unless the flag is set. Values above 1.0 are normal
(sensor clip sits at `gain` times diffuse white).

### 4.4 Auto white balance histogram (optional stage)

Reads the intermediate at 1/8 resolution: one sample per 8x8 pixel block,
the mean of the block's camera RGB (a 4x4 arrangement of 2x2 quads). Only
whole blocks count: floor(W/8) x floor(H/8) samples, any remainder at the
right and bottom edges is ignored, on the GPU and in the CPU builder alike.
For each sample:

    skip if max(r, g, b) >= 0.98 (clipped) or g < 0.002 (near black)
    rr = (r / wb_r) / (g / wb_g)                 raw ratio, the applied gains divided out
    bb = (b / wb_b) / (g / wb_g)                 (wb_g is 1.0 for every table seen so far)
    x  = ln(rr), y = ln(bb), both clamped into [LOG_MIN, LOG_MAX]
    ix = floor((x - LOG_MIN) / (LOG_MAX - LOG_MIN) * BINS), same for iy, clamped to BINS-1
    weight = min(g, 0.5)
    count[iy * BINS + ix]         += 1
    weight[iy * BINS + ix]        += round(weight * 65536)   (16.16 fixed point, atomics)
    weight_bright[iy * BINS + ix] += round(g^4 * 65536)      (same, emphasises light sources)

NaN ratios (negative debayer output) land in the first bin rather than out
of range. The 16.16 accumulator overflows above roughly 131k samples in one
bin; at 1/8 resolution that needs a flat neutral covering a quarter of an
8K frame, so it is documented rather than handled.

BINS = 128, LOG_MIN = -2.5, LOG_MAX = 1.0 (see `craw/awb.h`). The buffers
are zeroed by the pipeline at the start of each recording (vkCmdFillBuffer)
and copied to a host-visible readback buffer at the end.

## 5. Debayer definitions

Notation: `v(dx, dy)` is the prepped sample at offset (dx, dy) from the
current pixel, with mirrored borders. "Colour of the site" is the channel
native to a pixel.

### 5.1 Bilinear

At an R site: R = v(0,0); G = mean of the 4 edge neighbours; B = mean of
the 4 corner neighbours. At a B site: symmetric. At a G site in an R row
(site 1): G = v(0,0); R = mean of v(-1,0), v(1,0); B = mean of v(0,-1),
v(0,1). At a G site in a B row (site 2): G = v(0,0); B = mean of v(-1,0),
v(1,0); R = mean of v(0,-1), v(0,1).

### 5.2 MHC (Malvar, He, Cutler 2004), 5x5 kernels, divided by 8

G at R or B site (centre colour C):

    G = (4 v(0,0) + 2 [v(0,-1) + v(0,1) + v(-1,0) + v(1,0)]
         - [v(0,-2) + v(0,2) + v(-2,0) + v(2,0)]) / 8

C2 at C1 site where the two colours share a row (R at site 1, B at site 2),
i.e. the wanted colour is at horizontal neighbours:

    C2 = (5 v(0,0) + 4 [v(-1,0) + v(1,0)]
          - [v(-1,-1) + v(1,-1) + v(-1,1) + v(1,1) + v(-2,0) + v(2,0)]
          + (1/2) [v(0,-2) + v(0,2)]) / 8

C2 at C1 site where the wanted colour is at vertical neighbours (B at site
1, R at site 2): the transpose of the above (swap x and y offsets).

B at R site and R at B site (wanted colour at the diagonals):

    C2 = (6 v(0,0) + 2 [v(-1,-1) + v(1,-1) + v(-1,1) + v(1,1)]
          - (3/2) [v(0,-2) + v(0,2) + v(-2,0) + v(2,0)]) / 8

The native colour is copied. No clamping inside the debayer; the colour
stage sees whatever the kernels produce.

### 5.3 RCD: ratio-corrected directional debayer

Independently specified here from the published description of Sanz
Rodríguez's Ratio Corrected Demosaicing (its README: directional
discrimination on the CFA, a 3x3 binomial low-pass, green from ratios in
the low-pass domain, chroma from colour differences). No code from that
project is used. Where this text and that project differ, this text
wins; the official implementation is only a quality benchmark.

Three passes on the GPU, one function in the reference. eps = 1e-5.

Pass A, directions and low-pass, at every pixel:

    L    = binomial 3x3 of v: [1 2 1; 2 4 2; 1 2 1] / 16
    Vg   = |v(0,-1) - v(0,1)| + |2 v(0,0) - v(0,-2) - v(0,2)|
    Hg   = |v(-1,0) - v(1,0)| + |2 v(0,0) - v(-2,0) - v(2,0)|
    Pg   = |v(-1,1) - v(1,-1)| + |2 v(0,0) - v(-2,2) - v(2,-2)|     (NE-SW)
    Qg   = |v(-1,-1) - v(1,1)| + |2 v(0,0) - v(-2,-2) - v(2,2)|     (NW-SE)

then each of Vg, Hg, Pg, Qg is smoothed with a 3x3 box filter (mean of
the 9 values, mirrored borders) into Vs, Hs, Ps, Qs, and

    wv = Hs / (Hs + Vs + eps)        weight of the vertical estimate
    wp = Qs / (Ps + Qs + eps)        weight of the P (NE-SW) estimate

Pass B, green at R and B sites, using the neighbours' green and the
low-pass ratio (the published RCD-LPF formula applied to both neighbours
and averaged):

    Gv = 0.5 * [ v(0,-1) * (1 + (L(0,0) - L(0,-2)) / (L(0,0) + L(0,-2) + eps))
               + v(0, 1) * (1 + (L(0,0) - L(0, 2)) / (L(0,0) + L(0, 2) + eps)) ]
    Gh = 0.5 * [ v(-1,0) * (1 + (L(0,0) - L(-2,0)) / (L(0,0) + L(-2,0) + eps))
               + v( 1,0) * (1 + (L(0,0) - L( 2,0)) / (L(0,0) + L( 2,0) + eps)) ]
    G  = wv * Gv + (1 - wv) * Gh

At G sites G = v(0,0). After this pass a full green plane exists.

Pass C, red and blue.

At an R site (B wanted) and at a B site (R wanted), from the diagonal
neighbours' colour differences, where D(dx,dy) = v(dx,dy) - G(dx,dy) at
those neighbours (their native colour minus the green from pass B):

    Cp = G(0,0) + 0.5 * [ D(-1, 1) + D( 1,-1) ]
    Cq = G(0,0) + 0.5 * [ D(-1,-1) + D( 1, 1) ]
    C  = wp * Cp + (1 - wp) * Cq

At a G site, both R and B are wanted. For the colour whose native sites
are the horizontal neighbours (R at site 1, B at site 2):

    Ch = G(0,0) + 0.5 * [ D(-1,0) + D(1,0) ]
    Cv = G(0,0) + 0.5 * [ E(0,-1) + E(0,1) ]
    C  = wv * Cv + (1 - wv) * Ch

where E(dx,dy) is that same colour's difference to green at the vertical
neighbours, which are sites of the *other* colour, so E uses the value
interpolated for them in the first half of this pass (the diagonal
estimate at that neighbour minus its green). For the colour whose native
sites are the vertical neighbours, swap the roles: Cv from D at (0,±1),
Ch from E at (±1,0). Implementations therefore compute the R/B-site
estimates for the whole image before the G-site estimates (a separate
dispatch or a two-phase loop in the reference).

Rationale for the reference: this reproduces the published structure
(directional discrimination on the CFA, LPF-domain ratio correction for
green, colour-difference chroma) with fully specified stencils so that a
CPU and a GPU implementation written independently agree to float
rounding.

## 6. Profile evaluation

`craw_profile_eval()`:

- AS_SHOT: `wb = as_shot_wb`, `cam_to_xyz = as_shot_matrix`.
- REBALANCE: `cct` clamped to the table range; `wb` = linear
  interpolation in Kelvin of the wb table, then `wb[R] *= exp(tint/sqrt(2))`,
  `wb[B] *= exp(tint/sqrt(2))`; `cam_to_xyz` = linear interpolation in
  Kelvin of the matrix table, element by element, or `as_shot_matrix` when
  that table is empty.
- `gain = p->gain * (exposure_index / as_shot_ei)` when both > 0, else
  `p->gain`.

Locus mapping (`craw_profile_neutral_to_illuminant`): the locus is the
polyline through `(ln(1/wb[R]), ln(1/wb[B]))` of the wb knots, linear in
Kelvin between knots to match the vendor decoder. It is sampled so that
every knot is itself a sample and each knot interval is split into a whole
number of equal steps of at most 5 mired; a uniform global grid would miss
the knots and bias the table's own entries. For a raw neutral `(ln rg, ln bg)`
take the nearest sample; `cct` is its Kelvin value; `tint` is the dot
product of the residual (point minus sample) with `(-1, -1)/sqrt(2)`.
`craw_profile_illuminant_to_neutral` is the inverse:
`rg = (1/wb_R(cct)) * exp(-tint/sqrt(2))`, `bg` likewise.

## 7. Auto white balance estimator

Inputs: histogram, profile, `CRAWAwbParams`. Per bin with count > 0, the
bin centre is a candidate point. Compute its nearest locus sample as in
section 6 and its distance `d`. Bins with `d < band` are in the zone.
Two readings are taken from the same bins with different weights:

- zone: `w = weight_bin * exp(-(d / sigma)^2)`, every neutral-looking
  surface counts by its brightness, capped at 0.5;
- bright: the same with `weight_bright_bin`, i.e. g^4, so that the few
  brightest unclipped samples dominate. A lamp, its glare or a specular
  highlight carries the illuminant's own colour; a wood ceiling under a
  cool LED reads as a warm neutral to the zone reading and drags it down
  by thousands of Kelvin, but cannot compete with the lamp under g^4.

For each reading:

    in_zone_fraction = sum of that weighting's bin mass in zone / its total
    mired            = weighted mean over the zone of 1e6 / cct(sample)
    cct              = 1e6 / mired
    tint             = weighted mean over the zone of the residual dot (-1,-1)/sqrt(2)
    spread_mired     = weighted standard deviation of 1e6 / cct(sample)
    valid            = in_zone_fraction >= min_zone_fraction
                       and, for bright only, spread_mired <= bright_max_spread_mired

Selection: the bright reading when `prefer_bright` and it is valid, else
the zone reading. Confidence of the chosen reading:

    confidence = clamp(in_zone_fraction / 0.3, 0, 1) * clamp(1 - spread_mired / 60, 0, 1)

Ambience: if `cct < knee`, `cct = cct + ambience * (knee - cct)` after the
estimate, i.e. warm light is only partially corrected. The estimate is
then clamped to `[cct_min, cct_max]`.

Smoother: state is a mired value and a tint. On push: if `scene_cut` or no
state, adopt the estimate; when there is no state and the estimate is
invalid there is nothing to adopt and the call fails with EAGAIN, leaving
the output untouched. Else if `est.valid && est.confidence >=
min_confidence`: target = estimate; ignore the change if
`|target - state| < hysteresis_mired`; otherwise move the state towards the
target by `1 - exp(-dt / time_constant)` of the difference, limited to
`max_slew_mired_per_s * dt`. Tint follows the same rule with the same
fractions. `dt` is the difference in `pts_s` to the previous push, clamped
to [0, 1] seconds.

## 8. Colour spaces

`craw_colorspace_from_xyz_d65()` returns row-major matrices. D65 spaces use
the inverse of the RGB-to-XYZ matrix built from the primaries and a D65
white of (0.3127, 0.3290). The ACES spaces first adapt D65 to the ACES
white (0.32168, 0.33767) with a Bradford transform, then apply the inverse
of the AP0 or AP1 RGB-to-XYZ matrix. The OCIO studio config gives these
reference values, which the implementation must match to 1e-3:

XYZ (D65) to ACES2065-1 (AP0):

    +1.063496  +0.006409  -0.015807
    -0.492074  +1.368223  +0.091337
    -0.002816  +0.004644  +0.916419

XYZ (D65) to ACEScg (AP1):

    +1.660585  -0.315296  -0.241509
    -0.659926  +1.608392  +0.017299
    +0.009003  -0.003567  +0.913643

Primaries (x, y): BT.709 R(0.64,0.33) G(0.30,0.60) B(0.15,0.06);
BT.2020 R(0.708,0.292) G(0.170,0.797) B(0.131,0.046); P3-D65
R(0.680,0.320) G(0.265,0.690) B(0.150,0.060); V-Gamut R(0.730,0.280)
G(0.165,0.840) B(0.100,-0.030); AP0 R(0.7347,0.2653) G(0.0,1.0)
B(0.0001,-0.0770); AP1 R(0.713,0.293) G(0.165,0.830) B(0.128,0.044).

## 9. Test vectors (Panasonic DC-S1M2, clip P1000387.MOV, ProRes RAW)

Per-frame header: black 256/65535, white 63743/65535, wb_r 2.02539,
wb_b 1.66797, gain 23.04, cct 5500, matrix

    +0.695646 +0.152540 +0.102269
    +0.303238 +0.760187 -0.063425
    +0.005748 +0.031326 +1.051984

Container tables (identical in every clip from this camera):

    wb knots (cct: r, b): 2000: 0.8454 3.9482; 3200: 1.3763 2.4707;
    3400: 1.4477 2.2881; 5000: 1.9499 1.7195; 5500: 2.0253 1.6676;
    6300: 2.2051 1.5348; 7300: 2.3299 1.4186; 11500: 2.6469 1.2318

    matrix knots (cct: row-major 3x3):
    2300:  0.6535 0.1947 0.1022  0.2831 0.6975 0.0193  0.0113 -0.1074 1.1852
    2800:  0.6601 0.1881 0.1022  0.2862 0.7073 0.0064  0.0104 -0.0857 1.1644
    4000:  0.6759 0.1723 0.1023  0.2938 0.7308 -0.0246  0.0083 -0.0337 1.1144
    5000:  0.6891 0.1591 0.1023  0.3001 0.7504 -0.0505  0.0066 0.0096 1.0728
    6500:  0.7088 0.1394 0.1023  0.3095 0.7798 -0.0893  0.0040 0.0747 1.0104
    7500:  0.7218 0.1263 0.1023  0.3158 0.7991 -0.1149  0.0023 0.1176 0.9692
    10000: same as 7500
    15000: same as 7500

Facts a test can assert: REBALANCE at 5500 K with zero tint reproduces the
per-frame matrix to 5e-4 and the per-frame wb pair to 1e-3 (the camera
interpolates linearly in Kelvin); `neutral_to_illuminant(1/2.0253,
1/1.6676)` returns 5500 K and a tint of 0 to 1e-3; the estimator on a
synthetic histogram of a neutral scene at 3200 K plus 30% of saturated
coloured samples returns 3200 K within 100 K; a scene that is 90% warm surface reading as a
3000 K neutral at low luminance plus 10% glare of a 6000 K light at high
luminance returns about 6000 K from the bright reading, and about 3000 K
with prefer_bright off.

## 9b. Exposure model and parity with Apple's decoder

Verified in Apple's ProRes RAW plug-in (ProResRAW.dll, Ghidra, 2026-09-15):

    gain = GainFactor * (ISOExposureIndex / AsShotISOEI) * 2^ExposureOffset

with GainFactor the big-endian float at frame offset 0x48 (36.0 when a
frame lacks it), AsShotISOEI parsed only from the container key
`org.smpte.rdd18.camera.isosensitivity` (the plug-in refuses clips
without it), ISOExposureIndex defaulting to AsShotISOEI and restricted to
the standard third-stop ISO list, and ExposureOffset a bias of at most one
stop either way, default 0. Nothing else feeds the gain: no base ISO, no
EXIF ExposureIndex key, no vendor record. libcraw's profile maps onto
this one to one: `gain`, `as_shot_ei`, `exposure_index`, and
`CRAWFrameParams.exposure_stops` as the bias.

Apple's linear output comes in two domains selected by the container
key `com.apple.proresraw.scenelinearscaling`: "Reflection", where the
values are reflectances (18% grey at 0.18, sensor clip at GainFactor),
and the default "Normalized", which is the same times 10/9 so that a 90%
white card sits at 1.0. Every camera log and HLG output is generated
from the normalized domain with a 0.9 factor inside the curve; Apple Log
uses the reflection domain. libcraw outputs the reflection domain; a
consumer that wants Apple's normalized numbers multiplies by 10/9.

The plug-in has no display-referred output and no tone curve, knee,
soft clip or desaturation anywhere: the only clip is the per-channel
`min(x, 1.0)` in white balanced camera RGB before the matrix, which is
exactly section 4.1. Everything above 1.0 after the matrix and gain passes
through. Any highlight rendering seen in an NLE is that application's own
tone mapping downstream, which is where libcraw leaves it too.

Apple's decoder reads one vendor record from the frame trailer,
`com.apple.gdc`, a geometric distortion model (centre, scale, 15 radial
coefficients, radius limits), and ignores Panasonic's
`optical_correction` record entirely. The plug-in also carries built-in
per-CCT tables for the Panasonic 35LT1 and S1H and the Nikon Z 6 and Z 7,
for containers without them.

## 10. Vulkan runtime

- Barriers use synchronization2 with per-image tracked layout/stage/access
  (`craw_vk_image_barrier`). External images are wrapped for the duration
  of one recording and left in the layout the caller stated.
- Intermediates are `VK_IMAGE_LAYOUT_GENERAL` storage images. Formatless
  reads/writes (`GL_EXT_shader_image_load_formatted`) are required, and so
  is `shaderStorageImageExtendedFormats` for the R16 source and green plane.
  Formatless *reads* are what exclude Intel ANV today; every image the
  shaders read has a fixed format, so a future variant could annotate the
  reads and require only formatless writes if that ever matters.
- Buffers for histograms are storage buffers with device addresses; the
  readback buffer is host visible and cached.
- The pipeline records, per frame: barrier src to GENERAL/compute read,
  [fill histogram buffers], debayer passes, [awb histogram], colour pass,
  [copy histogram to readback], and a final barrier that returns dst to
  the caller's stated layout.
- Push constant structs are defined once in `lib/pipeline.c` and mirrored
  in GLSL with `layout(push_constant, scalar)`; keep them in the same
  order and pad explicitly.

## 11. Testing

- `test_profile`: evaluation, locus mapping, colour space matrices against
  section 8 and 9 values.
- `test_ref_debayer`: on synthetic mosaics (constant colour, horizontal
  and vertical linear ramps, a checkerboard) with known analytic results:
  bilinear and MHC reproduce a linear ramp exactly away from borders; all
  three methods reproduce a constant colour exactly; borders never NaN.
- `test_awb`: estimator and smoother on synthetic histograms/tracks.
- `test_pipeline`: GPU vs CPU reference on a synthetic 256x160 mosaic and,
  when `TEST_SOURCE_DIR/../samples` provides one, on a real frame dump.
  Tolerances: bilinear/MHC 2e-3 absolute per channel on the camera RGB
  intermediate (fp16), 1e-2 relative on the colour output; RCD 5e-3 /
  1.5e-2. lavapipe is acceptable for these tests; it exposes everything
  required.

## 11b. Known API limitations to revisit

- Histogram readback is multi-slot since `nb_awb_slots` was added to
  `CRAWPipelineParams`: with n slots `craw_pipeline_get_awb_histogram_n()`
  reads the histogram of a frame up to n - 1 recordings back, so a player
  waits only for that older frame's submission. The device-side histogram
  buffers themselves stay single because successive recordings on one
  queue are ordered by the pipeline's barriers.
- The debayer method is fixed at pipeline creation; switching it rebuilds
  the pipeline. Creating every method's shaders up front and selecting per
  frame would make it a live control.
- The pipeline's intermediates are not double-buffered, so a recording per
  submission is the rule (section 4 of pipeline.h).

## 12. Work split for the initial implementation

Contract files, owned by the integrator: everything under `include/`,
`lib/vk_runtime.h`, `lib/ref/ref.h`, `lib/meson.build`, `tests/vkdev.h`,
`tests/meson.build`, this file.

- Runtime: `lib/craw.c`, `lib/vk_functions.c`, `lib/vk_buffer.c`,
  `lib/vk_image.c`, `lib/vk_shader.c`, `lib/vma/vma.cpp`, `tests/vkdev.c`.
- Shaders and pipeline: `lib/shaders/*.glsl`, `lib/shaders/meson.build`,
  `lib/pipeline.c`, `tests/test_pipeline.c`.
- Colour science, reference, CPU tests: `lib/profile.c`,
  `lib/colorspace.c`, `lib/awb.c`, `lib/awb_smoother.c`,
  `lib/ref/ref_debayer.c`, `lib/ref/ref_color.c`, `tests/test_profile.c`,
  `tests/test_ref_debayer.c`, `tests/test_awb.c`.

Anyone needing a change to a contract file reports it instead of editing.

## 13. Film grain emulation

`tools/gui/grain.c` and `grain.h` turn a gauge and a stock into parameters
for libplacebo's film grain shaders. `tools/grain/grain_preview.c` is the
headless proof: it renders a chart and a developed frame through the same
`pl_render_image` path craw-view uses, measures what came out and writes
contact sheets. Every number below is from that tool unless it is marked as
a citation, and `--calibrate` re-measures the two tables `grain.c` carries
against whatever libplacebo is installed.

### 13.1 What libplacebo does, and where

`pl_needs_film_grain` / `pl_shader_film_grain` implement AV1 section 7.18.3
and H.274 (SMPTE RDD 5) model 0. `pl_renderer` applies `pl_frame.film_grain`
in `plane_film_grain()`, inside the per-plane setup, in the order
*deband -> film grain -> user hooks*: at the **source plane's own
resolution**, in the **source representation**, before colour conversion,
before the main scaler and before tone mapping.

Both models work on RGB: `channel_map()` sends G to the model's luma channel,
B to Cb and R to Cr. Luma-only parameters therefore give **green-only
grain**, so anything driving these shaders with an RGB image has to fill the
chroma channels too.

Neither model has a notion of floating point. `bit_depth()` falls back to 8
when `repr.bits` is unset, which is the case for our RGBA16F frames, so the
grain is quantised to 8-bit steps of the signal - with the scaling function's
own resolution on top of that, fine enough not to matter here. What does
matter is the range: the AV1 shader ends every component with
`clamp(color, 0.0, 1.0)` for full-range material. A scRGB RGBA16F frame whose
diffuse white is 1.0 would have every highlight above it clipped flat. The
H.274 shader has no clamp at all, but matches intensity intervals against
8-bit bounds in [0,1], so a value above 1.0 falls outside every interval and
silently gets no grain. Neither failure is acceptable on the display path.

Two implementation details found by reading and then confirmed by
measurement, both worth reporting upstream:

- **libplacebo's H.274 shader mis-assigns components for RGB.** It loops over
  the picture's components, maps each to a grain channel with
  `channel_map()`, and then subscripts `color[]` with that *channel* index
  rather than the component index. For YCbCr the two coincide; for RGB the
  effect is that model component 0 lands on red, 1 on green, 2 on blue.
  Measured: with only component 0 present, green came back with no grain and
  red carried it. The AV1 shader does this correctly. `build_h274()` works
  around it and says so.
- **libplacebo's H.274 shader does no deblocking** - there is a
  `// TODO: Deblocking?` where it would go - and lays grain down as 16x16
  patches cut from the 64x64 database with no blending between patches. It
  also indexes that database transposed relative to how it fills it, so the
  horizontal and vertical cut-off frequencies are swapped; harmless while
  they are equal, which is all this model asks for.

### 13.2 The physical model

Film grain differs between gauges for one reason: a smaller frame is
magnified more to fill the same screen. Two numbers follow.

**Blob size.** Pixel pitch measured on the film is
`p = 1000 * frame_width_mm / output_width_px` micrometres, and a dye cloud of
diameter `d` covers `d/p` pixels. Below one pixel the field is white noise.

**Amplitude.** Granularity is published as *diffuse RMS granularity*: the
standard deviation of density through a 48 um aperture at a net density of
1.0, times 1000. Selwyn's law - `sigma * sqrt(2A) = G` for aperture area A -
makes sigma scale as `1/a`, so

    sigma_D(a) = (rms48 / 1000) * 48 / a,   a = max(p, d)

The clamp at `a = d` is the physics: once the aperture is smaller than one
dye cloud you are no longer averaging clouds and the fluctuation saturates.
A density fluctuation is a transmittance ratio, so the relative fluctuation
of linear light is `ln(10) * sigma_D`, and with a display encoding
`v = L^(1/gamma)` the absolute amplitude the shader has to add is
`rho * v / gamma`. `signal_amp()` does that conversion; gamma is taken as
2.2, which sRGB is within a percent of over the range that matters.

The consequence worth internalising is that at a given output resolution the
gauges mostly share one amplitude - the Selwyn clamp saturates for every
gauge whose pixel is smaller than a dye cloud - and differ in blob size. That
is correct. Coarse grain looks far stronger than fine grain of the same
per-pixel standard deviation because the eye integrates over an area.

### 13.3 Gauge table

Camera apertures, with the standard each comes from in
`craw_grain_gauge_source()`. Blob sizes are for a 13 um dye cloud (the
VISION3 500T entry); "src" is the 5760-wide S1II source.

| gauge | width mm | blob @5760 | @3840 | @1920 | standard |
|---|---|---|---|---|---|
| Regular 8 | 4.80 | 15.60 | 10.40 | 5.20 | SMPTE ST 231; sources give 4.80-4.88 |
| Super 8 | 5.69 | 13.16 | 8.77 | 4.39 | SMPTE ST 157:1999 |
| 16 mm | 10.26 | 7.30 | 4.87 | 2.43 | SMPTE ST 7:2004 / ISO 69:1998 |
| Super 16 | 12.52 | 5.98 | 3.99 | 1.99 | SMPTE ST 201:1996 / ISO 5768 |
| 35 mm 2-perf | 22.00 | 3.40 | 2.27 | 1.13 | Techniscope; no standard exists |
| 35 mm 3-perf | 24.89 | 3.01 | 2.01 | 1.00 | ISO 2906:2002 style D |
| 35 mm Academy | 21.95 | 3.41 | 2.27 | 1.14 | SMPTE ST 59 / ISO 2906 style A |
| Super 35 | 24.89 | 3.01 | 2.01 | 1.00 | never standardised by SMPTE |
| 65 mm 5-perf | 52.63 | 1.42 | 0.95 | 0.47 | Todd-AO; another source says 48.59 |
| IMAX 15-perf | 70.41 | 1.06 | 0.71 | 0.35 | IMAX camera aperture, proprietary |

### 13.4 The resolution problem, and the two paths

Measured reach of the two models, as the full width at half maximum of the
grain's autocorrelation, at the resolution grain is generated at:

| asked for | AV1 gives | its amplitude | H.274 gives | its amplitude |
|---|---|---|---|---|
| 1.00 px | 1.00 px | 1.02 | 1.06 px | 1.00 |
| 2.01 | 2.16 | 1.00 | 2.07 | 1.00 |
| 4.01 | 4.15 | 0.97 | 3.62 | 1.00 |
| 5.67 | 5.70 | 0.97 | 5.82 | 1.00 |
| 8.02 | 7.69 | 0.95 | 5.82 | 1.00 |

Put that beside the gauge table. At **1080p** every gauge wants between 0.35
and 5.2 px and AV1 covers all of it. At the **5760 source** Regular 8 and
Super 8 want 13 to 16 px and nothing reaches 8, so they come out about three
times too fine. Worse, grain that coarse at 5760 is then reduced 3:1 for a
1080p window, which aliases the 64x64 template's own repeat down to a visible
regular weave about 10 px across - `sheet_paths_grain.png` shows it plainly,
a woven pattern on the source-path row against clean organic grain on the
output-path row. Add the AV1 clamp to [0,1] on an HDR frame and the native
path is not usable as the default.

So there are two paths and the API makes the choice explicit:

- **`CRAW_GRAIN_PATH_OUTPUT`** (default): a `pl_hook` at
  `PL_HOOK_PRE_OUTPUT` that calls libplacebo's own `pl_shader_film_grain()`
  on the hook's texture. A probe hook confirms rather than assumes what
  arrives there: for a 1920x1080 target the texture is 1920x1080, so target
  resolution, and its transfer is `PL_COLOR_TRC_SRGB`, so display encoded and
  in [0,1] by construction, which makes the shaders' clamps free. That it
  runs after colour management and the main scaler and before blending and
  rotation is from reading `pass_output_target()`, not from the probe.
  Because the synthesis is libplacebo's, this path is not a reimplementation
  - only the resolution and the domain differ.
- **`CRAW_GRAIN_PATH_SOURCE`**: fills `pl_frame.film_grain`. Physically the
  most correct place, since the display pixel then averages film area by
  itself, and the right choice for SDR material already in [0,1] rendered at
  or near its own resolution. Measured at 1920 from a 5760 source it delivers
  0.0054 where the output path delivers 0.0077, i.e. about 30% less, and the
  weave above.

### 13.5 Parameter mapping

**Blob size -> AR coefficients.** A separable first-order filter
`g = w + a*g[x-1] + a*g[y-1] - a^2*g[y-1][x-1]` has autocorrelation
`a^|dx| * a^|dy|`, so one number steers the blob. `ar_coeff_shift` is 7:
coefficients then reach 0.992 in steps of 1/128, where shift 6 would reach
1.98 in twice the step and shift 8 tops out at 0.496, too low for coarse
grain. `ar_solve()` bisects over the achievable integers against a
*measured* blob size, because the analytic inversion `a = 0.5^(2/b)` drifts
up to 30% at the coarse end.

A lag-3 filter is not the way to make big grain, despite its seven pixel
support: with coefficients small enough to stay stable it still produces an
autocorrelation about one pixel wide (measured: all 24 coefficients at 20
with shift 9 gives 1.1 px). libaom's photon noise tables use lag 0.

**Amplitude -> scaling function.** With `scaling_shift` 8 and
`grain_scale_shift` 0 at 8 bits the template is the Gaussian sequence
(sigma 512) shifted right by 4, so sigma is 32 in 8-bit units and a scaling
point of `y` contributes `y * 32 / (256 * 255) = y / 2040` of full scale.
libaom's `examples/photon_noise_table.c` independently arrives at
`y = round(255 * 7.88 * sigma)`, i.e. `y = 2009 * sigma`, from a photon shot
noise derivation; 2040 against 2009 is the calibration anchor for this file.

`grain.c` does not use the 2040 constant, because the AR filter changes the
template's variance. It generates the template exactly as the spec does,
synthesises a 192x192 piece of the field the shader will build from it - 32x32
blocks, the spec's per-row LFSR offsets, the 27/17 overlap - and solves
against *that*. Measuring the 64x64 template alone is not enough: subtracting
a 64x64 window's own mean removes the lowest frequencies, which is where a
coarse filter keeps most of its energy, and it made a five pixel grain come
out 40% low against the GPU.

**grain_scale_shift.** Chosen from the target blob size
(`gss_for_blob()`), not from a clipping test at the strongest coefficient -
testing there finds that a = 0.93 clips at every shift below 3 and then
solves the whole problem at shift 3, where the template is so quiet the
scaling function saturates before reaching the amplitude asked for.

**Response.** Four shapes, defined as amplitude against display-encoded
level and normalised so that mid grey always gets the physical amplitude,
whichever response is chosen:

| response | shape | measured at 2% / 18% / 50% / 90% scene |
|---|---|---|
| Negative | `x^1.1`, capped at 1.8 | 0.0049 / 0.0146 / 0.0211 / 0.0251 |
| Print | Gaussian about mid grey, width 0.42 | 0.0093 / 0.0174 / 0.0179 / 0.0160 |
| Reversal | asymmetric Gaussian peaked at 0.30 | 0.0238 / 0.0268 / 0.0130 / 0.0061 |
| Flat | constant | 0.0185 / 0.0185 / 0.0185 / 0.0185 |

Negative is the one with a physical derivation: Nutting's law puts
granularity at roughly `sqrt(D)` and on a camera negative density follows
scene exposure, so an inverted scan is grainiest in its highlights and clean
at black. Print and Reversal are **modelled shapes, not measurements** - the
print's transfer curve has slope only through its middle so the negative's
grain is swallowed in its toe and shoulder, and a reversal stock keeps its
silver where the scene was dark. The widths and the peak position were chosen
by eye and are the most obvious thing to refine.

**Chroma.** The three dye layers' grain is largely independent, which is why
colour negative grain is coloured. `chroma` sets both the amplitude relative
to luma and, through the AV1 chroma AR filter's luma cross term, how
correlated the three channels are: 0 gives one monochrome grain, 1 gives
three independent ones. The chroma templates are measured separately rather
than reusing the luma standard deviation, because the cross term is injected
*into* the recursion and therefore comes out filtered twice - with a strong
filter that more than doubled the chroma amplitude, which measurement caught
as "colour grain louder than luma grain".

**Seed.** A new seed every frame makes the grain boil. Measured: two seeds
give the same standard deviation (0.0123 and 0.0125) and an rms difference of
0.0246, i.e. `sqrt(2)` times either, so the two frames' grain is fully
independent. On AV1 only the low 16 bits reach the template, which libplacebo
regenerates and re-uploads whenever the seed changes; H.274's database does
not depend on the seed at all, so boiling is free there.

### 13.6 Calibrated, guessed, and unknown

**Calibrated against published data.** Only four stocks have a published
diffuse RMS granularity: Double-X 5222 at 14 (Kodak H-1-5222, March 2022, net
density 1.0, 48 um), Tri-X 400 at 17 (Kodak F-4017, February 2016), Tri-X 320
at 16 (same sheet), and still Ektachrome E100 at 8 (Kodak E-4000, August
2018, *gross* density 1.0 - the sheets are not consistent about that).

**Not published at all.** Kodak's motion picture colour negative data sheets
give a "Diffuse rms Granularity" *curve* and no number, for every VISION3
stock and for cine Ektachrome 100D, and that has been true back through
VISION2 and the 1996 VISION 500T sheet. The VISION3 entries in `stocks[]`
(50D 3, 200T and 250D 4, 500T 6) are **estimates**, ranked by speed, and each
carries a `source` string saying so. Kodak's still colour negatives publish
Print Grain Index instead, which their own literature says "cannot be
compared to rms granularity"; Portra 400's PGI of 59 at 8x10 inverts through
Kane & Cookingham's published fit
(`Gi = 100.8*log10(sigma) - 44.3` at mean visual density 0.8, IS&T PICS 1998)
to a print granularity of 10.6, but that is a *print* number at 8.8x
magnification and the film-to-print step (Einhaus 1990) could not be
obtained, so Portra 400's 9 is an estimate too.

**Unknown.** The dye cloud diameter. 10-15 um is repeated everywhere and
could not be traced to any primary source. What is documented is 0.2-2 um for
individual silver halide grains (tabular grain patent literature), a
single-source claim of "often 25 microns" for dye clouds, and an unpinned
patent range of 3-20 um. The table uses 10-16 um by stock. **Every blob size
in this section scales linearly with that number**, so it is the single most
valuable thing to pin down.

Also worth recording: a search found **no publicly available AV1 film grain
preset calibrated to a named film stock** - not in grav1synth, not in
SVT-AV1's or rav1e's documentation, not in the community threads. The
"photon noise" tables everyone uses are a sensor shot noise model with an
ISO knob, not a film measurement.

### 13.7 Measured against intended

All at 1920 output, VISION3 500T, print response, measured on a flat 18%
patch that tone maps to an encoded 0.401. "want" is `craw_grain_sigma_at()`
evaluated at the level the patch actually landed on.

| gauge | want px | got px | want sd | got sd | ratio | chroma/luma |
|---|---|---|---|---|---|---|
| Regular 8 | 5.20 | 5.71 | 0.01089 | 0.01061 | 0.97 | 0.78 |
| Super 8 | 4.39 | 4.55 | 0.01092 | 0.01054 | 0.97 | 0.73 |
| 16 mm | 2.43 | 2.62 | 0.01090 | 0.01079 | 0.99 | 0.69 |
| Super 16 | 1.99 | 2.16 | 0.01088 | 0.01085 | 1.00 | 0.66 |
| 35 mm 2-perf | 1.13 | 1.15 | 0.01089 | 0.01115 | 1.02 | 0.62 |
| Super 35 | 1.00 | 1.00 | 0.01090 | 0.01111 | 1.02 | 0.62 |
| 65 mm 5-perf | 0.47 | 1.00 | 0.00518 | 0.00528 | 1.02 | 0.62 |
| IMAX 15-perf | 0.35 | 1.00 | 0.00388 | 0.00396 | 1.02 | 0.62 |

Amplitude within 3% everywhere, blob size within 10% except where the gauge
asks for less than a pixel and gets white noise, which is correct. The
residual few percent is seed-to-seed variation: a 64x64 template holds only
about `(64/blob)^2` independent blobs, so at the coarse end any one frame's
realised variance strays a few percent from the process's. `ar_eval()`
averages three seeds to take the bias out of the solve; the per-frame wander
is a property of the model.

Amount 0 leaves a standard deviation of 0.00002, i.e. nothing.

### 13.8 Open questions

- **The dye cloud diameter** (13.6). Everything scales with it.
- **The VISION3 granularity numbers** are estimates. Kodak's curves could be
  read off the data sheets properly - they are graphs of granularity against
  density, which would also give a measured response shape instead of the
  modelled ones.
- **Print and Reversal response shapes** were chosen by eye. A real print
  stock's transfer curve, cascaded with the negative's granularity curve,
  would replace both with something derived.
- **H.274's intensity intervals are placed linearly** in the stored signal.
  On the source path, where the signal is scene-linear, a steep response can
  land a sixth of a step out at the reference level. Placing the boundaries
  perceptually - which H.274 allows, they are arbitrary - would fix it.
- **The AV1 template's white noise floor.** The AR recursion always adds one
  unfiltered copy of the white noise on top of the smoothed part, so coarse
  grain carries a fine texture at roughly `1/gain` relative amplitude. It is
  visible in the amplified sheets as a faint cross-hatch over the blobs. Real
  film has a fine component too, so this may be a feature; it has not been
  compared against a scan.
- **`grain_preview` does not check temporal behaviour on a moving picture.**
  Boil is verified as two independent still frames. Whether a per-frame seed
  at 24 or 60 fps reads as film boil or as digital noise is a judgement that
  needs the viewer.
- **Frame mixing.** `PL_HOOK_PRE_OUTPUT` runs once per output frame, after
  mixing, so grain is not blurred between input frames. That follows from
  where `pass_output_target()` sits and has not been measured with
  `pl_render_image_mix`.
