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
- Camera profile model with per-CCT tables (as-shot exact vs re-balance),
  including a sensor noise model by exposure index.
- Raw domain denoising: non-local means on the four CFA planes in a
  variance stabilised domain, optional, per-frame strength. Section 14.
- Geometric lens correction: a per-channel radial resampling warp on the
  linear output, driven per frame by the camera's own
  `com.panasonic.Semi-Pro.optical_correction` record, which the library
  also parses (`<craw/lens.h>`). Optional, per-frame, and a no-op on a
  frame that carries no record. Section 16.
- CPU reference of every stage, tests comparing it against the GPU.

Out, deliberately, with a slot reserved: temporal denoising (section 14.7
says how it attaches), a second chroma-only denoise stage (14.4 says why
not yet), scaling, learned debayer, highlight reconstruction beyond the
clip, OCIO-generated transforms, LUTs, other CFA phases, and ARRIRAW and
BRAW profiles. Vignetting is out too, and unlike distortion it is out for
good on this camera: the optical_correction record carries no shading
term, because Panasonic applies shading correction in the camera before the
sensels are written. (The GUI, cineview, exists under tools/gui as a
consumer of the library: it is not part of the library's scope.)

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

Denoised mosaic (internal, only with `CRAW_PIPELINE_FLAG_DENOISE`):
`VK_FORMAT_R16_UNORM`, the same raw encoding as the input, written by the
denoise stage and read by the debayer in its place. Section 14.1.

Intermediate image (internal): `VK_FORMAT_R16G16B16A16_SFLOAT`, camera RGB
after prep and debayer, alpha unused.

Linear image (internal, only with `CRAW_PIPELINE_FLAG_LENS`): the output
format, holding what the colour stage produced before the lens warp reads
it. Section 16.3.

Output image: `VK_FORMAT_R16G16B16A16_SFLOAT` or `R32G32B32A32_SFLOAT`,
linear light in the requested colour space, alpha 1.0.

Border handling, all stages: mirror without repeating the edge sample and
preserving CFA phase: `x < 0 -> -x`, `x >= W -> 2W - 2 - x`, same for y.
The denoise stage applies that same rule to each CFA plane in the plane's
own coordinates rather than to the mosaic; see 14.1 for why.

## 4. Stage math

### 4.0 Denoise (optional, before everything)

Filters the mosaic in the raw domain and hands the debayer a denoised
mosaic in the same encoding. Defined in section 14; nothing else in this
section changes when it runs.

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

### 4.5 Lens correction (optional, last)

A per-channel radial resampling warp of the linear output, in the
destination-to-source convention, driven by the frame's own record. Defined
in section 16; nothing else in this section changes when it runs, and the
auto white balance histogram of 4.4 is two stages ahead of it and is not
affected at all.

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
  [fill histogram buffers], [denoise pass], debayer passes, [awb
  histogram], colour pass, [lens pass], [copy histogram to readback], and a
  final barrier that returns dst to the caller's stated layout. With the
  lens pass the colour pass writes the internal linear image and the lens
  pass writes dst; without it, or on a frame whose correction is invalid,
  the colour pass writes dst and no lens pass is recorded at all. The denoise pass
  dispatches one workgroup slice per CFA plane over a half resolution grid
  (`gl_WorkGroupID.z` selects the plane) and the debayer then reads its
  output image instead of the caller's source.
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
- `test_lens`: the ProRes RAW record parser, on real camera bytes stored in
  `tests/data/prr_lens_frame0.bin`, plus a real packet through libavformat
  when `CRAW_TEST_CLIP` names a clip. Section 16.5.
- `test_pipeline`: GPU vs CPU reference on a synthetic 256x160 mosaic and,
  when `TEST_SOURCE_DIR/../samples` or `CRAW_TEST_FRAME` provides one, on a
  real frame dump. Tolerances: bilinear/MHC 2e-3 absolute per channel on
  the camera RGB intermediate (fp16), 1e-2 relative on the colour output;
  RCD 5e-3 / 1.5e-2. The denoise stage is held to the same bounds, plus the
  checks of section 14.6 that it ran, that it removed noise and that it
  kept detail. The lens stage is held to 8e-3 absolute plus the checks of
  section 16.5 that it ran, that switching it off reproduces the plain
  chain bit for bit, and that a mark lands within 0.25 px of where the
  polynomial's inverse says it should. lavapipe is acceptable for these tests; it exposes
  everything required.

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
- Colour science, noise, lens records, reference, CPU tests:
  `lib/profile.c`, `lib/colorspace.c`, `lib/awb.c`, `lib/awb_smoother.c`,
  `lib/noise.c`, `lib/lens.c`, `lib/ref/ref_debayer.c`,
  `lib/ref/ref_color.c`, `lib/ref/ref_denoise.c`, `lib/ref/ref_lens.c`,
  `tests/test_profile.c`, `tests/test_ref_debayer.c`, `tests/test_awb.c`,
  `tests/test_lens.c`.

Anyone needing a change to a contract file reports it instead of editing.

## 13. Film grain emulation

`tools/gui/grain.c` and `grain.h` turn a gauge and a stock into parameters
for libplacebo's film grain shaders. `tools/grain/grain_preview.c` is the
headless proof: it renders a chart and a developed frame through the same
`pl_render_image` path cineview uses, measures what came out and writes it
to disk. Every number below is from that tool unless it is marked as a
citation, and `--calibrate` re-measures the two tables `grain.c` carries
against whatever libplacebo is installed.

Frame-sized output is uncompressed 16-bit RGB TIFF, one file per frame: the
measurements come from the pixels in memory, and deflating a megapixel of
noise into a PNG costs about thirty times what storing it does. Only the
contact sheets - crops assembled to be looked at, each under a megapixel -
are PNGs. To look at a frame, crop it first: `ffmpeg -i frame_super8.tif -vf
crop=640:480:560:300 -frames:v 1 crop.png`.

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

**Blob size beyond a few pixels -> a reduced resolution.** AV1's luma
template is one 82x73 array of which libplacebo uploads a 64x64 window, and
every 32x32 block of the picture is a window into it at an offset of 0 to 30
in each axis. At a one pixel blob that window holds thousands of independent
blobs and no two blocks look alike. At ten it holds about thirty-six, every
block shows nearly the same handful, and the picture visibly tiles - a
repeated stamp rather than grain.

Measured on a flat field through the shader, as the autocorrelation of the
applied grain at lags past the 32 pixel block grid, which should be zero:

| blob | rho(32) before | rho(64) before | rho(32) after | rho(64) after |
|---|---|---|---|---|
| 1.0 px | 0.001 | 0.000 | 0.001 | 0.000 |
| 4.4 px | 0.042 | 0.027 | -0.025 | 0.021 |
| 13.2 px | 0.176 | 0.111 | 0.066 | -0.012 |

Note the shape of the "before" column: a flat pedestal out to a lag of 128,
not a spike at the block period. That rules out the other candidate - block
offsets that are not varying - and identifies the cause as every block
sharing the template's own low frequencies. The one pixel row is the control:
if the offsets were constant it would tile too, and it does not.

So above about 2.5 px the grain is synthesised on a copy of the picture
reduced by an integer factor - 2 or 4, chosen so the blob is at most 2.5 px
there - and the field is upsampled bilinearly. The template then covers four
or sixteen times the film area and the filter no longer has to strain: at
13.2 px the model reaches 13.1, where at full resolution it saturated at
10.7.

The grain shader does not produce a grain field on its own - it samples a
picture, works out the amplitude from what it read, and adds - so the hook
runs three passes: reduce the picture, grain the reduced copy, then add the
difference between the two reduced textures to the full resolution picture,
sampled bilinearly. The amplitude the reduced pass is asked for is raised by
a measured factor (1.20 at 1/2, 1.145 at 1/4) because upsampling low-passes
the field; after that compensation the delivered standard deviation is within
0.1% of the intended one at both factors. The panel's readout says which
factor is in use, because it changes how the grain is made and not only how
it looks.

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
- **The residual correlation at a 13 px blob.** rho(32) is 0.066 after the
  fix rather than zero. Part of that is the blob itself - a 13 px blob has
  real correlation out to 16 px - and part is the reduced-resolution
  template's own pedestal, which lands at four times the lag. A factor of
  eight would push it further out at the cost of another halving of the
  template's resolution; it has not been tried.
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

## 14. Denoising

### 14.1 Where it sits

Sensor noise is a property of a single sensel: a Poisson count of
photoelectrons plus a signal independent read term, independent from one
sensel to the next and from one CFA site to the next. A debayer is a set of
interpolation kernels, so it mixes neighbouring sensels together and turns
that independent per-sensel noise into *correlated* noise with a colour:
MHC's chroma kernels have coefficients of 6/8 and -3/2, RCD's ratios
divide by low-pass sums, and both hand a single noisy sensel to three
output channels at once. Once that has happened the noise is no longer
described by any per-channel model, and a filter downstream has to guess.

So the primary stage runs **before the debayer**, on the mosaic, split into
its four CFA planes (R, Gr, Gb, B at half resolution each), each filtered
on its own and reassembled. Three things follow from that choice and all
three are wanted:

- The noise model is exact there. Nothing has touched the samples, so
  `var = a * u + b` holds with the sensor's own a and b.
- The four planes are statistically independent, so filtering them
  separately loses nothing, and the filter removes the *source* of the
  output's chroma noise rather than its consequences.
- Everything downstream is unchanged. The pass writes a denoised mosaic in
  the same `VK_FORMAT_R16_UNORM` raw encoding the decoder produced, so the
  prep of section 4.1, all three debayers, the AWB histogram and the colour
  stage read it without knowing it happened.

It runs before the prep rather than after it because the prep multiplies by
the white balance gains and clips at 1.0. The gains would scale the model
per channel (a -> a*g, b -> b*g^2, which is easy but pointless), and the
clip would truncate the very distribution the model describes. The stage
therefore works in

    u = (raw - black_level[s]) / (white_level[s] - black_level[s])

which is the prep of 4.1 with the gain and the clip removed, and writes
`raw' = black_level[s] + u' * (white_level[s] - black_level[s])` back. The
16-bit re-encoding costs a quantization step of 1/65535 of the sample
range against a read noise of one to two code values; it is the price of
leaving five other stages alone.

Green is filtered as two planes, Gr and Gb, rather than as one quincunx
lattice. They are the same colour filter but not the same pixels, and the
two-plane form keeps every plane a plain rectangular image with a plain
rectangular stencil. A shared-lattice green filter has more samples to work
with and is the obvious thing to try if green ever looks softer than red
and blue.

Borders are mirrored **in each plane's own coordinates**: `i < 0 -> -i`,
`i >= W/2 -> W - 2 - i`. That is the rule of section 3 applied to the plane
as an image in its own right. It differs from mirroring the mosaic and then
extracting the plane, which for the two phases with an odd offset repeats
the edge sample instead of reflecting past it; the plane rule is uniform
across the four phases, which is what a per-plane filter wants.

### 14.2 The noise model

    var(u) = a * u + b

`u` is the signal above black in units of the sensor range, exactly as
above, so a profile's (a, b) survives a decoder rescaling the sample range.
`a` is the shot term and `b` the read term. One pair describes all four
planes because the model sits before the white balance.

The model lives in the profile as a table by exposure index
(`CRAWNoiseKnot`, up to 16 knots, sorted ascending), evaluated by
`craw_noise_model_eval()`. Interpolation is linear in log(iso) of log(a)
and log(b), i.e. a power law between knots, because both terms are power
laws in the sensor's gain. Outside the table the nearest knot is used
**unchanged, not extrapolated**: a model that is too small under-denoises,
which the strength parameter can correct, while one that is too large
destroys detail that cannot be recovered. The lookup uses the frame's
as-shot exposure index, not its working one — rating a clip at a different
index scales the picture afterwards, it does not change how the sensels
were read.

#### Calibration

`craw_noise_estimate()` fits (a, b) to one frame:

1. Split into the four CFA planes.
2. High-pass each plane with `[1 -2 1; -2 4 -2; 1 -2 1] / 6`. That kernel
   annihilates any locally bilinear signal, so a smooth part of the scene
   contributes nothing but its noise, and its gain on white noise variance
   is exactly 1, so the residual's standard deviation *is* the sensel's.
3. Bin every sample by a 9x9 plane mean. The kernel sums to zero over its
   3x3 support, so any box mean whose window contains that support is
   exactly uncorrelated with it; 9x9 makes the mean's own noise a ninth of
   a sample's, which keeps samples from being dragged into the wrong bin.
   Bins are equally spaced in sqrt(mean), since the standard deviation is
   what they have to resolve.
4. Per bin, sigma = median|residual| / 0.6745.
5. Fit the line robustly: a least trimmed squares seed over every pair of
   bins, then iteratively reweighted Gauss-Newton on ln(variance) with a
   Tukey biweight, with a and b carried as logarithms so neither leaves the
   physical quadrant. The log domain is right because the variances span
   three decades and what matters is being right by a percentage at every
   level. The robust loss is what makes this work on an ordinary scene at
   all: texture only ever *adds* variance, so the bins it contaminates sit
   above the line and get discarded, never below it.

Bins are weighted by log(1 + samples), not by the sample count: a bin with
fifty times the samples is worth more, but not fifty times more, because
its error is dominated by how smooth that part of the scene is and not by
counting statistics.

`cineraw --noise-estimate` runs this on every decoded frame, prints
the per-frame fit and then the median over the clip, formatted as a
`noise ISO A B` line ready to paste into a `--tables` file. Two diagnostics
decide whether to believe it: `fit_rms`, the relative error of the model
over the bins it kept, which is 10-15% on a real scene and should be a few
percent on a flat field; and `read_noise_lsb`, sqrt(b) in 16-bit code
values, which above roughly 10 means the frame has no region dark enough
to measure b and what was fitted there is scene texture.

#### Panasonic DC-S1M2, measured

Frame 0 of every ProRes RAW clip this camera has produced so far:

    clip       ISO    a          b          bins  fit   read noise
    P1093961   1000   1.2225e-5  0          36    10.3%  0.00
    P1094127   1000   1.2923e-5  2.24e-10   29    12.4%  0.95
    P1093971   1000   1.3016e-5  0          23     5.0%  0.00
    P1093870   1000   1.3154e-5  0          37    12.6%  0.00
    P1000387   1000   1.3322e-5  5.23e-10   17    13.9%  1.45
    P1094644   1000   1.4028e-5  0          80    15.9%  0.00
    P1030266   1250   1.4578e-5  1.08e-9    29    13.0%  2.09

    rejected, texture in every bin:
    P1094643   1000   2.7935e-5  0         107    24.8%  0.00
    P1093969   1000   3.7812e-5  8.20e-7    50     8.8% 57.50
    P1093970   1000   6.7972e-5  0          34    11.7%  0.00

Six of the seven ISO 1000 clips agree to within +-7% of their median,
1.3085e-5, which is the knot. The three rejected ones are scenes with fine
texture at every level; the first is caught by its fit error and the second
by an impossible read noise, and the third by neither, which is exactly why
the shipped table is a median over clips and not one clip's fit. The b
column is a floor: none of these clips has a region dark enough to resolve
it, and it only matters below u = b/a, about five code values above black.
The shipped knots are

    1000 K:  a = 1.30e-5,  b = 1.0e-9     (sqrt(b) = 2.0 code values)
    1250 K:  a = 1.46e-5,  b = 1.1e-9

Two knots is not a characterization. The camera writes the same GainFactor
(23.04) at every ISO, which points at a fixed analog gain with ISO carried
as metadata; the two knots are consistent with that (a moving by 12% for a
25% ISO change) and equally consistent with a weak power law. The planned
ISO series — 640, 1000, 3200, 12800 on white paper filling the frame —
decides it, and `--noise-estimate` on those clips produces the table lines
directly. Until then the clamp keeps a high-ISO clip under-denoised rather
than over-denoised, and `--noise A,B` is the override.

One property of this camera's files worth recording: the four residual
operators tried during calibration (the 3x3 above, a lag-1 difference, a
lag-2 difference and a 1D second difference) agree to within a few percent
at every level, so the noise in a *decoded* ProRes RAW frame is white at
the scale the filter works on. The codec is not visibly reshaping the noise
spectrum, which is what lets a model fitted on decoded frames be used to
filter decoded frames.

### 14.3 The raw domain stage

#### The variance stabilised domain

One threshold has to work in the shadows and in the highlights, where the
noise differs by a factor of thirty. The generalised Anscombe transform
makes the variance constant so that it can:

    c    = 3 a^2 / 8 + b
    T(u) = (2 / a) * sqrt(a u + c)          for u >= 0

with `Var(T(u)) ~ 1` wherever `a u >> 3 a^2 / 8`. In this domain one unit
is one noise standard deviation at every level, so the filter's threshold
is a plain number.

Below u = 0 the square root has neither a real value nor a bounded
derivative, and read noise puts a real fraction of the samples of a black
region there. **The transform is therefore continued below zero by its own
tangent at zero:**

    sc   = sqrt(c)
    t0   = 2 sc / a
    T(u) = t0 + u / sc                      for u < 0

which is C1 at the join, exactly invertible, and correct in its own right:
for u < 0 the variance is b and constant, so a constant slope is what a
stabilising transform should have there. The inverse is

    I(t) = a t^2 / 4 - c / a                for t >= t0
    I(t) = (t - t0) * sc                    for t <  t0

The algebraic inverse is used rather than an asymptotically unbiased one.
The bias of the algebraic inverse matters below about ten Poisson counts;
the darkest content in these clips sits at 20 code values above black with
an `a` of 0.83 code values per code value, which is 25 counts, and a white
paper field at any of the calibration ISOs is thousands.

Arithmetic is fp32 throughout. The stabilised values are a large offset
(t0 is 200 to 600 for this camera) with a small useful range around it —
that is the whole point of the transform — so fp16 would quantize away the
signal itself. There is no fp16 fast path for this stage.

#### The filter

Non-local means on each plane, in the stabilised domain. For an output at
p, over a search window of radius S and a patch of radius P:

    d2(q) = (1 / (2P+1)^2) * sum over the patch k of (t(p+k) - t(q+k))^2
    w(q)  = exp(-max(d2(q) - 2, 0) / h^2)
    t'(p) = sum over q of w(q) t(q)  /  sum over q of w(q)

The 2 is `2 * sigma^2` with sigma = 1: two stabilised samples of the same
true value differ by a variance of 2, so that much of every patch distance
is noise and is subtracted before it can count against a neighbour. It also
makes `w = 1` for every neighbour that matches within the noise, the centre
included, so the denominator is never smaller than 1 and no guard is
needed.

`h` is the strength parameter directly. A neighbourhood that differs from
the centre by a true step of D noise standard deviations gets
`w = exp(-D^2 / h^2)`, so at h = 1 detail below one sigma — which is
indistinguishable from noise anyway — is averaged away, a two sigma step
keeps 1.8% of its weight, and anything larger is rejected outright. Larger
h smooths across larger true differences. This is what "strength 1.0
removes noise at the modelled level" means.

S = 3 and P = 1 (a 7x7 search, a 3x3 patch) on the reference device; see
the timings below. Both are specialization constants 2 and 3, so changing
them is a host-side constant and a rebuild of `lib/pipeline.c`, not a
shader edit. **P = 0 turns the same binary into the bilateral fallback**:
the patch distance is then the plain difference of two stabilised samples,
which is a range filter with the same noise-adaptive threshold.

One workgroup per tile of one plane, the plane index in
`gl_WorkGroupID.z`, so one dispatch covers all four. The workgroup loads
its tile plus a halo of S + P into shared memory, already stabilised, and
each invocation keeps its own patch in registers; every search position is
then one pass over the patch in shared memory. The shared array is a fixed
28x28 floats, which is a 16x16 workgroup plus a halo of 6, and
`lib/pipeline.c` static-asserts that the configured radii fit.

#### The highlight guard

Near the sensor's clip point the noise distribution is truncated and the
model does not hold, and pulling a clipped sample down would un-clip it in
one channel only — which is the thing the balanced clip of section 4.1
exists to prevent. The filter is therefore ramped off over u in
[0.90, 1.00]:

    g = clamp((u0 - 0.90) / 0.10, 0, 1)
    u_out = mix(I(t'), u0, g)

with u0 the original sample. Values above the white level clamp to 1.0 on
the way back into 16 bits, exactly as they would have.

#### Measured cost

RX 6900 XT, 5760x3040 RGGB, RCD, XYZ D65 out, back-to-back submissions
(12 reps, mean; the whole pipeline, and the pass's own share):

    search  patch   pipeline   pass
    (none)          4.26 ms    --
    7x7     1x1     4.98 ms    0.71 ms     bilateral fallback
    5x5     3x3     5.24 ms    0.98 ms
    7x7     3x3     6.12 ms    1.86 ms     <- shipped
    9x9     3x3     7.37 ms    3.10 ms
    7x7     5x5     8.72 ms    4.46 ms
    9x9     5x5    11.48 ms    7.22 ms
    flag set, strength 0       4.14 ms    0.00 ms

The budget is 4 ms and 7x7/3x3 spends 1.86 of it, which leaves room for the
temporal stage of 14.7. 9x9/3x3 also fits and is the first thing to try if
the search window ever looks too small; 7x7/5x5 does not. The bilateral
fallback was measured and is not needed. A pipeline built with the flag but
run at strength 0 costs nothing measurable, which is what makes the
strength a live control rather than a rebuild.

`cineraw --bench`, which decodes on the CPU between frames and so
leaves the GPU idle 97% of the time, reads 5.07 ms without and 7.55 ms
with, i.e. 2.48 ms: the same pass, measured on a device that is not being
kept busy. `test_pipeline`'s single real-frame submission, which follows
fifteen seconds of CPU-only reference work, reads about 4.7 ms for the same
pass. Real playback looks like the first table.

#### Measured quality

Frame 0 of P1000387.MOV, ISO 1000, a dim room with a ceiling spotlight,
model (a, b) = (1.30e-5, 1.0e-9).

A flat dark patch of 128x128 mosaic samples at (2500, 1000), per CFA plane,
as the standard deviation of the residual after a least squares quadratic
surface — which counts the slow blotches an NLM leaves behind and not just
what a high pass would see:

    plane  mean (lsb)  u        sigma   model   off   1.0    2.0
    R      380.0       0.00195  10.34   10.31   1.00  2.01   1.54
    Gr     499.2       0.00383  15.23   14.31   1.06  3.24   2.35
    Gb     499.0       0.00383  14.87   14.30   1.04  3.10   2.21
    B      372.8       0.00184  10.35   10.02   1.03  2.27   1.71

The undenoised measurement is within 0 to 6% of what the model predicts on
all four planes, which is the model validating itself on a frame it was not
fitted to. Strength 1.0 leaves 20-23% of the noise (4.4x to 5.0x, about
2.2 stops); strength 2.0 leaves 15-17% (5.9x to 6.7x). The patch mean moves
by 0.3 code values out of 380, so the stage is not biasing the level.

In the output (linear BT.709), over a 320x200 crop of the same region, the
same quadratic-residual measure:

               R         G         B         R-G       B-G
    off        9.465e-3  5.004e-3  5.910e-3  1.001e-2  6.048e-3
    1.0        2.356e-3  1.185e-3  1.526e-3  2.613e-3  1.805e-3
    2.0        2.045e-3  9.450e-4  1.243e-3  2.291e-3  1.537e-3

Against a patch mean of 0.096, chroma noise goes from 10.4% to 2.7% at
strength 1.0.

Detail survives. Over the LED's fresnel rings (a 600x450 crop at
(2800, 1800), concentric arcs at the sampling limit), the RMS of the same
3x3 high pass on green is 2.76284e-1 undenoised, 2.76274e-1 at strength 1.0
and 2.76237e-1 at strength 2.0: 99.996% and 99.98% of the detail energy.
Over the whole 1000x1000 crop containing the LED, the rim and the wood, the
99th, 99.9th and 100th percentiles of the green gradient magnitude are
unchanged to six figures (1.76414e1, 2.29227e1, 3.08944e1) while the median
gradient, which is the noise floor, drops 4.8x at strength 1.0 and 7.5x at
strength 2.0. The filter is removing the noise and leaving the edges.

#### Integer output: the terracing, and why it is not in the data

The first denoised crops of this frame showed the smooth dark gradient
across the lamp housing broken into plateaus tens of pixels wide with
jagged boundaries, where the undenoised crop is smooth under its noise.
Two mechanisms could produce that and both were measured; neither is what
happened.

*Re-quantisation of the mosaic.* The stage writes back into R16_UNORM, so
the worry is that the noise that used to dither those steps is gone and one
code value becomes a band. In a 200x200 plane patch of that gradient the
denoised mosaic occupies 125 of the 125 code values it spans with one gap,
at a mean run length along a row of 1.35 (the input: 165 of 179, run 1.03).
In the linear output the denoised frame occupies 1722 distinct 16-bit sRGB
codes at a run length of 1.03. Nothing has collapsed. Note also that this
camera's black and white levels already fill 96.9% of the 16-bit range, so
rescaling the encoding could buy at most 3%; the option that would actually
add precision is a float intermediate, and the numbers say none is needed.

*An NLM staircase.* With `exp(-max(d^2 - 2, 0)/h^2)`, neighbours inside a
slow gradient all match within the noise and all get weight 1, so the
filter degenerates to a box mean over the search window; if the weight set
then changed discretely from pixel to pixel the output would step. It does
not. The method noise `nlm - in` has an autocorrelation of +1.000 at lag 0
and within +-0.032 at every lag from 1 to 14, with nothing at lag 7, the
search window's own period; a staircase cannot hide from that. And the
filter is not eating the slow signal either: `G20(nlm) - G20(in)` is 0.17
to 0.21 code values rms against a method noise of 9.8 to 13.7, with a mean
of -0.2 codes on a level of 380, which is the Anscombe inverse's own bias
and 0.05% of the signal. Adding the low-pass of the method noise back would
restore that 0.2 and cost a pass; the measurement says not to.

*What it actually is.* Eight-bit rendering. The same 700x500 region of the
gradient, quantised to 8 bits:

    field                       mean run   longest run   error acf lag 1
    undenoised                  1.22       9             +0.001
    denoised, plain             3.10       66            +0.062
    denoised, dithered          1.91       25            +0.000

while the linear float field the three come from got *smoother*, its
laplacian rms falling from 2.83% of the mean to 0.58%. The denoiser did its
job; 8 bits is simply too coarse a grid for a smooth shadow gradient once
the sensor noise that was dithering it has been deliberately removed, and
the correlated quantisation error is the contour. Every quantiser has this
property and a denoiser is what exposes it.

So the fix belongs at the quantiser, not in the filter: `cineraw`'s
8-bit writer now adds one step of triangular dither, hashed from the pixel
position so the output stays reproducible. That takes the error's lag-1
autocorrelation from +0.062 to +0.000, which is the whole point of TPDF
dither, and raises the quantisation noise from 1/sqrt(12) to 1/2 of a step,
which at 8 bits is the trade anyone would take. On an undenoised frame it
changes nothing measurable (run length 1.22 to 1.21): the sensor noise was
already dithering. The 16-bit writer is left alone, both because its step
is 1/256 of an 8-bit code and never visible and because dither has to be
applied at the *last* quantisation to help: a consumer that takes a 16-bit
PNG down to 8 bits has to dither there itself.

The same applies to the viewer: anything that puts this library's output on
an 8- or 10-bit surface needs dither at that step, which libplacebo does by
default.

### 14.4 A chroma stage after the debayer: measured, not built

The argument for a second, lighter stage is that chroma noise is what
people see and that a chroma-only filter on a luma/chroma split at reduced
resolution is cheap. The measurement above says it is not needed yet, and
says by how much.

The raw stage reduces the output's chroma noise by 3.8x (R-G) and 3.4x
(B-G) against 4.2x for luma, so the chroma-to-luma ratio moves from 2.00 to
2.20 at strength 1.0 and 2.42 at strength 2.0. There is a residual — the
chroma error the stage leaves is slightly more correlated than the luma
error, because a 7x7 search on a half resolution plane reaches 14x14 full
resolution pixels and cannot see a blotch larger than that — but in
absolute terms chroma noise in a dim shadow is down from 10.4% of the level
to 2.7%, and the second stage would be chasing the difference between 2.7%
and perhaps 1.4%.

So: not built this round. What it would be, when it is:

- Between the debayer and the colour stage, on the camera RGB intermediate,
  not on the output. The intermediate is always RGBA16F whatever the output
  space, G is a good luma for a Bayer sensor, and R-G and B-G are the
  colour difference signals the debayer already builds its chroma in.
- Three dispatches. Downsample (mean of each 4x4 block of R-G, B-G and G
  into a quarter resolution RGBA16F image); a joint bilateral blur of the
  two chroma planes at quarter resolution with the range term on the
  downsampled G, radius 4, which reaches 36x36 full resolution pixels;
  then a recombine pass that bilinearly upsamples the filtered differences
  and writes R = G + d_r, B = G + d_b, blended by the chroma strength.
- Quarter resolution means the blur itself is 1/16 of the work and the only
  full resolution passes are a read and a read-modify-write, so the whole
  thing should land near 0.4 ms, well inside what 14.3 leaves.
- The luma/chroma split guarantees it cannot soften a luma edge; the range
  term on G is what stops chroma bleeding across one.
- It needs its own parameter, `denoise_chroma_strength`, and the CLI
  spelling is already reserved as `--denoise STRENGTH,CHROMA`.

The measurement to repeat before building it is the chroma-to-luma ratio
above, on an ISO 12800 frame from the planned series rather than on ISO
1000: the residual grows with the noise, and a stage that is not worth 0.4
ms at ISO 1000 may well be at ISO 12800.

### 14.5 Parameters and API

`CRAW_PIPELINE_FLAG_DENOISE` builds the pass and its extra full resolution
R16_UNORM image. Per frame, `CRAWFrameParams` carries:

- `denoise_strength`, the h of 14.3. 0 skips the pass for that frame
  without rebuilding the pipeline and is the default; the skip is free
  (measured above), which is what makes this a live control.
- `noise_override`, a `CRAWNoiseModel`. When its `a` is positive it is used
  as given; otherwise the profile's table is evaluated at the profile's
  as-shot exposure index, and a profile with no table makes
  `craw_pipeline_record()` fail with `CRAW_ERROR(ENOTSUP)` whenever the
  strength is positive. Failing is right: there is no default noise level,
  and silently doing nothing would look like a broken pass.

`CRAWProfile` grows `noise_knots` and `nb_noise_knots` at the end of the
struct, which changes its size; it is plain caller-filled data with no
padding tail, so this is an ABI break and the soname covers it.
`CRAWFrameParams` and `CRAWPipelineParams` keep their sizes, the new fields
coming out of their padding tails as intended.

`cineraw` gains `--denoise STRENGTH`, `--noise A,B`, and
`--noise-estimate`, and its 8-bit writer gains the triangular dither of
14.3. It also gains `--crop X,Y,W,H`, which writes and takes `--stats` over
one rectangle of the output while the pipeline still runs on the whole
frame: it is how a 5.8K frame gets looked at 1:1 without a 5.8K
file, and every number in 14.3 was measured through it.
`craw_tool_profile_load_tables()` accepts a `noise ISO A B` line, and a
file without one leaves a built-in profile's noise table alone.

### 14.6 Testing

`test_pipeline` covers the stage four ways.

- GPU against `craw_ref_denoise()` on the synthetic mosaic at strengths 1.0
  and 2.0, tolerance 2e-3 absolute on the camera RGB intermediate, which is
  the same bound section 11 sets for the non-RCD debayers. What separates
  the two is the order 49 terms are summed in, an ulp or two in exp(), and
  samples that land within an ulp of a 16-bit rounding boundary and
  quantize the other way; the measured difference is 9.8e-4, which is the
  MHC debayer's own floor and not the denoise pass's.
- That the comparison is sensitive at all. The largest error of this stage
  sits on hard edges, which the highlight guard hands through untouched and
  where a denoised and an undenoised frame agree exactly, so max abs alone
  would read the same if the pass had never run. The test therefore also
  measures the denoised GPU frame against the *undenoised* reference and
  requires that to be more than three times the floor; it measures 6.7e-3
  against a floor of 9.8e-4.
- That it denoises and that it preserves detail, both on the reference's
  own mosaic so that the numbers describe the definition rather than one
  implementation: the flat neutral patch of the synthetic scene must keep
  at most 60% of its high-pass noise (it keeps 5%), and the 2x2
  checkerboard, which is real detail at the sampling limit, must keep at
  least 70% of its high-pass energy (it keeps 100%).
- The two ways off is off (strength 0, and the flag not set) reproduce the
  plain chain, and a profile with no noise table and no override fails with
  ENOTSUP.

On a real frame, `test_real_frame` fits the model with
`craw_noise_estimate()` on the frame itself, runs RCD plus denoise at
strength 1.0 and holds the result to 1.5e-2 *relative* against the
reference — the bound section 11 states for RCD output, because the frame's
highlights sit far above 1.0 where fp16 storage alone is worth tens of
millis absolute. It measures 3.06e-3, against 1.35e-1 relative between the
denoised frame and the undenoised one.

### 14.7 Temporal denoising, when it comes

A spatial filter has one frame's worth of samples. A temporal one has as
many as the scene holds still for, and for the shadows of a locked-off shot
that is the difference between 4.5x and something much larger. It is the
obvious next stage and it attaches to this one, not to the output.

Where it goes: **inside the raw stage, in the stabilised domain, before the
spatial filter.** The domain is the whole reason — in stabilised units one
number is one noise standard deviation at every level and in every frame,
so the decision "is this sensel the same as last frame's" is a comparison
against a constant, exactly as the spatial patch distance already is. Doing
it after the debayer would mean deciding that in a domain where the noise
has already been mixed and coloured.

The state that persists across frames is one image: the previous frame's
*filtered* stabilised planes, a full resolution R32_SFLOAT image, 70 MB at
5.8K. It has to be fp32 for the reason 14.3 gives. The pipeline owns it,
which means `craw_pipeline_record()` stops being idempotent and the
"recorded once per submission" rule of section 11b becomes load bearing,
and a seek or a switch of source has to be able to say so.

The filter is a motion-adaptive recursive average, the cheap end of the
family:

    d     = |t_prev(p) - t_cur(p)|          in stabilised units
    alpha = alpha_max * exp(-max(d^2 - 2, 0) / ht^2)
    t_out = alpha * t_prev(p) + (1 - alpha) * t_cur(p)

which is the spatial weight of 14.3 with a patch of one sample and a search
of one position, so the threshold and the strength mean the same thing and
`ht` can be the same number as `h`. `alpha_max` below 1 bounds how long a
sample can survive, which bounds the smearing a missed motion detection can
cause; 0.8 gives an effective 5 frames on a static scene and forgets a
wrong decision in three. A patch distance over 3x3 instead of a single
sample would make the detection far less trigger-happy at the cost of a
second tile load, and is the first thing to try if it ghosts.

Motion compensation — searching the previous frame over a window instead of
reading it at p — is the expensive end and needs a real motion estimate,
which this library does not have and should probably take from the caller
rather than compute.

Resetting. The state is wrong whenever the previous frame is not the
previous frame of the same scene, and there are three cases:

- The caller knows: a seek, a new clip, a jump in the pipeline's own frame
  counter. A `CRAW_FRAME_FLAG_TEMPORAL_RESET` makes the pass ignore the
  state for one frame (alpha = 0) and it refills from that frame on.
  `craw_awb_smoother_push()` already takes a `scene_cut` flag for the same
  reason and the two should be spelled the same way.
- A cut the caller does not know about. The pass already computes `d` at
  every sensel; the fraction of sensels with `d^2 > 2 + k` is a scene
  change detector for free, and a small reduction buffer next to the AWB
  histogram would hand it back to the host, one number a frame, to be read
  the frame *after* — the same one-frame lag the AWB estimator already
  accepts. On a cut alpha collapses to 0 by itself everywhere at once, so
  the detector is a diagnostic and a way to reset the *spatial* strength,
  not a correctness requirement.
- Any frame the pipeline was not given the previous frame for. The state
  image starts undefined; a `valid` flag in the pipeline, cleared on
  creation and on reset, keeps the first frame from reading it.

What it costs: one extra full resolution fp32 image read and written, plus
about ten instructions per sensel, so well under 0.5 ms — the state image's
bandwidth is the whole cost. That fits in what 14.3 leaves of the 4 ms.

## 15. Soft light and vibrance

`tools/gui/adjust.c` and `adjust.h` are a clean-room reimplementation of two
RawTherapee tools the owner reaches for, plus the move of the film
simulation LUT that had to happen for them to sit in the right order.

RawTherapee is GPL and this is BSD-2, so nothing of its code, comments or
tables is here. What was taken is the mathematics and the parameter
semantics, established by reading RawPedia and the two engine files, and
then written from scratch. Where the behaviour differs it is said so below,
with the reason.

### 15.1 Where the four hooks sit, and why the LUT moved

RawTherapee applies film simulation in its RGB stage, vibrance in the Lab
stage after it, and soft light late in that same Lab stage. Order matters:
vibrance on a LUT'd picture is not the picture a LUT on a vibrant one is.
The viewer now matches it, with the film grain last because grain is a
property of the recording medium and everything above it is a property of
the grade:

    film simulation LUT -> vibrance -> soft light -> film grain

All four are `pl_hook`s at `PL_HOOK_PRE_OUTPUT`, in one array, and
libplacebo runs the hooks of a stage in array order (`pass_hook()` walks
`params->hooks` forward), so the array *is* the order.

The LUT had to move to make that possible. It used to ride on
`pl_frame.lut` of the target frame, and libplacebo applies a target
`PL_LUT_NATIVE` at the very end of `pass_output_target()` - after every
`PL_HOOK_PRE_OUTPUT` hook has run. Nothing could ever be put in front of it.
So the two `target.lut` blocks in `gpu.c` are gone and `craw_adjust_lut()`
is the first hook instead.

Nothing is lost by the move, and this is checkable rather than merely
argued. Between the `PL_HOOK_PRE_OUTPUT` hook point and the place the target
LUT used to be applied, libplacebo runs exactly one thing:
`pl_shader_encode_color()`. That function's own `skip` test is true for an
RGB system at full range whose sample and colour depths agree and whose bit
shift is zero - which is what both the window and the 16-bit export target
are - so it emits nothing at all. The hook therefore holds the same values
the target LUT used to be handed, and it calls the same
`pl_shader_custom_lut()` with a `pl_shader_obj` of its own, so the same
tetrahedral interpolation runs over the same uploaded texture.

Measured: frame 0 of `P1000387.MOV` exported with `--lut lively-seaside` is
**byte for byte identical** before and after the move (md5
`b0ee990691898307bd171c5bb1022ed0` both ways), and so its PSNR against the
`ffmpeg -vf lut3d=...:interp=trilinear` reference is unchanged at **60.53 dB**
(per channel 63.15 / 64.13 / 57.48). The export with no LUT at all is also
unchanged (`4a1630379770e63a6a8b90a18f8ab0c2`).

A user `.cube` from `--lut` is untouched by all this: it still goes on the
*image* through `pl_render_params.lut`, where it always did, because that
one is authored against the source and not against the display.

### 15.2 Why `PL_HOOK_SIG_COLOR`

All three new hooks are pure per-pixel functions - no neighbour is read, no
resolution changes - so they take `PL_HOOK_SIG_COLOR` and append their
instructions to the shader the renderer is already building.
`PL_HOOK_SIG_TEX`, which the film grain needs because it does read a
neighbourhood, would instead force the renderer to finish that shader into
an intermediate texture and start another: one FBO round trip per hook, plus
the quantisation of storing a display-encoded value in it. That is also the
second half of why the LUT is still exactly as correct as it was.

### 15.3 Soft light

RawTherapee uses the **Pegtop** soft light blend, which for a backdrop `a`
and a blend layer `b` is

    S(a, b) = 2*a*b + a^2 * (1 - 2*b)

This is the continuous approximation to the W3C/SVG `soft-light`
compositing operator: same endpoints, same behaviour, but without the
piecewise branch at `b = 0.5` whose derivative is discontinuous. Pegtop
published it for that reason.

The picture is blended with *itself*, so `b = a` and the whole operator
collapses to one cubic in one variable:

    S(a, a) = 2*a^2 + a^2*(1 - 2*a) = 3*a^2 - 2*a^3 = a^2 * (3 - 2*a)

which is the smoothstep polynomial. It fixes 0, 0.5 and 1, has zero slope at
both ends, darkens below mid grey and lightens above it. Applied per channel
it raises contrast and, because the channels move by different amounts,
saturation with it.

This is worth pinning down because the variants agree on half the domain:
for `v <= 0.5` Pegtop, W3C and the Photoshop/illusions.hu formula all give
`3v^2 - 2v^3`, and they only diverge above it, where W3C and Photoshop
introduce a square root. The discriminating test point is `v = 0.75`:
Pegtop gives **0.84375**, the other two about **0.808**. A correct
implementation has no branch on 0.5 and no `sqrt`.

Strength is a percentage and enters as a plain mix:

    out = a + (strength/100) * (S(a, a) - a)

so 0 is the original and 100 the full blend. Default 0, i.e. off.
(RawTherapee's slider defaults to 30 but its tool starts disabled, which is
the same picture by a longer route.)

**Out of range.** RawTherapee does not clamp: a channel outside [0,1] -
after dividing by its 65535 working scale - **bypasses the effect entirely
and is passed through untouched**, and the decision is per channel, so a
pixel can have one channel blended and another not. That is reproduced here,
branch-free, as

    ok  = step(0, v) * step(v, 1)
    out = v + ok * (strength/100) * (v^2*(3 - 2*v) - v)

Clamping instead would drag out-of-gamut channels onto the boundary, which
is exactly where doing so invents a colour. At this hook point the picture
is display-encoded and mostly inside [0,1] anyway, so the guard costs two
instructions and matters only at the gamut edge and in clipped highlights.

**Where it differs from RawTherapee.** RawTherapee does the blend in
gamma-encoded working-space RGB and then mixes the result against the
*linear* original; here both the blend and the mix happen in the
display-encoded values the hook stage holds. The blend domain is the same
kind of signal either way - RawTherapee's gamma is an sRGB-shaped curve with
slightly shifted breakpoints (slope 12.92310, encode knee 0.003040) over the
user's working primaries, ours is the target's own transfer function over
the target's primaries - and at strength 0 and 100 the two are identical by
construction. Only the intermediate strengths differ, and mixing in the
encoded domain is what a soft-light layer at N% opacity does in every
display-referred editor. It is also what the hook stage can do without a
linearize/delinearize round trip it has no other use for.

Verified against the analytic formula: the 1024x1024 crop exported at
strength 100 matches `v^2*(3 - 2*v)` to a maximum of **1.25 LSB of 65535**
(mean 0.31, 103.7 dB), and at strength 50 to 1.12 LSB (103.1 dB). That is
16-bit quantisation and nothing else.

### 15.4 Vibrance

RawTherapee's vibrance works in CIE L\*a\*b\*, where a line of constant Lab
hue angle is *not* a line of constant perceived hue - the blue region bends
badly - so scaling chroma at fixed hue angle visibly rotates the hue, and it
has to carry a Munsell hue-correction dataset to put it back: on the order
of 195 one-dimensional tables of up to 140 entries, selected through a
nested cascade on lightness and hue leaf. That dataset is RawTherapee's, it
is GPL, and it cannot be reimplemented without equivalent data.

This implementation works in **Oklab** instead. Oklab was fitted to keep
lines of constant hue straight, which is precisely the property the Munsell
tables exist to restore, so scaling the `(a, b)` pair by a scalar already
leaves the hue where it was and no table is needed. That is not a
substitution of convenience: it is the one change that makes a from-scratch
version of this tool possible at all.

Chroma and hue are the polar form of Oklab's `(a, b)`:

    C = sqrt(a^2 + b^2),   h = atan2(b, a)

and the whole filter is one number, the factor `C` is multiplied by. `L` and
`h` are never touched, so the result is lightness- and hue-preserving by
construction. Measured on the exported crop: max `|dL|` **0.00003** and mean
`|dh|` **0.006 degrees** (max 0.24, at chroma so low the hue is
quantisation noise).

The Oklab matrices are kept in their XYZ form and composed at run time with
the *target's* own RGB<->XYZ matrices, rather than baked against sRGB, so a
BT.2020 swapchain gets the right hue angles instead of silently wrong ones.
That costs two 3x3 multiplies once per frame. It is exact for any D65
primaries; a target on a different white point would want a chromatic
adaptation that is not there, and no target the viewer can reach today has
one.

**Parameter mapping.** The two sliders run -100..100 and enter as
`k = slider/100`, which is RawTherapee's own mapping - not an exponential.
The chroma factor is `1 + m` where `m` is `k` after the skin factor and the
limiter:

    m = clamp(k * r, -0.93, +2.00)
    C' = C * (1 + m)

so +100 doubles chroma, -100 cuts it to 7% of itself, and 0 is exactly 1.
The `-0.93` floor is RawTherapee's and is worth keeping: it stops a full
negative slider collapsing a colour onto the neutral axis and losing its hue
with it. (RawTherapee's positive limit differs between the two regimes, 2.0
and 1.8. Without its 216-constant hue/lightness weighting table - see below
- `m` can never exceed 1, so neither limit is reachable and only the floor
is live.)

**What is deliberately not reproduced.** RawTherapee multiplies the slider
amount by one of 216 hand-tuned constants chosen by a hard branch on four
lightness bands and six hue sectors, asking for more gain in hues that
under-respond and less in hues that clip. Those constants are its data and
its expression, so they are not here; the hard branches are also its single
most visible artefact source, since nothing interpolates between adjacent
bands or sectors. The cost of leaving them out is that the response is
uniform across hue rather than perceptually weighted. In Oklab a good part
of what that table compensates for - Lab's uneven chroma scale across hue -
is already handled by the space.

**The threshold.** RawTherapee's pastel/saturated crossover is driven not by
chroma but by the RGB saturation `s = 1 - min(R,G,B)/max(R,G,B)` of the
gamut-mapped linear RGB, and its 9-node piecewise-linear response has its
crossover at `X = 0.07 + 0.93*T` for a threshold `T`. Here the abscissa is
Oklab chroma, which is the natural quantity in the space the filter already
works in and is perceptual where RGB saturation is not.

The default was calibrated rather than guessed. Over 400k random sRGB
colours, RawTherapee's default crossover of `s = 0.7675` (its threshold 75)
falls at a **median Oklab chroma of 0.135**, which is 45% of the 0.30 that a
saturated sRGB primary reaches. So:

    C_ref = 0.30                       (measured: the six sRGB primaries
                                        and secondaries run 0.15 to 0.32)
    C_t   = (threshold/100) * C_ref
    t     = smoothstep(0.75*C_t, C_t + 0.25*(C_ref - C_t), C)
    m     = mix(k_pastel, k_saturated, t)

and **threshold 45 is the default**, putting the crossover where
RawTherapee's default 75 puts it. Each regime hands a quarter of its own
span to the transition, which is the same geometry RawTherapee uses
(quarter-width sub-intervals either side of the crossover) expressed in
chroma. The transition is a smoothstep rather than RawTherapee's linear
ramp: a corner in the response is visible as a contour in any gradient that
crosses it. At the crossover the smoothstep gives `T^2(3-2T)` of the way
from the pastel amount to the saturated one where RawTherapee gives `T`;
at the default that is 0.84 against 0.75.

For reference, the chroma these land on: pastels measure 0.04 (pale sky) to
0.07 (light skin), mid foliage 0.13, a vivid sky 0.17, the primaries 0.15 to
0.32. The two regimes fall either side of 0.135.

**Protect skin tones.** RawTherapee classifies skin with a first-match
cascade of axis-aligned boxes in (L, hue, chroma) across seven lightness
bands, emitting one of three discrete reduction factors - 0.3 for its core
region, 0.4 and 0.6 for two wider envelopes. Hard boxes with no feathering
are the other visible artefact source in that tool.

Here it is a smooth window, and its geometry comes from measuring real skin
in Oklab rather than from RawTherapee's table. Reference skin swatches -
light and dark ColorChecker skin, pale, mid and deep complexions - come out
at hue **28 to 72 degrees** and chroma **0.05 to 0.12**. So the window is a
plateau over 25..75 degrees with a smoothstep roll-off of 15 degrees at each
edge, times a chroma ceiling that tapers from full at 0.12 to nothing by
0.18:

    w = smoothstep(10, 25, h) * (1 - smoothstep(75, 90, h))
                              * (1 - smoothstep(0.12, 0.18, C))
    m = m * (1 - 0.7 * w)

The chroma ceiling is what keeps a saturated orange from being mistaken for
a cheek; RawTherapee bounds its boxes in chroma for the same reason. The
0.7, i.e. skin keeping 30% of the effect, is RawTherapee's core factor, and
is worth matching because it is the one number about this control a user
carries over.

It defaults **on** here, where RawTherapee's defaults it off. Without the
Munsell table a hue window is all the protection there is, and it costs
nothing when there is no skin in the frame.

One consequence is worth stating plainly, because it shows up in the test
export: **warm low-chroma wood, leather and skin are the same colour**. The
sample frame is a wood ceiling at hue 50-65 degrees and chroma 0.03-0.09,
squarely inside the window, so with protection on `--vibrance 50,20` raises
its chroma by 1.15x rather than 1.5x. That is the feature working, not
failing, and RawTherapee's classifier would say the same thing about the
same pixels.

**Avoid colour shift.** In RawTherapee this toggle switches on the Munsell
hue correction, and a gamut-compression loop runs either way. In Oklab hue
constancy is free, so the toggle here governs only the gamut half, which is
the half the brief asks for: rather than letting the encoded RGB clip,
the chroma scale is walked back until the colour fits.

Clipping is the wrong answer because clipping one channel and not the others
*is* a hue rotation. Walking the scale back keeps `L` and `h` exactly - every
candidate on the walk has the same hue by construction - and gives up only
the chroma that could not be shown.

The walk is eight bisections on the scale, between 1 (always safe, since the
colour came from an in-gamut display value) and the requested factor, with
the invariant that the low end always fits. The requested factor is tested
once first and the walk only runs where it did not fit: bisection converges
on the boundary from inside, so a colour that was always going to fit would
otherwise come back a 256th of the interval short and the filter would
quietly under-deliver everywhere rather than only at the edge. The gamut
test is against the linear RGB in [0,1], which is equivalent to testing the
encoded RGB because every transfer function involved is monotonic.

Eight steps resolve the interval to a 256th of it, below the quantisation of
any window this lands in. Measured on the exported crop: the whole shader,
including the Oklab round trip, the threshold, the skin window, the limiter
and the walk, matches an independent model of the same algorithm to a
**mean chroma error of 0.026%** (max 2.1%, at chroma near zero). Pixels the
walk does not touch land on the asked-for factor exactly - 1.1498 measured
against 1.150 intended for the protected wood, 1.4997 against 1.500 for the
unprotected lamp body.

**Not implemented: the user curve.** RawTherapee has a skin-tone curve, a
hue-to-hue diagonal curve applied inside a fixed band with chroma and hue
feathering, at a fixed half strength. It is skipped in this round and
RawTherapee itself short-circuits the identity case, which is the default,
so nothing is missing at defaults. The hook for it is the place `m` is
computed in `adjust_vib_head`: a curve would enter as a 1D LUT sampled by
the normalised chroma position `t`, replacing the `mix()` there, and the
parameter would join `craw_vibrance_params`.

**Defaults are an absence, not an identity.** `craw_vibrance_is_noop()` is
true when both amounts are zero, and the caller then leaves the hook off the
array entirely rather than dispatching a shader that multiplies by one; the
same is true of soft light at strength 0 and of the LUT when no film
simulation is picked. So at defaults the render is instruction for
instruction what it was before this change, which the unchanged export md5
confirms.

This is also a deliberate improvement on RawTherapee, where the all-zero
guard is the *only* per-pixel identity path: move a vibrance slider there by
1 and every pixel in the image additionally goes through a full gamut
compression and a Munsell correction, so the step from 0 to 1 is far larger
than 1/100 of the effect. Here the gamut walk belongs to vibrance and scales
with it.

### 15.5 Cost

Both hooks are a few dozen ALU instructions per pixel appended to a shader
that was going to run anyway, with no extra pass and no extra render target.
`-v`'s per-frame `render` cannot resolve them. Over three 22-second
`--bench` runs each, at 1080p output: **13.90 ms** with the hooks off
(13.849/13.909/13.934) against **13.40 ms** with `--softlight 50 --vibrance
50,20` (13.290/13.324/13.582); at 4K output, 14.16 ms off against 13.80 ms
on. The difference is negative both times, i.e. the loop is decode-bound on
5.8K ProRes RAW, `render` is measuring GPU queue backpressure rather than
shader work, and the added instructions fit entirely in the slack. The
run-to-run spread, with other GPU work on the machine, is larger than
anything the hooks could cost.

## 16. Lens correction

Panasonic's cameras write a geometric lens correction into every ProRes RAW
frame, in the vendor metadata after the last tile, and it changes frame by
frame as the lens zooms and focuses. `tools/prr_lens.md` is the derivation
of the record from the bytes of real clips, from Apple's decoder and from
Panasonic's own in-camera block in RW2 stills; this section is what libcraw
does with it. Nothing below re-derives any of it.

### 16.1 The record, and where it is read

Frame layout, as `libavcodec/prores_raw.c` walks it: a be32 size, `prrf`, a
be16 header length, the header body, `nb_tiles` be16 tile sizes, the tile
data, and then a list of vendor metadata records that `eomd` ends. Each
record is a be32 size counting itself and its fourcc, the fourcc, and for a
`psim` record a Pascal-string key and the payload. The one that matters has
the key `com.panasonic.Semi-Pro.optical_correction` and a payload of a be32
version — 3 in every frame seen — followed by twenty big-endian doubles:
three channel blocks of six in the order R, G, B, each `k0 k1 k2 k3` and
two slots that are exactly +0.0 everywhere, then the optical centre, which
is exactly (0.5, 0.5) on every frame of every clip.

`craw_lens_from_prores_raw()` in `lib/lens.c` does that walk on a demuxed
packet and fills a `CRAWLensCorrection`. It derives the trailer's offset
from the header and the tile table rather than searching the frame for the
key string, because a search would also find those bytes inside a tile and
would say nothing about whether what follows them is a record. Every read
is bounds checked against the frame's own size field, which is itself
checked against the packet. No tile is decoded and no pixel is read; on a
5 MB frame it is a few microseconds.

It lives in the library rather than in either tool because both tools need
it, and it is the only part of libcraw that looks at a bitstream at all —
it is a metadata walk, not a decoder, and it is kept to that: it never
touches Vulkan and never allocates.

Three answers come back. A record found and decoded is 0. `ENOENT` is a
frame that carries no such record, which covers both a clip from another
camera whose metadata list simply lacks one and a frame with no list at
all: an FX6 recorded to an Atomos Ninja V ends its frames with a few dozen
bytes that are neither records nor zero and contain no `eomd` anywhere, and
"this frame has no record" is the useful answer there. `EINVAL` is kept for
a packet that is not a walkable ProRes RAW frame, and for an
optical_correction record that is present but the wrong shape — an unknown
version, a payload that is not 164 bytes, a coefficient that is not finite,
a centre outside the frame. On every failure the output is zeroed, i.e.
left invalid, so a caller that only wants "correct it if you can" can
ignore the return value entirely.

The record's doubles are narrowed to float in `CRAWLensCorrection`. At
r <= 1 with these magnitudes that is under a thousandth of a pixel on a
5.8K frame, and both the shader and the CPU reference evaluate in fp32
anyway, so carrying doubles would only move the rounding somewhere less
visible.

### 16.2 The model, and the coordinate conventions

Destination to source, per channel, about the optical centre:

    r_src = r_dst * (k0 + k1 r_dst^2 + k2 r_dst^4 + k3 r_dst^6)

For an output pixel at normalized radius `r_dst` from the centre, the
source is sampled on the same ray at `r_src`. The three channels share the
ray and differ only in the polynomial; that difference is the lens's
lateral chromatic aberration, and it runs to about two pixels at the corner
on the 24-60.

The conventions, spelled out because everything in the stage depends on
them:

- **The frame is the coded frame.** The pipeline develops the whole readout
  — the decoder is told not to crop (`apply_cropping = 0` in both tools),
  the debayer starts at sensel (0, 0), and the picture the frame header
  marks inside the readout is a *view* of the result, applied after the
  pipeline by `--crop`. The record normalizes against the frame the camera
  read out, so the stage does too: a `CRAWPipeline` created for
  `width x height` uses exactly that. If a future caller ever hands the
  pipeline a pre-cropped frame, the record would have to be re-normalized
  first, and that is the caller's problem, not the stage's: the stage has
  one frame and it is the one the pipeline was created for.
- **The centre is in normalized frame coordinates**, so in pixels it is
  `(cx * W, cy * H)` in a continuous coordinate system where integer pixel
  `p` sits at `p + 0.5`. With the camera's (0.5, 0.5) that is `(W/2, H/2)`,
  the exact geometric centre, and the geometry is symmetric about it.
- **r = 1 is the frame corner** as seen from that centre, i.e. the radius
  is normalized by `hypot(max(cx, 1-cx) * W, max(cy, 1-cy) * H)`. For a
  centred optic on 5760x3040 that is 3256.5 px. Both the shader and
  `craw_lens_corner_shift()` take the farther side on each axis, so an
  off-centre record would still put r = 1 at the corner that is actually
  farthest away.
- **The centre and that radius are computed on the host**, once per frame,
  and reach the shader as pixels and a reciprocal. A shader that recomputed
  them would be doing the same divide seventeen million times.

`k0` is not distortion; it is the crop. A pincushion correction pushes the
corners outward and would sample off the sensor, so the camera scales the
whole frame down by just enough to stop that: `k0` falls to 0.967 at the
long end of the 24-60 and sits at 0.999 at the wide end, where a barrel
correction samples inward anyway and no crop is needed. Two numbers follow
from the polynomial and `craw_lens_corner_shift()` returns either:
`poly(1) - 1` times the half diagonal is the *net* corner shift, a handful
of pixels, and `poly(1)/poly(0) - 1` times the half diagonal is the
distortion that is actually being corrected — -207 px at 24 mm, +100 px at
60 mm, +110 px on the Sigma 135. The second is the one to show a person.

Because `poly(1)` lands just below 1.0 on every frame measured, the source
position never leaves the frame; the border rule below exists for the
kernel's footprint, not for the sampling position.

### 16.3 Where the stage sits

After the colour pass, on the linear output image, and last. Section 1
originally reserved it as "a per-channel resampling warp after the debayer
and before any scaling", and after the colour pass is the earliest point
that is true of: the warp needs the three channels in the same image, which
is anything after the debayer, and it must not run on the mosaic, where
resampling a CFA would be meaningless. Running it after the colour pass
rather than on the camera RGB intermediate also means it resamples the
values the caller is going to look at — one interpolation, in the space the
picture is finally in, instead of one before a matrix that would spread its
error across channels.

With `CRAW_PIPELINE_FLAG_LENS` the colour pass writes an internal image and
the warp writes the caller's destination; without the flag nothing changes
and the colour pass writes the destination as it always did. The internal
image is created in the pipeline's own `out_format`, so RGBA16F by default
and RGBA32F when the caller asked for 32-bit output: a caller who asked for
32 bits asked for them all the way, and an internal image is no place to
quantize them away.

The auto white balance histogram is not affected. It reads the *camera RGB*
intermediate (section 4.4), two stages ahead of the warp, so it describes
the frame the camera saw and not the frame the lens correction produced.
That is the right way round: the estimator's locus is the camera's, and a
geometric resample of the picture does not change what illuminant lit it.

`CRAWFrameParams` carries a `CRAWLensCorrection` **by value**, not by
pointer. The record is different on every frame of a clip that is zooming,
and a player with several frames in flight must be able to keep each
frame's parameters self-contained; a pointer would mean the parameters
outliving whatever the panel or the decoder owned. A correction with
`valid` false means "leave this frame alone", and it is not a cheap copy
but no pass at all: the warp drops out of the recording and the colour pass
writes the caller's destination, so the frame is **bit for bit** what a
pipeline built without the flag produces. That is what makes the switch a
live control and what makes a clip with no record cost nothing.

### 16.4 Sampling

Sixteen tap Catmull-Rom bicubic, per channel, at the channel's own source
position. Catmull-Rom is the Mitchell-Netravali family at B = 0, C = 1/2:
it interpolates, it is C1, and its weights sum to one for every fractional
offset, so a flat field comes through flat.

The warp is within a few percent of the identity everywhere — a couple of
pixels of displacement over most of the frame, a couple of hundred at the
corner of a 24 mm frame — and that is exactly why the interpolator has to
be good. At a displacement of a fraction of a pixel a bilinear resample is
a low pass filter applied to the whole picture, and it would take out the
detail RCD was chosen to keep.

Measured, on the fresnel rings of the lamp in P1000387.MOV frame 0, which
is where the finest structure in that frame sits: a 512x512 crop at
(2750, 1850), local scale 0.9742, warped by the frame's own record twice,
once with these sixteen taps and once with plain bilinear. High-pass
energy, the squared 3x3 Laplacian of green summed over the crop:

    unwarped source   2.140e4
    bicubic warped    1.717e4     80.3% of the source
    bilinear warped   1.324e4     61.9% of the source

So bilinear throws away **22.9%** of the detail the bicubic keeps. Between
the two warped crops that is a 0.92% RMS difference, with the worst sample
6.4% of the crop's peak and 7.5% of samples differing by more than 1% of
it — all of it concentrated on the ring edges, which is exactly where it
shows. Side by side at 3x it is not subtle: the dark rings come out wider
and lower in contrast, and the finer secondary rings wash out. (Both
numbers being under 100% is the warp's own 2.6% shrink at that radius,
which no interpolator can undo; the comparison between the two is the
point.) Bicubic costs three times the taps, and the whole pass is 0.8 ms
(16.6), so it is worth paying for.

Borders follow the mirror of section 3, `i < 0 -> -i` and
`i >= n -> 2n - 2 - i`, one reflection. Only the outer ring of the 4x4
footprint can ever reach past an edge, because the sampling position itself
stays inside the frame (16.2), so the rule is there to keep the shader
total rather than to shape the picture.

Alpha is **not** warped: it is read from the output pixel's own position.
The colour stage writes 1.0 and it stays exactly 1.0, rather than becoming
the sum of sixteen weights that agree to the last bit only in exact
arithmetic.

The geometry is evaluated once per pixel and the polynomial three times:

    d   = (p + 0.5) - centre_px
    r2  = dot(d, d) * inv_half_diag^2
    f_c = k0 + r2 * (k1 + r2 * (k2 + r2 * k3))          per channel
    s_c = centre_px + d * f_c - 0.5                     pixel index space

`r2` rather than `r` because nothing in the model needs the square root.

### 16.5 The CPU reference and the tests

`craw_ref_lens()` in `lib/ref/ref_lens.c` is the same arithmetic in the
same order, in fp32, with the same border rule; `craw_ref_process()` runs
it when the pipeline parameters carry the flag and the frame's correction
is valid, which is the condition `craw_pipeline_record()` applies.

`test_lens` (CPU only, always runs) parses the record. Its frame,
`tests/data/prr_lens_frame0.bin`, is frame 0 of P1000387.MOV with the tile
*payload* removed: the first 8836 bytes verbatim — the size field, `prrf`,
the 88 byte header and all 4370 be16 tile sizes — followed by the 232 byte
trailer verbatim, 9068 bytes instead of 5133984. The test puts
`sum(tile sizes)` zero bytes back where the tiles were, so every byte the
parser reads is the camera's own and the whole walk runs on it. It checks
the twelve coefficients against `prr_lens.md` section 5 bit for bit, the
corner shifts to 0.01 px, that every prefix of the frame is refused rather
than read past, and each way of breaking the frame. With libavformat
available and `CRAW_TEST_CLIP` set it parses a real packet too.

`test_pipeline` covers the stage three ways.

- GPU against `craw_ref_lens()` on the synthetic frame, with the real
  P1000387 record and with an exaggerated one, tolerance 8e-3 absolute. The
  warp is a weighted sum of sixteen fp16 samples whose weights sum to one
  but whose absolute values sum to as much as 1.25 per axis, so it carries
  at most 1.56 times the error of the image it reads; against the 5e-3
  section 11 allows the RCD intermediate that is 7.8e-3. Measured:
  **1.9e-3** and **2.0e-3**. As with the denoise stage the comparison is
  also shown to be sensitive: the corrected frame must be more than five
  times that from the *uncorrected* one, and it is 1.1 and 1.0.
- That "off" is off, exactly. A pipeline built with the flag and handed an
  invalid correction must reproduce a pipeline built without it **bit for
  bit**, and the test compares the downloaded frames with `memcmp`.
- Geometry against arithmetic rather than against the other
  implementation. A black 512x320 frame carries two Gaussian marks on the
  corner diagonal, at 0.35 and 0.78 of the half diagonal. The frame is
  developed twice, once without the warp and once with a correction whose
  three channels differ (`1 - 0.090 r^2`, `1 - 0.100 r^2`,
  `1 - 0.110 r^2`), and each mark's centroid is measured in both. Where it
  has to land is the inverse of `r * poly(r)`, found by bisection, so the
  expected position is arithmetic and not a second warp; measuring the mark
  in both frames rather than trusting where it was drawn cancels the
  debayer's own reconstruction of a Bayer-sampled Gaussian. Tolerance
  0.25 px. Measured: the outer mark moves 15.64 px in R, 17.83 in G and
  20.16 in B — four and a half pixels of deliberate lateral aberration,
  each channel following its own polynomial — and the worst placement error
  is **0.071 px** on the GPU and 0.070 px in the reference.

### 16.6 Cost

`cineraw --bench` on the RX 6900 XT, RCD, 5760x3040, alternating runs on
P1000387.MOV: **5.212 / 5.107 / 5.078 / 5.161 ms** per frame for the whole
pipeline with `--lens off` against **5.943 / 5.960 / 5.947 / 5.932 ms**
with the stage running, so the pass is **0.73 to 0.87 ms**, call it 0.8.
The target was under 1.5 and the frame budget at 30p is 33. In cineview,
which measures libcraw's own timestamps rather than the whole submission,
the same frames read 5.3 to 6.2 ms with the stage on.

That is 49 image loads per pixel — sixteen taps for each of three channels
plus one for alpha — on 17.5 million pixels, which only fits in 0.8 ms
because neighbouring output pixels read overlapping footprints and the
cache absorbs almost all of it: the traffic that actually reaches memory is
one read and one write of an RGBA16F frame, 280 MB, about 0.55 ms at this
card's bandwidth. There is a shared-memory tiling to be had here if the
budget ever gets tight — the source footprint of a 16x16 output tile is a
16x16 source tile plus a three pixel halo, wherever the warp has moved it —
but at 0.8 ms it would be optimizing the wrong stage.

A frame with no record costs nothing at all, because the pass is not
recorded: measured at **0.037 ms** difference on the 256x160 test frame,
i.e. inside the noise of the measurement.

### 16.7 Channels after the colour stage, and what is not verified

Two things about the lateral chromatic aberration are worth stating plainly,
because neither is settled by anything here.

**The record's channels are the camera's; the warp's channels are the output
space's.** The stage runs after the colour pass (16.3), so it applies the R
polynomial to the output image's R, which is a linear combination of the
camera's R, G and B. For the radial distortion this is exact: the common
part of the three polynomials is the same for every channel, so warping a
mixture is warping each part by the same amount. For the *difference*
between the channels it is not: correcting the camera's aberration properly
would mean warping the camera channels and then applying the matrix.
Whether that matters in practice depends on how diagonal the combined
matrix is, and this version does not attempt to find out; the stage is
where section 1 reserved it and where the distortion wants it.

**The aberration correction is not verified on a real frame.** On the
synthetic frame it is exact: `test_pipeline` puts three deliberately
different polynomials on the three channels, the outer mark moves 15.64 px
in R, 17.83 in G and 20.16 in B, and each lands within 0.071 px of the
inverse of its own polynomial (16.5). On a real one it is below the noise
of any estimator tried. On the long window mullion of P1000352.MOV frame
29, where the record separates R from B by 0.46 px along the scan, the
measured R minus B is -0.128 px before the warp and -0.132 px after, on
camera RGB, from cross-correlating the two channels' profiles over 2721
scan lines. That estimator was calibrated by amplifying the record's
per-channel deviation: a nominal 4.56 px change reads 2.63 px, 2.28 px
reads 0.75, 1.37 px reads 0.25, and 0.46 px reads 0.00. It has a dead zone
right where the real effect is, on a defocused edge in a scene with its own
colour fringing. So: the machinery demonstrably applies a different
polynomial per channel, and whether the camera's numbers are the right ones
for a real lens remains open — as does the channel ordering, which
`tools/prr_lens.md` section 8 already lists as inferred rather than
established.
