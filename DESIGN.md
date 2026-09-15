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
