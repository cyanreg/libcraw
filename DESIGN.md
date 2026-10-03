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
  balanced clip point, or instead highlight reconstruction: the clipped
  channels of every sensel rebuilt from the channels that did not clip and
  the colour of the unclipped light around the blown area, before the
  debayer, optional, per frame. Section 18.
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
- Temporal denoising in front of it, in the same domain: every sample
  averaged with the previous frame's filtered value where a 3x3 patch of
  the two agrees within the noise, optional, per-frame strength, with state
  the pipeline keeps from one recording to the next and a reset and a
  replay to steer it. Section 14.7.
- Chroma denoising after the debayer: the quarter resolution chroma band
  replaced by its luma-guided bilateral smoothing, optional, per-frame
  strength. Section 14.4.
- False colour suppression: a median and a blur of the colour differences
  after the debayer, one to five passes, optional, per frame. Section 17.
- Geometric lens correction: a per-channel radial resampling warp of the
  camera RGB intermediate after the debayer, driven per frame by the
  camera's own `com.panasonic.Semi-Pro.optical_correction` record, which
  the library also parses (`<craw/lens.h>`). Optional, per-frame, and a
  no-op on a frame that carries no record. Section 16.
- Defringe: along strong edges, the colour of every pixel held to the
  range of colours on the two sides of the edge, the level kept; takes out
  the rims chromatic aberration leaves. Optional, per-frame strength.
  Section 19.
- Capture sharpening: Richardson-Lucy deconvolution of the luma by a
  Gaussian PSF after every other stage on the camera RGB, each pixel's
  channels scaled by its luma's ratio, held to the range around it and
  masked by the local contrast in units of the modelled noise. Optional,
  per-frame radius, iteration count and threshold. Section 20.
- CPU reference of every stage, tests comparing it against the GPU.

Out, deliberately, with a slot reserved: motion compensated temporal
denoising (section 14.7 says why not yet), scaling, learned debayer, OCIO-generated transforms, LUTs, other CFA phases, and ARRIRAW and
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
  (preprocessor only, for `#include "common.glsl"`) are allowed, and so
  are `GL_KHR_shader_subgroup_basic`, `_shuffle_relative` and `_quad`
  (Vulkan 1.1 core operations, checked on the device by
  `craw_vulkan_get_requirements()`, section 21), `GL_EXT_shader_16bit_storage`
  (the sharpening's fp16 planes, on the 16-bit storage feature the library
  already requires) and `GL_EXT_control_flow_attributes` (`[[unroll]]` on
  the register rings); anything else needs a note here.
- Precision: fp32 arithmetic in shaders unless a stage documents an fp16
  fast path; fp16 storage for the intermediate camera RGB image.

## 3. Data model

Coordinates: x is the column, y the row, (0, 0) top left. CFA quad site
index `s = (y & 1) * 2 + (x & 1)`. For RGGB: site 0 = R, 1 = G (in the R
row, "Gr"), 2 = G (in the B row, "Gb"), 3 = B.

Input image: `VK_FORMAT_R16_UNORM`, the mosaic, sample `raw(x, y)` in
[0, 1]. This is what FFmpeg's Vulkan hwcontext maps `AV_PIX_FMT_BAYER_RGGB16`
to. The samples need not fill the 16 bits: a ProRes RAW frame's do (black
256, white 63743 of 65535), a Panasonic RW2 still's are the camera's 14-bit
codes in the 16-bit container (black 512, white 16383), and each is
normalised by its own levels, `(raw - black) / (white - black)`, which the
profile carries in the same units. The tools take the CFA phase from the
frame's pixel format, the software format behind a Vulkan frame
(`craw_tool_profile_from_frame()`), so a GRBG, GBRG or BGGR mosaic is
named as what it is and then refused by `craw_pipeline_create()` (section
1); every frame either tool has seen so far is RGGB. libavutil has a Vulkan
format for the RGGB phase only, so cineview could not even upload another.

Denoised mosaic (internal, only with `CRAW_PIPELINE_FLAG_DENOISE` or
`CRAW_PIPELINE_FLAG_TEMPORAL_DENOISE`): `VK_FORMAT_R16_UNORM`, the same raw
encoding as the input, written by the denoise stage and read by the
debayer in its place. Section 14.1.

Temporal state (internal, only with `CRAW_PIPELINE_FLAG_TEMPORAL_DENOISE`):
two sets, each a `VK_FORMAT_R32_SFLOAT` and a `VK_FORMAT_R8_UNORM` image at
full resolution, indexed like the mosaic, holding one frame's temporally
filtered level of every sample and the square root of the noise variance
left in it. They persist from one recording to the next. Section 14.7.

Intermediate image (internal): `VK_FORMAT_R16G16B16A16_SFLOAT`, camera RGB
after prep and debayer, alpha unused.

Scratch intermediate (internal, only with `CRAW_PIPELINE_FLAG_FALSE_COLOR`,
`CRAW_PIPELINE_FLAG_LENS`, `CRAW_PIPELINE_FLAG_DEFRINGE` or
`CRAW_PIPELINE_FLAG_CHROMA_DENOISE`): the intermediate's format, a second
image for the stages that cannot run in place; each writes it and the two
swap roles, so what everything downstream reads is always the latest. The
defringe keeps its first pass there and writes the intermediate in place
(19.3). It is an image of its own with every method; RCD, which used
to lend its auxiliary image for it, has none since its passes became one
dispatch (5.3). Section 17.3.

Quarter resolution chroma images (internal, only with
`CRAW_PIPELINE_FLAG_CHROMA_DENOISE`): two RGBA16F images of ceil(W/4) x
ceil(H/4), the block means of (R - G, B - G, G) and the same with the
differences filtered. Section 14.4.

Rebuilt mosaic (internal, only with
`CRAW_PIPELINE_FLAG_HIGHLIGHT_RECONSTRUCTION`): `VK_FORMAT_R16_SFLOAT`,
full resolution, balanced and unclipped, written by the highlight
reconstruction and read by the debayer in place of the mosaic. Its colour
field: RGBA32F, one texel per 4x4 block, and a pyramid down to 1x1.
Section 18.

Output image: `VK_FORMAT_R16G16B16A16_SFLOAT` or `R32G32B32A32_SFLOAT`,
linear light in the requested colour space, alpha 1.0.

Border handling, all stages: mirror without repeating the edge sample and
preserving CFA phase: `x < 0 -> -x`, `x >= W -> 2W - 2 - x`, same for y.
The denoise stage applies that same rule to each CFA plane in the plane's
own coordinates rather than to the mosaic; see 14.1 for why.

## 4. Stage math

### 4.0 Denoise (optional, before everything)

Filters the mosaic in the raw domain, in time (14.7) and then in space
(14.3), and hands the debayer a denoised mosaic in the same encoding.
Defined in section 14; nothing else in this section changes when it runs.

### 4.0b Highlight reconstruction (optional, before the debayer)

Reads the mosaic (the denoised one when 4.0 ran) and writes a rebuilt
one, balanced and unclipped, in which every clipped channel has been
estimated from what did not clip; the debayer reads it with the prep of
4.1 reduced to the identity. Defined in section 18.

### 4.1 Prep (fused into the debayer loads)

For a sample at site `s` with channel `c(s)`:

    v  = (raw - black_level[s]) / (white_level[s] - black_level[s])
    v  = max(v, 0)
    v  = v * wb_asshot[c(s)]
    v  = min(v, 1.0)

`wb_asshot` is a misnomer kept for the shader field: it is
`CRAWColorParams.wb` evaluated for the frame's own wb_mode and illuminant.
The clip at 1.0 after white balance keeps blown highlights neutral under
that illuminant. The highlight reconstruction of section 18 replaces this
whole step when it runs: the debayer then reads the rebuilt mosaic, which
already holds the balanced values, with a prep of black 0, range 1, gains
1 and no clip. Note
the clip point differs per channel in raw units. An earlier revision
balanced with the as-shot gains and re-balanced in the colour stage; that
turned every clipped lamp the colour of the gain ratio, so the debayer
output is now tied to the frame's white balance and a grade-time change
re-runs it, which costs a few milliseconds.

The debayer operates on `v`. What counts as clipped, and how softly, is
section 18.2's business.

### 4.2 Debayer

Produces R, G, B at every pixel from `v`. Definitions in section 5.

### 4.2b False colour suppression (optional, after the debayer)

Replaces the intermediate with a copy whose colour differences have been
median filtered and blurred, G untouched. Defined in section 17; the
histogram of 4.4 and the colour stage of 4.3 read the copy.

### 4.2c Lens correction (optional, after 4.2b)

A per-channel radial resampling warp of the intermediate, in the
destination-to-source convention, driven by the frame's own record.
Defined in section 16; everything after it, the histogram of 4.4
included, reads the warped copy.

### 4.2d Defringe (optional, after 4.2c)

At every pixel near a strong edge in level, the colour clamped to the
range the two sides of the edge span, the level kept; every other pixel
untouched. Defined in section 19; in place in the intermediate.

### 4.2e Chroma denoise (optional, after 4.2d)

Replaces the intermediate with a copy whose quarter resolution chroma
band has been smoothed by a luma-guided bilateral, G untouched. Defined
in section 14.4; the histogram of 4.4 and the colour stage of 4.3 read
the copy.

### 4.2f Capture sharpening (optional, after 4.2e)

The luma deconvolved by Richardson-Lucy iterations with a Gaussian PSF,
and the three channels of each pixel multiplied by its luma's ratio, held
to the range of the luma around it and weighed by a local contrast mask; a
pixel the mask leaves alone is untouched. Defined in section 20; in place
in the intermediate, so the histogram of 4.4 and the colour stage of 4.3
read it.

### 4.3 Colour

With `c` the camera RGB from the intermediate and the frame's
`CRAWColorParams` `p` (as-shot or re-balance) plus the as-shot params
`a`:

    c'  = c                               the gains are already in c
    xyz = p.cam_to_xyz * c'
    xyz = xyz * p.gain * 2^exposure_stops
    out = M_out * xyz                     M_out from craw_colorspace_from_xyz_d65()
    if (out_space is ACEScg and !(flags & NO_GAMUT_COMPRESSION))
        out = gamut_compress(out)         ACES 1.3 RGC, section 8.1
    if (flags & CLAMP_NEGATIVE) out = max(out, 0)
    alpha = 1

No clamp of any kind unless the flag is set. Values above 1.0 are normal
(sensor clip sits at `gain` times diffuse white).

`out_space` is the frame's own `CRAWFrameParams.out_space` when it set
`out_space_set`, and the pipeline's `CRAWPipelineParams.out_space`
otherwise. Nothing the pipeline owns depends on the choice — it is this one
3x3, composed on the host per frame, plus the compression above — so a
caller may alternate spaces frame by frame on one pipeline without a
rebuild and without a second set of intermediates. That is what a viewer
showing ACEScg while writing an ACES2065-1 archive of the same frames
needs. The flag is a separate bool rather than a sentinel in the enum
because 0 is `CRAW_CS_XYZ_D65`, a real space, so a zeroed CRAWFrameParams
has to keep meaning "whatever the pipeline was created with".

### 4.4 Auto white balance histogram (optional stage)

Reads the intermediate (the filtered one when the stages of sections 17
and 14.4 run) at 1/8 resolution: one sample per 8x8 pixel block,
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

On the GPU the three passes are one dispatch (`rcd.comp.glsl`). A
workgroup of 32x16 loads the prepped mosaic of its tile with an apron of
five into shared memory and works inwards a ring at a time: L on an apron
of 4, the directional values on 3, the weights and pass B's green on 2,
pass C's R and B site estimates on 1, and the G sites and the output on
the tile. Wherever a pass read what an earlier one wrote at a mirrored
position, the tile holds at the position outside the frame the value
computed at the one inside it; the directional values, which pass A
computed from the mirrored patch at every offset, are computed at the
position itself. Four dispatches used to write and read back an RGBA16F
image of L and the weights, an R16F green plane and the intermediate
twice, a gigabyte at 5.8K for the 175 MB of mosaic in and intermediate
out, and the first pass was bound by its 49 image loads a pixel.

The values the passes kept in fp16 images are rounded to fp16 where they
were stored, the way those stores did it, so the output is the four
passes' bit for bit. On RDNA2 that is toward zero, not to nearest: an
RGBA16F store of 1 + 7/8 of an fp16 step reads back as 1, and denormals
are kept. Left in fp32 the steps are the reference's arithmetic, and the
intermediate's error against it on the synthetic scene falls from 1.57e-3
to 9.8e-4 and the colour output's from 6.7e-3 relative to 1.7e-3 (on the
real frame from 5.96e-2 absolute to 3.94e-2); but the real frame's false
colour and chroma test holds its stage to twice the plain chain's error,
and the plain chain's falling pulls that bound under the stage's own
8.9e-2. Keeping the fp32 steps needs that bound reconsidered, so for now
the truncation stays.

At 5.8K on the 6900 XT, back to back, the four passes took 3.54 ms and
the one takes 1.38 to 1.49 (in fp32, 1.31 to 1.37; a 16x16 workgroup
1.51, wave64 1.57), and the aux image and the green plane, 175 MB, are
gone. What is left is the tile's shared memory work, five barriers and
the aprons' share of it, not memory: the mosaic in and the intermediate
out are 0.4 ms of bandwidth.

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
the knots and bias the table's own entries. A tint moves a neutral along
`(-1, -1)/sqrt(2)` and nothing else, so for a raw neutral `(ln rg, ln bg)`
the locus point is the one it lies from along that direction: walking the
samples from warm to cool, the first pair between which the residual
(point minus locus) changes sign across it - its dot product with `(1, -1)`
- brackets the point, and bisection in mired on the interpolated gains
finds it. `cct` is that point's Kelvin value and `tint` the dot product of
the residual with `(-1, -1)/sqrt(2)`, which is then the whole residual; a
point past either end goes to that end. The nearest sample would not do:
its residual is perpendicular to the locus, which is the green direction
only where the locus happens to cross it at a right angle, and the rest of
the residual is lost when it becomes a tint (on the S1M2 table, 2300 K at
tint -0.15 came back as 2269 K at -0.1475). `craw_profile_illuminant_to_neutral`
is the exact inverse: `rg = (1/wb_R(cct)) * exp(-tint/sqrt(2))`, `bg`
likewise.

## 7. Auto white balance estimator

Inputs: histogram, profile, `CRAWAwbParams`. Per bin with count > 0, the
bin centre is a candidate point. Compute its locus point as in section 6
and its distance `d` from it, which is the size of its tint. Bins with
`d < band` are in the zone.
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

Smoother: state is a mired value and a tint. On push: with no state, adopt
the estimate if it is valid, whatever its confidence - the first reading is
the track until a better one moves it; a track that waited for a confident
start was never there at all on a clip the estimator is unsure of
throughout - and fail with EAGAIN, output untouched, if it is invalid. With
`scene_cut`, adopt the estimate when it is valid and its confidence reaches
`min_confidence`. Else if `est.valid && est.confidence >= min_confidence`:
target = estimate; ignore the change if `|target - state| <
hysteresis_mired`; otherwise move the state towards the target by `1 -
exp(-dt / time_constant)` of the difference, limited to `max_slew_mired_per_s
* dt`. Tint follows the same rule with the same fractions. `dt` is the
difference in `pts_s` to the previous push, clamped to [0, 1] seconds. A
reading below `min_confidence` therefore holds the track (a night scene's
unsure readings swing by thousands of kelvin from frame to frame), and
nothing ever jumps: cineview feeds a paused picture one push per
re-development until the track settles, so it glides to its own frame's
reading through the same smoothing as when playing, and a seek is a glide
from where the picture was to where it is.

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

`craw_planckian_xy()` is the chromaticity a correlated colour temperature
names: Krystek's rational approximation of the Planckian locus in CIE 1960
uv (1000 K to 15000 K, about 1e-4 in uv from the integrated locus),
converted to xy. `craw_bradford_adaptation()` is the von Kries adaptation in
the Bradford cone space between two whites, the one the ACES spaces above
are reached with, made public so that a caller can move a picture between
whites the way the library does; it matches the published D65 to D50
matrix to 2e-4. cineview uses the two for the white balance of an already
developed clip, whose recorded temperature is a point on that locus.

`craw_colorspace_to_xyz_d65()` is the other direction, built from the same
primaries and the same adaptation and inverted from the same double
precision matrix, so the two compose to the identity to about 1e-7 in
float. Chaining one with the other is how a caller moves between two output
spaces; for the ACES pair the two adaptations cancel and the composition
reproduces the published ACEScg to ACES2065-1 matrix to 1e-7.

`test_profile` checks all of this: both ACES matrices against the values
above, every space's round trip, BT.709 through XYZ into AP1 against the
published sRGB to ACEScg matrix, and sRGB white landing on AP1 (1, 1, 1).

### 8.1 Why ACEScg, and the reference gamut compression

The pipeline's hand-off to the display path is ACEScg, not linear BT.2020.
BT.2020 does not contain what this camera's matrix produces: a saturated
LED blue or cyan comes out with a negative component, and a display side
gamut mapper then clips it hard, which is where the flat, plastic edges on
coloured light sources come from. ACES2065-1 is not a working space either
— its primaries are imaginary and its hull is enormous, so a hull-based
perceptual gamut mapper spends its compression budget on colours that do
not exist and desaturates the ones that do.

So the two ACES spaces do the two jobs the Academy intended. ACEScg (AP1)
is the rendering and display hand-off: wide enough that almost everything
real is inside it, bounded enough that a gamut mapper downstream has a
meaningful hull to work against. ACES2065-1 (AP0) is the archive: it is
never compressed and keeps every value the camera matrix produced,
negative components included, which is what an archival export and a
re-grade years later need.

What pulls the rare colours outside AP1 inward is the ACES 1.3 Reference
Gamut Compression, applied to the AP1 linear RGB at the end of the colour
stage. It is transcribed from the Academy's own CTL
(`transforms/ctl/lmt/LMT.Academy.ReferenceGamutCompress.ctl` in
ampas/aces-dev at v1.3, functions `compress()` and `gamut_compress()`);
OpenColorIO's built-in `ACES-GamutComp-v13` is the same operator with the
same seven numbers, and libcraw implements no variant of it. With
`ach = max(R, G, B)` and, per channel, `d = (ach - c) / |ach|`:

    scl  = (lim - thr) / (((1 - thr) / (lim - thr))^-pwr - 1)^(1/pwr)
    d'   = d                                            if d < thr
    d'   = thr + scl * nd / (1 + nd^pwr)^(1/pwr)        otherwise,
                                            with nd = (d - thr) / scl
    c'   = ach - d' * |ach|

Parameters, per channel in the order R, G, B, i.e. the cyan, magenta and
yellow directions: limits 1.147, 1.264, 1.312; thresholds 0.815, 0.803,
0.880; power 1.2. `scl` is solved so that a distance of exactly `lim`
compresses to exactly 1.0, which puts the channel on zero: that is what
"distance limit" means and it is the whole reason anything nearer stops
being negative. The solved scales are 0.327302, 0.285938 and 0.146821.

`lib/gamut.c` derives the scales once per frame and the pipeline pushes
them to the colour shader, so the shader and the CPU reference cannot
disagree about a constant. The curve is C1 at the threshold (its slope
there is exactly 1 from both sides), it is monotone, and it never expands a
distance, so it cannot reorder channels or move the achromatic axis.

Three consequences worth stating rather than discovering:

- Below the threshold nothing moves at all, bit for bit. Above it,
  saturated *in-gamut* colours do move: BT.709 blue is at distance 0.946 in
  the cyan direction, past the 0.815 threshold, and the operator pulls it
  by 0.050 in AP1 units. That is the specification's own trade — the
  thresholds were fitted to protect the ColorChecker Classic, not the whole
  of BT.709 — and it is why the display primaries are not a useful
  invariance test. Real surfaces are well inside: every one of the test's
  BT.709 and BT.2020 primaries and secondaries pulled 40% towards neutral
  is untouched exactly, except the two most saturated, which move by 1.6e-4.
- The guarantee is bounded by the limits, not unconditional. A distance
  past `lim` still compresses to more than 1.0 and leaves the channel
  negative. For this camera the reach is a property of the composed
  camera-RGB-to-AP1 matrix, and because the distance is a scale invariant
  ratio of linear forms its maximum over the cone of non-negative camera
  RGB sits on one of the three camera primaries. The as-shot matrix reaches
  (1.0668, 1.1572, 0.9901) against limits of (1.147, 1.264, 1.312), so the
  whole sensor is inside and ACEScg output is non-negative. Across the
  rebalance table the reach grows to (1.1131, 1.2656, 1.0997): the 7500 K
  knot's blue primary is a hair past the magenta limit and leaves 2.5e-4 of
  the pixel negative, on a colour of negative luminance that no light
  produces.
- The compression is a colour stage operation, and two stages around it
  can put negatives back. A debayer with negative kernel weights (MHC, RCD)
  can ring the camera RGB itself below zero at a hard edge, which leaves
  the cone the bound above is taken over; and the lens warp of section 16
  runs *after* the colour stage with a Catmull-Rom kernel whose lobes are
  negative. On P1000387 frame 0 the colour stage alone leaves zero negative
  pixels in ACEScg, RCD adds 10 of 17.5 M and the lens warp 1; with both,
  26. A consumer that needs the output strictly non-negative under any
  chain should add `CRAW_FRAME_FLAG_CLAMP_NEGATIVE`, which then only ever
  has ringing to clamp rather than real colour.

ACES2065-1 is never compressed: every real light is inside AP0, so on real
footage it does not need it, and compressing the archive would be exactly
the wrong thing to do to it. It is not unconditionally non-negative either
— the camera matrix is a 3x3 fit and extrapolates outside the spectral
locus for the most saturated sources — but the counts are an order of
magnitude below the uncompressed ACEScg ones.

`CRAW_FRAME_FLAG_NO_GAMUT_COMPRESSION` turns the operator off for one
frame, which is what a caller comparing the two wants; it is a frame flag
rather than a pipeline parameter because it costs nothing (a bit in the
colour shader's push constant, and the branch is uniform across the
dispatch) and because comparing with and without should not mean rebuilding
a pipeline. Measured on an RX 6900 XT over 40 frames of P1000387, three
runs each: BT.2020 5.943 to 6.019 ms a frame, ACEScg 5.906 to 5.990, AP0
5.923. The spread within one space is larger than the difference between
them.

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

The built-in table in `tools/profiles.c` has two knots more, 2500: 1.1055
3.1675 and 10000: 2.5537 1.2813, which are the camera's own per-frame
pairs from clips shot in Kelvin mode at those settings (the lamp series of
2026-10-01, P1115218-223, two clips at each; the RW2 stills' EXIF levels
agree to 0.5%). Between the container's knots the camera does not
interpolate linearly: its 2500 K pair is 3.4% (r) and 5.4% (b) from the
line between the 2000 and 3200 K knots, its 10000 K pair 1% and 2% from
the 7300-11500 line. The pair also moves by up to 1.5% between clips shot
at one setting, so a knot taken from one clip is good to about that. A
clip at any other Kelvin setting is one more sample of the curve, whatever
is in front of the lens.

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

## 9c. Panasonic RW2 stills

An RW2 from the same camera is the same sensor's mosaic, one picture of it,
and FFmpeg's rw2 decoder (in Lynne's tree; upstream has neither the decoder
nor the record, so the tools build without RW2 there, `CRAW_TOOL_HAVE_RW2`)
exports less about it than a ProRes RAW frame header says. What it gives,
in `AV_FRAME_DATA_RAW_COLOR_PARAMS` of type `AV_RAW_COLOR_PARAMS_RW2`: the
levels (the green sites' black for all four, `black[1]/65535`, and the
largest linearity limit as the white), and the as-shot white balance, R/G
and B/G, from the file's WB tags. No matrix, no colour temperature
(`wb_cct` is 0), no gain. The camera model and the ISO are in the Exif IFD
the decoder attaches to the frame, `Model` and `ExifIFD/ISOSpeedRatings`;
the demuxer has no tags at all. The picture is cropped by the file's
border, even aligned: 6000x4000 inside a 6008x4008 readout on the DC-S1M2,
which is also the size of the camera's own JPEG (ffprobe reports 6004x4000
because `av_frame_apply_cropping()` gives up four columns of the left crop
to keep the plane pointer aligned; the tools never let it crop).

`craw_tool_profile_from_frame()` (tools/profiles.c) makes the rest out of
the camera's own tables, which are the same sensor's:

- as_shot_wb = { R/G, 1, B/G } as the file has them;
- as_shot_cct = the temperature the as-shot gains lie at on the table's
  locus, `craw_profile_neutral_to_illuminant(1/R, 1/B)`, the tint left
  across it being already in the gains;
- as_shot_matrix = the table's matrix at that temperature,
  `craw_profile_eval(CRAW_WB_REBALANCE)`'s `cam_to_xyz`, which is how the
  camera itself picks a ProRes RAW frame's matrix (section 9: REBALANCE at
  a frame's CCT reproduces its header matrix to 5e-4);
- gain = the camera's own GainFactor for the gain mode the ISO is in,
  23.04 at ISO 1000 and above and 14.51 below, divided by eight: a still
  is rated three stops closer to the sensor clip than a V-Log recording
  (the measurement below), so the video factor itself made every still
  three stops brighter than the camera's own JPEG. Both factors are this
  camera's video ones (14.2's clips: 23.04 in every clip from 1000 up,
  14.51 in every 640 clip). ISO 100 to 500 exist only for stills, so no
  clip says which mode they are, and the low mode is an assumption; where
  between 640 and 1000 the camera changes mode is not known either;
- as_shot_ei = the ISO, so that a working index re-rates a still exactly as
  it re-rates a clip, `gain * (ei / iso)`.

The camera's tables come from the model in the Exif when the container
names none, which for a still it never does; without tables there is no
matrix, and the frame is refused with a message that names `--profile` and
`--tables`, never developed through an identity.

What the gain rule does, measured against the camera's own JPEGs on flat
mid-tone patches (64x64, the JPEG's green between 0.25 and 0.75 of full
scale, flat in both; the JPEG taken to linear through the sRGB curve, which
ignores its picture style's S curve):

    still      ISO  exposure         patches  ours / JPEG, linear G
    _1000406   100  1/15 f/2.8 (M)       152  9.66  (+3.27 stops)
    _1000036   100  1/640 f/6.3 (P)        3  6.52  (+2.71 stops)
    _1000042   100  1/200 f/4 (scene)     40  7.97  (+3.00 stops)
    _1093946   640  1/8000 f/1.4         152  7.76  (+2.96 stops)

So a still developed with its camera's video GainFactor itself sits about
three stops above the camera's JPEG of it in the mid-tones, at ISO 100 and
at 640 alike: the ISO 640 still, whose factor is measured, is as far off
as the ISO 100 ones, whose factor is assumed. The gap is the rating, not
the assumption: the video factor places the sensor clip 6.3 stops above
18% grey (14.51 / 0.18), the log-style headroom the camera gives a V-Log
recording at the same ISO number, where its stills rating leaves about
three. The tools therefore take the video factor over eight for a still
(measured as 2.7 to 3.3 stops over the four stills; three is the rating
convention and the number used). With it cineraw's SDR file shows the
same mid-tone patches at 0.34 of full scale against the JPEG's 0.31 on
_1000406, and at 0.62 against 0.63 on _1093946; with the video factor
itself they came out at 0.87. The price is that a still and a clip from
the same body at the same ISO number and exposure no longer have the same
brightness: the clip is three stops brighter, because the camera's two
ratings are three stops apart, and `--exposure 3` on the still (or the
panel's Exposure section) puts the two side by side.

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
  [fill histogram buffers], [denoise pass, temporal and spatial in one
  dispatch], [highlight field, pull and
  push over its pyramid, fill], the debayer, [false colour pass], [lens
  pass], [defringe prep and clamp], [chroma down, blur and up], [awb
  histogram], colour pass, [copy histogram to readback], and a final
  barrier that returns dst to the caller's stated layout. The colour pass
  always writes dst; a frame whose lens correction is invalid records no
  lens pass at all. The denoise pass
  dispatches four workgroups per tile of the half resolution grid, one per
  CFA plane, side by side in x (the low two bits of `gl_WorkGroupID.x`
  select the plane; 14.7 says why), and the debayer then reads its
  output image instead of the caller's source. The highlight
  reconstruction reads whichever of the two the debayer would have read
  and the debayer then reads the rebuilt mosaic instead.
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
  polynomial's inverse says it should. The false colour stage is held to
  twice the debayer's own intermediate bound, and on the real frame to
  twice the plain chain's absolute error, plus the checks of section 17.5 that it ran,
  that both ways of switching it off reproduce the plain chain, that a
  single site spike loses its colour by the amount the definition predicts
  and that G is untouched bit for bit. The highlight reconstruction is
  held to the relative bounds above, on a scene of its own, plus the
  checks of section 18.5 that the definition rebuilds what it says it
  does, that both ways of switching it off are the plain chain bit for bit
  and that an unclipped frame moves by no more than its mosaic's fp16
  rounding. The defringe is held to 1e-2 absolute against the reference
  run on the GPU's own undefringed intermediate, on the real frame to a
  count of pixels past that bound, plus the checks of section 19.5 that
  the definition does what it says, that both ways of switching it off
  are the plain chain bit for bit and that it ran. The temporal denoise is
  tested on a sequence rather than a frame: eight frames against the
  reference at the denoise bound, the noise falling and the moving bars
  leaving no trail, and its state rules — off, reset, replay, a frame
  without the pass, four recordings in flight — bit for bit on the GPU
  against the GPU (section 14.7). lavapipe is acceptable for these
  tests; it exposes everything required.
- `test_pipeline`, gamut compression (section 8.1), in three parts. The
  operator alone, on the CPU: the thresholds and power against the
  specification, nothing below the threshold moving by a single bit, a
  distance of exactly the limit landing on 1.0, monotone and never
  expanding across a sweep that crosses the threshold, and no jump there at
  a step of 1e-6. What it leaves alone: BT.709 and BT.2020 primaries and
  secondaries, each at four levels and each also pulled 40% towards
  neutral, checked for scale invariance, for never being pushed away from
  achromatic, for never moving the achromatic value, and — for the ones
  inside the thresholds — for not moving at all. And the camera's reach,
  which bounds the whole sensor from the three camera primaries.
  Then the chain on the device, on a scene of eight saturated camera-RGB
  bands at four levels: BT.2020 and uncompressed ACEScg must produce
  negatives (or the scene is not testing anything), compressed ACEScg must
  produce none except where the debayer rang the camera RGB out of the
  non-negative cone, ACES2065-1 must be far cleaner than uncompressed
  ACEScg and must be unchanged bit for bit by the compression flag, and all
  of it must match the reference within the tolerances above. Finally one
  pipeline created for BT.2020 records ACEScg, ACES2065-1 and ACEScg again
  through `CRAWFrameParams.out_space`, each against its own reference.

  The bands cross-fade over the last fifth of their width. A one-pixel step
  between two saturated colours is a test of RCD's behaviour when its
  directional ratios divide by a near-zero low pass sum, not of colour, and
  it pushes GPU and reference apart by 2.5e-2 relative for reasons that
  have nothing to do with this section.

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
  submission is the rule (section 4 of pipeline.h), and recordings are
  submitted to one queue in the order they were made.
- With `CRAW_PIPELINE_FLAG_TEMPORAL_DENOISE` the pipeline is no longer a
  function of its arguments: each recording reads the state the previous
  one left (section 14.7). The caller has to say when the frame it hands
  over is not the next one — `CRAW_FRAME_FLAG_TEMPORAL_RESET` after a
  seek, a loop or a new clip, `denoise_temporal_replay` for the same
  frame developed again — and a recording that is made but never
  submitted has moved the state all the same. Two consumers of one clip
  that want the same state (a viewer and an exporter) need a pipeline
  each, or have to replay each other's frames in order; cineview builds
  its exporter its own pipeline and starts both at the export's first
  frame. A caller-visible way to save and restore the state, or to hand
  it from one pipeline to another, does not exist yet.

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
laws in the sensor's gain. Outside the table the model is **extrapolated
the way a gain moves it**: a in proportion to the index on both sides, b
as the square of the index above the last knot and unchanged below the
first. The read term is two noises, one ahead of the gain, which the gain
squares, and one after it, which the gain does not touch, so
b = b0 + c * iso^2; at the top of a table the first is nearly all of b and
at the bottom the second is most of it. The S1M2's measurements below
follow that law, within their scatter, over the whole of its main gain
mode. An exposure index that is not positive gets the first knot.

The table used to hold the nearest knot unchanged instead, on the argument
that a model too small only under-denoises, which the strength can
correct. That stopped being true once the temporal average (14.7) and the
sharpening mask (20) read the model too: the first takes noise it was not
told about for motion and averages next to nothing, the second takes it for
texture and sharpens it, and with two knots at 1000 and 1250 an ISO 10000
clip was eight times short. The lookup uses the frame's as-shot exposure
index, not its working one — rating a clip at a different index scales the
picture afterwards, it does not change how the sensels were read.

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
values, which at a low index above roughly 10 means the frame has no region
dark enough to measure b and what was fitted there is scene texture. The
real read noise rises with the index (below), to about 10 at ISO 10000 and
12 at 12800 on this camera, so the bound scales with it. The other failure
is the opposite one: a frame with no shadows at all fits b = 0, a read
noise of 0.00, which is no measurement of b.

#### Panasonic DC-S1M2, measured

`cineraw --noise-estimate --frames 10` on every ProRes RAW clip on disk that
records an ISO other than 1000, 28 of them from ISO 640 to 12800, and on
fourteen at 1000; per clip, the median over the ten frames. A clip counts
if its fit error is at most 15%, the fit keeps at least 90% of its bins and
a moves by at most 16% over the ten frames (it does in a scene that
moves); its b
counts only if the fit resolved one, a read noise above half a code value.
At ISO 1000 the frame 0 fits of four clips no longer on disk from the
earlier table count as well (starred). The clips that count:

    clip       ISO    a          b          fit    read noise
    P1000111     640  1.2749e-5  1.719e-8   13.1%   8.32
    P1094715    1000  1.1699e-5  7.81e-10    3.1%   1.77
    P1094753    1000  1.1762e-5  9.86e-10    5.8%   1.99
    P1093961*   1000  1.2225e-5  -          10.3%   0.00
    P1094127*   1000  1.2923e-5  2.24e-10   12.4%   0.95
    P1093971*   1000  1.3016e-5  -           5.0%   0.00
    P1093870*   1000  1.3154e-5  -          12.6%   0.00
    P1000387    1000  1.3302e-5  5.31e-10   14.0%   1.46
    P1094725    1000  1.3499e-5  -           9.8%   0.00
    P1094729    1000  1.3600e-5  -           9.2%   0.00
    P1001123    1000  1.5830e-5  1.20e-9    10.8%   2.20
    P1094844    1000  1.6053e-5  -          14.1%   0.00
    P1030266    1250  1.4941e-5  9.68e-10   12.7%   1.98
    P1094731    1250  1.6695e-5  -           4.4%   0.00
    P1000255    1600  2.1129e-5  1.22e-9     9.2%   2.22
    P1000167    2000  2.2271e-5  2.47e-9     3.5%   3.16
    P1000168    2000  2.5992e-5  1.37e-9     5.4%   2.35
    P1094738    3200  3.4042e-5  2.02e-10    4.7%   0.90
    P1094727    3200  3.8290e-5  2.80e-9     5.6%   3.36
    P1094724    3200  3.9979e-5  2.05e-9    10.0%   2.87
    P1094764    4000  4.7596e-5  -           2.4%   0.00
    P1094754    8000  8.3590e-5  -           9.7%   0.00
    P1000165   10000  1.1557e-4  2.37e-8     1.5%   9.77
    P1094352   10000  1.1721e-4  3.00e-8    13.3%  10.99
    P1001122   10000  1.1868e-4  5.03e-9     4.5%   4.50
    P1094353   10000  1.1928e-4  2.93e-8    13.3%  10.87
    P1000170   12800  1.4693e-4  5.22e-8     6.5%  14.50
    P1094757   12800  1.4743e-4  3.28e-8     4.3%  11.49
    P1000164   12800  1.5518e-4  4.37e-8     1.6%  13.27
    P1094756   12800  1.5630e-4  2.99e-8     3.0%  10.98

Sixteen did not count: P1000110 and P1000123 at 640; P1094740, P1094744,
P1094797, P1094823 and P1094854 at 1000 (fit errors of 18 to 24%, a third
to seven tenths of the bins kept), P1094763 and P1094770 there too (a
moving by 26 and 28%); P1094366, P1094367 and P1094758 at 1250, P1030111 at
2500, P1094765 at 4000, P1000436 at 6400 and P1000376 at 10000, on the fit.
Every one of them reads an a above the median at its ISO (the law below at
2500 and 6400), by 7% to 3.2 times: texture only ever adds. P1094644 of the
earlier table misses the fit rule too, at 15.9%, and its three rejects stay
out; one of those, P1093970, a = 6.8e-5 at a fit error of 11.7% and a
plausible read noise, is caught by neither statistic, which is why a knot
is a median over clips and not one clip's fit.

Per ISO, the medians:

    ISO     a          a / ISO    b          clips (a, b)
      640   1.27e-5    19.9e-9    1.72e-8    1, 1
     1000   1.32e-5    13.2e-9    7.8e-10   11, 5
     1250   1.58e-5    12.6e-9    9.7e-10    2, 1
     1600   2.11e-5    13.2e-9    1.22e-9    1, 1
     2000   2.41e-5    12.1e-9    1.92e-9    2, 2
     3200   3.83e-5    12.0e-9    2.05e-9    3, 3
     4000   4.76e-5    11.9e-9    -          1, 0
     8000   8.36e-5    10.4e-9    -          1, 0
    10000   1.18e-4    11.8e-9    2.65e-8    4, 4
    12800   1.51e-4    11.8e-9    3.82e-8    4, 4

From ISO 1000 up this is one gain mode — every one of those clips carries
the same GainFactor, 23.04 — and the two terms follow

    a = 12e-9 * ISO                    (ISO^0.96 fitted; every knot within 10%)
    b = 6.1e-10 + 2.3e-16 * ISO^2      (17% rms over the medians)

The shot term goes as the index: ISO is a gain applied before the file is
written, a rising 8.9 times from 1000 to 10000. The read term is the usual
pair, 1.6 code values after the gain, which no index changes, and 9.6e-4 *
ISO code values ahead of it, which the gain carries: 0.96 at 1000 and 9.6
at 10000. At 1000 the first is three quarters of b and at 12800 the second
is 98% of it, which is why b rises 34 times from 1000 to 10000 and not 100;
it is the floor, not a second native ISO. a / ISO holds, the GainFactor
holds and b stays on its law all the way from 1000 to 12800, so there is
no mode change inside that range.

ISO 640 is the other mode. Its GainFactor is 14.51, the 23.04 of the rest
times 640/1000 to within 2%; its a is ISO 1000's (1.27e-5 against
1.32e-5), not two thirds of it, and its b is 22 times ISO 1000's, 8.3
code values, in all three 640 clips (1.60e-8 to 1.72e-8, the two whose
texture disqualified their a included). Where between 640 and 1000 the
camera switches no clip says: the table interpolates the two knots as a
power law, which puts ISO 800 at a = 1.30e-5, b = 3.7e-9, and an ISO 800
clip would settle it.

The knots are the medians at every ISO with both terms measured: 640,
1000, 1250, 1600, 2000, 3200, 10000 and 12800. The two with a alone check
the interpolation: at 4000 the table gives 4.77e-5 and P1094764 reads 0.3%
below it, at 8000 9.47e-5 and P1094754 12% below. Above 12800 the
evaluation follows the law out (b0 is 2% of b there, so b ~ iso^2 is the
law to that), and below 640 it scales a and keeps b. P1094353's own fit
at 10000 was 1.19e-4, 2.95e-8, and the table gives 1.18e-4, 2.65e-8.
`--noise A,B` still overrides all of it.

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

One workgroup per tile of one plane, the four planes of a tile four
neighbouring workgroups in x, so one dispatch covers all four (14.7 says
why the planes are interleaved rather than in `gl_WorkGroupID.z`, as they
were first). The workgroup loads its tile plus a halo of S + P into
shared memory, already stabilised; with the temporal pass of 14.7 the
tile holds what that pass made of the plane.

The workgroup is four subgroups of 32 stacked, each a band of M = 4 rows;
a lane owns a column of its band and filters its four samples. For one
search offset the lane forms the squared difference of its column
against the offset column once per row of the band's patches, sums it
along x with its neighbours' through `subgroupShuffleUp/Down` (a
`ds_swizzle` each in wave32 on RDNA) and along y in registers, which is
every patch distance of the band for two adds and two shuffles a row and
two adds a sample, where each sample used to read (2P + 1)^2 = 9 values
of shared memory per search position; and the offset column is read from
the tile once for all 2S + 1 offsets that share its dx. The P lanes at
either end of a subgroup have a neighbour on one side only and filter
nothing, so a tile is 30 x 16 samples. The weight exp(-max(d2 - thr, 0)
/ h^2) is evaluated as exp2(min(d2 c1 + c0, 0)) with the constants folded
per sample, two multiplies fewer a neighbour. The search is then bound by
its VALU: 13 instructions a neighbour, one of them the exp, 14.5 with a
history, and three swizzles alongside. The sums are in
another order than before, left to right along x as the reference sums a
row but not the reference's raster order over the patch, so the result
moves by rounding: on the real frame 0.18% of the output pixels change,
by at most 4e-3 of the pixel, and every error `test_pipeline` prints is
the same to the digits it prints. `lib/pipeline.c` sizes the tile and
checks it against the device's shared memory (16 KB here), halving the
bands on a device with less, and static-asserts the radii against the
shader's register arrays (S up to 3, P up to 2, 16 rows).

Measured against the tiled version it replaces, the pass alone at 5.8K,
back to back, median of 20 over three runs: spatial 2.02 ms to 1.18,
spatial and temporal 2.60 to 1.59, temporal alone 0.71 to 0.73 (the
temporal pass alone is memory bound, and the new tile loses two lanes in
32 to the patch it does not use). DPP instead of the swizzles
(`subgroupClusteredRotate` over clusters of 16, which ACO folds into the
adds as a row rotate) measured 1.54 ms against 1.27 for the swizzles at
the same instruction count and was dropped; wave64, where a shuffle is
three `ds_bpermute`, over 5 ms. The table below is the tiled version's.

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

### 14.4 The chroma stage after the debayer

Measured first, and not built then: at ISO 1000 the raw stage leaves the
chroma noise of a dim shadow at 2.7% of the level (below), and a second
stage would have been chasing the difference between that and half of it.
What changed the case is section 17: on a wet street in the rain the
colour that survives the false colour suppression is not single pixels
but blobs of three and more — the glints two and three sensels wide that
the debayer measured on two sites, and the correlated chroma noise of a
dark road — and a 4K downscale leaves that flicker whole. A quarter
resolution chroma stage works at exactly that scale. It is the same class
of operation as Adobe's colour noise reduction, the one setting a photo
converter turns on for every file, and RawTherapee's wavelet chrominance
denoise.

Where it sits: after the false colour stage of section 17, the lens warp
of section 16 and the defringe of section 19, on the camera RGB
intermediate, before the histogram of 4.4 and the colour stage of 4.3,
which read what it leaves. The intermediate is always RGBA16F whatever
the output space, G is the channel the sensor sampled twice as densely,
and R - G and B - G are the colour differences the debayer built its
chroma in.

Definition, three passes:

    quarter grid: Wq = ceil(W / 4), Hq = ceil(H / 4); texel q holds the
    mean over the pixels of block [4q, 4q + 4) that lie inside the frame of
        (d_r, d_b, G) = (R - G, B - G, G)

    bilateral, at every texel q, borders mirrored on the quarter grid,
    with L = max(G_q, 0) + eps the centre's level:
        w(s) = exp(-(dx^2 + dy^2) / (2 sigma_s^2)
                   - ((G_s - G_q) / (K L))^2
                   - ((d_r,s - d_r,q) / (K_c L))^2 - ((d_b,s - d_b,q) / (K_c L))^2)
        f(q) = sum w(s) d(s) / sum w(s)        over |dx|, |dy| <= R
    with R = 4, sigma_s = 2 quarter texels, K = 0.35, K_c = 0.35,
    eps = 0.02, the sums row by row, left to right

    recombine, at every pixel p, with `up` the bilinear interpolation of
    a quarter image at (p + 0.5) / 4 - 0.5 in texel centre units, the
    coordinates clamped to the grid, the four corners summed top left
    first:
        d' = d + strength * (up(f) - up(d_quarter))
        R = G + d'_r,  B = G + d'_b                G and alpha unchanged

Only the band the quarter grid holds is replaced: what the stage puts
back is the difference between the smoothed and the unsmoothed block
means, so chroma detail finer than a block, which section 17 has already
dealt with, passes through unchanged, and a strength of one replaces the
band completely. The range terms are on the *block means*, relative to
the centre's own G so that they mean the same in a shadow and a
highlight, and that is the one difference from the design as first
written: a range term on G per pixel would fence off every glint and
keep the colour this stage exists to remove. On the block mean a glint
smaller than a block is invisible to it, while a real luminance edge,
which spans blocks, still stops chroma from crossing. The second range
term, on the colour differences themselves, is what stops it at a colour
edge of equal luminance, which the G term alone cannot see: on the
synthetic scene the neutral flat patch sits beside a strong gradient at
a luminance 24% away, and with the G term alone the gradient's chroma
walked 16 pixels into it and moved the warm patch's centre by 6%; with
both terms the warm patch moves by 0.0%. K_c is a trade: tighter
protects glints too, since their block means differ in chroma as a
small coloured object's do, and looser lets the surround's chroma into a
flat patch as a trend, which on the synthetic scene is what happens at
0.5. At 0.35 the road's flicker reduction is 6% less than with the G
term alone and the neutral patch keeps 49% of blotches painted onto it
against 62% at 0.5. The frame must be at least
20 pixels on each side, so that the quarter grid's mirror is good for
one reflection at radius 4.

Measured on the GPU output of a 128x128 crop of the road of 17.1 over
four consecutive frames, RCD, on top of two false colour passes: the
mean frame-to-frame change of the colour differences goes 0.055 (plain)
to 0.034 (two passes) to 0.023 (both stages at full strength), the mean
saturation of the brightest tenth 0.29 to 0.25 to 0.22, and the
frame-to-frame change of G stays at 0.05, the rain. On the 320x240 crop
of the neon signs, two passes move 2.5% of pixels' chroma by more than
0.1 and both stages 4.1%, the crop's mean saturation going 0.250, 0.236,
0.224: the thin ornaments lose a little, the flat reds and the text
keep theirs. An unguided Gaussian of the same width, tried in emulation,
reaches the same flicker but bleeds red into the white of the signs
until the crop's saturation goes back up. What remains after both stages
is at least partly real: the wet road reflects the signs, and moving
water changes which colour lands where.

Cost: one read of the full resolution intermediate into 1/16 of an
RGBA16F image, an 81 tap bilateral over that quarter image, and one more
read and write of the full resolution intermediate for the recombine; the
two quarter images are 8.75 MB each at 5.8K, and the full resolution
scratch image is shared with section 17.
Measured with `cineraw --bench` at 5.8K, RCD: 5.8 ms a frame without it,
7.2 ms with it at full strength, 1.4 ms for the stage, and the same 1.4
ms on top of two false colour passes. That is above the 0.4 ms first
estimated, and the difference is the two full resolution reads and one
write: folding the recombine into the colour stage, which reads the
intermediate and writes the output anyway, would take most of it back,
at the price of the histogram seeing the unsmoothed chroma. Not done;
1.4 ms fits. The downsample was an invocation per block reading its
sixteen pixels, 0.65 ms of that; it is an invocation per column of a
block now, whose four loads are runs of a row across the subgroup, and a
quad broadcast (DPP) gathers the block's columns into its first lane,
which sums them in the reference's order: 0.49 ms, the same bits.

### 14.5 Parameters and API

`CRAW_PIPELINE_FLAG_DENOISE` builds the raw pass and its extra full
resolution R16_UNORM image; `CRAW_PIPELINE_FLAG_CHROMA_DENOISE` builds the
chroma stage of 14.4, its two quarter resolution images and, without RCD
and without section 17's flag, the full resolution scratch image. Per
frame, `CRAWFrameParams` carries:

- `denoise_strength`, the h of 14.3. 0 skips the pass for that frame
  without rebuilding the pipeline and is the default; the skip is free
  (measured above), which is what makes this a live control.
- `noise_override`, a `CRAWNoiseModel`. When its `a` is positive it is used
  as given; otherwise the profile's table is evaluated at the profile's
  as-shot exposure index, and a profile with no table makes
  `craw_pipeline_record()` fail with `CRAW_ERROR(ENOTSUP)` whenever the
  strength, or the temporal strength of 14.7, is positive. Failing is
  right: there is no default noise level, and silently doing nothing would
  look like a broken pass.
- `denoise_chroma_strength`, the blend of 14.4, 0 to 1. 0 skips the stage
  for that frame and is the default; above 1 is `CRAW_ERROR(EINVAL)`, the
  band cannot be replaced by more than all of itself. It needs no noise
  model: the bilateral's range term is relative to the level.
- `denoise_temporal_strength` and `denoise_temporal_replay`, with
  `CRAW_PIPELINE_FLAG_TEMPORAL_DENOISE` and
  `CRAW_FRAME_FLAG_TEMPORAL_RESET`: the temporal pass, section 14.7.

`CRAWProfile` grows `noise_knots` and `nb_noise_knots` at the end of the
struct, which changes its size; it is plain caller-filled data with no
padding tail, so this is an ABI break and the soname covers it.
`CRAWFrameParams` and `CRAWPipelineParams` keep their sizes, the new fields
coming out of their padding tails as intended.

`cineraw` gains `--denoise STRENGTH[,CHROMA]`, `--noise A,B`, and
`--noise-estimate`, and its 8-bit writer gains the triangular dither of
14.3. It also gains `--crop X,Y,W,H`, which writes and takes `--stats` over
one rectangle of the output while the pipeline still runs on the whole
frame: it is how a 5.8K frame gets looked at 1:1 without a 5.8K
file, and every number in 14.3 was measured through it.
`craw_tool_profile_load_tables()` accepts a `noise ISO A B` line, and a
file without one leaves a built-in profile's noise table alone.

### 14.6 Testing

`test_pipeline` covers the raw stage four ways, and the chroma stage of
14.4 the way section 17.5 covers its neighbour: both ways off reproduce
the plain chain, half and full strength on MHC (its own scratch image) and
on RCD (RCD's aux) match the reference within twice the debayer's
intermediate bound and sit at least three floors from the unfiltered
chain, a strength above one is refused, and on the CPU chroma blotches of
+-0.02 per 8x8 block painted over the neutral flat patch keep at most 60%
of the spread of their 4x4 block means while the centre of the warm flat
patch keeps its colour differences within 5% and G is untouched bit for
bit. On the real frame it runs at full strength
on top of two false colour passes and is held to twice the plain chain's
absolute error.

The raw stage:

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

### 14.7 Temporal denoising

A spatial filter has one frame's worth of samples. A temporal one has as
many as the scene holds still for, and for the shadows of a locked-off shot
that is the difference between 4.5x and something much larger. It attaches
to the raw stage of 14.3, not to the output.

#### Where it sits

**Inside the raw stage, in the stabilised domain, before the spatial
filter**, in the same dispatch. The domain is the whole reason: in
stabilised units one number is one noise standard deviation at every level
and in every frame, so the decision "is this sensel the same as last
frame's" is a comparison against a constant, exactly as the spatial patch
distance already is. After the debayer the same decision would be made on
noise that has already been mixed and coloured.

The spatial filter then works on what the temporal one left, and the state
carried to the next frame is the temporal result *before* the spatial
filter: fed back through it, a static scene would be filtered again every
frame and smoothed further without end.

The state holds the level u of every sample (`I(t)`), not its stabilised
value, and the next frame puts it through its own transform. A frame with
another noise model — a different exposure index, a `noise_override` the
caller changed — then still reads last frame's state correctly. It is fp32
for the reason 14.3 gives. Beside it the state keeps the noise variance
left in each sample, in stabilised units, which the spatial filter needs
(below): its square root in an R8_UNORM image, a step of 2.4% of the
variance at an alpha_max of 0.8.

#### Definition

For every sample p of each CFA plane, with t = T(u) of 14.3, s the state's
level and v_s its variance, and the plane mirrored as in 14.1:

    tc(p)  = T(u(p))                                   this frame
    tp(p)  = T(s(p))                                   last frame, filtered
    d2(p)  = (1/9) sum over the 3x3 k of (tp(p+k) - tc(p+k))^2
    a(p)   = alpha_max * exp(-max(d2(p) - 2, 0) / ht^2)
    t'(p)  = (1 - a) tc(p) + a tp(p)
    v(p)   = max((1 - a)^2 + a^2 v_s(p), 1e-4)
    state' = (I(t'(p)), v(p))

With no history (the first frame, a reset) a = 0, t' = tc and v = 1. The
floor on v only keeps the division below finite at an alpha_max of 1. The
2 is the `2 sigma^2` of 14.3 and `ht` is the spatial `h` when the spatial
filter runs and 1.0 when it does not, so that one strength means the same
in both. The sum over the patch is taken row by row, left to right, as the
spatial one is. alpha_max bounds how long a sample survives, which bounds
what a missed motion can smear: on a static scene the variance converges
to (1 - a) / (1 + a), 0.11 at 0.8, a third of the noise in standard
deviation, and a wrong decision is forgotten at a rate of a per frame. At
1.0 a static sample keeps its first frame for ever and never takes in
another, which is why cineview's row stops at 0.95.

The spatial filter of 14.3 then runs on t' with two changes, both only
where the pass had a history: the patch distance loses the noise the two
samples actually carry, v(p) + v(q) instead of 2, and the strength is
taken in units of the noise left in the sample being filtered:

    w(q) = exp(-max(d2(q) - (v(p) + v(q)), 0) / (h^2 v(p)))

Where nothing was averaged v = 1 and this is 14.3 exactly. The highlight
guard of 14.3 compares against the sample as it came in and hands it
through as it came in, temporal average and all.

**A patch, not the sample alone.** The design first compared the single
sample, `d = |tp - tc|`. That compares a sample against its own noise:
every excursion past a sigma and a half reads as motion and passes through
unaveraged, and those are exactly the samples the average was for. On
static Gaussian noise the single sample converges on 0.84 of the noise and
the 3x3 reaches 0.44 by the tenth frame (a numpy model of both); on 16 raw
frames of the flat panel of P1094353 below, 0.85 on every CFA plane
against 0.50 to 0.58. The 3x3 is also the less trigger-happy with a
moving edge: in the same model a bar 3 sigma above its background leaves
0.14 sigma behind it with the single sample and 0.02 with the patch.

**The spatial filter's noise.** The first version left the spatial filter
alone, subtracting 2 and measuring h against the noise of one frame. On a
temporally averaged tile, whose samples carry a third of that, it took
every true difference up to 1.3 sigma for noise and removed the fine
detail the average had just uncovered: on the P1094353 panel below it kept
0.5% to 2% of the fine detail, where the spatial filter alone keeps 13% to
14%.
Subtracting the variances alone gets some of it back (6% to 9% in a model
on the raw frames); scaling h with the remaining noise as well is what
makes the combination better than the spatial filter on both counts. The
variance has to be tracked rather than inferred from a: the steady-state
guess (1 - a) / (1 + a) takes the first frames after a reset, and after
motion stops, for already averaged, and those frames came out noisier
than with the spatial filter alone (13.3 against 7.6 codes on the frame
after a reset, in the same model); tracked, every frame is at or below it.

Motion compensation — searching the previous frame over a window instead
of reading it at p — is the expensive end, needs a real motion estimate,
which this library does not have and should probably take from the
caller, and is not done.

#### State, replay and reset

The pipeline keeps two sets of state and two indices: `tn_cur`, the set
the last recording wrote, and `tn_prev`, the set it read; -1 means nothing
valid, which is where both start. A recording that is the next frame
reads `tn_cur` and writes the other set; one flagged as a replay
(`denoise_temporal_replay`) reads `tn_prev` and writes `tn_cur` over; a
`CRAW_FRAME_FLAG_TEMPORAL_RESET` reads nothing. The two indices move only
once the recording has succeeded. That gives:

- The first frame, and the frame after a reset, are developed as if the
  pipeline had just been created.
- A frame developed again is bit for bit its first development, however
  often and whatever was changed and changed back in between, and the
  frame after it follows the first development. A grade changed while
  paused therefore never averages the frame with itself.
- A next frame recorded with the pass off leaves nothing valid, so the
  frame after it starts again; a replay with the pass off leaves the
  state as it was, so turning the pass off and on again while paused
  returns the same picture. Turning it on while paused on a frame that
  was developed without it, after frames developed without it, has no
  history to use and shows the spatial filter alone until the next frame.
- Two sets rather than one are what the replay needs, and the fused tile
  load needs them anyway: a workgroup reads last frame's state in its
  halo, which a neighbouring workgroup writing in place could already
  have overwritten.

The state is read and written only by recordings, so the barriers that
order consecutive recordings order it too, across submissions in flight:
this frame's read after the last one's write, this frame's write after the
last one's read, and a replay's write after the first development's, which
is why the pass leaves the image it wrote in the read state. Recordings
have to be submitted in the order they were made, which was already true
of the intermediates (11b). A recording made but never submitted has
moved the indices all the same, and the caller should reset.

What breaks the run is the caller's to say, and cineview says it on a seek
to another frame, the loop back to the start, a new clip, and the first
frame of a video export, which is sought even when it is the picture on
screen, so that the same range exported twice is the same file. It does
not start the average at the frame a seek lands on, though, which would
make the picture after every seek, and the first frames of every export,
the noisiest of their run: a jump starts it five frames earlier and
develops those frames without showing them, the first with the reset and
the rest as next frames, and the target is then the sixth frame of an
average rather than the first. Five is 1 / (1 - alpha_max) at the 0.8 the
section is meant for, the average's reach, and leaves the frame it
started from a weight of 0.8^5 = 0.33 in the target's state; a higher
alpha_max is warmed only partly. It is a constant, `VIEW_PREROLL`, not a
setting. A frame of the pre-roll is developed for its state and nothing
else: the passes after the raw stage, which cannot change the state, are
left out (the target comes out bit for bit the same as after a pre-roll
with them, 20 iterations of sharpening among them), it is recorded into
the output slot the next picture will use, which is never the one on
screen, without moving the slot on, and it is not timed, counted, read
for the white balance or shown. The seek decodes its target before it
returns, and the clock only starts after that, so the audio is not sent
off ahead of a picture the pre-roll made late. A scene-referred export
puts the same frames through its own pipeline, which is built for them
rather than when the output is opened, so the two average the same
frames. A pre-roll that cannot finish - a frame that does not decode,
the end of the clip, a medium that delivers nothing for a second - is
abandoned, the decoder is sent to the target again and the target starts
the average, as every seek did before: an injected failure gives a still
and a scene-referred range bit for bit the ones the viewer wrote before
there was a pre-roll. While the scrub bar's knob is held its seeks come
as a stream and go without, since each would pay for a pre-roll and the
picture would follow the knob at half the rate (7 positions on a drag
that shows 13 without); the frame it is let go on is sought once more,
with one. The first frame of a clip has nothing before it.

The pre-roll took 70 to 170 ms a jump in the window at 5.8K on the 6900 XT,
82 ms the median of 63 and the slowest with other work on the GPU, the
target's own decode included: six decodes and five passes that the device
runs one after another, the wait for the medium no more than 10 ms of it.
The raw path has no software decode of a clip to cap it for: ProRes RAW is
decoded by the Vulkan hwaccel, so every frame of a pre-roll is a decode on
the device and none of them can cost what a decode on the CPU would. The
one raw source decoded on the CPU and uploaded is a still (section 24),
which has nothing before it to pre-roll; a ProRes RAW clip whose hwaccel
would not start takes the same upload, frame by frame, and pays for it.

A frame the player drops to keep up is not a break: the next one is still
the next frame of the same scene, further on, and the pass takes what
moved as motion. A development of the frame last developed is a replay
whether it comes from a grade change, a white pick or a seek to the
picture already on screen. A still export of the frame on screen is the
picture on screen; of any other frame, a seek and its pre-roll, except
that a frame up to eight ahead is reached by developing the frames
between, which carries the average forward.

A scene change detector — the fraction of samples whose patch distance
exceeds the threshold, handed back like the AWB histogram — was designed
here and is not built. A cut moves nearly every patch far past the
threshold, so the pass already takes nothing from the frame before it
everywhere at once; the detector would only be a diagnostic. Not measured
on a real cut.

#### Dispatch and cost

One dispatch for both filters, as in 14.3, with a mode in the push
constant saying which run and whether there is a history. With a history
the workgroup loads this frame and last frame's state, stabilised, one
sample further out than the spatial tile (26x26 for a 16x16 workgroup and
the 7x7/3x3 filter), works out t' and v for the tile, and the spatial
filter runs on that; the temporal value of a sample is a function of the
3x3 around it, mirrored, so every tile that holds a sample holds the same
number. Shared memory is 13.5 KB (16 KB since the subgroup version of
14.3, whose tile is 30 x 16). With the spatial filter off the tile is
the workgroup plus the one sample.

The four planes of a tile are four neighbouring workgroups in x rather
than a slice of `gl_WorkGroupID.z` each, as they were. Each plane touches
every other sample of every other row of the mosaic and of the state, so a
plane at a time streamed the whole 70 MB state four times over for a
quarter of every line it fetched; side by side the four share the lines.
That took the temporal pass alone from 1.9 ms to 0.84 in `cineraw
--bench`, and the whole pipeline with the spatial filter, which is not
bandwidth bound, from between 6.15 and 6.18 ms to between 6.09 and 6.12.

RX 6900 XT, 5760x3040, RCD, two decoded frames of P1094353 alternating,
back-to-back submissions, median of 20; four runs, the last two with
nothing else on the GPU:

    raw stage                         pipeline         stage
    built, both at 0 (nothing runs)   4.13 to 4.16     --
    spatial 1.0                       6.06 to 6.12     1.95
    spatial 1.0 + temporal 0.8        6.65 to 6.71     +0.58 on top
    temporal 0.8 alone                4.79 to 4.81     0.65

Of the 0.58, about 0.08 is reading and writing the variance and 0.12 its
use in the spatial filter; the version without either measured +0.38.
Temporal alone includes the stage's own read of the mosaic and write of
its output, which the spatial filter pays too. `cineraw --bench --frames
60` on the same clip, which keeps the GPU idle between frames (14.3),
reads 7.40 to 7.43 ms off, 9.76 spatial, 10.46 both and 8.43 to 8.45
temporal alone over two runs. The state is 175 MB at 5.8K.

#### API and tools

`CRAW_PIPELINE_FLAG_TEMPORAL_DENOISE` builds the state and, with or
without `CRAW_PIPELINE_FLAG_DENOISE`, the raw stage it runs in.
`CRAWFrameParams.denoise_temporal_strength` is alpha_max, 0 to 1: zero
dispatches nothing the pipeline would not otherwise and is free (above),
above one is EINVAL at record time, and it needs the noise model exactly
as `denoise_strength` does, ENOTSUP without one.
`CRAWFrameParams.denoise_temporal_replay` and
`CRAW_FRAME_FLAG_TEMPORAL_RESET` are the replay and the reset above. Both
fields come out of the padding tail.

`cineraw` takes the temporal strength as a third component of `--denoise
STRENGTH[,CHROMA[,TEMPORAL]]`; it hands the pipeline its frames one after
another, and a rebuild for a new frame size or a lens record starts the
average by itself. cineview's Denoise section gets a `temporal` row, 0 to
0.95 and 0 by default, under the section's switch like the rest, and its
strength row now reaches 0, the spatial filter off, so that the temporal
average and the chroma stage can each be looked at alone; `--denoise`
takes the same third component, and a strength of 0 with a chroma or
temporal amount above it switches the section on with the spatial filter
off. The viewer builds the pass into both its pipelines, the display's
and the scene-referred exporter's.

#### Testing

`test_pipeline` runs a sequence of its own: eight frames of a scene whose
noise is drawn afresh every frame at exactly the model's level (the model
has to be the truth here, since what the pass decides is whether a
difference is noise), with a bar far above its background (0.40 on 0.10)
and one 2 sigma above it sliding 12 pixels a frame, a checkerboard far above the
noise and a faint one 1.5 sigma from square to square. Through one
pipeline and through `craw_ref_denoise_temporal()`, frame by frame, at
spatial strength 0 and 1:

- GPU against the reference within the denoise bound of 14.6, 2e-3 on the
  MHC intermediate: 5.4e-4 and 8.6e-4 at worst over the eight frames.
- Temporal alone, on the reference's own mosaics: a flat patch keeps 71.7,
  60.0, 50.2, 43.2, 40.0, 36.3, 34.8, 31.9 counts of noise over the
  frames, every frame quieter than the first and the last under half of
  it; both checkerboards keep their detail (the faint one 1.01 of it, by
  the regression of the high pass on the truth's).
- With the spatial filter: the patch keeps 3.2 counts where the spatial
  filter alone keeps 5.2 (at most 0.75 of it is allowed), and the faint
  checkerboard 0.77 of its detail where the spatial filter alone keeps
  0.21 (it may not keep less).
- What the bars leave behind them, the strip each vacated since the last
  frame: the strong one at most 0.09 sigma from the background (0.25
  allowed), the faint one 0.25 to 0.31 sigma (0.6 allowed), and on the
  GPU's own output 0.15 to 0.18.

Then the state, bit for bit, the GPU against the GPU: strength 0, the flag
not set, and a first frame of the temporal average alone are all the plain
chain; frame 4 developed six times, with other settings, with the pass off
and back, at the original settings is its first development every time,
and frame 5 after it is frame 5 of an unreplayed run; a reset is a first
frame and changes the output of a warm pipeline; the frame after one
without the pass is a first frame; four frames recorded into four command
buffers and submitted back to back without a wait, each into its own
images, match the sequential run. A strength of 1.5 is refused, and the
pass without a noise model is ENOTSUP.

#### On real footage

P1094353, a phone booth at night, locked off, ISO 10000, with the model
fitted to the clip itself (`--noise 1.19e-4,2.95e-8`; the profile's table
then ended at the 1250 knot, eight times too small in a here, and now
gives 1.18e-4, 2.65e-8, below). A flat metal
panel, 600x290 pixels of the linear BT.709 output at a mean G of 0.035,
the residual after a quadratic surface over frames 20 to 29, the frame to
frame difference, and how much of the panel's fine detail survives (the
regression of the 3x3 high pass of frames 27 to 29 on that of the mean of
the undenoised frames 0 to 15):

                  R        G        B        R-G      B-G      frame diff G  detail
    off           1.98e-2  9.38e-3  1.18e-2  2.11e-2  1.27e-2  1.30e-2       1.00
    spatial 1     5.60e-3  2.50e-3  3.55e-3  6.05e-3  4.09e-3  3.05e-3       0.13
    temporal 0.8  1.04e-2  4.78e-3  6.66e-3  1.09e-2  7.16e-3  3.73e-3       1.04
    both          3.89e-3  1.90e-3  2.68e-3  3.99e-3  2.90e-3  1.42e-3       0.22

The frame difference, which static texture cannot enter, is the cleanest
of these: the two together flicker 2.2 times less than the spatial filter
alone, while keeping 1.7 to 1.8 times the detail (0.22 against 0.13 on
the flat part, 0.27 against 0.14 on the panel's slotted lower half). After the start the difference is 2.00e-3,
1.71e-3, 1.54e-3 and 1.40e-3 by the fifth frame, against a steady 3.05e-3.
The temporal average alone reaches half the noise rather than the third
the definition allows, and the raw frames say why: this clip's noise is 3%
to 6% above even its own fitted model in stabilised units, which puts
more patches over the threshold (a mean a of 0.757 at the sixteenth frame
against 0.780 for noise at exactly the model), and the residual counts
the panel's own texture, which the average rightly keeps.

P1094352, the same booth with three people walking past at 50 to 100
pixels a frame. Behind the bright edges of an arm and a bag against the
dark foliage nothing is left: at 1:1 the frame after them is the spatial
filter's alone. Dark hair passing over the dark foliage is different: the
two differ by 0.56 of the undenoised noise, less than the pass can tell
from noise, and two frames after the hair has gone the foliage there sits
0.21 sigma above where the spatial filter alone puts it, 38% of the
contrast and 0.74 of the spatial filter's own residual noise — a faint
lighter smear, visible at 1:1 with the picture brightened four times. That is the price of the
single threshold: detail that moves at less than a noise standard deviation
of contrast is averaged in time as the spatial filter averages it in space.
Lowering alpha_max shortens it; so would subtracting the noise the two
frames actually carry (1 + v_s) from the patch distance instead of 2,
which would make the test about twice as sensitive to a sub-sigma change
and cost part of the noise reduction on a static scene. Not tried. The
static foliage beside the walkers goes from a frame difference of 3.02e-3
to 0.80e-3 with the spatial filter and 0.45e-3 with both.

The model matters more here than to the spatial filter. cineview takes it
from the profile's table, which for this ISO 10000 clip used to be the
1250 knot: the pass saw every frame's noise as motion and an export's frame
to frame difference did not fall at all (0.032 on every frame of 10 to 25).
On P1000387 (ISO 1000, the table's own knot) an export of frames 5 to 20
goes from a frame difference of 0.0020 with the spatial filter alone to
0.0013, 0.0011, 0.0009 and 0.0007 by the end with both, through the
display's pipeline, and from 0.00111 to 0.00075 and then 0.00040 through
the scene-referred exporter's. The ISO series of 14.2 is what makes the
pass work at every ISO in the viewer. With its table and no override the
panel above reads, spatial at 1 and temporal at 0.8, a frame difference of
1.50e-3 against 1.41e-3 with the clip's own fit and 1.27e-2 with the old
1250 knot (off, 1.30e-2). Most of the gap is the table's b, 10% under the
clip's: the table's a with the clip's b reads 1.43e-3. Through cineview's
video export of the panel alone, the display look, the difference is
0.043 off and 0.0103 with the spatial filter, and with both 0.0071, 0.0060
and 0.0054 over the first frames and 0.0050 to 0.0056 after them.

The viewer's pre-roll, on the same two clips with the clip's own model
(given to cineview as a `--tables` noise knot), spatial 1 and temporal
0.8. A still of P1094353's frame 40 against the same frame developed after
all forty before it, which is what playing to it gives: started cold, the
sRGB still is 512 16-bit codes away on average and the lower panel's
high-pass noise is 1.071 of the warm frame's; after the pre-roll 182 codes
and 0.997. Eight frames of pre-roll bring the difference to 111 codes and
ten to 80, for proportionally more time. The scene-referred frames 30 to
34 against the same frames of an export of 0 to 34: an RMS difference of
6.58e-3 at frame 30 started cold and 1.99e-3 after the pre-roll, 2.01e-3
and 0.81e-3 by frame 34, and the median frame to frame difference of G
from 30 to 31 is 1.31e-3 cold, 0.96e-3 after the pre-roll and 0.98e-3 in
the whole clip. On P1094352, frames 60 to 64 with the walkers in them,
5.78e-3 and 1.27e-3 at frame 60 and 1.87e-3 and 0.51e-3 at 64. Exported
twice, or with its first frame already on screen, a range is the same to
the bit.

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

On the camera RGB intermediate, after the debayer and the false colour
stage of section 17 and before the defringe of section 19. Section 1
originally reserved it as "a per-channel resampling warp after the debayer
and before any scaling"; the warp needs the three channels in the same
image, which is anything after the debayer, and it must not run on the
mosaic, where resampling a CFA would be meaningless.

The first version ran it after the colour pass, on the linear output, on
the argument that it then resamples the values the caller is going to
look at. That was wrong for the part of the record that matters most here.
The record's three polynomials are the camera's: the lateral chromatic
aberration is a property of the light each of the sensor's filters passes,
so its correction belongs to the camera's R, G and B, and warping the
output space's R, which is a mixture of all three, moves the camera's G
and B inside it by R's amount (16.7). Nothing else pulls the other way: the
common, distorting part of the three polynomials commutes with the colour
pass, which is per pixel and linear up to the gamut compression, so where
it runs changes nothing about it.

It also has to come before the defringe. The defringe clamps the colour
of an edge to what its two sides have, which aligns the three channels'
edges at the level's edge; a warp that shifted R and B afterwards would
pull them off it again, into a fringe of the opposite colour as wide as
the correction. At the top left corner of P1001259.MOV, leaves against the
sky, where the record separates R from B by about 2 px, the mean colour
at the strong edges of a 500x300 crop (the distance in ln R/G, ln B/G)
was 0.293 with the warp on the output and 0.274 with the defringe ahead of
it, against 0.167 with the defringe run on the warped output; with the
warp on the intermediate, ahead of the defringe, it is 0.205. The false
colour stage stays ahead of the warp: its medians work on the sensor's
own grid, where the debayer's single pixel errors are single pixels.

The warp cannot run in place, so it writes the second intermediate of
section 3 and the two swap, like the false colour stage: the pipeline's
scratch image (RCD's aux image until RCD became one dispatch). The colour pass then
writes the caller's destination whatever the flags. Running on the fp16
intermediate costs a caller who asked for RGBA32F output nothing it had:
the intermediate the colour pass reads was fp16 before the warp too.

Being ahead of the colour pass, the warp's overshoot now goes through the
gamut compression of section 8.1 like everything else. Its Catmull-Rom
kernel has negative lobes, and at a hard edge in the camera's channels
that can still come out as a small negative in ACEScg: on P1000387 frame
0 with cineraw's defaults, 25 pixels in 17.5 M have a negative component
with the warp and without the defringe, against 14 without the warp, and
none with both. `CRAW_FRAME_FLAG_CLAMP_NEGATIVE` removes the rest.

The auto white balance histogram reads the intermediate after the warp
(section 4.4). A geometric resample of the picture does not change what
illuminant lit it; 16.8's black border is black in all three channels, so
a block entirely inside it falls below the histogram's floor and one that
straddles it keeps the ratios of its picture pixels.

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
the same frames read 5.3 to 6.2 ms with the stage on. Those were measured
with the warp on the output; on the intermediate (16.3), over 30 frames
of P1000352.MOV with the highlight reconstruction and the defringe off,
it is 5.18 against 5.87 ms, 0.69 ms, and with RGBA32F output 5.47 against
6.27 ms, 0.80 ms, where it used to read and write the 32-bit image.

That is 49 image loads per pixel — sixteen taps for each of three channels
plus one for alpha — on 17.5 million pixels, which only fits in 0.8 ms
because neighbouring output pixels read overlapping footprints and the
cache absorbs almost all of it: the traffic that actually reaches memory is
one read and one write of an RGBA16F frame, 280 MB, about 0.55 ms at this
card's bandwidth. There is a shared-memory tiling to be had here if the
budget ever gets tight — the source footprint of a 16x16 output tile is a
16x16 source tile plus a three pixel halo, wherever the warp has moved it —
but at 0.8 ms it would be optimizing the wrong stage.

Both were tried in the round of section 21, back to back at 5.8K with a
60 mm record. The alpha load was not free: the warp now writes the 1.0
every debayer puts there and every stage keeps, and the pass went from
0.77 ms to 0.70. The tile (the box of every tap of a workgroup's pixels,
found with subgroup minimums and maximums and a shared atomic per
subgroup, loaded once and read 48 times a pixel) took it to 0.68, which
was not worth its hundred lines: the loads were never the bound, the
cache having absorbed them as said above, and the pass is the one read
and one write of the frame plus the taps' arithmetic.

A frame with no record costs nothing at all, because the pass is not
recorded: measured at **0.037 ms** difference on the 256x160 test frame,
i.e. inside the noise of the measurement.

### 16.7 The channels, and what is not verified

**The record's channels are the camera's, and so are the warp's.** The
first version ran the stage after the colour pass and so applied the R
polynomial to the output image's R, a linear combination of the camera's
R, G and B. For the radial distortion that is exact, the common part of
the three polynomials being the same for every channel; for the
*difference* between them it is not, and the stage now warps the camera
channels ahead of the matrix (16.3).

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

The corners say the correction is at least not wrong way round, and
leave its size open. With the warp on the camera channels and the
defringe off, the lateral part of the record scaled by a factor a
(k_R and k_B moved to k_G + a (k - k_G)), the mean colour at the strong
edges of the 1100x700 corners of three clips:

    corner           a = 0    0.35    0.6     1      R, B swapped   negated
    P1001259 TL      0.300   0.257   0.311   0.486       0.403       1.016
    P1001259 BR      0.492   0.331   0.270   0.302
    P1000352 BL      1.016   0.837   0.841   0.988
    P1000352 BR      0.929   0.802   0.813   0.989

(the swapped and negated columns on the 500x300 crop of 16.3, where a = 0
reads 0.329 and a = 1 0.478). Negated or swapped is worse than no
correction at all, so the order and the sign stand. But the best a is
0.35 at one corner and 0.6 to 1 at the opposite one of the same frame,
which is not what a record that is merely too large would do; a centre
that is not where the record says, which the body's stabilisation would
move from frame to frame, would. On P1001123, a night street, every
column reads within 1% of the others: the corners' edges are noise.

### 16.8 Keeping everything, with black borders

The stage as described so far fills the output frame: every output pixel
gets a source, and at the wide end, where the correction is a barrel
correction pulling the periphery outward, the source's own corners are
pulled past the frame and lost — 233 px past the corner on the 24 mm
frame of P1000352.MOV, which is what `k0` sitting at 0.999 there means.
The camera chooses that trade. `CRAW_FRAME_FLAG_LENS_KEEP_ALL` makes the
other choice: shrink the corrected picture until the whole of the source
fits inside the frame, and paint everything outside it black.

**The mapping.** One global scale `s <= 1` per frame, the same for all
three channels so that the chromatic correction stays consistent. For an
output pixel at normalized position `p` (centred, half-diagonal units),

    r' = |p| / s
    source_c = centre + p * poly_c(r') / s

which is the fill mapping of 16.2 evaluated at `p / s`. It extrapolates the
polynomial a little past r = 1 — to r = 1.07 at the corner of the 24 mm
frame — which is fine: it is a smooth even polynomial and the coefficients
are small.

**The scale.** `s` is the largest value for which every source pixel has an
output position inside the frame. The source rectangle's boundary is
walked, 512 segments a side and each side carrying its starting corner, so
2048 points; an even number of segments puts a sample exactly on every
corner and every edge midpoint. For each boundary point at radius `R` (half
diagonals from the centre):

    needed = max over channels of invert_c(R),    invert_c = inverse of r * poly_c(r)
    s      = min over the boundary of frame_radius(theta) / needed,  capped at 1

`invert_c` is found by bisection over [0, 2], forty halvings, in double.
`frame_radius(theta)` is where the ray leaves the frame rectangle, which
for the camera's centred record is `min(half_w / |cos|, half_h / |sin|)`
in pixels over the half diagonal; `craw_lens_keep_all_scale()` writes it as
a per-side ray-rectangle intersection so that an off-centre record measures
against the edge it is actually heading for.

Because the source rectangle and the output frame *are* the same rectangle,
`frame_radius(theta) = R` at every boundary point, and the ratio reduces to
`poly_c(invert_c(R))`: it depends on the boundary only through `R`. For a
polynomial that is monotone over the range — every record seen — the
smallest ratio is therefore at the smallest or the largest `R`, i.e. at an
edge midpoint or at a corner, and the sampling in between is insurance
against a polynomial that turns over. Which one binds says what the record
is doing:

- **barrel** (24 mm, poly falling with r): the corner binds.
  `s = 1 / invert(1)`. Measured on P1000352.MOV at 0.96 s: **0.9332**.
- **pincushion** (60 mm, poly rising with r): the short edge midpoint
  binds. Measured on P1000387.MOV: **0.9772**.
- **a pure crop** (poly constant at k0): every boundary point binds equally
  and `s = k0` exactly.
- **a correction that samples outward everywhere**: the source already fits
  and the cap holds `s` at 1; the corners then come out black rather than
  being magnified to fill, because "keep everything" is not "fill with
  something".

The cap at 1 is what makes fill mode the default and this mode a shrink.

**The border.** An output pixel is black when the source position of *any*
channel falls outside the source rectangle — one boundary for all three, so
the border is one shape and not three, which is what would show as a
coloured fringe along it. The cost is a sliver: the scale is set by the
channel needing the *largest* output radius and the border by the channel
whose source leaves *first*, and between them lies the lateral chromatic
aberration, 2.4 px at the corner of the 24 mm frame. That sliver is black
although one channel still had data there, which is the right way round.
Inside the border nothing changes: the same sixteen-tap Catmull-Rom with
the same mirrored edge rule.

Alpha stays exactly 1.0. It is read from the output pixel's own position,
as it is everywhere else in this stage (16.4), and the colour stage wrote
1.0 there: the border is black, not transparent, and the library has no
transparent output to offer.

**Where the scale is computed.** On the host, in
`craw_lens_keep_all_scale()` in `lib/lens.c`, and it reaches the shader as
`1 / s` in the push constants. It is a property of the record and the frame
size, so it changes frame by frame on a clip that is zooming and not at all
on one that is not; the pipeline keeps the last record and its scale and
recomputes only when the record's 64 bytes change. The walk costs **0.87 ms**
on this machine for a fresh record — 2048 points times three channels times
forty halvings — which is why the three bisections run in one loop with
their brackets moved by conditional assignment rather than by a branch:
written the obvious way it was 2.07 ms. Both tools call the same function
to show the number, `cineraw --lens keep -v` once per move of the lens and
cineview's Lens section in its readout.

**Filling the frame is bit for bit what it was.** The two modes are written
out as separate branches of the shader, and the filling one is the code
that was there before the mode existed, down to the expression. Sharing one
code path — computing the source position into an array and tapping from it
— left 1557 samples of a 5760x3040 frame one unit in the last place of the
fp16 output away from the old ones, because the driver's compiler contracts
`centre + d * poly - 0.5` into a multiply-add differently depending on what
else reads it. The push constant `inv_scale` is exactly 1.0 when filling
and a multiply by one is exact, so the two paths start from the same `d`.
Verified: `cineraw --lens on` on P1000352.MOV at 0.96 s, full frame,
linear float out, md5 `e7f1d3b6741c8add850992655e3e4348` from the build
before this change and from the build after it.

**Tests.** `test_lens` checks the scale itself against a second derivation:
the ratio identity above, with the inverse found by Newton rather than
bisection, evaluated at the corner and at both edge midpoints; it names
which one binds, checks that the ratio really is monotone between them, and
compares. On the real 60 mm record it is the short edge midpoint at
0.977178; on a ten-times barrel it is the corner at 0.846483, which is
`1 / r_c` with `r_c + k1 r_c^3 = 1` solved in closed form; on a pure crop it
is k0 exactly. Then the contract itself, on boundary points the library
does not sample (331 a side, offset half a step): every source point lands
inside the frame, and a scale a thousandth larger puts one outside, so the
answer is the largest one.

`test_pipeline` runs the mode on the 512x320 synthetic frame with the
exaggerated record of 16.5, whose corner binds at s = 0.846483. A mark at
0.95 of the half diagonal — which fill mode would have to show at output
radius 1.074, past the frame corner, and therefore does not have at all —
comes back at 0.21 px from where the scaled inverse puts it, well inside
the frame: 23 px from the right edge and 15 from the bottom. The two measurable marks land within 0.057 px of their
predictions, the same tolerance the fill-mode geometry check uses. 27496
pixels, 16.8% of that frame, are outside the mapped source and every one of
them is exactly black with alpha exactly 1; every pixel inside still
carries the flat field it was given; 36 pixels sit within a hundredth of a
pixel of the boundary and are only counted. The GPU is within 1.4e-3 of
`craw_ref_lens()`, against the stage's 8e-3, and the border is in the same
place in both, pixel for pixel. Filling the frame through the same pipeline
before and after the mode has run is compared with `memcmp`.

**On real frames.** P1000352.MOV at 0.96 s, 24 mm, s = 0.9332, exported by
`cineraw --lens keep`: the black mask of three 1000x1000 crops — the
top-left corner, the middle of the left edge and the middle of the top edge
— matches the mask computed from the record and the scale in double
**pixel for pixel**, 0 disagreements in 3 million pixels. 17 px of black on
the row through the left edge midpoint against 16.67 predicted, 66 px on
the column through the top edge midpoint against 66.46, and at the corner
only the 1.5 px chromatic sliver, because that is where this record binds.
On P1000387.MOV, 60 mm, s = 0.9772, the binding point moves to the short
edge midpoint and the corner carries 69 px of black against 68.79
predicted; there 27 pixels of 640000 disagree, all within 0.026 px of the
boundary, which is fp32 against double. A cloud edge 80 px from the left
edge of the 24 mm frame, which fill mode puts 99 px off the frame and which
is not in its export at all, is found in the keep export 0.64 px from its
predicted place with a correlation of 0.995.

**Cost.** Nothing measurable on the GPU. `cineraw --bench`, 12 frames of
P1000352.MOV, four alternating passes: fill 5.917 / 6.619 / 6.353 / 6.424
ms a frame, keep 6.684 / 6.302 / 5.994 / 6.375, which is the same number
inside a ±0.35 ms spread — the in-bounds test is six comparisons against
49 image loads, and the border pixels skip the taps entirely. The cost of
the mode is the host-side scale above.

## 17. False colour suppression

### 17.1 The artefact

A glint smaller than a CFA quad — rain hitting a wet road under street
lights, a specular on a wet leaf, a distant point source — lands on one
sensel, or on two or three straddling sites, and so is measured by one
colour and guessed for the others. No debayer can recover what was never
sampled: bilinear makes a saturated dot of the site's own colour, MHC and
RCD make a paler one because their correction terms push part of the spike
into the other channels, but every one of them makes a coloured dot, and
in a moving scene the next frame's glints land on other sites and come out
other colours. On P1094366.MOV, a wet street in the rain at ISO 1250, a
256x256 patch of the road holds about 250 glints of four times their
same-colour surroundings, most of them one to three sensels wide, and the
road shimmers in colour from frame to frame.

The frequency-domain view (Alleysson, Süsstrunk and Hérault, IEEE TIP
2005; Dubois, IEEE SPL 2005) says why the information is really gone: the
CFA signal is luminance at baseband plus two chrominance signals modulated
onto carriers at the corners of the spectrum, and a point source has energy
at the carriers that no demodulator can tell from chrominance. What a
debayer can do is decide, and the decision every raw converter makes is
that chrominance is smooth: an isolated pixel whose colour differs from
all of its neighbours is wrong, whatever its luminance. That is a median
of the colour differences, Freeman's 1988 patent, the final step of
Hirakawa and Parks' AHD (IEEE TIP 2005), dcraw's `-m`, darktable's
"colour smoothing" and RawTherapee's "false colour suppression steps".
RawTherapee's step, read from `rtengine/rawimagesource.cc`, is a 3x3
median of the two chroma planes followed by a 3x3 box blur of them, in
YIQ with Y kept, one to five steps, and it is the blur that reaches what a
median alone leaves: a glint two or three sensels wide is a blob the
median rightly keeps and the blur spreads and desaturates. This stage is
that step on the differences to G, with mirrored borders.

Two things it is not. It is not the chroma denoise of section 14.4: that
works at the scale of blobs, on what this stage leaves, and runs after it
for that reason. And it is not a temporal filter: the temporal stage of 14.7 sees a
glint as motion and passes it through, and a glint lasts one frame. The
video demosaicking literature (Wu and Zhang, IEEE TCSVT 2006) fetches the
missing colours from motion-compensated neighbouring frames, which needs
the feature to persist; rain does not.

### 17.2 Definition

On the camera RGB intermediate of section 3, after the debayer and before
anything reads it, with `passes` in 1..5:

    d_r = R - G,  d_b = B - G                     at every pixel
    repeat `passes` times:
        d_r = median of the 3x3 neighbourhood of d_r, borders mirrored (section 3)
        d_b = the same for d_b
        d_r = mean of the 3x3 neighbourhood of d_r, borders mirrored
        d_b = the same for d_b
    R = G + d_r,  B = G + d_b                     G and alpha unchanged

Every sweep reads the whole result of the sweep before it. The mean is
the nine summed in stencil order, top row first, left to right, then
scaled by the float nearest 1/9, so that the two implementations sum in
the same order. G is not touched at all: the stage cannot soften a
luminance edge or move a neutral value, and on a region where the colour
differences are affine — every gradient, every flat patch — it is the
identity, because the centre of a symmetric stencil is both the median
and the mean of affine samples.

The colour differences and not the ratios, because that is what the
debayers build their chroma in (5.2, 5.3); to G and not to a weighted
luma, because G is the channel the sensor sampled twice as densely and
the one every debayer built its chroma from, and RawTherapee's YIQ moves
all three channels for nothing; and on the white-balanced intermediate
rather than after the matrix, so that "neutral" is the frame's illuminant
and a glint that loses its colour comes out the colour of the light.

What the two sweeps do. A single-sensel spike of amplitude A at an R site
comes out of bilinear as an R - G of A at the pixel, A/2 at its four edge
neighbours and A/4 at the corners: the median halves the centre and
quarters the edges, the blur then leaves a sixth of A at the centre, and
the maximum leaves nothing measurable through any debayer. The median is
edge preserving and removes the isolated dots outright; the blur is not
and is what reaches the wider glints, at the price of bleeding colour a
pixel across every colour edge per pass. Measured on the GPU output of a
128x128 crop of the road of 17.1 over four consecutive frames, RCD, the
mean frame-to-frame change of the colour differences goes 0.055, 0.039,
0.034, 0.031, 0.028 for zero, one, two, three and five passes, where the
median alone had converged at 0.036 by three; the mean saturation of the
brightest tenth goes 0.29 to 0.24, and the frame-to-frame change of G
stays at 0.05, because it is the rain itself and the stage never touches
it. On a 320x240 crop of the neon signs in the same frame, two passes
move 1.7% of the pixels' chroma by more than 0.1 and five move 3.4%,
which at 6x is a slightly softer one-pixel red border and nothing else.
One or two passes cost nothing visible in a delivered file: 4K 4:2:0
from a 5.8K frame keeps one chroma sample per two source pixels, below
what one 3x3 blur removes. Five is a visible chroma blur at 1:1, which
is why the default is zero, as it is in every raw converter.

### 17.3 Where it sits, and the fused dispatch

Between the debayer and everything that reads the intermediate: the auto
white balance histogram of 4.4 and the colour stage of 4.3 both read the
filtered copy, so a road full of glints stops voting on the illuminant too.
The copy is a second RGBA16F image, because a pass that reads its
neighbours cannot write the image it reads. It was RCD's aux image with
RCD, which pass A wrote and pass C last read, so that the default pipeline
paid no memory for the stage; since RCD's passes are one dispatch (5.3)
there is no aux image, and every method gets the pipeline's scratch, 140
MB at 5.8K, shared with the other stages that need one. An RCD pipeline
with the stage is still 35 MB smaller than it was, the aux image and the
green plane being gone.

The passes run in one dispatch. Each workgroup loads a tile with an apron
of two pixels per pass into shared memory, as the two differences in one
vec2 so that the componentwise min and max of a selection network sort
both planes at once, and sweep s — the median of a pass when s is odd, its
blur when s is even — is computed on the part of the tile that lies s
inside its edge: that is where its stencil stays inside what sweep s - 1
produced, and it is all that sweep s + 1, or the final store, reads. The
apron is loaded through the mirror of section 3, and the fused result
equals the sequential definition at the borders as well: the mirror is a
reflection, both stencils are symmetric under it, so a sweep computed at a
mirrored position is that sweep at the position it mirrors, by induction
over the sweeps.

The tile's side is specialization constant 4, the workgroup's side plus
four times the maximum passes, and the shared array is sized by it: for a
16x16 workgroup that is 36x36 and 20 KB, which is more than the 16 KB
Vulkan guarantees, so the host halves the workgroup until the tile fits
the device's `maxComputeSharedMemorySize`, and a device with only the
guarantee runs 8x8 workgroups on a 28x28 tile of 12.5 KB. The picture is
the same either way; only the occupancy differs.

The median of nine is Smith's 20 exchange selection network, the one in
McGuire's median shader: each step discards the minimum and the maximum of
a shrinking set, neither of which can be the median, until three are left.
The reference sorts; both select, and both sum the blur in the same order,
so they agree bit for bit on equal inputs. On the fp16 intermediate they
differ, in absolute terms, by no more than the debayer already differed by
within the stencil's reach plus one more fp16 store, because neither a
median nor a mean moves by more than its inputs do. In relative terms they
can differ by more: the difference selected or averaged for a dark pixel
beside a clipped lamp is the lamp's, error included, and the same absolute
error is several times larger against the dark pixel. The tests are
written accordingly (17.5).

### 17.4 Cost and API

One read of the intermediate with the apron's overhead, which the L2
mostly absorbs, one write, and two sweeps per pass over the tile, twenty
min/max or nine adds on a vec2 per entry, all from shared memory.
Measured with `cineraw --bench` on the 6900 XT at 5.8K, RCD, 5.8 ms a
frame without the stage:

    passes        1     2     3     5
    ms a frame    6.6   7.0   7.6   9.5
    the stage     0.8   1.2   1.8   3.7

The apron grows with the passes, so five is not five times one: the tile
is 36x36 for 16x16 pixels and ten sweeps run over it. Nothing at all on a
frame with zero passes, which is not dispatched. There is a pipeline per
pass count, each with the tile its count needs: with one tile for five,
a frame of one pass held 20.7 KB of shared memory per workgroup for a
20x20 tile of 6.4 KB, and gave up occupancy for it. At one pass that was
0.83 ms back to back and is 0.70, the same bits.

`CRAW_PIPELINE_FLAG_FALSE_COLOR` builds the stage and
`CRAWFrameParams.false_color_passes`, 0 to `CRAW_FALSE_COLOR_MAX_PASSES`,
switches it per frame; more than the maximum is EINVAL at record time
rather than a clamp, because the shader's tile has no apron for it. The
CLI spelling is `--false-colour PASSES` in both tools, and cineview's
Debayer section carries the same number.

### 17.5 Testing

`test_pipeline`, on the synthetic mosaic: the two ways of switching the
stage off (no flag with passes set; the flag with zero passes) reproduce
the plain chain within its floor; one pass and the maximum on MHC (its
own image) and on RCD (the borrowed aux) match the reference within twice
the debayer's own intermediate tolerance — the output at a pixel is its G
plus a difference selected and averaged from its neighbours, so it
carries other pixels' disagreement, and the copy is stored in fp16 once
more — and sit at least three floors away from the unfiltered chain, so
the pass cannot have silently not run; six passes are refused with
EINVAL. On the CPU, three single-site spikes injected into the neutral
flat patch — one per site colour — keep at most 10% of their colour after
the maximum number of passes through every debayer, and through bilinear
at most 65% after one, which is what 17.2 predicts, while the G plane is
unchanged bit for bit. With `CRAW_TEST_FRAME` the real frame runs RCD
with the maximum number of passes, timed, and is held to twice the plain
chain's absolute error against the reference rather than to the relative
bound of section 11, for the reason 17.3 ends with.

## 18. Highlight reconstruction

### 18.1 What the clip throws away

Every sensel saturates at the same raw level whatever its colour filter.
On the S1II that level is the frame header's white level, 63743: a
histogram of the raw codes of any frame with a blown area has a spike
there in all four sites, and the ProRes RAW codec smears it into a
plateau, 0.993 to 1.009 of the sensor range in the HQ flavour and 0.98 to
1.03 in the other one. After the white balance gains of 4.1 the channels
therefore clip at different levels: on the camera's as-shot pair G clips
at 1.0, B at 1.668 and R at 2.025.

The clip of 4.1 cuts all three at 1.0. That keeps a blown area neutral,
which is its purpose, and it throws away everything R and B still
measured above G's clip. On P1000352.MOV, an overcast sky through a
window, G is clipped over 17% of the frame and R over 5%: across most of
the sky R and B still carry the clouds, and the clip turns it into a flat
grey plane. Where every channel has clipped the plateau is flat at 1.0 and
has no colour, and a tone mapper renders it as a flat white shape with a
hard outline. Unclipped, the same sky is magenta: G stops at 1.0 while R
and B carry on.

This stage rebuilds the clipped channels of every sensel from the
channels that did not clip at that place and from the colour of the
unclipped light around the blown area, and lets the result exceed 1.0.

The ideas are old and published: estimating a clipped channel from the
unclipped ones through the colour of the neighbourhood is what
darktable's "inpaint opposed" and RawTherapee's "colour propagation"
methods do (both described in their manuals), and filling holes in an
image by pulling a pyramid of weighted averages up and pushing it back
down is Gortler, Grzeszczuk, Szeliski and Cohen's pull-push ("The
Lumigraph", SIGGRAPH 1996). No code from either program is used; the
definitions below are complete in themselves.

### 18.2 Definition

Constants:

    U_LO    = 0.92    start of the clip ramp, as a fraction of the sensor range
    U_HI    = 0.98    end of the ramp: a sample above it is clipped
    EPS     = 1e-4    floor under a balanced value before a logarithm
    LOG_MAX = 2.0     bound on a logarithmic colour ratio
    W_SAT   = 1.0     field weight at which a texel ignores the coarser level
    MARGIN  = 3       quads around a clipped one that the colour field leaves out

The ramp sits below the codec's plateau: a sample the codec pulled down to
0.98 of the range is still fully clipped, and one at 0.92 is still fully
trusted.

Per site s at (x, y), with c = c(s) its channel and `wb` the gains the
prep of 4.1 would apply:

    u = max((raw - black_level[s]) / (white_level[s] - black_level[s]), 0)
    x = u * wb[c]                           balanced, not clipped
    a = smoothstep(U_LO, U_HI, u)           how clipped the site is, 0 to 1

with smoothstep(e0, e1, v) = t^2 (3 - 2t), t = clamp((v - e0) / (e1 - e0), 0, 1).

**The colour field.** One texel per 4x4 block of sites, the block (bx, by)
covering x in [4 bx, 4 bx + 3] and y in [4 by, 4 by + 3]; the grid is
ceil(W/4) x ceil(H/4) and the last column or row of blocks may hold only
half its quads (W and H are even, so a block always holds whole quads).
Each quad, the 2x2 sites with an even top left corner, gives

    R, G, B = x of its R site, the mean of its two G sites, x of its B site
    a_q     = the largest a over the sites of the 7x7 quads centred on this
              one, i.e. within MARGIN quads of it (those that exist; none
              outside the frame)
    m_q     = min(the largest u of its four sites, 1)
    w_q     = (1 - a_q) * m_q^2
    r_q     = clamp(ln(max(R, EPS) / max(G, EPS)), -LOG_MAX, LOG_MAX)
    b_q     = the same with B

and the texel holds `(S_r, S_b, W) = (sum w_q r_q, sum w_q b_q, sum w_q)`
over its quads, summed top row first, left to right.

A clipped quad has no weight, and neither has any quad within three
quads of one. Two things put the margin there. The ring of unclipped quads
touching a clipped area reads off colour, always towards green and cyan:
the median ln(R/G) of the quads one quad away from a clip, against those
two to twenty away, is 0.46 lower around the lamp of P1000387 (its
reflector's rings are unclipped gaps one or two sensels wide between
clipped lines, all of them in that ring), and ln(B/G) 0.10 lower at the
edge of the sky of P1001259; from two quads out the bias is below 0.03 on
every clip measured. Taken into the field, that ring painted the lamp's
rebuilt rings saturated cyan. Whether it is the codec, crosstalk or the
optics was not settled. And a small light's field is made of the few
brightest unclipped pixels around it, which are its lateral chromatic
aberration fringe and its flare: noisy, different on every side, and
different in every frame. In an area blown in every channel the level is
set by the smallest ratio (L_full below divides by it), so that noise came
out as flicker, the rebuilt core of a lamp jumping between cream and
orange and by up to a quarter in level from one frame to the next. Measured on the 22
blown lights of the first twelve frames of P1001123.MOV, a night street,
followed from frame to frame, as the median (and 90th percentile) change
of the rebuilt core between consecutive frames, in natural log units of
the level and of the colour (the distance in ln R/G, ln B/G):

    margin (quads)      1              2              3              4
    level          0.028 (0.220)  0.014 (0.074)  0.005 (0.054)  0.004 (0.070)
    colour         0.114 (0.316)  0.029 (0.090)  0.008 (0.062)  0.007 (0.083)

Three is where both stop improving. Against the truth of 18.6 the margin
costs little either way; the numbers there are the three-quad ones.

The m_q^2 makes the colour that fills a
blown area the colour of the brightest unclipped light around it — the
glow around a lamp, the band of sky beside the clipped part of the sky —
and not of whatever dark object happens to border it. It is the argument
behind the bright reading of the white balance estimator (section 7).

**Pull.** Level 0 is that grid. Level l + 1 has ceil(n / 2) texels along
each axis that level l has n along, and each of its texels holds the sum,
component by component, of the level l texels (2i, 2j), (2i + 1, 2j),
(2i, 2j + 1) and (2i + 1, 2j + 1) that exist, added in that order. Levels
are built until one is 1x1; a 5760x3040 frame has twelve, 1440x760 down.

**Push.** From the top down every texel gets a colour `c`, the pair of
logarithmic ratios (ln R/G, ln B/G) of the light there:

    top level:  c = (S_r, S_b) / W, or (0, 0) when W = 0
    level l:    c_up = level l + 1's c, bilinear at ((i + 0.5) / 2 - 0.5, (j + 0.5) / 2 - 0.5)
                t    = min(W / W_SAT, 1)
                c    = c_up + t * ((S_r, S_b) / W - c_up),  or c_up when W = 0

Bilinear sampling, here and in the fill below: with p the position,
i0 = floor(p) and f = p - i0 on each axis, the four texels at i0 and
i0 + 1 with each index clamped into the level, mixed along x first and
then along y.

This is pull-push hole filling with one change: the pull sums the weights
instead of capping them at every level, so the brightness weighting
survives into the coarse levels a large blown area is filled from. A
texel whose own block has enough unclipped light keeps its own colour, one
with none takes the coarser level's, and so on up: every blown area ends
up in the colour of the nearest bright unclipped light, at whatever scale
it has to reach for it.

**Fill.** Every site with a = 0 keeps x. At a site with a > 0:

    (lr, lb)  = level 0's c, bilinear at (qx / 2 - 0.25, qy / 2 - 0.25),
                the centre of the site's quad (qx, qy) = (x >> 1, y >> 1)
    lrho      = (lr, 0, lb)                  ln of the light's ratios to G

and for each of the two channels k other than c, over its sites in the
3x3 around the site (borders mirrored as in section 3; at an R or B site
G is at the four edge neighbours and the opposite colour at the four
corners, at a G site one colour is at the horizontal pair and the other at
the vertical pair):

    m_k = mean of their x
    a_k = 1 - the largest of their a
    l_k = ln(max(m_k, EPS)) - lrho[k]

then, with l_c = ln(max(x, EPS)) - lrho[c] and the other two channels
numbered 1 and 2,

    A      = a_1 + a_2
    L_full = exp(max(l_c, l_1, l_2))
    L_part = exp((a_1 l_1 + a_2 l_2) / A),  or L_full when A = 0
    L      = L_full + min(A, 1) * (L_part - L_full)
    out    = x + a * max(L * exp(lrho[c]) - x, 0)

The field says what colour the light is at the site, and L is its level:
each channel is L times its ratio, `x_k = L * exp(lrho[k])`. The channels
that did not clip measure L; L_part is their geometric mean, weighted by
how unclipped each one is. Where none is left every channel is only known
from below, `x_k >= m_k`, and L_full is the lowest level at which the
field's colour satisfies all three bounds. The clipped channel then gets
its share of L, never less than it measured, and the ramp blends that in:
a sample below U_LO is left alone and one above U_HI is entirely rebuilt.

The bounds of the *other* clipped channels deliberately do not raise
L_part. When they disagree with the unclipped channels the local colour
is not the field's colour, and raising L to satisfy them would push the
rebuilt channel above what the measurement allows; each clipped channel is
instead held to its own bound only, by the final max.

What it produces on a neutral highlight brightening past the clip, with
the S1II's as-shot gains: G clips at 1.0 and is rebuilt from R and B up to
1.668, where B clips; from R alone up to 2.025, where R clips; and above
that the area is a plateau at 2.025 in every channel. The picture gains
the stop the gains leave between G's clip and R's, and an area blown in
every channel is flat at 2.025 in the colour of its surroundings instead
of at 1.0 in no colour at all.

### 18.3 Where it sits

After the raw denoise of 14.3, whose output it reads when that runs (the
denoise's highlight guard leaves clipped samples alone and the unclipped
light the field is built from is the cleaner for it), and before the
debayer, which then reads the rebuilt mosaic in place of the caller's.
The rebuilt mosaic is `VK_FORMAT_R16_SFLOAT`, full resolution, and holds
`out`: the black level, the range and the gains are already in it and
nothing is clipped, so the debayer's prep of 4.1 becomes the identity
(black 0, range 1, gains 1, no clip).

On the mosaic and not after the debayer, because a debayer run on clipped
data has already smeared the clip across every edge of a blown area, into
channels and pixels that were never clipped, and no repair afterwards can
tell those apart. On the rebuilt mosaic the debayer sees continuous data
across the edge.

Everything downstream sees values above 1.0 in the camera RGB
intermediate. The auto white balance histogram's 0.98 test (4.4) still
excludes them; the colour stage has never clipped; the output extends
further above the old clip point, which at base ISO on the S1II sits at
23 times diffuse white.

A frame with no clipped sample goes through the stage as the fp16
rounding of its mosaic: at most 2^-11 relative, the rounding the fp16
intermediate of section 3 applies anyway, and 7.4 / sqrt(u) times below
the sensor's noise at u with the S1II's model of section 14.2, so at
least seven times below it anywhere.

### 18.4 Dispatches, cost and API

Four shaders: `highlight_field` (one invocation per 4x4 block; the
workgroup first puts the largest a of every quad it covers, and of the
three quads of margin around them, into a shared tile of at most 38x38,
and each block then reads its 7x7 windows from there rather than 256
samples of its own), `highlight_pull`
and `highlight_push` (one dispatch per pair of levels each way, eleven of
each at 5.8K), and `highlight_fill` (one invocation per quad, reading the
4x4 sites its four stencils cover, and storing its four samples straight
through without touching the field when none of them is clipped). The
push rewrites each level in place: every invocation reads and writes only
its own texel of the finer level, and the coarser one was finished by the
previous dispatch. The field is RGBA32F because the pull's sums over a
whole frame need fp32's range and precision; level 0 is 1440x760 at 5.8K
and the pyramid adds a third, 23.3 MB in all, plus 35.0 MB for the rebuilt
mosaic.

Measured with `cineraw --bench` over 30 frames of P1000352.MOV on the 6900
XT, RCD and lens correction: 6.00 ms a frame without the stage, 6.33 ms
with it. Leaving the field pass out saves 0.03 ms of that; the first
version, with one quad of margin read sample by sample instead of the
shared tile, cost 0.44 ms. A single cold submission right after the
pipeline is created, as the test times it, takes 2 ms more than that.

`CRAW_PIPELINE_FLAG_HIGHLIGHT_RECONSTRUCTION` builds the stage and
`CRAW_FRAME_FLAG_HIGHLIGHT_RECONSTRUCTION` switches it per frame; an
unflagged frame dispatches nothing the plain chain does not and is bit for
bit the plain chain. A flag rather than a strength, because there is
nothing to dial: a rebuilt channel is the estimate or the clip. The CLI
spelling is `--highlights on|off` in both tools, on by default, and
cineview's Debayer section carries the same switch; it is a correction,
like the lens, rather than a look. cineview raises the peak it tells
libplacebo about by the largest gain over the smallest, the plateau a
neutral area blown in every channel is rebuilt to; libplacebo's peak
detection replaces that with the measured peak anyway (it only clamps a
measured peak to PQ's 10000 nits, not to the stated one).

### 18.5 Testing

`test_pipeline` on a scene of its own: two Gaussian blobs of constant
chroma on a dim gradient, the first peaking at G = 1.4 with R and B below
their clips, the second at G = 4.0 and blown in every channel at its core,
inverted to raw codes that saturate at the white level with the usual
+-8 counts of noise on top, plateau included. On the CPU, the reference's
rebuilt mosaic: every sample below the ramp is its balanced value to
within the fp16 store (4.8e-4 relative); the 540 G sites clipped with
their R and B neighbours untouched are rebuilt to within 0.09% of the
scene's own G (bound 2%); the blown core, saturated in every channel over
the whole 3x3, comes out at R/G 1.1062 and B/G 1.0478 against the blob's
1.10 and 1.05 and at G 1.831 against the 1.841 of 18.2's plateau (bound 3%
each), with no sample below its clip; no clipped sample is ever lowered.
On the GPU: both ways of switching the stage off are the plain chain bit
for bit; MHC and RCD match the reference within the relative bounds of
section 11 (2.2e-3 and 2.4e-3 measured) and sit 1.12 away from the clipped
chain, so the pass cannot have silently not run; and the same scene
dimmed until nothing clips moves by 9.8e-4 at most, the fp16 rounding of
the mosaic, within the debayers' own absolute bounds. With
`CRAW_TEST_FRAME` the real frame runs RCD with the stage and is held to
the relative bound of section 11: 3.2e-3 on P1000387 and 3.3e-3 on
P1000352.

### 18.6 Measured against the truth

A clip cannot say what its blown highlights should have been, so the
definition was measured on frames whose highlights did not clip, pushed
over the clip on purpose: the frame's u multiplied by k, everything above
1.0 set to 1.0 plus the HQ codec's plateau noise (0.3%), the stage run,
and the result compared quad by quad with the frame's own values times k.
Quads within two quads of anything the original frame had clipped are left
out of the comparison, because there the "truth" is itself clipped. Median
absolute errors in natural log units, green-magenta being half the sum of
the errors in ln R/G and ln B/G, warm-cool half their difference, and the
level the error in ln G; each against what the clip of 4.1 gives:

    frame, k                 quads   green-magenta   warm-cool    level
    P1001493 lake, 2.5    part 0.92%   0.010 (0.017)  0.000 (0.044) 0.013 (0.158)
                          full 0.003%  0.014 (0.014)  0.055 (0.046) 0.085 (0.755)
    P1000123 sunset, 4    part 2.05%   0.018 (0.084)  0.026 (0.359) 0.024 (0.402)
                          full 0.22%   0.030 (0.086)  0.142 (0.377) 0.053 (1.113)
    P1001497 beach, 3     part 0.67%   0.020 (0.029)  0.000 (0.079) 0.026 (0.211)
                          full 0.016%  0.021 (0.052)  0.050 (0.086) 0.074 (0.797)
    P1001123 night, 12    part 0.08%   0.109 (0.160)  0.018 (0.203) 0.064 (0.313)
                          full 0.054%  0.059 (0.154)  0.117 (0.225) 0.530 (1.338)

(part: clipped in one or two channels; full: in all three.) Where a
channel survives, the level comes back to within 1% to 7% and the colour
to within what the unclipped neighbours already say; where none does the
level is, by construction, the lowest the colour allows, and misses by
5% to 41% in the median, low almost everywhere, instead of by 53% to
74%. The night street's lamps are the hard case: small sources, their
colour known only from a thin halo.

One alternative was measured and rejected: taking the green-magenta
component of the light's colour from the camera's Planckian locus (which
in this camera's balanced log space is the line ln R/G + ln B/G = 0 to
within 0.02 from 2000 K to 11500 K) instead of from the field, on the
argument that skies and light sources lie on it. Its green-magenta error
was larger than the field's on every row above but one, where it tied,
three to four times larger on the sunset, and its level error larger on
six rows of eight; the two it won, the blown cores of the lake and the
beach, it won by 0.026 and 0.014.

A second alternative was measured for the blown areas alone: shrinking
the field's log colour by a factor s towards neutral wherever no channel
survives, on the argument that the colour of a light source is closer to
the white balance than the colour of what it lights. Median colour error
(the distance in ln R/G, ln B/G) and level error on the blown quads of
the four frames:

    s        lake           sunset         beach          night
    0      0.076 0.051    0.545 0.406    0.154 0.100    0.395 0.637
    0.5    0.071 0.065    0.186 0.305    0.083 0.075    0.275 0.621
    0.75   0.076 0.071    0.070 0.157    0.075 0.068    0.234 0.589
    1      0.082 0.085    0.206 0.053    0.088 0.074    0.197 0.530

Neutral is the worst choice by far on the sunset and the night street,
and 0.75 trades a fifth less colour error for a fifth more level error
on average, so the field's colour stands as it is. It does not always
win: the blown core of P1000387's lamp is rebuilt warm (R/G 1.22, B/G
0.80) from the warm glow on the housing around it, while the white
balance estimator's bright reading puts the lamp itself near 5600 K, i.e.
close to neutral under the clip's 5500 K. The frame does not record which
is right.

Estimating the clipped channel through the field's colour takes only the
field's green-magenta component where both other channels survive: with
R and B measured and G unknown, the output keeps the measured R/B and
takes (ln R/G + ln B/G) / 2 from the field. A field whose colour is off
only along the locus, warmer or cooler than the truth, therefore costs
nothing there. On P1000352's sky, which is clipped over most of the frame
and whose nearest unclipped light is the warm band along the horizon,
the rebuilt sky still shows lavender and pink patches of the kind the
green-magenta column measures.

## 19. Defringe

### 19.1 The fringe

A lens images the three colours slightly differently: shifted (lateral
chromatic aberration, which section 16 corrects after the colour stage) and
blurred by different amounts (axial chromatic aberration, which no warp can
undo, and which is largest for whatever is out of focus). At a hard edge
between something bright and something dark the three channels therefore
cross the edge in slightly different places, and the pixels in between take
a colour that neither side has: the purple rim of a branch against the
sky, the blue line along the bottom of a lit window, the cyan and yellow
edges of a street light.

Highlight reconstruction (section 18) makes these more visible. The old
clip of 4.1 flattened every channel above the balanced clip to 1.0, which
turned the fringe at a blown edge white; kept, it is as coloured as the
optics made it. On P1081807.MOV, a fluorescent tube close to the camera and
out of focus, R's image of the tube is almost a pixel narrower than G's
and B's is shifted by 0.7 px, and the rows along its long, nearly
horizontal edge come out pink where R has reached its clip and G has not,
in a staircase that follows the sensel rows.

### 19.2 Definition

On the camera RGB intermediate of section 3, per pixel:

    Y  = (R + 2 G + B) / 4                         the Bayer-weighted level
    c  = (ln(max(R, YMIN) / max(G, YMIN)), ln(max(B, YMIN) / max(G, YMIN)))
    l  = ln(max(Y, YMIN) + Y0)
    E  = |Sobel(l)| / 8                            edge strength, per pixel, in ln units

with YMIN = 1e-3 and Y0 = 0.02 of the balanced clip, about half of diffuse
white at base ISO, so that the log gradient of noise and of detail in the
shadows is not an edge. Then at every pixel p:

    En = the largest E over the 5x5 around p       near a strong edge
    Yx = the largest Y over the 3x3 around p       and bright enough to matter
    t  = strength * smoothstep(E_LO, E_HI, En) * smoothstep(B_LO, B_HI, Yx)

with E_LO = 0.15, E_HI = 0.35, B_LO = 0.005 and B_HI = 0.02: the gate is
fully open from half of diffuse white at base ISO, and Yx is the 3x3's
largest so that a dark pixel on the dark side of a light's edge still
counts. A pixel with t = 0 is left exactly as it was. Otherwise the edge
normal n is the gradient of l smoothed over the 5x5, i.e. l convolved with
the 7x7 kernel whose columns weigh (-1, -1, 0, 0, 0, 1, 1) and rows
(1, 3, 4, 4, 4, 3, 1) for x (the transpose for y), normalized to unit
length. Each side of the edge is sampled where the edge has ended:

    for s = +1 and s = -1:
        k_s = the k in D_MIN .. D_MAX whose pixel round(p + s k n) has the
              smallest E, the nearer on a tie
        S_s = the 3x3 mean of (Y, c) at p + s k_s n, bilinear between the
              four integer positions around it

with D_MIN = 2 and D_MAX = 6. A side whose Y is below DARK = 0.1 of the
brighter side's counts as neutral: its colour is taken as c = (0, 0). The
allowed range of c is [min, max] over the two sides, per component,
widened by MARGIN = 0.03 on both ends, and

    c' = c + t * (clamp(c, range) - c)
    RGB' = Y * (e^c'r, 1, e^c'b) * 4 / (e^c'r + 2 + e^c'b)      Y kept

A pixel whose c the clamp does not move is left exactly as it was.

The rule is that the colour of a pixel on an edge must lie between the
colours on the two sides of that edge. A pixel between a green leaf and a
white sky is between green and white and is not touched; a pink pixel
between a white tube and a grey wall is outside anything either side has
and is pulled back to the nearest colour they span. Only pixels near an
edge of at least E_LO in ln units per pixel are considered at all, which
is several times the steepest colour edge of an ordinary scene (the orange
paint on the road of P1081807 peaks at 0.03) and well below a lamp or a
branch against the sky (0.6 to 1.5).

The sides are found rather than placed. The fringe must not vote on its
own colour, and the fringe of something out of focus, like the tube, is
three to five pixels wide, so a side taken at a fixed three pixels lands
inside it; one taken at a fixed six crosses a leaf four pixels wide and
reads the sky beyond it. The point of least E along the normal, two to
six pixels out, is where the edge has ended on that side, whichever of
the two it is.

A dark side is neutral rather than left out. Its measured colour is the
bright side's light spilled into it by the blur, the fringe itself, and
noise, so taking it lets the fringe vote after all; leaving it out would
hold the dark pixels beside a red light to red, and tint them. Neutral is
the colour of the illuminant once the white balance has run, which is
what a dark surface next to a light most likely has: a grey pixel beside
a red bar stays grey, since grey lies between red and neutral, and a blue
rim beside a white lamp is pulled to white. What it costs is the real
colour of a dark object beside a bright one, within the three pixels of
the edge where the stage acts at all.

The level is kept, in the debayer's own Bayer weighting. The fringe is a
fault of colour; the level of the edge is the lens's blur of the scene
and is right as it is, and keeping it means the stage can neither move an
edge nor draw a halo in level. It also means the stage cannot repair a
fault of level: where highlight reconstruction rebuilt the core of the
tube one sensel row at a time, the rows keep their levels (19.6).

### 19.3 Where it sits, and the two passes

On the camera RGB intermediate after the false colour stage of section 17
and the lens warp of section 16, and before the chroma denoise of 14.4,
the histogram of 4.4 and the colour pass of 4.3, which read what it
leaves. After the warp, because the warp re-registers the channels and a
defringe ahead of it would align them at the level's edge only for the
warp to pull them apart again (16.3). Before the chroma denoise, because
that smooths colour over blocks of four pixels: a fringe it had spread
would be wider than the side search reaches and paler, harder to tell
from a real colour, and defringed first it is gone before it can be
spread. On the camera RGB rather than the output, because there every
channel is positive, neutral is c = (0, 0) whatever the output space, and
Y is the weighting the debayer used.

Two passes, which were two dispatches. The first writes (l, c, E) per
pixel as fp16; l stays within +-4, so its fp16 is good to 2^-9, 0.2% of
Y, and c and E need no more. The second reads the first's values around
its pixel and moves its pixel's colour or leaves it alone, so every pixel
the stage leaves alone is bit for bit what came in. Most pixels leave
after the 5x5 of E. The reference in `lib/ref/ref_defringe.c` rounds the
first pass's four values to fp16 as the GPU stores them.

The first pass used to write the second intermediate of section 3 as an
RGBA16F image, which the second read back around every pixel and then
wrote the intermediate in place. Now (section 21) both are one dispatch,
`defringe.comp.glsl`: a 32x32 workgroup (wave32) loads its tile of the
intermediate with an apron of nine, keeps l in fp32 and works out the
first pass's values on the apron of eight inside it into shared memory,
rounded to fp16 the way the RGBA16F image stored them (toward zero on
RDNA2, as in 5.3), and each invocation then treats its pixel from shared
memory and writes it, moved or not, to the second image, which becomes
the intermediate as with the stages that cannot run in place. The side
search reaches six pixels along the normal and the side means two more,
which is the apron of eight, and the Sobel one more for l; a position
outside the frame holds what the first pass computed at the one inside
it that mirrors there. On the real frame the output is the two passes'
bit for bit, but for one pixel of 17.5 million when the lens warp runs
first, moved by 6e-4 of itself: the side search is an argmin (19.5).

The second image is the pipeline's scratch, 140 MB at 5.8K, shared with
the other stages that need it (it was RCD's aux with RCD, until RCD's
passes became one dispatch, 5.3).

### 19.4 Cost and API

Measured with `cineraw --bench` over 30 frames of P1081807.MOV on the
6900 XT, RCD, highlight reconstruction and lens correction on: 6.32 ms a
frame without the stage, 7.37 ms with it. The first pass is about 0.76 ms
of that, measured with the second made to return at once: a read and a
write of an RGBA16F frame, 280 MB, is 0.55 ms at this card's bandwidth,
and the nine logarithms per pixel of its Sobel read the 3x3 through the
cache. The second pass, 25 loads for the gate at every pixel and the rest
at the few percent near a strong edge, is 0.3 ms.

In one dispatch (19.3) the stage reads the intermediate once and writes
the second image once. Back to back at 5.8K, the two passes took 1.20 ms
and the one takes 0.90 (0.92 in wave64, 0.99 at 32x16, 1.19 at 16x16:
the aprons of nine are 4.5 times the pixels of a 16x16 tile and 2.4 times
those of a 32x32 one, and every pixel of an apron costs a load and three
logarithms). Its tiles take 37.6 KB of shared memory; a device with less,
or with fewer than 1024 invocations to a workgroup, gets a smaller tile.

`CRAW_PIPELINE_FLAG_DEFRINGE` builds the stage and
`CRAWFrameParams.defringe_strength`, 0 to 1, switches it per frame: zero
dispatches nothing and is the plain chain bit for bit, anything above one
is EINVAL at record time. The strength scales t, so a half moves each
colour halfway to its clamp. `craw_frame_params_init` leaves it at zero.
The CLI spelling is `--defringe STRENGTH` in both tools, 1 by default,
and cineview's Debayer section carries the same number under the
highlights switch: it is on by default there because the highlight
reconstruction, also on by default, is what uncovers the fringes it
removes.

### 19.5 Testing

`test_pipeline` on a scene of its own, in balanced camera RGB inverted to
raw codes with the usual +-8 counts of noise: a bright neutral bar (0.9
on 0.03) whose R is imaged a pixel right and down and whose B a pixel left
and up, the way a lens misregisters them; a saturated red bar (0.9, 0.08,
0.06) on 0.01, a hard edge whose colour is real; and an orange and a grey
half of equal Y side by side, a colour edge with no edge in level at all.
On the CPU, through the reference RCD and the reference stage: the 836
pixels within three of the neutral bar's outline carry colour up to 4.55
before and 0.085 at the 99th percentile after (bound 0.12); the red bar's
rim keeps its ln R/G, 2.289 before and after (it may not lose more than
0.05); not a pixel of the orange and grey patch moves, nor one far from
every edge; and no neutral pixel of the dark background comes out
coloured. On the GPU: both ways of switching it off are the plain chain
bit for bit; MHC and RCD match the reference run on the GPU's own
undefringed intermediate to 4.2e-3 and 1.8e-3 (bound 1e-2) and sit 0.47
and 0.52 away from the plain chain, so the pass cannot have silently not
run; a strength of 1.5 is refused.

Against the whole reference chain the same frames differ by 2.0e-2 and
6.8e-3, and that is a comparison of the debayers: the side search is an
argmin over fp16 values of E, and a debayer disagreement of a millesimal
moves it at the corners of the scene, where the normal turns fast. For
the same reason the real frame (with `CRAW_TEST_FRAME`) is held to a
count rather than a maximum: on P1081807 frame 0 the stage moves 276241
pixels, 1.6%, by up to 0.41, and against the reference on the same
intermediate 22 of them differ by more than 1e-2 (at most one in a
thousand of those moved is allowed), up to 0.12.

### 19.6 On real frames

GPU output through cineraw, the colour at the strong edges of each crop
(the mean distance in ln R/G, ln B/G, BT.709) without and with the stage,
and the share of the crop's pixels whose colour moved by more than 0.05:

    crop                                      edge colour        moved
    P1081807 tube, 380x100                    0.435 -> 0.174     11.7%
    P1081807 road, 700x300                    no strong edge      0.0%
    P1001259 leaves against the sky, 500x400  0.417 -> 0.262     25.8%
    P1000352 window, 500x400                  no strong edge      0.0%
    P1001123 railing under a lamp, 700x400    2.333 -> 0.290      0.5%
    P1001123 lit window, 260x240              1.821 -> 0.855      2.8%

The rainbow along the railing's specular line goes; the blue and purple
rims of the lit window and the purple rims of the leaves go, the leaves
keeping their green; the blue ends and most of the pink of the tube go.
What stays in the tube is a staircase in level along its lit core, the
rows that the highlight reconstruction rebuilt at different heights (18.2)
because R reached its clip on some and not others; that is a fault of
level and the stage keeps level. Fringes wider than D_MAX, the purple
bloom around a strongly defocused highlight, are wider than the side
search and stay.

## 20. Capture sharpening

### 20.1 The blur

Before anything is done to it, the picture is already soft: the lens's
point spread, the sensel's own aperture, the debayer's interpolation, the
codec, and the lens warp's resample at a fraction of a pixel (16.4) each
blur it a little, and together they are close to Gaussian. Fitting an
error function across the edges of P1094366, a street under signs at ISO
1250 in focus (the edges at the 99th percentile of the log gradient whose
sides are flat, in the camera RGB luma), gives a spread of 0.72 px in the
median over 697 edges of the signs and 0.57 over 702 of the street below
them; taken over a stricter set of the same signs' edges it reads 0.88, and
P1081807, much of it out of focus, reads 1.16. So the capture blur of this
camera in focus is a Gaussian of 0.6 to 0.9 px.

Capture sharpening undoes that blur and nothing more: it is not the
unsharp mask of an output sharpening, which adds contrast at whatever
scale it is set to, but deconvolution by the blur the capture made, which
RawTherapee has as a step of its own right after the demosaic for the same
reason. The deconvolution is Richardson's and Lucy's (Richardson, JOSA
1972; Lucy, AJ 1974): the maximum likelihood estimate under Poisson noise,
found by multiplicative updates that keep every value positive. What this
section adds to the textbook iteration, the luma it runs on, the clamp and
the mask, is its own.

### 20.2 Definition

Constants:

    P        = 2^-14    pedestal, and the floor under the estimate
    CORR_MAX = 2        bound on one iteration's correction
    RANGE_R  = 1        the clamp's window, 3x3
    MASK_R   = 2        the contrast statistic's window, 5x5
    KAPPA    = 0.5      the statistic's reading of the noise model, below

The PSF is the Gaussian of the frame's radius sigma, integrated over each
pixel's footprint and cut at R = ceil(3 sigma), at most 6:

    k_i = Phi((i + 1/2) / sigma) - Phi((i - 1/2) / sigma),   i = 0 .. R
    k_i = k_i / (k_0 + 2 (k_1 + .. + k_R))

with Phi the standard normal distribution. K * f is the separable blur, x
first and then y, each tap sum the centre term first and then the pairs
k_i (f(-i) + f(+i)) from i = 1 outwards, borders mirrored as in section 3.

On the camera RGB intermediate:

    d   = fp16(max((R + 2 G + B) / 4, 0) + P)     the observed luma
    u_0 = d
    for n = 0 .. N - 1:
        c       = K * (d / (K * u_n))
        u_(n+1) = max(u_n * clamp(c, 1 / CORR_MAX, CORR_MAX), P)
        rounded to fp16, except u_N

and then at every pixel, with the windows mirrored:

    lo, hi = the smallest and largest d over the 3x3
    E      = sum over the 5x5 of (d(x+1) - d(x-1))^2 + (d(y+1) - d(y-1))^2,
             summed row by row, left to right, times 1 / (4 * 25)
    L      = the mean of d over the 5x5, minus P
    V      = n_a * max(L, 0) + n_b
    m      = smoothstep(C, 2 C, sqrt(E / V)),  or 1 when C = 0
    rho    = 1 + m * (clamp(u_N, lo, hi) / d - 1)
    RGB    = rho * RGB                           alpha unchanged

with C the frame's contrast threshold and, from the frame's noise model
(a, b) of section 14.2 and its white balance gains,

    n_a = KAPPA * a * (wb_R + 2 wb_G + wb_B) / 4
    n_b = KAPPA * b * (wb_R^2 + 2 wb_G^2 + wb_B^2) / 4

A pixel with rho = 1 is left exactly as it was.

*The luma, and the ratio.* The deconvolution runs on one plane, the
Bayer-weighted luma the defringe of 19.2 uses, and every channel of a
pixel is scaled by the same factor, so R/G and B/G do not move at all
(to 2e-7 in the tests). The channels have had four stages of their own by
then, the false colour, the lens warp, the defringe and the chroma
denoise, and a deconvolution per channel would undo each of them: that re-sharpens
the chroma noise 14.4 smoothed and the fringes 19 took out. The luma is
where the detail is.

*The pedestal.* Richardson-Lucy divides by the blurred estimate and
multiplies by the ratio, which needs everything positive. The debayer
leaves a black area's noise either side of zero, so the luma is floored at
zero and raised by P, the smallest normal fp16 value: every value the
planes hold then keeps fp16's full relative precision, no store can flush
to zero, and P is 4 code values of the 16-bit range, about the read noise
of section 14.2. Because the ratio rho is taken against d = Y + P, the
change a pixel gets is Y / (Y + P) of its luma's, which damps the deepest
shadows and nothing else. The estimate is floored at P for the same
reasons.

*The clamp.* The target luma is held to the range of the observed luma in
the 3x3 around the pixel: sharpening may steepen an edge but never makes
a new extremum. It is what keeps the halos out. Near a light, the lens's
blur is not the modelled Gaussian but a wider glow, and the deconvolution,
trying to explain that glow as blur, digs a dark moat into the
surroundings a few pixels out; on a hard clipped edge, which is sharper
than any blur allows, it rings. Measured on the fluorescent tube of
P1081807 over 4655 edges of more than 20:1, as the darkest luma one to
four pixels out on the dark side over that side's level six to eight
pixels out, at the tenth percentile, and on the signs of 20.1 as the edge
spread, radius 0.8, 20 iterations, no mask:

    clamp                               tube moat   signs spread   street
    none                                0.507       0.370          0.251
    3x3 range                           0.915       0.426          0.334
    5x5 range                           0.882       0.410          0.322
    3x3, 25% beyond it                  0.817       0.376          0.277
    3x3, 50% beyond while hi/lo < 2,    0.794       0.376          0.279
      nothing beyond from hi/lo = 8
    unsharpened                         0.961       0.719          0.571

Without the clamp one edge in ten gets a moat at half its surroundings'
level. Any allowance past the range, even one that shrinks to nothing next
to a light, lets the moat back, because the pixels that dip are two to
four pixels out, where their own 3x3 sees only the gentle slope of the
glow. The strict 3x3 keeps 84% of the unclamped sharpening of the signs
and is the one kept. It also limits the noise: a noise peak is a local
extremum and cannot rise, which is visible as a much quieter flat area
than the unclamped result gives. What it cannot do is raise the peak of a
line one pixel wide; such a line comes out narrower, not brighter.

*The correction bound.* Clamping one iteration's c to [1/2, 2] made no
difference to the numbers above, the signs' spread or the lamp of 20.6
from a bound of 1.25 to one of 100: the range clamp decides. It stays as a guard, so that no single
iteration moves the estimate by more than a factor of two whatever the
data.

*The mask.* Sharpening noise is the other artefact, and in video it
flickers. E is the mean squared central difference over a 5x5, which on
white noise of variance s^2 is s^2; V is what the camera's noise model says
it should read at the window's level. On noise sqrt(E / V) is about 1,
on detail it is larger, and the threshold C is in those units, the units
of denoise_strength: 1 is "the modelled noise level" at every level and,
as far as the model is right, at every ISO. The ramp to 2 C is scale free.

KAPPA is measured. A debayered luma is correlated noise, of which central
differences see only a part: over a flat dark patch of P1000387 (320x200
at level 0.0037 of the clip, RCD) E reads 0.50 of the per-sensel variance
in the mean, 0.48 in the median, 0.375 at the lower quartile. On three
whole ISO 1000 frames (P1000387, P1094725, P1094715), taken level by level
from 5e-4 to 2e-2 of the clip, the lower quartile reads 0.35 to 0.46 and
the 5th percentile 0.25 to 0.30, against the patch's 0.375 and 0.265: the
model's linear dependence on the level is the right shape, and noise sits
at the bottom of every level's distribution. With KAPPA = 0.5 the flat
patch's sqrt(E / V) is 0.98 at the median, 1.21 at the 90th percentile and
1.43 at the 99th.

The raw denoise of section 14 changes what the mask sees: at strength 1
the statistic on the same patch falls to 0.02 of the model, a seventh of
its standard deviation. The threshold stays in units of the modelled
noise rather than guessing at what the denoise left, which is scene
dependent; 20.6 measures what a lower threshold does then.

### 20.3 Where it sits, and the dispatches

After the chroma denoise of 14.4, the last stage on the camera RGB, and
before the histogram of 4.4 and the colour stage of 4.3. After every
denoise, because a deconvolution amplifies whatever noise it is given and
the denoises remove it at the scale the deconvolution works at. After the
lens warp, because the warp's resample at a fraction of a pixel is part of
the blur and a sharpened picture resampled afterwards is soft again, with
its overshoot interpolated. After the defringe, whose edge gate was set on
edges as the debayer leaves them (19.2) and would open wider on sharpened
ones, and whose clamp of colour the ratio cannot undo. On the camera RGB,
where every channel is positive and the luma has the debayer's weighting.
The histogram then reads the sharpened intermediate, and since rho moves
no ratio of channels its coordinates do not move; only a sample's weight
does, with its level.

Dispatches: `sharpen_prep`, one invocation per pixel of the plane's
rows (it was one per eight pixels, whose loads a subgroup scattered over
the row: 0.80 ms at 5.8K against 0.50 now, the same bits), writes d;
then N of `sharpen`, one Richardson-Lucy iteration each, the
first reading d as its estimate and each writing the plane the one before
it did not; the last is a second pipeline of the same shader
(specialization constant 5) that writes no plane but applies the clamp,
the mask and rho to the intermediate in place, each invocation to its own
texel only, and does not write a texel whose rho is 1. The iterations
before the last are `sharpen_rows.comp.glsl` wherever the device can run
it, below; it computes the same numbers in another order of work.

The tiled iteration (`sharpen.comp.glsl`): the workgroup loads its tile of the estimate
with an apron of 2R and of d with R (3 at least, for the mask), blurs the
estimate along x and y over the part of the tile R inside its edge, takes
the ratio there, and blurs the ratio down to its own pixels, all in shared
memory, so that the planes are read once an iteration instead of four
times. As in 17.3 the apron is the mirror of section 3 and the fused
result is the sequential definition at the borders too: the mirror is a
reflection and the Gaussian symmetric, so the blur at a mirrored position
is the blur at the position it mirrors.

The three planes, d and two estimates, are fp16 (a rounding of at most
2^-11, seven times below the sensor's noise even at the clip, the argument
of 18.3) in storage buffers rather than images, eight pixels to a 16-byte
chunk, so that a tile is loaded a chunk at a time; the part of a tile
outside the frame is then filled from the positions the mirror maps it
to, on the border workgroups only. The conversion to fp16 is written out
as round to nearest even: packHalf2x16() promises no rounding and on this
GPU truncates, which biased every iteration's estimate low and put 2500
pixels of the synthetic scene past 1e-3 of the reference. The reach R is
specialization constant 6, twelve pipelines in all, so that every tile
size, loop bound and divisor is a constant; and the blurs along x and the
first along y take four outputs to an invocation, from one run of 4 + 2R
values held in registers. At 16x16 the tiles need 9.7 KB of shared memory
at R = 3 and 16.4 KB at R = 6, a little over the 16 KB Vulkan guarantees,
so the host halves a reach's workgroup to 8x8 on a device with only the
guarantee. The planes cost 105 MB at 5.8K.

What each of those steps bought, per iteration at R = 3, 5.8K, on the
6900 XT, back-to-back submissions: a formatless image load per pixel of
the tiles and the reach in the push constants, 0.86 ms; the reach a
specialization constant, 0.60 ms; fp16 buffers in chunks, 0.39 ms; four
outputs to an invocation, 0.37 ms. Of the last, the loads and stores are
0.16 ms, which at 105 MB an iteration is what the card's memory and cache
deliver, and the blurs the rest.

The streamed iteration (`sharpen_rows.comp.glsl`) does the blurs without
shared memory. One subgroup of 32 owns a vertical strip of the planes,
4 x 32 columns wide and 32 output rows tall, and walks down it a row at
a time, each lane holding four adjacent columns. Along x a blur takes the
R columns it needs past the lane's own from the neighbouring lanes with
`subgroupShuffleUp/Down`, which in wave32 on RDNA is one `ds_swizzle`
each: the lanes' crossbar, no shared memory traffic and no barrier.
Along y the last 2R + 1 rows of each x blur stay in registers, a ring
indexed by row modulo 2R + 1 whose loop is unrolled by its depth
(`GL_EXT_control_flow_attributes`, since NIR would otherwise leave the
ring's indices dynamic and lower them to selects), so nothing moves as
the strip advances. Row y of u enters, b and q leave for row y - R and u'
for row y - 2R; the first 4R rows of a strip fill the rings, and 2R
columns on either side are computed and thrown away, as the tiles'
aprons are. Every sum is the tiled shader's in the same order, and the
store the same rounding, so the two write the same bits: the chain's
output with either is identical, and the tests read the same errors.

Wave32 is chosen, not left to the driver: in wave64 a constant shuffle
is three `ds_bpermute` and the iteration took 0.21 ms instead of 0.14
(before the unroll). The strip height is a plateau from 28 to 56 rows,
0.119 to 0.120 ms, and falls off below and above (0.129 at 16, 0.132 at
76), from the rows spent filling the rings on one side and too few
subgroups for 160 SIMDs on the other. A load issued a row ahead made it
slower (0.130 ms, the extra registers cost occupancy): the shader is
bound by its VALU, a row of four columns taking about as many cycles as
it has instructions, not by memory. A device whose subgroups cannot be 32 gets the
nearest size it has; one whose strip would then have no room for output
(below 8 lanes at R = 6) keeps the tiled shader for that reach.

### 20.4 Cost, the iteration count, and the API

Per iteration, and the rest of the stage (the luma pass, and the last
iteration's read and write of the RGBA16F intermediate and its mask), at
5.8K on the 6900 XT, back-to-back submissions, each measured twice while
another agent's runs shared the GPU:

    radius     0.3    0.6    0.8    1.2         1.5         2.0
    reach R    1      2      3      4           5           6
    ms         0.24   0.26   0.37   0.41-0.48   0.57-0.63   0.74

and about 0.9 ms for the rest. The two runs agree to 0.01 ms except at
reaches 4 and 5, where one of them was evidently shared. With the
streamed iterations of 20.3, per iteration, the median of 20 recordings
of the full chain's sharpening on one frame, the tiled shader measured
in the same session beside it:

    radius     0.3    0.6    0.8    1.2    1.5    2.0
    tiled      0.228  0.447  0.353  0.393  0.548  0.722
    streamed   0.092  0.109  0.120  0.152  0.203  0.273

so twenty iterations at radius 0.8 are 2.3 ms plus the rest where they
were 6.6 ms, and five, cineview's while playing, 0.5 ms. The figures
below are the tiled shader's. `cineraw --bench`
over 60 frames of P1000387, which idles the GPU between frames (14.3),
reads 7.37 ms without the stage, 8.81 ms with one iteration, 10.52 ms with
5 and 16.62 ms with 20 at radius 0.8, identically on two runs: 0.41 ms an
iteration and 1.0 ms for the rest.

Convergence, as the edge spread of the signs and the street of 20.1 at
radius 0.8 and no mask:

    iterations   0      1      2      4      6      10     20     40
    signs        0.719  0.601  0.558  0.507  0.482  0.457  0.426  0.403
    street       0.571  0.487  0.448  0.407  0.385  0.364  0.334  0.322

Twenty is where more stops paying: forty cost twice as much for a tenth
more. Twenty is 8 to 9 ms at 5.8K, which a still or an export can pay and
playback cannot: the pipeline is about 5 ms without the stage, libplacebo's
render 13 to 19 ms, and the frame 33 ms. So the iteration count is a
per-frame parameter, and cineview runs at most five while playing, 2.7 to
3.2 ms and three quarters of the edge sharpening twenty do, and the full
count on a paused picture and on every exported frame. Pausing is a
change of look and puts the held picture through again (the same
mechanism a grade change while paused uses), so what is paused, looked at
and exported is the full count, and what plays is a little softer.

`CRAW_PIPELINE_FLAG_SHARPEN` builds the stage and its planes; per frame,
`CRAWFrameParams` carries `sharpen_radius`, the sigma, above 0 and at
most `CRAW_SHARPEN_MAX_RADIUS` (2.0); `sharpen_iterations`, at most
`CRAW_SHARPEN_MAX_ITERATIONS` (100, a bound on how long one recording can
hold the GPU); and `sharpen_contrast`, C, at least 0. A radius or a count
of zero, the default of `craw_frame_params_init`, dispatches nothing and is
the plain chain bit for bit; out of range is EINVAL at record time. A
positive contrast needs a noise model exactly as the denoise does, the
override or the profile's table, and fails with ENOTSUP without one; the
three fields come out of the padding tail.

The CLI spelling is `--sharpen RADIUS[,ITERATIONS[,CONTRAST]]` in both
tools, off by default as in every raw converter, the two left out taking
20 and 1. cineview's Sharpen section, after Lens, has the standard switch
and the three rows; on a camera with no noise model it greys the contrast
row and sharpens unmasked. Radius 0.8 is the default for the in-focus blur
of 20.1.

### 20.5 Testing

`test_pipeline`, on the CPU, through the reference: a scene of bars six
pixels wide, a block and a ramp, blurred by the PSF the stage assumes at
sigma 1.0 and deconvolved with it, falls from an rms error of 0.0537
against the unblurred scene to 0.0484, 0.0429, 0.0340 and 0.0265 after 1,
4, 16 and 64 iterations (every step must fall, and 64 must halve it). A
flat patch carrying white noise at exactly the modelled level beside a
blurred step: unmasked, 16 iterations raise the noise 1.26 times; with a
threshold of 1.5 the patch is untouched but for one pixel in 15872 (the
bound is one in a thousand) while all 768 pixels of the step are
sharpened; and the colour ratios of every pixel hold to 1.9e-7.

On the GPU: three ways of switching it off (no flag, radius 0, no
iterations) are the plain chain bit for bit; MHC and RCD, at radius 0.6
with 25 iterations unmasked and 1.3 with 12 masked, match the reference
run on the GPU's own unsharpened intermediate within 4e-3 relative (1.6e-3
to 2.1e-3 measured; one iteration alone is one fp16 step, 9.7e-4) and sit
a whole pixel's value away from the plain chain; a 246x150 frame, whose
rows end in a part-filled chunk and whose last workgroups hang over the
right and bottom edges, at radius 2.0, matches within 1.5e-3, and so, to
8.9e-4, does the smallest frame the stage takes, 14x14 from the middle of
the checkerboard, where one tile holds the whole frame and the mirror
reaches across it; a 12x12 pipeline, a radius of 2.5, 101 iterations, a
negative contrast and a positive one without a model are refused. With
`CRAW_TEST_FRAME` the real frame runs RCD with 20 iterations at radius 0.8
and threshold 1, moves 30.7% of the pixels of P1000387, and holds to
3.0e-3 of the reference with no pixel past 4e-3.

### 20.6 On real frames

Radius 0.8, threshold 1, the edge spread of 20.1 and the high-pass noise
of flat areas (the rms of the 3x3 Laplacian of the luma) over what the
unsharpened frame reads:

    frame                                    5 iterations  20 iterations
    P1094366 signs, spread 0.718             0.493         0.426
    P1094366 street, spread 0.571            0.397         0.335
    P1000387 dark flat patch, noise          x1.008        x1.015
      the same, raw denoise at 1             x1.000        x1.000
    P1094352 dark trees, ISO 10000, noise    x1.001        x1.002
      with the old table's 1250 knot         x1.092        x1.187

The detail comes out: edges lose two fifths of their spread, the sign's text
and the edges of the lit panel in P1094352 are crisp at 1:1, and a flat
area moves by 1.5% of its noise or not at all. The threshold was chosen on
these numbers, the same measure at 20 iterations, no denoise:

    threshold                        0      1      1.5    2
    P1000387 flat patch noise        x1.35  x1.02  x1.00  x1.00
    P1000387 wood grain, band        x1.35  x1.03  x1.00  x1.00
    P1094738 hair, ISO 3200, band    x1.35  x1.04  x1.02  x1.01

(the band being the rms of the difference of two Gaussians of 0.6 and 2
px, P1094738 with its own fitted model). Strong edges sharpen identically
at every threshold up to 3; what the threshold decides is texture at the
level of the noise, which at 1 comes through a little and at 1.5 not at
all. 1 is the default, the modelled level, and costs 2% on a flat patch.
With the raw denoise at strength 1 the noise that is left is a seventh of
the model's and a lower threshold is safe: at 0.5 the flat patch and the
wood are untouched and the hair gains 13%, at 0.25 the flat patch's noise
rises 1%.

The last row of the first table was the open problem. The noise table of
section 14.2 had knots at ISO 1000 and 1250 only and held the last knot
above them, which is the right failure for the denoise (too little) and
the wrong one for the mask: this clip's own fit is a = 1.18e-4, eight
times the knot, the mask read its noise at nearly three times the
modelled level, and 20 iterations raised it 19%. P1094738 at ISO 3200 was
2.3 times the knot. The ISO series of 14.2 fixed both stages at once: the
table now gives this ISO 1.18e-4, 2.65e-8, and with nothing but the table
the same measure reads, 5 and 20 iterations, on two flat patches of the
dark trees (240x360 at 4680,240 and 180x300 at 5220,660; own fit and 1250
knot for comparison, on the same patches)

    table of 14.2         x1.002  x1.004
    the clip's own fit    x1.002  x1.003
    the 1250 knot         x1.12   x1.26

and on P1094353's flat panel of 14.7 x1.004 and x1.007 (own fit x1.003
and x1.006, 1250 knot x1.22 and x1.50). The 1250 knot also pulled the
dark patches' level down by a tenth at 20 iterations; the table leaves it
where it was. `--noise`, which the mask uses as the denoise does, is still
the override.

Lights, over the edges of more than 20:1 contrast, the dark side's moat
of 20.2 at the tenth percentile and the bright side's peak one to four
pixels in over its level six to eight pixels in, at the ninetieth:

    frame                       moat, 0 / 5 / 20        peak, 0 / 5 / 20
    P1000387 lamp               1.152 1.132 1.095       1.794 1.981 1.973
    P1081807 tube               0.961 0.945 0.922       1.360 1.447 1.468
    P1094352 booth lights       0.431 0.372 0.351       1.904 2.083 2.142
    P1094715 leaves, sun        1.107 1.087 1.075       0.988 1.001 0.997

The moats that are there before sharpening are the scene's (the booth's
dark frame against its lights); the stage deepens them by at most the 3x3
range allows, which at 1:1 on the tube, the booth and a string of bulbs
reads as steeper edges and no ring. The bright side rises by 8% to 12% at
the ninetieth percentile, never past the brightest pixel of its 3x3, on
the rims of lights far above white. Two things are not done: an estimate of the radius from the frame itself, for
which the edge fit of 20.1 is the obvious start, and any attempt to make
the five iterations of playback look like twenty.

## 21. Profiling and cost

### 21.1 Timing every dispatch

`CRAW_PIPELINE_FLAG_PROFILE` gives the pipeline a timestamp query pool and
has every dispatch it records wrapped in a pair of timestamps, both written
at `VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT`: the first once everything
recorded before it has left the compute stage, i.e. once the previous pass
has finished, the second once this one has. All dispatches go through one
function in `lib/pipeline.c`, so that is the only place that knows; without
the flag there is no pool and the function costs one predicted branch.

The pool holds `CRAW_PIPELINE_PROFILE_SLOTS` (4) recordings, each a run of
two queries per dispatch the pipeline can make: every pass it was built
with, the highlight pyramid at its depth and `CRAW_SHARPEN_MAX_ITERATIONS`
sharpening iterations. A recording resets its run in its own command
buffer ahead of the timestamps, which needs no barrier (query commands on
one queue are ordered among themselves), and remembers which shader each
pair timed. `craw_pipeline_get_stage_times()` reads the last recording's
run once the caller's fence has signalled, without `WAIT_BIT` (a
submission that has not completed is an `EAGAIN`, not a hang), and folds
the dispatches of one shader into one entry: its name, the number of
dispatches and their summed time, in the order the passes first ran. The
`_n` variant reads a recording up to three back, for a player that keeps
several in flight, exactly as the histogram's does.

What the numbers are: the GPU time of each pass alone, barriers excluded.
The sum of the passes against a pair of timestamps around the whole
recording is what the barriers and the launches between passes cost,
under 0.2 ms on the full chain. Timing each dispatch serialises the two
that could overlap, the histogram and the colour pass, so a profiled
recording can be marginally slower than an unprofiled one; the output is
the same bit for bit (`test_pipeline` checks both).

`cineraw --bench` builds its pipeline with the flag and prints, after the
per frame lines, every pass with its dispatches per frame and its average
and median time over the run, then the sum of the passes and the whole
recording. It runs cold: the software decode takes 170 ms to 500 ms a
frame, the GPU drops its clocks in between, and every frame starts on an
idle GPU. The passes are comparable with each other and from run to run,
but a player whose frames arrive back to back is faster: by up to a third
on the passes the shader core bounds (the denoise, RCD's first pass, the
sharpening), by next to nothing on the ones memory bounds (below).

### 21.2 The full chain, measured

P1000387 (ISO 1000, 5760x3040) and P1094724 from the card, both with the
lens record on every frame, through `--denoise 1,0.5,0.8 --highlights on
--false-colour 1 --defringe 1 --sharpen 0.8,20` (the reach of a 0.8 px
Gaussian is 3), 30 frames, on the RX 6900 XT under RADV (Mesa 26.3),
wave64, with the GPU shared with other work. "Warm" is the same chain on
one frame of P1000387 recorded 30 times back to back with RGBA32F output
and the histogram on (a scratch harness, not in the tree), median. The
bytes are the images each pass reads and writes once, at 17.5 Mpx; a pass
is called bandwidth bound when those bytes over its warm time come within
reach of the 450 GB/s the card sustains out of memory, more where the
128 MB Infinity Cache still holds what the previous pass wrote.

    pass              n    cold 387  cold 724   warm     MB    GB/s  bound by
    denoise_nlm       1      3.90      3.91     2.61    245     94   LDS: 49 x 9 patch reads a sample
    highlight_field   1      0.12      0.12     0.11     53    473   bandwidth
    highlight_pull   11      0.09      0.08     0.08      -      -   launches, 7 us each
    highlight_push   11      0.06      0.05     0.05      -      -   launches
    highlight_fill    1      0.14      0.14     0.12     88    754   bandwidth (cache)
    rcd_dir           1      1.48      1.48     1.19    175    147   texture fetch: 49 image loads a pixel
    rcd_green         1      0.48      0.48     0.51    210    416   bandwidth
    rcd_rb_rb         1      0.88      0.88     0.88    280    318   bandwidth, half the lanes idle
    rcd_rb_g          1      0.84      0.85     0.85    385    455   bandwidth, half the lanes idle
    false_color       1      0.94      0.94     0.83    280    337   bandwidth and LDS
    lens              1      0.76      0.77     0.71    280    396   texture fetch: 49 image loads a pixel
    defringe_prep     1      0.64      0.64     0.65    280    432   bandwidth
    defringe          1      0.50      0.55     0.49    150    305   texture fetch: 25 loads for the 5x5 max
    chroma_down       1      0.66      0.66     0.65    149    228   access pattern: 16 loads an invocation
    chroma_blur       1      0.16      0.16     0.14      9      -   ALU: 81 exp a quarter texel
    chroma_up         1      0.51      0.52     0.52    298    577   bandwidth
    sharpen_prep      1      0.81      0.81     0.81    175    217   access pattern: 8 loads an invocation
    sharpen          19      7.34      7.36     6.56    105    304   LDS and barriers, 0.345 ms each
    sharpen_last      1      0.67      0.68     0.63      -      -   LDS and the mask
    awb_hist          -        -         -      0.38    140    368   bandwidth, 64 loads an invocation
    color             1      0.47      0.48     0.74    280    594   bandwidth (RGBA32F out: 420 MB)
    sum of passes           21.45     21.56    19.49
    whole recording         21.52     21.63    19.64

Half the frame time is two stages: the sharpening's iterations, a third of
it, and the raw denoise, an eighth. Both are bound by shared memory, not
by memory: an iteration of the sharpening moves 105 MB in 0.35 ms and the
denoise 245 MB in 2.6 ms. RCD's first pass and the lens warp looked bound
by the texture units' rate of image loads, 49 a pixel each; for the warp
that turned out wrong (16.6). The rest is bandwidth, where subgroup
operations cannot help and only moving fewer bytes can.

### 21.3 What was done, and what subgroups bought

Each change was measured before and after with the table above's warm
harness, median of 20 recordings, several runs alternated between the two
builds with the GPU shared, and kept only when it measured faster; each
writes the same bits as before unless it says otherwise, checked on the
real frame's full chain against the previous build.

- The sharpening's iterations but the last (20.3): streamed down strips
  by subgroups of 32, the x blurs' neighbours from the next lanes with
  `subgroupShuffleUp/Down`, the y blurs in a register ring. 0.353 to
  0.120 ms an iteration at radius 0.8, 6.6 to 2.3 ms for twenty. The same
  bits. This is where subgroups pay most: the tiled version was shared
  memory traffic and barriers, and the streamed one has neither.
- The raw denoise (14.3): the patch distances of a band of four rows a
  lane, summed along x across the lanes with the same shuffles and along
  y in registers, the offset column read once per dx. 2.60 to 1.59 ms
  with the temporal pass, 2.02 to 1.18 without, 0.71 to 0.73 for the
  temporal pass alone. Rounding moves 0.18% of the output pixels by at
  most 4e-3 of the pixel; every test reads the same.
- RCD (5.3): the three passes in one dispatch with shared memory tiles.
  3.54 to 1.38 ms and 175 MB of images fewer; the same bits. No subgroup
  operations: this was fusion, and bandwidth.
- The defringe (19.3): both passes in one dispatch, 1.20 to 0.90 ms; the
  same bits but one pixel of the real frame when the lens warp runs first.
- The false colour stage (17.4): a tile sized for the frame's pass count,
  0.83 to 0.70 ms at one pass. Occupancy.
- The luma pass of the sharpening, the chroma downsample and the AWB
  histogram (20.3, 14.4, 4.4): invocations placed so that a subgroup's
  loads are runs of a row, the blocks gathered with quad broadcasts in the
  order they were summed. 0.80 to 0.47, 0.65 to 0.44 and 0.39 to 0.31 ms
  in the full chain; the same bits.
- The lens warp (16.6): its alpha written rather than read back, 0.77 to
  0.70 ms.

Tried and dropped: DPP instead of the swizzles in the denoise
(`subgroupClusteredRotate` over 16 lanes, folded into the adds: 1.54 ms
against 1.27 at the same instruction count); wave64 for either streamed
kernel (a constant shuffle is three `ds_bpermute` there: 0.21 ms against
0.14 an iteration, over 5 ms for the denoise); a load a row ahead in the
streamed sharpening (0.130 against 0.120, the registers cost occupancy);
the lens taps in a shared memory tile found with subgroup min/max (0.70
to 0.68, not worth its size); and RCD in fp32 throughout, which would
cut the chain's error against the reference threefold but tightens one
test's relative bound below the chroma stage's own error (5.3).

Where subgroups did not help: every pass that is bound by bandwidth
(highlight field and fill, the colour pass, the chroma upsample, the
false colour stage, the warp), where the bytes are the cost and only
moving fewer of them helped, which is what the fusions did; and RCD and
the defringe, whose stencils reach five and nine pixels in both
directions, which a tile in shared memory serves directly where lanes
would have to exchange across rows as well. The AWB histogram's atomics
(three a block) were never the bound, so a reduction before them would
buy nothing. What subgroups did buy is the two passes whose inner loops
were shared memory traffic, and the three whose loads were scattered.

The subgroup operations used are basic, relative shuffle and quad,
checked on the device as section 10 says; the streamed kernels require a
subgroup size of 32 through `requiredSubgroupSize` and fall back to the
nearest size the device has (lavapipe's 8 passes the tests). The shaders
now use `GL_KHR_shader_subgroup_basic`, `_shuffle_relative` and `_quad`,
`GL_EXT_control_flow_attributes` (the unrolled rings) and
`GL_EXT_shader_16bit_storage` (the luma pass's 16-bit stores, under the
storageBuffer16BitAccess the library already required), which section
2's list of allowed extensions does not name yet.

### 21.4 The chain after

The same runs as 21.2 (cold: `cineraw --bench`, 30 frames of each clip;
warm: the harness, RGBA32F output and the histogram on), the GPU shared
with other work throughout:

    pass              n    cold 387  cold 724   warm
    denoise_nlm       1      2.53      2.39     1.62
    highlight_field   1      0.14      0.16     0.11
    highlight_pull   11      0.08      0.09     0.08
    highlight_push   11      0.05      0.05     0.05
    highlight_fill    1      0.15      0.14     0.12
    rcd               1      1.79      1.75     1.42
    false_color       1      0.67      0.67     0.64
    lens              1      0.77      0.76     0.70
    defringe          1      1.12      1.17     0.93
    chroma_down       1      0.42      0.42     0.44
    chroma_blur       1      0.19      0.17     0.16
    chroma_up         1      0.51      0.51     0.51
    sharpen_prep      1      0.47      0.47     0.47
    sharpen          19      2.74      2.67     2.32
    sharpen_last      1      0.75      0.75     0.64
    awb_hist          -        -         -      0.31
    color             1      0.48      0.49     0.74
    sum of passes           12.97     12.89    11.26
    whole recording         13.03     12.95    11.39

21.5 ms to 13.0 a frame cold and 19.8 to 11.4 warm. The passes the shader
core bounds lose the most on a cold GPU (the denoise 2.5 cold against 1.6
warm), which is why the cold total fell less.

### 21.5 What is left

- The sharpening's last iteration, 0.64 ms, is still the tiled shader:
  its contrast mask reads 125 values of shared memory a pixel. Streamed
  like the others, with the observed luma's last seven rows in registers
  and its columns from the neighbouring lanes, the mask would be register
  arithmetic in the same order; about 0.2 ms to gain.
- The highlight pyramid's 22 dispatches of small levels cost 0.13 ms and
  the barriers between them; the levels below a workgroup's size could
  pull and push in one dispatch each.
- The luma pass reads the intermediate the chroma upsample (or whichever
  stage ran last) has just written; writing the luma from that stage
  would save a read of the frame, 0.3 ms, at the price of coupling them.
- The denoise's temporal pass is bound by its state's bandwidth, R32F
  read and written; nothing here changes that.

## 22. The window's colour

### 22.1 The model

A Wayland compositor composites every window into a framebuffer of its
own, and with `wp_color_manager_v1` it knows what that framebuffer is and
what each window's pixels are: it converts from the one to the other. A
window that says nothing is "compositor implementation defined, should be
sRGB", and wlroots takes it as sRGB primaries with the gamma 2.2 display
curve (`types/scene/surface.c`), which is also what its SDR outputs are.
cineview used to render that untagged window with libplacebo's target for
`VK_COLOR_SPACE_SRGB_NONLINEAR_KHR`, which is the piecewise sRGB curve at
libplacebo's assumed 1000:1: a picture encoded for one curve and one black
level, shown through another.

Now the window asks the compositor for the surface's *preferred* image
description (`get_surface_feedback`, `get_preferred_parametric`,
`get_information`), decides what of it can be rendered
(`view_colour_resolve()` in `tools/gui/colour_map.c`), renders
libplacebo's target into exactly that - primaries, curve, luminances,
target gamut - and tags the surface with a parametric description saying
so (`get_surface`, `set_image_description`). When the preference is
rendered as it is, the compositor has nothing left to convert. When it
changes - the window moved to another output, HDR switched on - the
feedback object says `preferred_changed2` and the same happens again while
the window keeps playing. Everything protocol-side is `tools/gui/colour.c`,
on a `wl_event_queue` of its own made off SDL's `wl_display` (a proxy
wrapper of the display on that queue gets the registry, and everything
made from the manager inherits the queue), so SDL, Mesa's WSI and the
module read one socket and see only their own events. Opening waits on
roundtrips of that queue; at run time nothing waits: the loop dispatches
what has arrived at the top of every iteration, and each step sends its
requests and returns.

X11 (XWayland included: SDL hands out no `wl_display` there), a
compositor without the manager, without parametric descriptions or
preferring nothing renderable, a surface without a pass-through format,
and `--no-colour-management` all leave the window as it was: untagged
sRGB on an eight-bit swapchain.

### 22.2 Pass-through, 16-bit UNORM, and the one colour surface

A `wl_surface` may have one `wp_color_management_surface_v1` and no more;
a second is a protocol error, which ends the connection and SDL's window
with it. Mesa's Wayland WSI creates one of its own for every swapchain
colour space except `VK_COLOR_SPACE_PASS_THROUGH_EXT` (and sRGB, on a
compositor without the deprecated `srgb` curve, which sway is), parented to
the swapchain; with pass-through it never touches colour management. So
the swapchain is switched to pass-through before our surface object is
made. libplacebo does that for a `pl_swapchain_colorspace_hint()` with
unknown primaries and unknown transfer, and only then: it recreates the
swapchain on the spot and destroys the retired one at once, and Mesa
destroys the old swapchain's colour surface with it. The frame's
`color_space` is then unknown/unknown, and cineview overrides the target's
colour itself. The order at opening is: read everything, decide, create
the description and wait for `ready2`, and only then hint the swapchain
and create the surface object - so that every refusal on the way leaves
the window exactly as it was, and the one step after the hint cannot fail.

libplacebo scores pass-through formats 16-bit over 10 over 8 and UNORM
over float, and there is no parameter to ask for another, so on this
machine the window becomes `VK_FORMAT_R16G16B16A16_UNORM` (Mesa offers
B8G8R8A8/R8G8B8A8 UNORM and SRGB, A2B10G10R10, A2R10G10B10 and RGBA16
UNORM and SFLOAT with pass-through). That holds 0 to 1 and nothing
outside it, so extended linear - scRGB - is not a target: a compositor's
HDR is reached through PQ or HLG instead.

### 22.3 The dither depth

The dither has to meet the grid the compositor rounds to, not the
image's: sway composites into an eight-bit framebuffer (render_bit_depth
8, and 10 once an output is HDR) and rounds without dithering, so a
picture dithered against a finer grid lands with exactly the contours
the dither was there to break. For the untagged window that is had with
an eight-bit format (`disable_10bit_sdr`, whose comment in `gpu.c` tells
both halves). With RGBA16 the frame's bit encoding is set by hand:
`sample_depth = color_depth = 8`, `bit_shift = 0` for an SDR description,
10 for PQ and HLG. libplacebo's output stage then dithers to that depth
and floors onto its grid (`floor(scale * color + bias) / scale`, with no
further normalisation since the two depths are equal), so each sixteen-bit
texel holds an exact code k/255 (k x 257), which the compositor's rounding
reproduces.

### 22.4 The choice

The candidates, first acceptable wins, where acceptable means a curve
libplacebo can encode into a UNORM image that the compositor advertises
(a named one, or a power with `set_tf_power`), and named primaries the
compositor advertises and libplacebo has:

1. the preferred description as it is (exact);
2. BT.2020 PQ, when the preference is HDR: PQ or HLG, or a maximum above
   the reference white;
3. sRGB primaries, gamma 2.2;
4. sRGB primaries, BT.1886;
5. sRGB primaries, the `srgb` curve, which only a version 1 compositor
   still names.

Renderable are bt1886, gamma22, gamma28, srgb and compound_power_2_4 (the
piecewise IEC 61966-2-1 curve, which is libplacebo's `PL_COLOR_TRC_SRGB`
exactly), st2084_pq, hlg, and a power within 1e-3 of 1.8, 2.0, ... 2.8.
Not renderable: ext_linear (no values above 1), st428 (XYZ, a colour
system rather than a curve), the log curves, st240, xvYCC and ext_srgb.
The named primaries map to libplacebo's by their xy - srgb is BT.709, pal
BT.601-625, ntsc BT.601-525, pal_m BT.470M, generic_film film C, and
bt2020, dci_p3, display_p3 and adobe_rgb their namesakes - with one
exception: the protocol's cie1931_xyz is ST 428-1's XYZ "primaries", (1,
0), (0, 1), (0, 0) with the equal-energy white, while
`PL_COLOR_PRIM_CIE_1931` is CIE 1931 RGB, 700, 546.1 and 435.8 nm, another
triangle altogether (the two only share a name, and libplacebo's libav
mapping of AVCOL_PRI_SMPTE428). It goes with st428 anyway. A preference
that sends only xy is taken as the named set it matches within
`pl_raw_primaries_similar()`.

The gamut libplacebo maps into, `hdr.prim`, is the compositor's target
gamut where that is smaller than the container, the container otherwise,
and the same primaries go out as the mastering display's, so that what the
compositor is told is what was mapped into. A target reaching outside the
container cannot be encoded in UNORM or sent without
extended_target_volume, and is clipped into the container rather than
dropped (`pl_primaries_clip()`, then pulled a ten-thousandth of the way to
white so it stays strictly inside once rounded to the protocol's
millionths). The case that matters is the one that needs it: a P3 panel
behind a BT.2020 container. Both reds lie on the spectral locus, P3's near
615 nm and BT.2020's at 630, and the locus bulges between them, so P3's red
is outside BT.2020's red-green edge by 0.0017 in y; refused, the picture
would be mapped into all of BT.2020 and left to the panel to clip.

The intent is relative colorimetric where the compositor offers it,
perceptual (which every compositor must) otherwise: the picture is already
tone and gamut mapped into the target volume, and a second perceptual
squeeze is not wanted.

### 22.5 Luminances, and why nothing is scaled

libplacebo's conventions decide the rest. Its SDR white is
`PL_COLOR_SDR_WHITE`, 203 cd/m², the linear 1.0; PQ is absolute, nits in
and nits out; an SDR curve and HLG are relative and "black scaled": the
target's [min, max] is mapped onto the signal's [0, 1] (BT.1886 builds its
black lift from the same two, and HLG its system gamma from the peak). So
libplacebo is given the luminances of the description the surface will
actually carry:

- SDR: the compositor's own when they can be sent (`set_luminances`, on
  an exact match), else the ones the chosen curve implies, since those are
  what the compositor reads the surface with - BT.1886's 0.01/100/100 from
  BT.2035, everything else's 0.2/80/80 from srgb.xalter - scaled so that
  the reference white is 203: `max = 203 * max / ref`, `min = 203 * min /
  ref` (at least `PL_COLOR_HDR_BLACK`). For sway's 0.2/80/80 that is 203
  and 0.5075 cd/m², 400:1 and no headroom; a preference with its maximum
  above its reference white gets that headroom as a target above 203.
- PQ: the compositor's target volume, which is where the tone mapper aims.
  A peak of 10000, PQ's own reach, or a range upside down means the
  compositor has no idea - wlroots sends the curve's defaults as the
  target, 10000 included - and a 1000 cd/m² display is assumed and logged.
  Luminances, where they can be sent, are 0.0001/10000/203: the protocol
  takes PQ's maximum as the minimum plus 10000, i.e. the minimum is an
  offset under the whole curve, while libplacebo's PQ is the absolute one,
  so the pair only agrees at a minimum of zero, of which 1e-4 cd/m² is the
  smallest step. The display's real black goes out as the mastering
  luminance instead.
- HLG: encoded for the display the description names: the compositor's
  target, sent as the luminances, where `set_luminances` exists; the
  protocol's nominal 0.005 to 1000 cd/m² HLG display otherwise.

Nothing needs scaling between the two sides: 203 is libplacebo's SDR
white, BT.2408's reference and the protocol's default reference for PQ
and HLG.

### 22.6 What is sent, and what is not

`set_primaries_named` and `set_tf_named` (or `set_tf_power`) always;
`set_luminances` in the cases above, and only with the feature;
`set_mastering_display_primaries`, with the feature, for HDR and for an SDR
target gamut smaller than the container; `set_mastering_luminance` with it,
for HDR only. Never `set_max_cll` or `set_max_fall`: this is a window, not
a mastered stream, and its peak moves with every frame and every grade.
Never a target volume outside the container. Every request is checked
against what the manager advertised before the first of a creator's goes
out, because a property skipped half way leaves the set incomplete, and
`create` on an incomplete set is itself a protocol error: a guard that is
missing is not a graceful failure but a dead window.

### 22.7 Run time

`preferred_changed2` (or version 1's `preferred_changed`) with an identity
other than the one held starts the same sequence without a single
roundtrip: `get_preferred_parametric`, on `ready2` `get_information`, on
`done` the resolver. wlroots does not de-duplicate identities (every copy
gets a new one), so the comparison that actually keeps a repeat from
re-tagging the surface is of the choices: protocol-side fields and the
whole target colour. A different choice makes a new description; on its
`ready2` it is set on the surface, the old one destroyed, and the new
target colour and depth published, which the picture pass's signature
notices and renders again. `failed`, on either description, is logged and
the old one kept.

`set_image_description` is double-buffered and lands with the next
`wl_surface.commit`, which is normally the present of the frame rendered
next against the new target, so colour and tag arrive together. A commit
of SDL's in between - a configure being acknowledged - can show one frame
of the old picture under the new tag; that is the price of not owning the
commits, and one frame is all it can cost.

The panel overlay is tagged with the target itself on an SDR target, so
its code values pass straight through as they always did (a target with
headroom has its peak brought down to 203 first), and on an HDR target as
BT.709 gamma 2.2, which libplacebo puts at 203 cd/m²: an SDR element, what
the compositor would make of an SDR window there.

### 22.8 Verified, and what is not

On sway 1.13 (wlroots, Vulkan renderer) with its headless 1280x720 SDR
output, nested, on the RX 6900 XT: the manager at version 2 with
parametric and set_mastering_display_primaries, curves gamma22, st2084_pq,
ext_linear, bt1886 and compound_power_2_4, primaries srgb and bt2020,
perceptual only; the output and the surface prefer srgb gamma22 0.2/80/80,
target the same. The window resolves that exactly, renders BT.709 gamma 2.2
from 0.5075 to 203 cd/m² with an eight-bit dither into RGBA16 UNORM
pass-through, and is tagged srgb gamma22, perceptual, with nothing else
sent. No protocol error, and no colour surface but ours on the connection.

Screenshots of the last frame of P1094366 (`--no-loop`), grim on the
output, tagged against `--no-colour-management`: the panel is identical
(0 of 227430 pixels outside its two live readout rows differ), which also
shows sway converts neither window - it takes the untagged one as the same
sRGB/gamma 2.2 it is told for the tagged one. The picture differs where the
two renders do: a dark patch goes from luma 34.4 to 42.9 codes, a mid grey
from 107.1 to 109.6, and nothing above code 160 moves by more than one.
That is the curve (piecewise sRGB against gamma 2.2) and the target's
black level (libplacebo's 1000:1 for an untagged sRGB target against the
output's 400:1, which moves the spline's black point): the spline sampled
from the clip's 9995 cd/m² to both targets and encoded both ways predicts
+8.5 codes at code 35 and +2.4 at 111. After resizing the window
(floating, 900x600) the swapchain is recreated twice, stays pass-through,
no second `get_surface` is sent, and the same comparison gives the same
table (dark +8.3, mid grey +2.4).

`--bench --loops 1`, three runs each, alternated: 39.0, 39.0 and 39.6 fps
tagged against 39.1, 39.1 and 39.2 untagged, the display running at the
pipeline's rate either way; the compose's overlay pass costs 0.021 ms of GPU
on RGBA16 against 0.010 on RGBA8, its dither pass 0.018 against 0.017.

The run-time path was exercised by a scratch build (not committed) that
announced a preferred change at 3, 7 and 11 seconds and replaced the
parsed preference with BT.2020 PQ, 0.005/10000/203, target Display P3
0.005 to 800 cd/m², for the first: the window went through
`get_preferred_parametric`, `get_information`, a new description with the
clipped P3 as mastering primaries (red 0.678762, 0.319411) and mastering
luminance 50/800, `set_image_description` and a ten-bit dither, back to
srgb gamma22 at 7 s, and the third announcement changed nothing. sway
converted the PQ window back to its SDR output and anchored 203 cd/m² to
its white: the panel's static pixels came out identical under both tags.

Untested: a real HDR output (sway's headless output refuses `hdr on`, and
only wlroots' DRM backend has output image descriptions, so the preferred
never changes on its own here), HLG, set_luminances, set_tf_power, the
relative intent and the version 1 events, none of which sway offers.

## 23. Retiming

### 23.1 The model

A clip shot at 59.94 or 29.97 fps can be conformed to a lower frame rate:
every source frame becomes one output frame, shown and written at the
chosen rate. A 59.94 clip at 23.976 plays 2.5 times slower and a range of
it exports as a 23.976 file 2.5 times as long. No frame is dropped,
repeated or synthesised, and nothing about the picture changes; the still
export does not see the rate at all.

The retime is two numbers. The output rate R is the rate the picture is
shown and written at. The speed S is the clip seconds that pass per output
second. Today S = R / the clip's own rate and is at most 1, one source
frame per output frame, so the picture for output frame k is source frame
k. Both mappings, output frame to clip time and clip time to output time,
are worked out in a handful of helpers in `tools/gui/main.c` and nowhere
else: `view_rate_native()` (the clip's own rate, 29.97 when it says none,
as `view_frame_duration()` assumes), `view_rate_out()` (R),
`view_speed()` (S, 1 when not slowed), `view_slowed()` (S != 1) and
`view_out_frame_duration()` (1/R, in seconds of what is watched).

The fixed rates are 23.976, 24, 25, 29.97, 30, 48, 50, 59.94 and 60. A
clip's stops are every one of them strictly below its own rate, compared
with `av_cmp_q`, and then the clip's own as the last; a fixed rate within
1/1000 fps of the clip's own is the clip's own, because the stream says
30000/1001 where a container may say 2997/100, while 30 and 29.97, 0.03
apart, stay two. A 59.94 clip has eight stops, a 29.97 one four, a 23.976
one only its own, and that row is greyed.

What is chosen - by the panel's row or `--rate` - is kept apart from what
is in effect: the choice carries from clip to clip, and each clip that
opens is set to it if it is below its own rate and to its own otherwise.
Kept as one value, the clamp would become the choice: a 23.976 clip in a
playlist of 59.94 ones would set it to 23.976's own, and the next 59.94
clip would play at its own. A `--rate` above a clip's own is refused when
that clip opens, with the clip's rate in the message; a headless run exits
non-zero, a window plays the clip at its own rate. The rate is the clip's
transport, not its look: it is in `struct view_state` rather than in
`struct view_look`, so neither the picture pass's signature, which hashes
the look, nor anything that kept a look would carry it.

The frame readout stays in clip frames and clip seconds, the clip's own
timeline. Left and Right step five seconds and Shift with them one second
as watched: frames = seconds x R, rounded as before, so five seconds of a
59.94 clip at 23.976 is 120 frames, two seconds of the clip.

### 23.2 The clock

The master clock stays in clip seconds: `st->pts` is clip time, the
decoder's timestamps are clip time and every seek is a clip frame number.
It runs at S. On the wall path `view_clock()` is `wall_pts + S x elapsed`,
and the smoother that steps the clock between reads steps at S as well:
stepped at one clip second per wall second under a raw clock running at 0.4
it would drift ahead towards (1 - S) times its 0.3 s time constant, 0.18 s,
and cross its 0.1 s jump threshold on the way. Everything that compares
`st->pts` with the clock - the drop threshold in `view_step_frame()`, the
catch-up loop and the hold in the run loop - is unchanged, because a clip
frame period of lag in clip time is an output frame period of lag in wall
time. The one conversion is the hold, which is in clip seconds while the
wait is in wall milliseconds: `ms = hold / S x 1000`. The pause folds the
wall time since the start into `wall_pts` times S.

The wall clock no longer runs before the first picture is up, which the
audio clock never did. Running from the clip's opening, it had put 0.25 s
of clip behind the first picture of P1081307 at 23.976 by the time it was
drawn, and fifteen frames were dropped to catch up.

The panel writes only the choice. The run loop makes it the rate at the
top of an iteration, before anything in it reads the clock, and in the
same step moves the sound and the clock (`view_rate_follow()`): to a
slowed rate the track is paused and the wall clock restarted from the
picture on screen at the new speed; back to the clip's own the track is
sought to that picture, and the loop's own start logic plays it. Neither
seeks the picture nor pauses it. Acted on an iteration after the panel set
it, a change ran the wall clock at the new speed from the old anchor in
between - 0.23 s ahead on going from 24 to 25 fps after 14 s at 24,
fifteen frames dropped - and read the paused track as the clock on the way
back, 14.3 s behind.

A slowed clip has no sound. The camera's own slow and quick motion and
variable frame rate modes record none, and a track slowed 1.25 times with
the picture is 12 log2 1.25 = 3.9 semitones down, slowed 2.5 times 15.9.
Its track is still opened, and neither pumped nor played. `view_audio_live()`,
a track and not slowed, stands in for "there is a track" wherever the
audio is driven or read as the clock: `view_clock()`, `view_audio_start()`,
the pump, the end of the clip (a slowed clip ends when its last picture
has been shown; its paused ring would never drain), the play and pause
block and the pause's fold. The seek in `view_step_frame()` still sends the
paused track to its target, which costs a demuxer seek and is undone by
the seek on the way back; closing a clip still pauses a playing device; the
mute and volume rows stay as they are, shown when the clip has a track.

### 23.3 The export

The file's frame rate is R: `e->fr = view_rate_out()`, and 25/1 as before
for a clip that says no rate and is not slowed. A frame's pts is its place
in the range, the frame counter, so the time base, `avc->framerate` and the
stream's `avg_frame_rate` follow R as they followed the clip's rate, and
the clip's own timestamps never reach the file. A slowed run writes no
audio stream at all, whatever the mode says. The mode is not changed - the
panel greys its rows with `no sound while slowed`, and the next run at the
clip's own rate has it again - and `view_encode_begin()` does not refuse a
container that cannot carry the mode, since nothing will be written to it.
The status line ends `no sound at 23.98 fps` and `-v` says `24000/1001 fps,
slowed from 60000/1001 ... no sound (slowed)`. The default file name is
unchanged.

What the file says: QuickTime keeps the time base, so 150 frames of
P1081307 at 23.976 are r_frame_rate 24000/1001, time base 1/24000, 6.25625
s, against 60000/1001 and 2.5025 s at its own rate; Matroska rounds its
timestamps to milliseconds, so the same range is 6.257 s there, and
ffprobe guesses the unslowed one's rate as 19001/317.

### 23.4 What motion estimation would change

libplacebo is growing motion estimation (merge request 876,
`pl_motion_estimate` in `libplacebo/shaders/motion.h`): block-matching
flow fields between a pair of frames, forward and backward, at block
granularity. With a frame interpolator on top of it, S stops following
from R and is chosen on its own - a 24 fps clip at 24 fps at half speed -
and the picture for output frame k is the clip at t0 + k x S / R,
synthesised between the two source frames either side of that time. The
sampler is the only thing that changes: which source frames an output
frame needs, two rather than one, and the pass that makes it from them.
The clock is already in clip seconds running at S, the hold already
converts at S, every seek is already a clip frame, the export's pts are
already output frames, and `view_rate_out()`, `view_speed()`,
`view_slowed()` and `view_out_frame_duration()` keep their meaning; only
what sets S changes. Nothing of it is built.

### 23.5 Verified

On the RX 6900 XT, with P1081307 (HEVC, 59.94 fps, 1440 frames) and
P1081308 (ProRes RAW, 29.97, 720 frames):

- Headless, frames 0 to 149 of P1081307, the default FFv1 (Vulkan) L3
  preset into `.mkv` (QuickTime does not carry FFV1): at 23.976 150
  packets, r_frame_rate 24000/1001, 6.257 s, no audio stream; at its own
  rate 150 packets, 2.503 s and the copied pcm_s24be. Frame 0 of the two,
  decoded to rgb48le, is the same 117006336 bytes; so is frame 149. The
  same range through ProRes (prores_ks) HQ at 1920 wide into `.mov`:
  24000/1001, 150 frames, 6.25625 s, no audio, against 60000/1001, 2.5025
  s and 120120 samples of sound; all 150 frames identical between the two
  (PSNR infinite).
- P1081308, frames 0 to 59 at 24: 60 frames at 24/1, 2.500 s, no audio.
  `--rate 60` on it is refused, `runs at 29.97 fps; --rate can only slow a
  clip down`, exit 1, no file. `--rate 23.98` writes 24000/1001;
  `--rate 26` is refused with the nine names and `clip`.
- In a window on sway's headless 1280x720 output at 60 Hz, nested, with
  SDL's dummy audio driver: P1081307 at `--rate 23.976` for 23 s, 541
  pictures, none dropped, one to the next 41.709 ms on average with a
  standard deviation of 0.477 ms (1/23.976 is 41.708), the clock
  advancing 0.40000 clip seconds per wall second, |a/v| at most 1.86 ms
  of clip time. At its own rate 16.699 ms, sd 0.873 ms; there the dummy
  driver's clock ran 1.0083 clip seconds per wall second, more than a 60
  Hz output can show of a 59.94 clip, and 9 frames were dropped in 22 s,
  as a build without this section dropped 9 in 16 s with its clock at
  1.0081.
- The row, dragged with a virtual pointer while playing with sound: from
  the clip's own rate to 24 (the drag passes 50, 30 and 25 on the way) and
  back (through 25 and 48), seven changes, every frame shown in order
  across them and |a/v| at most 4.7 ms around them, under a third of a
  frame; at 24, 41.667 ms with sd 0.435 ms and 0.4004 clip seconds per
  wall second (24/59.94 is 0.40040); at the clip's own rate before and
  after, 16.562 and 16.667 ms. `-v` says the sound was paused and resumed
  at the picture on screen.
- A 23.976 clip (P1000350) shows its one stop greyed, `23.98 fps, the
  clip's own`; `--rate 24` on it in a window prints the refusal and plays
  it at its own rate.

Not verified: an export started from the panel while slowed (the engine is
the same as the headless one, which was run), and a playlist switch
between clips of different rates, for want of a way to send keys to the
nested compositor.

## 24. Stills in cineview

### 24.1 What a still is

A clip of one picture: `nb_frames` 1, from the container's count or from
its duration at its rate. An RW2 is one by construction (its demuxer gives
it a duration of one second at 1 fps), and so is a one-frame cut of any
clip. `struct view_decoder`'s `still` says so, and four things follow.

- It is not looped. At the end of its stream - the step after its picture
  - it pauses instead, and the pause is the still's rather than the
  user's (`still_paused`): the next clip a playlist opens plays if the
  playlist was playing. Paused, everything a paused picture does works on
  it: a grade change develops it again, the auto white balance settles on
  it through replays. `--loops N` counts it as every pass at once and
  quits after it, `--no-loop` stops at it as at the end of any clip.
- A seek to it while it is on screen (Home, an arrow key, `,`) is the end
  of the stream at once rather than a read and a decode of the same
  picture; a cold seek, an export's, still decodes it.
- Its rate is a placeholder, so the Clip block leaves the frame rate out
  and the panel has no rate row for it; the scrub bar already needed more
  than one frame.
- The end of its stream does not take the picture away. The decoder
  receives into a frame of its own (`recv`) and hands it over only when a
  picture came: `avcodec_receive_frame()` unreferences what it is given
  before it looks, which at the end of a stream used to be the picture on
  screen, so a still, or a clip's last frame with `--no-loop`, could no
  longer be developed again for a grade change.

Why a one-frame clip never came up before: `view_decoder_feed()` drained
the decoder at the end of the stream only when the codec has
`AV_CODEC_CAP_DELAY`, and a decoder running frame threads holds every
picture back until each thread has had a packet or it is drained,
whatever its own delay. With one packet - or with fewer packets than
threads, a ten-frame ProRes 422 clip on 13 threads - nothing ever came
out: the end of the stream arrived, the loop sought back to the start, the
flush threw away the picture in flight, and round again (a one-frame
ProRes 422 clip: 1041 passes and no picture in 35 s; the ten-frame one:
three passes, no picture). A frame-threaded decoder is now drained too,
and a clip of more frames no longer loses its last ones at every loop
point either. A one-frame ProRes RAW clip did show its picture (one
thread, nothing held back), and then re-decoded and re-developed it about
twenty times a second at the loop point.

### 24.2 The RW2 path

The rw2 decoder has no hwaccel, so every RW2 decodes into system memory.
`view_get_format()` takes the software format for a raw clip when no
Vulkan one is offered (it used to refuse), and `view_decoder_next()`
uploads the picture into a Vulkan frames context of the clip's own on the
decoders' device (`view_decoder_upload()`): sw_format the frame's Bayer
format, the frame's coded size, storage and transfer usage, made on the
first such picture and released with the clip. `av_hwframe_transfer_data()`
keeps the hwcontext's contract on the AVVkFrame it fills - the copy waits
for the frame's timeline semaphore at its value and signals the next, and
writes the layout, access and queue family it leaves back into the frame,
or copies on the host when the device can - which is exactly what
`view_gpu_develop()` waits for and transitions from under the frames lock,
as it does for a decoded picture. `av_frame_copy_props()` carries the side
data, the metadata, the crop and the timestamps across. Everything after
the decoder therefore sees what the hwaccel would have handed over, the
frame pool change detection (`frames`) included. A 24 MP mosaic uploads in
4.5 to 8 ms, a 96 MP one in 19 to 20 ms.

The decoder runs slice threads, as many as FFmpeg picks: a still is one
packet, its stripes are the slices, and frame threads would give it one
core. One thread is kept for a decoder with a Vulkan hwaccel, whose
reason (the decode threads' submissions moving the AVVkFrame's state
while the develop reads it under the frames lock) does not apply to a
software decoder, which never touches a frames context. The panel's decode
time for a 24 MP still, the upload in it, was 56 to 119 ms, and 246 to 251
ms for the 96 MP one.

A decoder in the pool (view.h, `view_codec`) takes the next clip of its
codec through a flush and the stream's extradata on its first packet. The
rw2 decoder cannot: it reads the file's whole header - the stripes, the
levels, the white balance - from the context's extradata on every picture
and looks at nothing on the packet, so a pooled decoder handed a second
still decoded its stripes with the first one's header ("Error decoding
stripe 1", and the viewer gave up on the playlist). It is kept only for a
stream with the same extradata, the same file coming round again, and
replaced by a new one otherwise, which for a software decoder costs
nothing (`view_decoder_takes()`).

The model, and with it the camera's tables, is read from the first
picture's metadata when the container names no camera (`profile_late`),
before that picture is developed, along with the Clip block's camera,
lens, ISO, aperture, shutter and date out of the Exif; the white balance
row is the as-shot temperature of 9c. A display matrix on the picture -
the decoder's translation of the Exif orientation - turns it the way a
track header's turns a clip, when the track had none.

### 24.3 Verified

On the nested compositor, with `-v`:

- `_1000406.RW2` alone: one picture presented, then nothing decoded or
  developed while it stayed up (the panel's pipeline rate 0.0 fps); `q`
  quits.
- A playlist P1000319.MOV, _1083412.RW2 (96 MP), P1000319.MOV with `t`,
  three runs: the clip plays, the still comes up, and the next clip plays
  again from the pool, every frame of it shown in order. Two runs dropped
  nothing; the first dropped 25 frames in the first second of the first
  clip, before the still had been opened, and that did not happen again.
- Four stills in a row, two of them portraits with Exif orientations 6 and
  8: each opens a decoder of its own and comes up upright, matching the
  camera's JPEGs.
- `--wb auto` on _1000406: the first replay reads the histogram and the
  balance goes from the as-shot 3371 K to the estimate, 2741 K (confidence
  0.01: the first reading starts the track whatever its confidence, as on a
  clip), the second replay finds it there and the replays stop.
- `--loops 1` quits after the picture, `--no-loop` keeps it, play and Home
  on a still decode nothing.
- The one-frame ProRes RAW and ProRes 422 clips and the ten-frame ProRes
  422 one above, before and after.
- Headless: `--export 0 out.tif` of an RW2 writes the 6000x4000 picture;
  `--export-video 0 0` writes one frame; the scene-referred XYZ of that one
  frame (`--colour xyz`, FFv1 (Vulkan) L4, gbrpf32le) is bit for bit
  cineraw's `--space xyz --encode linear --float32` PFM of the same still,
  same md5 of the float samples. The display-referred still export is not
  cineraw's SDR file and never was: libplacebo's tone mapping against
  cineraw's per-channel roll-off, a mean difference of 0.105 of full scale
  on _1000406 and 0.150 on P1000387 frame 13.
