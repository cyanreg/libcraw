# libcraw

Cinema RAW processing library. Turns camera-native Bayer sensor data into
scene-linear colour on the GPU with Vulkan compute, and nothing else: no
container parsing, no decoding, no display. Decoding is FFmpeg's job,
rendering is libplacebo's; libcraw sits between them.

First target: Panasonic ProRes RAW as decoded by FFmpeg (Lumix S1II).
ARRIRAW and BRAW are planned.

## What it does

- Black/white normalization, white balance, clip handling.
- Debayering: bilinear, Malvar-He-Cutler, and a ratio-corrected
  directional method (RCD). Full resolution output.
- Denoising in the raw domain, before the debayer can spread sensor noise
  into colour: non-local means on the four CFA planes in a variance
  stabilised domain, so one strength works from the shadows to the clip
  point. About 1.9 ms on a 5.8K frame on an RX 6900 XT.
- A sensor noise model, variance = a * signal + b by exposure index, held
  in the camera profile, with a calibrator that fits it to a clip.
- Camera profiles with per-CCT white balance and matrix tables, as-shot
  exact and re-balance modes, CCT plus tint illuminant model.
- Colour: camera matrix, gain, exposure, output to XYZ D65, BT.709,
  BT.2020, P3-D65, V-Gamut, ACES AP0 or AP1, linear float.
- Auto white balance: GPU histogram with two readings, one over every
  neutral-looking surface and one weighted towards light sources and
  glare, a locus-based estimator, and a temporal smoother.
- Exposure the way Apple's ProRes RAW plug-in rates a clip: frame gain
  times working index over as-shot ISO, times an exposure bias.
- A CPU reference of every stage, used by the tests.

See DESIGN.md for the exact definitions, conventions and the reasoning.

## Ownership model

The caller owns the VkInstance, VkDevice, queues and command buffers.
libcraw records dispatches and barriers into a command buffer it is given
and allocates its intermediates on the caller's device through the AMD
Vulkan Memory Allocator. Vulkan 1.4 is required.

## Building

    meson setup build
    ninja -C build
    ninja -C build test

Requires meson, a C23 compiler, a C++17 compiler (one translation unit for
VMA), the Vulkan headers and loader, and glslang. Tests need a Vulkan 1.4
device; llvmpipe is enough.

## Tools

`cineraw` decodes a clip with FFmpeg, runs every frame through the
GPU pipeline and writes PNG (16 bit), PFM (linear float) or PPM (8 bit,
dithered, because a denoised shadow gradient bands without it) sequences.
The options that matter most:

    cineraw --wb auto -v --frames 30 clip.MOV        estimate the illuminant
    cineraw --wb 5600:0.02 --ei 400 -o out_%05d.png clip.MOV
    cineraw --bench --frames 60 clip.MOV             GPU and decode timing
    cineraw --denoise 1.0 --crop 2750,1450,1000,1000 -o c_%d.png clip.MOV
    cineraw --noise-estimate --frames 10 flatfield.MOV   fit the noise model

`--denoise` takes a strength in units of the camera's modelled noise
level, 1.0 meaning "remove exactly that much"; `--noise A,B` overrides the
model and `--noise-estimate` measures one, printing a line ready for a
`--tables` file. `--crop X,Y,W,H` writes one rectangle of the output while
the pipeline still runs on the whole frame, which is how a 5.8K frame gets
looked at 1:1 without a 5.8K file.

`--wb` takes `as-shot`, `auto` or `CCT[:TINT]`; `--ei` re-rates the clip
at a working exposure index, `--exposure` adds a bias in stops; `--encode`
picks the transfer function for image output, `sdr` by default for PNG,
which rolls highlights off per channel before the sRGB curve. The
Panasonic DC-S1M2 per-CCT tables are built in and selected from the
container's model name; `--tables` loads others from a text file.

`cineview` is the interactive viewer and exporter built on SDL3, nuklear
and libplacebo, when those are found. It decodes with FFmpeg's Vulkan
hwaccel on libplacebo's device, runs libcraw on the same device, reads the
file on its own thread and loops by default (`--no-loop`). Space plays and
pauses; Up and Down walk the panel's rows and Left and Right adjust the
focused one; Shift with Left and Right seeks one second; Page Up and Page
Down zoom in and out, and Shift with them goes to the first and last frame;
Control with the arrows pans; Tab or `0` fits the picture again and `1` is
one to one; Enter opens the film simulation menu on its row and Escape
closes it; the wheel zooms about the pointer and a left drag pans; `m`
mutes; `q`, or Escape with no menu open, quits. Zoom and pan only touch the
view, never the export. Its "Denoise" section switches the raw stage on and
sets its strength, and reports the noise model it found for the clip's own
ISO — the stage stays off when the camera has none, which the section says
rather than leaving you guessing; `--denoise STRENGTH` sets it from the
command line. With four mono audio tracks the last two are taken
as the stereo pair (`--audio-map` overrides). `cineview -h` lists the
options.

It also carries 296 film simulation LUTs from
[filmgrid.org](https://filmgrid.org/), which derive from the open-source
G'MIC film emulation collection — see `tools/gui/luts/LICENSE.md` for the
provenance and the attribution they carry. The panel's "film simulation"
row opens a cascading menu over the same two category trees the site uses,
Films (colour negative, slide, black and white, instant, cinema print,
Fujifilm simulations) and Looks (warm, cool, vintage, cinematic and so on).
`--lut` takes a rendering's name or slug as well as a file path, and
`--list-luts` prints both trees. The `.cube` files live in the tree; the
compact pack that ends up in the executable is built by
`tools/luts_build.py` at build time.

The panel's "Film grain" section emulates film grain through libplacebo's
AV1 grain synthesis, applied after scaling and colour management so that
what is graded is what is on screen, at the size it is on screen. It is
driven by gauge and stock rather than by codec parameters: a gauge sets how
much the frame is magnified and so how coarse the grain looks, a stock sets
how much of it there is, and a response says how it varies with exposure.
`--grain GAUGE[,STOCK][:AMOUNT]` sets it from the command line, as in
`--grain super8,500t:0.8`. It is off until it is turned on, and it has
nothing to do with the film simulation LUTs: the two are independent, and
neither ever changes the other. The grain's seed comes from the frame
number, so a frame exported twice with the same settings is the same file
twice. DESIGN.md section 13 has the derivation, the published granularity
figures behind the stocks, and which of them are estimates.

The picture can be cropped for the export. `c` turns crop mode on, a left
drag over the picture then draws the rectangle instead of panning, and
everything outside it is shaded with a thin border around what is kept; the
panel's crop row reports the region and clears it. The rectangle is held in
source pixels with even edges, so zoom, pan and a resized window leave it
where it is, and the export writes exactly that region at source
resolution. `--crop WxH+X+Y` sets the same rectangle from the command line.

`--export FRAME OUT` writes one frame to one file and exits without opening
a window; the Export panel does the same for the frame on screen. The file
name picks the format through libavformat, which knows both the muxer and
the codec an extension implies: `.png` and `.tif` take the sixteen bit RGB
the frame is rendered as, `.jpg`, `.webp` and `.avif` whatever their encoder
supports, deepest first, through libswscale where that is not RGB;
`--codec` overrides the encoder and `--enc KEY=VALUE,...` passes it options
(`cpu-used=8,threads=2` for an AVIF). PNG is written at zlib level 1 (a
fifth of the time of the default level for a per cent of the size) and
TIFF uncompressed. The stream is tagged as what the export really is: full
range, BT.709 primaries, sRGB transfer, display referred. A sequence of
stills is deliberately not offered — that is a video encode.

## Layout

    include/craw/   public headers
    lib/            library sources, shaders under lib/shaders
    lib/ref/        CPU reference implementation
    tests/          tests and the test device helper
    tools/          build helpers, cineraw, and the cineview GUI

## License

BSD-2-Clause, see LICENSE.md. The vendored Vulkan Memory Allocator is MIT.
