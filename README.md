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

`craw-process` decodes a clip with FFmpeg, runs every frame through the
GPU pipeline and writes PNG, PFM or PPM sequences. The options that
matter most:

    craw-process --wb auto -v --frames 30 clip.MOV        estimate the illuminant
    craw-process --wb 5600:0.02 --ei 400 -o out_%05d.png clip.MOV
    craw-process --bench --frames 60 clip.MOV             GPU and decode timing

`--wb` takes `as-shot`, `auto` or `CCT[:TINT]`; `--ei` re-rates the clip
at a working exposure index, `--exposure` adds a bias in stops; `--encode`
picks the transfer function for image output, `sdr` by default for PNG,
which rolls highlights off per channel before the sRGB curve. The
Panasonic DC-S1M2 per-CCT tables are built in and selected from the
container's model name; `--tables` loads others from a text file.

`craw-view` is the interactive viewer and exporter built on SDL3, nuklear
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
view, never the export. With four mono audio tracks the last two are taken
as the stereo pair (`--audio-map` overrides). `craw-view -h` lists the
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

## Layout

    include/craw/   public headers
    lib/            library sources, shaders under lib/shaders
    lib/ref/        CPU reference implementation
    tests/          tests and the test device helper
    tools/          build helpers, craw-process, and the craw-view GUI

## License

BSD-2-Clause, see LICENSE.md. The vendored Vulkan Memory Allocator is MIT.
