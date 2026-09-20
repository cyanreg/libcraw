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
`--tables` file. `--false-colour PASSES` runs one to five passes of a
median and a blur over the colour differences after the debayer,
RawTherapee's false colour suppression steps, which turn the coloured
dots that rain glints and other sub-pixel highlights leave into neutral
ones and touch nothing else. `--crop X,Y,W,H` writes one rectangle of the output while
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
file on its own thread and loops by default (`--no-loop`). The panel is
worked with the pointer: its switches, expanders, sliders, fields and
cascading menus are clicked, and no key reaches a panel row. The keys are
the transport's and the view's. Space plays and pauses; Left and Right
seek five seconds back and forward, and Shift with them one second; Page
Up and Page Down zoom in and out, and Shift with them goes to the first and
last frame; Control with the arrows pans; Tab or `0` fits the picture again
and `1` is one to one; the wheel zooms about the pointer and a left drag
pans; `r` and `t` go to the previous and the next file; Ctrl+1 and Ctrl+2
mark the start and the end of the video export at the frame on screen, the
whole clip being the default; `m` mutes; Escape closes an open menu, calls
off an armed "pick white", and otherwise quits, as `q` does. Zoom and pan
only touch the view, never the export. While a video export runs the whole panel and every key but `q` are
locked, the picture is the export's; `q` then stops the export, finishes
the file and quits.

The panel is set in whatever face fontconfig names for `sans-serif` on the
machine, at 15 px, the way every other desktop application finds its UI
font: there is no option and nothing to configure, and nuklear's own
bitmap face is only what remains when fontconfig has nothing or the file
will not load. Under the playback readouts sits a "Clip" block with what
the file says about the recording, one row per fact and only the rows the
file has: camera, format (the codec with its bit depth and coded size,
"ProRes RAW 16-bit 5760x3040"), frame rate, ISO, aperture, the white
balance as recorded, the exposure index when it differs from the ISO, the
capture gamma and gamut of a developed clip, the timecode at the clip's
start and the recording date - the camera's own, with its zone, when the
clip carries the Panasonic XML, else the container's creation time in the
machine's zone. All of it comes through FFmpeg's demuxer: the Apple and
RDD 18 tags a ProRes RAW clip carries, the stream's own tags and the XML,
which a developed clip has too and an MP4 does not. The shutter angle, the
focal length and the lens name sit in the maker note the demuxer does not
export from these files; they arrive with a libavformat change, not with a
box parser here. `-v` prints the same rows once per clip.

The panel is a column of sections — White balance, Exposure, Debayer,
Denoise, Lens, Look, Adjustments, Film grain, Crop, Export — each behind a
header with a power switch at the left of its title, the way RawTherapee
puts an enable button on every tool. Off bypasses that section while every
setting in it is kept, so switching it back on returns the picture it had:
Exposure off is a bias of zero at the as-shot index with the slider still
where it was, Look off drops the film simulation or the `--lut` cube
without unselecting it, Adjustments off drops soft light and vibrance, Film
grain, Denoise and Lens off turn their stages off, and Crop is the same
switch `c` is. White balance and Debayer have nothing to bypass and carry
an empty slot so the titles stay in line. A lit switch always means the
section is doing something:
Exposure, Look and Adjustments start off and switch themselves on the
moment one of their parameters is touched — a slider moved, a film
simulation picked, `--exposure`, `--ei`, `--lut`, `--softlight` or
`--vibrance` given — the way RawTherapee's tools do. `--denoise 0`, `--lens
off` and the absence of `--grain` start those three off, and `--bypass
exposure,look,adjustments` pins the other three off whatever the other
options set, which is how a headless export is handed the state the panel
is in.

The White balance section's "pick white" button is the spot white balance
of every raw developer: press it, and the next left click on the picture
— the pointer is a crosshair while it waits, and Escape or a second press
calls it off — takes the 24x24 source pixels under the click as something
that ought to be white and sets the manual white balance to the illuminant
that makes them so. The patch is read back out of libcraw's own output for
the frame on screen (paused first: a pick is a decision about one frame),
the colour stage is undone through the profile's matrix and gains for the
illuminant the frame was developed with, and the raw ratios that leaves go
to `craw_profile_neutral_to_illuminant()`; the temperature and tint land
in the manual sliders. Pixels within 2% of the clip or below the camera's
noise floor are left out, and a patch with fewer than a quarter of its
pixels usable is refused with the one word beside the button, "clipped" or
"too dark". The click does not pan and does not draw a crop, whatever mode
the picture is in. `--wb pick:X,Y` is the same pick on the source pixel
X,Y of the first frame the run develops — frame 0 in the window, the
exported frame headless — and `-v` prints what it read and found, then
checks it: the same patch under the picked illuminant read again, which
comes out neutral to within a percent (the locus is sampled every 5
mired, and what is left is the half step between samples), and picked
again, which returns the same illuminant.

The "Debayer" section picks the method and carries the false colour
suppression row: the number of median and blur passes over the colour
differences after the debayer, zero to five, which is what stops a wet street from
shimmering in colour as the rain hits it; `--false-colour PASSES` sets it
from the command line. The "Denoise" section sets the strength and reports the noise model it
found for the clip's own ISO — the stage stays off when the camera has
none, which the section says rather than leaving you guessing; `--denoise
STRENGTH` sets both the switch and the strength from the command line.
With four mono audio tracks the last two are taken as the stereo pair
(`--audio-map` overrides). `cineview -h` lists the options.

It takes any number of files, which makes a playlist: `r` and `t` step to
the previous and the next clip and wrap at both ends, and the panel's clip
line reads `3/7 P1000347.MOV [hevc video, gpu]`. The clips need not match —
ProRes RAW and already developed, different sizes, stereo or four-channel or
no audio at all — and a switch keeps every setting exactly as it was: the
white balance mode and a manual CCT, the exposure index and bias, debayer
and its false colour passes, denoise, lens correction and its fit, the
export's colour space, size,
pixel format, audio mode and threads, the two export file paths once typed,
the film simulation, the tone map, the adjustments, the grain, every
section's switch and the zoom. A setting a clip's kind
cannot use is kept rather than dropped, so the white balance a raw clip was
graded with is still there after a developed one. Only what is read from the
clip is read again: its tables and as-shot values, its crop, its lens
records, its metadata. A file that cannot be opened is reported and leaves
the viewer on the clip it is showing, and `--export` and `--export-video`
take the first file only.

It also carries 296 film simulation LUTs from
[filmgrid.org](https://filmgrid.org/), which derive from the open-source
G'MIC film emulation collection — see `tools/gui/luts/LICENSE.md` for the
provenance and the attribution they carry. The panel's "film simulation"
row opens a cascading menu over the same two category trees the site uses,
Films (colour negative, slide, black and white, instant, cinema print,
Fujifilm simulations) and Looks (warm, cool, vintage, cinematic and so on),
by a click on the row; a click elsewhere, a choice or Escape closes it.
`--lut` takes a rendering's name or slug as well as a file path, and
`--list-luts` prints both trees. The `.cube` files live in the tree; the
compact pack that ends up in the executable is built by
`tools/luts_build.py` at build time.

The Look section also holds the "tone map" row: the curve the window is
tone mapped with, out of libplacebo's own list — spline, the default;
bt2390; bt2446a; st2094-40; st2094-10; reinhard; mobius; hable; gamma;
linear; linearlight; clip — with `--tone-map NAME` as the starting value.
It is a display setting rather than a look: it sits above the section's
switch and is applied whatever else is bypassed, and every display-referred
export is mapped with the same curve, so the file keeps matching the window
whichever one is picked. Nothing else of the mapping has a row; the gamut
map and the peak detection stay libplacebo's defaults.

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

A clip shot with the camera on its side carries a display matrix in its
track header, and cineview turns the picture by it the way mpv and ffmpeg's
autorotate do: a 5760x3040 clip with a quarter turn is 3040 wide and 5760
tall everywhere it is looked at or written. The fit, the zoom, the pan about
the pointer, `1` and Tab, the crop rectangle as it is drawn and every size
the panel reports are in those pixels, and every export is written upright
and carries no display matrix of its own. Nothing upstream of the renderer
turns: libcraw's pipeline, the lens correction, the auto white balance
histogram and the denoise all run on the mosaic the sensor read out, in its
own orientation. The clip line says which turn it is — `P1001019.MOV
[prores_raw raw, 90°]` — and a matrix that is not a multiple of ninety
degrees, or that mirrors as well as turns, is reported and left alone rather
than approximated.

The picture can be cropped for the export. `c` turns crop mode on, a left
drag over the picture then draws the rectangle instead of panning, and
everything outside it is shaded with a thin border around what is kept; the
Crop section's switch is that same mode, and its body reports the region.
Switching the mode on clears whatever rectangle there was, so the whole
frame is back and a new one can be drawn; switching it off keeps the one
that was drawn, shaded, for the export — `c`, draw, `c` leaves the
rectangle in place with the mode off, and `c` again starts afresh. There
is no clear button: on again is the clearing. The rectangle is held in
source pixels with even edges, so zoom, pan and a resized window leave it
where it is, and the export writes exactly that region at source
resolution. On a rotated clip it is drawn on the picture as it is shown and
mapped through the turn both ways, so the shade and the readout follow what
was drawn and the file holds exactly that region, upright. `--crop WxH+X+Y`
sets the same rectangle from the command line, in source pixels — which on a
rotated clip is the readout's rectangle turned back.

The Export section holds two exports on two tabs, **Screenshot** and
**Video**, independent of each other: the screenshot is the frame on
screen, one image, and the video is a range, whose start and end fields
are bounded by the clip's last frame — Ctrl+1, Ctrl+2 and `--export-video`
hold to it the same way, so a range cannot reach past the clip. The tab
row sits at the top of the section, the chosen tab lit and the other
dimmed, and only the chosen tab's rows are laid out under it; the choice
is kept for the session and starts on Screenshot. Each tab has its own
file field, its own button and its own status line, neither field names
the other's file, and a field keeps what was typed into it while the other
tab is up. Each field follows its own default name until it is typed into
— the screenshot's the clip and the frame, the video's the clip and the
range in the clip's directory — and once a path is typed it is the user's
and is kept, across clips too. A typed path always wins. Nothing judges a
name while it is being typed: the extension picks the container when the
button is pressed, a name with no extension libavformat knows gets the
family's container appended (`.png` on the Screenshot tab) in the field
itself, and a container that will not carry the encoder or the sound is
refused on the status line under the button, in a sentence.

`--export FRAME OUT` is the screenshot from the command line: it writes one frame
to one file and exits without opening a window, upright and at the crop
rectangle's own size. The file name picks the format through libavformat,
which knows both the muxer and the codec an extension implies: `.png` and
`.tif` take the sixteen bit RGB the frame is rendered as, `.jpg`, `.webp`
and `.avif` whatever their encoder supports, deepest first, through
libswscale where that is not RGB; `--codec` overrides the encoder and `--enc
KEY=VALUE,...` passes it options (`cpu-used=8,threads=2` for an AVIF). PNG is
written at zlib level 1 (a fifth of the time of the default level for a per
cent of the size) and TIFF uncompressed. The stream is tagged as what the
export really is: full range, BT.709 primaries, sRGB transfer, display
referred. A sequence of stills is deliberately not offered — that is a video
encode. `--export-video START END OUT` is the video export headless. The
Screenshot tab has no codec box: the extension is the choice.

Every video export into an `.mp4` or `.mov` is written with `movflags
+faststart`, always: the `moov` atom is moved ahead of the media so the file
plays before it has finished downloading. There is no option to turn it off.

The **Video** block also carries the sound (see below), the number of
encoder **threads** — 0 for FFmpeg's own auto, 1 to the machine's core
count, applied to the software encoders and greyed for a Vulkan one, which
has none; `--threads N` sets it and `-v` prints the count the encoder
settled on since some clamp it — and, while a run is going, an *exporting: N
of M frames* line, a progress bar and a stop button. From the moment the
export starts until it ends or is stopped, every panel section and every row
but the stop button is disabled, the tab row with them, held on Video:
nothing can be scrubbed, seeked or changed by mouse, wheel or keyboard, the
scrub bar takes no input, and the key
handler ignores everything but `q` (which stops the export cleanly and then
quits) and Escape closing a menu. When the export ends everything unlocks
and the picture returns to where it was.

### Two kinds of export

There are two, and they are not two settings of one thing.

A **display-referred** export — BT.709, BT.2020 SDR, BT.2020 PQ, BT.2020
HLG — is the picture as the viewer shows it. libplacebo renders it into the
target exactly the way it renders the window: the same tone mapping — the
Look section's curve — the same gamut mapping and the same per-frame peak
detection, with the film simulation, the adjustments and the grain on it,
encoded with the target's own transfer. The export has no mapping choice of
its own, because it does not get to decide how the picture is mapped any
more than the window does: the mapping is the display's, so the file is the
frame on screen. The one mapping parameter the selection still holds is
PQ's target peak in nits, because that is part of what the HDR target *is*
rather than a choice about the picture. (An export used to default to a
bt.2390 map with peak detection off, and a file that came out thirteen per
cent brighter in the mid-tones than the frame on screen is what that
bought.) One caveat is libplacebo's: when the window shows the picture
downscaled it measures the peak after the scaler, on the smaller picture,
while a full-size export measures it on every pixel, so a curve that leans
on the peak — bt2390 above all — can come out a percent or two brighter in
the file than on a fitted window; spline, the default, is all but
insensitive to it.

A **scene-referred** export — ACEScg, ACES2065-1, CIE XYZ D65 — never goes
near the renderer. The frame is put through the pipeline a second time with
the output space asked for, into an RGBA32F texture, and those floats are
read back and handed to the encoder: no tone curve, no gamut mapping, no
film simulation, no grain, nothing display-referred anywhere between
libcraw's output and the file. ACEScg carries the ACES 1.3 reference gamut
compression the display hand-off carries, so the file holds the numbers the
screen was shown; ACES2065-1 and XYZ are uncompressed, and none of the three
clamp negatives. A rotated clip is turned during the de-interleave that
splits those floats into the encoder's planes — a strided walk of the
read-back rather than a pass on the device, which at 5.8K takes the
de-interleave from 77 ms a frame to 225.

Because those four display-referred operators cannot reach such a file, the
Look, Adjustments and Film grain sections grey out while one is selected,
each with a line saying why, and their hooks come off the *display* too — so
what is on screen is what the file holds, up to the display transform. Greyed
rows take no input, and the film simulation cascade will not open.

Linear light needs float samples, which of the encoder families here means
FFV1 at version 4 — `FFv1 (Vulkan)` or `FFv1` with an L4 preset, writing
gbrpf32. Any other family or preset is refused with the reason rather than
opened with whatever format it happened to accept. The scene-referred paths
are written at the source's own resolution only; the size row is refused for
them, while the crop rectangle still works, because a crop is a rectangle of
the read-back and a resize is a display decision. H.273 has no code points
for the ACES spaces, so those files carry unspecified primaries with a linear
transfer and a `COLOUR_SPACE` container tag naming the space; XYZ uses the
ST 428-1 primaries code point and the same tag.

The Export section's size row — full, 1/2, 1/4, 3840 wide, 1920 wide, or a
typed WxH — applies to a display-referred export. `--size` sets the same
thing from the command line and still takes a bare `WxH`. It exists because a
software encoder at 5.8K wants three and a half gigabytes. The row and its
`writes` readout are about the picture as it comes out, so on a clip with a
quarter turn `1/2` of a 5760x3040 frame is 1520x2880 and `1920 wide` is 1920
across the turned frame — and a Vulkan H.264 or HEVC encoder meets its 4096
limit on what is now the height.

### The audio

The video export optionally carries the clip's sound, chosen by the Video
block's audio row and `--audio MODE`: `none`; `copy` (the default), the
source track remuxed and cut to the exported range on a sample boundary;
`opus:KBPS` (libopus, 160 by default, in 120 ms frames); `aac:KBPS` (160);
`flac:LEVEL` (0 to 12, 5); or `wav`, PCM at the source's own depth — 24 bit
little endian, or 16 bit for a 16 bit clip. The four encoders decode the
source (it is PCM in these MOVs), take it through libswresample only where
the encoder wants another sample format or rate — opus wants 48 kHz float,
aac planar float, while flac and PCM take the samples as they are — encode
it cut to the same range as `copy`, and interleave the packets with the
picture. One numeric field beside the mode is its bitrate or its level as
the mode needs. Which container carries which codec is asked of the muxer
itself, so a mode the file cannot hold is greyed with the muxer's own
reason: this FFmpeg's QuickTime muxer takes neither Opus nor FLAC (*only
supported in MP4*), while `.mp4` and `.mkv` carry all of aac, opus, flac and
PCM. The four-mono-track rule holds — the last two of four are the stereo
pair — and a single four-channel stream is taken whole.

### The pixel format

The pixel format row under the colour one says what the picture is written
as — `4:2:0 8-bit (nv12)`, `4:2:2 10-bit (yuv422p10le)`, `RGB 16-bit
(gbrp16le)` — and `--pix-fmt NAME` says the same thing from the command
line, by FFmpeg's own names. The row starts on the format the chosen preset
implies and remembers an explicit choice until the family changes, since the
next family's encoder need not take the same formats at all.

The list is fixed per family, in `presets.c`. A software encoder declares
what it takes, but a Vulkan encoder declares only `vulkan`, and the only way
to ask it is to open it with a frames context of each candidate — 136 ms an
open for `ffv1_vulkan` on an RX 6900 XT, about half a second for the ProRes
and APV ones, over a hundred opens for the table — which is not a thing to
do at run time. The lists are what those opens found on that GPU, trimmed to
the layouts a graded picture is written in: 4:2:0, 4:2:2, 4:4:4 and RGB at
the depths each encoder has, no gray, no alpha. `--list-presets` prints
them. A format an encoder refuses for a particular frame, as `ffv1_vulkan`
does for 4:2:0 because its slice grid must divide the chroma planes, is
reported in the encoder's own words at export time.

A format the encoder does not accept is never offered. One it accepts that
the preset's profile forbids is offered greyed, with the reason, because the
profiles that *are* a chroma layout have to win: ProRes proxy, LT, standard
and HQ are 4:2:2 and 4444 and 4444 XQ are 4:4:4, and `prores_ks` reads the
layout off the frame rather than off the profile, so a 4:4:4 frame under
`-profile hq` comes out subsampled 4:4:4 and labelled HQ. AV1 Main is 4:2:0
and High is 4:4:4. x264 sets no profile of its own precisely so that this row
means something: it picks High, High 10, High 4:2:2 or High 4:4:4 Predictive
from the format it is given. APV takes its profile from the format too, so
there is nothing there to refuse.

The engine renders straight into whatever was chosen. On the Vulkan path it
is the frames context's software format, so libplacebo converts to YCbCr and
subsamples on the way into the encoder's own image; on the software path
libswscale takes the read-back to it. Either way the stream is tagged as what
it is: RGB full range, YCbCr studio range with the colour entry's matrix, and
left chroma siting where there is chroma to site and none stated where there
is not. A scene-referred export has one format and the row shows it: the
family's float one, `gbrpf32`.

## Layout

    include/craw/   public headers
    lib/            library sources, shaders under lib/shaders
    lib/ref/        CPU reference implementation
    tests/          tests and the test device helper
    tools/          build helpers, cineraw, and the cineview GUI

## License

BSD-2-Clause, see LICENSE.md. The vendored Vulkan Memory Allocator is MIT.
