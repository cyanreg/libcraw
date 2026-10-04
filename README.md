# libcraw

Cinema RAW processing library. Turns camera-native Bayer sensor data into
scene-linear colour on the GPU with Vulkan compute, and nothing else: no
container parsing, no decoding, no display. Decoding is FFmpeg's job,
rendering is libplacebo's; libcraw sits between them.

First target: Panasonic ProRes RAW as decoded by FFmpeg (Lumix S1II), and
the same camera's RW2 stills where the installed FFmpeg has an RW2 decoder
(upstream does not yet). ARRIRAW and BRAW are planned.

## What it does

- Black/white normalization, white balance, clip handling.
- Debayering: bilinear, Malvar-He-Cutler, and a ratio-corrected
  directional method (RCD). Full resolution output.
- Denoising in the raw domain, before the debayer can spread sensor noise
  into colour: non-local means on the four CFA planes in a variance
  stabilised domain, so one strength works from the shadows to the clip
  point. About 1.2 ms on a 5.8K frame on an RX 6900 XT. In front of it,
  optionally, a temporal average in the same domain: every sample averaged
  with the previous frame's wherever the 3x3 around it agrees within the
  noise, so a static shot averages over the last few frames and whatever
  moved is left to the spatial filter, which then keeps more of the
  detail. About 0.4 ms more.
- A sensor noise model, variance = a * signal + b by exposure index, held
  in the camera profile, with a calibrator that fits it to a clip. The
  S1II's is measured on its own clips from ISO 640 to 12800 and carried past
  both ends the way a gain moves it (a with the ISO, the read noise ahead
  of the gain with its square), so the denoise, its temporal average and
  the sharpening all know how noisy an ISO 10000 night shot is
  (DESIGN.md 14.2).
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

`cineraw` decodes a clip or a still with FFmpeg, runs every frame through
the GPU pipeline and writes PNG (16 bit), TIFF (16 bit, uncompressed), PFM
(linear float) or PPM (8 bit, dithered, because a denoised shadow gradient
bands without it) sequences. A still - an RW2, or any input with one frame
- and a run of `--frames 1` may name one file instead of a sequence
pattern. Every frame is written upright, turned by the display matrix a
clip carries in its stream and a still in its Exif orientation, so a
portrait comes out 4000x6000. `--wb auto` reads the first frame and
develops it again at its own reading, so a still is balanced too; the
reading is taken when the estimator is confident of it and the camera's
own balance kept when it is not. The options that matter most:

    cineraw --wb auto -v --frames 30 clip.MOV        estimate the illuminant
    cineraw --wb 5600:0.02 --ei 400 -o out_%05d.png clip.MOV
    cineraw -v -o still.tif _1000406.RW2             a still, with its timings
    cineraw --bench --frames 60 clip.MOV             GPU time per pass, decode timing
    cineraw --denoise 1.0 --crop 2750,1450,1000,1000 -o c_%d.png clip.MOV
    cineraw --noise-estimate --frames 10 flatfield.MOV   fit the noise model

A Panasonic RW2 carries its levels and its as-shot white balance and
nothing else of a profile, so the rest comes from the camera's per-CCT
tables, picked by the model in the still's Exif: its colour temperature is
where the as-shot gains lie on the table's locus, its matrix the table's at
that temperature, and its gain the camera's video GainFactor for the ISO's
gain mode (23.04 from ISO 1000 up and 14.51 below, measured at 640 and
assumed below it) over eight, because the camera rates a still three stops
closer to the sensor clip than a V-Log recording. A still therefore comes
out as the camera's own JPEG does; a clip shot at the same ISO number and
exposure is three stops brighter than it, and `--exposure 3` on the still
puts them side by side. Without tables an RW2 has no matrix and is refused
(DESIGN.md 9c). The
decode runs slice threads; `-v` prints the profile it built and every
frame's decode, upload, develop, download and write time.

`--denoise STRENGTH[,CHROMA[,TEMPORAL]]` takes a strength in units of the
camera's modelled noise level, 1.0 meaning "remove exactly that much", an
optional chroma denoise amount from 0 to 1, the quarter resolution chroma
band replaced by its luma-guided smoothing, which takes the colour
blotches out of a dark scene and needs no noise model, and an optional
temporal amount from 0 to 1, how much of the previous frame a sample keeps
where the two agree within the noise (0.8 is about five frames of a static
shot; it needs the model, works with STRENGTH 0 too, and leaves a faint
trail behind things that move with less contrast than the noise;
DESIGN.md 14.7); `--noise A,B`
overrides the model and `--noise-estimate` measures one, printing a line
ready for a `--tables` file. `--false-colour PASSES` runs one to five passes of a
median and a blur over the colour differences after the debayer,
RawTherapee's false colour suppression steps, which turn the coloured
dots that rain glints and other sub-pixel highlights leave into neutral
ones and touch nothing else. Clipped highlights are rebuilt before the
debayer by default: every channel that clipped is estimated from the
channels that did not clip around it and from the colour of the unclipped
light around the blown area, and may exceed the old clip point, so a sky
where only green clipped keeps its clouds instead of turning flat grey
(DESIGN.md 18); `--highlights off` clips as before. `--defringe STRENGTH`,
0 to 1 and 1 by default, pulls the colour of every pixel along a strong
edge back into the range of colours on the two sides of that edge,
keeping its level: the purple, blue and pink rims the lens's chromatic
aberration leaves along bright edges, which the rebuilt highlights no
longer hide under a white clip, go, and a real colour edge is left alone
(DESIGN.md 19); `--defringe 0` turns it off. `--sharpen
RADIUS[,ITERATIONS[,CONTRAST]]` turns on capture sharpening, off by
default: Richardson-Lucy deconvolution of the luma by a Gaussian of RADIUS
pixels (0.8 matches this camera's in-focus edges), ITERATIONS of it (20 by
default, about 0.12 ms each at 5.8K), with every channel scaled by the
luma's change so that no colour moves, no pixel taken past the range of
its neighbours so that a lamp gets no dark ring, and nothing sharpened
where the local contrast is below CONTRAST times the camera's modelled
noise (1 by default; with `--denoise` on, 0.5 sharpens more texture and
none of the noise that is left), DESIGN.md 20. `--crop X,Y,W,H` writes one rectangle of the output while
the pipeline still runs on the whole frame, which is how a 5.8K frame gets
looked at 1:1 without a 5.8K file.

`--bench` prints the GPU and decode time of every frame and, at the
end, a table of every pass of the pipeline with its average and median
GPU time over the run, from `CRAW_PIPELINE_FLAG_PROFILE` and
`craw_pipeline_get_stage_times()`. The software decode leaves the GPU idle
between frames, so the frames run on a cold GPU and a player's back to
back frames are faster; the passes still compare with each other and from
run to run (DESIGN.md 21).

`--wb` takes `as-shot`, `auto` or `CCT[:TINT]`; `--ei` re-rates the clip
at a working exposure index, `--exposure` adds a bias in stops; `--encode`
picks the transfer function for image output, `sdr` by default for PNG,
which rolls highlights off per channel before the sRGB curve. The
Panasonic DC-S1M2 per-CCT tables are built in and selected from the
container's model name, or for a still from its Exif; `--tables` loads
others from a text file.

`cineview` is the interactive viewer and exporter built on SDL3, nuklear
and libplacebo, when those are found. It decodes with FFmpeg's Vulkan
hwaccel on libplacebo's device, runs libcraw on the same device, reads the
file on its own thread and loops by default (`--no-loop`). An RW2 still,
whose decoder has no hwaccel, is decoded on the CPU with slice threads and
uploaded to the device once, and from there it is developed like a ProRes
RAW frame. A still - an RW2, or any file of one frame - is not looped: it
comes up, stays up and is paused at the end of its stream, so a grade
change or the auto white balance develops it again, and a playlist that
was playing plays on at the next file. It has no rate row, and the Clip
block shows what its Exif says, its white balance being the as-shot
temperature on the camera's locus; a portrait still is turned upright by
its Exif orientation. Every decoder
it opens stays open until it quits, one per codec: a decoder is flushed
when its clip closes, a clip of a codec seen before takes it back from the
pool, only a codec not seen yet opens a new one, and a loop point is a
flush too, the way libavcodec takes one stream after another. The RW2
decoder is the exception: it reads a file's whole header from its own
setup, so it is taken back only for the same file, and another still
replaces it with a new one, which for a software decoder costs nothing. The panel is
worked with the pointer: its switches, expanders, sliders, fields and
cascading menus are clicked, and no key reaches a panel row. The keys are
the transport's and the view's. Space plays and pauses; Left and Right
seek five seconds back and forward, and Shift with them one second; Page
Up and Page Down zoom in and out, and Shift with them goes to the first and
last frame; `,` and `.` step one frame back and forward and pause the clip;
Control with the arrows pans; Tab or `0` fits the picture again
and `1` is one to one; the wheel zooms about the pointer and a left drag
pans; `r` and `t` go to the previous and the next file, and held, walk on
through them, a file per key repeat once the one before is on screen;
`f` goes through all frames, the odd ones and the even ones (below);
Shift+Alt+Delete marks the file on screen for deletion and unmarks it again;
Ctrl+1 and Ctrl+2
mark the start and the end of the video export at the frame on screen, the
whole clip being the default; `m` mutes; Escape closes an open menu, calls
off an armed "pick white", and otherwise quits, as `q` does. Zoom and pan
only touch the view, never the export. While a video export runs the whole panel and every key but `q` are
locked, the picture is the export's; `q` then stops the export, finishes
the file and quits.

The scrub bar goes where it is pressed: the frame under the pointer comes
up and stays for as long as the button is held, the pointer moving while
held takes it along, and when it is let go the clip plays on from there if
it was playing before.
Under the scrub bar the "rate" row slows the clip down, frame for frame.
Its stops are those of 23.976, 24, 25, 29.97, 30, 48, 50, 59.94 and 60
that are below the clip's own rate, and the clip's own at the right: a
59.94 clip at 23.976 shows every one of its frames two and a half times
slower, and the video export writes them as a 23.976 file two and a half
times as long. Nothing is dropped, repeated or made up, and nothing about
the picture changes. While the clip is slowed its sound is left out, as the
camera leaves it out of its own slow and quick motion; moving the row
neither seeks nor pauses, and back at the clip's own rate the sound picks
up at the picture on screen. Left and Right then step five seconds as
watched, and the frame readout above stays in the clip's own frames and
seconds. A 23.976 clip has nothing slower to go to and its row is greyed.
Like the scrub bar, the row goes where it is pressed and follows the
pointer while the button is held; there is no knob to catch first.
`--rate FPS` sets the same from the command line, one of those nine or
`clip`, for every clip of a playlist, and a rate above a clip's own is
refused when that clip opens: a headless run stops there, a window plays
the clip at its own rate (DESIGN.md 23).

At the right end of the rate row a checkbox with three states plays and
exports all of a clip's frames, or every other one: its odd-numbered frames
or its even-numbered ones, counted from frame 0 as the frame readout counts.
Each click moves it on, as `f` does: empty for all frames, the upper left
half of the box filled for the odd ones, the lower right half for the even
ones. That is half the clip's rate at its own speed, with its sound, and the
rate row's readout beside it says which, "14.99 fps, odd frames". It is for
light that pulses in step with half the frame rate, LED street lamps on a
mains-rate driver under a high-speed clip, which leave every second frame
darker than the ones either side; one half is a clip without that flicker,
and the checkbox picks which. The rate row keeps the stops the clip has
without it, with the half rate in its place among them, so the kept frames
can still be slowed - or shown faster than they were shot, which nothing
else does: a 29.97 clip on its odd frames plays at 14.99 and can be set to
23.98, 24 or 25, "25 fps, 1.67x faster", without sound like any other rate
than its own. The frames left out are still decoded, since a decoder needs them,
and dropped before they are developed. A seek lands on the shown frame at
or before where it was sent, picking the other half while a frame of it is
on screen goes back one frame, `,` and `.` step from one shown frame to the
next, and the temporal denoise averages only shown frames, after a seek as
well. The video export writes the shown frames of its range, numbered in
the file at the half rate: frames 0 to 59 of a 60 fps clip are 30 frames
lasting one second, sound and all, and file frame k is the clip's frame 2k
for the even half and 2k + 1 for the odd. A range that starts on a frame
left out starts at the next shown one. `--every-other-frame odd` or `even`
sets it from the command line for every clip of a playlist, and the choice
carries from clip to clip like the rate.

On Wayland the window follows the compositor's colour management
(`wp_color_manager_v1`): it asks for the image description the compositor
would like the window in, renders the picture into exactly that - the
primaries, the curve, the black level and peak, the display's own gamut -
and tags the window with it, so the compositor has nothing left to
convert, and does it again when the preference changes, when the window
goes to another output or HDR is switched on, without a restart. What the
compositor asks for is rendered as it is when it can be (sway's SDR outputs
want sRGB primaries in gamma 2.2, which is what the window then is), an HDR
preference that cannot be falls back to BT.2020 PQ, and anything else to
the sRGB display in gamma 2.2; `-v` prints what was offered, what was
preferred and what was made of it (DESIGN.md 22). The panel's colours are
the same either way. `--no-colour-management` turns this off and leaves the
window untagged sRGB, which is what it is on X11 (XWayland included) and on
a compositor without colour management: the compositor then decides what
the picture's pixels mean, and on sway that is gamma 2.2.

The panel is set in whatever face fontconfig names for `sans-serif` on the
machine, at 15 px, the way every other desktop application finds its UI
font: there is no option and nothing to configure, and nuklear's own
bitmap face is only what remains when fontconfig has nothing or the file
will not load. Under the playback readouts sits a "Clip" block with what
the file says about the recording, one row per fact and only the rows the
file has: lens (with the first frame's focal length), format (the
codec with its bit depth and coded size, "ProRes RAW 16-bit 5760x3040"),
frame rate, the exposure on one row ("ISO 100  f/2.8  1/15"), the white
balance as recorded (with
the camera's mode, "auto, 3100 K", unless it was set in Kelvin), the
exposure index when it differs from the ISO, the capture gamma and gamut
of a developed clip, the timecode at the clip's start and the recording
date - the camera's own, with its zone, when the clip carries the
Panasonic XML, else the container's creation time in the machine's zone.
Most of it comes through FFmpeg's demuxer: the Apple and RDD 18 tags a
ProRes RAW clip carries, the stream's own tags and the XML. The rest the
viewer reads out of the file itself, because the demuxer skips both
places it is in: the Exif block and Panasonic maker note in
`moov/udta/PANA`, which every clip from the camera carries and which is a
developed clip's only record of its lens, shutter and white balance, and
the copy of the XML an MP4 keeps in a top-level `psmd` meta box rather
than as a tag. `-v` prints the same rows once per clip, and the maker
note's white balance multipliers with them.

The panel is a column of sections — White balance, Exposure, Debayer,
Denoise, Lens, Sharpen, Look, Adjustments, Film grain, Crop, Export image, Export video — each behind a
header with a power switch at the left of its title, the way RawTherapee
puts an enable button on every tool. Off bypasses that section while every
setting in it is kept, so switching it back on returns the picture it had:
Exposure off is a bias of zero at the as-shot index with the slider still
where it was, Look off drops the film simulation or the `--lut` cube
without unselecting it, Adjustments off drops soft light and vibrance, Film
grain, Denoise, Lens and Sharpen off turn their stages off, and Crop is the same
switch `c` is. Debayer has nothing to bypass, and neither has a raw clip's
White balance, which is always applied; they carry an empty slot so the
titles stay in line. A developed clip's White balance does have a switch,
and off shows the picture balanced as the camera balanced it. A lit switch
always means the section is doing something:
Exposure, Look, Adjustments and a developed clip's White balance start off
and switch themselves on the moment one of their parameters is touched — a
slider moved, a film simulation picked, a white balance mode chosen or
picked, `--exposure`, `--ei`, `--lut`, `--softlight`, `--vibrance` or a
`--wb` other than `as-shot` given — the way RawTherapee's tools do. `--denoise 0`, `--lens
off` and the absence of `--grain` start those three off, and `--bypass
exposure,look,adjustments` pins the other three off whatever the other
options set, which is how a headless export is handed the state the panel
is in.

Auto white balance follows the smoother while the clip plays: the track
starts and moves only on readings the estimator is confident of, and slews
between them, so a run never jumps after a reading that might be wrong. A
paused picture is shown its own frame's estimate whatever the confidence
— the one looking at it is the gate, and the section shows the estimate,
its confidence and the track — and a seek while paused re-estimates for
the frame it lands on.

Auto starts where the camera was - its as-shot balance - and the clip's
first reading is taken whole when the estimator is confident of it and
left alone when it is not (the panel prints the confidence beside the
result). From there on it never jumps: the balance glides to each new
confident reading through the same smoothing whether the clip is playing
or paused, so a paused picture settles to its own frame's reading and a
seek settles to the new frame's, and readings the estimator is unsure of
hold it. A scene it is unsure of throughout, a filament on black, stays at
the camera's balance, which is what RawTherapee does with it too.

The White balance section's "pick white" button, a row of its Manual mode,
is the spot white balance
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
comes out neutral to a few hundredths of a per cent, and picked again,
which returns the same illuminant.

A developed clip - HEVC, H.264, ProRes 422 - has the same section, the same
three modes and the same pick, applied to a picture the camera has already
balanced. The camera adapted the scene's white to the space's; which white
it adapted from is the colour temperature the clip records - the maker
note's, which on an auto clip is the one the camera settled on, or the
XML's in Kelvin mode - taken as a point on the Planckian locus
(`craw_planckian_xy()`). Under an illuminant of T Kelvin the camera
therefore renders a neutral as the Planckian white of T adapted from the
recorded white, and sampled from 2000 to 12000 K that is the clip's own
white balance locus in the linear RGB of its primaries, handed to libcraw
as a profile's white balance table: a temperature along it and a tint
across it mean what they mean on a raw clip, in the same units, and the
pick and the auto estimator run on it unchanged. Rebalancing to an
illuminant is a Bradford adaptation (`craw_bradford_adaptation()`) from
the colour the camera gave that illuminant's neutral to the space's white,
one matrix at libplacebo's LINEAR stage together with the exposure; the
recorded temperature is a knot of the locus and comes back as no
correction at all, which is where Manual starts - picking it, on either
kind of clip, starts from the balance on screen. The pick and the auto
white balance read the picture back as the LINEAR stage holds it, before
the correction and with an HLG signal's black level resolved against the
window as the window's own render resolves it; the auto reading is a
960x540 shrink collected a frame later, and while paused the same picture
is read again until the track settles. A clip that records no temperature
is taken as balanced for its space's white, so the numbers are relative
to D65. A picture style's tone curve and saturation are not undone,
because nothing records them: on anything but V-Log this is the correction
a grading tool makes on a developed picture, not a development.

The "Debayer" section picks the method and carries the false colour
suppression row: the number of median and blur passes over the colour
differences after the debayer, zero to five, which is what stops a wet street from
shimmering in colour as the rain hits it; `--false-colour PASSES` sets it
from the command line. Its "rebuild clipped highlights" switch, on by
default, is the highlight reconstruction; `--highlights on|off` sets it.
The defringe row under it, 1 by default, is the defringe strength;
`--defringe STRENGTH` sets it. The "Denoise" section sets the raw stage's strength and reports the
noise model it found for the clip's own ISO — that stage stays off when
the camera has none, which the section says rather than leaving you
guessing — the temporal amount in front of it, 0 to 0.95 and 0 by
default, which needs the model too, and the chroma amount, the luma-guided
smoothing of the quarter resolution chroma band after the debayer, which
needs no model; a strength of 0 leaves the spatial filter out, so the
other two can be looked at alone. `--denoise STRENGTH[,CHROMA[,TEMPORAL]]`
sets the switch and all three amounts from the command line. The temporal
average starts again on a new clip, at the loop point and after a seek,
but not at the frame the seek lands on: the five frames before it are
developed first, unseen, so that the picture is about as clean as if the
clip had been played to it, at about a tenth of a second a seek. A video
export does the same before its first frame, and so does a still of a
frame that is not on screen; while the scrub bar is held its seeks skip
it, and the frame it is let go on is developed again with it. A
grade changed while paused develops the frame on screen from the same
history it was first developed from. The "Sharpen" section is the capture
sharpening, off until its switch is lit: the radius, the iteration count
and the contrast threshold of `cineraw --sharpen`, 0.8, 20 and 1 until
changed. The count is what a paused picture and every exported frame get;
while the clip plays the pass runs at most five iterations, which is what
fits the frame time, and pausing puts the picture through again at the
full count. On a camera with no noise model the contrast row greys out and
everything is sharpened. `--sharpen RADIUS[,ITERATIONS[,CONTRAST]]` sets
the switch and the numbers from the command line.
With four mono audio tracks the last two are taken as the stereo pair
(`--audio-map` overrides). `cineview -h` lists the options.

It takes any number of files, which makes a playlist: `r` and `t` step to
the previous and the next clip and wrap at both ends, and the panel's clip
line reads `3/7 P1000347.MOV [hevc video, gpu]`. The clips need not match —
ProRes RAW and already developed, different sizes, stereo or four-channel or
no audio at all — and a switch keeps every setting exactly as it was: the
white balance mode and a manual CCT, the exposure index and bias, debayer
and its false colour passes, highlight switch and defringe, denoise, lens correction
and its fit, the sharpening, the
export's colour space, size,
pixel format, audio mode and threads, the two export file paths once typed,
the film simulation, the tone map, the adjustments, the grain, every
section's switch, the zoom and the rate, which each clip clamps to its own
without forgetting it: a 23.976 clip among 59.94 ones plays at its own
rate, and the next 59.94 one is slowed again. A setting a clip's kind
cannot use is kept rather than dropped, so the debayer method and the lens
correction a raw clip was graded with are still there after a developed
one. Only what is read from the
clip is read again: its tables and as-shot values, its crop, its lens
records, its metadata. A file that cannot be opened is reported and leaves
the viewer on the clip it is showing, and `--export` and `--export-video`
take the first file only.

Going through a card's worth of files is also where the bad ones get
thrown out. Shift+Alt+Delete marks the file on screen for deletion, and
the same keys unmark it; a still is marked with the camera's JPEG of the
same name beside it, `_1000406.JPG` with `_1000406.RW2`, when there is
one. Nothing is deleted on a mark. A marked file is shown with a red frame
round the picture. While anything is marked the bottom of
the panel has a "delete N marked" button, and beside it whether the file
on screen is one of them ("this one", or "this one, and its JPEG"). The
button deletes every marked file, permanently rather than to a trash, and
takes them out of the playlist; the clip on screen stays if it was not
marked, and otherwise the next file left comes up. A file that will not go
stays marked and the reason is said under the button. Deleting every file
of the playlist ends the run, there being nothing left to show.

It also carries 296 film simulation LUTs from
[filmgrid.org](https://filmgrid.org/), which derive from the open-source
G'MIC film emulation collection — see `tools/gui/luts/LICENSE.md` for the
provenance and the attribution they carry. The panel's "film simulation"
row opens a cascading menu over the same two category trees the site uses,
Films (colour negative, slide, black and white, instant, cinema print,
Fujifilm simulations) and Looks (warm, cool, vintage, cinematic and so on),
by a click on the row; a click elsewhere, a choice or Escape closes it.
The rendering the highlight rests on is previewed on the picture while it
does, with the menu open and wherever the pointer goes from there, so the
collection can be walked rather than picked through; a film previews as
its base a moment after the pointer stops on it, before its column opens,
so the films column can be walked too. A click keeps what is previewed and
closing the menu any other way puts the chosen one back. Exports never see
a preview.
`--lut` takes a rendering's name or slug as well as a file path, and
`--list-luts` prints both trees. The `.cube` files live in the tree; the
compact pack that ends up in the executable is built by
`tools/luts_build.py` at build time.

The Look section also holds the "tone map" row: the curve the window is
tone mapped with, out of libplacebo's own list — st2094-10, the default;
st2094-40; spline; bt2390; bt2446a; reinhard; mobius; hable; gamma;
linear; linearlight; clip — with `--tone-map NAME` as the starting value.
The default is ST 2094-10 because its curve runs through the black, the
knee and the peak and nothing else, so highlights keep their separation
up to the display's white where ST 2094-40's Bézier flattens everything
above about a thousand nits into one white and the spline sits darker
through the mid-tones.
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
rotated clip is the readout's rectangle turned back. A clip that marks its
picture inside the readout (a ProRes RAW frame header can, an RW2 does:
6000x4000 inside 6008x4008) starts with that as the rectangle, and that
rectangle is the picture on screen rather than a shaded one over the
readout, so there is no border round it; the Crop section reports it as
the clip's. `c` clears it like any other, which is how the sensels around
the picture are looked at.

Two sections hold two exports, **Export image** and **Export video**,
independent of each other: the image is the frame on screen, one file, and
the video is a range, whose start and end fields are bounded by the clip's
last frame — Ctrl+1, Ctrl+2 and `--export-video` hold to it the same way,
so a range cannot reach past the clip. Each section has its own file
field, its own button and its own status line, neither field names the
other's file, and a field keeps what was typed into it while its section
is folded. Each field follows its own default name — the image's the clip
and the frame as a `.jpg`, the video's the clip and the range in the
family's container, both in `~/Processed`, which is made on first use (the
clip's own directory when there is no home to put it under) — except for
what is typed into it, and the two parts of a name are kept differently.
An extension typed is the format from then on, on this clip and every one
after. A name typed before it is kept on the clip it was typed on and on
no other: the next clip gets its own name again, under that extension, so
a name typed for one clip is never the next one's and exporting the next
never writes over the first. A typed path always wins while it stands.
Nothing judges a name while it is being typed: the extension picks the
container when the button is pressed, a name with no extension libavformat
knows gets the family's container appended (`.jpg` for the image) in the
field itself, and a container that will not carry the encoder or the sound
is refused on the status line under the button, in a sentence.

`--export FRAME OUT` is the image export from the command line: it writes one frame
to one file and exits without opening a window, upright and at the crop
rectangle's own size. The file name picks the format through libavformat,
which knows both the muxer and the codec an extension implies: `.png` and
`.tif` take the sixteen bit RGB the frame is rendered as, `.jpg`, `.webp`
and `.avif` whatever their encoder supports, deepest first, through
libswscale where that is not RGB; `--codec` overrides the encoder and `--enc
KEY=VALUE,...` passes it options (`cpu-used=8,threads=2` for an AVIF). PNG is
written at zlib level 1 (a fifth of the time of the default level for a per
cent of the size), TIFF uncompressed, and JPEG at 4:4:4 with the encoder's
quantiser fixed at 2 (`--enc global_quality=N` sets another, in the
encoder's lambda units, 118 a step). The stream is tagged as what the
export really is: full range, BT.709 primaries, sRGB transfer, display
referred. A sequence of stills is deliberately not offered — that is a video
encode. `--export-video START END OUT` is the video export headless, and
honours `--rate`: the range is written frame for frame at that rate, with no
sound. The image export has no codec box: the extension is the choice.

Every video export into an `.mp4` or `.mov` is written with `movflags
+faststart`, always: the `moov` atom is moved ahead of the media so the file
plays before it has finished downloading. There is no option to turn it off.

The **Export video** section also carries the sound (see below), the number of
encoder **threads** — 0 for FFmpeg's own auto, 1 to the machine's core
count, applied to the software encoders and greyed for a Vulkan one, which
has none; `--threads N` sets it and `-v` prints the count the encoder
settled on since some clamp it — and, while a run is going, an *exporting: N
of M frames* line, a progress bar and a stop button. From the moment the
export starts until it ends or is stopped, every panel section and every row
but the stop button is disabled and the Export video section held open:
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
on the peak — bt2390 above all, and st2094-10, the default, which also
follows the frame's average — can come out a percent or two brighter in
the file than on a fitted window; spline is all but insensitive to it.

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

A slowed clip — the rate row below the scrub bar, or `--rate` — is written
with no sound stream at all, whatever the mode says. The rows are greyed and
say so, `no sound while slowed`; the mode itself is kept, and a container
that could not carry it is no reason to refuse a run that writes no sound.
At the clip's own rate it is there again. The file runs at the slowed rate,
its frames numbered in it: 150 frames of a 59.94 clip at 23.976 are a
6.25625 s file, where the clip's own rate makes them 2.5025 s. The status
line and `-v` say `no sound at 23.98 fps` and `slowed from 60000/1001`.

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
from the format it is given. Every x264 preset starts on `4:2:0 8-bit
(nv12)`, which is High and plays everywhere, where High 10 plays on almost
no hardware; an H.264 encoder named with `--codec` - `h264_vulkan`, say -
starts on NV12 too, since only `--pix-fmt` pins a format for an encoder that
is not the family's own. APV takes its profile from the format too, so
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
