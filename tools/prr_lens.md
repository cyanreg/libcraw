# Panasonic's per-frame lens record in ProRes RAW

How `com.panasonic.Semi-Pro.optical_correction` is framed and what its
payload means, derived from the bytes of real S1II clips, from Apple's own
ProRes RAW decoder, and cross-checked against Panasonic's in-camera
distortion block in RW2 stills.

`tools/prr_lens.py` is the probe that produced every number below.

    prr_lens.py CLIP [--frames N] [--ss T] [--hexdump]
    prr_lens.py CLIP --rw2 STILL.RW2

It demuxes with `ffmpeg -i CLIP -map 0:v -c copy -f data -`, splits frames
on the be32 size plus `prrf` tag, walks the frame header and tile table the
way `libavcodec/prores_raw.c` does, and reads the records that follow the
last tile.  It never decodes pixels.


## 1. Where the record sits

`prores_raw.c` gives the frame walk: be32 size, `prrf`, a be16 header
length, the header body, then `nb_tiles` be16 tile sizes, then the tile
data.  `nb_tiles` follows from the width, the height and the alignment
field of the header flags:

    tw16    = (w + 15) >> 4
    nb_tw   = (tw16 >> align) + popcount(tw16 & ((1 << align) - 1))
    nb_th   = (h + 15) >> 4

Everything after the last tile is vendor metadata.  For P1000387.MOV frame
0 that puts the trailer at 0x4e55b8 of a 0x4e56a0 byte frame, 232 bytes,
and the record is the whole of it bar the terminator.  `prr_lens.py`
derives that offset rather than searching for the key string, and the two
agree on every frame tried.


## 2. Record framing

Raw bytes of the record, P1000387.MOV frame 0:

    004e55b8: 0000 00d6 7073 696d 2963 6f6d 2e70 616e  ....psim)com.pan
    004e55c8: 6173 6f6e 6963 2e53 656d 692d 5072 6f2e  asonic.Semi-Pro.
    004e55d8: 6f70 7469 6361 6c5f 636f 7272 6563 7469  optical_correcti
    004e55e8: 6f6e 0000 0003 3fef 24e9 8132 2e7d 3f8c  on....?.$..2.}?.
    004e55f8: 50e7 43a2 b41c 3f90 5f72 5e21 d558 bf76  P.C...?._r^!.X.v
    004e5608: 54bc e211 b5a1 0000 0000 0000 0000 0000  T...............
    004e5618: 0000 0000 0000 3fef 2886 ddf2 41be 3f8c  ......?.(...A.?.
    004e5628: 0db6 3ff9 2e5d 3f90 e0e5 44ae 8b54 bf78  ..?..]?...D..T.x
    004e5638: 6bc1 2ea5 fcc0 0000 0000 0000 0000 0000  k...............
    004e5648: 0000 0000 0000 3fef 2d8a 6bdd 94a7 3f8b  ......?.-.k...?.
    004e5658: 5e89 7695 fe05 3f91 84bc 6fda d636 bf7b  ^.v...?...o..6.{
    004e5668: 2bf6 23ac a363 0000 0000 0000 0000 0000  +.#..c..........
    004e5678: 0000 0000 0000 3fe0 0000 0000 0000 3fe0  ......?.......?.
    004e5688: 0000 0000 0000 0000 0008 656f 6d64       ......    eomd

So:

    be32  size        0x000000d6 = 214, counting the size field and the 4cc
    4cc   'psim'
    u8    key length  0x29 = 41
    ...   key         "com.panasonic.Semi-Pro.optical_correction", no NUL
    ...   payload     size - 9 - keylen = 164 bytes

    be32  size        8
    4cc   'eomd'      end of metadata
    ...   zero padding to the frame's alignment, inside the frame size

214 = 4 + 4 + 1 + 41 + 4 + 20*8 exactly, which is what fixes the payload as
a 4-byte prefix plus twenty 8-byte quantities with nothing left over.

Apple's own decoder confirms the framing independently.  In
`ProResRAW.dll` (Ghidra, image base 0x180000000), `FUN_180006ad0` walks two
parallel arrays built by the frame parser, a 4cc array and a payload
pointer array.  Its loop terminates on `0x656f6d64` (`eomd`) and bodies
only on `0x7073696d` (`psim`).  Inside, `bVar1 = *pbVar3` is the pascal
length byte, the key comparison is
`strncmp((char *)(pbVar3 + 1), "com.apple.gdc", 0xd)`, and the size check
is `iVar2 - (uint)bVar1 == 0x56`, i.e. size minus key length is a constant.
For `com.apple.gdc` that constant is 0x56 = 86 = 8 framing bytes + 1 length
byte + 77 payload bytes, so `size` includes the be32 and the 4cc and the
pointer handed to the record parser is the length byte.  Same framing as
the Panasonic record, one record type later.

Apple's decoder does **not** parse Panasonic's record.  `ProResRAW.dll`
contains no string matching `optical_correction`, `Semi-Pro`, `panasonic`
(as a record key), `distortion` or `optical`; the only geometric record it
knows is `com.apple.gdc` at 0x18007c080, referenced once, from 0x180006bc5
inside `FUN_180006ad0`.  So the Panasonic record is carried through the
file untouched, and there is no Apple-side code to read the layout off.

For contrast, Apple's own model, from the OpenCL kernel source embedded at
0x18070a5d0, is a centre `cx, cy`, a radius scale `oneOverM`, fifteen
radial coefficients, `minRadius`/`maxRadius` clamps and a
`reciprocalScaling` flag, applied as
`pp = c + (reciprocalScaling ? dp / f : f * dp)` with `f` the polynomial in
the **destination** radius.  Same family as Panasonic's: a backward warp
whose polynomial is evaluated at the output radius.


## 3. Payload layout

    be32      version     0x00000003 in every frame seen
    20 x be64 IEEE doubles

Decoding the twenty doubles of the frame above:

    R  0.9732558750   1.38262e-02   1.59891e-02  -5.45191e-03   0.0   0.0
    G  0.9736971221   1.36980e-02   1.64829e-02  -5.96214e-03   0.0   0.0
    B  0.9743091685   1.33639e-02   1.71079e-02  -6.63372e-03   0.0   0.0
    centre  0.5   0.5

Three blocks of six, then two.  The split is forced by the values, not
assumed:

* Doubles 18 and 19 are exactly 0.5 in every frame of every clip tried,
  bit-for-bit `3fe0000000000000`.  Nothing else in the record is an exact
  power of two.  They are the optical centre in normalised frame
  coordinates, and the camera always reports the geometric centre.
  A "centre first" layout is ruled out: the exact halves are at the end.
* Doubles 4, 5, 10, 11, 16 and 17 are exactly +0.0 in every frame of every
  clip tried, across 24 to 135 mm and across barrel and pincushion.  A
  higher-order k4, k5 that is identically zero over a 5.6x zoom range and
  two different lenses is not a plausible reading; they are pad.
* Within each block of six the first entry is near 1 and the rest are
  small, so the first is the constant term of a polynomial in even powers
  and the layout is ascending: k0, k1, k2, k3.
* The three blocks differ from each other only in the fourth decimal, which
  is the signature of three channels of one lens, not three unrelated
  quantities.  Evaluated at the corner the three differ by about a pixel
  and a half, which is lateral chromatic aberration of the right size for
  a full-frame lens.

Order is R, G, B.  Nothing in the record labels the channels, but R and B
straddle G at the corner in every frame, which is what a real lens does and
what neither of the other two orderings would give.


## 4. The model

    r_src = r_dst * (k0 + k1 r_dst^2 + k2 r_dst^4 + k3 r_dst^6)

per channel, about the centre, destination to source: for each pixel of the
corrected output at normalised radius `r_dst`, sample the source at
`r_src`.  `r` is normalised by the **half diagonal of the coded frame**, so
r = 1 is the frame corner, 3256.5 px for 5760x3040.

Three independent lines of evidence fix the normalisation.

**The record's own design.** Across every pincushion frame measured, the
polynomial at r = 1 lands just below 1.0 and never above:

    clip          focal    k0          f(1)       margin
    P1001272      36 mm    0.99993866  0.9966713  0.333%
    P1001124      38 mm    0.99607944  0.9986212  0.138%
    P1001209      43 mm    0.98796143  0.9980408  0.196%
    P1001125      47 mm    0.98323852  0.9979057  0.209%
    P1000387      60 mm    0.97369712  0.9979149  0.209%
    P1000319      60 mm    0.96800243  0.9973837  0.262%
    P1093969     135 mm    0.96722345  0.9998772  0.012%

f(1) ≈ 1 means the corner of the corrected frame samples the corner of the
source, to within a fraction of a percent.  That is the camera spending
exactly as much crop as the correction needs and no more, and it is only a
meaningful statement if r = 1 *is* the corner.  If r = 1 meant the half
width instead, the 60 mm frame would want a source radius 0.6% outside the
sensor corner, about 20 px of black in each corner.

**k0 is that crop.** At 24 mm, where the lens is barrel and correction
samples inward anyway, no crop is needed and k0 = 0.9991, essentially 1.
As the lens goes pincushion k0 falls to 0.967, a 3.3% crop.  k0 tracks the
requirement, not the distortion.

**The RW2 cross-check**, section 6, which puts the best-fit normalisation
radius within 1.3% of the video half diagonal and rules out the
alternatives by 3x to 100x.

Two derived numbers are worth separating.  `poly(1) - 1` times the half
diagonal is the *net* corner shift, a few pixels, mostly cancelled by the
crop.  `poly(1)/poly(0) - 1` times the half diagonal is the distortion the
record actually corrects, up to 210 px.  `prr_lens.py` prints both as
"net px" and "dist px".


## 5. What the coefficients do

One frame per focal length, LUMIX S 24-60 f/2.8 except the last:

    clip        focal    ch  k0            k1            k2            k3            net px  dist px
    P1000261     24 mm   R    0.99965795  -1.19016e-01   6.71727e-02  -1.21451e-02  -209.49  -208.45
                         G    0.99908949  -1.18440e-01   6.67320e-02  -1.19222e-02  -210.18  -207.40
                         B    0.99902956  -1.17697e-01   6.61432e-02  -1.16425e-02  -208.96  -206.00
    P1001272     36 mm   R    1.00005202  -2.44132e-02   2.68090e-02  -5.66174e-03   -10.47   -10.63
                         G    0.99993866  -2.43367e-02   2.68724e-02  -5.80187e-03   -10.84   -10.64
                         B    1.00027248  -2.44520e-02   2.70659e-02  -6.10293e-03   -10.47   -11.36
    P1001124     38 mm   R    0.99613963  -1.47150e-02   2.22941e-02  -5.00054e-03    -4.17     8.43
                         G    0.99607944  -1.46753e-02   2.23697e-02  -5.15375e-03    -4.49     8.31
                         B    0.99642897  -1.48737e-02   2.26392e-02  -5.51378e-03    -4.30     7.36
    P1001209     43 mm   R    0.98792869  -7.52288e-04   1.44991e-02  -3.56335e-03    -6.15    33.57
                         G    0.98796143  -7.88030e-04   1.46263e-02  -3.75939e-03    -6.38    33.22
                         B    0.98833029  -1.12054e-03   1.49981e-02  -4.18586e-03    -6.44    31.93
    P1001125     47 mm   R    0.98315256   6.92747e-03   1.03298e-02  -2.46930e-03    -6.71    48.98
                         G    0.98323852   6.88582e-03   1.04559e-02  -2.67512e-03    -6.82    48.58
                         B    0.98359239   6.55917e-03   1.08006e-02  -3.10382e-03    -7.01    47.20
    P1000387     60 mm   R    0.97325587   1.38262e-02   1.59891e-02  -5.45191e-03    -7.75    81.52
                         G    0.97369712   1.36980e-02   1.64829e-02  -5.96214e-03    -6.79    81.00
                         B    0.97430917   1.33639e-02   1.71079e-02  -6.63372e-03    -6.03    79.68
    P1030266     60 mm   R    0.96737251   2.64504e-02   3.88586e-03  -4.72019e-04    -9.00   100.53
                         G    0.96764802   2.64300e-02   4.13925e-03  -8.66953e-04    -8.63    99.96
                         B    0.96792927   2.61398e-02   4.53701e-03  -1.45556e-03    -9.28    98.31
    P1093969    135 mm   R    0.96727822   2.60283e-02   2.83489e-03   3.68910e-03    -0.55   109.59
                         G    0.96722345   2.63190e-02   2.67723e-03   3.65618e-03    -0.40   109.94
                         B    0.96696760   2.68170e-02   2.39848e-03   3.52871e-03    -0.94   110.27

    P1093969 is the Sigma 135mm F1.4 DG Art; the rest are the LUMIX S 24-60.
    Centre is exactly (0.5, 0.5) and slots 4, 5 are exactly zero in all of them.

Barrel at the wide end, -207 px at the corner, through neutral at about
37 mm, to pincushion at the long end, +100 px.  Sign, magnitude and
crossover are what a 24-60 f/2.8 does.

Lateral chromatic aberration, R minus B at the corner, runs from +0.01 px
to -1.72 px across these frames: small, correctly signed, and of the size a
full-frame lens shows.

The 135 mm Sigma is neither absent nor neutral.  It carries a full record,
a 3.4% pincushion correction, because the camera has a profile for the
lens over the L-mount protocol.  The record is present on every ProRes RAW
frame of every clip tried, including two clips whose container metadata
carries no lens name at all.

**Zoom.** Three clips zoom while recording.  P1000253, frame by frame from
t = 9 s, the corner distortion in pixels:

    -207.1 ... -207.5  -202.4  -167.0  -126.3   -93.8   -66.5   -34.2
     -12.4    -6.3 ... -6.2     +8.2   +24.4   +37.0   +50.7   +63.3
     +73.2   +77.5 ... +77.9   +82.8   +91.1   +97.7  +102.2 ... +101.3

A continuous sweep from the wide end to the long end with plateaus where
the ring stopped.  The coefficients follow the zoom frame by frame.

**Focus.** At fixed focal length the coefficients still move, in a
correlated one-parameter family: over P1001497 at 60 mm, k0 ranges 0.96682
to 0.96793 while k1 falls 0.028171 to 0.025835.  Within a contiguous run
the values snap between a small set of table entries rather than varying
continuously, so the camera is sampling a stored table on a quantised
index.

**The zoom position itself is not recorded anywhere per frame.**  The frame
header has no focal length field; the only trailer records are
`optical_correction` and `eomd`; the third track in the container is
`tmcd`, timecode.  The container's Exif focal length is written once, from
the start of the clip, so it is wrong for the rest of a clip that zooms.
The polynomial is the only per-frame record of the lens state, and since it
moves monotonically with zoom over most of the range it can be inverted for
an approximate focal length, though not uniquely, because focus moves it
too.


## 6. Panasonic's own model in RW2 stills, and what RawTherapee does

RawTherapee's Panasonic path is `PanasonicMetadataLensCorrection` in
`rtengine/lensmetadata.cc` (the class at lines 613-700), with the radius
machinery in its base `CenterRadiusMetadataLensCorrection` in the same file
(lines 59-103) and the warp loop in `rtengine/iptransform.cc`
(`transformGeneral`, lines 1070-1360).  Described in prose, nothing quoted:

* It reads one tag, IFD0 **0x0119** of the RW2, Panasonic's `DistortionInfo`
  block, as sixteen signed little-endian 16-bit words.  Words 0, 1, 14 and
  15 are checksums; exiftool's `PanasonicRaw.pm` verifies them.
* Four words are used.  Word 5 becomes a scale as the reciprocal of one
  plus the word over 32768.  Words 8, 4 and 11 are each divided by 32768
  and become the coefficients of r^2, r^4 and r^6 respectively, in that
  deliberately scrambled order.  All four are signed; the divisor is 2^15
  in RawTherapee and in exiftool 13.55 alike.  exiftool names them
  `DistortionScale`, `DistortionParam08`, `DistortionParam04` and
  `DistortionParam11`.
* Word 7's low nibble is exiftool's `DistortionCorrection` on/off flag.
  RawTherapee never reads it; it treats the presence of the tag as the
  enable.
* Word 12 is 3606 in every S1II still.  The still's active area is
  6000x4000, whose half diagonal is 3605.55 px.  So the block states its
  own normalisation radius, and it is the half diagonal.
* The radius comes from the base class as the distance from the geometric
  image centre divided by the half diagonal of the output image, so r = 1
  at the corner.  No optical centre is read from the makernote; the centre
  is the image centre.
* The applied factor is one over the product of the scale and the
  polynomial, so it is a reciprocal of a cubic in r^2, not a cubic.
* It is applied **destination to source**: the transform loop iterates over
  output pixels, hands each output radius to the correction, and gathers
  from the source at the returned radius.  Same direction as the ProRes RAW
  record.
* It is a **single geometric warp for all three channels**.  The class
  reports no CA correction and no vignetting; the comments say the CA block
  has not been reverse engineered and that Panasonic applies shading
  correction in camera.
* The scale is a radius-independent multiplier sitting outside the
  polynomial, the vendor's own crop term.  RawTherapee's separate
  auto-fill, which bisects a zoom until the corners stop clipping, is a
  different mechanism layered on top.
* No clamping anywhere: no bound on the factor, no range check on the
  coefficients, and a 0/0 at exactly r = 0.

One caveat, and it matters for the comparison: **the whole path is dead
code**.  The constructor throws unconditionally before parsing, the factory
catches that and returns null, and the GUI greys the option out.  So a
Panasonic RW2 gets no metadata lens correction in RawTherapee today.  What
follows is what the code would compute, not what a user sees.

Written out, with `w5`, `a`, `b`, `c` the scaled words:

    r_src = r_dst * (1 + w5/32768) / (1 + a r_dst^2 + b r_dst^4 + c r_dst^6)

versus the ProRes RAW record's

    r_src = r_dst * (k0 + k1 r_dst^2 + k2 r_dst^4 + k3 r_dst^6)

Same direction, same centre convention, same half-diagonal normalisation,
same role for the leading scale.  The functional families differ: ours is a
polynomial, Panasonic's still block is the reciprocal of one.


## 7. The numbers, same lens, same focal length

`prr_lens.py CLIP --rw2 STILL.RW2` evaluates both over r in 0..1 on the
video frame's radius axis, mapping the still's radius by 3256.5 / 3606.

Stills are on the LUMIX card, `DCIM/109_PANA`; all are f/2.8, focus
distance not reported by either side, so focal length is the only thing
matched.

**24 mm.** P1000261.MOV frame 0 against `_1093926.RW2`.
Still block: scale 1.001008, a +0.126434, b -0.045258, c -0.008179,
correction on, norm 3606 px.

    centre scale        record 1/k0 1.000911    still 1.001008
    corner, crops out   record 0.936312         still 0.931968
    max |r_src difference| over r in 0..1, each with its own crop:  5.33 px
    max |r_src difference| with both crops divided out:             5.55 px

On a 207 px correction, 5.5 px is 2.7% of the effect and 0.17% of the
radius.  The two curves track point by point across the whole range.

**60 mm.** P1000387.MOV frame 0 against `_1094195.RW2`.
Still block: scale 1.034278, a -0.031158, b +0.000122, c +0.000366.

    centre scale        record 1/k0 1.027013    still 1.034278
    crop needed         record 1.024873         still 1.031641
    margin left         record 0.208%           still 0.255%
    max |r_src difference|, own crops:                             19.97 px
    max |r_src difference|, crops divided out:                      5.52 px

The centre scales differ by 0.7 pp because the two frames have different
diagonals: the still needs more crop than the 17:9 video crop of the same
sensor, and both leave about a quarter of a percent of margin.  The shapes
agree to 5.5 px.

**135 mm**, Sigma Art, P1093969.MOV against `_1093871.RW2`: shapes agree to
0.01 px out to r = 0.5, diverging to 12 px at the corner; both leave about
0.01% of margin, the crop being the exact minimum.

**Which radius r = 1 means.**  Refitting the 24 mm pair with the
normalisation ratio free, the best fit is 3300 px, 1.3% from the video half
diagonal.  Holding it at each candidate:

    video half-diagonal   3256.5 px    max error    5.56 px
    still half-diagonal   3606.0 px    max error   16.10 px
    video half-width      2880.0 px    max error   26.66 px
    video half-height     1520.0 px    max error  140.45 px
    video full width      5760.0 px    max error  601.22 px

**Which direction.**  Using the still's polynomial directly as a
destination-to-source multiplier, the way the ProRes RAW record is used,
gives the wrong sign of correction and a 163 px error at 60 mm.  Only
RawTherapee's reciprocal reading matches.  So the two records are in
opposite senses: the ProRes RAW polynomial multiplies, the RW2 polynomial
divides.  Treating the RW2 form instead as a forward map to be inverted
exactly fits worse than RawTherapee's one-step reciprocal (13.5 px against
5.6 px at 24 mm), so the reciprocal is how Panasonic means it, not an
approximation RawTherapee is getting away with.

What this establishes: the record's layout, its direction, its
normalisation and the meaning of k0, all confirmed against an independent
vendor statement of the same lens at the same focal length.  What it does
not: the residual few pixels cannot be attributed, because the focus
distance of the stills is not recorded and the video record moves with
focus by about the same amount.  Nor does it confirm the per-channel split,
since Panasonic's still block is monochrome and RawTherapee's Panasonic
path has no CA term at all; the channel reading rests on the internal
evidence in section 3.


## 8. Not established

* Doubles 4, 5, 10 and 11, 16 and 17 are zero in every frame reachable
  here.  They are pad on that evidence, but a lens or a body that used them
  would not be distinguishable from one that does not.
* The version prefix is 3 in every frame.  No other value has been seen, so
  nothing is known about what a different version would change.
* All ProRes RAW clips on this machine are 5760x3040.  The claim that r = 1
  follows the coded frame's corner rather than some fixed sensor radius is
  supported by the RW2 comparison across two different diagonals, but not
  by two ProRes RAW frame sizes.
* The lateral CA channel ordering is R, G, B on the strength of R and B
  straddling G; no independent source confirms it.
* Nothing here was applied to an image.  The model has not been tested by
  warping a frame and checking that straight lines come out straight.
