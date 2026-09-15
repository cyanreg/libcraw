#!/usr/bin/env python3
# Copyright (c) 2026, Lynne. BSD-2-Clause, see LICENSE.md.
"""Extract Panasonic's per-frame lens correction record from ProRes RAW.

Demuxes a clip's video packets with the ffmpeg CLI, walks each ProRes RAW
frame (header -> tile table -> tile data), reads the vendor metadata that
follows the last tile, and decodes every

    com.panasonic.Semi-Pro.optical_correction

record it finds.  Nothing here touches libcraw; it is a standalone probe.

Frame layout, as walked by libavcodec/prores_raw.c:

    be32  frame size (includes this field and any trailing pad)
    4cc   'prrf'
    be16  header_len  (counts itself; header body is header_len - 2 bytes)
    ...   header body: reserved, version, vendor 4cc, w, h, crop[4],
          bayer, sensel range, wb factors, 3x3 matrix, gain, cct, flags,
          optional 64-byte qmat, optional 8x be16 linearisation curve
    be16 * nb_tiles   tile size table
    ...   tile data, concatenated in table order
    ...   vendor metadata records, then 'eomd', then zero padding

Vendor metadata record framing (confirmed against Apple's own reader):

    be32  size   -- counts the size field and the 4cc
    4cc   'psim' -- or 'eomd' (size 8) to end the list
    u8    key length
    ...   key, that many bytes, not NUL terminated
    ...   payload, size - 9 - keylen bytes

Usage:
    prr_lens.py CLIP [CLIP ...] [--frames N] [--skip N] [--hexdump]
    prr_lens.py CLIP --rw2 STILL.RW2      compare against Panasonic's
                                          in-camera RW2 distortion block
"""

import argparse
import json
import math
import struct
import subprocess
import sys

KEY = b"com.panasonic.Semi-Pro.optical_correction"

# Only frames this wide are reported in pixels; overridden by the real size.
DEFAULT_W, DEFAULT_H = 5760, 3040


class FrameError(Exception):
    pass


# ---------------------------------------------------------------- demuxing


def ffprobe_clip(path):
    """Container facts: frame count, size, and whatever names the lens."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,width,height,r_frame_rate,"
         "codec_name", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    info = {
        "codec": st.get("codec_name", "?"),
        "nb_frames": int(st.get("nb_frames", 0) or 0),
        "width": int(st.get("width", 0) or 0),
        "height": int(st.get("height", 0) or 0),
        "rate": st.get("r_frame_rate", ""),
    }
    info.update(exif_lens(path))
    return info


def exif_lens(path):
    """Lens identity out of the Panasonic makernote QuickTime carries."""
    keys = ("LensType", "LensModel", "FocalLength", "FocusDistance",
            "LensSerialNumber", "FNumber")
    try:
        out = subprocess.run(
            ["exiftool", "-a", "-u", "-s", "-s", "-json"] +
            ["-" + k for k in keys] + [path],
            capture_output=True, text=True, check=True).stdout
        d = json.loads(out)[0]
    except Exception:
        return {}
    return {k.lower(): d[k] for k in keys if k in d}


def packet_stream(path, frames=None, skip=0, ss=None):
    """Yield raw ProRes RAW frames from the clip, one bytes object each.

    `-f data` writes the packets back to back with nothing in between, so
    the be32 size at the head of each frame is enough to separate them.
    Read incrementally: clips here run to tens of gigabytes.
    """
    cmd = ["ffmpeg", "-v", "error"]
    if ss is not None:
        cmd += ["-ss", str(ss)]         # input seek; ProRes RAW is all-intra
    cmd += ["-i", path, "-map", "0:v", "-c", "copy"]
    if frames is not None:
        cmd += ["-frames:v", str(skip + frames)]
    cmd += ["-f", "data", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    n = 0
    try:
        while True:
            head = read_exactly(proc.stdout, 8)
            if head is None:
                return
            size, tag = struct.unpack(">I4s", head)
            if tag != b"prrf":
                raise FrameError("frame %d: tag %r, not 'prrf'" % (n, tag))
            if size < 8 or size > 1 << 30:
                raise FrameError("frame %d: implausible size %d" % (n, size))
            body = read_exactly(proc.stdout, size - 8)
            if body is None:
                raise FrameError("frame %d: truncated" % n)
            if n >= skip:
                yield n, head + body
            n += 1
            if frames is not None and n >= skip + frames:
                return
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()


def read_exactly(fh, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = fh.read(n - len(buf))
        if not chunk:
            return None if not buf else bytes(buf)
        buf += chunk
    return bytes(buf)


# ------------------------------------------------------------ frame walking


def parse_frame(pkt):
    """Walk header and tile table; return frame facts and the trailer slice.

    Mirrors the arithmetic in libavcodec/prores_raw.c so the trailer offset
    is derived, not searched for.
    """
    size, tag = struct.unpack_from(">I4s", pkt, 0)
    if tag != b"prrf" or size != len(pkt):
        raise FrameError("bad frame framing")

    hdr_len = struct.unpack_from(">H", pkt, 8)[0]
    if hdr_len < 62:
        raise FrameError("header_len %d too small" % hdr_len)
    h = pkt[10:8 + hdr_len]           # header body, hdr_len - 2 bytes

    o = 0
    o += 1                            # reserved
    version = h[o]; o += 1
    vendor = bytes(h[o:o + 4]); o += 4
    w, ht = struct.unpack_from(">HH", h, o); o += 4
    crop = tuple(h[o:o + 4]); o += 4
    bayer = struct.unpack_from(">H", h, o)[0] & 3; o += 2
    sensel = struct.unpack_from(">H", h, o)[0]; o += 2
    wb_r, wb_b = struct.unpack_from(">ff", h, o); o += 8
    matrix = struct.unpack_from(">9f", h, o); o += 36
    gain = struct.unpack_from(">f", h, o)[0]; o += 4
    cct = struct.unpack_from(">H", h, o)[0]; o += 2
    flags = struct.unpack_from(">H", h, o)[0]; o += 2
    align = (flags >> 1) & 7
    if flags & 1:
        o += 64                       # quantisation matrix
    if (flags >> 4) & 1:
        o += 16                       # 8 be16 linearisation control points

    tw16 = (w + 15) >> 4
    nb_tw = (tw16 >> align) + bin(tw16 & ((1 << align) - 1)).count("1")
    nb_th = (ht + 15) >> 4
    nb_tiles = nb_tw * nb_th

    tbl = 8 + hdr_len
    if tbl + nb_tiles * 2 > len(pkt):
        raise FrameError("tile table overruns frame")
    sizes = struct.unpack_from(">%dH" % nb_tiles, pkt, tbl)
    trailer_off = tbl + nb_tiles * 2 + sum(sizes)
    if trailer_off > len(pkt):
        raise FrameError("tile data overruns frame")

    return {
        "size": size, "version": version, "vendor": vendor,
        "w": w, "h": ht, "crop": crop, "bayer": bayer,
        "sensel": sensel, "wb": (wb_r, wb_b), "matrix": matrix,
        "gain": gain, "cct": cct, "flags": flags, "align": align,
        "nb_tiles": nb_tiles, "hdr_len": hdr_len,
        "trailer_off": trailer_off,
        "trailer": pkt[trailer_off:],
    }


def parse_records(trailer):
    """Split the vendor metadata into (fourcc, key, payload, raw) records."""
    recs = []
    o = 0
    while o + 8 <= len(trailer):
        size, tag = struct.unpack_from(">I4s", trailer, o)
        if tag == b"eomd":
            recs.append(("eomd", b"", b"", trailer[o:o + size]))
            break
        if size < 9 or o + size > len(trailer):
            raise FrameError("record at %d: size %d out of range" % (o, size))
        if tag != b"psim":
            recs.append((tag.decode("latin1"), b"",
                         bytes(trailer[o + 8:o + size]), trailer[o:o + size]))
            o += size
            continue
        klen = trailer[o + 8]
        key = bytes(trailer[o + 9:o + 9 + klen])
        payload = bytes(trailer[o + 9 + klen:o + size])
        recs.append(("psim", key, payload, trailer[o:o + size]))
        o += size
    return recs


# --------------------------------------------------------------- the record


def decode_optical(payload):
    """Decode the optical_correction payload.

    4-byte big-endian version, then 20 big-endian doubles:
    three channel blocks of six (k0..k3 then two slots that are zero in
    every frame seen), then the optical centre x, y.
    """
    if len(payload) < 4:
        raise FrameError("optical payload too short")
    ver = struct.unpack_from(">I", payload, 0)[0]
    body = payload[4:]
    if len(body) % 8:
        raise FrameError("optical payload %d bytes, not a whole number of "
                         "doubles after the version" % len(body))
    vals = list(struct.unpack(">%dd" % (len(body) // 8), body))
    if len(vals) != 20:
        return {"version": ver, "n": len(vals), "raw": vals}
    chans = {}
    for i, name in enumerate("RGB"):
        blk = vals[i * 6:i * 6 + 6]
        chans[name] = {"k": blk[:4], "tail": blk[4:]}
    return {
        "version": ver, "n": 20, "raw": vals,
        "chans": chans, "centre": (vals[18], vals[19]),
    }


def poly(k, r):
    """The record's radial factor at normalised radius r."""
    r2 = r * r
    return k[0] + k[1] * r2 + k[2] * r2 * r2 + k[3] * r2 * r2 * r2


def half_diag(w, h, cx, cy):
    """Radius that r = 1 corresponds to: the corner, from the given centre."""
    return math.hypot(max(cx, 1.0 - cx) * w, max(cy, 1.0 - cy) * h)


def corner_shift(k, hd):
    """Net corner displacement in pixels under r_src = r_dst * poly(r_dst).

    Negative means the output corner samples inside the source corner, i.e.
    the frame stays covered.
    """
    return hd * (poly(k, 1.0) - 1.0)


def shape_shift(k, hd):
    """Corner displacement with the centre scale divided out.

    poly(1)/poly(0) - 1 is the distortion the record actually corrects;
    poly(0) = k0 on its own is just the crop that keeps it in frame.
    Positive means pincushion correction (the corner is pulled outward
    relative to the centre), negative means barrel.
    """
    return hd * (poly(k, 1.0) / poly(k, 0.0) - 1.0)


# --------------------------------------------- Panasonic's RW2 still model


def rw2_distortion(path):
    """Read IFD0 tag 0x0119, Panasonic's in-camera distortion block.

    Sixteen signed little-endian 16-bit words; words 0, 1, 14 and 15 are
    checksums.  Parametrisation as RawTherapee reads it (its
    PanasonicMetadataLensCorrection, rtengine/lensmetadata.cc) and as
    exiftool names it (Image/ExifTool/PanasonicRaw.pm):

        word  5  DistortionScale   scale = 1 / (1 + w5/32768)
        word  8  DistortionParam08 a, the r^2 coefficient
        word  4  DistortionParam04 b, the r^4 coefficient
        word 11  DistortionParam11 c, the r^6 coefficient
        word  7  DistortionCorrection, low nibble, 1 = on
        word 12  the normalisation radius in pixels (the still's half
                 diagonal), which is what fixes r = 1 at the corner

    RawTherapee's applied factor, destination to source, r normalised by
    the half diagonal and measured from the geometric centre:

        r_src = r_dst / (scale * (1 + a r^2 + b r^4 + c r^6))
              = r_dst * (1 + w5/32768) / (1 + a r^2 + b r^4 + c r^6)
    """
    typesz = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8,
              11: 4, 12: 8, 13: 4}
    with open(path, "rb") as f:
        hdr = f.read(8)
        if hdr[:2] != b"II":
            raise FrameError("%s: not a little-endian TIFF/RW2" % path)
        f.seek(struct.unpack("<I", hdr[4:8])[0])
        n = struct.unpack("<H", f.read(2))[0]
        ent = f.read(n * 12)
        for i in range(n):
            tag, typ, cnt, val = struct.unpack_from("<HHI4s", ent, i * 12)
            if tag != 0x0119:
                continue
            size = typesz.get(typ, 1) * cnt
            if size <= 4:
                blob = val[:size]
            else:
                f.seek(struct.unpack("<I", val)[0])
                blob = f.read(size)
            if len(blob) < 32:
                raise FrameError("%s: 0x0119 is %d bytes, want 32"
                                 % (path, len(blob)))
            w = struct.unpack("<16h", blob[:32])
            return {
                "words": w, "raw": blob[:32],
                "scale": 1.0 / (1.0 + w[5] / 32768.0),
                "a": w[8] / 32768.0, "b": w[4] / 32768.0,
                "c": w[11] / 32768.0,
                "on": bool(w[7] & 0x0F), "norm_px": w[12],
            }
    raise FrameError("%s: no 0x0119 distortion block" % path)


def rw2_factor(d, r):
    """RawTherapee's destination-to-source radial factor at radius r."""
    r2 = r * r
    return 1.0 / (d["scale"] * (1.0 + r2 * (d["a"] + r2 * (d["b"] +
                                                           r2 * d["c"]))))


# ------------------------------------------------------------------ report


def run_clip(path, args):
    info = ffprobe_clip(path)
    lens = info.get("lenstype") or info.get("lensmodel") or "?"
    print("=" * 108)
    print("clip   %s" % path)
    print("lens   %s   focal %s   focus %s   f %s   serial %s" % (
        lens, info.get("focallength", "?"), info.get("focusdistance", "?"),
        info.get("fnumber", "?"), info.get("lensserialnumber", "?")))
    print("video  %s %dx%d  %s fps  %d frames (ffprobe)" % (
        info["codec"], info["width"], info["height"], info["rate"],
        info["nb_frames"]))
    print("=" * 108)
    if info["codec"] != "prores_raw":
        print("not a ProRes RAW clip, nothing to read")
        return {"path": path, "lens": lens, "rows": [], "seen": 0}

    seen = 0
    missing = 0
    rows = []
    for n, pkt in packet_stream(path, args.frames, args.skip, args.ss):
        fr = parse_frame(pkt)
        recs = parse_records(fr["trailer"])
        hit = None
        for tag, key, payload, raw in recs:
            if tag == "psim" and key == KEY:
                hit = (payload, raw)
                break
        if seen == 0:
            print("frame 0: %dx%d v%d vendor %r flags 0x%04x align %d "
                  "tiles %d" % (fr["w"], fr["h"], fr["version"], fr["vendor"],
                                fr["flags"], fr["align"], fr["nb_tiles"]))
            print("         trailer starts at 0x%x of 0x%x, %d bytes, "
                  "records: %s" % (
                      fr["trailer_off"], fr["size"], len(fr["trailer"]),
                      ", ".join("%s%s(%d)" % (
                          t, "/" + k.decode("latin1") if k else "", len(p))
                          for t, k, p, _ in recs)))
            if hit and args.hexdump:
                print("         record bytes (%d):" % len(hit[1]))
                print(hexdump(hit[1], fr["trailer_off"] + 0))
        seen += 1
        if not hit:
            missing += 1
            continue
        d = decode_optical(hit[0])
        rows.append((n, fr, d))

    if not rows:
        print("no optical_correction record in any of the %d frames read"
              % seen)
        return {"path": path, "lens": lens, "rows": [], "seen": seen}

    w, h = rows[0][1]["w"], rows[0][1]["h"]
    cx0, cy0 = rows[0][2]["centre"]
    hd = half_diag(w, h, cx0, cy0)
    print()
    print("r = 1 is the frame corner, %.1f px from the centre (%.4f, %.4f)."
          % (hd, cx0, cy0))
    print("net px  = corner shift of r_src = r_dst*poly(r_dst), crop included")
    print("dist px = the same with k0 divided out: what is actually corrected")
    print()
    print("%-6s %-2s %-13s %-12s %-12s %-12s %8s %8s" %
          ("frame", "ch", "k0", "k1", "k2", "k3", "net px", "dist px"))
    print("-" * 82)
    for n, fr, d in rows:
        if d["n"] != 20:
            print("%-6d unexpected payload: %d doubles" % (n, d["n"]))
            continue
        cx, cy = d["centre"]
        hdn = half_diag(fr["w"], fr["h"], cx, cy)
        for ch in "RGB":
            k = d["chans"][ch]["k"]
            print("%-6s %-2s % .10f % .5e % .5e % .5e %8.2f %8.2f" %
                  (n if ch == "R" else "", ch, k[0], k[1], k[2], k[3],
                   corner_shift(k, hdn), shape_shift(k, hdn)))
        tails = [v for ch in "RGB" for v in d["chans"][ch]["tail"]]
        kr, kb = d["chans"]["R"]["k"], d["chans"]["B"]["k"]
        print("%-6s    centre (%.6f, %.6f)  slots 4,5 %s  "
              "lateral CA R-B at corner %+.2f px" %
              ("", cx, cy, "zero" if not any(tails) else repr(tails),
               hdn * (poly(kr, 1.0) - poly(kb, 1.0))))
    if missing:
        print("(%d of %d frames carried no record)" % (missing, seen))
    return {"path": path, "lens": lens, "rows": rows, "seen": seen, "w": w,
            "h": h, "hd": hd}


def compare_rw2(clip, rw2_path, args):
    """Evaluate our record and Panasonic's still model on the same axis."""
    res = run_clip(clip, args)
    if not res["rows"]:
        print("no record to compare against")
        return
    d = rw2_distortion(rw2_path)
    ex = exif_lens(rw2_path)
    n, fr, rec = res["rows"][0]
    w, h = fr["w"], fr["h"]
    cx, cy = rec["centre"]
    hd_v = half_diag(w, h, cx, cy)

    print()
    print("=" * 108)
    print("RW2  %s" % rw2_path)
    print("     lens %s   focal %s   f %s" % (
        ex.get("lenstype", "?"), ex.get("focallength", "?"),
        ex.get("fnumber", "?")))
    print("     0x0119 words: %s" % " ".join("%6d" % x for x in d["words"]))
    print("     scale %.6f  a(r^2) %+.6f  b(r^4) %+.6f  c(r^6) %+.6f  "
          "correction %s  norm %d px"
          % (d["scale"], d["a"], d["b"], d["c"],
             "on" if d["on"] else "off", d["norm_px"]))
    print("=" * 108)

    # Panasonic normalises the still by its own half diagonal, which word 12
    # states in pixels.  Put both models on the video frame's radius axis.
    hd_s = float(d["norm_px"])
    ratio = hd_v / hd_s
    print("still half diagonal %.1f px, video %.1f px, ratio %.6f"
          % (hd_s, hd_v, ratio))
    print()
    print("%6s | %-11s %-11s %8s | %-11s %-11s %8s" %
          ("r", "prr f(r)", "rw2 cf(r)", "diff px",
           "prr shape", "rw2 shape", "diff px"))
    print("-" * 78)
    k = rec["chans"]["G"]["k"]
    worst_abs = worst_shape = 0.0
    for i in range(0, 21):
        r = i / 20.0
        pf = poly(k, r)
        cf = rw2_factor(d, r * ratio)
        ps = pf / poly(k, 0.0)
        cs = cf / rw2_factor(d, 0.0)
        da = (pf - cf) * r * hd_v
        ds = (ps - cs) * r * hd_v
        worst_abs = max(worst_abs, abs(da))
        worst_shape = max(worst_shape, abs(ds))
        print("%6.2f | %11.8f %11.8f %8.2f | %11.8f %11.8f %8.2f"
              % (r, pf, cf, da, ps, cs, ds))
    print()
    print("max |r_src difference| over r in 0..1, with each model's own "
          "crop: %.2f px" % worst_abs)
    print("max |r_src difference| with both crops divided out (pure "
          "distortion shape): %.2f px" % worst_shape)
    print("centre scale: record k0 %.6f (1/k0 %.6f), RW2 crop %.6f"
          % (poly(k, 0.0), 1.0 / poly(k, 0.0), 1.0 / rw2_factor(d, 0.0)))
    print("crop needed to keep the corner in frame: record %.6f, "
          "RW2 %.6f (margin %.3f%% and %.3f%%)"
          % (poly(k, 1.0) / poly(k, 0.0), rw2_factor(d, 1.0) /
             rw2_factor(d, 0.0),
             100.0 * (1.0 - poly(k, 1.0)),
             100.0 * (1.0 - rw2_factor(d, 1.0))))


def hexdump(b, base=0, limit=256):
    out = []
    for i in range(0, min(len(b), limit), 16):
        chunk = b[i:i + 16]
        hexs = " ".join("%02x%02x" % (chunk[j], chunk[j + 1])
                        for j in range(0, len(chunk) - 1, 2))
        txt = "".join(chr(c) if 32 <= c < 127 else "." for c in chunk)
        out.append("  %08x: %-40s %s" % (base + i, hexs, txt))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--frames", type=int, default=None,
                    help="stop after N frames (default: whole clip)")
    ap.add_argument("--skip", type=int, default=0,
                    help="skip N frames first")
    ap.add_argument("--ss", default=None,
                    help="seek to this timestamp before reading")
    ap.add_argument("--hexdump", action="store_true",
                    help="dump the raw record bytes of the first frame")
    ap.add_argument("--rw2", default=None,
                    help="compare the first frame against this RW2 still's "
                         "in-camera distortion block")
    args = ap.parse_args()
    for path in args.clips:
        try:
            if args.rw2:
                compare_rw2(path, args.rw2, args)
            else:
                run_clip(path, args)
        except (FrameError, subprocess.CalledProcessError) as e:
            print("%s: %s" % (path, e), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
