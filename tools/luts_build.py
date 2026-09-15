#!/usr/bin/env python3
# Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md.
"""Fetches and packs the embedded film simulation LUTs from filmgrid.org.

Two halves that run at different times:

  fetch   Run by hand, rarely. Crawls https://filmgrid.org/ politely (one
          request at a time, a delay between them, a real User-Agent),
          enumerates the Films and Looks category trees, and stores every
          .cube byte for byte as the site serves it under
          tools/gui/luts/films/<category>/<film>/<rendering>.cube, next to
          a manifest.json describing what was found. Those files are what
          lives in the repository.

  pack    Run by meson on every build where the inputs changed. Reads the
          manifest and the .cube files and writes one binary pack into the
          build directory; nothing generated is stored in the tree.

The pack keeps each LUT as its own independently compressed blob, because
cineview decompresses exactly one of them — the one the user picked — and
never the other 295. Each blob is the LUT quantized to 16 bits over its own
value range (so lossless to one part in 65535 of that range), delta coded
along the fastest axis (red, the way .cube orders its samples) and deflated.

Usage:
    luts_build.py fetch --dir tools/gui/luts [--cache DIR]
    luts_build.py pack --from-dir tools/gui/luts --out build/filmgrid.pack
"""

import argparse
import array
import datetime
import hashlib
import html
import json
import os
import re
import struct
import sys
import time
import urllib.parse
import urllib.request
import zlib

SITE = "https://filmgrid.org"
UA = ("libcraw-luts_build/1.0 (film simulation pack for libcraw's cineview; "
      "contact dev@lynne.ee)")
DELAY = 0.6          # seconds between requests; politeness, not a rate limit
PACK_MAGIC = b"CRAWLUT1"
PACK_VERSION = 1
MANIFEST = "manifest.json"

# ---------------------------------------------------------------- fetching

class Fetcher:
    """Sequential, delayed, cached HTTP GET. Never more than one in flight."""

    def __init__(self, cache_dir):
        self.cache = cache_dir
        self.last = 0.0
        self.gets = 0
        self.hits = 0
        os.makedirs(cache_dir, exist_ok=True)

    def get(self, url, binary=False):
        key = hashlib.sha256(url.encode()).hexdigest()[:24]
        path = os.path.join(self.cache, key)
        if os.path.exists(path):
            self.hits += 1
            data = open(path, "rb").read()
            return data if binary else data.decode("utf-8", "replace")
        dt = time.time() - self.last
        if dt < DELAY:
            time.sleep(DELAY - dt)
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
        finally:
            self.last = time.time()
        self.gets += 1
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        return data if binary else data.decode("utf-8", "replace")


def text(s):
    """HTML fragment to plain text."""
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def slugify(s):
    s = s.lower().replace("&", " and ").replace("+", " plus ")
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


CARD_RE = re.compile(
    r'<a class="card" href="(?P<href>[^"]+)">.*?<h3>(?P<name>.*?)</h3>'
    r'.*?<span class="count">(?P<count>[^<]*)</span>', re.S)
VARIANT_RE = re.compile(
    r'<td class="v-name">(?P<name>.*?)</td>\s*'
    r'<td class="v-label">(?P<label>.*?)</td>\s*'
    r'<td class="v-dl"><a href="(?P<href>[^"]+)"', re.S)


def crawl(fetcher):
    """Walks /lut/ and /look/ and returns the catalogue in the site's order."""
    # /lut/ is the Films index: an <h2> per category, then one card per film.
    index = fetcher.get(SITE + "/lut/")
    body = index[index.find("<main"):]
    parts = re.split(r"<h2[^>]*>(.*?)</h2>", body)
    cat_films = []
    for i in range(1, len(parts), 2):
        films = [(m.group("href"), text(m.group("name")), text(m.group("count")))
                 for m in CARD_RE.finditer(parts[i + 1])]
        if films:
            cat_films.append((text(parts[i]), films))
    if not cat_films:
        raise RuntimeError("no film categories found on /lut/")
    print(f"[fetch] {len(cat_films)} film categories, "
          f"{sum(len(f) for _, f in cat_films)} films", file=sys.stderr)

    categories, films, renderings = [], [], []
    by_slug = {}
    for cname, cards in cat_films:
        cat = {"name": cname, "slug": None, "url": None, "films": []}
        categories.append(cat)
        for href, fname, count in cards:
            fslug = href.strip("/").split("/")[-1]
            if fslug in by_slug:
                raise RuntimeError("film listed twice: " + fslug)
            page = fetcher.get(SITE + href)
            crumbs = re.search(r'<nav class="crumbs">(.*?)</nav>', page, re.S)
            m = re.search(r'href="/film/([a-z0-9-]+)/">(.*?)</a>',
                          crumbs.group(1) if crumbs else "")
            if m:
                # The crumb is the category's canonical slug and spelling;
                # cross-check it against the heading we grouped under.
                if cat["slug"] is None:
                    cat["slug"] = m.group(1)
                    cat["url"] = SITE + "/film/" + m.group(1) + "/"
                elif cat["slug"] != m.group(1):
                    raise RuntimeError(f"{fslug}: category slug disagrees")
                if text(m.group(2)) != cname:
                    raise RuntimeError(f"{fslug}: category name disagrees")
            h1 = re.search(r"<h1>(.*?)</h1>", page, re.S)
            eyebrow = re.search(r'<p class="eyebrow">(.*?)</p>', page, re.S)
            variants = list(VARIANT_RE.finditer(page))
            if not variants:
                raise RuntimeError("no renderings on " + href)
            head = count.split()[0]
            if head.lower() != "one" and int(head) != len(variants):
                raise RuntimeError(
                    f"{fslug}: card says {head}, table has {len(variants)}")
            film = {"slug": fslug, "name": text(h1.group(1)) if h1 else fname,
                    "url": SITE + href, "category": cname,
                    "about": text(eyebrow.group(1)) if eyebrow else "",
                    "renderings": []}
            by_slug[fslug] = film
            films.append(film)
            cat["films"].append(fslug)
            # Rendering file names must be stable and unique inside the
            # film's directory; the display names are, but guard anyway.
            used = set()
            for v in variants:
                name = text(v.group("name"))
                slug = slugify(name)
                while slug in used:
                    slug += "-2"
                used.add(slug)
                r = {"index": len(renderings), "name": name, "slug": slug,
                     "label": text(v.group("label")), "film": fslug,
                     "film_category": cname,
                     "source_url": SITE + v.group("href"),
                     "file": "films/%s/%s/%s.cube" % (cat["slug"] or
                                                      slugify(cname),
                                                      fslug, slug)}
                film["renderings"].append(r["index"])
                renderings.append(r)
        print(f"[fetch]   {cname}: {len(cat['films'])} films", file=sys.stderr)

    # /look/ regroups the same films by style. A look page is a flat list of
    # film cards, so a rendering is in a look exactly when its film is.
    look_index = fetcher.get(SITE + "/look/")
    looks = []
    hrefs = []
    for m in re.finditer(r'href="(/look/[a-z0-9-]+/)"', look_index):
        if m.group(1) not in hrefs:
            hrefs.append(m.group(1))
    for href in hrefs:
        page = fetcher.get(SITE + href)
        h1 = re.search(r"<h1>(.*?)</h1>", page, re.S)
        members = [m.group("href").strip("/").split("/")[-1]
                   for m in CARD_RE.finditer(page)]
        unknown = [f for f in members if f not in by_slug]
        if unknown:
            raise RuntimeError(f"{href}: films not in the Films tree: {unknown}")
        looks.append({"name": text(h1.group(1)),
                      "slug": href.strip("/").split("/")[-1],
                      "url": SITE + href, "films": members,
                      "renderings": [i for f in members
                                     for i in by_slug[f]["renderings"]]})
        print(f"[fetch]   look {looks[-1]['name']}: {len(members)} films, "
              f"{len(looks[-1]['renderings'])} renderings", file=sys.stderr)
    for r in renderings:
        r["looks"] = [l["name"] for l in looks if r["index"] in l["renderings"]]
    return categories, films, looks, renderings


def do_fetch(args):
    cache = args.cache or os.path.join(args.dir, ".httpcache")
    fetcher = Fetcher(cache)
    robots = fetcher.get(SITE + "/robots.txt")
    if re.search(r"(?im)^\s*disallow:\s*/\s*$", robots):
        raise SystemExit("robots.txt disallows crawling; stopping")

    categories, films, looks, renderings = crawl(fetcher)
    by_slug = {f["slug"]: f for f in films}

    # Mirror the .cube files byte for byte. Several renderings can point at
    # the same URL, so fetch each distinct one once and copy it into place.
    blobs = {}
    for n, r in enumerate(renderings):
        url = r["source_url"]
        if url not in blobs:
            blobs[url] = fetcher.get(url, binary=True)
        data = blobs[url]
        path = os.path.join(args.dir, r["file"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        size, dmin, dmax, title, vals = parse_cube(path)
        vmin, vmax = min(vals), max(vals)
        r.update(lut_size=size, domain_min=dmin, domain_max=dmax, title=title,
                 value_min=round(vmin, 9), value_max=round(vmax, 9),
                 outside_unit_range=bool(vmin < 0.0 or vmax > 1.0),
                 bytes=len(data),
                 sha256=hashlib.sha256(data).hexdigest())
        if (n + 1) % 50 == 0:
            print(f"[fetch]   {n + 1}/{len(renderings)} cubes", file=sys.stderr)

    man = {
        "source": SITE,
        "source_pages": {"films": SITE + "/lut/", "looks": SITE + "/look/"},
        "generator": "tools/luts_build.py fetch",
        "fetched": datetime.date.today().isoformat(),
        "counts": {"renderings": len(renderings), "distinct_cubes": len(blobs),
                   "films": len(films), "film_categories": len(categories),
                   "look_categories": len(looks)},
        # Films -> category -> film -> rendering, in the site's own order.
        "films_tree": [
            {"category": c["name"], "slug": c["slug"], "url": c["url"],
             "films": [{"name": by_slug[s]["name"], "slug": s,
                        "url": by_slug[s]["url"], "about": by_slug[s]["about"],
                        "renderings": by_slug[s]["renderings"]}
                       for s in c["films"]]}
            for c in categories],
        # Looks -> category -> renderings (with the film grouping kept too).
        "looks_tree": [
            {"category": l["name"], "slug": l["slug"], "url": l["url"],
             "films": l["films"], "renderings": l["renderings"]}
            for l in looks],
        "renderings": renderings,
    }
    with open(os.path.join(args.dir, MANIFEST), "w") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
        f.write("\n")
    total = sum(r["bytes"] for r in renderings)
    print(f"[fetch] {len(films)} films, {len(renderings)} renderings, "
          f"{len(blobs)} distinct cubes, {total / 1e6:.1f} MB of .cube text; "
          f"{fetcher.gets} requests ({fetcher.hits} from cache)",
          file=sys.stderr)

# ----------------------------------------------------------------- parsing

def parse_cube(path):
    """Minimal .cube reader. Returns (size, domain_min, domain_max, title, values).

    The values come back as one flat list in file order: blue slowest, red
    fastest, three floats per sample."""
    size = None
    dmin, dmax = [0.0] * 3, [1.0] * 3
    title = ""
    body = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == "#":
                continue
            c = line[0]
            if c == "-" or c == "." or c.isdigit():
                body.append(line)
                continue
            key = line.split(None, 1)[0].upper()
            if key == "TITLE":
                m = re.search(r'"(.*)"', line)
                title = m.group(1) if m else line[5:].strip()
            elif key == "LUT_3D_SIZE":
                size = int(line.split()[1])
            elif key == "LUT_1D_SIZE":
                raise ValueError(path + ": 1D LUT, not supported")
            elif key == "DOMAIN_MIN":
                dmin = [float(x) for x in line.split()[1:4]]
            elif key == "DOMAIN_MAX":
                dmax = [float(x) for x in line.split()[1:4]]
            elif key == "LUT_3D_INPUT_RANGE":
                lo, hi = (float(x) for x in line.split()[1:3])
                dmin, dmax = [lo] * 3, [hi] * 3
            else:
                raise ValueError(f"{path}: unknown keyword {key}")
    if size is None:
        raise ValueError(path + ": no LUT_3D_SIZE")
    # One str.split() over the whole body, then one C-level float conversion;
    # a per-line Python loop here costs seconds across 296 files.
    vals = [float(x) for x in " ".join(body).split()]
    if len(vals) != size ** 3 * 3:
        raise ValueError(f"{path}: {len(vals)} values, expected {size ** 3 * 3}")
    return size, dmin, dmax, title, vals


def encode_lut(size, vals):
    """Quantizes to 16 bits over the LUT's own range and delta codes on red.

    .cube stores red fastest, so consecutive triples are neighbours along
    red and their difference is small — which is the whole reason the
    deflate afterwards buys anything. Returns (bytes, vmin, vmax)."""
    vmin, vmax = min(vals), max(vals)
    scale = 65535.0 / (vmax - vmin) if vmax > vmin else 0.0
    q = [min(65535, max(0, int((v - vmin) * scale + 0.5))) for v in vals]
    # prev[i] is the sample one red step back, or 0 at the start of a run,
    # where the sample is stored absolutely so decoding can restart there.
    prev = [0, 0, 0] + q[:-3]
    for row in range(0, len(q) // 3, size):
        prev[row * 3] = prev[row * 3 + 1] = prev[row * 3 + 2] = 0
    d = array.array("H", [(a - b) & 0xFFFF for a, b in zip(q, prev)])
    if sys.byteorder != "little":
        d.byteswap()
    return d.tobytes(), vmin, vmax

# ----------------------------------------------------------------- packing

class Strings:
    """Deduplicating NUL terminated string table."""

    def __init__(self):
        self.buf = bytearray(b"\0")        # offset 0 is the empty string
        self.map = {"": 0}

    def add(self, s):
        s = s or ""
        if s not in self.map:
            self.map[s] = len(self.buf)
            self.buf += s.encode("utf-8") + b"\0"
        return self.map[s]


def zstd_compress_all(blobs, level=19):
    """Compresses with libzstd through ctypes, or returns None if absent."""
    try:
        import ctypes
        import ctypes.util
        lib = ctypes.CDLL(ctypes.util.find_library("zstd") or "libzstd.so.1")
    except OSError:
        return None
    lib.ZSTD_compressBound.restype = ctypes.c_size_t
    lib.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
    lib.ZSTD_compress.restype = ctypes.c_size_t
    lib.ZSTD_compress.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.c_int]
    lib.ZSTD_isError.restype = ctypes.c_uint
    lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
    out = []
    for b in blobs:
        cap = lib.ZSTD_compressBound(len(b))
        dst = ctypes.create_string_buffer(cap)
        n = lib.ZSTD_compress(dst, cap, b, len(b), level)
        if lib.ZSTD_isError(n):
            return None
        out.append(dst.raw[:n])
    return out


def do_pack(args):
    src = args.from_dir
    man = json.load(open(os.path.join(src, MANIFEST)))
    renderings = man["renderings"]
    films_tree = man["films_tree"]
    looks_tree = man["looks_tree"]

    films = [f for c in films_tree for f in c["films"]]
    film_ix = {f["slug"]: i for i, f in enumerate(films)}
    if len(looks_tree) > 32:
        raise SystemExit("more than 32 look categories, widen the bitmask")

    # Renderings are laid out film by film in the site's order so a film's
    # renderings are one contiguous run the menu can slice.
    order = [i for c in films_tree for f in c["films"] for i in f["renderings"]]
    if sorted(order) != list(range(len(renderings))):
        raise SystemExit("films_tree does not cover every rendering exactly once")
    pos = {ix: n for n, ix in enumerate(order)}
    luts = [renderings[i] for i in order]

    # Display names repeat across films, so a lookup slug that collides gets
    # its film prepended; that slug is what --lut NAME matches on.
    counts = {}
    for l in luts:
        counts[l["slug"]] = counts.get(l["slug"], 0) + 1
    used = set()
    for l in luts:
        s = l["slug"] if counts[l["slug"]] == 1 else l["film"] + "-" + l["slug"]
        while s in used:
            s += "-2"
        used.add(s)
        l["key"] = s

    t0 = time.time()
    st = Strings()
    blobs = []
    text_bytes = 0
    for l in luts:
        path = os.path.join(src, l["file"])
        size, dmin, dmax, title, vals = parse_cube(path)
        if size != l["lut_size"]:
            raise SystemExit(f"{l['file']}: size {size} != manifest "
                             f"{l['lut_size']}")
        text_bytes += os.path.getsize(path)
        raw, vmin, vmax = encode_lut(size, vals)
        blobs.append(raw)
        l.update(_size=size, _dmin=dmin, _dmax=dmax, _title=title,
                 _vmin=vmin, _vmax=vmax, _raw=len(raw))
    t_parse = time.time() - t0

    t0 = time.time()
    zl = [zlib.compress(b, 9) for b in blobs]
    ztotal = sum(len(x) for x in zl)
    comp, comp_flag, stotal = zl, 0, None
    if not args.no_zstd:
        zs = zstd_compress_all(blobs)
        if zs is not None:
            stotal = sum(len(x) for x in zs)
            # A second decompressor in cineview only earns its place on a
            # real win, so hold zstd to 20% better than deflate.
            if stotal <= ztotal * 0.8:
                comp, comp_flag = zs, 1
    t_comp = time.time() - t0

    # ---- assemble ------------------------------------------------------
    # Header, four fixed-size tables, the look index lists, the string
    # table, then the blobs. Little endian throughout, every offset from
    # the start of the file, so the loader just points into the array.
    HDR = "<8sIIIIIIIIIIIIII"
    LUT = "<IIIIIIIIIIIHHff3f3f"
    FILM = "<IIIIIII"
    FCAT = "<IIII"
    LCAT = "<IIIIIII"
    hdr_size = struct.calcsize(HDR)

    n_luts, n_films = len(luts), len(films)
    n_fcat, n_lcat = len(films_tree), len(looks_tree)
    dir_off = hdr_size
    films_off = dir_off + n_luts * struct.calcsize(LUT)
    fcat_off = films_off + n_films * struct.calcsize(FILM)
    lcat_off = fcat_off + n_fcat * struct.calcsize(FCAT)
    lists_off = lcat_off + n_lcat * struct.calcsize(LCAT)

    lists = bytearray()
    lcat_rows = []
    look_bit = {}
    for bit, lc in enumerate(looks_tree):
        look_bit[lc["category"]] = bit
        fl = [film_ix[s] for s in lc["films"]]
        ll = [pos[i] for i in lc["renderings"]]
        foff = lists_off + len(lists)
        lists += struct.pack("<%dI" % len(fl), *fl)
        loff = lists_off + len(lists)
        lists += struct.pack("<%dI" % len(ll), *ll)
        lcat_rows.append((st.add(lc["category"]), st.add(lc["slug"]),
                          st.add(lc["url"]), len(fl), foff, len(ll), loff))

    lut_rows = []
    for l in luts:
        mask = 0
        for name in l["looks"]:
            mask |= 1 << look_bit[name]
        flags = 0
        if l["_dmin"] != [0.0] * 3 or l["_dmax"] != [1.0] * 3:
            flags |= 1 << 0                       # non-unit domain
        if l["_vmin"] < 0.0 or l["_vmax"] > 1.0:
            flags |= 1 << 1                       # values outside [0,1]
        lut_rows.append([st.add(l["name"]), st.add(l["key"]),
                         st.add(l["label"]), film_ix[l["film"]],
                         st.add(l["source_url"]), st.add(l["_title"]),
                         st.add(l["file"]), mask,
                         0, 0, l["_raw"], l["_size"], flags,
                         l["_vmin"], l["_vmax"], *l["_dmin"], *l["_dmax"]])

    film_rows = []
    first = 0
    cat_of = {}
    for ci, c in enumerate(films_tree):
        for f in c["films"]:
            cat_of[f["slug"]] = ci
    for f in films:
        n = len(f["renderings"])
        film_rows.append((st.add(f["name"]), st.add(f["slug"]),
                          st.add(f["url"]), st.add(f["about"]),
                          cat_of[f["slug"]], first, n))
        first += n

    fcat_rows = []
    ff = 0
    for c in films_tree:
        fcat_rows.append((st.add(c["category"]), st.add(c["slug"]), ff,
                          len(c["films"])))
        ff += len(c["films"])

    strtab = bytes(st.buf)
    strtab_off = lists_off + len(lists)
    o = strtab_off + len(strtab)
    blobs_off = o
    for row, c in zip(lut_rows, comp):
        row[8] = o
        row[9] = len(c)
        o += len(c)

    out = bytearray()
    out += struct.pack(HDR, PACK_MAGIC, PACK_VERSION, comp_flag, n_luts,
                       n_films, n_fcat, n_lcat, dir_off, films_off, fcat_off,
                       lcat_off, lists_off, strtab_off, len(strtab), o)
    for row in lut_rows:
        out += struct.pack(LUT, *row)
    for row in film_rows:
        out += struct.pack(FILM, *row)
    for row in fcat_rows:
        out += struct.pack(FCAT, *row)
    for row in lcat_rows:
        out += struct.pack(LCAT, *row)
    out += lists + strtab
    for c in comp:
        out += c
    if len(out) != o:
        raise SystemExit("pack layout mismatch")

    d = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(d, exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "wb") as f:
        f.write(out)
    os.replace(tmp, args.out)

    # Let ninja track the 296 .cube files without meson having to list them:
    # a depfile names exactly what this run read, so a re-pack happens when
    # one of them or the manifest changes and never otherwise.
    if args.depfile:
        def esc(p):
            return p.replace("\\", "\\\\").replace(" ", "\\ ")
        deps = [os.path.join(src, MANIFEST)] + \
               [os.path.join(src, l["file"]) for l in luts] + [__file__]
        with open(args.depfile, "w") as f:
            f.write(esc(args.out) + ": " +
                    " ".join(esc(os.path.abspath(x)) for x in deps) + "\n")

    raw_total = sum(l["_raw"] for l in luts)
    print(f"[pack] {n_luts} LUTs from {text_bytes / 1e6:.1f} MB of .cube "
          f"text, sizes {sorted(set(l['_size'] for l in luts))}",
          file=sys.stderr)
    print(f"[pack] quantized {raw_total / 1e6:.2f} MB -> zlib "
          f"{ztotal / 1e6:.2f} MB" +
          (f", zstd {stotal / 1e6:.2f} MB" if stotal else "") +
          f"; using {'zstd' if comp_flag else 'zlib'}", file=sys.stderr)
    print(f"[pack] {args.out}: {len(out)} bytes ({len(out) / 1e6:.2f} MB) "
          f"in {t_parse:.1f}s parse + {t_comp:.1f}s compress", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "gui", "luts")

    f = sub.add_parser("fetch", help="crawl filmgrid.org into the source tree")
    f.add_argument("--dir", default=default_dir)
    f.add_argument("--cache", help="HTTP cache directory (default DIR/.httpcache)")

    p = sub.add_parser("pack", help="pack the source tree; never uses the network")
    p.add_argument("--from-dir", default=default_dir)
    p.add_argument("--out", required=True)
    p.add_argument("--depfile", help="write a Makefile dependency list here")
    p.add_argument("--no-zstd", action="store_true",
                   help="do not even measure zstd, just deflate")

    args = ap.parse_args()
    (do_fetch if args.cmd == "fetch" else do_pack)(args)


if __name__ == "__main__":
    main()
