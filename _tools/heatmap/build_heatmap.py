#!/usr/bin/env python3
"""Build running/tracks.json for the /running heatmap.

Usage:
  build_heatmap.py <strava export .zip | export folder | folder of .fit/.gpx/.tcx> [out.json]

Reads every run (Run / Trail Run / Track Run style activities; walks, hikes,
rides and treadmill runs are skipped), trims the first and last TRIM_M meters
of each one so start/end points (home) never get published, splits on GPS
jumps, simplifies, and writes quantized Web Mercator tracks plus per-zoom
color lookup tables. The lookup tables are histogram-equalized over the
SC/RWC core box so the orange -> white ramp is evenly spread there.

Needs: pip install fitdecode gpxpy numpy
"""
import csv
import gzip
import io
import json
import math
import os
import random
import sys
import warnings
import xml.etree.ElementTree as ET
import zipfile

import fitdecode
import gpxpy
import numpy as np

warnings.filterwarnings("ignore")

TRIM_M = 400          # cut this much off both ends of every run
JUMP_M = 150          # split the track if two consecutive fixes are farther apart
SIMPLIFY_M = 1.5      # Douglas-Peucker tolerance
Q = 1 << 26           # world is quantized to Q x Q integer units
# San Carlos + Redwood City (incl. Redwood Shores). Colors are equalized here.
CORE = {"south": 37.452, "west": -122.300, "north": 37.535, "east": -122.190}
LUT_ZOOMS = range(8, 20)  # effective zoom = map zoom + log2(devicePixelRatio)

RUN_TYPES = {"run", "trail run", "virtual run", "track run", "running"}


# ---------- reading ----------

def read_fit(data):
    pts = []
    with fitdecode.FitReader(io.BytesIO(data), check_crc=fitdecode.CrcCheck.DISABLED,
                             error_handling=fitdecode.ErrorHandling.IGNORE) as fr:
        sport = None
        for frame in fr:
            if not isinstance(frame, fitdecode.FitDataMessage):
                continue
            if frame.name == "sport" or frame.name == "session":
                if frame.has_field("sport"):
                    sport = sport or frame.get_value("sport")
                if frame.has_field("sub_sport"):
                    sub = frame.get_value("sub_sport")
                    if sub in ("treadmill", "indoor_running", "virtual_activity"):
                        sport = "indoor"
            if frame.name != "record":
                continue
            if not (frame.has_field("position_lat") and frame.has_field("position_long")):
                continue
            la, lo = frame.get_value("position_lat"), frame.get_value("position_long")
            if la is None or lo is None:
                continue
            pts.append((la * 180.0 / 2**31, lo * 180.0 / 2**31))
    return pts, sport


def read_gpx(data):
    g = gpxpy.parse(data.decode("utf-8", "replace"))
    pts = [(p.latitude, p.longitude) for t in g.tracks for s in t.segments for p in s.points]
    sport = g.tracks[0].type if g.tracks and g.tracks[0].type else None
    return pts, sport


def read_tcx(data):
    root = ET.fromstring(data.decode("utf-8", "replace").strip())
    pts, sport = [], None
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag == "Activity" and sport is None:
            sport = el.get("Sport")
        if tag == "Position":
            la = lo = None
            for c in el:
                ct = c.tag.split("}")[-1]
                if ct == "LatitudeDegrees":
                    la = float(c.text)
                elif ct == "LongitudeDegrees":
                    lo = float(c.text)
            if la is not None and lo is not None:
                pts.append((la, lo))
    return pts, sport


def parse_file(name, data):
    n = name.lower()
    if n.endswith(".gz"):
        data = gzip.decompress(data)
        n = n[:-3]
    if n.endswith(".fit"):
        return read_fit(data)
    if n.endswith(".gpx"):
        return read_gpx(data)
    if n.endswith(".tcx"):
        return read_tcx(data)
    return [], None


def iter_source(src):
    """Yield (name, bytes, csv_type_or_None)."""
    if zipfile.is_zipfile(src):
        z = zipfile.ZipFile(src)
        names = set(z.namelist())
        types = {}
        csv_name = next((n for n in names if n.endswith("activities.csv")), None)
        if csv_name:
            prefix = csv_name[: -len("activities.csv")]
            rows = csv.DictReader(io.TextIOWrapper(z.open(csv_name), "utf-8", errors="replace"))
            for r in rows:
                fn = (r.get("Filename") or "").strip()
                if fn:
                    types[prefix + fn] = (r.get("Activity Type") or "").strip().lower()
            for fn, t in types.items():
                if fn in names:
                    yield fn, z.read(fn), t
        else:
            for fn in sorted(names):
                yield fn, z.read(fn), None
        return
    types = {}
    csv_path = os.path.join(src, "activities.csv")
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                fn = (r.get("Filename") or "").strip()
                if fn:
                    types[os.path.normpath(os.path.join(src, fn))] = (r.get("Activity Type") or "").strip().lower()
    if types:
        for path, t in types.items():
            if os.path.exists(path):
                with open(path, "rb") as f:
                    yield path, f.read(), t
        return
    for root, _, files in os.walk(src):
        for fn in sorted(files):
            path = os.path.join(root, fn)
            with open(path, "rb") as f:
                yield path, f.read(), None


def is_run(csv_type, file_sport):
    if csv_type is not None:
        return csv_type in RUN_TYPES
    if file_sport is None:
        return True
    s = str(file_sport).lower()
    return s in RUN_TYPES or s == "running"


# ---------- geometry ----------

def merc(lat, lon):
    lat = max(min(lat, 85.0), -85.0)
    x = (lon + 180.0) / 360.0
    s = math.sin(math.radians(lat))
    y = 0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)
    return x, y


def hav(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 12742000 * math.asin(math.sqrt(h))


def trim_ends(pts):
    if len(pts) < 2:
        return []
    cum = [0.0]
    for a, b in zip(pts, pts[1:]):
        d = hav(a, b)
        cum.append(cum[-1] + (d if d < JUMP_M else 0))
    total = cum[-1]
    if total <= 2 * TRIM_M + 200:
        return []
    return [p for p, c in zip(pts, cum) if TRIM_M <= c <= total - TRIM_M]


def split_jumps(pts):
    out, cur = [], []
    for p in pts:
        if cur and hav(cur[-1], p) > JUMP_M:
            if len(cur) > 1:
                out.append(cur)
            cur = []
        cur.append(p)
    if len(cur) > 1:
        out.append(cur)
    return out


def rdp(xy, tol):
    n = len(xy)
    if n < 3:
        return xy
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = xy[i], xy[j]
        seg = b - a
        L = np.hypot(*seg)
        mid = xy[i + 1:j]
        if L == 0:
            d = np.hypot(*(mid - a).T)
        else:
            d = np.abs(seg[0] * (mid[:, 1] - a[1]) - seg[1] * (mid[:, 0] - a[0])) / L
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return xy[keep]


# ---------- equalization ----------

def raster_counts(tracks, zoom, core_q):
    """Distinct-activity count per pixel inside the core box at an effective zoom,
    rasterized the same way the browser does (DDA, floor)."""
    scale = 256 * 2**zoom / Q
    x0, y0, x1, y1 = core_q
    W = int((x1 - x0) * scale) + 1
    allpix = []
    for act_id, segs in tracks:
        pix = []
        for q in segs:
            p = q.astype(np.float64) * scale
            a, b = p[:-1], p[1:]
            n = np.maximum(np.ceil(np.abs(b - a).max(axis=1)), 1).astype(np.int64)
            tot = int(n.sum())
            if tot > 5_000_000:
                continue
            idx = np.repeat(np.arange(len(n)), n)
            start = np.concatenate(([0], np.cumsum(n)[:-1]))
            t = (np.arange(tot) - np.repeat(start, n)) / np.repeat(n, n)
            pts = a[idx] + (b - a)[idx] * t[:, None]
            px = np.floor(pts[:, 0] - x0 * scale).astype(np.int64)
            py = np.floor(pts[:, 1] - y0 * scale).astype(np.int64)
            ok = (px >= 0) & (py >= 0) & (px < W) & (py < int((y1 - y0) * scale) + 1)
            pix.append(py[ok] * W + px[ok])
        if pix:
            allpix.append(np.unique(np.concatenate(pix)))
    if not allpix:
        return np.array([], np.int64)
    _, counts = np.unique(np.concatenate(allpix), return_counts=True)
    return counts


def lut_for(counts):
    """Ramp position (0 = orange, 1 = white) for counts 1..max, from the share
    of core pixels with a lower count. Index 0 is count 1."""
    if len(counts) == 0:
        return [0]
    hist = np.bincount(counts)[1:].astype(np.float64)
    below = np.concatenate(([0.0], np.cumsum(hist)[:-1])) / hist.sum()
    if below[-1] > 0:
        below /= below[-1]
    return [round(float(v), 3) for v in below]


# ---------- main ----------

def main():
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(__file__), "..", "..", "running", "tracks.json")
    tracks = []
    seen = skipped = 0
    km = 0.0
    for name, data, ctype in iter_source(src):
        try:
            pts, sport = parse_file(name, data)
        except Exception as e:
            print("skip (parse)", name, e, file=sys.stderr)
            continue
        if not pts:
            continue
        seen += 1
        if not is_run(ctype, sport) or sport == "indoor":
            skipped += 1
            continue
        for a, b in zip(pts, pts[1:]):
            d = hav(a, b)
            if d < JUMP_M:
                km += d / 1000
        segs = []
        for part in split_jumps(trim_ends(pts)):
            xy = np.array([merc(la, lo) for la, lo in part])
            # simplify in meters (local scale), then quantize
            mpu = 40075016.7 * math.cos(math.radians(part[0][0]))
            s = rdp(xy * mpu, SIMPLIFY_M) / mpu
            q = np.round(s * Q).astype(np.int64)
            q = q[np.r_[True, np.any(np.diff(q, axis=0) != 0, axis=1)]]
            if len(q) > 1:
                segs.append(q)
        if segs:
            tracks.append((len(tracks), segs))

    random.seed(7)
    random.shuffle(tracks)
    tracks = [(i, s) for i, (_, s) in enumerate(tracks)]

    cx0, cy0 = merc(CORE["north"], CORE["west"])
    cx1, cy1 = merc(CORE["south"], CORE["east"])
    core_q = (cx0 * Q, cy0 * Q, cx1 * Q, cy1 * Q)
    luts = {}
    for z in LUT_ZOOMS:
        luts[str(z)] = lut_for(raster_counts(tracks, z, core_q))
        print(f"lut z{z}: max={luts[str(z)][-1]}", file=sys.stderr)

    enc = []
    npts = 0
    for _, segs in tracks:
        t = []
        for q in segs:
            d = np.vstack([q[:1], np.diff(q, axis=0)]).ravel().tolist()
            t.append(d)
            npts += len(q)
        enc.append(t)

    doc = {
        "q": Q,
        "core": CORE,
        "runs": len(tracks),
        "miles": round(km / 1.609344),
        "luts": luts,
        "tracks": enc,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(f"{seen} files with GPS, {skipped} non-runs skipped, {len(tracks)} runs, "
          f"{npts} points, {round(km / 1.609344)} mi -> {out} "
          f"({os.path.getsize(out) / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
