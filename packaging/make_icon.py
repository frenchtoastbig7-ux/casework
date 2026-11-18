"""Generate packaging/casework.icns from scratch.

Kept as code rather than a committed binary so the icon is reviewable and
reproducible. The motif is a small connection graph with one highlighted
node -- the identified subject and the people who recur around them, which is
what the tool is actually for. Deliberately unlike the geolocator's globe so
the two apps are not confused in the Dock.
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
S = 1024


def rounded_mask(size: int, radius_ratio: float = 0.2237) -> Image.Image:
    """macOS squircle-ish mask. 0.2237 approximates the Big Sur corner."""
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle(
        [0, 0, size - 1, size - 1], radius=int(size * radius_ratio), fill=255)
    return m


def draw_icon() -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Vertical gradient background: near-black graphite -> muted steel, a
    # colder and darker ground than the geolocator's navy.
    top, bottom = (22, 24, 28), (44, 50, 60)
    for y in range(S):
        t = y / S
        d.line([(0, y), (S, y)],
               fill=tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)))

    cx = cy = S / 2
    accent = (255, 106, 74)        # the subject node
    muted = (150, 170, 195)        # everyone else
    edge = (110, 130, 160, 150)

    # A ring of satellite nodes around one central subject, with edges from
    # the subject to each -- a co-occurrence star, the graph view in miniature.
    r = S * 0.30
    satellites = []
    for i in range(6):
        a = -math.pi / 2 + i * (2 * math.pi / 6)
        satellites.append((cx + r * math.cos(a), cy + r * math.sin(a)))

    for sx, sy in satellites:
        d.line([(cx, cy), (sx, sy)], fill=edge, width=10)
    # One edge between two satellites, so the graph reads as a network rather
    # than a simple star.
    d.line([satellites[1], satellites[2]], fill=edge, width=8)

    nr = S * 0.052
    for sx, sy in satellites:
        d.ellipse([sx - nr, sy - nr, sx + nr, sy + nr], fill=muted)

    # The subject node: larger, accent-coloured, with a thin reticle bracket
    # to read as "identified".
    br = S * 0.092
    d.ellipse([cx - br, cy - br, cx + br, cy + br], fill=accent)
    hr = br * 0.42
    d.ellipse([cx - hr, cy - hr, cx + hr, cy + hr], fill=(22, 24, 28))

    reach = br * 1.75
    tick = br * 0.55
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        x0, y0 = cx + dx * reach, cy + dy * reach
        d.line([(x0, y0), (x0 - dx * tick, y0)], fill=(255, 106, 74, 220), width=9)
        d.line([(x0, y0), (x0, y0 - dy * tick)], fill=(255, 106, 74, 220), width=9)

    img.putalpha(rounded_mask(S))
    return img


def main() -> int:
    icon = draw_icon()
    iconset = HERE / "casework.iconset"
    if iconset.exists():
        for f in iconset.iterdir():
            f.unlink()
    iconset.mkdir(exist_ok=True)

    for size in (16, 32, 64, 128, 256, 512):
        icon.resize((size, size), Image.LANCZOS).save(iconset / f"icon_{size}x{size}.png")
        icon.resize((size * 2, size * 2), Image.LANCZOS).save(
            iconset / f"icon_{size}x{size}@2x.png")
    icon.save(iconset / "icon_512x512@2x.png")

    out = HERE / "casework.icns"
    res = subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out)],
                         capture_output=True)
    if res.returncode != 0:
        print(res.stderr.decode(), file=sys.stderr)
        return 1
    icon.resize((512, 512), Image.LANCZOS).save(HERE / "casework-icon.png")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
