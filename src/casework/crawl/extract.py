"""Pull links and image URLs out of a fetched HTML page.

Image discovery is deliberately broad in *which attributes* it reads and
narrow in *what it accepts*. Modern pages almost never put the usable image
in a plain ``src``: it hides in ``srcset``, in ``data-src`` behind a lazy
loader, or in an Open Graph meta tag, and a naive ``img[src]`` scrape on such
a page collects a wall of 1x1 placeholder GIFs and tracking pixels.

From a ``srcset`` the *largest* candidate is taken. Face embedding quality
falls off a cliff below roughly 100px of inter-ocular distance, so collecting
the 320w thumbnail when a 1600w original was offered on the same line throws
away most of the signal the rest of the pipeline depends on.
"""
from __future__ import annotations

import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

# Attributes lazy-loading scripts stash the real URL in.
_SRC_ATTRS = ("src", "data-src", "data-original", "data-lazy-src", "data-image", "data-url")
_SRCSET_ATTRS = ("srcset", "data-srcset")
_TRACKING_HINTS = re.compile(r"(1x1|pixel|spacer|blank|transparent|beacon|analytics)", re.I)


def _best_from_srcset(value: str, base: str) -> str | None:
    """Highest-resolution candidate in a srcset, by width or pixel density."""
    best: tuple[float, str] | None = None
    for candidate in value.split(","):
        parts = candidate.strip().split()
        if not parts:
            continue
        url = urljoin(base, parts[0])
        weight = 1.0
        if len(parts) > 1:
            token = parts[1].strip().lower()
            try:
                if token.endswith("w"):
                    weight = float(token[:-1])
                elif token.endswith("x"):
                    weight = float(token[:-1]) * 1000
            except ValueError:
                weight = 1.0
        if best is None or weight > best[0]:
            best = (weight, url)
    return best[1] if best else None


def image_urls(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    found: list[str] = []

    def add(url: str | None) -> None:
        if not url:
            return
        if url.startswith("data:") or _TRACKING_HINTS.search(url):
            return
        absolute = urljoin(base_url, url)
        if absolute not in found:
            found.append(absolute)

    for tag in soup.find_all(["img", "source"]):
        for attr in _SRCSET_ATTRS:
            if tag.get(attr):
                add(_best_from_srcset(tag[attr], base_url))
        for attr in _SRC_ATTRS:
            if tag.get(attr):
                add(tag[attr])

    # Open Graph and Twitter card images: on profile pages this is very often
    # the single highest-quality portrait the page exposes.
    for meta in soup.find_all("meta"):
        prop = (meta.get("property") or meta.get("name") or "").lower()
        if prop in {"og:image", "og:image:url", "twitter:image", "twitter:image:src"}:
            add(meta.get("content"))

    # Thumbnails that link to their own full-size original.
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].lower().split("?")[0]
        if href.endswith((".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")):
            add(anchor["href"])

    return found


def page_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    hrefs = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if href and not href.startswith(("#", "javascript:", "mailto:", "tel:")):
            hrefs.append(urljoin(base_url, href))
    return hrefs


def page_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()[:200]
    return ""
