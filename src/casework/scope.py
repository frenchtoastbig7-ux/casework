"""URL scope enforcement.

This is the module that makes the tool what it is, so it is small, pure and
heavily tested on purpose.

The rule has two tiers, because a one-tier rule is either useless or wrong:

**Pages** are in scope only if their URL sits at or beneath an authorised
source prefix, on the same host. Following a link off the prefix is refused
even when the link is on the same site -- a profile page's "people you may
know" strip is exactly the drift this tool exists to prevent.

**Assets** (images, mostly) are allowed from any host, but only when they are
embedded in a page that was itself in scope, and they are recorded with that
referring page. This exception is necessary rather than convenient: nearly
every real site serves its images from a separate CDN host, so a same-host
rule for assets would collect no pictures at all. It is narrow in the way
that matters -- an asset fetch cannot introduce new pages, cannot recurse,
and carries a referer proving which authorised page pulled it in.

Normalisation matters more than it looks. ``/users/alice`` must not authorise
``/users/alice-smith``, and ``..`` must not climb out of the prefix, so paths
are resolved and compared segment-wise, never by string prefix.
"""
from __future__ import annotations

import posixpath
from dataclasses import dataclass
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse

from .config import ScopeError

# Fetching these adds nothing to a face-collection case and drags in tracking
# and telemetry endpoints, so they are refused before the network is touched.
_BLOCKED_SCHEMES = {"javascript", "data", "mailto", "tel", "ftp", "file", "about"}
_ASSET_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic"}


def normalise(url: str) -> str:
    """Canonical form for comparison and dedup.

    Drops the fragment, lowercases scheme and host, strips a default port and
    a trailing dot on the host, and resolves dot segments in the path. Query
    strings are kept -- on paginated profiles the query *is* the page.
    """
    url, _frag = urldefrag(url.strip())
    parts = urlparse(url)
    scheme = parts.scheme.lower()
    host = parts.hostname or ""
    host = host.rstrip(".").lower()
    if parts.port and not (
        (scheme == "http" and parts.port == 80) or (scheme == "https" and parts.port == 443)
    ):
        host = f"{host}:{parts.port}"
    path = posixpath.normpath(parts.path or "/")
    if path == ".":
        path = "/"
    if parts.path.endswith("/") and not path.endswith("/"):
        path += "/"
    return urlunparse((scheme, host, path, "", parts.query, ""))


def _segments(path: str) -> list[str]:
    return [seg for seg in path.split("/") if seg]


def is_asset_url(url: str) -> bool:
    suffix = posixpath.splitext(urlparse(url).path)[1].lower()
    return suffix in _ASSET_SUFFIXES


@dataclass(frozen=True)
class Prefix:
    """One authorised source, precompiled for matching."""

    source_id: str
    subject_id: str
    scheme: str
    host: str
    segments: tuple[str, ...]
    depth: int

    @classmethod
    def from_url(cls, url: str, source_id: str, subject_id: str, depth: int) -> Prefix:
        parts = urlparse(normalise(url))
        if parts.scheme not in {"http", "https"}:
            raise ScopeError(f"Source URL must be http(s): {url!r}")
        if not parts.hostname:
            raise ScopeError(f"Source URL has no host: {url!r}")
        return cls(
            source_id=source_id,
            subject_id=subject_id,
            scheme=parts.scheme,
            host=parts.hostname.lower(),
            segments=tuple(_segments(parts.path)),
            depth=max(0, depth),
        )

    def covers(self, url: str) -> bool:
        """True when ``url`` sits at or beneath this prefix.

        Host must match exactly -- no subdomain widening. Comparing whole
        path segments is what stops ``/users/alice`` from authorising
        ``/users/alice-smith``.
        """
        parts = urlparse(normalise(url))
        if (parts.hostname or "").lower() != self.host:
            return False
        if parts.scheme not in {"http", "https"}:
            return False
        segs = _segments(parts.path)
        if len(segs) < len(self.segments):
            return False
        return tuple(segs[: len(self.segments)]) == self.segments

    def depth_of(self, url: str) -> int:
        """How many path segments below the prefix ``url`` sits."""
        segs = _segments(urlparse(normalise(url)).path)
        return max(0, len(segs) - len(self.segments))


class Scope:
    """The set of URLs one case is permitted to fetch."""

    def __init__(self, prefixes: list[Prefix]) -> None:
        self.prefixes = prefixes

    @classmethod
    def from_case(cls, case) -> Scope:
        return cls(
            [
                Prefix.from_url(s.url, s.id, s.subject_id, s.depth)
                for s in case.sources
                if s.enabled
            ]
        )

    def match(self, url: str) -> Prefix | None:
        """The most specific prefix covering ``url``, or None."""
        candidates = [p for p in self.prefixes if p.covers(url)]
        if not candidates:
            return None
        return max(candidates, key=lambda p: len(p.segments))

    def check_page(self, url: str) -> Prefix:
        """Authorise a page fetch, or raise.

        Also enforces the per-source depth limit, so a source declared with
        ``depth=1`` cannot be walked into a whole-site mirror.
        """
        parts = urlparse(url)
        if parts.scheme.lower() in _BLOCKED_SCHEMES:
            raise ScopeError(f"Refusing non-web scheme: {url!r}")
        prefix = self.match(url)
        if prefix is None:
            raise ScopeError(
                f"Out of scope: {url!r} is not beneath any authorised source for this case."
            )
        if prefix.depth_of(url) > prefix.depth:
            raise ScopeError(
                f"Out of scope: {url!r} is deeper than source {prefix.source_id} "
                f"authorises (depth {prefix.depth})."
            )
        return prefix

    def check_asset(self, url: str, referer: str) -> Prefix:
        """Authorise an asset fetch pulled in by an in-scope page.

        The referring page is re-checked rather than trusted, so an asset can
        never be laundered in by passing a referer the crawler never visited.
        """
        parts = urlparse(url)
        if parts.scheme.lower() not in {"http", "https"}:
            raise ScopeError(f"Refusing non-web asset scheme: {url!r}")
        if not is_asset_url(url):
            raise ScopeError(f"Not a recognised image asset: {url!r}")
        return self.check_page(referer)

    def allows_page(self, url: str) -> bool:
        try:
            self.check_page(url)
        except ScopeError:
            return False
        return True

    def partition_links(
        self, page_url: str, hrefs: list[str]
    ) -> tuple[list[str], list[str]]:
        """Split a page's links into (in scope, refused).

        Refusals are returned rather than dropped because the number of links
        a collection declined to follow is an audit fact in its own right --
        "the crawler saw 4,000 links to other people's profiles and followed
        none of them" is the evidence that scope was actually enforced.
        """
        allowed: list[str] = []
        refused: list[str] = []
        seen: set[str] = set()
        for href in hrefs:
            if not href or href.startswith("#"):
                continue
            try:
                candidate = normalise(urljoin(page_url, href))
            except ValueError:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            (allowed if self.allows_page(candidate) else refused).append(candidate)
        return allowed, refused

    def expand_links(self, page_url: str, hrefs: list[str]) -> list[str]:
        """Just the in-scope links. See :meth:`partition_links`."""
        return self.partition_links(page_url, hrefs)[0]
