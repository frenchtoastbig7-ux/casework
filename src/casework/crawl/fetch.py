"""The only module in this package that touches the network.

Everything here is deliberately slow and conservative. A scoped case crawl is
tens to hundreds of URLs across a handful of hosts; there is nothing to gain
from concurrency and a great deal to lose, since an aggressive fetch pattern
is both discourteous and the most likely way for a collection to announce
itself to the operator of the site being collected from.

``robots.txt`` is honoured by default. An investigator with authority may
have grounds to override it, so the override exists -- but it is an explicit
per-run flag that writes its own audit entry, never a default and never
silent.
"""
from __future__ import annotations

import hashlib
import time
import urllib.robotparser
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

import httpx

from ..config import SETTINGS, require_network


@dataclass
class FetchResult:
    url: str
    status: int
    content: bytes
    content_type: str
    sha256: str
    elapsed: float
    from_cache: bool = False

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300 and bool(self.content)

    @property
    def is_image(self) -> bool:
        return self.content_type.split(";")[0].strip().lower().startswith("image/")

    @property
    def suffix(self) -> str:
        """File suffix from the served content type, not from the URL.

        URLs lie about their type constantly (extensionless CDN paths,
        ``?format=jpg`` query params). The response header is what the bytes
        actually are, and a wrong suffix makes OpenCV silently fail to decode.
        """
        base = self.content_type.split(";")[0].strip().lower()
        return {
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
            "image/bmp": ".bmp",
            "image/tiff": ".tif",
            "image/heic": ".heic",
            "text/html": ".html",
            "application/xhtml+xml": ".html",
            "application/json": ".json",
        }.get(base, ".bin")


class RobotsCache:
    """One robots.txt per host, fetched once per run.

    A host whose robots.txt cannot be fetched is treated as permitting the
    crawl. That is the conventional reading, and the alternative -- refusing
    to collect because a server returned 500 -- turns an unrelated outage
    into a blocked investigation.
    """

    def __init__(self, client: httpx.Client) -> None:
        self._client = client
        self._parsers: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def allowed(self, url: str) -> bool:
        parts = urlparse(url)
        host_key = f"{parts.scheme}://{parts.netloc}"
        if host_key not in self._parsers:
            self._parsers[host_key] = self._load(host_key)
        parser = self._parsers[host_key]
        if parser is None:
            return True
        return parser.can_fetch(SETTINGS.user_agent, url)

    def crawl_delay(self, url: str) -> float:
        parts = urlparse(url)
        parser = self._parsers.get(f"{parts.scheme}://{parts.netloc}")
        if parser is None:
            return 0.0
        try:
            delay = parser.crawl_delay(SETTINGS.user_agent)
        except Exception:
            return 0.0
        return float(delay or 0.0)

    def _load(self, host_key: str) -> urllib.robotparser.RobotFileParser | None:
        parts = urlparse(host_key)
        robots_url = urlunparse((parts.scheme, parts.netloc, "/robots.txt", "", "", ""))
        try:
            resp = self._client.get(robots_url, timeout=15.0)
        except httpx.HTTPError:
            return None
        if resp.status_code != 200 or not resp.text.strip():
            return None
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(resp.text.splitlines())
        return parser


class Fetcher:
    """Polite, rate-limited, size-capped HTTP client."""

    def __init__(self, respect_robots: bool = True) -> None:
        require_network()
        self.client = httpx.Client(
            timeout=SETTINGS.request_timeout,
            headers={
                "User-Agent": SETTINGS.user_agent,
                "Accept-Language": "en",
            },
            follow_redirects=True,
            max_redirects=5,
        )
        self.respect_robots = respect_robots
        self.robots = RobotsCache(self.client)
        self._last_hit: dict[str, float] = {}

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> Fetcher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _wait(self, url: str) -> None:
        """Space out requests per host, honouring any Crawl-delay directive."""
        host = urlparse(url).netloc
        pause = max(SETTINGS.request_pause, self.robots.crawl_delay(url))
        last = self._last_hit.get(host)
        if last is not None:
            remaining = pause - (time.monotonic() - last)
            if remaining > 0:
                time.sleep(remaining)
        self._last_hit[host] = time.monotonic()

    def blocked_by_robots(self, url: str) -> bool:
        return self.respect_robots and not self.robots.allowed(url)

    def get(self, url: str, referer: str | None = None) -> FetchResult | None:
        """Fetch one URL. Returns None when it could not be retrieved.

        Scope is *not* checked here. That is the caller's job, and keeping the
        check out of this class means a scope bug cannot be masked by a
        fetcher that quietly declines to fetch.
        """
        headers = {"Referer": referer} if referer else {}
        last_error = ""

        for attempt in range(SETTINGS.max_retries):
            self._wait(url)
            started = time.monotonic()
            try:
                with self.client.stream("GET", url, headers=headers) as resp:
                    declared = resp.headers.get("content-length")
                    if declared and int(declared) > SETTINGS.max_artifact_bytes:
                        return FetchResult(
                            url=url, status=resp.status_code, content=b"",
                            content_type=resp.headers.get("content-type", ""),
                            sha256="", elapsed=0.0,
                        )
                    chunks: list[bytes] = []
                    total = 0
                    oversize = False
                    for chunk in resp.iter_bytes():
                        total += len(chunk)
                        # Enforce the cap even when Content-Length was absent
                        # or lied; streaming means we stop paying for the
                        # download rather than discovering the size after it.
                        if total > SETTINGS.max_artifact_bytes:
                            oversize = True
                            break
                        chunks.append(chunk)
                    if oversize:
                        return FetchResult(
                            url=str(resp.url), status=resp.status_code, content=b"",
                            content_type=resp.headers.get("content-type", ""),
                            sha256="", elapsed=time.monotonic() - started,
                        )
                    body = b"".join(chunks)
                    return FetchResult(
                        url=str(resp.url),
                        status=resp.status_code,
                        content=body,
                        content_type=resp.headers.get("content-type", ""),
                        sha256=hashlib.sha256(body).hexdigest(),
                        elapsed=time.monotonic() - started,
                    )
            except httpx.HTTPError as exc:
                last_error = type(exc).__name__
                time.sleep(1.5 * (attempt + 1))

        if last_error:
            return None
        return None
