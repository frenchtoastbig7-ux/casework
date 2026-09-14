"""Shared machinery for API-based collectors.

Some platforms cannot be collected by fetching HTML -- either because the
page is a client-side shell, or because the API exposes the material in a
form with better provenance than the rendered page does (full-resolution
media, authored-vs-boosted, real post timestamps rather than collection
timestamps).

What every such collector must do, and what lives here:

* **archive its own inputs.** The API response is the evidence for what was
  collected and why. Keeping the images while discarding the JSON leaves an
  artifact whose provenance stops at "the API said so".
* **notice when its parse has gone stale.** An API whose response shape moves
  makes a collector return nothing. That is a safe failure and an invisible
  one, so entries seen and media found are counted separately and compared.
* **verify authorship rather than infer it.** Each subclass implements the
  check appropriate to its platform; the base class only insists there is one.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import quote

from .fetch import Fetcher, FetchResult


class CollectorError(RuntimeError):
    """The API could not be reached or returned something unusable."""


@dataclass
class MediaItem:
    """One image, with the provenance needed to defend collecting it."""

    url: str
    kind: str                     # avatar | banner | post
    post_url: str | None          # permalink, as the referring context
    post_uri: str | None          # durable identifier, where one exists
    created_at: str | None        # when it was *posted*, not collected
    alt: str = ""
    sensitive: bool = False


class ApiCollector:
    """Base for one-actor API collectors."""

    #: Human-readable platform name, used in messages.
    platform = "api"
    #: Subclasses raise their own error type, so a caller can catch one
    #: platform's failures without catching every collector's.
    error_class: type[CollectorError] = CollectorError

    def __init__(
        self, fetcher: Fetcher, actor: str,
        on_response: Callable[[FetchResult], None] | None = None,
    ) -> None:
        self.fetcher = fetcher
        self.actor = actor
        self.on_response = on_response
        self.api_calls: list[str] = []
        self.entries_seen = 0
        self.media_found = 0

    # -- transport --------------------------------------------------------
    def _get_json(self, url: str) -> dict | list:
        self.api_calls.append(url)
        result = self.fetcher.get(url)
        if result is None:
            raise self.error_class(f"{self.platform}: no response after retries")
        if not result.ok:
            raise self.error_class(f"{self.platform}: HTTP {result.status}")
        if self.on_response:
            self.on_response(result)
        try:
            return json.loads(result.content)
        except json.JSONDecodeError as exc:
            raise self.error_class(f"{self.platform}: response was not JSON") from exc

    @staticmethod
    def query(**params: str | int | bool) -> str:
        parts = []
        for key, value in params.items():
            if value is None or value == "":
                continue
            text = "true" if value is True else ("false" if value is False else str(value))
            parts.append(f"{key}={quote(text, safe='')}")
        return "&".join(parts)

    # -- health -----------------------------------------------------------
    def parse_looks_wrong(self) -> bool:
        """Entries came back, but none yielded a usable image URL.

        Collectors request media-only feeds, so this combination means the
        response shape has moved -- not that the account is quiet.
        """
        return self.entries_seen > 0 and self.media_found == 0
