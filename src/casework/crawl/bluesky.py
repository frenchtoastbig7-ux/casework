"""Collection from Bluesky's public XRPC API.

`bsky.app` renders client-side, so the HTML crawler fetches a profile and
gets an empty shell. The AT Protocol's public read-only endpoints do serve
the underlying data without authentication, so Bluesky gets its own collector
rather than a source that is authorised, crawled, and silently returns
nothing.

Three hosts are involved and each is pinned:

* ``bsky.app``            -- what the analyst declares as the source
* ``public.api.bsky.app`` -- the XRPC endpoint actually queried
* ``cdn.bsky.app``        -- where the images live

That fan-out is why this is a collector and not a scope exception. The
general crawler's rule for off-host assets is "an in-scope page referred to
it", which is a heuristic. Here we can do better, because **a Bluesky CDN URL
contains the DID of the account that uploaded the image**:

    https://cdn.bsky.app/img/feed_fullsize/plain/did:plc:abc.../bafkrei...@jpeg

So every image URL is checked against the DID resolved from the declared
source, and one that does not match is refused before it is fetched. That is
a structural guarantee rather than an inference: a quote-post carrying
someone else's photograph cannot be collected into this subject's case even
though it appears in their feed.

Reposts are skipped for the same reason. An author feed contains material the
subject amplified but did not create, and collecting it would attribute other
people's photographs to this subject's case file.
"""
from __future__ import annotations

from collections.abc import Iterator
from urllib.parse import urlparse

from ..config import ScopeError
from .collector import ApiCollector, CollectorError, MediaItem

XRPC_HOST = "public.api.bsky.app"
XRPC_BASE = f"https://{XRPC_HOST}/xrpc"
CDN_HOSTS = frozenset({"cdn.bsky.app"})
WEB_HOST = "bsky.app"

# The API caps page size at 100. Pages are bounded separately so a prolific
# account cannot turn one source into an unbounded collection.
PAGE_LIMIT = 100
MAX_PAGES = 40


class BlueskyError(CollectorError):
    """The Bluesky API could not be reached or returned something unusable."""


def actor_from_source_url(url: str) -> str | None:
    """Extract the actor from a declared ``bsky.app/profile/<actor>`` source."""
    parts = urlparse(url)
    if (parts.hostname or "").lower() not in {WEB_HOST, f"www.{WEB_HOST}"}:
        return None
    segments = [s for s in parts.path.split("/") if s]
    if len(segments) >= 2 and segments[0] == "profile":
        return segments[1]
    return None


def did_of_cdn_url(url: str) -> str | None:
    """The uploading account's DID, read out of a Bluesky CDN path."""
    parts = urlparse(url)
    if (parts.hostname or "").lower() not in CDN_HOSTS:
        return None
    for segment in parts.path.split("/"):
        if segment.startswith("did:"):
            return segment
    return None


def post_permalink(handle_or_did: str, post_uri: str) -> str | None:
    """``at://did/app.bsky.feed.post/<rkey>`` -> the bsky.app permalink."""
    if not post_uri.startswith("at://"):
        return None
    rkey = post_uri.rsplit("/", 1)[-1]
    if not rkey:
        return None
    return f"https://{WEB_HOST}/profile/{handle_or_did}/post/{rkey}"


class BlueskyCollector(ApiCollector):
    """Walks one actor's public media. Touches only the three pinned hosts."""

    platform = "bluesky"
    error_class = BlueskyError

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.did: str | None = None
        self.handle: str | None = None

    # -- transport --------------------------------------------------------
    def _call(self, method: str, **params: str | int) -> dict:
        payload = self._get_json(f"{XRPC_BASE}/{method}?{self.query(**params)}")
        if not isinstance(payload, dict):
            raise BlueskyError(f"{method}: expected a JSON object")
        return payload

    # -- identity ---------------------------------------------------------
    def resolve(self) -> dict:
        """Resolve the actor to a DID. Everything else is checked against it.

        A Bluesky handle is a domain and can be reassigned to another person,
        so the DID -- which cannot -- is what the collection is pinned to.
        """
        profile = self._call("app.bsky.actor.getProfile", actor=self.actor)
        did = profile.get("did")
        if not did:
            raise BlueskyError(f"getProfile returned no DID for {self.actor!r}")
        self.did = did
        self.handle = profile.get("handle") or self.actor
        return profile

    def check_media_url(self, url: str) -> None:
        """Refuse any image not uploaded by the resolved account."""
        if self.did is None:
            raise BlueskyError("resolve() must run before media URLs are checked")
        host = (urlparse(url).hostname or "").lower()
        if host not in CDN_HOSTS:
            raise ScopeError(f"Bluesky media must come from the Bluesky CDN, got {url!r}")
        owner = did_of_cdn_url(url)
        if owner is None:
            raise ScopeError(f"Bluesky CDN URL carries no DID, cannot attribute it: {url!r}")
        if owner != self.did:
            raise ScopeError(
                f"Image belongs to {owner}, not to this source's account ({self.did}). "
                "Refusing: a repost or quoted post is not this subject's material."
            )

    # -- traversal --------------------------------------------------------
    def profile_media(self, profile: dict) -> Iterator[MediaItem]:
        for kind in ("avatar", "banner"):
            url = profile.get(kind)
            if url:
                yield MediaItem(
                    url=url, kind=kind,
                    post_url=f"https://{WEB_HOST}/profile/{self.handle}",
                    post_uri=None, created_at=None,
                )

    def _embed_images(self, embed: dict | None) -> list[dict]:
        """Images the post's own author attached.

        ``recordWithMedia`` is the subtle one: the *media* half is this
        author's, the *record* half is the quoted post and belongs to someone
        else. Only the media half is read. The DID check would catch the
        difference anyway, but not reaching for it keeps the refusal counts
        honest.
        """
        if not isinstance(embed, dict):
            return []
        kind = embed.get("$type", "")
        if kind.startswith("app.bsky.embed.images"):
            return [i for i in embed.get("images", []) if isinstance(i, dict)]
        if kind.startswith("app.bsky.embed.recordWithMedia"):
            return self._embed_images(embed.get("media"))
        if kind.startswith("app.bsky.embed.video"):
            thumb = embed.get("thumbnail")
            return [{"fullsize": thumb, "alt": embed.get("alt", "")}] if thumb else []
        return []

    def post_media(self, max_pages: int = MAX_PAGES) -> Iterator[MediaItem]:
        """Canonical name used by the runner; see :meth:`feed_media`."""
        return self.feed_media(max_pages)

    def feed_media(self, max_pages: int = MAX_PAGES) -> Iterator[MediaItem]:
        cursor: str | None = None
        for _page in range(max_pages):
            params: dict[str, str | int] = {
                "actor": self.actor, "limit": PAGE_LIMIT, "filter": "posts_with_media",
            }
            if cursor:
                params["cursor"] = cursor
            payload = self._call("app.bsky.feed.getAuthorFeed", **params)

            for entry in payload.get("feed", []):
                if not isinstance(entry, dict):
                    continue
                self.entries_seen += 1
                # Reposts: amplified, not authored. Not this subject's material.
                if entry.get("reason"):
                    continue
                post = entry.get("post")
                if not isinstance(post, dict):
                    continue
                if (post.get("author") or {}).get("did") != self.did:
                    continue

                uri = post.get("uri")
                created = (post.get("record") or {}).get("createdAt")
                permalink = post_permalink(self.handle or self.actor, uri or "")
                for image in self._embed_images(post.get("embed")):
                    url = image.get("fullsize") or image.get("thumb")
                    if url:
                        self.media_found += 1
                        yield MediaItem(
                            url=url, kind="post", post_url=permalink,
                            post_uri=uri, created_at=created,
                            alt=str(image.get("alt") or "")[:300],
                        )

            cursor = payload.get("cursor")
            if not cursor:
                return


__all__ = [
    "BlueskyCollector",
    "BlueskyError",
    "MediaItem",
    "actor_from_source_url",
    "did_of_cdn_url",
    "post_permalink",
]
