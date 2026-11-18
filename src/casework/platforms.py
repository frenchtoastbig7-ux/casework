"""Handle-to-URL expansion for known platforms.

This module does **string templating and nothing else**. It performs no
lookup, no search and no enumeration: it does not ask a platform whether a
handle exists, and it cannot be pointed at a handle to discover which
platforms carry it. Expanding ``@alice@mastodon.social`` yields exactly one
URL, the same one every time, whether or not that account exists. A handle
that is wrong produces a source that 404s on first collection.

That limit is the point. Sweeping one username across many platforms to find
which accounts exist is a discovery operation, and discovery is what turns
scoped collection back into open-ended searching. Expansion saves the analyst
from typing a URL they already know; it does not tell them anything they did
not already have.

The ``collection`` rating on each platform is a separate, practical warning.
Producing a correct profile URL is easy; getting images back from it is not.
Several major platforms serve no meaningful HTML to an unauthenticated
client, so a source pointed at them will be authorised, will be fetched, and
will yield nothing. Saying so here is more useful than letting an analyst
discover it after a twenty-minute crawl.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

# How much a plain, unauthenticated HTTP crawl actually gets back:
#   open     -- server-rendered HTML with real image URLs
#   partial  -- some content served, but truncated, gated or JS-dependent
#   blocked  -- login wall, aggressive bot detection, or an empty JS shell
COLLECTION_RATINGS = ("open", "partial", "blocked")


class HandleError(ValueError):
    """A handle could not be expanded into a profile URL."""


@dataclass(frozen=True)
class Platform:
    key: str
    name: str
    template: str
    example: str
    collection: str
    note: str = ""
    default_depth: int = 1
    needs_instance: bool = False
    # Handles are lowercased unless the platform's URLs are case-sensitive.
    lowercase: bool = True
    pattern: str = r"^[A-Za-z0-9_.-]{1,64}$"

    def expand(self, handle: str, instance: str | None = None) -> str:
        handle = clean_handle(handle)
        if self.lowercase:
            handle = handle.lower()
        if not re.match(self.pattern, handle):
            raise HandleError(
                f"{handle!r} does not look like a {self.name} handle "
                f"(expected e.g. {self.example})"
            )
        if self.needs_instance:
            if not instance:
                raise HandleError(
                    f"{self.name} is federated, so the instance is part of the identity. "
                    f"Give it as {self.example}, or pass --instance."
                )
            host = clean_instance(instance)
            return self.template.format(handle=handle, instance=host)
        return self.template.format(handle=handle)


def clean_handle(raw: str) -> str:
    """Accept the forms people actually paste.

    Analysts copy handles out of message headers, address bars and case
    notes, so ``@alice``, ``alice``, and a full profile URL all arrive here.
    Taking the last path segment off a pasted URL is deliberate: rejecting it
    would just mean the analyst hand-edits the string, which is where typos
    come from.
    """
    value = raw.strip().strip(",;")
    if value.startswith(("http://", "https://")):
        path = urlparse(value).path.rstrip("/")
        value = path.rsplit("/", 1)[-1] if path else ""
    return value.lstrip("@")


def clean_instance(raw: str) -> str:
    value = raw.strip().lower()
    if value.startswith(("http://", "https://")):
        value = urlparse(value).hostname or ""
    return value.strip("/").lstrip("@")


def split_fediverse(raw: str) -> tuple[str, str | None]:
    """``@alice@mastodon.social`` -> ``("alice", "mastodon.social")``."""
    value = raw.strip().lstrip("@")
    if value.startswith(("http://", "https://")):
        parts = urlparse(value)
        user = parts.path.rstrip("/").rsplit("/", 1)[-1].lstrip("@")
        return user, (parts.hostname or None)
    if "@" in value:
        user, _, host = value.partition("@")
        return user, host or None
    return value, None


# Handle grammars that differ from the default.
_DOMAINISH = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,252}\.[A-Za-z]{2,63}$|^did:[a-z]+:[A-Za-z0-9._%-]+$"
_LOOSE = r"^[A-Za-z0-9_.\-]{1,80}$"

PLATFORMS: tuple[Platform, ...] = (
    # --- the two that actually collect well --------------------------------
    Platform(
        key="mastodon", name="Mastodon", needs_instance=True,
        template="https://{instance}/@{handle}",
        example="@alice@mastodon.social",
        collection="open", default_depth=1,
        note="Collected through the instance REST API (full-resolution media, "
             "real post times, boosts excluded), falling back to crawling the "
             "profile page if the instance will not answer. The instance is "
             "part of the identity -- the same local name elsewhere is another "
             "person.",
    ),
    Platform(
        key="bluesky", name="Bluesky",
        template="https://bsky.app/profile/{handle}",
        example="alice.bsky.social (or did:plc:...)",
        collection="open", default_depth=2, pattern=_DOMAINISH,
        note="Collected through the public XRPC API, not by crawling bsky.app "
             "(which is a JS shell). Reposts and other accounts' images are "
             "refused on the DID in the CDN URL. Handles are reassignable "
             "domains -- pin the did:plc: where identity matters.",
    ),
    Platform(
        key="lemmy", name="Lemmy", needs_instance=True,
        template="https://{instance}/u/{handle}",
        example="@alice@lemmy.world", collection="open",
        note="Server-rendered, like Mastodon. Federated: instance is identity.",
    ),
    Platform(
        key="pixelfed", name="Pixelfed", needs_instance=True,
        template="https://{instance}/{handle}",
        example="@alice@pixelfed.social", collection="open",
        note="Fediverse image host; server-rendered and image-dense.",
    ),
    # --- open enough to be worth pointing at -------------------------------
    Platform(
        key="github", name="GitHub", template="https://github.com/{handle}",
        example="alice", collection="open", lowercase=False,
        note="Avatar and profile only. Useful for corroboration, not imagery.",
    ),
    Platform(
        key="flickr", name="Flickr",
        template="https://www.flickr.com/people/{handle}/",
        example="alice", collection="open", default_depth=2, pattern=_LOOSE,
        note="Image-dense and largely open. One of the better sources here.",
    ),
    Platform(
        key="tumblr", name="Tumblr", template="https://{handle}.tumblr.com/",
        example="alice", collection="partial", default_depth=2,
        note="Blog lives on a subdomain. Some blogs interpose an age gate.",
    ),
    Platform(
        key="vk", name="VK", template="https://vk.com/{handle}",
        example="alice", collection="partial",
        note="Public walls often render; much is gated behind login.",
    ),
    Platform(
        key="telegram", name="Telegram",
        template="https://t.me/s/{handle}", example="alicechannel",
        collection="partial", note="Public *channels* preview at /s/<name>. "
                                   "Private groups and user accounts do not.",
    ),
    Platform(
        key="twitch", name="Twitch", template="https://www.twitch.tv/{handle}",
        example="alice", collection="partial", note="Mostly a JS shell; avatar only.",
    ),
    Platform(
        key="reddit", name="Reddit",
        template="https://www.reddit.com/user/{handle}/",
        example="alice", collection="partial", default_depth=2,
        note="old.reddit.com renders far more to a plain client than www does.",
    ),
    # --- authorised, but expect little -------------------------------------
    Platform(
        key="x", name="X (Twitter)", template="https://x.com/{handle}",
        example="alice", collection="blocked",
        note="Unauthenticated fetches get a login wall. Expect nothing.",
    ),
    Platform(
        key="instagram", name="Instagram",
        template="https://www.instagram.com/{handle}/",
        example="alice", collection="blocked",
        note="Login wall and aggressive bot detection. Expect nothing.",
    ),
    Platform(
        key="facebook", name="Facebook",
        template="https://www.facebook.com/{handle}",
        example="alice", collection="blocked", note="Login wall.",
    ),
    Platform(
        key="threads", name="Threads",
        template="https://www.threads.net/@{handle}",
        example="alice", collection="blocked", note="JS shell behind a login wall.",
    ),
    Platform(
        key="tiktok", name="TikTok", template="https://www.tiktok.com/@{handle}",
        example="alice", collection="blocked",
        note="Heavy bot detection; imagery is video frames this tool does not extract.",
    ),
    Platform(
        key="linkedin", name="LinkedIn",
        template="https://www.linkedin.com/in/{handle}/",
        example="alice-doe-12345", collection="blocked", pattern=_LOOSE,
        note="Login wall, and unusually litigious about scraping.",
    ),
    Platform(
        key="youtube", name="YouTube",
        template="https://www.youtube.com/@{handle}",
        example="alice", collection="partial",
        note="Channel page yields avatar and thumbnails only, not video frames.",
    ),
    Platform(
        key="pinterest", name="Pinterest",
        template="https://www.pinterest.com/{handle}/",
        example="alice", collection="partial", default_depth=2,
    ),
    Platform(
        key="snapchat", name="Snapchat",
        template="https://www.snapchat.com/add/{handle}",
        example="alice", collection="blocked", note="Public profile card only.",
    ),
)

BY_KEY = {p.key: p for p in PLATFORMS}
# Aliases for what people actually type.
BY_KEY.update({
    "twitter": BY_KEY["x"], "bsky": BY_KEY["bluesky"], "fb": BY_KEY["facebook"],
    "ig": BY_KEY["instagram"], "yt": BY_KEY["youtube"], "li": BY_KEY["linkedin"],
    "toot": BY_KEY["mastodon"], "tg": BY_KEY["telegram"],
})


def get(key: str) -> Platform:
    platform = BY_KEY.get(key.strip().lower())
    if platform is None:
        known = ", ".join(sorted(p.key for p in PLATFORMS))
        raise HandleError(f"Unknown platform {key!r}. Known: {known}")
    return platform


def expand(platform_key: str, handle: str, instance: str | None = None) -> tuple[str, Platform]:
    """Expand one handle into one profile URL. No lookup, no discovery."""
    platform = get(platform_key)
    if platform.needs_instance:
        local, embedded = split_fediverse(handle)
        return platform.expand(local, instance or embedded), platform
    return platform.expand(handle, instance), platform


def catalogue() -> list[dict[str, object]]:
    return [
        {
            "key": p.key, "name": p.name, "example": p.example,
            "collection": p.collection, "depth": p.default_depth,
            "needs_instance": p.needs_instance, "note": p.note,
        }
        for p in PLATFORMS
    ]
