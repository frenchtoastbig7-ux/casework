"""Collection from a Mastodon instance's public REST API.

Unlike Bluesky, Mastodon's profile HTML is real and does contain images, so
this collector is an *upgrade* rather than a rescue -- and it degrades to the
HTML crawler when an instance declines to answer (see the runner). What the
API buys over scraping the page:

* ``only_media=true`` asks for exactly the posts worth collecting;
* attachments carry the full-resolution ``url``, not the display size the
  page happened to render;
* ``created_at`` is when the subject *posted*, which is what a timeline
  needs -- the HTML path can only record when we collected;
* ``reblog`` distinguishes authored from boosted, so other people's images
  are refusable rather than merely improbable;
* pagination reaches the whole history instead of the first page.

**The authorship check.** Mastodon has no equivalent of Bluesky's DID-in-URL,
because an instance may serve media from any object store its admin
configured. So the bound is derived from the instance's own behaviour rather
than hardcoded: the hosts serving *this account's* avatar and header are the
hosts trusted to serve its posts, plus the instance itself. Combined with
``status.account.id`` matching the resolved account and ``reblog`` being
absent, that gives authorship at the post level and a bounded host set at the
media level.

That is honestly weaker than the Bluesky check, which verifies each image
individually. It is stated here rather than glossed: on Mastodon the
guarantee is "this account posted it, and it came from where this instance
serves its media", not "this account uploaded these exact bytes".
"""
from __future__ import annotations

from collections.abc import Iterator
from urllib.parse import urlparse

from ..config import ScopeError
from .collector import ApiCollector, CollectorError, MediaItem

PAGE_LIMIT = 40          # the API's own maximum for this endpoint
MAX_PAGES = 40

# Attachment types worth collecting, and where the still image lives on each.
_STILL_FIELD = {"image": "url", "video": "preview_url", "gifv": "preview_url"}


class MastodonError(CollectorError):
    """The instance API could not be reached or returned something unusable."""


def account_from_source_url(url: str) -> tuple[str, str] | None:
    """``https://instance/@alice`` -> ``("instance", "alice")``.

    Returns None for anything that is not a bare profile URL, including
    ``/@alice/12345`` (a single status) and ``/@alice/media`` (a tab), so a
    source pointed at those keeps using the HTML crawler.
    """
    parts = urlparse(url)
    host = (parts.hostname or "").lower()
    segments = [s for s in parts.path.split("/") if s]
    if not host or len(segments) != 1 or not segments[0].startswith("@"):
        return None
    user = segments[0][1:]
    # A remote handle (@alice@other.social) on this instance is a cached copy;
    # the home instance is the authoritative source and is what the platform
    # expander produces, so only local handles take the API path.
    if not user or "@" in user:
        return None
    return host, user


class MastodonCollector(ApiCollector):
    """Walks one account's public media on its home instance."""

    platform = "mastodon"
    error_class = MastodonError

    def __init__(self, fetcher, instance: str, user: str, on_response=None) -> None:
        super().__init__(fetcher, user, on_response=on_response)
        self.instance = instance.lower()
        self.user = user
        self.account_id: str | None = None
        self.acct: str | None = None
        self.profile_url: str | None = None
        self.media_hosts: set[str] = {self.instance}

    def _api(self, path: str, **params: str | int | bool) -> dict | list:
        query = self.query(**params)
        url = f"https://{self.instance}/api/v1/{path}" + (f"?{query}" if query else "")
        return self._get_json(url)

    # -- identity ---------------------------------------------------------
    def resolve(self) -> dict:
        account = self._api("accounts/lookup", acct=self.user)
        if not isinstance(account, dict) or not account.get("id"):
            raise MastodonError(f"lookup returned no account for {self.user!r}")
        self.account_id = str(account["id"])
        self.acct = account.get("acct") or self.user
        self.profile_url = account.get("url") or f"https://{self.instance}/@{self.user}"

        # Trust, for media, whatever hosts this instance uses for this
        # account's own avatar and header.
        for field in ("avatar", "avatar_static", "header", "header_static"):
            host = (urlparse(str(account.get(field) or "")).hostname or "").lower()
            if host:
                self.media_hosts.add(host)
        return account

    def check_media_url(self, url: str) -> None:
        if self.account_id is None:
            raise MastodonError("resolve() must run before media URLs are checked")
        host = (urlparse(url).hostname or "").lower()
        if host not in self.media_hosts:
            raise ScopeError(
                f"{url!r} is not served from a host this instance uses for "
                f"{self.acct}'s own media ({', '.join(sorted(self.media_hosts))})."
            )

    # -- traversal --------------------------------------------------------
    def profile_media(self, account: dict) -> Iterator[MediaItem]:
        for field, kind in (("avatar_static", "avatar"), ("header_static", "banner")):
            url = account.get(field) or account.get(field.replace("_static", ""))
            if url:
                yield MediaItem(
                    url=str(url), kind=kind, post_url=self.profile_url,
                    post_uri=None, created_at=account.get("created_at"),
                )

    def _attachments(self, status: dict) -> Iterator[tuple[str, str]]:
        for item in status.get("media_attachments") or []:
            if not isinstance(item, dict):
                continue
            field = _STILL_FIELD.get(str(item.get("type")))
            if not field:
                continue                      # audio, or a type we cannot use
            url = item.get(field) or item.get("url")
            if url:
                yield str(url), str(item.get("description") or "")[:300]

    def post_media(self, max_pages: int = MAX_PAGES) -> Iterator[MediaItem]:
        """Canonical name used by the runner; see :meth:`account_media`."""
        return self.account_media(max_pages)

    def account_media(self, max_pages: int = MAX_PAGES) -> Iterator[MediaItem]:
        max_id: str | None = None
        for _page in range(max_pages):
            statuses = self._api(
                f"accounts/{self.account_id}/statuses",
                only_media=True, limit=PAGE_LIMIT, exclude_reblogs=True,
                max_id=max_id or "",
            )
            if not isinstance(statuses, list) or not statuses:
                return

            for status in statuses:
                if not isinstance(status, dict):
                    continue
                self.entries_seen += 1
                # Boosted, not authored. `exclude_reblogs` should have caught
                # these; it is re-checked because a server that ignores the
                # parameter must not quietly widen the collection.
                if status.get("reblog"):
                    continue
                if str((status.get("account") or {}).get("id")) != self.account_id:
                    continue

                for url, alt in self._attachments(status):
                    self.media_found += 1
                    yield MediaItem(
                        url=url, kind="post",
                        post_url=status.get("url") or status.get("uri"),
                        post_uri=status.get("uri"),
                        created_at=status.get("created_at"),
                        alt=alt, sensitive=bool(status.get("sensitive")),
                    )

            last = statuses[-1].get("id") if isinstance(statuses[-1], dict) else None
            if not last or len(statuses) < PAGE_LIMIT:
                return
            max_id = str(last)
