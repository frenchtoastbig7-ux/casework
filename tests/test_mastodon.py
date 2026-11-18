"""Mastodon REST collector, and the fallback that Bluesky cannot have.

Mastodon's HTML is real, so an instance refusing the API is a reason to crawl
the page rather than to abandon the source. That downgrade must be visible:
the page path cannot tell an authored post from a boost and records only when
we collected, not when the subject posted.
"""
from __future__ import annotations

import hashlib
import io
import json

import pytest

from casework.config import ScopeError
from casework.crawl.fetch import FetchResult
from casework.crawl.mastodon import (
    MastodonCollector,
    MastodonError,
    account_from_source_url,
)

INSTANCE = "infosec.exchange"
ACCOUNT_ID = "109820371"
OTHER_ID = "999999999"
MEDIA_HOST = "files.infosec.exchange"


def media(name: str, host: str = MEDIA_HOST) -> str:
    return f"https://{host}/media_attachments/files/{name}.jpg"


def json_result(url: str, payload) -> FetchResult:
    body = json.dumps(payload).encode()
    return FetchResult(url=url, status=200, content=body,
                       content_type="application/json",
                       sha256=hashlib.sha256(body).hexdigest(), elapsed=0.01)


def jpeg_bytes(seed: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (48, 32), (seed % 255, 100, 160)).save(buf, "JPEG")
    return buf.getvalue()


class FakeFetcher:
    """Serves canned JSON by API path, plus optional image and HTML bytes."""

    def __init__(self, routes: dict[str, list], blobs: dict[str, tuple[bytes, str]] | None = None):
        self.routes = {k: list(v) for k, v in routes.items()}
        self.blobs = blobs or {}
        self.requested: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def blocked_by_robots(self, url: str) -> bool:
        return False

    def get(self, url: str, referer: str | None = None):
        self.requested.append(url)
        if url in self.blobs:
            body, ctype = self.blobs[url]
            return FetchResult(url=url, status=200, content=body, content_type=ctype,
                               sha256=hashlib.sha256(body).hexdigest(), elapsed=0.01)
        for key, queue in self.routes.items():
            if key in url and queue:
                return json_result(url, queue.pop(0))
        return FetchResult(url=url, status=404, content=b"", content_type="",
                           sha256="", elapsed=0.0)


def account(**extra) -> dict:
    return {
        "id": ACCOUNT_ID, "username": "alice", "acct": "alice",
        "url": f"https://{INSTANCE}/@alice",
        "avatar_static": media("avatar"), "header_static": media("header"),
        **extra,
    }


def status(sid: str, urls: list[str], account_id: str = ACCOUNT_ID, **extra) -> dict:
    return {
        "id": sid, "created_at": "2026-04-01T09:00:00.000Z",
        "url": f"https://{INSTANCE}/@alice/{sid}",
        "uri": f"https://{INSTANCE}/users/alice/statuses/{sid}",
        "account": {"id": account_id, "acct": "alice"},
        "sensitive": False,
        "media_attachments": [
            {"id": f"m{i}", "type": "image", "url": u, "preview_url": u,
             "description": "alt text"}
            for i, u in enumerate(urls)
        ],
        **extra,
    }


def collector(routes) -> MastodonCollector:
    return MastodonCollector(FakeFetcher(routes), INSTANCE, "alice")


# --------------------------------------------------------------------- URLs
class TestSourceUrlParsing:
    def test_bare_profile_is_claimed(self):
        assert account_from_source_url(f"https://{INSTANCE}/@alice") == (INSTANCE, "alice")

    def test_trailing_slash_is_fine(self):
        assert account_from_source_url(f"https://{INSTANCE}/@alice/") == (INSTANCE, "alice")

    def test_single_status_url_is_left_to_the_html_crawler(self):
        # A source deliberately narrowed to one post should stay narrowed;
        # taking the API path would widen it to the whole account.
        assert account_from_source_url(f"https://{INSTANCE}/@alice/12345") is None

    def test_remote_handle_is_not_claimed(self):
        # A cached copy on someone else's instance; the home instance is
        # authoritative and is what the handle expander produces.
        assert account_from_source_url(f"https://{INSTANCE}/@alice@other.social") is None

    def test_non_profile_paths_are_not_claimed(self):
        assert account_from_source_url(f"https://{INSTANCE}/about") is None
        assert account_from_source_url("https://example.org/users/alice") is None


# ------------------------------------------------------------------- scope
class TestMediaScope:
    def test_media_hosts_are_derived_from_the_accounts_own_assets(self):
        c = collector({"accounts/lookup": [account()]})
        c.resolve()
        assert MEDIA_HOST in c.media_hosts
        assert INSTANCE in c.media_hosts

    def test_media_from_a_derived_host_is_allowed(self):
        c = collector({"accounts/lookup": [account()]})
        c.resolve()
        c.check_media_url(media("photo1"))

    def test_media_from_an_unrelated_host_is_refused(self):
        c = collector({"accounts/lookup": [account()]})
        c.resolve()
        with pytest.raises(ScopeError, match="not served from a host"):
            c.check_media_url("https://elsewhere.example/img/x.jpg")

    def test_checking_before_resolving_is_an_error(self):
        with pytest.raises(MastodonError, match="resolve"):
            collector({}).check_media_url(media("x"))

    def test_lookup_without_an_id_fails_loudly(self):
        with pytest.raises(MastodonError, match="no account"):
            collector({"accounts/lookup": [{"username": "alice"}]}).resolve()


# ------------------------------------------------------------------ statuses
class TestStatusWalk:
    def _walk(self, pages, acct=None):
        c = collector({"accounts/lookup": [acct or account()],
                       "/statuses": pages})
        c.resolve()
        return c, list(c.post_media())

    def test_own_media_is_collected_with_post_time(self):
        _c, items = self._walk([[status("1", [media("p1")])]])
        assert [i.url for i in items] == [media("p1")]
        # The API's value, not the collection time -- this is the point of it.
        assert items[0].created_at == "2026-04-01T09:00:00.000Z"
        assert items[0].post_url.endswith("/@alice/1")
        assert items[0].alt == "alt text"

    def test_boosts_are_skipped_even_if_the_server_ignores_the_filter(self):
        entry = status("1", [media("p1")])
        entry["reblog"] = {"id": "99", "account": {"id": OTHER_ID}}
        _c, items = self._walk([[entry]])
        assert items == []

    def test_statuses_by_another_account_are_skipped(self):
        _c, items = self._walk([[status("1", [media("p1")], account_id=OTHER_ID)]])
        assert items == []

    def test_only_media_is_requested(self):
        c, _ = self._walk([[]])
        assert any("only_media=true" in u for u in c.fetcher.requested)

    def test_video_uses_the_preview_frame_and_audio_is_skipped(self):
        entry = status("1", [])
        entry["media_attachments"] = [
            {"id": "a", "type": "video", "url": "https://x/v.mp4",
             "preview_url": media("vthumb"), "description": ""},
            {"id": "b", "type": "audio", "url": "https://x/a.mp3",
             "preview_url": None, "description": ""},
        ]
        _c, items = self._walk([[entry]])
        assert [i.url for i in items] == [media("vthumb")]

    def test_sensitive_flag_is_carried_through(self):
        entry = status("1", [media("p1")], sensitive=True)
        _c, items = self._walk([[entry]])
        assert items[0].sensitive is True

    def test_pagination_uses_max_id_and_stops_on_a_short_page(self):
        full = [status(str(i), [media(f"p{i}")]) for i in range(40)]
        c, items = self._walk([full, [status("99", [media("p99")])]])
        assert len(items) == 41
        status_calls = [u for u in c.fetcher.requested if "/statuses" in u]
        assert len(status_calls) == 2
        assert "max_id=39" in status_calls[1]

    def test_a_single_short_page_makes_no_second_request(self):
        c, _ = self._walk([[status("1", [media("p1")])]])
        assert len([u for u in c.fetcher.requested if "/statuses" in u]) == 1

    def test_profile_avatar_and_header_are_collected(self):
        c = collector({"accounts/lookup": [account()]})
        acct = c.resolve()
        assert [(i.kind, i.url) for i in c.profile_media(acct)] == [
            ("avatar", media("avatar")), ("banner", media("header"))
        ]

    def test_drift_is_flagged_when_posts_yield_no_media(self):
        entry = status("1", [])
        entry["media_attachments"] = [{"id": "a", "type": "somethingNew"}]
        c, items = self._walk([[entry]])
        assert items == []
        assert c.parse_looks_wrong()

    def test_a_quiet_account_is_not_flagged(self):
        c, items = self._walk([[]])
        assert items == []
        assert not c.parse_looks_wrong()


# ------------------------------------------------------- runner integration
class TestRunnerIntegration:
    @pytest.fixture(autouse=True)
    def _home(self, tmp_path, monkeypatch):
        import casework.config as config

        monkeypatch.setattr(config.SETTINGS, "root", tmp_path)
        yield

    def _case(self, url: str = f"https://{INSTANCE}/@alice"):
        from datetime import date, timedelta

        from casework.case import CaseStore
        from casework.models import Authorisation, Source, Subject

        store = CaseStore.create(
            "op-masto", "Mastodon",
            Authorisation(holder="h", instrument="W-1", granted=date.today(),
                          expires=date.today() + timedelta(days=30)),
        )
        case = store.load()
        case.subjects.append(Subject(id="alice", label="Alice Doe"))
        case.sources.append(Source(
            id="src-001", subject_id="alice", url=url,
            attribution="handle named in the schedule to the instrument", depth=1,
        ))
        store.save(case)
        return store

    def _run(self, monkeypatch, routes, blobs):
        import casework.crawl.runner as runner

        fake = FakeFetcher(routes, blobs)
        monkeypatch.setattr(runner, "Fetcher", lambda **kw: fake)
        store = self._case()
        return store, runner.crawl("op-masto"), fake

    def test_media_is_collected_with_provenance(self, monkeypatch):
        store, run, _ = self._run(
            monkeypatch,
            {"accounts/lookup": [account()], "/statuses": [[status("1", [media("p1")])]]},
            {media("p1"): (jpeg_bytes(1), "image/jpeg"),
             media("avatar"): (jpeg_bytes(2), "image/jpeg"),
             media("header"): (jpeg_bytes(3), "image/jpeg")},
        )
        case = store.load()
        urls = {a.url for a in case.artifacts if a.is_image}
        assert media("p1") in urls
        post = next(a for a in case.artifacts if a.url == media("p1"))
        assert post.subject_id == "alice"
        assert post.referer.endswith("/@alice/1")
        assert run.fetched >= 1

    def test_api_responses_are_archived(self, monkeypatch):
        store, _run, _ = self._run(
            monkeypatch,
            {"accounts/lookup": [account()], "/statuses": [[]]},
            {media("avatar"): (jpeg_bytes(2), "image/jpeg"),
             media("header"): (jpeg_bytes(3), "image/jpeg")},
        )
        api = [a for a in store.load().artifacts if "/api/v1/" in a.url]
        assert any("accounts/lookup" in a.url for a in api)

    def test_media_on_an_unrelated_host_is_refused_before_fetching(self, monkeypatch):
        foreign = "https://elsewhere.example/img/stolen.jpg"
        store, run, fake = self._run(
            monkeypatch,
            {"accounts/lookup": [account()],
             "/statuses": [[status("1", [media("p1"), foreign])]]},
            {media("p1"): (jpeg_bytes(1), "image/jpeg"), foreign: (jpeg_bytes(9), "image/jpeg"),
             media("avatar"): (jpeg_bytes(2), "image/jpeg"),
             media("header"): (jpeg_bytes(3), "image/jpeg")},
        )
        collected = {a.url for a in store.load().artifacts}
        assert media("p1") in collected
        assert foreign not in collected
        assert run.refused >= 1
        assert foreign not in fake.requested

    def test_api_failure_falls_back_to_the_page_crawl(self, monkeypatch):
        """An instance that will not answer the API is a downgrade, not a dead end."""
        page = (
            f"<html><head><title>alice</title>"
            f"<meta property='og:image' content='{media('ogimg')}'></head>"
            f"<body><img src='{media('inline')}'></body></html>"
        ).encode()
        store, run, _fake = self._run(
            monkeypatch,
            {},   # every API call 404s
            {f"https://{INSTANCE}/@alice": (page, "text/html"),
             media("ogimg"): (jpeg_bytes(4), "image/jpeg"),
             media("inline"): (jpeg_bytes(5), "image/jpeg")},
        )
        urls = {a.url for a in store.load().artifacts}
        assert f"https://{INSTANCE}/@alice" in urls, "the profile page should have been crawled"
        assert media("inline") in urls

        # And the downgrade is on the record, not silent.
        assert any("falling back" in n for n in run.notes)
        events = [e["event"] for e in store.audit.entries()]
        assert "mastodon.fallback" in events

    def test_a_single_status_source_keeps_using_the_html_crawler(self, monkeypatch):
        """A source deliberately narrowed to one post must stay narrowed."""
        import casework.crawl.runner as runner

        url = f"https://{INSTANCE}/@alice/12345"
        page = f"<html><body><img src='{media('one')}'></body></html>".encode()
        fake = FakeFetcher(
            {"accounts/lookup": [account()]},
            {url: (page, "text/html"), media("one"): (jpeg_bytes(6), "image/jpeg")},
        )
        monkeypatch.setattr(runner, "Fetcher", lambda **kw: fake)
        self._case(url)
        runner.crawl("op-masto")
        # The account lookup must never have happened: taking the API path
        # here would widen a one-post source to the whole account.
        assert not any("accounts/lookup" in u for u in fake.requested)

    def test_posting_time_reaches_the_artifact_not_just_the_log(self, monkeypatch):
        """Real post times are the point of the API path; a timeline needs
        them on the artifact, not buried in the audit log."""
        store, _run, _ = self._run(
            monkeypatch,
            {"accounts/lookup": [account()], "/statuses": [[status("1", [media("p1")])]]},
            {media("p1"): (jpeg_bytes(1), "image/jpeg"),
             media("avatar"): (jpeg_bytes(2), "image/jpeg"),
             media("header"): (jpeg_bytes(3), "image/jpeg")},
        )
        post = next(a for a in store.load().artifacts if a.url == media("p1"))
        assert post.posted == "2026-04-01T09:00:00.000Z"
        assert post.posted != post.fetched.isoformat()

    def test_page_crawled_artifacts_have_no_posting_time(self, monkeypatch):
        """None, not the collection time -- they are different facts."""
        page = f"<html><body><img src='{media('inline')}'></body></html>".encode()
        store, _run, _ = self._run(
            monkeypatch, {},
            {f"https://{INSTANCE}/@alice": (page, "text/html"),
             media("inline"): (jpeg_bytes(5), "image/jpeg")},
        )
        img = next(a for a in store.load().artifacts if a.url == media("inline"))
        assert img.posted is None
