"""Bluesky XRPC collector.

Everything here runs against a canned API, deliberately: exercising this
against the live service would mean collecting a real person's photographs to
run a test, which is the behaviour the whole tool exists to constrain.

The tests that matter most are the refusals. An author feed contains material
the subject amplified but did not create, and the DID embedded in every CDN
URL is what lets that be refused structurally rather than guessed at.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from casework.config import ScopeError
from casework.crawl.bluesky import (
    BlueskyCollector,
    BlueskyError,
    actor_from_source_url,
    did_of_cdn_url,
    post_permalink,
)
from casework.crawl.fetch import FetchResult

SUBJECT_DID = "did:plc:subjectaaaaaaaaaaaaaaaaaaa"
OTHER_DID = "did:plc:strangerbbbbbbbbbbbbbbbbbb"


def cdn(did: str, cid: str, kind: str = "feed_fullsize") -> str:
    return f"https://cdn.bsky.app/img/{kind}/plain/{did}/{cid}@jpeg"


def result_for(url: str, payload: dict) -> FetchResult:
    body = json.dumps(payload).encode()
    return FetchResult(
        url=url, status=200, content=body, content_type="application/json",
        sha256=hashlib.sha256(body).hexdigest(), elapsed=0.01,
    )


class FakeFetcher:
    """Serves canned JSON keyed by XRPC method; records what was requested."""

    def __init__(self, responses: dict[str, list[dict]]) -> None:
        self.responses = {k: list(v) for k, v in responses.items()}
        self.requested: list[str] = []

    def get(self, url: str, referer: str | None = None) -> FetchResult | None:
        self.requested.append(url)
        method = url.split("/xrpc/", 1)[-1].split("?", 1)[0]
        queue = self.responses.get(method)
        if not queue:
            return FetchResult(url=url, status=404, content=b"", content_type="",
                               sha256="", elapsed=0.0)
        return result_for(url, queue.pop(0))


def profile(did: str = SUBJECT_DID, **extra) -> dict:
    return {"did": did, "handle": "alice.bsky.social", **extra}


def post(did: str, cid: str, rkey: str = "abc", **extra) -> dict:
    return {
        "post": {
            "uri": f"at://{did}/app.bsky.feed.post/{rkey}",
            "author": {"did": did},
            "record": {"createdAt": "2026-05-01T10:00:00Z"},
            "embed": {
                "$type": "app.bsky.embed.images#view",
                "images": [{"fullsize": cdn(did, cid), "alt": "a photo"}],
            },
        },
        **extra,
    }


def collector(responses: dict[str, list[dict]]) -> BlueskyCollector:
    return BlueskyCollector(FakeFetcher(responses), "alice.bsky.social")


# --------------------------------------------------------------------- URLs
class TestUrlParsing:
    def test_actor_extracted_from_profile_url(self):
        assert actor_from_source_url("https://bsky.app/profile/alice.bsky.social") == \
            "alice.bsky.social"

    def test_actor_extracted_when_a_did_is_used(self):
        assert actor_from_source_url(f"https://bsky.app/profile/{SUBJECT_DID}") == SUBJECT_DID

    def test_non_bluesky_url_is_not_claimed(self):
        assert actor_from_source_url("https://example.org/profile/alice") is None

    def test_bluesky_url_that_is_not_a_profile_is_not_claimed(self):
        assert actor_from_source_url("https://bsky.app/search?q=alice") is None

    def test_did_read_from_cdn_url(self):
        assert did_of_cdn_url(cdn(SUBJECT_DID, "bafkrei1")) == SUBJECT_DID

    def test_did_of_non_cdn_url_is_none(self):
        assert did_of_cdn_url(f"https://evil.example/img/plain/{SUBJECT_DID}/x@jpeg") is None

    def test_permalink_built_from_at_uri(self):
        assert post_permalink("alice.bsky.social", f"at://{SUBJECT_DID}/app.bsky.feed.post/xyz") \
            == "https://bsky.app/profile/alice.bsky.social/post/xyz"


# ------------------------------------------------------------------- scope
class TestMediaScope:
    def test_own_image_is_allowed(self):
        c = collector({"app.bsky.actor.getProfile": [profile()]})
        c.resolve()
        c.check_media_url(cdn(SUBJECT_DID, "bafkrei1"))

    def test_another_accounts_image_is_refused(self):
        c = collector({"app.bsky.actor.getProfile": [profile()]})
        c.resolve()
        with pytest.raises(ScopeError, match="not to this source's account"):
            c.check_media_url(cdn(OTHER_DID, "bafkrei2"))

    def test_image_from_a_foreign_host_is_refused(self):
        c = collector({"app.bsky.actor.getProfile": [profile()]})
        c.resolve()
        with pytest.raises(ScopeError, match="Bluesky CDN"):
            c.check_media_url(f"https://evil.example/img/plain/{SUBJECT_DID}/x@jpeg")

    def test_checking_before_resolving_is_an_error_not_a_pass(self):
        c = collector({})
        with pytest.raises(BlueskyError, match="resolve"):
            c.check_media_url(cdn(SUBJECT_DID, "bafkrei1"))

    def test_missing_did_means_no_profile_and_no_collection(self):
        c = collector({"app.bsky.actor.getProfile": [{"handle": "alice.bsky.social"}]})
        with pytest.raises(BlueskyError, match="no DID"):
            c.resolve()


# -------------------------------------------------------------------- feed
class TestFeedWalk:
    def _walk(self, feed_pages: list[dict], prof: dict | None = None):
        c = collector({
            "app.bsky.actor.getProfile": [prof or profile()],
            "app.bsky.feed.getAuthorFeed": feed_pages,
        })
        c.resolve()
        return c, list(c.feed_media())

    def test_own_posts_are_collected(self):
        _c, items = self._walk([{"feed": [post(SUBJECT_DID, "bafkrei1")]}])
        assert [i.url for i in items] == [cdn(SUBJECT_DID, "bafkrei1")]
        assert items[0].post_url.endswith("/post/abc")
        assert items[0].created_at == "2026-05-01T10:00:00Z"

    def test_reposts_are_skipped(self):
        """Amplified, not authored -- not this subject's material."""
        entry = post(SUBJECT_DID, "bafkrei1")
        entry["reason"] = {"$type": "app.bsky.feed.defs#reasonRepost"}
        _c, items = self._walk([{"feed": [entry]}])
        assert items == []

    def test_posts_by_another_author_are_skipped(self):
        _c, items = self._walk([{"feed": [post(OTHER_DID, "bafkrei9")]}])
        assert items == []

    def test_quote_post_takes_the_media_not_the_quoted_record(self):
        """recordWithMedia: the media half is the author's, the record half
        is someone else's post and must not be harvested."""
        entry = {
            "post": {
                "uri": f"at://{SUBJECT_DID}/app.bsky.feed.post/q1",
                "author": {"did": SUBJECT_DID},
                "record": {"createdAt": "2026-05-02T10:00:00Z"},
                "embed": {
                    "$type": "app.bsky.embed.recordWithMedia#view",
                    "media": {
                        "$type": "app.bsky.embed.images#view",
                        "images": [{"fullsize": cdn(SUBJECT_DID, "mine")}],
                    },
                    "record": {
                        "$type": "app.bsky.embed.record#view",
                        "record": {
                            "author": {"did": OTHER_DID},
                            "embeds": [{
                                "$type": "app.bsky.embed.images#view",
                                "images": [{"fullsize": cdn(OTHER_DID, "theirs")}],
                            }],
                        },
                    },
                },
            }
        }
        _c, items = self._walk([{"feed": [entry]}])
        assert [i.url for i in items] == [cdn(SUBJECT_DID, "mine")]

    def test_video_thumbnail_is_taken(self):
        entry = {
            "post": {
                "uri": f"at://{SUBJECT_DID}/app.bsky.feed.post/v1",
                "author": {"did": SUBJECT_DID}, "record": {},
                "embed": {"$type": "app.bsky.embed.video#view",
                          "thumbnail": cdn(SUBJECT_DID, "vthumb")},
            }
        }
        _c, items = self._walk([{"feed": [entry]}])
        assert [i.url for i in items] == [cdn(SUBJECT_DID, "vthumb")]

    def test_pagination_follows_the_cursor_then_stops(self):
        pages = [
            {"feed": [post(SUBJECT_DID, "p1", "r1")], "cursor": "c1"},
            {"feed": [post(SUBJECT_DID, "p2", "r2")]},          # no cursor -> stop
        ]
        c, items = self._walk(pages)
        assert [i.url for i in items] == [cdn(SUBJECT_DID, "p1"), cdn(SUBJECT_DID, "p2")]
        feed_calls = [u for u in c.fetcher.requested if "getAuthorFeed" in u]
        assert len(feed_calls) == 2
        assert "cursor=c1" in feed_calls[1]

    def test_media_filter_is_requested(self):
        c, _items = self._walk([{"feed": []}])
        assert any("filter=posts_with_media" in u for u in c.fetcher.requested)

    def test_profile_avatar_and_banner_are_collected(self):
        prof = profile(avatar=cdn(SUBJECT_DID, "av", "avatar"),
                       banner=cdn(SUBJECT_DID, "bn", "banner"))
        c = collector({"app.bsky.actor.getProfile": [prof]})
        resolved = c.resolve()
        kinds = [(i.kind, i.url) for i in c.profile_media(resolved)]
        assert kinds == [("avatar", cdn(SUBJECT_DID, "av", "avatar")),
                         ("banner", cdn(SUBJECT_DID, "bn", "banner"))]

    def test_api_responses_are_handed_back_for_archiving(self):
        archived = []
        c = BlueskyCollector(
            FakeFetcher({"app.bsky.actor.getProfile": [profile()]}),
            "alice.bsky.social", on_response=archived.append,
        )
        c.resolve()
        assert len(archived) == 1
        assert json.loads(archived[0].content)["did"] == SUBJECT_DID

    def test_http_error_becomes_a_readable_failure(self):
        c = collector({})   # every method 404s
        with pytest.raises(BlueskyError, match="HTTP 404"):
            c.resolve()


# ------------------------------------------------------- runner integration
class FullFakeFetcher(FakeFetcher):
    """Serves the API *and* image bytes, and stands in for Fetcher itself."""

    def __init__(self, responses, images):
        super().__init__(responses)
        self.images = images

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def blocked_by_robots(self, url: str) -> bool:
        return False

    def get(self, url: str, referer: str | None = None):
        if url in self.images:
            self.requested.append(url)
            body = self.images[url]
            return FetchResult(
                url=url, status=200, content=body, content_type="image/jpeg",
                sha256=hashlib.sha256(body).hexdigest(), elapsed=0.01,
            )
        return super().get(url, referer)


def jpeg_bytes(seed: int) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (seed % 255, 90, 140)).save(buf, "JPEG")
    return buf.getvalue()


class TestRunnerIntegration:
    @pytest.fixture(autouse=True)
    def _home(self, tmp_path, monkeypatch):
        import casework.config as config

        monkeypatch.setattr(config.SETTINGS, "root", tmp_path)
        yield

    def _case_with_bluesky_source(self):
        from datetime import date, timedelta

        from casework.case import CaseStore
        from casework.models import Authorisation, Source, Subject

        store = CaseStore.create(
            "op-bsky", "Bluesky",
            Authorisation(holder="h", instrument="W-1", granted=date.today(),
                          expires=date.today() + timedelta(days=30)),
        )
        case = store.load()
        case.subjects.append(Subject(id="alice", label="Alice Doe"))
        case.sources.append(Source(
            id="src-001", subject_id="alice",
            url="https://bsky.app/profile/alice.bsky.social",
            attribution="handle named in the schedule to the instrument", depth=2,
        ))
        store.save(case)
        return store

    def _run(self, monkeypatch, feed_pages, extra_images=None):
        import casework.crawl.runner as runner

        mine = cdn(SUBJECT_DID, "bafkrei1")
        theirs = cdn(OTHER_DID, "bafkrei2")
        images = {mine: jpeg_bytes(1), theirs: jpeg_bytes(2)}
        images.update(extra_images or {})

        fake = FullFakeFetcher(
            {"app.bsky.actor.getProfile": [profile()],
             "app.bsky.feed.getAuthorFeed": feed_pages},
            images,
        )
        monkeypatch.setattr(runner, "Fetcher", lambda **kw: fake)
        store = self._case_with_bluesky_source()
        run = runner.crawl("op-bsky")
        return store, run, fake

    def test_own_images_are_collected_with_provenance(self, monkeypatch):
        store, run, _ = self._run(monkeypatch, [{"feed": [post(SUBJECT_DID, "bafkrei1")]}])
        case = store.load()
        images = [a for a in case.artifacts if a.is_image]
        assert len(images) == 1
        assert images[0].url == cdn(SUBJECT_DID, "bafkrei1")
        assert images[0].subject_id == "alice"
        assert images[0].referer.endswith("/post/abc")
        assert run.fetched >= 1

    def test_the_api_response_is_archived_as_evidence(self, monkeypatch):
        store, _run, _ = self._run(monkeypatch, [{"feed": []}])
        case = store.load()
        api = [a for a in case.artifacts if "xrpc" in a.url]
        assert api, "the XRPC responses should be kept, not just their conclusions"
        assert any("getProfile" in a.url for a in api)

    def test_another_accounts_image_is_refused_and_counted(self, monkeypatch):
        """The feed hands back a foreign image; the DID check must refuse it.

        Constructed by forging an embed on the subject's own post, which is
        the case a referer-based rule would wave through.
        """
        entry = post(SUBJECT_DID, "bafkrei1")
        entry["post"]["embed"]["images"].append({"fullsize": cdn(OTHER_DID, "bafkrei2")})
        store, run, fake = self._run(monkeypatch, [{"feed": [entry]}])

        collected = {a.url for a in store.load().artifacts}
        assert cdn(SUBJECT_DID, "bafkrei1") in collected
        assert cdn(OTHER_DID, "bafkrei2") not in collected
        assert run.refused >= 1
        # Refused before the network was touched, not fetched and discarded.
        assert cdn(OTHER_DID, "bafkrei2") not in fake.requested

    def test_refusal_is_on_the_audit_record(self, monkeypatch):
        entry = post(SUBJECT_DID, "bafkrei1")
        entry["post"]["embed"]["images"].append({"fullsize": cdn(OTHER_DID, "bafkrei2")})
        store, _run, _ = self._run(monkeypatch, [{"feed": [entry]}])
        events = [e["event"] for e in store.audit.entries()]
        assert "bluesky.refused" in events
        assert "bluesky.resolved" in events

    def test_the_html_crawler_is_not_used_for_bluesky(self, monkeypatch):
        _store, _run, fake = self._run(monkeypatch, [{"feed": []}])
        # bsky.app itself is a JS shell; fetching it would waste a request and
        # store an empty page as if it were collected material.
        assert not any(u.startswith("https://bsky.app/") for u in fake.requested)

    def test_rerun_does_not_recollect_the_same_image(self, monkeypatch):
        import casework.crawl.runner as runner

        store, _run, _ = self._run(monkeypatch, [{"feed": [post(SUBJECT_DID, "bafkrei1")]}])
        before = len([a for a in store.load().artifacts if a.is_image])

        fake2 = FullFakeFetcher(
            {"app.bsky.actor.getProfile": [profile()],
             "app.bsky.feed.getAuthorFeed": [{"feed": [post(SUBJECT_DID, "bafkrei1")]}]},
            {cdn(SUBJECT_DID, "bafkrei1"): jpeg_bytes(1)},
        )
        monkeypatch.setattr(runner, "Fetcher", lambda **kw: fake2)
        run2 = runner.crawl("op-bsky")

        assert len([a for a in store.load().artifacts if a.is_image]) == before
        assert run2.skipped >= 1

    def test_expired_authorisation_blocks_the_bluesky_path_too(self, monkeypatch):
        from datetime import date, timedelta

        import casework.crawl.runner as runner
        from casework.config import AuthorisationError

        store = self._case_with_bluesky_source()
        case = store.load()
        case.authorisation.expires = date.today() - timedelta(days=1)
        store.save(case)
        monkeypatch.setattr(runner, "Fetcher", lambda **kw: None)

        with pytest.raises(AuthorisationError):
            runner.crawl("op-bsky")


class TestParseDriftDetection:
    """A response shape that moved must not look like a quiet account."""

    def test_posts_with_no_readable_media_is_flagged(self):
        entry = {
            "post": {
                "uri": f"at://{SUBJECT_DID}/app.bsky.feed.post/x",
                "author": {"did": SUBJECT_DID}, "record": {},
                # Plausible future shape this code does not understand.
                "embed": {"$type": "app.bsky.embed.somethingNew#view", "items": []},
            }
        }
        c = collector({"app.bsky.actor.getProfile": [profile()],
                       "app.bsky.feed.getAuthorFeed": [{"feed": [entry]}]})
        c.resolve()
        assert list(c.feed_media()) == []
        assert c.parse_looks_wrong()

    def test_a_genuinely_empty_feed_is_not_flagged(self):
        c = collector({"app.bsky.actor.getProfile": [profile()],
                       "app.bsky.feed.getAuthorFeed": [{"feed": []}]})
        c.resolve()
        assert list(c.feed_media()) == []
        assert not c.parse_looks_wrong()

    def test_a_working_parse_is_not_flagged(self):
        c = collector({"app.bsky.actor.getProfile": [profile()],
                       "app.bsky.feed.getAuthorFeed": [{"feed": [post(SUBJECT_DID, "b1")]}]})
        c.resolve()
        assert len(list(c.feed_media())) == 1
        assert not c.parse_looks_wrong()
