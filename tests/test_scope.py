"""Scope is the control the whole tool rests on, so it gets the most tests."""
from __future__ import annotations

import pytest

from casework.config import ScopeError
from casework.scope import Prefix, Scope, normalise


def prefix(url: str, depth: int = 1) -> Prefix:
    return Prefix.from_url(url, "src-001", "alice", depth)


def scope(url: str = "https://example.org/users/alice", depth: int = 1) -> Scope:
    return Scope([prefix(url, depth)])


class TestNormalise:
    def test_drops_fragment_and_lowercases_host(self):
        assert normalise("HTTPS://Example.ORG/a#frag") == "https://example.org/a"

    def test_strips_default_port(self):
        assert normalise("https://example.org:443/a") == "https://example.org/a"

    def test_keeps_nonstandard_port(self):
        assert normalise("https://example.org:8443/a") == "https://example.org:8443/a"

    def test_keeps_query_because_pagination_lives_there(self):
        assert normalise("https://example.org/a?page=2") == "https://example.org/a?page=2"

    def test_resolves_dot_segments(self):
        assert normalise("https://example.org/users/alice/../bob") == "https://example.org/users/bob"


class TestPrefixMatching:
    def test_covers_itself_and_children(self):
        p = prefix("https://example.org/users/alice")
        assert p.covers("https://example.org/users/alice")
        assert p.covers("https://example.org/users/alice/photos")

    def test_sibling_with_shared_string_prefix_is_not_covered(self):
        # The bug a naive startswith() check would ship with.
        p = prefix("https://example.org/users/alice")
        assert not p.covers("https://example.org/users/alice-smith")

    def test_other_paths_on_same_host_are_not_covered(self):
        p = prefix("https://example.org/users/alice")
        assert not p.covers("https://example.org/users/bob")
        assert not p.covers("https://example.org/")

    def test_subdomains_are_not_covered(self):
        p = prefix("https://example.org/users/alice")
        assert not p.covers("https://cdn.example.org/users/alice")

    def test_dot_segments_cannot_climb_out(self):
        p = prefix("https://example.org/users/alice")
        assert not p.covers("https://example.org/users/alice/../bob")


class TestScopeChecks:
    def test_in_scope_page_passes(self):
        assert scope().check_page("https://example.org/users/alice/photos").source_id == "src-001"

    def test_out_of_scope_page_raises(self):
        with pytest.raises(ScopeError, match="Out of scope"):
            scope().check_page("https://example.org/users/bob")

    def test_depth_limit_is_enforced(self):
        s = scope(depth=1)
        s.check_page("https://example.org/users/alice/photos")
        with pytest.raises(ScopeError, match="deeper than"):
            s.check_page("https://example.org/users/alice/photos/2024/june")

    def test_non_web_schemes_refused(self):
        for bad in ("javascript:alert(1)", "file:///etc/passwd", "data:text/html,x"):
            with pytest.raises(ScopeError):
                scope().check_page(bad)

    def test_asset_on_foreign_cdn_allowed_via_in_scope_referer(self):
        # The CDN exception: necessary, because almost no site serves its own
        # images, and narrow, because the referring page is re-checked.
        s = scope()
        assert s.check_asset(
            "https://cdn.other-host.net/img/1.jpg",
            "https://example.org/users/alice",
        ).subject_id == "alice"

    def test_asset_cannot_be_laundered_by_a_forged_referer(self):
        with pytest.raises(ScopeError):
            scope().check_asset(
                "https://cdn.other-host.net/img/1.jpg",
                "https://example.org/users/bob",
            )

    def test_asset_must_look_like_an_image(self):
        with pytest.raises(ScopeError, match="recognised image"):
            scope().check_asset(
                "https://cdn.other-host.net/tracker.js",
                "https://example.org/users/alice",
            )

    def test_expand_links_drops_everything_off_prefix(self):
        s = scope()
        links = s.expand_links(
            "https://example.org/users/alice",
            [
                "/users/alice/photos",          # kept
                "/users/bob",                   # dropped: another person
                "https://elsewhere.net/x",      # dropped: another host
                "#top",                         # dropped: fragment
            ],
        )
        assert links == ["https://example.org/users/alice/photos"]

    def test_relative_links_resolve_against_the_base_directory(self):
        """RFC 3986 resolution, which is stricter here than it first looks.

        On a base of ``/users/alice`` (no trailing slash) a relative
        ``photos`` resolves to ``/users/photos`` -- a sibling of alice, not a
        child -- and is therefore correctly refused. On ``/users/alice/`` the
        same href resolves beneath the prefix and is kept.
        """
        s = scope()
        assert s.expand_links("https://example.org/users/alice", ["photos"]) == []
        assert s.expand_links("https://example.org/users/alice/", ["photos"]) == [
            "https://example.org/users/alice/photos"
        ]

    def test_expand_links_also_applies_the_depth_limit(self):
        s = scope(depth=1)
        assert s.expand_links("https://example.org/users/alice/", ["photos/2024"]) == []
        assert s.expand_links("https://example.org/users/alice/", ["photos"]) == [
            "https://example.org/users/alice/photos"
        ]

    def test_most_specific_prefix_wins(self):
        s = Scope([
            Prefix.from_url("https://example.org/users", "src-broad", "alice", 3),
            Prefix.from_url("https://example.org/users/bob", "src-bob", "bob", 1),
        ])
        assert s.check_page("https://example.org/users/bob/photos").subject_id == "bob"
