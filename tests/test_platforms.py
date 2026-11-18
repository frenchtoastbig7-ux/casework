"""Handle expansion: templating only, and the federation rule.

The important negative here is that this module cannot discover anything.
There is no function that takes a handle and returns which platforms carry
it, and these tests are where that stays true.
"""
from __future__ import annotations

import pytest

from casework.platforms import PLATFORMS, HandleError, catalogue, expand


class TestExpansion:
    def test_mastodon_instance_embedded_in_handle(self):
        url, plat = expand("mastodon", "@alice@mastodon.social")
        assert url == "https://mastodon.social/@alice"
        assert plat.key == "mastodon"

    def test_mastodon_instance_passed_separately(self):
        url, _ = expand("mastodon", "alice", "infosec.exchange")
        assert url == "https://infosec.exchange/@alice"

    def test_mastodon_without_an_instance_is_refused(self):
        # Federated: the same local name on another instance is another
        # person, so guessing a default instance would be an identity error.
        with pytest.raises(HandleError, match="federated"):
            expand("mastodon", "alice")

    def test_explicit_instance_overrides_the_embedded_one(self):
        url, _ = expand("mastodon", "@alice@mastodon.social", "infosec.exchange")
        assert url == "https://infosec.exchange/@alice"

    def test_bluesky_domain_handle(self):
        url, _ = expand("bluesky", "alice.bsky.social")
        assert url == "https://bsky.app/profile/alice.bsky.social"

    def test_bluesky_did(self):
        url, _ = expand("bluesky", "did:plc:ewvi7nxzyoun6zhxrhs64oiz")
        assert url == "https://bsky.app/profile/did:plc:ewvi7nxzyoun6zhxrhs64oiz"

    def test_bluesky_bare_word_is_refused(self):
        # Bluesky handles are domains; a bare word is a mistake, not a handle.
        with pytest.raises(HandleError, match="does not look like"):
            expand("bluesky", "alice")

    def test_leading_at_is_optional(self):
        assert expand("x", "@alice")[0] == expand("x", "alice")[0]

    def test_pasted_profile_url_is_accepted(self):
        url, _ = expand("mastodon", "https://mastodon.social/@bob")
        assert url == "https://mastodon.social/@bob"

    def test_tumblr_uses_a_subdomain(self):
        assert expand("tumblr", "alice")[0] == "https://alice.tumblr.com/"

    def test_aliases_resolve(self):
        assert expand("twitter", "alice")[0] == expand("x", "alice")[0]
        assert expand("bsky", "a.bsky.social")[0] == expand("bluesky", "a.bsky.social")[0]

    def test_unknown_platform_lists_the_known_ones(self):
        with pytest.raises(HandleError, match="mastodon"):
            expand("myspace", "alice")

    def test_expansion_is_deterministic(self):
        assert expand("mastodon", "@a@b.social")[0] == expand("mastodon", "@a@b.social")[0]


class TestNoDiscovery:
    def test_expansion_never_touches_the_network(self, monkeypatch):
        """A nonexistent handle expands happily -- proof nothing is looked up."""
        import casework.config as config

        monkeypatch.setattr(config.SETTINGS, "allow_network", False)
        url, _ = expand("mastodon", "@definitely-not-a-real-account@example.social")
        assert url == "https://example.social/@definitely-not-a-real-account"

    def test_there_is_no_cross_platform_search_function(self):
        import casework.platforms as mod

        # Guards against someone later adding the one feature that would turn
        # expansion into enumeration.
        suspicious = [
            name for name in dir(mod)
            if any(w in name.lower() for w in ("search", "discover", "enumerate", "find_all"))
        ]
        assert suspicious == []


class TestCatalogue:
    def test_every_platform_declares_a_collection_rating(self):
        from casework.platforms import COLLECTION_RATINGS

        assert all(p.collection in COLLECTION_RATINGS for p in PLATFORMS)

    def test_every_platform_has_a_usable_example(self):
        for plat in PLATFORMS:
            assert plat.example, f"{plat.key} has no example handle"

    def test_the_named_platforms_are_present(self):
        keys = {row["key"] for row in catalogue()}
        assert {"mastodon", "bluesky"} <= keys

    def test_templates_all_produce_valid_scoped_prefixes(self):
        """Every template must survive Prefix parsing, or the platform is
        unusable however good its expansion looks."""
        from casework.scope import Prefix

        for plat in PLATFORMS:
            handle = "did:plc:abc123" if plat.key == "bluesky" else "alicetest"
            url, _ = expand(plat.key, handle, "example.social" if plat.needs_instance else None)
            prefix = Prefix.from_url(url, "src-001", "alice", plat.default_depth)
            assert prefix.host
            assert prefix.covers(url)
