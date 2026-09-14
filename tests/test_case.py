"""Case store, authorisation gating, audit chain, and label durability."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

import casework.config as config

SHARED_MODELS = config.Settings().model_cache


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Isolate case storage, but keep the real (shared) model cache."""
    monkeypatch.setattr(config.SETTINGS, "root", tmp_path)
    monkeypatch.setattr(config.SETTINGS, "model_root", SHARED_MODELS)
    yield tmp_path


def make_auth(days: int = 30):
    from casework.models import Authorisation

    return Authorisation(
        holder="DI Okafor",
        instrument="WARR-2026-114",
        granted=date.today() - timedelta(days=1),
        expires=date.today() + timedelta(days=days),
        scope_note="Collection limited to accounts named in the schedule.",
    )


def new_case(case_id="op-test", days=30):
    from casework.case import CaseStore

    return CaseStore.create(case_id, "Test case", make_auth(days))


class TestAuthorisation:
    def test_live_authorisation_passes(self):
        make_auth(10).check()

    def test_expired_authorisation_raises(self):
        with pytest.raises(config.AuthorisationError, match="expired"):
            make_auth(-1).check()

    def test_not_yet_in_force_raises(self):
        from casework.models import Authorisation

        auth = Authorisation(
            holder="x", instrument="y",
            granted=date.today() + timedelta(days=5),
            expires=date.today() + timedelta(days=40),
        )
        with pytest.raises(config.AuthorisationError, match="does not take effect"):
            auth.check()

    def test_crawl_refuses_on_expired_authorisation(self):
        from casework.crawl.runner import crawl
        from casework.models import Source

        store = new_case("op-expired", days=-1)
        case = store.load()
        case.subjects.append(_subject())
        case.sources.append(
            Source(id="src-001", subject_id="alice", url="https://example.org/users/alice",
                   attribution="handle matches seized header, confirmed by exhibit 4")
        )
        store.save(case)

        with pytest.raises(config.AuthorisationError):
            crawl("op-expired")
        # The refusal is itself recorded.
        assert any(e["event"] == "crawl.refused" for e in store.audit.entries())


def _subject():
    from casework.models import Subject

    return Subject(id="alice", label="Alice Doe")


class TestSourceValidation:
    def test_attribution_must_be_substantive(self):
        from casework.models import Source

        with pytest.raises(ValueError, match="attribution"):
            Source(id="s", subject_id="alice", url="https://example.org/u/a", attribution="dunno")

    def test_reasonable_attribution_accepted(self):
        from casework.models import Source

        source = Source(
            id="s", subject_id="alice", url="https://example.org/u/a",
            attribution="username matches handle in seized message header",
        )
        assert source.subject_id == "alice"


class TestAuditChain:
    def test_chain_verifies_when_untouched(self):
        store = new_case("op-audit")
        for i in range(5):
            store.audit.append("test.event", index=i)
        ok, detail = store.audit.verify()
        assert ok, detail

    def test_edited_entry_is_detected(self):
        store = new_case("op-tamper")
        store.audit.append("fetch.ok", url="https://example.org/a")
        store.audit.append("fetch.ok", url="https://example.org/b")

        lines = store.audit.path.read_text().splitlines()
        # Not lines[0] -- creating the case writes its own entry first.
        target = next(i for i, line in enumerate(lines) if "example.org/a" in line)
        lines[target] = lines[target].replace("example.org/a", "example.org/z")
        store.audit.path.write_text("\n".join(lines) + "\n")

        ok, detail = store.audit.verify()
        assert not ok
        assert "hash does not match" in detail

    def test_deleted_entry_is_detected(self):
        store = new_case("op-delete")
        for i in range(4):
            store.audit.append("test.event", index=i)
        lines = store.audit.path.read_text().splitlines()
        del lines[2]
        store.audit.path.write_text("\n".join(lines) + "\n")

        ok, detail = store.audit.verify()
        assert not ok
        assert "chain" in detail.lower()


class TestManualLabelsSurviveReclustering:
    def test_labels_are_keyed_by_face_not_cluster(self):
        """The reason manual_labels is keyed by face id.

        Re-clustering renumbers clusters. If a decision were stored against a
        cluster number, changing the threshold would silently move a human
        judgement onto a different person.
        """
        from casework.models import FaceRecord

        store = new_case("op-labels")
        case = store.load()
        case.subjects.append(_subject())
        faces = [
            FaceRecord(id=f"face{i}", artifact_id="a1", box=(0, 0, 10, 10),
                       det_score=0.9, embedding_row=i, cluster=0)
            for i in range(3)
        ]
        case.faces = faces
        case.manual_labels = {"face1": "alice"}
        store.save(case)

        # Simulate a re-cluster that renumbers everything.
        case = store.load()
        for i, face in enumerate(case.faces):
            face.cluster = 7 + i
        store.save(case)

        case = store.load()
        labelled = [f for f in case.faces if f.id in case.manual_labels]
        assert [f.id for f in labelled] == ["face1"]
        assert case.manual_labels["face1"] == "alice"


class TestStore:
    def test_case_roundtrips(self):
        store = new_case("op-round")
        case = store.load()
        case.subjects.append(_subject())
        store.save(case)
        assert store.load().subjects[0].label == "Alice Doe"

    def test_duplicate_case_refused(self):
        new_case("op-dupe")
        with pytest.raises(FileExistsError):
            new_case("op-dupe")

    def test_embedding_dimension_change_is_caught(self):
        import numpy as np

        store = new_case("op-dims")
        store.append_embeddings(np.zeros((2, 128), dtype=np.float32))
        with pytest.raises(ValueError, match="Embedding dimension changed"):
            store.append_embeddings(np.zeros((1, 512), dtype=np.float32))

    def test_blob_paths_are_fanned_out(self):
        store = new_case("op-blobs")
        path = store.blob_path("abcdef0123", ".jpg")
        assert path.parent.name == "ab"
        assert path.name == "abcdef0123.jpg"


class TestCaseCreationInput:
    """Bugs found by actually launching the app: a bad date 500'd, and a
    case could be created already expired without saying so."""

    def test_malformed_expiry_is_rejected_with_a_readable_message(self):
        from fastapi.testclient import TestClient

        from casework.server import create_app

        client = TestClient(create_app())
        r = client.post("/api/cases", json={
            "id": "op-bad", "title": "t", "holder": "h",
            "instrument": "i", "expires": "next tuesday",
        })
        assert r.status_code == 422
        assert "YYYY-MM-DD" in r.text

    def test_empty_expiry_is_rejected(self):
        from fastapi.testclient import TestClient

        from casework.server import create_app

        client = TestClient(create_app())
        r = client.post("/api/cases", json={
            "id": "op-empty", "title": "t", "holder": "h",
            "instrument": "i", "expires": "",
        })
        assert r.status_code == 422

    def test_case_born_expired_is_created_but_warns(self):
        from fastapi.testclient import TestClient

        from casework.case import CaseStore
        from casework.server import create_app

        client = TestClient(create_app())
        r = client.post("/api/cases", json={
            "id": "op-old", "title": "t", "holder": "h",
            "instrument": "OLD-1", "expires": "2020-01-01",
        })
        assert r.status_code == 200
        assert "cannot collect" in r.json()["warning"]
        # And the fact is on the record, not just in a dismissable toast.
        events = [e["event"] for e in CaseStore("op-old").audit.entries()]
        assert "case.created_expired" in events

    def test_valid_expiry_creates_without_warning(self):
        from datetime import date, timedelta

        from fastapi.testclient import TestClient

        from casework.server import create_app

        client = TestClient(create_app())
        r = client.post("/api/cases", json={
            "id": "op-good", "title": "t", "holder": "h", "instrument": "i",
            "expires": str(date.today() + timedelta(days=30)),
        })
        assert r.status_code == 200
        assert r.json()["warning"] is None
