"""Face pipeline: the ONNX integration, and the match decision rules.

The decision tests matter most. They pin down the three verdicts -- matched,
unknown, contested -- against constructed embeddings, so the rule that a
match must also beat the runner-up cannot be quietly lost in a refactor.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

import casework.config as config

SHARED_MODELS = config.Settings().model_cache


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Isolate case storage, but keep the real (shared) model cache."""
    monkeypatch.setattr(config.SETTINGS, "root", tmp_path)
    monkeypatch.setattr(config.SETTINGS, "model_root", SHARED_MODELS)
    yield tmp_path


def unit(*indices: int, dim: int = 128) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for i in indices:
        v[i] = 1.0
    return v / np.linalg.norm(v)


def yunet_row(box, score=0.95) -> np.ndarray:
    """A detection row in YuNet's layout: box, five landmarks, score."""
    x, y, w, h = box
    return np.array([
        x, y, w, h,
        x + 0.32 * w, y + 0.38 * h,
        x + 0.68 * w, y + 0.38 * h,
        x + 0.50 * w, y + 0.58 * h,
        x + 0.36 * w, y + 0.76 * h,
        x + 0.64 * w, y + 0.76 * h,
        score,
    ], dtype=np.float32)


def make_case(case_id="op-faces"):
    from casework.case import CaseStore
    from casework.models import Authorisation

    return CaseStore.create(
        case_id, "Faces",
        Authorisation(holder="h", instrument="i", granted=date.today(),
                      expires=date.today() + timedelta(days=30)),
    )


# --------------------------------------------------------------- integration
def _models_present() -> bool:
    """Checked against the shared cache, not the per-test tmp home."""
    from casework.modelmgr import ALL_MODELS

    return all((SHARED_MODELS / m.filename).exists() for m in ALL_MODELS)


models_required = pytest.mark.skipif(
    not _models_present(), reason="ONNX models not cached; run `casework models fetch`",
)


@models_required
class TestRecogniserIntegration:
    def test_embedding_is_128d_and_l2_normalised(self):
        from casework.faces import recognise

        rng = np.random.default_rng(7)
        img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        det = recognise.Detection(box=(200, 140, 160, 200), score=0.95,
                                  raw=yunet_row((200, 140, 160, 200)))
        vec = recognise.embed(img, det)
        assert vec.shape == (128,)
        assert np.isfinite(vec).all()
        assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-5)

    def test_embedding_is_deterministic(self):
        from casework.faces import recognise

        rng = np.random.default_rng(11)
        img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        det = recognise.Detection(box=(180, 120, 150, 190), score=0.9,
                                  raw=yunet_row((180, 120, 150, 190)))
        assert np.allclose(recognise.embed(img, det), recognise.embed(img, det))

    def test_detect_on_a_blank_image_finds_nothing(self):
        from casework.faces import recognise

        assert recognise.detect(np.full((400, 400, 3), 200, dtype=np.uint8)) == []

    def test_unreadable_file_returns_none_not_an_exception(self, tmp_path):
        from casework.faces import recognise

        bad = tmp_path / "not-an-image.jpg"
        bad.write_bytes(b"nonsense")
        assert recognise.load_image(bad) is None


# ------------------------------------------------------------------ matching
class TestMatchDecisions:
    """Constructed embeddings, so each verdict is reached for one reason."""

    def _setup(self, monkeypatch, cluster_vectors: dict[int, np.ndarray]):
        from casework.faces import match as match_mod
        from casework.faces import recognise
        from casework.models import FaceRecord, Subject

        store = make_case()
        case = store.load()
        case.subjects = [
            Subject(id="alice", label="Alice", reference_images=["refs/a.jpg"]),
            Subject(id="bob", label="Bob", reference_images=["refs/b.jpg"]),
        ]

        vectors, faces = [], []
        for cluster_id, vec in cluster_vectors.items():
            for n in range(3):
                faces.append(FaceRecord(
                    id=f"f{cluster_id}-{n}", artifact_id=f"a{cluster_id}",
                    box=(0, 0, 60, 60), det_score=0.9,
                    embedding_row=len(vectors), cluster=cluster_id,
                ))
                vectors.append(vec)
        case.faces = faces
        store.save(case)
        store.save_embeddings(np.vstack(vectors))

        refs = {"alice": unit(0).reshape(1, -1), "bob": unit(1).reshape(1, -1)}
        monkeypatch.setattr(match_mod, "reference_embeddings",
                            lambda _store, sid: refs[sid])
        monkeypatch.setattr(recognise, "models_ready", lambda: True)
        return store

    def test_clear_match_is_assigned(self, monkeypatch):
        from casework.faces.match import match_case

        self._setup(monkeypatch, {0: unit(0)})
        result = match_case("op-faces")
        decision = result["decisions"][0]
        assert decision["verdict"] == "matched"
        assert decision["subject"] == "alice"

    def test_face_matching_nobody_is_unknown_not_nearest(self, monkeypatch):
        from casework.faces.match import match_case

        # Orthogonal to both references: the tool must not reach for the
        # nearest enrolled name.
        self._setup(monkeypatch, {0: unit(5)})
        assert match_case("op-faces")["decisions"][0]["verdict"] == "unknown"

    def test_equidistant_face_is_contested_not_the_higher_number(self, monkeypatch):
        from casework.faces.match import match_case

        # cos 0.707 to each reference: over threshold for both, so no margin.
        self._setup(monkeypatch, {0: unit(0, 1)})
        decision = match_case("op-faces")["decisions"][0]
        assert decision["verdict"] == "contested"
        assert set(decision["candidates"]) == {"alice", "bob"}
        assert "subject" not in decision

    def test_no_face_is_attributed_when_a_cluster_is_contested(self, monkeypatch):
        from casework.faces.match import match_case

        store = self._setup(monkeypatch, {0: unit(0, 1)})
        match_case("op-faces")
        assert all(f.matched_subject is None for f in store.load().faces)

    def test_rerun_clears_stale_attributions(self, monkeypatch):
        from casework.faces.match import match_case

        store = self._setup(monkeypatch, {0: unit(0)})
        match_case("op-faces")
        assert all(f.matched_subject == "alice" for f in store.load().faces)

        # A threshold above any achievable similarity must retract them.
        match_case("op-faces", threshold=1.01)
        assert all(f.matched_subject is None for f in store.load().faces)

    def test_analyst_decision_overrides_the_matcher(self, monkeypatch):
        from casework.faces.match import match_case

        store = self._setup(monkeypatch, {0: unit(0)})
        case = store.load()
        case.manual_labels = {"f0-1": "bob"}
        store.save(case)

        result = match_case("op-faces")
        assert result["manual_overrides"] == 1
        by_id = {f.id: f for f in store.load().faces}
        assert by_id["f0-1"].matched_subject == "bob"     # analyst
        assert by_id["f0-0"].matched_subject == "alice"   # matcher

    def test_matching_without_any_reference_is_refused(self, monkeypatch):
        from casework.faces import match as match_mod
        from casework.faces import recognise
        from casework.faces.match import match_case
        from casework.models import Subject

        store = make_case("op-noref")
        case = store.load()
        case.subjects = [Subject(id="alice", label="Alice")]
        store.save(case)
        monkeypatch.setattr(match_mod, "reference_embeddings",
                            lambda _s, _i: np.zeros((0, 128), dtype=np.float32))
        monkeypatch.setattr(recognise, "models_ready", lambda: True)

        with pytest.raises(ValueError, match="reference image"):
            match_case("op-noref")
