"""The offline guard: analysis must never need the network."""
from __future__ import annotations

import pytest

import casework.config as config


def test_fetcher_refuses_when_offline(monkeypatch):
    from casework.crawl.fetch import Fetcher

    monkeypatch.setattr(config.SETTINGS, "allow_network", False)
    with pytest.raises(config.OfflineError):
        Fetcher()


def test_clustering_needs_no_network(monkeypatch, tmp_path):
    """Clustering runs with the network hard-disabled."""
    import numpy as np

    monkeypatch.setattr(config.SETTINGS, "root", tmp_path)
    monkeypatch.setattr(config.SETTINGS, "allow_network", False)

    from datetime import date, timedelta

    from casework.case import CaseStore
    from casework.faces.cluster import cluster_case
    from casework.models import Authorisation, FaceRecord

    store = CaseStore.create(
        "op-offline", "Offline",
        Authorisation(holder="h", instrument="i", granted=date.today(),
                      expires=date.today() + timedelta(days=5)),
    )
    case = store.load()

    # Two tight groups, far apart: clustering must find exactly two.
    rng = np.random.default_rng(0)
    a = np.tile(np.array([1.0] + [0.0] * 127, dtype=np.float32), (5, 1))
    b = np.tile(np.array([0.0, 1.0] + [0.0] * 126, dtype=np.float32), (4, 1))
    vectors = np.vstack([a, b]) + rng.normal(0, 0.01, (9, 128)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    store.save_embeddings(vectors)

    case.faces = [
        FaceRecord(id=f"f{i}", artifact_id=f"a{i}", box=(0, 0, 10, 10),
                   det_score=0.9, embedding_row=i)
        for i in range(9)
    ]
    store.save(case)

    result = cluster_case("op-offline")
    assert result["clusters"] == 2
    # Relabelled by size, so the five-face group is cluster 0.
    case = store.load()
    assert sum(1 for f in case.faces if f.cluster == 0) == 5
