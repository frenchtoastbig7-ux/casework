"""Group faces in the collection by who they appear to be.

Agglomerative average-linkage on cosine distance, with a fixed distance
threshold rather than a fixed number of clusters -- the whole point is that
the number of distinct people in a collection is unknown, which is what
K-means-style methods require you to already know.

Average linkage rather than single linkage is a deliberate choice with a
concrete failure behind it. Single linkage chains: one ambiguous face that
sits between two people welds their clusters together, and the result is a
single "cluster" containing two identities, which then defeats every
downstream check because the analyst sees one person's label on another
person's photographs. Average linkage requires the groups to be close on the
whole, so one bridging face cannot merge them.

Clusters are *hypotheses about sameness*, not identities. Naming a cluster is
a separate, human act -- see match.py.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from ..case import CaseStore
from ..config import SETTINGS


def cluster_case(case_id: str, threshold: float | None = None) -> dict[str, int]:
    store = CaseStore(case_id)
    case = store.load()
    vectors = store.load_embeddings()

    usable = [f for f in case.faces if f.has_embedding and f.embedding_row < len(vectors)]
    if len(usable) < 2:
        for face in usable:
            face.cluster = 0 if usable else None
        store.save(case)
        return {"faces": len(usable), "clusters": 1 if usable else 0, "singletons": len(usable)}

    matrix = vectors[[f.embedding_row for f in usable]]
    distance = threshold if threshold is not None else SETTINGS.cluster_threshold

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance,
        metric="cosine",
        linkage="average",
    )
    labels = model.fit_predict(matrix)

    # Relabel by descending size so cluster 0 is always the most-photographed
    # person in the collection; stable ordering makes the GUI and any written
    # report agree between runs.
    order = {
        old: new
        for new, (old, _n) in enumerate(
            sorted(
                ((lbl, int((labels == lbl).sum())) for lbl in set(labels.tolist())),
                key=lambda kv: (-kv[1], kv[0]),
            )
        )
    }
    for face, label in zip(usable, labels, strict=True):
        face.cluster = order[int(label)]

    sizes = np.bincount(np.array([order[int(x)] for x in labels]))
    store.save(case)
    store.audit.append(
        "faces.clustered", faces=len(usable), clusters=len(sizes),
        threshold=distance, largest=int(sizes.max()),
    )
    return {
        "faces": len(usable),
        "clusters": len(sizes),
        "singletons": int((sizes == 1).sum()),
        "largest": int(sizes.max()),
    }


def cluster_summary(case_id: str) -> list[dict[str, object]]:
    """Per-cluster rollup for the GUI: size, spread, artifacts, subject."""
    store = CaseStore(case_id)
    case = store.load()
    vectors = store.load_embeddings()
    artifacts = case.artifacts_by_id()

    groups: dict[int, list] = {}
    for face in case.faces:
        if face.cluster is not None:
            groups.setdefault(face.cluster, []).append(face)

    out = []
    for cluster_id, faces in sorted(groups.items()):
        rows = [f.embedding_row for f in faces if 0 <= f.embedding_row < len(vectors)]
        if len(rows) > 1:
            block = vectors[rows]
            centroid = block.mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroid = centroid / norm if norm else centroid
            # Mean cosine similarity to the centroid. A low value on a large
            # cluster is the signal that two people have been merged.
            tightness = float((block @ centroid).mean())
        else:
            tightness = 1.0

        subjects = {f.matched_subject for f in faces if f.matched_subject}
        out.append(
            {
                "cluster": cluster_id,
                "size": len(faces),
                "artifacts": len({f.artifact_id for f in faces}),
                "sources": sorted(
                    {artifacts[f.artifact_id].source_id for f in faces if f.artifact_id in artifacts}
                ),
                "tightness": round(tightness, 3),
                "subject": sorted(subjects)[0] if len(subjects) == 1 else None,
                "contested": len(subjects) > 1,
                "faces": [f.id for f in faces[:60]],
                "best_face": max(faces, key=lambda f: f.det_score).id,
            }
        )
    return out
