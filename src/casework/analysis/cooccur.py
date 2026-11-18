"""Who appears with whom, and where.

For most investigations this is the payoff rather than the face matching
itself. Knowing that an unnamed cluster appears in eleven photographs
alongside an enrolled subject, across two independent sources, is a lead;
knowing that a face scored 0.41 is not.

Edges are counted per *artifact*, not per source page, so a gallery that
embeds the same photograph on six pages contributes one co-occurrence rather
than six. Without that, a site's own layout dominates the graph.
"""
from __future__ import annotations

from itertools import combinations

from ..case import CaseStore


def cooccurrence(case_id: str, min_weight: int = 1) -> dict[str, object]:
    store = CaseStore(case_id)
    case = store.load()
    artifacts = case.artifacts_by_id()

    # Deduplicate by content hash: the same image collected from two URLs is
    # one photograph and must not double every edge inside it.
    by_photo: dict[str, set[int]] = {}
    for face in case.faces:
        if face.cluster is None:
            continue
        artifact = artifacts.get(face.artifact_id)
        if artifact is None:
            continue
        by_photo.setdefault(artifact.sha256, set()).add(face.cluster)

    labels: dict[int, str | None] = {}
    for face in case.faces:
        if face.cluster is not None and face.matched_subject:
            labels[face.cluster] = face.matched_subject

    weights: dict[tuple[int, int], int] = {}
    for clusters in by_photo.values():
        for a, b in combinations(sorted(clusters), 2):
            weights[(a, b)] = weights.get((a, b), 0) + 1

    nodes = sorted({c for clusters in by_photo.values() for c in clusters})
    photo_counts = {
        cluster: sum(1 for clusters in by_photo.values() if cluster in clusters)
        for cluster in nodes
    }

    return {
        "nodes": [
            {
                "cluster": c,
                "subject": labels.get(c),
                "photos": photo_counts[c],
            }
            for c in nodes
        ],
        "edges": [
            {"a": a, "b": b, "photos": w}
            for (a, b), w in sorted(weights.items(), key=lambda kv: -kv[1])
            if w >= min_weight
        ],
    }


def cluster_locations(case_id: str, cluster_id: int) -> list[dict[str, object]]:
    """Every place one cluster was seen, with full provenance for each hit."""
    store = CaseStore(case_id)
    case = store.load()
    artifacts = case.artifacts_by_id()

    rows = []
    for face in case.faces:
        if face.cluster != cluster_id:
            continue
        artifact = artifacts.get(face.artifact_id)
        if artifact is None:
            continue
        rows.append(
            {
                "face_id": face.id,
                "url": artifact.url,
                "referer": artifact.referer,
                "source_id": artifact.source_id,
                "attributed_subject": artifact.subject_id,
                "collected": artifact.fetched.isoformat(timespec="seconds"),
                "posted": artifact.posted,
                "sha256": artifact.sha256,
                "det_score": round(face.det_score, 3),
                "match_score": face.match_score,
            }
        )
    # Order by when it was posted where that is known, falling back to when
    # it was collected -- so an API-collected account reads as a timeline
    # rather than as the order the crawler happened to walk it.
    return sorted(rows, key=lambda r: r["posted"] or r["collected"])


def unknowns_near_subjects(case_id: str) -> list[dict[str, object]]:
    """Unnamed clusters ranked by how often they appear with enrolled subjects.

    This is the lead list. It answers "who is around this person that I have
    not identified yet" using only material already lawfully collected, and
    it deliberately stops there -- resolving those clusters to names is
    investigative work with other evidence, not something this tool does.
    """
    graph = cooccurrence(case_id)
    named = {n["cluster"] for n in graph["nodes"] if n["subject"]}
    by_cluster = {n["cluster"]: n for n in graph["nodes"]}

    tally: dict[int, dict[str, object]] = {}
    for edge in graph["edges"]:
        for this, other in ((edge["a"], edge["b"]), (edge["b"], edge["a"])):
            if this in named or other not in named:
                continue
            row = tally.setdefault(
                this,
                {"cluster": this, "photos": by_cluster[this]["photos"],
                 "with_subjects": {}, "shared_photos": 0},
            )
            subject = by_cluster[other]["subject"]
            row["with_subjects"][subject] = row["with_subjects"].get(subject, 0) + edge["photos"]
            row["shared_photos"] += edge["photos"]

    return sorted(tally.values(), key=lambda r: -r["shared_photos"])
