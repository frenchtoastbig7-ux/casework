"""Attribute clusters to named subjects, against a bounded reference set.

This is the only place the tool assigns an identity, and it is built around
three constraints that follow from what it is for.

**Matching is against subjects already in the case.** The reference set is
whatever the analyst enrolled -- typically a booking photo, an exhibit, or a
frame the subject was already identified in. There is no global index and no
way to ask "who is this?" of the world. If a face matches nobody enrolled,
the answer is "unknown", and unknown is a normal, common, correct result.

**Decisions are made per cluster, not per face.** A single face at an awkward
angle is a weak observation; forty faces of the same person across a
collection is a strong one. Deciding at the cluster level and requiring
agreement across its best members means one flattering profile shot cannot
carry an identification on its own.

**A match must also beat the runner-up.** Scoring above threshold for two
different enrolled subjects is not a match, it is a warning that the
reference images are too similar or too poor to separate them. That case
returns ``contested`` rather than picking the higher number, because picking
the higher number is precisely how a tool like this produces a confident,
documented, wrong identification.

The default threshold is OpenCV's published cosine operating point for
SFace. It is not a probability. It does not mean "96% certain". It is a
similarity score whose error rate on your material is unknown until you
measure it on your material, which is what ``calibrate`` is for.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..case import CaseStore
from ..config import SETTINGS
from . import recognise

Progress = Callable[[str, str], None]

# How many of a cluster's best-scoring faces vote on its identity, and how far
# ahead of the runner-up subject the winner must be.
VOTE_TOP_K = 5
MIN_MARGIN = 0.05


def _noop(_e: str, _d: str) -> None:
    return None


def reference_embeddings(store: CaseStore, subject_id: str) -> np.ndarray:
    """Embeddings of every enrolled reference image for one subject.

    The largest face in each reference image is used. Reference images are
    supposed to be portraits of the subject; when one contains several
    people, the biggest face is the subject far more often than not, and the
    analyst can see what was enrolled in the GUI and remove a wrong one.
    """
    case = store.load()
    subject = case.subject(subject_id)
    if subject is None:
        raise KeyError(f"No subject {subject_id!r} in case {store.id!r}")

    vectors: list[np.ndarray] = []
    for relative in subject.reference_images:
        path = store.resolve(relative)
        if not path.exists():
            continue
        bgr = recognise.load_image(path)
        if bgr is None:
            continue
        detections = recognise.detect(bgr)
        if not detections:
            continue
        largest = max(detections, key=lambda d: d.box[2] * d.box[3])
        vectors.append(recognise.embed(bgr, largest))

    if not vectors:
        return np.zeros((0, 128), dtype=np.float32)
    return np.vstack(vectors)


def match_case(
    case_id: str,
    threshold: float | None = None,
    progress: Progress = _noop,
) -> dict[str, object]:
    if not recognise.models_ready():
        raise recognise.ModelsUnavailable("Run `casework models fetch` first.")

    store = CaseStore(case_id)
    case = store.load()
    vectors = store.load_embeddings()
    limit = threshold if threshold is not None else SETTINGS.match_threshold

    enrolled: dict[str, np.ndarray] = {}
    for subject in case.subjects:
        refs = reference_embeddings(store, subject.id)
        if len(refs):
            enrolled[subject.id] = refs
            progress("subject", f"{subject.label}: {len(refs)} reference face(s)")
        else:
            progress(
                "warn",
                f"{subject.label}: no usable reference face -- cannot be matched",
            )

    if not enrolled:
        raise ValueError(
            "No subject has a usable reference image. Enrol one with "
            "`casework subject ref <case> <subject> <image>`; without a reference "
            "there is nothing to match against, which is the point."
        )

    # Clear any previous attribution so a re-run with a different threshold
    # cannot leave stale identities behind on faces that no longer qualify.
    for face in case.faces:
        face.matched_subject = None
        face.match_score = None

    clusters: dict[int, list] = {}
    for face in case.faces:
        if face.cluster is not None and 0 <= face.embedding_row < len(vectors):
            clusters.setdefault(face.cluster, []).append(face)

    decisions: list[dict[str, object]] = []

    for cluster_id, faces in sorted(clusters.items()):
        block = vectors[[f.embedding_row for f in faces]]

        # Per-subject: each face's best similarity to any of that subject's
        # references, then the mean of the top K faces as the cluster score.
        scores: dict[str, float] = {}
        per_face: dict[str, np.ndarray] = {}
        for subject_id, refs in enrolled.items():
            face_best = (block @ refs.T).max(axis=1)
            per_face[subject_id] = face_best
            top = np.sort(face_best)[::-1][:VOTE_TOP_K]
            scores[subject_id] = float(top.mean())

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        winner, top_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_score - runner_up

        if top_score < limit:
            decisions.append(
                {"cluster": cluster_id, "verdict": "unknown", "size": len(faces),
                 "best_subject": winner, "score": round(top_score, 3)}
            )
            continue

        if len(ranked) > 1 and margin < MIN_MARGIN:
            decisions.append(
                {"cluster": cluster_id, "verdict": "contested", "size": len(faces),
                 "candidates": [r[0] for r in ranked[:2]],
                 "score": round(top_score, 3), "margin": round(margin, 3)}
            )
            store.audit.append(
                "match.contested", cluster=cluster_id,
                candidates=[r[0] for r in ranked[:2]], score=round(top_score, 3),
            )
            continue

        for face, score in zip(faces, per_face[winner], strict=True):
            face.matched_subject = winner
            face.match_score = round(float(score), 4)

        decisions.append(
            {"cluster": cluster_id, "verdict": "matched", "subject": winner,
             "size": len(faces), "score": round(top_score, 3),
             "margin": round(margin, 3)}
        )
        store.audit.append(
            "match.assigned", cluster=cluster_id, subject=winner, faces=len(faces),
            score=round(top_score, 3), margin=round(margin, 3), threshold=limit,
        )

    # Analyst adjudications win, always, and are applied last so a re-run at
    # a different threshold cannot quietly overturn a human decision.
    overridden = 0
    for face in case.faces:
        if face.id in case.manual_labels:
            decided = case.manual_labels[face.id]
            face.matched_subject = decided or None
            face.match_score = None
            overridden += 1

    store.save(case)
    matched = sum(1 for d in decisions if d["verdict"] == "matched")
    return {
        "manual_overrides": overridden,
        "threshold": limit,
        "clusters": len(decisions),
        "matched": matched,
        "unknown": sum(1 for d in decisions if d["verdict"] == "unknown"),
        "contested": sum(1 for d in decisions if d["verdict"] == "contested"),
        "decisions": decisions,
    }


def calibrate(case_id: str, subject_id: str, positives: list[Path],
              negatives: list[Path]) -> dict[str, object]:
    """Measure the score separation on *your* material before trusting it.

    Give it images you know contain the subject and images you know do not.
    It reports the two score distributions and where they overlap, which is
    the only honest basis for choosing a threshold. A published operating
    point was measured on a benchmark, and your collection -- CCTV stills,
    compressed social media reposts, group photos at distance -- is not that
    benchmark.
    """
    store = CaseStore(case_id)
    refs = reference_embeddings(store, subject_id)
    if not len(refs):
        raise ValueError(f"Subject {subject_id!r} has no usable reference image.")

    def best_scores(paths: list[Path]) -> list[float]:
        out = []
        for path in paths:
            for _detection, vector in recognise.embed_file(path):
                out.append(float((refs @ vector).max()))
        return out

    pos = best_scores(positives)
    neg = best_scores(negatives)
    if not pos or not neg:
        raise ValueError("Need at least one detectable face in each of positives and negatives.")

    overlap = max(neg) >= min(pos)
    suggested = (max(neg) + min(pos)) / 2 if not overlap else None

    return {
        "subject": subject_id,
        "positive_faces": len(pos),
        "negative_faces": len(neg),
        "positive_min": round(min(pos), 3),
        "positive_mean": round(float(np.mean(pos)), 3),
        "negative_max": round(max(neg), 3),
        "negative_mean": round(float(np.mean(neg)), 3),
        "separable": not overlap,
        "suggested_threshold": round(suggested, 3) if suggested else None,
        "note": (
            "Distributions overlap: no threshold separates these examples cleanly. "
            "Enrol better reference images before relying on any match."
            if overlap
            else "Distributions are separable on this sample. This is a small "
                 "sample, not a validation study."
        ),
    }
