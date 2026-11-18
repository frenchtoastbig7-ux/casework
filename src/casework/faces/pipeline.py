"""Run detection and embedding across a case's collected images.

Incremental by design: a second pass only looks at artifacts collected since
the first. Face analysis is the expensive stage, and re-embedding a
ten-thousand-image case because two new photos arrived makes the tool
unusable in the way that actually matters -- an analyst stops re-running it,
and then works from stale clusters.
"""
from __future__ import annotations

import uuid
from collections.abc import Callable

import cv2
import numpy as np

from ..case import CaseStore
from ..models import FaceRecord
from . import recognise

Progress = Callable[[str, str], None]


def _noop(_e: str, _d: str) -> None:
    return None


def process(case_id: str, reset: bool = False, progress: Progress = _noop) -> dict[str, int]:
    if not recognise.models_ready():
        raise recognise.ModelsUnavailable(
            "Face models are not cached. Run `casework models fetch` first "
            "(about 39 MB, downloaded once, then everything runs offline)."
        )

    store = CaseStore(case_id)
    case = store.load()

    if reset:
        case.faces = []
        case.processed_artifacts = []
        store.save_embeddings(np.zeros((0, 128), dtype=np.float32))
        for old in store.thumbs_dir.glob("*.jpg"):
            old.unlink()

    done = set(case.processed_artifacts)
    todo = [a for a in case.artifacts if a.is_image and a.id not in done]
    if not todo:
        return {"images": 0, "faces": 0, "unreadable": 0}

    store.thumbs_dir.mkdir(parents=True, exist_ok=True)
    new_vectors: list[np.ndarray] = []
    new_records: list[FaceRecord] = []
    unreadable = 0
    base_row = len(store.load_embeddings())

    for index, artifact in enumerate(todo, start=1):
        path = artifact.file(store.dir)
        progress("image", f"{index}/{len(todo)} {path.name}")

        bgr = recognise.load_image(path)
        if bgr is None:
            unreadable += 1
            case.processed_artifacts.append(artifact.id)
            continue

        for detection in recognise.detect(bgr):
            vector = recognise.embed(bgr, detection)
            record = FaceRecord(
                id=uuid.uuid4().hex[:16],
                artifact_id=artifact.id,
                box=detection.box,
                det_score=detection.score,
                embedding_row=base_row + len(new_vectors),
            )
            new_vectors.append(vector)
            new_records.append(record)

            thumb = recognise.crop(bgr, detection.box)
            if thumb.size:
                cv2.imwrite(
                    str(store.thumbs_dir / f"{record.id}.jpg"), thumb,
                    [cv2.IMWRITE_JPEG_QUALITY, 88],
                )

        case.processed_artifacts.append(artifact.id)

    if new_vectors:
        store.append_embeddings(np.vstack(new_vectors))
    case.faces.extend(new_records)
    store.save(case)
    store.audit.append(
        "faces.processed", images=len(todo), faces=len(new_records), unreadable=unreadable,
    )
    progress("done", f"{len(new_records)} faces across {len(todo)} images")
    return {"images": len(todo), "faces": len(new_records), "unreadable": unreadable}
