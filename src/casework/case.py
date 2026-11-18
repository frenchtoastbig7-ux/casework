"""Case storage: one directory per investigation.

Layout::

    ~/.casework/cases/<case-id>/
        case.json           the Case model
        audit.jsonl         hash-chained action log
        embeddings.npy      float32 [n_faces, dim], row-indexed by FaceRecord
        raw/<sha2>/<sha>.<ext>   collected artifacts, content-addressed
        refs/               subject reference images
        thumbs/             face crops for the GUI
        exports/            generated reports

Content-addressed storage means refetching the same image twice costs one
copy and one artifact row, and that an artifact's bytes can be verified
against its recorded hash years later.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .audit import AuditLog
from .config import SETTINGS, safe_name
from .models import Authorisation, Case


class CaseStore:
    def __init__(self, case_id: str) -> None:
        self.id = safe_name(case_id)
        self.dir = SETTINGS.case_dir(self.id)
        self.audit = AuditLog(self.dir / "audit.jsonl")

    # -- paths ------------------------------------------------------------
    @property
    def case_file(self) -> Path:
        return self.dir / "case.json"

    @property
    def raw_dir(self) -> Path:
        return self.dir / "raw"

    @property
    def refs_dir(self) -> Path:
        return self.dir / "refs"

    @property
    def thumbs_dir(self) -> Path:
        return self.dir / "thumbs"

    @property
    def exports_dir(self) -> Path:
        return self.dir / "exports"

    @property
    def embeddings_file(self) -> Path:
        return self.dir / "embeddings.npy"

    def blob_path(self, sha: str, suffix: str) -> Path:
        """Content-addressed path, fanned out two hex chars deep.

        A flat directory of ten thousand images is slow to list on most
        filesystems and painful in a file browser when an analyst wants to
        look at the raw collection by hand.
        """
        return self.raw_dir / sha[:2] / f"{sha}{suffix}"

    # -- lifecycle --------------------------------------------------------
    def exists(self) -> bool:
        return self.case_file.exists()

    @classmethod
    def create(cls, case_id: str, title: str, authorisation: Authorisation) -> CaseStore:
        store = cls(case_id)
        if store.exists():
            raise FileExistsError(f"Case {store.id!r} already exists at {store.dir}")
        for d in (store.raw_dir, store.refs_dir, store.thumbs_dir, store.exports_dir):
            d.mkdir(parents=True, exist_ok=True)
        case = Case(id=store.id, title=title, authorisation=authorisation)
        store.save(case)
        store.audit.append(
            "case.created",
            case=store.id,
            title=title,
            instrument=authorisation.instrument,
            holder=authorisation.holder,
            expires=authorisation.expires,
        )
        return store

    def load(self) -> Case:
        if not self.exists():
            raise FileNotFoundError(
                f"No case {self.id!r} at {self.dir}. Create it with `casework case new`."
            )
        return Case.model_validate_json(self.case_file.read_text(encoding="utf-8"))

    def save(self, case: Case) -> None:
        """Write atomically -- a crash mid-write must not lose the case file."""
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.case_file.with_suffix(".json.tmp")
        tmp.write_text(case.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self.case_file)

    # -- embeddings -------------------------------------------------------
    def load_embeddings(self) -> np.ndarray:
        if not self.embeddings_file.exists():
            return np.zeros((0, 128), dtype=np.float32)
        return np.load(self.embeddings_file)

    def save_embeddings(self, matrix: np.ndarray) -> None:
        np.save(self.embeddings_file, matrix.astype(np.float32))

    def append_embeddings(self, rows: np.ndarray) -> int:
        """Append rows, returning the index the first new row landed at."""
        existing = self.load_embeddings()
        if rows.size == 0:
            return len(existing)
        rows = rows.astype(np.float32)
        if existing.size == 0:
            merged = rows
        else:
            if existing.shape[1] != rows.shape[1]:
                raise ValueError(
                    f"Embedding dimension changed ({existing.shape[1]} -> {rows.shape[1]}). "
                    "The recogniser model differs from the one this case was built with; "
                    "re-run `casework faces --reset` to rebuild consistently."
                )
            merged = np.vstack([existing, rows])
        self.save_embeddings(merged)
        return len(existing)

    # -- reference images -------------------------------------------------
    def add_reference(self, subject_id: str, image: Path) -> str:
        self.refs_dir.mkdir(parents=True, exist_ok=True)
        dest = self.refs_dir / f"{safe_name(subject_id)}-{image.name}"
        shutil.copy2(image, dest)
        return str(dest.relative_to(self.dir))

    def resolve(self, relative: str) -> Path:
        return self.dir / relative

    # -- housekeeping -----------------------------------------------------
    def stats(self) -> dict[str, int]:
        case = self.load()
        return {
            "subjects": len(case.subjects),
            "sources": len(case.sources),
            "artifacts": len(case.artifacts),
            "images": sum(1 for a in case.artifacts if a.is_image),
            "faces": len(case.faces),
            "clusters": len({f.cluster for f in case.faces if f.cluster is not None}),
            "matched": sum(1 for f in case.faces if f.matched_subject),
        }


def list_cases() -> list[dict[str, object]]:
    if not SETTINGS.cases.exists():
        return []
    out = []
    for d in sorted(SETTINGS.cases.iterdir()):
        cf = d / "case.json"
        if not cf.exists():
            continue
        try:
            raw = json.loads(cf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        auth = raw.get("authorisation", {})
        out.append(
            {
                "id": raw.get("id", d.name),
                "title": raw.get("title", d.name),
                "created": raw.get("created", ""),
                "expires": auth.get("expires", ""),
                "subjects": len(raw.get("subjects", [])),
                "artifacts": len(raw.get("artifacts", [])),
            }
        )
    return out

