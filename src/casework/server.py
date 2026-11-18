"""Local HTTP API behind the review GUI.

Binds to loopback only. There is no authentication because there is no
multi-user mode and no remote access: the tool holds collected material about
identified people, and the correct place for that is one analyst's machine,
not a shared service. If it needs to be shared, share the case directory
under whatever controls the material actually requires.

Long operations (crawl, face processing) run on a worker thread and report
through a job record the GUI polls, so a collection that takes twenty minutes
does not hold an HTTP request open for twenty minutes.
"""
from __future__ import annotations

import threading
import traceback
import uuid
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from .case import CaseStore, list_cases
from .config import SETTINGS, safe_name
from .models import Authorisation, Source, Subject

STATIC = Path(__file__).parent / "web" / "static"

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def _job_start(kind: str, case_id: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id, "kind": kind, "case": case_id, "state": "running",
            "lines": [], "result": None, "error": None,
        }
    return job_id


def _job_log(job_id: str, event: str, detail: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job["lines"].append({"event": event, "detail": detail})
        # Keep the tail only; a large crawl would otherwise grow this
        # unboundedly in memory for output nobody scrolls back through.
        if len(job["lines"]) > 400:
            del job["lines"][:-400]


def _run_job(job_id: str, fn, *args, **kwargs) -> None:
    def worker() -> None:
        try:
            result = fn(*args, **kwargs)
            with JOBS_LOCK:
                JOBS[job_id]["result"] = _jsonable(result)
                JOBS[job_id]["state"] = "done"
        except Exception as exc:
            with JOBS_LOCK:
                JOBS[job_id]["error"] = f"{type(exc).__name__}: {exc}"
                JOBS[job_id]["state"] = "error"
                JOBS[job_id]["trace"] = traceback.format_exc()[-2000:]

    threading.Thread(target=worker, daemon=True).start()


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


class NewCase(BaseModel):
    id: str
    title: str
    holder: str
    instrument: str
    expires: str
    scope_note: str = ""

    @field_validator("expires")
    @classmethod
    def _parseable_date(cls, v: str) -> str:
        # Parsed here rather than in the handler so a typo comes back as a
        # readable 422 instead of a 500 the GUI can only render as "HTTP 500".
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError(
                f"expiry must be a date as YYYY-MM-DD, got {v!r}"
            ) from None
        return v


class NewSubject(BaseModel):
    id: str
    label: str
    note: str = ""


class NewSource(BaseModel):
    subject_id: str
    attribution: str
    url: str = ""
    handle: str = ""
    platform: str = ""
    instance: str = ""
    depth: int = -1          # -1 means "use the platform default, else 1"


class CrawlRequest(BaseModel):
    sources: list[str] | None = None
    max_pages: int = 200
    max_images: int = 2000
    respect_robots: bool = True
    refresh_pages: bool = True


class ClusterRequest(BaseModel):
    threshold: float | None = None


class AssignRequest(BaseModel):
    cluster: int
    subject_id: str | None = None
    reason: str = ""


def create_app() -> FastAPI:
    app = FastAPI(title="casework", docs_url=None, redoc_url=None)

    def store_for(case_id: str) -> CaseStore:
        store = CaseStore(safe_name(case_id))
        if not store.exists():
            raise HTTPException(404, f"No case {case_id!r}")
        return store

    # ---------------------------------------------------------------- pages
    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- cases
    @app.get("/api/cases")
    def api_cases() -> list[dict[str, Any]]:
        return list_cases()

    @app.post("/api/cases")
    def api_case_new(body: NewCase) -> dict[str, Any]:
        auth = Authorisation(
            holder=body.holder, instrument=body.instrument,
            granted=date.today(), expires=date.fromisoformat(body.expires),
            scope_note=body.scope_note,
        )
        try:
            store = CaseStore.create(safe_name(body.id), body.title, auth)
        except FileExistsError as exc:
            raise HTTPException(409, str(exc)) from exc

        # A case may legitimately be opened over an expired instrument to
        # analyse material collected while it was live -- analysis is offline
        # and ungated. But it must not be a silent state: collection is
        # already blocked, and the reason should be said out loud once.
        warning = None
        if auth.days_left() < 0:
            warning = (
                f"Authorisation {auth.instrument} expired on {auth.expires}. "
                "This case can be reviewed but cannot collect."
            )
            store.audit.append("case.created_expired", instrument=auth.instrument,
                               expires=auth.expires)
        return {"id": store.id, "warning": warning}

    @app.get("/api/case/{case_id}")
    def api_case(case_id: str) -> dict[str, Any]:
        store = store_for(case_id)
        case = store.load()
        auth = case.authorisation
        return {
            "id": case.id,
            "title": case.title,
            "authorisation": {
                "holder": auth.holder, "instrument": auth.instrument,
                "granted": str(auth.granted), "expires": str(auth.expires),
                "days_left": auth.days_left(), "scope_note": auth.scope_note,
                "expired": auth.days_left() < 0,
            },
            "stats": store.stats(),
            "subjects": [
                {
                    "id": s.id, "label": s.label, "note": s.note,
                    "references": len(s.reference_images),
                    "faces": sum(1 for f in case.faces if f.matched_subject == s.id),
                }
                for s in case.subjects
            ],
            "sources": [s.model_dump(mode="json") for s in case.sources],
            "runs": [r.model_dump(mode="json") for r in case.runs[-10:]],
            "models_ready": _models_ready(),
        }

    def _models_ready() -> bool:
        from .faces import recognise

        return recognise.models_ready()

    @app.post("/api/case/{case_id}/subjects")
    def api_subject_new(case_id: str, body: NewSubject) -> dict[str, Any]:
        store = store_for(case_id)
        case = store.load()
        sid = safe_name(body.id)
        if case.subject(sid):
            raise HTTPException(409, f"Subject {sid!r} already exists")
        case.subjects.append(Subject(id=sid, label=body.label, note=body.note))
        store.save(case)
        store.audit.append("subject.added", subject=sid, label=body.label, via="gui")
        return {"id": sid}

    @app.post("/api/case/{case_id}/sources")
    def api_source_new(case_id: str, body: NewSource) -> dict[str, Any]:
        store = store_for(case_id)
        case = store.load()
        sid = safe_name(body.subject_id)
        if case.subject(sid) is None:
            raise HTTPException(400, "Every source must belong to an enrolled subject")
        if bool(body.url) == bool(body.handle):
            raise HTTPException(400, "Give exactly one of url or handle")

        url, depth, expanded = body.url, body.depth, None
        if body.handle:
            from .platforms import HandleError, expand

            if not body.platform:
                raise HTTPException(400, "A handle needs a platform")
            try:
                url, plat = expand(body.platform, body.handle, body.instance or None)
            except HandleError as exc:
                raise HTTPException(400, str(exc)) from exc
            expanded = {"platform": plat.key, "collection": plat.collection, "note": plat.note}
            if depth < 0:
                depth = plat.default_depth
        if depth < 0:
            depth = 1

        try:
            source = Source(
                id=f"src-{len(case.sources) + 1:03d}", subject_id=sid, url=url,
                attribution=body.attribution, depth=depth,
            )
            from .scope import Prefix

            Prefix.from_url(source.url, source.id, sid, depth)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        case.sources.append(source)
        store.save(case)
        store.audit.append(
            "source.added", source=source.id, subject=sid, url=url,
            attribution=body.attribution, via="gui",
            **({"handle": body.handle, "platform": body.platform} if body.handle else {}),
        )
        return {**source.model_dump(mode="json"), "expanded": expanded}

    @app.get("/api/case/{case_id}/scope")
    def api_scope(case_id: str, url: str) -> dict[str, Any]:
        from .crawl.runner import preview_scope

        store_for(case_id)
        return preview_scope(safe_name(case_id), url)

    # ------------------------------------------------------------ pipeline
    @app.post("/api/case/{case_id}/crawl")
    def api_crawl(case_id: str, body: CrawlRequest) -> dict[str, Any]:
        store = store_for(case_id)
        case = store.load()
        try:
            case.authorisation.check()
        except Exception as exc:
            raise HTTPException(403, str(exc)) from exc

        from .crawl.runner import crawl

        job_id = _job_start("crawl", store.id)
        _run_job(
            job_id, crawl, store.id, source_ids=body.sources,
            respect_robots=body.respect_robots, max_pages=body.max_pages,
            max_images=body.max_images, refresh_pages=body.refresh_pages,
            progress=lambda e, d: _job_log(job_id, e, d),
        )
        return {"job": job_id}

    @app.post("/api/case/{case_id}/faces")
    def api_faces(case_id: str, reset: bool = False) -> dict[str, Any]:
        store = store_for(case_id)
        from .faces.pipeline import process

        job_id = _job_start("faces", store.id)
        _run_job(
            job_id, process, store.id, reset=reset,
            progress=lambda e, d: _job_log(job_id, e, d),
        )
        return {"job": job_id}

    @app.post("/api/case/{case_id}/cluster")
    def api_cluster(case_id: str, body: ClusterRequest) -> dict[str, Any]:
        store = store_for(case_id)
        from .faces.cluster import cluster_case

        return _jsonable(cluster_case(store.id, body.threshold))

    @app.post("/api/case/{case_id}/match")
    def api_match(case_id: str, body: ClusterRequest) -> dict[str, Any]:
        store = store_for(case_id)
        from .faces.match import match_case

        try:
            return _jsonable(match_case(store.id, body.threshold))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/job/{job_id}")
    def api_job(job_id: str) -> dict[str, Any]:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is None:
                raise HTTPException(404, "No such job")
            return dict(job)

    # ------------------------------------------------------------- review
    @app.get("/api/case/{case_id}/clusters")
    def api_clusters(case_id: str) -> list[dict[str, Any]]:
        store = store_for(case_id)
        from .faces.cluster import cluster_summary

        return _jsonable(cluster_summary(store.id))

    @app.get("/api/case/{case_id}/cluster/{cluster_id}")
    def api_cluster_detail(case_id: str, cluster_id: int) -> dict[str, Any]:
        store = store_for(case_id)
        case = store.load()
        from .analysis.cooccur import cluster_locations

        faces = [f for f in case.faces if f.cluster == cluster_id]
        return {
            "cluster": cluster_id,
            "faces": [
                {
                    "id": f.id, "artifact": f.artifact_id,
                    "det_score": round(f.det_score, 3),
                    "match_score": f.match_score,
                    "subject": f.matched_subject,
                    "manual": f.id in case.manual_labels,
                }
                for f in sorted(faces, key=lambda f: -f.det_score)
            ],
            "locations": _jsonable(cluster_locations(store.id, cluster_id)),
        }

    @app.post("/api/case/{case_id}/assign")
    def api_assign(case_id: str, body: AssignRequest) -> dict[str, Any]:
        """Record an analyst's decision about a whole cluster.

        Stored per face so it survives re-clustering, and audited as a human
        judgement distinct from an algorithmic match.
        """
        store = store_for(case_id)
        case = store.load()
        if body.subject_id and case.subject(body.subject_id) is None:
            raise HTTPException(400, f"No subject {body.subject_id!r}")

        touched = 0
        for face in case.faces:
            if face.cluster != body.cluster:
                continue
            case.manual_labels[face.id] = body.subject_id or ""
            face.matched_subject = body.subject_id or None
            face.match_score = None
            touched += 1

        store.save(case)
        store.audit.append(
            "match.manual", cluster=body.cluster, subject=body.subject_id or "(none)",
            faces=touched, reason=body.reason,
        )
        return {"cluster": body.cluster, "faces": touched, "subject": body.subject_id}

    @app.get("/api/case/{case_id}/graph")
    def api_graph(case_id: str, min_weight: int = 1) -> dict[str, Any]:
        store = store_for(case_id)
        from .analysis.cooccur import cooccurrence, unknowns_near_subjects

        return {
            **_jsonable(cooccurrence(store.id, min_weight)),
            "leads": _jsonable(unknowns_near_subjects(store.id)),
        }

    @app.get("/api/case/{case_id}/audit")
    def api_audit(case_id: str, limit: int = 200) -> dict[str, Any]:
        store = store_for(case_id)
        ok, detail = store.audit.verify()
        return {"chain_ok": ok, "chain_detail": detail, "entries": store.audit.entries(limit)}

    @app.post("/api/case/{case_id}/report")
    def api_report(case_id: str, redact: bool = True) -> dict[str, Any]:
        store = store_for(case_id)
        from .report import write_report

        path = write_report(store.id, redact_unmatched=redact)
        return {"path": str(path)}

    # --------------------------------------------------------------- media
    @app.get("/api/case/{case_id}/thumb/{face_id}")
    def api_thumb(case_id: str, face_id: str) -> FileResponse:
        store = store_for(case_id)
        path = store.thumbs_dir / f"{safe_name(face_id)}.jpg"
        if not path.exists():
            raise HTTPException(404, "No thumbnail")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/case/{case_id}/image/{artifact_id}")
    def api_image(case_id: str, artifact_id: str) -> FileResponse:
        store = store_for(case_id)
        case = store.load()
        artifact = case.artifact(artifact_id)
        if artifact is None:
            raise HTTPException(404, "No such artifact")
        path = artifact.file(store.dir)
        # Confine to the case directory: artifact paths come from the case
        # file, and a case file could be hand-edited or copied in.
        try:
            path.resolve().relative_to(store.dir.resolve())
        except ValueError as exc:
            raise HTTPException(403, "Artifact path escapes the case directory") from exc
        if not path.exists():
            raise HTTPException(404, "Artifact bytes are missing")
        return FileResponse(path, media_type=artifact.content_type or "application/octet-stream")

    @app.get("/api/platforms")
    def api_platforms() -> list[dict[str, Any]]:
        from .platforms import catalogue

        return catalogue()

    @app.post("/api/platforms/preview")
    def api_platform_preview(body: NewSource) -> dict[str, Any]:
        """Expand a handle without storing anything, so the GUI can show the
        resulting URL before the analyst commits to authorising it."""
        from .platforms import HandleError, expand

        try:
            url, plat = expand(body.platform, body.handle, body.instance or None)
        except HandleError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"url": url, "platform": plat.key, "collection": plat.collection,
                "note": plat.note, "depth": plat.default_depth}

    @app.get("/api/models")
    def api_models() -> dict[str, Any]:
        from .modelmgr import status

        return {"models": _jsonable(status()), "offline": not SETTINGS.allow_network}

    @app.post("/api/models/fetch")
    def api_models_fetch() -> dict[str, Any]:
        from .modelmgr import ensure_models

        job_id = _job_start("models", "-")
        _run_job(job_id, ensure_models)
        return {"job": job_id}

    @app.exception_handler(PermissionError)
    def _permission(_request, exc: PermissionError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=403)

    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

    return app
