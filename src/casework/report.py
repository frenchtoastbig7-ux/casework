"""Self-contained HTML case report.

Two deliberate properties:

**Unattributed faces are blurred by default.** A collection gathered about
two subjects routinely contains bystanders who are nobody's business. A
report that leaves them visible turns a scoped collection into an
unscoped disclosure the moment it is emailed to anyone. ``--no-redact``
exists for the case where the whole image is the evidence, and choosing it
is recorded.

**Every image carries its provenance inline.** Source URL, referring page,
collection time and content hash sit next to the picture, because a report
whose images cannot be traced back is not usable as anything.
"""
from __future__ import annotations

import base64
import html
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from .case import CaseStore
from .faces import recognise

MAX_WIDTH = 900


def _render_image(store: CaseStore, artifact, faces: list, redact: bool) -> str | None:
    """Return a data: URI for one artifact, blurring unattributed faces."""
    bgr = recognise.load_image(artifact.file(store.dir))
    if bgr is None:
        return None

    if redact:
        for face in faces:
            if face.matched_subject:
                continue
            x, y, w, h = face.box
            pad_x, pad_y = int(w * 0.25), int(h * 0.25)
            x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
            x1 = min(bgr.shape[1], x + w + pad_x)
            y1 = min(bgr.shape[0], y + h + pad_y)
            roi = bgr[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            # Blur then pixelate: a Gaussian alone is partially invertible on
            # a known kernel, and this material may be disclosed.
            kernel = max(15, (min(x1 - x0, y1 - y0) // 3) | 1)
            roi = cv2.GaussianBlur(roi, (kernel, kernel), 0)
            small = cv2.resize(
                roi, (max(1, roi.shape[1] // 12), max(1, roi.shape[0] // 12)),
                interpolation=cv2.INTER_LINEAR,
            )
            bgr[y0:y1, x0:x1] = cv2.resize(
                small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST
            )

    height, width = bgr.shape[:2]
    if width > MAX_WIDTH:
        scale = MAX_WIDTH / width
        bgr = cv2.resize(bgr, (MAX_WIDTH, int(height * scale)), interpolation=cv2.INTER_AREA)

    ok, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(np.asarray(buffer).tobytes()).decode()


def write_report(case_id: str, redact_unmatched: bool = True, dest: Path | None = None) -> Path:
    store = CaseStore(case_id)
    case = store.load()
    artifacts = case.artifacts_by_id()

    faces_by_artifact: dict[str, list] = {}
    for face in case.faces:
        faces_by_artifact.setdefault(face.artifact_id, []).append(face)

    generated = datetime.now(UTC)
    esc = html.escape
    auth = case.authorisation
    chain_ok, chain_detail = store.audit.verify()

    parts: list[str] = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>{esc(case.title)} — casework report</title>",
        """<style>
        :root{--ink:#1a1a1c;--ink2:#5c5f66;--line:#e3e0da;--bg:#fbfaf8;--panel:#fff;
              --warn:#8a6100;--warn-bg:#fdf1d8;--bad:#a52626;--bad-bg:#fbeaea}
        body{margin:0;background:var(--bg);color:var(--ink);
             font:14px/1.6 ui-sans-serif,-apple-system,"Segoe UI",system-ui,sans-serif}
        .wrap{max-width:980px;margin:0 auto;padding:44px 26px 90px}
        h1{font-size:23px;margin:0 0 4px;letter-spacing:-.01em}
        h2{font-size:16px;margin:38px 0 14px;padding-bottom:7px;border-bottom:1px solid var(--line)}
        h3{font-size:14px;margin:26px 0 8px}
        .muted{color:var(--ink2)}
        .banner{background:var(--warn-bg);color:var(--warn);border-radius:8px;
                padding:13px 16px;margin:20px 0;font-size:13px}
        .banner.bad{background:var(--bad-bg);color:var(--bad)}
        dl{display:grid;grid-template-columns:auto 1fr;gap:5px 16px;font-size:13px;margin:14px 0}
        dt{color:var(--ink2)} dd{margin:0;overflow-wrap:anywhere}
        table{width:100%;border-collapse:collapse;font-size:12.5px;margin:12px 0}
        th{text-align:left;color:var(--ink2);font-size:10.5px;text-transform:uppercase;
           letter-spacing:.07em;padding:0 9px 6px}
        td{padding:7px 9px;border-top:1px solid var(--line);vertical-align:top;overflow-wrap:anywhere}
        figure{margin:0 0 22px;background:var(--panel);border:1px solid var(--line);
               border-radius:9px;overflow:hidden}
        figure img{width:100%;display:block}
        figcaption{padding:10px 13px;font-size:11.5px;color:var(--ink2);
                   font-family:ui-monospace,Menlo,monospace;line-height:1.7}
        .mono{font-family:ui-monospace,Menlo,monospace}
        </style></head><body><div class='wrap'>""",
        f"<h1>{esc(case.title)}</h1>",
        f"<div class='muted'>casework report · generated {generated.isoformat(timespec='seconds')}</div>",
    ]

    if redact_unmatched:
        parts.append(
            "<div class='banner'>Faces not attributed to an enrolled subject are "
            "redacted in this report.</div>"
        )
    else:
        parts.append(
            "<div class='banner bad'>Redaction was disabled for this export. Every face "
            "in the collection is visible, including people who are not subjects of this case.</div>"
        )
    if not chain_ok:
        parts.append(f"<div class='banner bad'>Audit chain failure: {esc(chain_detail)}</div>")

    parts.append("<h2>Authorisation</h2><dl>")
    for key, value in [
        ("Instrument", auth.instrument), ("Holder", auth.holder),
        ("Granted", str(auth.granted)), ("Expires", str(auth.expires)),
        ("Scope", auth.scope_note or "—"),
        ("Audit chain", chain_detail),
    ]:
        parts.append(f"<dt>{esc(key)}</dt><dd>{esc(str(value))}</dd>")
    parts.append("</dl>")

    parts.append("<h2>Collection scope</h2>")
    parts.append(
        "<p class='muted'>Every source below was declared before collection and attributed "
        "to a named subject. The crawler does not leave these prefixes.</p>"
    )
    parts.append("<table><thead><tr><th>source</th><th>subject</th><th>prefix</th>"
                 "<th>depth</th><th>attribution</th></tr></thead><tbody>")
    for source in case.sources:
        subject = case.subject(source.subject_id)
        parts.append(
            f"<tr><td class='mono'>{esc(source.id)}</td>"
            f"<td>{esc(subject.label if subject else source.subject_id)}</td>"
            f"<td class='mono'>{esc(source.url)}</td><td>{source.depth}</td>"
            f"<td>{esc(source.attribution)}</td></tr>"
        )
    parts.append("</tbody></table>")

    totals = store.stats()
    refused = sum(run.refused for run in case.runs)
    parts.append(
        f"<p class='muted'>{totals['artifacts']} artifacts collected "
        f"({totals['images']} images), {totals['faces']} faces detected, "
        f"{totals['clusters']} distinct people grouped, {totals['matched']} faces "
        f"attributed to an enrolled subject. {refused} candidate URLs were refused "
        f"as outside the authorised scope.</p>"
    )

    parts.append("<h2>Subjects</h2>")
    for subject in case.subjects:
        subject_faces = [f for f in case.faces if f.matched_subject == subject.id]
        subject_artifacts = sorted(
            {f.artifact_id for f in subject_faces},
            key=lambda aid: artifacts[aid].fetched if aid in artifacts else generated,
        )
        parts.append(f"<h3>{esc(subject.label)} <span class='muted'>({esc(subject.id)})</span></h3>")
        if subject.note:
            parts.append(f"<p class='muted'>{esc(subject.note)}</p>")
        manual = sum(1 for f in subject_faces if f.id in case.manual_labels)
        parts.append(
            f"<p class='muted'>{len(subject_faces)} face(s) across "
            f"{len(subject_artifacts)} photograph(s); {manual} attributed by analyst decision, "
            f"{len(subject_faces) - manual} by score.</p>"
        )

        for artifact_id in subject_artifacts:
            artifact = artifacts.get(artifact_id)
            if artifact is None:
                continue
            data_uri = _render_image(
                store, artifact, faces_by_artifact.get(artifact_id, []), redact_unmatched
            )
            if data_uri is None:
                continue
            scores = [
                f.match_score for f in faces_by_artifact.get(artifact_id, [])
                if f.matched_subject == subject.id and f.match_score is not None
            ]
            score_text = f"match {max(scores):.3f}" if scores else "analyst decision"
            parts.append(
                f"<figure><img src='{data_uri}' alt=''><figcaption>"
                f"{esc(artifact.url)}<br>"
                f"collected {artifact.fetched.isoformat(timespec='seconds')} · "
                f"source {esc(artifact.source_id)} · {score_text}<br>"
                f"sha256 {artifact.sha256}"
                + (f"<br>referer {esc(artifact.referer)}" if artifact.referer else "")
                + "</figcaption></figure>"
            )

    parts.append("<h2>Method and limits</h2>")
    parts.append(
        "<p class='muted'>Faces were detected with YuNet and embedded with SFace, grouped by "
        "average-linkage agglomerative clustering on cosine distance, and compared only against "
        "reference images enrolled in this case. Scores are cosine similarities, not "
        "probabilities: a score above the threshold means the embedding is close to an enrolled "
        "reference, not that the identification is correct at any stated confidence. No external "
        "index was consulted and no query left this machine. Every attribution here should be "
        "corroborated by other evidence before it is relied on.</p>"
    )
    parts.append("</div></body></html>")

    store.exports_dir.mkdir(parents=True, exist_ok=True)
    out = dest or store.exports_dir / f"report-{generated.strftime('%Y%m%d-%H%M%S')}.html"
    out.write_text("".join(parts), encoding="utf-8")
    store.audit.append(
        "report.exported", path=str(out), redacted=redact_unmatched,
        subjects=len(case.subjects), images=totals["images"],
    )
    return out
