"""Case data model.

The shape here encodes the tool's central constraint: **every source is
attributed to a named subject before it can be collected**. There is no
"search the web for this face" object because there is no such operation.
A ``Source`` cannot exist without a ``subject_id`` and an ``attribution``
string saying how it was established that the source belongs to that
subject, which is the same thing an analyst has to justify on paper.
"""
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(UTC)


class Authorisation(BaseModel):
    """The mandate the case is worked under, and when it lapses.

    ``expires`` is enforced, not decorative: collection refuses to run once
    the date has passed. Authority to collect is nearly always time-boxed and
    the box is nearly always missed, because nothing in the toolchain
    normally knows about it. Here it does.
    """

    holder: str
    instrument: str = Field(description="Warrant, engagement or approval reference")
    granted: date
    expires: date
    scope_note: str = ""

    def check(self, when: date | None = None) -> None:
        from .config import AuthorisationError

        today = when or utcnow().date()
        if today < self.granted:
            raise AuthorisationError(
                f"Authorisation {self.instrument} does not take effect until {self.granted}."
            )
        if today > self.expires:
            raise AuthorisationError(
                f"Authorisation {self.instrument} expired on {self.expires}. "
                "Collection is blocked. Renew the instrument and update the case."
            )

    def days_left(self, when: date | None = None) -> int:
        return (self.expires - (when or utcnow().date())).days


class Subject(BaseModel):
    """A person already identified in the case, by name or by case handle."""

    id: str
    label: str
    note: str = ""
    reference_images: list[str] = Field(default_factory=list)
    added: datetime = Field(default_factory=utcnow)


class Source(BaseModel):
    """One authorised collection point, tied to one subject.

    ``url`` is a prefix, not a seed: the crawler will not leave it. A source
    of ``https://example.org/users/alice`` authorises that page and pages
    beneath it, and nothing else on the host.
    """

    id: str
    subject_id: str
    url: str
    attribution: str = Field(
        description="How it was established this source belongs to the subject"
    )
    depth: int = 1
    added: datetime = Field(default_factory=utcnow)
    enabled: bool = True

    @field_validator("attribution")
    @classmethod
    def _attribution_is_substantive(cls, v: str) -> str:
        if len(v.strip()) < 8:
            raise ValueError(
                "attribution must say how the source was tied to the subject "
                "(e.g. 'username matches handle in seized message header, "
                "confirmed by profile photo in exhibit 4')"
            )
        return v.strip()


class Artifact(BaseModel):
    """One fetched file, with the provenance needed to defend it later."""

    id: str
    source_id: str
    subject_id: str
    url: str
    referer: str | None = None
    fetched: datetime
    status: int
    content_type: str = ""
    sha256: str
    bytes: int = 0
    path: str
    is_image: bool = False
    # When the subject *posted* it, as reported by an API that knows. The
    # HTML crawler cannot supply this and leaves it None, which is the honest
    # answer -- `fetched` is when we collected, and the two are not the same
    # fact. A timeline built from `fetched` would say every photograph in a
    # ten-year account appeared on the afternoon of the crawl.
    posted: str | None = None

    def file(self, case_dir: Path) -> Path:
        return case_dir / self.path


class FaceRecord(BaseModel):
    """One detected face inside one artifact.

    ``embedding_row`` indexes the case's embedding matrix rather than
    carrying the vector, so the JSON stays readable and the vectors stay in a
    single contiguous ``.npy`` that clustering can load in one read.
    """

    id: str
    artifact_id: str
    box: tuple[int, int, int, int]
    det_score: float
    embedding_row: int = -1
    cluster: int | None = None
    matched_subject: str | None = None
    match_score: float | None = None

    @property
    def has_embedding(self) -> bool:
        return self.embedding_row >= 0


class CrawlRun(BaseModel):
    """One collection pass, for the audit trail and for resumability."""

    id: str
    started: datetime
    finished: datetime | None = None
    sources: list[str] = Field(default_factory=list)
    fetched: int = 0
    skipped: int = 0
    failed: int = 0
    refused: int = 0
    notes: list[str] = Field(default_factory=list)


class Case(BaseModel):
    """Everything about one investigation except the collected bytes."""

    id: str
    title: str
    created: datetime = Field(default_factory=utcnow)
    authorisation: Authorisation
    subjects: list[Subject] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    faces: list[FaceRecord] = Field(default_factory=list)
    runs: list[CrawlRun] = Field(default_factory=list)
    # Artifacts the face pipeline has already looked at. Tracked explicitly
    # rather than inferred from `faces`, so an image that genuinely contains
    # no face is not re-analysed on every subsequent pass.
    processed_artifacts: list[str] = Field(default_factory=list)
    # Analyst adjudications, keyed by **face id** rather than cluster id.
    # Cluster numbering is an artefact of the last clustering run and changes
    # when the threshold changes; a human judgement about a photograph must
    # not silently move to a different person because of that. Value is a
    # subject id, or "" for an explicit "this is not any enrolled subject".
    manual_labels: dict[str, str] = Field(default_factory=dict)

    def subject(self, subject_id: str) -> Subject | None:
        return next((s for s in self.subjects if s.id == subject_id), None)

    def source(self, source_id: str) -> Source | None:
        return next((s for s in self.sources if s.id == source_id), None)

    def artifact(self, artifact_id: str) -> Artifact | None:
        return next((a for a in self.artifacts if a.id == artifact_id), None)

    def artifacts_by_id(self) -> dict[str, Artifact]:
        return {a.id: a for a in self.artifacts}
