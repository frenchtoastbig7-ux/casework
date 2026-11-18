"""Runtime settings and the two global guards every other module honours.

Two things are deliberately global rather than passed around, because a
capability that can be forgotten at one call site is not a control:

``OfflineError``
    Nothing in this package reaches the network except through the crawl
    fetcher, and the fetcher refuses to run when ``allow_network`` is off.
    Face detection, embedding, clustering and reporting are pure local
    computation over material already collected, so an analyst can work a
    case with networking disabled entirely and know that no request left the
    machine.

``ScopeError``
    Raised when something asks to fetch a URL the case's authorisation does
    not cover. It is an exception rather than a filtered-out no-op so that an
    off-scope request is loud in logs and tests instead of silently doing
    nothing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


class OfflineError(RuntimeError):
    """A network fetch was attempted while the tool is in offline mode."""


class ScopeError(PermissionError):
    """A fetch was attempted outside the case's authorised scope."""


class AuthorisationError(PermissionError):
    """The case's authorisation is absent, expired, or not yet in force."""


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


@dataclass
class Settings:
    """Filesystem layout and crawl politeness knobs.

    Defaults put everything under ``~/.casework``. Cases are directories, one
    per investigation, so an analyst can archive or hand over a whole case by
    copying a folder, and so deleting a case really deletes the collected
    material rather than orphaning rows in a shared database.
    """

    root: Path = field(
        default_factory=lambda: _env_path("CASEWORK_HOME", Path.home() / ".casework")
    )
    # Model weights are shared across every case and are not case material,
    # so they get their own location rather than living inside a case home.
    # That also means moving or wiping a case home does not force a redownload.
    model_root: Path | None = field(
        default_factory=lambda: (
            Path(os.environ["CASEWORK_MODELS"]).expanduser()
            if os.environ.get("CASEWORK_MODELS")
            else None
        )
    )
    allow_network: bool = field(
        default_factory=lambda: os.environ.get("CASEWORK_OFFLINE", "") == ""
    )

    # Politeness. These are intentionally slow. A scoped crawl covers tens or
    # hundreds of pages belonging to a handful of subjects, not millions, so
    # there is no reason to hit a host hard, and hammering a site is both
    # rude and an excellent way to get an investigation noticed.
    request_pause: float = 1.5
    request_timeout: float = 45.0
    max_retries: int = 3
    max_artifact_bytes: int = 40 * 1024 * 1024

    user_agent: str = (
        "casework/0.1 (scoped investigative collection; contact via case authorisation)"
    )

    # Face pipeline. `match_threshold` is cosine *similarity* against a
    # subject's reference embeddings and defaults to OpenCV's published
    # operating point for the SFace recogniser; `cluster_threshold` is cosine
    # *distance* for agglomerative grouping. See faces/match.py for why these
    # sit where they do and why raising them is the safer error.
    det_min_score: float = 0.70
    det_min_pixels: int = 48
    cluster_threshold: float = 0.55
    match_threshold: float = 0.363

    @property
    def cases(self) -> Path:
        return self.root / "cases"

    @property
    def model_cache(self) -> Path:
        return self.model_root or (self.root / "models")

    def case_dir(self, case_id: str) -> Path:
        return self.cases / safe_name(case_id)


def safe_name(name: str) -> str:
    """Filesystem-safe slug. Case ids reach here from CLI and HTTP input."""
    cleaned = "".join(c if c.isalnum() or c in "-_" else "-" for c in name.strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned[:64].lower() or "case"


SETTINGS = Settings()


def require_network() -> None:
    if not SETTINGS.allow_network:
        raise OfflineError(
            "Network access is disabled (CASEWORK_OFFLINE is set). "
            "Collection needs the network; analysis of already-collected "
            "material does not."
        )
